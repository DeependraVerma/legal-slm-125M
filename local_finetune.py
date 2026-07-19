"""On-prem port of finetune.py (Phase 8, step 1): build a grounded Q&A SFT
dataset with a Gemini teacher, then tokenize it with our own Phase 3
tokenizer. Same logic as finetune.py, ported from Modal `@app.function`
steps + `.local_entrypoint()`s fanned out across cloud containers to plain
functions run in-process, and from Modal Volume writes to plain local disk
under config.DATA_ROOT (set SLM_DATA_ROOT to control where that is, same as
local_pipeline.py). The Modal `gemini-api` secret is replaced by reading
GEMINI_API_KEY straight from the environment — source .env.local and export
it first, same convention as HUGGINGFACE_TOKEN elsewhere in this repo:

    source .env.local && export GEMINI_API_KEY
    export SLM_DATA_ROOT=/raid/llm_sec/legal-slm-125M/data

Pipeline (unchanged from finetune.py):
    chunk   -> $SLM_DATA_ROOT/sft/passages.jsonl          (free, no API calls)
    pilot   -> tiny end-to-end run (~20 passages) + live cost projection
    build   -> full raw-set build: chunk -> generate (Gemini) -> judge (Gemini)
    curate  -> dedup + decontam + chat JSONL + tokenize -> $SLM_DATA_ROOT/sft/dataset/
    verify  -> print one tokenized train example, decoded

    .venv/bin/python3 local_finetune.py chunk --n-passages 20   # free, safe to run anytime
    .venv/bin/python3 local_finetune.py pilot                   # PAID: real Gemini calls
    .venv/bin/python3 local_finetune.py build                   # PAID: real Gemini calls
    .venv/bin/python3 local_finetune.py curate
    .venv/bin/python3 local_finetune.py verify

Always run `pilot` before `build` — it prints a live cost projection from
actual Gemini pricing before the full run commits spend. Per this project's
rules, get explicit user go-ahead immediately before running `pilot`/`build`
(they call the paid Gemini API) — `chunk`, `curate`, and `verify` make no
network calls and are free.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import config

BASE_MODEL_DIR = config.BASE_CKPT_DIR       # our own Phase 5 pretrained model
GEN_MODEL = "gemini-flash-lite-latest"      # cheap, high-volume generation
JUDGE_MODEL = "gemini-flash-latest"         # stronger validator
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

SYSTEM_PROMPT = "You are a knowledgeable legal and financial assistant. Answer accurately and concisely."

SFT_DIR = f"{config.DATA_ROOT}/sft"
PASSAGES_PATH = f"{SFT_DIR}/passages.jsonl"
RAW_QA_DIR = f"{SFT_DIR}/raw_qa"
JUDGED_DIR = f"{SFT_DIR}/judged"

# How the raw generation is sized. ~4 kept pairs/passage after judging.
PAIRS_PER_PASSAGE = 5
# Domain-weighted sampling of source passages (legal/financial first).
SOURCE_WEIGHTS = {"case-law": 0.45, "sec": 0.45, "fineweb-edu": 0.10}
PASSAGE_CHARS = 2800          # ~700-800 tokens per grounded passage


# --------------------------------------------------------------------------- #
# Gemini REST helper (thinking disabled for cost; retries on transient errors)
# --------------------------------------------------------------------------- #
def _gemini(model: str, prompt: str, *, temperature: float, max_tokens: int,
            api_key: str) -> tuple[str, dict]:
    import time

    import requests

    url = GEMINI_ENDPOINT.format(model=model) + f"?key={api_key}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    last = ""
    for attempt in range(4):
        try:
            r = requests.post(url, json=body, timeout=120)
            if r.status_code == 200:
                data = r.json()
                usage = data.get("usageMetadata", {})
                cand = (data.get("candidates") or [{}])[0]
                parts = cand.get("content", {}).get("parts", [{}])
                text = "".join(p.get("text", "") for p in parts)
                return text, {
                    "in": usage.get("promptTokenCount", 0),
                    "out": usage.get("candidatesTokenCount", 0),
                }
            last = f"{r.status_code}: {r.text[:160]}"
            if r.status_code in (429, 500, 503):
                time.sleep(2 * (attempt + 1))
                continue
            break
        except Exception as e:  # network hiccup
            last = str(e)[:160]
            time.sleep(2 * (attempt + 1))
    print(f"  [gemini {model}] failed: {last}", flush=True)
    return "", {"in": 0, "out": 0}


def _parse_json_array(text: str) -> list:
    import json
    import re

    if not text:
        return []
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else obj.get("pairs", []) if isinstance(obj, dict) else []
    except Exception:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return []
    return []


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def _gen_prompt(passage: str, k: int) -> str:
    return f"""You are building a supervised fine-tuning dataset for a legal & financial assistant.

Read the PASSAGE and write {k} high-quality question-answer pairs answerable USING ONLY the passage.

Vary them across:
- task_type: one of "qa", "extraction", "summarization", "rewrite"
- difficulty: one of "easy", "medium", "hard" (hard = multi-step reasoning over the passage)

Strict rules:
- The answer MUST be fully supported by the passage. Never use outside knowledge or invent facts, names, numbers, or citations.
- The QUESTION must be self-contained: a reader who cannot see the passage must understand it. Name the entity/company/case explicitly; do NOT say "this passage" or "the document".
- Answers must be correct, complete, and concise (1-4 sentences, or a short list for extraction).
- If the passage is boilerplate or cannot yield good questions, return fewer pairs or an empty array.

Return ONLY a JSON array:
[{{"q":"...","a":"...","task_type":"qa","difficulty":"easy"}}]

PASSAGE:
\"\"\"{passage}\"\"\""""


def _judge_prompt(passage: str, pairs: list) -> str:
    import json

    compact = [{"i": i, "q": p["q"], "a": p["a"]} for i, p in enumerate(pairs)]
    return f"""You are a strict validator for a fine-tuning dataset. Given a PASSAGE and candidate Q&A pairs, judge EACH pair on:
- grounded: is the answer fully supported by the passage, with no outside facts or hallucinations?
- correct: is the answer factually correct and directly responsive to the question?
- self_contained: is the question understandable WITHOUT seeing the passage (names the entity, not "this document")?

Keep a pair only if ALL THREE hold. Give an integer score 1-5 (5 = perfect). keep=true requires score>=4.

Return ONLY a JSON array, one object per pair:
[{{"i":0,"keep":true,"score":5,"reason":"..."}}]

PASSAGE:
\"\"\"{passage}\"\"\"

PAIRS:
{json.dumps(compact, ensure_ascii=False)}"""


# --------------------------------------------------------------------------- #
# Step 1: chunk the cleaned corpus into grounded passages (free, no API calls)
# --------------------------------------------------------------------------- #
def chunk_corpus(n_passages: int = 2000, seed: int = 1337) -> dict:
    import glob
    import json
    import os
    import random

    rng = random.Random(seed)
    os.makedirs(SFT_DIR, exist_ok=True)
    picked: list[dict] = []
    for source, weight in SOURCE_WEIGHTS.items():
        want = int(n_passages * weight)
        files = sorted(glob.glob(f"{config.CORPUS_DIR}/{source}/*.txt"))
        # reservoir-sample documents, then cut one passage from each
        docs: list[str] = []
        for path in files:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if len(line) >= PASSAGE_CHARS // 2:
                        if len(docs) < want * 4:
                            docs.append(line)
                        else:
                            j = rng.randint(0, len(docs) * 2)
                            if j < len(docs):
                                docs[j] = line
        rng.shuffle(docs)
        for doc in docs[:want]:
            start = 0 if len(doc) <= PASSAGE_CHARS else rng.randint(0, len(doc) - PASSAGE_CHARS)
            passage = doc[start:start + PASSAGE_CHARS].strip()
            picked.append({"source": source, "passage": passage})
    rng.shuffle(picked)
    with open(PASSAGES_PATH, "w", encoding="utf-8") as fh:
        for i, p in enumerate(picked):
            p["id"] = i
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    by_src = {s: sum(1 for p in picked if p["source"] == s) for s in SOURCE_WEIGHTS}
    print(f"chunked {len(picked)} passages: {by_src}")
    return {"n": len(picked), "by_source": by_src}


# --------------------------------------------------------------------------- #
# Step 2 + 3: generate and judge (one shard worker per thread group)
# --------------------------------------------------------------------------- #
def generate_shard(shard_id: int, passages: list, k: int = PAIRS_PER_PASSAGE) -> dict:
    import json
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    api_key = os.environ["GEMINI_API_KEY"]
    out_rows: list[dict] = []
    usage = {"in": 0, "out": 0, "calls": 0}

    def work(p):
        text, u = _gemini(GEN_MODEL, _gen_prompt(p["passage"], k),
                          temperature=0.85, max_tokens=2048, api_key=api_key)
        pairs = _parse_json_array(text)
        good = []
        for pr in pairs:
            if isinstance(pr, dict) and pr.get("q") and pr.get("a"):
                good.append({"q": str(pr["q"]).strip(), "a": str(pr["a"]).strip(),
                             "task_type": pr.get("task_type", "qa"),
                             "difficulty": pr.get("difficulty", "medium"),
                             "source": p["source"], "passage": p["passage"], "pid": p["id"]})
        return good, u

    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(work, p) for p in passages]
        for f in as_completed(futs):
            good, u = f.result()
            out_rows.extend(good)
            usage["in"] += u["in"]; usage["out"] += u["out"]; usage["calls"] += 1

    os.makedirs(RAW_QA_DIR, exist_ok=True)
    path = f"{RAW_QA_DIR}/shard-{shard_id:03d}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[gen {shard_id:03d}] {len(passages)} passages -> {len(out_rows)} raw pairs "
          f"| tokens in={usage['in']} out={usage['out']}")
    return {"shard": shard_id, "pairs": len(out_rows), "usage": usage}


def judge_shard(shard_id: int, groups: list) -> dict:
    """groups: list of {passage, pairs:[...]} grouped by source passage."""
    import json
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    api_key = os.environ["GEMINI_API_KEY"]
    kept: list[dict] = []
    usage = {"in": 0, "out": 0, "calls": 0}

    def work(g):
        pairs = g["pairs"]
        text, u = _gemini(JUDGE_MODEL, _judge_prompt(g["passage"], pairs),
                          temperature=0.0, max_tokens=2048, api_key=api_key)
        verdicts = _parse_json_array(text)
        keep = []
        vmap = {v.get("i"): v for v in verdicts if isinstance(v, dict)}
        for i, pr in enumerate(pairs):
            v = vmap.get(i, {})
            if v.get("keep") and int(v.get("score", 0)) >= 4:
                keep.append({"q": pr["q"], "a": pr["a"], "task_type": pr["task_type"],
                             "difficulty": pr["difficulty"], "source": pr["source"],
                             "score": int(v.get("score", 0))})
        return keep, u

    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(work, g) for g in groups]
        for f in as_completed(futs):
            keep, u = f.result()
            kept.extend(keep)
            usage["in"] += u["in"]; usage["out"] += u["out"]; usage["calls"] += 1

    os.makedirs(JUDGED_DIR, exist_ok=True)
    path = f"{JUDGED_DIR}/shard-{shard_id:03d}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[judge {shard_id:03d}] kept {len(kept)} | tokens in={usage['in']} out={usage['out']}")
    return {"shard": shard_id, "kept": kept, "usage": usage}


# --------------------------------------------------------------------------- #
# Cost helper (approximate public Flash rates, USD per 1M tokens)
# --------------------------------------------------------------------------- #
GEN_RATE = {"in": 0.10, "out": 0.40}     # flash-lite
JUDGE_RATE = {"in": 0.30, "out": 2.50}   # flash


def _cost(usage: dict, rate: dict) -> float:
    return usage["in"] / 1e6 * rate["in"] + usage["out"] / 1e6 * rate["out"]


def read_passages(n: int) -> list:
    import json
    rows = []
    with open(PASSAGES_PATH, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def read_jsonl(path: str) -> list:
    import json
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line))
    return out


def _map_threaded(fn, work: list[tuple]) -> list:
    """Run fn(*args) once per item in `work`, one thread per shard. Gemini calls
    are network I/O (not CPU-bound), so threads stand in for what were separate
    Modal containers -- each still internally fans out its own passage-level
    ThreadPoolExecutor, same as finetune.py."""
    if not work:
        return []
    with ThreadPoolExecutor(max_workers=len(work)) as ex:
        futs = [ex.submit(fn, *args) for args in work]
        return [f.result() for f in futs]


# --------------------------------------------------------------------------- #
# PILOT: tiny end-to-end run to check quality + real cost before scaling
# (PAID: makes real Gemini API calls)
# --------------------------------------------------------------------------- #
def cmd_pilot(n_passages: int = 20) -> None:
    print(f"== PILOT: {n_passages} passages ==")
    chunk_corpus(n_passages=n_passages)
    rows = read_passages(n_passages)
    gen = generate_shard(0, rows)
    raw = read_jsonl(f"{RAW_QA_DIR}/shard-000.jsonl")
    # regroup by passage for judging
    by_pid: dict = {}
    for r in raw:
        by_pid.setdefault(r["pid"], {"passage": r["passage"], "pairs": []})
        by_pid[r["pid"]]["pairs"].append(r)
    groups = list(by_pid.values())
    judged = judge_shard(0, groups)

    gen_cost = _cost(gen["usage"], GEN_RATE)
    judge_cost = _cost(judged["usage"], JUDGE_RATE)
    kept = judged["kept"]
    print("\n=== PILOT SAMPLES (kept) ===")
    for r in kept[:8]:
        print(f"\n[{r['source']} | {r['task_type']} | {r['difficulty']} | score {r['score']}]")
        print(f"  Q: {r['q']}")
        print(f"  A: {r['a']}")

    raw_n = gen["pairs"]
    kept_n = len(kept)
    print("\n=== PILOT ECONOMICS ===")
    print(f"passages={n_passages}  raw_pairs={raw_n}  kept={kept_n}  keep_rate={kept_n/max(1,raw_n):.0%}")
    print(f"gen cost   ${gen_cost:.4f}  ({gen['usage']})")
    print(f"judge cost ${judge_cost:.4f}  ({judged['usage']})")
    total = gen_cost + judge_cost
    per_kept = total / max(1, kept_n)
    print(f"pilot total ${total:.4f}  |  ${per_kept:.5f} per kept pair")
    print(f"PROJECTION for 5,000 kept pairs: ~${per_kept*5000:.2f}")


def cmd_build(n_passages: int = 1500, shards: int = 12) -> None:
    """Full raw-set build: chunk -> generate (parallel) -> judge (parallel).
    (PAID: makes real Gemini API calls)"""
    print(f"== BUILD raw SFT: {n_passages} passages across {shards} shards ==")
    chunk_corpus(n_passages=n_passages)
    passages = read_passages(n_passages)

    gen_work = [(i, passages[i::shards]) for i in range(shards)]
    gen = _map_threaded(generate_shard, gen_work)
    raw_total = sum(g["pairs"] for g in gen)
    gen_cost = sum(_cost(g["usage"], GEN_RATE) for g in gen)
    print(f"\ngenerated {raw_total} raw pairs | gen cost ${gen_cost:.3f}")

    all_raw = []
    for i in range(shards):
        all_raw.extend(read_jsonl(f"{RAW_QA_DIR}/shard-{i:03d}.jsonl"))
    by_pid: dict = {}
    for r in all_raw:
        by_pid.setdefault(r["pid"], {"passage": r["passage"], "pairs": []})
        by_pid[r["pid"]]["pairs"].append(r)
    groups = list(by_pid.values())

    jwork = [(i, groups[i::shards]) for i in range(shards)]
    jud = _map_threaded(judge_shard, jwork)
    kept_total = sum(len(j["kept"]) for j in jud)
    judge_cost = sum(_cost(j["usage"], JUDGE_RATE) for j in jud)
    print(f"\njudged: kept {kept_total} / {raw_total} ({kept_total/max(1,raw_total):.0%}) "
          f"| judge cost ${judge_cost:.3f}")
    print(f"TOTAL Gemini cost so far: ${gen_cost + judge_cost:.3f}")
    print("next: .venv/bin/python3 local_finetune.py curate")


# --------------------------------------------------------------------------- #
# Step 4 + 5: curate (dedup + decontaminate) -> chat format -> tokenize
# (free, no API calls)
# --------------------------------------------------------------------------- #
DATASET_DIR = f"{SFT_DIR}/dataset"
MAX_LEN = 1024
VAL_FRACTION = 0.05


def _norm_q(q: str) -> str:
    import re
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", q.lower())).strip()


def curate() -> dict:
    import glob
    import json
    import os
    import random

    from datasketch import MinHash, MinHashLSH
    from transformers import AutoTokenizer

    # ---- load all judged pairs ----
    pairs = []
    for path in sorted(glob.glob(f"{JUDGED_DIR}/*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                pairs.append(json.loads(line))
    print(f"loaded {len(pairs)} judged pairs")

    # ---- format validation ----
    def valid(p):
        q, a = p.get("q", "").strip(), p.get("a", "").strip()
        return 8 <= len(q) <= 400 and 3 <= len(a) <= 1500
    pairs = [p for p in pairs if valid(p)]

    # ---- exact-normalized dedup on the question ----
    seen, uniq = set(), []
    for p in pairs:
        n = _norm_q(p["q"])
        if n and n not in seen:
            seen.add(n)
            uniq.append(p)
    print(f"after exact-question dedup: {len(uniq)}")

    # ---- near-duplicate dedup (MinHash-LSH over question word-shingles) ----
    lsh = MinHashLSH(threshold=0.7, num_perm=64)
    kept = []
    for i, p in enumerate(uniq):
        words = _norm_q(p["q"]).split()
        shingles = {" ".join(words[j:j + 4]) for j in range(max(1, len(words) - 3))} or set(words)
        m = MinHash(num_perm=64)
        for s in shingles:
            m.update(s.encode())
        if lsh.query(m):
            continue
        lsh.insert(str(i), m)
        kept.append(p)
    print(f"after near-dup dedup: {len(kept)}")

    # ---- shuffle + split (train/val disjoint => decontaminated by construction) ----
    random.Random(1337).shuffle(kept)
    n_val = max(100, int(len(kept) * VAL_FRACTION))
    val_pairs, train_pairs = kept[:n_val], kept[n_val:]

    # ---- tokenize with our own Phase 3 tokenizer, loss-masked on the answer ----
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    sid = tok.convert_tokens_to_ids
    BOS, EOS = sid("<|bos|>"), sid("<|eos|>")
    SYS, USER, ASST = sid("<|system|>"), sid("<|user|>"), sid("<|assistant|>")
    sys_ids = tok(SYSTEM_PROMPT, add_special_tokens=False)["input_ids"]

    def encode(p):
        q = tok(p["q"], add_special_tokens=False)["input_ids"]
        a = tok(p["a"], add_special_tokens=False)["input_ids"]
        prompt = [BOS, SYS] + sys_ids + [USER] + q + [ASST]
        answer = a + [EOS]
        input_ids = prompt + answer
        labels = [-100] * len(prompt) + answer      # learn only the answer
        return input_ids[:MAX_LEN], labels[:MAX_LEN]

    os.makedirs(DATASET_DIR, exist_ok=True)

    def write_split(name, rows):
        toks = 0
        path = f"{DATASET_DIR}/{name}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for p in rows:
                ii, ll = encode(p)
                toks += sum(1 for x in ll if x != -100)   # supervised (answer) tokens
                fh.write(json.dumps({"input_ids": ii, "labels": ll}) + "\n")
        return toks

    train_answer_tokens = write_split("train", train_pairs)
    val_answer_tokens = write_split("val", val_pairs)

    # also keep a human-readable chat JSONL
    with open(f"{DATASET_DIR}/chat.jsonl", "w", encoding="utf-8") as fh:
        for p in kept:
            fh.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": p["q"]},
                {"role": "assistant", "content": p["a"]},
            ], "meta": {k: p.get(k) for k in ("source", "task_type", "difficulty", "score")}}) + "\n")

    def dist(key):
        d = {}
        for p in kept:
            d[p.get(key, "?")] = d.get(p.get(key, "?"), 0) + 1
        return d

    meta = {
        "final_pairs": len(kept), "train": len(train_pairs), "val": len(val_pairs),
        "train_answer_tokens": train_answer_tokens, "val_answer_tokens": val_answer_tokens,
        "by_source": dist("source"), "by_task": dist("task_type"), "by_difficulty": dist("difficulty"),
        "max_len": MAX_LEN, "system_prompt": SYSTEM_PROMPT, "tokenizer": config.TOKENIZER_DIR,
    }
    with open(f"{DATASET_DIR}/meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))
    return meta


def cmd_curate() -> None:
    curate()


def verify_example(idx: int = 0) -> None:
    import json

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    with open(f"{DATASET_DIR}/train.jsonl", encoding="utf-8") as fh:
        row = json.loads(fh.readlines()[idx])
    ii, ll = row["input_ids"], row["labels"]
    supervised = [t for t, l in zip(ii, ll) if l != -100]
    print(f"seq_len={len(ii)}  supervised_tokens={len(supervised)}")
    print("\n--- FULL SEQUENCE (decoded, special tokens visible) ---")
    print(tok.decode(ii, skip_special_tokens=False))
    print("\n--- SUPERVISED PART ONLY (what the model learns to produce) ---")
    print(tok.decode(supervised, skip_special_tokens=False))


def cmd_verify(idx: int = 0) -> None:
    verify_example(idx)


def cmd_chunk(n_passages: int = 2000, seed: int = 1337) -> None:
    chunk_corpus(n_passages=n_passages, seed=seed)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("chunk", help="chunk corpus into passages only (free, no API calls)")
    s.add_argument("--n-passages", type=int, default=2000)
    s.add_argument("--seed", type=int, default=1337)

    s = sub.add_parser("pilot", help="tiny end-to-end run + cost projection (PAID: real Gemini calls)")
    s.add_argument("--n-passages", type=int, default=20)

    s = sub.add_parser("build", help="full raw-set build: chunk -> generate -> judge (PAID: real Gemini calls)")
    s.add_argument("--n-passages", type=int, default=1500)
    s.add_argument("--shards", type=int, default=12)

    sub.add_parser("curate", help="dedup + decontam + chat JSONL + tokenize (free, no API calls)")

    s = sub.add_parser("verify", help="print one tokenized train example, decoded")
    s.add_argument("--idx", type=int, default=0)

    args = p.parse_args()
    print(f"DATA_ROOT={config.DATA_ROOT}\n")

    if args.cmd == "chunk":
        cmd_chunk(args.n_passages, args.seed)
    elif args.cmd == "pilot":
        cmd_pilot(args.n_passages)
    elif args.cmd == "build":
        cmd_build(args.n_passages, args.shards)
    elif args.cmd == "curate":
        cmd_curate()
    elif args.cmd == "verify":
        cmd_verify(args.idx)


if __name__ == "__main__":
    main()
