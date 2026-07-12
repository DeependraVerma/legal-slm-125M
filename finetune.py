"""Phase 1 of fine-tuning: build a grounded Q&A SFT dataset with a Gemini teacher,
then tokenize it with the MENTOR's tokenizer (thesreedath/slm-125m-base).

Pipeline (all fanned out on Modal):
    chunk_corpus  -> /data/sft/passages.jsonl
    generate      -> /data/sft/raw_qa/*.jsonl   (Gemini writes grounded Q&A)
    judge         -> /data/sft/judged/*.jsonl   (Gemini LLM-as-judge keeps good ones)
    curate        -> /data/sft/dataset/*        (dedup + decontam + chat JSONL + tokens)

Run pieces with:  modal run finetune.py::pilot   (tiny end-to-end sanity + cost)
"""

from __future__ import annotations

import modal

import config

app = modal.App("slm-125m-sft")

MENTOR_MODEL = "thesreedath/slm-125m-base"
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

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "requests==2.32.3",
        "huggingface_hub==0.34.4",
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "numpy==1.26.4",
        "datasketch==1.6.5",
    )
    .add_local_python_source("config")
)
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}
gemini_secret = modal.Secret.from_name("gemini-api")


# --------------------------------------------------------------------------- #
# Gemini REST helper (thinking disabled for cost; retries on transient errors)
# --------------------------------------------------------------------------- #
def _gemini(model: str, prompt: str, *, temperature: float, max_tokens: int,
            api_key: str) -> tuple[str, dict]:
    import json
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
# Step 1: chunk the cleaned corpus into grounded passages
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes=VOLUMES, timeout=60 * 20)
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
    volume.commit()
    by_src = {s: sum(1 for p in picked if p["source"] == s) for s in SOURCE_WEIGHTS}
    print(f"chunked {len(picked)} passages: {by_src}")
    return {"n": len(picked), "by_source": by_src}


# --------------------------------------------------------------------------- #
# Step 2 + 3: generate and judge (one worker per shard, threaded within)
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes=VOLUMES, secrets=[gemini_secret],
              timeout=60 * 60, cpu=2.0)
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
    volume.commit()
    print(f"[gen {shard_id:03d}] {len(passages)} passages -> {len(out_rows)} raw pairs "
          f"| tokens in={usage['in']} out={usage['out']}")
    return {"shard": shard_id, "pairs": len(out_rows), "usage": usage}


@app.function(image=image, volumes=VOLUMES, secrets=[gemini_secret],
              timeout=60 * 60, cpu=2.0)
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
    volume.commit()
    print(f"[judge {shard_id:03d}] kept {len(kept)} | tokens in={usage['in']} out={usage['out']}")
    return {"shard": shard_id, "kept": kept, "usage": usage}


# --------------------------------------------------------------------------- #
# Cost helper (approximate public Flash rates, USD per 1M tokens)
# --------------------------------------------------------------------------- #
GEN_RATE = {"in": 0.10, "out": 0.40}     # flash-lite
JUDGE_RATE = {"in": 0.30, "out": 2.50}   # flash


def _cost(usage: dict, rate: dict) -> float:
    return usage["in"] / 1e6 * rate["in"] + usage["out"] / 1e6 * rate["out"]


# --------------------------------------------------------------------------- #
# PILOT: tiny end-to-end run to check quality + real cost before scaling
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def pilot(n_passages: int = 20):
    import json

    print(f"== PILOT: {n_passages} passages ==")
    chunk_corpus.remote(n_passages=n_passages)
    # read passages back locally via a tiny function to keep everything on-volume
    rows = list(read_passages.remote(n_passages))
    gen = generate_shard.remote(0, rows)
    raw = list(read_jsonl.remote(f"{RAW_QA_DIR}/shard-000.jsonl"))
    # regroup by passage for judging
    by_pid: dict = {}
    for r in raw:
        by_pid.setdefault(r["pid"], {"passage": r["passage"], "pairs": []})
        by_pid[r["pid"]]["pairs"].append(r)
    groups = list(by_pid.values())
    judged = judge_shard.remote(0, groups)

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


@app.local_entrypoint()
def build(n_passages: int = 1500, shards: int = 12):
    """Full raw-set build: chunk -> generate (parallel) -> judge (parallel)."""
    print(f"== BUILD raw SFT: {n_passages} passages across {shards} shards ==")
    chunk_corpus.remote(n_passages=n_passages)
    passages = list(read_passages.remote(n_passages))

    gen_work = [(i, passages[i::shards]) for i in range(shards)]
    gen = list(generate_shard.starmap(gen_work))
    raw_total = sum(g["pairs"] for g in gen)
    gen_cost = sum(_cost(g["usage"], GEN_RATE) for g in gen)
    print(f"\ngenerated {raw_total} raw pairs | gen cost ${gen_cost:.3f}")

    all_raw = []
    for i in range(shards):
        all_raw.extend(read_jsonl.remote(f"{RAW_QA_DIR}/shard-{i:03d}.jsonl"))
    by_pid: dict = {}
    for r in all_raw:
        by_pid.setdefault(r["pid"], {"passage": r["passage"], "pairs": []})
        by_pid[r["pid"]]["pairs"].append(r)
    groups = list(by_pid.values())

    jwork = [(i, groups[i::shards]) for i in range(shards)]
    jud = list(judge_shard.starmap(jwork))
    kept_total = sum(len(j["kept"]) for j in jud)
    judge_cost = sum(_cost(j["usage"], JUDGE_RATE) for j in jud)
    print(f"\njudged: kept {kept_total} / {raw_total} ({kept_total/max(1,raw_total):.0%}) "
          f"| judge cost ${judge_cost:.3f}")
    print(f"TOTAL Gemini cost so far: ${gen_cost + judge_cost:.3f}")
    print("next: modal run finetune.py::curate")


# --------------------------------------------------------------------------- #
# Step 4 + 5: curate (dedup + decontaminate) -> chat format -> tokenize
# --------------------------------------------------------------------------- #
DATASET_DIR = f"{SFT_DIR}/dataset"
MENTOR_TOK_DIR = f"{SFT_DIR}/mentor_tokenizer"
MAX_LEN = 1024
VAL_FRACTION = 0.05


def _norm_q(q: str) -> str:
    import re
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", q.lower())).strip()


@app.function(image=image, volumes=VOLUMES, timeout=60 * 30, cpu=4.0, memory=8_192)
def curate() -> dict:
    import glob
    import json
    import os
    import random

    import numpy as np
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

    # ---- tokenize with the MENTOR tokenizer, loss-masked on the answer ----
    tok = AutoTokenizer.from_pretrained(MENTOR_MODEL, cache_dir=MENTOR_TOK_DIR)
    tok.save_pretrained(MENTOR_TOK_DIR)
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
        "max_len": MAX_LEN, "system_prompt": SYSTEM_PROMPT, "tokenizer": MENTOR_MODEL,
    }
    with open(f"{DATASET_DIR}/meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    volume.commit()
    print(json.dumps(meta, indent=2))
    return meta


@app.local_entrypoint()
def curate_run():
    curate.remote()


@app.function(image=image, volumes=VOLUMES)
def verify_example(idx: int = 0) -> None:
    import json

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MENTOR_TOK_DIR)
    with open(f"{DATASET_DIR}/train.jsonl", encoding="utf-8") as fh:
        row = json.loads(fh.readlines()[idx])
    ii, ll = row["input_ids"], row["labels"]
    supervised = [t for t, l in zip(ii, ll) if l != -100]
    print(f"seq_len={len(ii)}  supervised_tokens={len(supervised)}")
    print("\n--- FULL SEQUENCE (decoded, special tokens visible) ---")
    print(tok.decode(ii, skip_special_tokens=False))
    print("\n--- SUPERVISED PART ONLY (what the model learns to produce) ---")
    print(tok.decode(supervised, skip_special_tokens=False))


@app.local_entrypoint()
def verify(idx: int = 0):
    verify_example.remote(idx)


@app.function(image=image, volumes=VOLUMES)
def read_passages(n: int) -> list:
    import json
    rows = []
    with open(PASSAGES_PATH, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


@app.function(image=image, volumes=VOLUMES)
def read_jsonl(path: str) -> list:
    import json
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line))
    return out
