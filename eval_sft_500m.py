"""Full held-out evaluation of the 500M build's SFT model -- identical
methodology to eval_sft.py (same CUAD F1 / LEDGAR exact-match / Llama-70B-
judged closed-book correctness), just pointed at the new model. The val
split itself is read from the ORIGINAL 125M dataset location (same reasoning
as local_train_sft_500m.py: the dataset is tokenizer-dependent only, and the
tokenizer is shared -- nothing to rebuild).

    export SLM_DATA_ROOT=/raid/llm_sec/legal-slm-125M/data
    .venv/bin/python3 eval_sft_500m.py --device cuda:1
"""

from __future__ import annotations

import argparse
import json
import re

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

SFT_DIR = f"{config.DATA_ROOT}/sft"
DATASET_DIR = f"{SFT_DIR}/dataset"
MODEL_DIR = "/raid/llm_sec/legal-slm-125M/data_500m/sft/model"
RESULTS_PATH = "/raid/llm_sec/legal-slm-125M/data_500m/sft/eval_results.jsonl"
SYSTEM_PROMPT = "You are a knowledgeable legal and financial assistant. Answer accurately and concisely."
LLAMA70B_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"

REFUSAL = "this excerpt does not appear to address that."


def load_val() -> list[dict]:
    with open(f"{DATASET_DIR}/meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    n_val = meta["val"]
    rows = []
    with open(f"{DATASET_DIR}/chat.jsonl", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= n_val:
                break
            rows.append(json.loads(line))
    return rows


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def token_f1(pred: str, gold: str) -> float:
    p, g = _norm(pred).split(), _norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = {}
    for t in p:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in g:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    prec = overlap / len(p)
    rec = overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def generate_batch(model, tok, device, user_msgs: list[str], max_new_tokens: int = 150) -> list[str]:
    sys_ids = tok(SYSTEM_PROMPT, add_special_tokens=False)["input_ids"]
    bos, sysid, userid, asstid, eos, pad = (
        tok.convert_tokens_to_ids(t) for t in
        ("<|bos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|eos|>", "<|pad|>")
    )
    seqs = []
    for u in user_msgs:
        u_ids = tok(u, add_special_tokens=False)["input_ids"][:800]
        seqs.append([bos, sysid] + sys_ids + [userid] + u_ids + [asstid])
    maxlen = max(len(s) for s in seqs)
    input_ids = torch.tensor([[pad] * (maxlen - len(s)) + s for s in seqs], device=device)
    attn = torch.tensor([[0] * (maxlen - len(s)) + [1] * len(s) for s in seqs], device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(input_ids, attention_mask=attn, max_new_tokens=max_new_tokens,
                              do_sample=False, eos_token_id=eos, pad_token_id=pad)
    preds = []
    for i, s in enumerate(seqs):
        gen = out[i][maxlen:]
        preds.append(tok.decode(gen, skip_special_tokens=True).strip())
    return preds


def llama_judge(question: str, gold: str, pred: str) -> bool | None:
    prompt = (
        "You are grading a small language model's answer against a reference answer.\n"
        f"QUESTION: {question}\n"
        f"REFERENCE ANSWER: {gold}\n"
        f"MODEL ANSWER: {pred}\n\n"
        "Is the MODEL ANSWER factually consistent with and substantively responsive to the "
        "REFERENCE ANSWER (paraphrase/partial-but-correct counts as yes; missing/contradictory/"
        "hallucinated facts count as no)? Reply with ONLY one word: yes or no."
    )
    try:
        r = requests.post(LLAMA70B_ENDPOINT, json={
            "model": "llama_70b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": 5,
        }, timeout=60)
        if r.status_code != 200:
            return None
        text = r.json()["choices"][0]["message"]["content"].strip().lower()
        return text.startswith("y")
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    print(f"MODEL_DIR={MODEL_DIR}")
    val = load_val()
    print(f"loaded {len(val)} val examples")

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).to(args.device)
    model.eval()

    results = []
    for i in range(0, len(val), args.batch_size):
        batch = val[i:i + args.batch_size]
        user_msgs = [b["messages"][1]["content"] for b in batch]
        preds = generate_batch(model, tok, args.device, user_msgs)
        for b, pred in zip(batch, preds):
            results.append({
                "source": b["meta"]["source"], "task_type": b["meta"]["task_type"],
                "question": b["messages"][1]["content"], "gold": b["messages"][2]["content"],
                "pred": pred,
            })
        print(f"  generated {min(i + args.batch_size, len(val))}/{len(val)}", flush=True)

    cuad_extract_f1, cuad_refusal_correct, cuad_refusal_total = [], 0, 0
    ledgar_correct, ledgar_total = 0, 0
    judge_pool = []
    for r in results:
        if r["source"] == "cuad":
            if _norm(r["gold"]) == _norm(REFUSAL):
                cuad_refusal_total += 1
                if _norm(r["pred"]) == _norm(REFUSAL) or "does not" in r["pred"].lower() or "not addressed" in r["pred"].lower():
                    cuad_refusal_correct += 1
            else:
                cuad_extract_f1.append(token_f1(r["pred"], r["gold"]))
        elif r["source"] == "ledgar":
            ledgar_total += 1
            if _norm(r["pred"]) == _norm(r["gold"]):
                ledgar_correct += 1
        else:
            judge_pool.append(r)

    print(f"\njudging {len(judge_pool)} closed-book (case-law/sec/fineweb-edu) examples via local Llama-70B...")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    judged = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(llama_judge, r["question"], r["gold"], r["pred"]): idx
                for idx, r in enumerate(judge_pool)}
        done = 0
        for f in as_completed(futs):
            judged[futs[f]] = f.result()
            done += 1
            if done % 50 == 0:
                print(f"  judged {done}/{len(judge_pool)}", flush=True)

    by_source_judge = {}
    for idx, r in enumerate(judge_pool):
        v = judged.get(idx)
        d = by_source_judge.setdefault(r["source"], {"correct": 0, "total": 0, "unjudged": 0})
        d["total"] += 1
        if v is True:
            d["correct"] += 1
        elif v is None:
            d["unjudged"] += 1

    print("\n" + "=" * 70)
    print("EVAL SUMMARY (full val split, n=%d)" % len(val))
    print("=" * 70)
    if cuad_extract_f1:
        print(f"CUAD extraction token-F1 (n={len(cuad_extract_f1)}): "
              f"mean={sum(cuad_extract_f1)/len(cuad_extract_f1):.3f}")
    if cuad_refusal_total:
        print(f"CUAD refusal accuracy (n={cuad_refusal_total}): "
              f"{cuad_refusal_correct}/{cuad_refusal_total} = {cuad_refusal_correct/cuad_refusal_total:.1%}")
    if ledgar_total:
        print(f"LEDGAR classification exact-match (n={ledgar_total}): "
              f"{ledgar_correct}/{ledgar_total} = {ledgar_correct/ledgar_total:.1%}")
    for src, d in by_source_judge.items():
        graded = d["total"] - d["unjudged"]
        rate = d["correct"] / graded if graded else float("nan")
        print(f"{src} closed-book Llama-70B-judged correctness (n={graded}, "
              f"{d['unjudged']} unjudged/errored): {d['correct']}/{graded} = {rate:.1%}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nfull per-example results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
