"""Builds a full per-question comparison between the 125M and 500M SFT
models' held-out eval predictions -- both ran on the exact same 1,059-example
val split in the same order (verified: zero question/gold mismatches), so
predictions pair up by index directly.

CUAD/LEDGAR correctness is recomputed directly (has ground truth, no LLM
call needed). Closed-book (case-law/sec/fineweb-edu) needs a fresh Llama-70B
judge pass for BOTH models' answers -- the original eval runs only saved
aggregate correctness, not per-example verdicts, so this re-judges all of
them (both models' predictions, same question) to get a real paired
correct/incorrect signal for the report, not just side-by-side text.

Writes a single JSON file consumed by the HTML report artifact.

    .venv/bin/python3 compare_125m_500m.py
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

REFUSAL = "this excerpt does not appear to address that."
LLAMA70B_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"


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
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def llama_judge(question: str, gold: str, pred: str) -> bool | None:
    prompt = (
        "You are grading a small language model's answer against a reference answer.\n"
        f"QUESTION: {question}\nREFERENCE ANSWER: {gold}\nMODEL ANSWER: {pred}\n\n"
        "Is the MODEL ANSWER factually consistent with and substantively responsive to the "
        "REFERENCE ANSWER (paraphrase/partial-but-correct counts as yes; missing/contradictory/"
        "hallucinated facts count as no)? Reply with ONLY one word: yes or no."
    )
    try:
        r = requests.post(LLAMA70B_ENDPOINT, json={
            "model": "llama_70b", "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": 5}, timeout=60)
        if r.status_code != 200:
            return None
        return r.json()["choices"][0]["message"]["content"].strip().lower().startswith("y")
    except Exception:
        return None


def main() -> None:
    a = [json.loads(l) for l in open("/raid/llm_sec/legal-slm-125M/data/sft/eval_results.jsonl")]
    b = [json.loads(l) for l in open("/raid/llm_sec/legal-slm-125M/data_500m/sft/eval_results.jsonl")]
    assert len(a) == len(b)
    assert all(x["question"] == y["question"] and x["gold"] == y["gold"] for x, y in zip(a, b))
    print(f"paired {len(a)} examples")

    merged = []
    judge_jobs = []  # (row_idx, which, question, gold, pred)
    for i, (ra, rb) in enumerate(zip(a, b)):
        row = {"source": ra["source"], "task_type": ra["task_type"],
               "question": ra["question"], "gold": ra["gold"],
               "pred_125m": ra["pred"], "pred_500m": rb["pred"]}
        if ra["source"] == "cuad":
            if _norm(ra["gold"]) == _norm(REFUSAL):
                row["correct_125m"] = _norm(ra["pred"]) == _norm(REFUSAL) or "does not" in ra["pred"].lower()
                row["correct_500m"] = _norm(rb["pred"]) == _norm(REFUSAL) or "does not" in rb["pred"].lower()
                row["metric"] = "refusal"
            else:
                f1a, f1b = token_f1(ra["pred"], ra["gold"]), token_f1(rb["pred"], ra["gold"])
                row["correct_125m"], row["correct_500m"] = f1a >= 0.5, f1b >= 0.5
                row["f1_125m"], row["f1_500m"] = round(f1a, 3), round(f1b, 3)
                row["metric"] = "f1"
        elif ra["source"] == "ledgar":
            row["correct_125m"] = _norm(ra["pred"]) == _norm(ra["gold"])
            row["correct_500m"] = _norm(rb["pred"]) == _norm(ra["gold"])
            row["metric"] = "exact_match"
        else:
            judge_jobs.append((i, "125m", ra["question"], ra["gold"], ra["pred"]))
            judge_jobs.append((i, "500m", ra["question"], ra["gold"], rb["pred"]))
            row["metric"] = "llm_judge"
        merged.append(row)

    print(f"judging {len(judge_jobs)} closed-book predictions (both models) via local Llama-70B...")
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(llama_judge, q, g, p): (i, which) for i, which, q, g, p in judge_jobs}
        done = 0
        for f in as_completed(futs):
            i, which = futs[f]
            merged[i][f"correct_{which}"] = f.result()
            done += 1
            if done % 100 == 0:
                print(f"  judged {done}/{len(judge_jobs)}", flush=True)

    # ---- aggregate summary ----
    by_source: dict[str, dict] = {}
    for row in merged:
        d = by_source.setdefault(row["source"], {"n": 0, "125m": 0, "500m": 0, "both_wrong": 0,
                                                    "500m_fixed": 0, "125m_only": 0, "both_right": 0})
        d["n"] += 1
        c125, c500 = row.get("correct_125m"), row.get("correct_500m")
        if c125 is None or c500 is None:
            continue
        d["125m"] += int(c125)
        d["500m"] += int(c500)
        if c125 and c500:
            d["both_right"] += 1
        elif c125 and not c500:
            d["125m_only"] += 1
        elif c500 and not c125:
            d["500m_fixed"] += 1
        else:
            d["both_wrong"] += 1

    print("\n=== per-source accuracy comparison ===")
    for src, d in by_source.items():
        print(f"  {src:<12} n={d['n']:>4}  125M={d['125m']/d['n']:.1%}  500M={d['500m']/d['n']:.1%}  "
              f"(500M fixed {d['500m_fixed']}, regressed {d['125m_only']}, both right {d['both_right']}, both wrong {d['both_wrong']})")

    out = {"summary": by_source, "rows": merged}
    with open("/raid/llm_sec/legal-slm-125M/data_500m/sft/comparison_125m_500m.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\nwrote data_500m/sft/comparison_125m_500m.json")


if __name__ == "__main__":
    main()
