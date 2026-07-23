"""Locate and print the ACTUAL overlapping 13-word phrase (not just a
truncated document prefix) for a small sample of contamination hits, to
tell genuine LegalBench-document leakage apart from generic-legal-boilerplate
false positives (a common phrase like "shall be governed by and construed in
accordance with the laws of" legitimately recurs across thousands of
unrelated real contracts -- a match on that is not evidence the LegalBench
test document itself leaked into training).
"""

from __future__ import annotations

from dedup import word_ngrams, words

LEGALBENCH_TASKS = [
    "contract_nli_confidentiality_of_agreement",
    "contract_qa",
    "consumer_contracts_qa",
    "citation_prediction_classification",
]

SAMPLE_FILES = [
    "data_500m/clean/fineweb-edu/shard-000.txt",
    "data_500m/clean/case-law/shard-000.txt",
    "data_500m/clean/caselaw-new/shard-008.txt",  # the 1020-hit outlier
]


def build_gram_index() -> dict[int, tuple[str, list[str]]]:
    """gram_hash -> (source_task, the actual 13 words)"""
    from datasets import load_dataset

    idx: dict[int, tuple[str, list[str]]] = {}
    for task in LEGALBENCH_TASKS:
        for split in ("train", "test"):
            try:
                ds = load_dataset("nguha/legalbench", task, split=split, trust_remote_code=True)
            except Exception:
                continue
            for rec in ds:
                text = " ".join(str(v) for v in rec.values() if isinstance(v, str))
                toks = words(text)
                if len(toks) < 13:
                    continue
                for i in range(len(toks) - 12):
                    gram = tuple(toks[i : i + 13])
                    idx.setdefault(hash(gram), (task, list(gram)))
    return idx


def find_matching_window(line_tokens: list[str], target_words: list[str]) -> str | None:
    n = len(target_words)
    for i in range(len(line_tokens) - n + 1):
        if line_tokens[i : i + n] == target_words:
            lo = max(0, i - 8)
            hi = min(len(line_tokens), i + n + 8)
            return " ".join(line_tokens[lo:hi])
    return None


def main():
    idx = build_gram_index()
    contam = set(idx.keys())
    print(f"[diag] {len(idx):,} unique legalbench 13-grams")

    for path in SAMPLE_FILES:
        print(f"\n=== {path} ===")
        shown = 0
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                toks = words(line)
                if len(toks) < 13:
                    continue
                grams = word_ngrams(toks, 13)
                hits = grams & contam
                if hits:
                    g = next(iter(hits))
                    task, target_words = idx[g]
                    window = find_matching_window(toks, target_words)
                    print(f"  [{task}] matched phrase: {' '.join(target_words)!r}")
                    print(f"  corpus context: ...{window}...")
                    shown += 1
                    if shown >= 4:
                        break


if __name__ == "__main__":
    main()
