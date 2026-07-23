"""For the 356 strict (40-word) contamination hits, identify which
LegalBench task each came from, and whether it's genuine document-template
leakage (contract_nli/contract_qa/consumer_contracts_qa -- would mean an
actual NDA/ToS document text appears in both places) vs. expected same-
public-corpus overlap (citation_prediction_classification's "text" field is
itself an excerpt of real historical case law, which the pretraining corpus
independently also contains -- overlap there isn't test-answer leakage,
it's two datasets drawing from the same public domain source).
"""

from __future__ import annotations

import glob
from collections import Counter

from dedup import words

RUN_LEN = 40

LEGALBENCH_TASKS = [
    "contract_nli_confidentiality_of_agreement",
    "contract_nli_explicit_identification",
    "contract_nli_limited_use",
    "contract_nli_no_licensing",
    "contract_nli_notice_on_compelled_disclosure",
    "contract_nli_permissible_copy",
    "contract_nli_return_of_confidential_information",
    "contract_nli_sharing_with_employees",
    "contract_nli_survival_of_obligations",
    "contract_qa",
    "consumer_contracts_qa",
    "citation_prediction_classification",
]

HOT_FILES = [
    "data_500m/clean/case-law/shard-001.txt",
    "data_500m/clean/sec-edgar-new/shard-004.txt",
    "data_500m/clean/caselaw-new/shard-008.txt",
]


def build_index():
    from datasets import load_dataset

    idx = {}
    for task in LEGALBENCH_TASKS:
        for split in ("train", "test"):
            try:
                ds = load_dataset("nguha/legalbench", task, split=split, trust_remote_code=True)
            except Exception:
                continue
            for row_i, rec in enumerate(ds):
                text = " ".join(str(v) for v in rec.values() if isinstance(v, str))
                toks = words(text)
                if len(toks) < RUN_LEN:
                    continue
                for i in range(len(toks) - RUN_LEN + 1):
                    idx.setdefault(hash(tuple(toks[i : i + RUN_LEN])), (task, split, row_i))
    return idx


def main():
    idx = build_index()
    contam = set(idx.keys())
    print(f"[attribute] {len(idx):,} 40-word runs indexed")

    task_counts = Counter()
    doc_ids_by_task = {}
    example_shown = {}

    for path in HOT_FILES:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                toks = words(line)
                if len(toks) < RUN_LEN:
                    continue
                for i in range(len(toks) - RUN_LEN + 1):
                    h = hash(tuple(toks[i : i + RUN_LEN]))
                    if h in contam:
                        task, split, row_i = idx[h]
                        task_counts[task] += 1
                        doc_ids_by_task.setdefault(task, set()).add((split, row_i))
                        if task not in example_shown:
                            example_shown[task] = " ".join(toks[i : i + RUN_LEN])

    print("\nHits by LegalBench task:")
    for task, n in task_counts.most_common():
        n_docs = len(doc_ids_by_task[task])
        print(f"  {task}: {n} 40-word hits, from {n_docs} distinct legalbench doc(s)")
        print(f"    example: {example_shown[task][:250]}")


if __name__ == "__main__":
    main()
