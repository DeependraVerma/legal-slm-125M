"""13-gram contamination check: does any selected LegalBench task text
(train or test split) appear in this project's pretraining corpus?

Same method as dedup_500m.py's _build_contamination_ngrams()/decontam check
(13-word n-grams, same words()/word_ngrams() tokenizer from dedup.py), just
run standalone, after the fact, against LegalBench instead of CaseHOLD/LexGLUE
-- those two were already excluded at pretraining time (see dedup_500m.py),
LegalBench was not, so this has to be checked before trusting any LegalBench
eval number.

Scans the union of cleaned-corpus source directories across both the 125M
and 500M builds (case-law, fineweb-edu, sec, caselaw-new, sec-edgar-new) --
case-law/fineweb-edu/sec are shared verbatim between the two builds, so one
pass covers both models' contamination status.

    .venv/bin/python3 benchmarks/check_legalbench_contamination.py
"""

from __future__ import annotations

import glob
from concurrent.futures import ProcessPoolExecutor

from dedup import word_ngrams, words

NGRAM = 13

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

# Union of cleaned-corpus source dirs across both builds -- case-law/sec/
# fineweb-edu are shared verbatim between data/clean and data_500m/clean
# (confirmed identical sizes), so scanning data_500m/clean's copies covers
# the 125M model too; caselaw-new/sec-edgar-new only exist in the 500M build.
CORPUS_DIRS = [
    "data_500m/clean/case-law",
    "data_500m/clean/fineweb-edu",
    "data_500m/clean/sec",
    "data_500m/clean/caselaw-new",
    "data_500m/clean/sec-edgar-new",
]


def build_legalbench_ngrams() -> set[int]:
    from datasets import load_dataset

    grams: set[int] = set()
    n_docs = 0
    for task in LEGALBENCH_TASKS:
        for split in ("train", "test"):
            try:
                ds = load_dataset("nguha/legalbench", task, split=split, trust_remote_code=True)
            except Exception as e:
                print(f"  [legalbench] {task}/{split}: could not load ({e})")
                continue
            for rec in ds:
                text = " ".join(str(v) for v in rec.values() if isinstance(v, str))
                grams |= word_ngrams(words(text), NGRAM)
                n_docs += 1
    print(f"[legalbench-contam] {n_docs} legalbench docs -> {len(grams):,} unique {NGRAM}-grams")
    return grams


def scan_shard(args: tuple[str, set[int]]) -> tuple[str, int]:
    path, contam = args
    hits = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if word_ngrams(words(line), NGRAM) & contam:
                hits += 1
    return path, hits


def main():
    contam = build_legalbench_ngrams()

    shard_paths = []
    for d in CORPUS_DIRS:
        shard_paths += sorted(glob.glob(f"{d}/shard-*.txt"))
    print(f"[legalbench-contam] scanning {len(shard_paths)} shard files across {CORPUS_DIRS}")

    total_hits = 0
    hit_files = []
    with ProcessPoolExecutor(max_workers=min(32, __import__("os").cpu_count() or 4)) as ex:
        for path, hits in ex.map(scan_shard, [(p, contam) for p in shard_paths]):
            if hits:
                hit_files.append((path, hits))
                total_hits += hits
            print(f"  {path}: {hits} contaminated lines")

    print()
    if total_hits == 0:
        print("[legalbench-contam] RESULT: CLEAN -- zero 13-gram overlaps found. "
              "Safe to report LegalBench results on both models.")
    else:
        print(f"[legalbench-contam] RESULT: CONTAMINATED -- {total_hits} overlapping lines "
              f"across {len(hit_files)} shard files:")
        for path, hits in hit_files:
            print(f"  {path}: {hits}")


if __name__ == "__main__":
    main()
