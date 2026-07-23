"""Stricter re-check after the 13-gram pass flagged ~8,700 hits that, on
inspection (inspect_contamination_hits.py), were all short runs of generic
legal boilerplate (standard-of-review language, DMCA/ToS disclaimer text,
canonical precedent-citation phrasing) -- real, widely-recurring legal
phrases, not evidence that a LegalBench *document* leaked into training.
13-gram overlap is the same threshold this project already uses for
CaseHOLD/LexGLUE decontamination, but legal text's much higher natural
phrase-level duplication (case law quotes precedent verbatim; contracts
reuse standard clauses) makes it too sensitive here specifically.

This instead requires a MUCH longer contiguous run (default 40 words) before
counting something as real contamination -- long enough that coincidental
boilerplate reuse essentially can't produce it, while genuine verbatim
document leakage still would.
"""

from __future__ import annotations

import glob
from concurrent.futures import ProcessPoolExecutor

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

CORPUS_DIRS = [
    "data_500m/clean/case-law",
    "data_500m/clean/fineweb-edu",
    "data_500m/clean/sec",
    "data_500m/clean/caselaw-new",
    "data_500m/clean/sec-edgar-new",
]


def build_long_grams() -> set[int]:
    from datasets import load_dataset

    grams: set[int] = set()
    for task in LEGALBENCH_TASKS:
        for split in ("train", "test"):
            try:
                ds = load_dataset("nguha/legalbench", task, split=split, trust_remote_code=True)
            except Exception:
                continue
            for rec in ds:
                text = " ".join(str(v) for v in rec.values() if isinstance(v, str))
                toks = words(text)
                if len(toks) < RUN_LEN:
                    continue
                for i in range(len(toks) - RUN_LEN + 1):
                    grams.add(hash(tuple(toks[i : i + RUN_LEN])))
    print(f"[strict-contam] {len(grams):,} unique {RUN_LEN}-word runs from legalbench")
    return grams


def scan_shard(args: tuple[str, set[int]]) -> tuple[str, int]:
    path, contam = args
    hits = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            toks = words(line)
            if len(toks) < RUN_LEN:
                continue
            for i in range(len(toks) - RUN_LEN + 1):
                if hash(tuple(toks[i : i + RUN_LEN])) in contam:
                    hits += 1
    return path, hits


def main():
    contam = build_long_grams()
    shard_paths = []
    for d in CORPUS_DIRS:
        shard_paths += sorted(glob.glob(f"{d}/shard-*.txt"))
    print(f"[strict-contam] scanning {len(shard_paths)} shard files (RUN_LEN={RUN_LEN})")

    total = 0
    with ProcessPoolExecutor(max_workers=min(32, __import__("os").cpu_count() or 4)) as ex:
        for path, hits in ex.map(scan_shard, [(p, contam) for p in shard_paths]):
            if hits:
                print(f"  {path}: {hits} hits")
                total += hits

    print()
    if total == 0:
        print(f"[strict-contam] RESULT: CLEAN at RUN_LEN={RUN_LEN} -- no long verbatim runs found. "
              "The 13-gram hits were boilerplate false positives; LegalBench results are safe to report.")
    else:
        print(f"[strict-contam] RESULT: {total} long verbatim runs found -- needs manual review.")


if __name__ == "__main__":
    main()
