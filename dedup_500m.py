"""Phase 2 (dedup + decontam) for the 500M build -- adapted from
local_pipeline.py's cmd_dedup(), generalized in three ways the original
hardcoded for exactly the 3 original sources:

1. Shard counts are discovered dynamically (glob), not a hardcoded
   {"case-law": 10, "sec": 5, "fineweb-edu": 5} dict -- the two new sources
   have 28 shards each (one per ingestion worker), not 5 or 10.
2. Near-dup MinHash dedup now runs across {case-law, caselaw-new} TOGETHER,
   not just case-law alone. Real reason, not just generalization for its own
   sake: HFforLegal/case-law and common-pile/caselaw_access_project both
   ultimately draw from the same underlying pool of US court opinions (the
   Caselaw Access Project / CourtListener), so the same opinion could
   plausibly appear in both sources under different formatting -- training
   on it twice would be a real duplication the original single-source
   near-dup pass can't catch.
3. Decontamination (against CaseHOLD/LexGLUE) now covers all 4 case-law-like
   and SEC-like sources (case-law, caselaw-new, sec, sec-edgar-new), not just
   the original 2 -- same contamination risk applies to the new sources
   since they're the same domain.

Exact-dedup stays per-shard (same scope limitation as the original -- a
`seen` hash set is local to each shard file, not global across the whole
corpus). Matches the original design's accepted tradeoff, not a new gap
introduced here.

    export SLM_DATA_ROOT_500M=/raid/llm_sec/legal-slm-125M/data_500m
    .venv/bin/python3 dedup_500m.py
"""

from __future__ import annotations

import glob
import json
import os
import urllib.request
from concurrent.futures import ProcessPoolExecutor

import config_500m as config

SHINGLE_K = 5
MINHASH_PERM = 32
MINHASH_THRESHOLD = 0.8
DECONTAM_NGRAM = 13
SIG_DIR = f"{config.DATA_ROOT}/tmp/minhash_sigs"
NEAR_DUPS_PATH = f"{config.DATA_ROOT}/tmp/near_dups.json"

# The real generalization vs. the 125M pipeline: both case-law-flavored and
# both SEC-flavored sources participate in near-dup + decontam together.
CASELAW_SOURCES = {"case-law", "caselaw-new"}
DECONTAM_SOURCES = {"case-law", "caselaw-new", "sec", "sec-edgar-new"}


def _nproc() -> int:
    return min(32, os.cpu_count() or 4)


def _shard_files(source_name: str) -> list[str]:
    return sorted(os.path.basename(p) for p in
                  glob.glob(f"{config.CLEAN_DIR}/{source_name}/shard-*.txt"))


def _parquet_urls(hf_id: str, config_name: str, split: str) -> list[str]:
    """Fetches pre-converted parquet URLs from HF's datasets-server API,
    exactly matching local_pipeline.py's original helper. Deliberately NOT
    using load_dataset(hf_id, ...) directly -- casehold/casehold requires
    trust_remote_code=True via its own loading script (confirmed: crashed a
    full dedup run on this), while its datasets-server-converted parquet
    files load with zero custom code."""
    api = f"https://datasets-server.huggingface.co/parquet?dataset={hf_id}"
    req = urllib.request.Request(api, headers={"User-Agent": "slm-500m"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return [f["url"] for f in data.get("parquet_files", [])
            if f.get("config") == config_name and f.get("split") == split]


def _build_contamination_ngrams() -> set:
    from datasets import load_dataset

    from dedup import word_ngrams, words

    grams: set = set()
    for hf_id, cfg_name in [("casehold/casehold", "all"), ("coastalcph/lex_glue", "case_hold")]:
        try:
            urls = _parquet_urls(hf_id, cfg_name, "test")
            if not urls:
                urls = _parquet_urls(hf_id, cfg_name, "train")
            ds = load_dataset("parquet", data_files=urls, split="train", streaming=True)
            for rec in ds:
                text = " ".join(str(v) for v in rec.values() if isinstance(v, str))
                grams |= word_ngrams(words(text), DECONTAM_NGRAM)
        except Exception as e:
            print(f"  [decontam] could not load {hf_id}: {e}")
    print(f"  [decontam] {len(grams):,} eval 13-grams loaded")
    return grams


def minhash_shard(source_name: str, shard_basename: str) -> dict:
    import numpy as np
    from datasketch import MinHash

    from dedup import shingles, words

    path = f"{config.CLEAN_DIR}/{source_name}/{shard_basename}"
    sigs, idxs = [], []
    with open(path, encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.rstrip("\n")
            if not line:
                continue
            m = MinHash(num_perm=MINHASH_PERM)
            sh = list(shingles(words(line), SHINGLE_K))
            if sh:
                m.update_batch(sh)
            sigs.append(m.hashvalues.astype(np.uint64))
            idxs.append(idx)
    os.makedirs(SIG_DIR, exist_ok=True)
    key = f"{source_name}__{shard_basename}"
    np.savez(f"{SIG_DIR}/{key}.npz",
             sigs=np.vstack(sigs) if sigs else np.zeros((0, MINHASH_PERM), dtype=np.uint64),
             idxs=np.asarray(idxs, dtype=np.int64))
    print(f"[minhash {key}] {len(idxs):,} docs", flush=True)
    return {"key": key, "n": len(idxs)}


def build_near_dups() -> int:
    import numpy as np
    from datasketch import MinHash, MinHashLSH

    near: dict[str, list[int]] = {}
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_PERM)
    for npz_path in sorted(glob.glob(f"{SIG_DIR}/*.npz")):
        key = os.path.basename(npz_path)[: -len(".npz")]
        data = np.load(npz_path)
        for row, idx in zip(data["sigs"], data["idxs"]):
            m = MinHash(num_perm=MINHASH_PERM, hashvalues=row)
            if lsh.query(m):
                near.setdefault(key, []).append(int(idx))
            else:
                lsh.insert(f"{key}:{int(idx)}", m)
    os.makedirs(os.path.dirname(NEAR_DUPS_PATH), exist_ok=True)
    with open(NEAR_DUPS_PATH, "w", encoding="utf-8") as fh:
        json.dump(near, fh)
    total = sum(len(v) for v in near.values())
    print(f"[near-dups] {total:,} near-duplicates across {CASELAW_SOURCES}")
    return total


def write_corpus_shard(source_name: str, shard_basename: str) -> dict:
    from dedup import exact_hash, word_ngrams, words

    near: set[int] = set()
    if source_name in CASELAW_SOURCES:
        key = f"{source_name}__{shard_basename}"
        with open(NEAR_DUPS_PATH, encoding="utf-8") as fh:
            near = set(json.load(fh).get(key, []))
    # matches the original 125M pipeline's approach exactly: rebuild the
    # contamination n-gram set fresh inside each worker (not passed in) --
    # a set this size doesn't survive being closed over in a function passed
    # to ProcessPoolExecutor (nested closures aren't picklable), and the
    # original design already accepted the redundant-fetch cost per shard.
    contam = _build_contamination_ngrams() if source_name in DECONTAM_SOURCES else None
    in_path = f"{config.CLEAN_DIR}/{source_name}/{shard_basename}"
    out_dir = f"{config.CORPUS_DIR}/{source_name}"
    os.makedirs(out_dir, exist_ok=True)
    seen: set[str] = set()
    kept = clean_chars = 0
    reasons = {"near_dup": 0, "exact_dup": 0, "contaminated": 0, "kept": 0}
    with open(in_path, encoding="utf-8") as fin, \
            open(f"{out_dir}/{shard_basename}", "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            text = line.rstrip("\n")
            if not text:
                continue
            if idx in near:
                reasons["near_dup"] += 1
                continue
            h = exact_hash(text)
            if h in seen:
                reasons["exact_dup"] += 1
                continue
            if contam and (word_ngrams(words(text), DECONTAM_NGRAM) & contam):
                reasons["contaminated"] += 1
                continue
            seen.add(h)
            fout.write(text + "\n")
            kept += 1
            clean_chars += len(text)
            reasons["kept"] += 1
    print(f"[corpus {source_name}/{shard_basename}] kept={kept} drops={reasons}", flush=True)
    return {"source": source_name, "shard": shard_basename, "kept": kept,
            "est_tokens": int(clean_chars / config.CHARS_PER_TOKEN), "reasons": reasons}


def cmd_dedup() -> None:
    sources = [s.name for s in config.DATA_MIX]
    caselaw_work = [(src, sb) for src in sources if src in CASELAW_SOURCES for sb in _shard_files(src)]
    print(f"1/3 MinHash signatures for {len(caselaw_work)} case-law-flavored shards "
          f"(sources: {sorted(CASELAW_SOURCES)})...")
    with ProcessPoolExecutor(max_workers=_nproc()) as ex:
        list(ex.map(minhash_shard, *zip(*caselaw_work)))

    print("2/3 building near-dup set (LSH)...")
    build_near_dups()

    all_work = [(src, sb) for src in sources for sb in _shard_files(src)]
    print(f"3/3 writing final corpus ({len(all_work)} shards, parallel; "
          f"decontam applied to {sorted(DECONTAM_SOURCES)})...")
    with ProcessPoolExecutor(max_workers=_nproc()) as ex:
        results = list(ex.map(write_corpus_shard, *zip(*all_work)))

    report: dict[str, dict] = {}
    for r in results:
        agg = report.setdefault(r["source"], {"kept": 0, "est_tokens": 0,
              "reasons": {"near_dup": 0, "exact_dup": 0, "contaminated": 0, "kept": 0}})
        agg["kept"] += r["kept"]
        agg["est_tokens"] += r["est_tokens"]
        for k, v in r["reasons"].items():
            agg["reasons"][k] = agg["reasons"].get(k, 0) + v
    total = sum(v["est_tokens"] for v in report.values())
    print("PHASE 2 REPORT")
    for name, a in report.items():
        print(f"  {name:<16} kept={a['kept']:>8} est_tokens={a['est_tokens']/1e9:.2f}B drops={a['reasons']}")
    print(f"  TOTAL corpus est tokens: {total/1e9:.2f}B")
    with open(f"{config.CORPUS_DIR}/phase2_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    cmd_dedup()
