"""Phase 1 (clean) for the two new 500M-build sources only -- the original 3
sources' cleaned output was copied over from the 125M build's /data/clean
(same source, same cleaning config, no reason to re-stream/re-clean).

Deliberately NOT using local_pipeline.py's existing clean_shard()/_parquet_urls()
path: that assumes HF's datasets-server parquet API cleanly enumerates shards
by (config, split), which doesn't fit TeraflopAI/SEC-EDGAR's per-filing-type
subfolder layout, and more importantly a real fraction of its parquet shards
are corrupt ("Couldn't deserialize thrift", reproduced 3x during measurement)
-- the existing streaming path has no way to skip a bad shard mid-stream, it
just crashes. This script lists files explicitly and processes file-by-file
with per-file error handling, matching what measure_new_sources.py already
proved works.

Output format matches the existing pipeline exactly: one cleaned document per
line, written to $DATA_ROOT/clean/{source_name}/shard-NNN.txt -- so Phase 2
(dedup) and beyond need zero changes to consume this alongside the copied-over
original 3 sources.

    export SLM_DATA_ROOT_500M=/raid/llm_sec/legal-slm-125M/data_500m
    .venv/bin/python3 ingest_500m_new_sources.py --workers 24
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import config_500m as config
from cleaning import clean_document

SOURCES_BY_NAME = {s.name: s for s in config.DATA_MIX}


def _list_files(source_name: str) -> list[str]:
    from huggingface_hub import HfApi

    source = SOURCES_BY_NAME[source_name]
    api = HfApi()
    all_files = api.list_repo_files(source.hf_id, repo_type="dataset")
    if source.data_files:
        pats = source.data_files if isinstance(source.data_files, tuple) else (source.data_files,)
        files = [f for f in all_files if any(fnmatch.fnmatch(f, p) for p in pats)]
    else:
        files = [f for f in all_files if f.endswith((".parquet", ".jsonl.gz", ".jsonl.xz"))]
    random.Random(1337).shuffle(files)
    return files


def _rows_from_file(hf_id: str, filename: str):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(hf_id, filename, repo_type="dataset")
    if filename.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            yield row.to_dict()
    elif filename.endswith(".jsonl.gz"):
        import gzip
        import json
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)
    else:
        raise ValueError(f"unsupported file type: {filename}")


def clean_worker(source_name: str, files: list[str], worker_id: int, token_cap: int) -> dict:
    source = SOURCES_BY_NAME[source_name]
    out_dir = f"{config.CLEAN_DIR}/{source_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/shard-{worker_id:03d}.txt"

    streamed = kept = clean_chars = bad_files = 0
    reasons: dict[str, int] = {}
    with open(out_path, "w", encoding="utf-8") as fh:
        for filename in files:
            if clean_chars / config.CHARS_PER_TOKEN >= token_cap:
                break
            try:
                rows = list(_rows_from_file(source.hf_id, filename))
            except Exception as e:
                bad_files += 1
                print(f"  [w{worker_id:02d}] skip corrupt file {filename}: {str(e)[:100]}", flush=True)
                continue
            for record in rows:
                streamed += 1
                text = record.get(source.text_field) or ""
                if not isinstance(text, str):
                    text = str(text)
                r = clean_document(text, strict_ocr=source.strict_ocr)
                reasons[r.reason] = reasons.get(r.reason, 0) + 1
                if r.kept:
                    fh.write(r.text.replace("\n", " ").strip() + "\n")
                    kept += 1
                    clean_chars += r.clean_chars
                if clean_chars / config.CHARS_PER_TOKEN >= token_cap:
                    break
    est_tokens = int(clean_chars / config.CHARS_PER_TOKEN)
    print(f"[{source_name} worker {worker_id:03d}] streamed={streamed} kept={kept} "
          f"est_tokens={est_tokens/1e6:.1f}M bad_files={bad_files}", flush=True)
    return {"source": source_name, "worker": worker_id, "streamed": streamed,
            "kept": kept, "est_tokens": est_tokens, "bad_files": bad_files, "reasons": reasons}


def ingest_source(source_name: str, n_workers: int) -> dict:
    source = SOURCES_BY_NAME[source_name]
    files = _list_files(source_name)
    print(f"{source_name}: {len(files)} files available, token_budget={source.token_budget/1e9:.1f}B, "
          f"splitting across {n_workers} workers", flush=True)
    # round-robin files across workers so each worker has a large, diverse pool
    # to draw from (needed since per-file doc counts vary a lot, esp. for SEC-EDGAR)
    worker_files = [files[i::n_workers] for i in range(n_workers)]
    per_worker_cap = source.token_budget // n_workers

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(clean_worker, source_name, worker_files[i], i, per_worker_cap): i
                for i in range(n_workers)}
        for f in as_completed(futs):
            results.append(f.result())

    total_tokens = sum(r["est_tokens"] for r in results)
    total_kept = sum(r["kept"] for r in results)
    total_streamed = sum(r["streamed"] for r in results)
    total_bad = sum(r["bad_files"] for r in results)
    print(f"\n{source_name} DONE: streamed={total_streamed} kept={total_kept} "
          f"est_tokens={total_tokens/1e9:.2f}B bad_files_skipped={total_bad}")
    return {"source": source_name, "est_tokens": total_tokens, "kept": total_kept,
            "streamed": total_streamed, "bad_files": total_bad}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--only", default="", help="restrict to one source name")
    args = ap.parse_args()

    print(f"DATA_ROOT: {config.DATA_ROOT}\n")
    targets = ["sec-edgar-new", "caselaw-new"]
    if args.only:
        targets = [args.only]
    summary = {}
    for name in targets:
        summary[name] = ingest_source(name, args.workers)
    print("\n=== FINAL SUMMARY ===")
    for name, s in summary.items():
        print(f"  {name}: {s['est_tokens']/1e9:.2f}B tokens, {s['bad_files']} bad files skipped")


if __name__ == "__main__":
    main()
