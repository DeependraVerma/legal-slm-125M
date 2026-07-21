"""Measure real, cleaned-token yield for the two candidate new sources
(TeraflopAI/SEC-EDGAR, common-pile/caselaw_access_project) for the planned
500M-param model -- mirrors local_pipeline.py's measure_sources() exactly
(same clean_document() call, same config.CHARS_PER_TOKEN extrapolation), but
deliberately does NOT touch config.DATA_MIX, so the live 125M pipeline's
config is untouched. Read-only: streams a sample, prints stats, writes
nothing to disk (same as the existing measure step).

Real total document counts (needed to extrapolate from a sample to the full
source), pulled directly from each dataset's own README/stats table, not
estimated:
  - TeraflopAI/SEC-EDGAR: 8,055,455 (README's own per-filing-type table)
  - common-pile/caselaw_access_project: 6,919,240 (README's own stats table;
    78GB raw -- NOT the 24.3GB figure from an earlier search result, which
    must have referred to a filtered variant)

    export SLM_DATA_ROOT=/raid/llm_sec/legal-slm-125M/data
    .venv/bin/python3 measure_new_sources.py
"""

from __future__ import annotations

import config
from config import Source

CANDIDATES: tuple[Source, ...] = (
    Source("sec-edgar-new", "TeraflopAI/SEC-EDGAR", 0, "text", split="train"),
    Source("caselaw-new", "common-pile/caselaw_access_project", 0, "text", split="train"),
)

# TeraflopAI/SEC-EDGAR: streaming the FULL repo (2,556 parquet files across
# all 10 filing types, no loading script) is unreliable -- hits either slow
# file-list discovery or a genuinely bad shard somewhere in the full set
# (confirmed: "Couldn't deserialize thrift" error, reproduced twice).
# Targeting specific filing-type folders via `data_files` works cleanly and
# is also the better corpus choice: 10-K/10-Q/8-K are the prose-rich
# disclosure filings (similar register to the existing SEC source), vs.
# S-1/20-F/Forms 3-4-5 which are registration boilerplate / terse insider
# trading forms.
SEC_DATA_FILES = ["10-K/*.parquet", "10-Q/*.parquet", "8-K/*.parquet"]
SEC_TOTAL_DOCS = 223_275 + 674_240 + 1_952_207  # 10-K + 10-Q + 8-K, from the dataset's own README table

TOTAL_ROWS = {
    "sec-edgar-new": SEC_TOTAL_DOCS,
    "caselaw-new": 6_919_240,
}


def _stream_sec_edgar_robust(n: int):
    """Per-file streaming with per-file error handling -- confirmed (3x
    reproduced) that some parquet shards in TeraflopAI/SEC-EDGAR are
    corrupt ("Couldn't deserialize thrift"). The high-level
    load_dataset(streaming=True) iterator can't skip a bad shard mid-stream,
    it just crashes the whole run -- so this lists files explicitly and
    catches errors per file instead."""
    import fnmatch

    import pandas as pd
    from huggingface_hub import HfApi

    api = HfApi()
    all_files = api.list_repo_files("TeraflopAI/SEC-EDGAR", repo_type="dataset")
    files = [f for f in all_files if any(fnmatch.fnmatch(f, pat) for pat in SEC_DATA_FILES)]
    import random
    random.Random(1337).shuffle(files)

    yielded = 0
    bad_files = 0
    for f in files:
        if yielded >= n:
            break
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download("TeraflopAI/SEC-EDGAR", f, repo_type="dataset")
            df = pd.read_parquet(path)
        except Exception as e:
            bad_files += 1
            print(f"  [skip corrupt shard] {f}: {str(e)[:100]}", flush=True)
            continue
        for _, row in df.iterrows():
            yield row.to_dict()
            yielded += 1
            if yielded >= n:
                break
    if bad_files:
        print(f"  ({bad_files} corrupt shard(s) skipped out of shards sampled)", flush=True)


def _stream_source(source: Source, n: int):
    from datasets import load_dataset

    if source.name == "sec-edgar-new":
        yield from _stream_sec_edgar_robust(n)
        return
    ds = load_dataset(source.hf_id, source.config_name, split=source.split, streaming=True)
    for i, record in enumerate(ds):
        if i >= n:
            break
        yield record


def measure(n_per_source: int = 2000) -> dict:
    from cleaning import clean_document

    out: dict[str, dict] = {}
    for source in CANDIDATES:
        clean_chars = kept = streamed = 0
        for record in _stream_source(source, n_per_source):
            streamed += 1
            text = record.get(source.text_field) or ""
            if not isinstance(text, str):
                text = str(text)
            r = clean_document(text)
            if r.kept:
                kept += 1
                clean_chars += r.clean_chars
        avg_clean = clean_chars / streamed if streamed else 0
        total = TOTAL_ROWS[source.name]
        est = total * avg_clean / config.CHARS_PER_TOKEN
        out[source.name] = {
            "sampled": streamed, "keep_rate": round(kept / streamed, 3) if streamed else 0,
            "avg_clean_chars_per_doc": round(avg_clean), "total_docs": total,
            "est_clean_tokens": int(est),
        }
        print(f"{source.name:<16} sampled={streamed:>5} keep={kept/streamed:.0%}  "
              f"avg_clean={avg_clean:>7.0f} ch/doc  total_docs={total:>10,}  "
              f"est_clean_tokens={est/1e9:.2f}B", flush=True)
    total_est = sum(v["est_clean_tokens"] for v in out.values())
    print(f"\nTOTAL new-source est clean tokens: {total_est/1e9:.2f}B")
    print(f"Plus existing corpus (~2.04B unique from case-law/SEC/fineweb-edu already in {config.CORPUS_DIR})")
    print(f"Combined estimate: {(total_est + 2_040_000_000)/1e9:.2f}B unique tokens")
    return out


if __name__ == "__main__":
    measure()
