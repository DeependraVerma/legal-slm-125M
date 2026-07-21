"""Phase 4 (tokenize + pack) for the 500M build -- adapted from
local_pipeline.py's cmd_tokenize(), generalized to discover which sources
and how many corpus shards exist dynamically (glob), instead of a hardcoded
{"case-law": 4, "sec": 6, "fineweb-edu": 4} dict that doesn't know about the
2 new sources. Everything else (window packing, 99/1 train/val split, uint16
dtype) is identical to the original -- reuses the SAME tokenizer
(config_500m.TOKENIZER_DIR points at the existing 125M build's tokenizer, a
deliberate reuse decision, not a placeholder).

    export SLM_DATA_ROOT_500M=/raid/llm_sec/legal-slm-125M/data_500m
    .venv/bin/python3 tokenize_500m.py
"""

from __future__ import annotations

import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import config_500m as config

ENCODE_BATCH = 1_000
SHARDS_PER_SOURCE = 8  # parallelism per source; actual shard COUNT of corpus
#  files varies per source (10-27), this just controls tokenize-phase fan-out


def _nproc() -> int:
    return min(32, os.cpu_count() or 4)


def tokenize_shard(source_name: str, shard_index: int, num_shards: int) -> dict:
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    eos_id = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
    seq_len = config.SEQ_LEN
    os.makedirs(config.TRAIN_TOKENS_DIR, exist_ok=True)
    os.makedirs(config.VAL_TOKENS_DIR, exist_ok=True)
    train_path = f"{config.TRAIN_TOKENS_DIR}/{source_name}-{shard_index:03d}.bin"
    val_path = f"{config.VAL_TOKENS_DIR}/{source_name}-{shard_index:03d}.bin"
    buf: list[int] = []
    win_count = n_train = n_val = 0
    corpus_files = sorted(glob.glob(f"{config.CORPUS_DIR}/{source_name}/*.txt"))

    def _doc_iter():
        for path in corpus_files:
            with open(path, encoding="utf-8") as fh:
                for idx, line in enumerate(fh):
                    if idx % num_shards == shard_index:
                        line = line.rstrip("\n")
                        if line:
                            yield line

    with open(train_path, "wb") as ftr, open(val_path, "wb") as fva:
        batch: list[str] = []

        def _flush():
            nonlocal win_count, n_train, n_val
            if not batch:
                return
            for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                buf.extend(ids)
                buf.append(eos_id)
            while len(buf) >= seq_len:
                window = np.asarray(buf[:seq_len], dtype=np.uint16)
                del buf[:seq_len]
                if win_count % config.VAL_EVERY_N_WINDOWS == 0:
                    window.tofile(fva)
                    n_val += 1
                else:
                    window.tofile(ftr)
                    n_train += 1
                win_count += 1

        for doc in _doc_iter():
            batch.append(doc)
            if len(batch) >= ENCODE_BATCH:
                _flush()
                batch = []
        _flush()
    print(f"[{source_name} {shard_index:03d}] train_win={n_train} val_win={n_val} "
          f"train_tok={n_train*seq_len/1e6:.1f}M", flush=True)
    return {"source": source_name, "shard": shard_index, "train_windows": n_train,
            "val_windows": n_val, "train_tokens": n_train * seq_len, "val_tokens": n_val * seq_len}


def cmd_tokenize() -> None:
    sources = [s.name for s in config.DATA_MIX]
    work = [(name, i, SHARDS_PER_SOURCE) for name in sources for i in range(SHARDS_PER_SOURCE)]
    print(f"Launching {len(work)} tokenize workers across {len(sources)} sources "
          f"({SHARDS_PER_SOURCE} shards/source, {_nproc()} parallel)...")
    with ProcessPoolExecutor(max_workers=_nproc()) as ex:
        results = list(ex.map(tokenize_shard, *zip(*work)))
    total = {"seq_len": config.SEQ_LEN, "dtype": config.TOKENS_DTYPE,
             "train_windows": sum(r["train_windows"] for r in results),
             "val_windows": sum(r["val_windows"] for r in results),
             "train_tokens": sum(r["train_tokens"] for r in results),
             "val_tokens": sum(r["val_tokens"] for r in results), "shards": results}
    with open(f"{config.TOKENS_DIR}/index.json", "w", encoding="utf-8") as fh:
        json.dump(total, fh, indent=2)
    print(f"index: train={total['train_tokens']/1e9:.2f}B tok ({total['train_windows']} win), "
          f"val={total['val_tokens']/1e6:.1f}M tok ({total['val_windows']} win)")


if __name__ == "__main__":
    cmd_tokenize()
