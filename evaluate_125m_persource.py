"""Per-source held-out perplexity breakdown for the ORIGINAL 125M base model
-- needed for a fair comparison against the 500M build's per-source numbers,
since evaluate_local.py only ever reported one blended aggregate (7.763)
across its 3-source val set. Same exact-prefix file matching as
evaluate_500m.py (no shell-glob prefix-collision risk here since case-law/
sec/fineweb-edu don't share prefixes, but using the same safe method anyway).

    .venv/bin/python3 evaluate_125m_persource.py
"""

from __future__ import annotations

import glob
import math
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


def _source_files(src: str) -> list[str]:
    pat = re.compile(rf"^{re.escape(src)}-\d+\.bin$")
    return sorted(f for f in glob.glob(f"{config.VAL_TOKENS_DIR}/*.bin")
                  if pat.match(f.split("/")[-1]))


def full_val_perplexity(model, device, files: list[str]) -> tuple[float, int, int]:
    seq = config.SEQ_LEN
    total_loss, total_tok, nwin_all = 0.0, 0, 0
    with torch.no_grad():
        for f in files:
            arr = np.memmap(f, dtype=np.uint16, mode="r")
            nwin = arr.shape[0] // seq
            nwin_all += nwin
            for i in range(0, nwin, 32):
                rows = [arr[j * seq:(j + 1) * seq].astype(np.int64)
                        for j in range(i, min(i + 32, nwin))]
                x = torch.tensor(np.stack(rows), device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=x, labels=x).loss
                ntok = x.numel() - x.shape[0]
                total_loss += loss.item() * ntok
                total_tok += ntok
    return total_loss / max(1, total_tok), total_tok, nwin_all


def main() -> None:
    device = "cuda:2"
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_CKPT_DIR, torch_dtype=torch.bfloat16).to(device).eval()

    print("\n=== 125M per-source held-out perplexity ===")
    for src in ["case-law", "sec", "fineweb-edu"]:
        files = _source_files(src)
        vloss, ntok, nwin = full_val_perplexity(model, device, files)
        print(f"  {src:<16} val_loss={vloss:.4f} ppl={math.exp(vloss):.2f} "
              f"({nwin} windows, {ntok/1e6:.2f}M tokens)")


if __name__ == "__main__":
    main()
