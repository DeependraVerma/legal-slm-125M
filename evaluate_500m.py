"""Phase 6 (evaluate) for the 500M build. Same full-val-set perplexity
methodology as evaluate_local.py (not the cheap 512-window sample train.py's
own in-loop eval uses), adapted for config_500m, plus a per-source
breakdown -- this model's val set spans 5 domains (vs the 125M model's 3),
so an aggregate number alone can't say whether the new sources specifically
helped or dragged things down.

    export SLM_DATA_ROOT_500M=/raid/llm_sec/legal-slm-125M/data_500m
    export SLM_DATA_ROOT=/raid/llm_sec/legal-slm-125M/data
    .venv/bin/python3 evaluate_500m.py
"""

from __future__ import annotations

import argparse
import glob
import math

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config_500m as config

DEFAULT_PROMPTS = [
    "The plaintiff shall bear the burden of",
    "Pursuant to the terms of this Agreement, the parties",
    "The Company's net revenues for the fiscal year",
    "IN THE UNITED STATES DISTRICT COURT FOR THE",
    "Notwithstanding any provision herein to the contrary,",
    "The defendant moved for summary judgment on the grounds that",
]


def _source_files(src: str) -> list[str]:
    """Exact-prefix match on `{src}-` followed by digits, NOT a shell glob --
    glob(f'{src}-*.bin') for src='sec' also matches 'sec-edgar-new-*.bin'
    (shares the 'sec-' prefix), silently blending two sources' windows
    together. Confirmed: this bug inflated the first run's 'sec' row into a
    sec+sec-edgar-new mix (96,132 windows reported vs. the true ~8,176)."""
    import re
    pat = re.compile(rf"^{re.escape(src)}-\d+\.bin$")
    return sorted(f for f in glob.glob(f"{config.VAL_TOKENS_DIR}/*.bin")
                  if pat.match(f.split("/")[-1]))


def full_val_perplexity(model, device, files: list[str] | None = None) -> tuple[float, int, int]:
    seq = config.SEQ_LEN
    total_loss, total_tok, nwin_all = 0.0, 0, 0
    file_list = files if files is not None else sorted(glob.glob(f"{config.VAL_TOKENS_DIR}/*.bin"))
    with torch.no_grad():
        for f in file_list:
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


def generate(model, tok, device, prompt: str, n_gen_tokens: int) -> str:
    eos_id = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n_gen_tokens, do_sample=True,
                              top_k=50, top_p=0.95, temperature=0.8,
                              eos_token_id=eos_id, pad_token_id=eos_id)
    return tok.decode(out[0], skip_special_tokens=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", action="append", dest="prompts", default=None)
    p.add_argument("--n-gen-tokens", type=int, default=120)
    p.add_argument("--skip-ppl", action="store_true")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    prompts = args.prompts or list(DEFAULT_PROMPTS)
    device = args.device

    print(f"loading tokenizer from {config.TOKENIZER_DIR}")
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    print(f"loading model from {config.BASE_CKPT_DIR}")
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_CKPT_DIR, torch_dtype=torch.bfloat16).to(device).eval()

    if not args.skip_ppl:
        sources = [s.name for s in config.DATA_MIX]
        print("\n=== per-source held-out perplexity ===")
        for src in sources:
            files = _source_files(src)
            vloss, ntok, nwin = full_val_perplexity(model, device, files=files)
            if nwin == 0:
                continue
            print(f"  {src:<16} val_loss={vloss:.4f} ppl={math.exp(vloss):.2f} "
                  f"({nwin} windows, {ntok/1e6:.2f}M tokens)")

        print("\n=== full aggregate held-out perplexity ===")
        vloss, ntok, nwin = full_val_perplexity(model, device)
        print(f"ALL SOURCES: val_loss={vloss:.4f} ppl={math.exp(vloss):.3f} "
              f"({nwin} windows, {ntok/1e6:.1f}M tokens)")

    print("\n=== sample generations ===")
    for pr in prompts:
        out = generate(model, tok, device, pr, args.n_gen_tokens)
        print(f"\nPROMPT: {pr}")
        print(f"OUTPUT: {out}")


if __name__ == "__main__":
    main()
