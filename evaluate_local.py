"""On-prem port of modal_app.py's evaluate_model (Phase 6): full held-out
perplexity + sample generations from the pretrained base model. Single GPU,
inference only (no DDP needed).

    .venv/bin/python3 evaluate_local.py
    .venv/bin/python3 evaluate_local.py --prompt "Some legal sentence to continue"
    .venv/bin/python3 evaluate_local.py --skip-ppl --prompt "..." --prompt "..."
"""

from __future__ import annotations

import argparse
import glob
import math

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

DEFAULT_PROMPTS = [
    "The plaintiff shall bear the burden of",
    "Pursuant to the terms of this Agreement, the parties",
    "The Company's net revenues for the fiscal year",
    "IN THE UNITED STATES DISTRICT COURT FOR THE",
    "Notwithstanding any provision herein to the contrary,",
    "The defendant moved for summary judgment on the grounds that",
]


def full_val_perplexity(model, device) -> tuple[float, int, int]:
    seq = config.SEQ_LEN
    total_loss, total_tok, nwin_all = 0.0, 0, 0
    with torch.no_grad():
        for f in sorted(glob.glob(f"{config.VAL_TOKENS_DIR}/*.bin")):
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
    return total_loss / total_tok, total_tok, nwin_all


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
    p.add_argument("--prompt", action="append", dest="prompts", default=None,
                    help="Custom prompt (repeatable). Defaults to a fixed legal/financial set.")
    p.add_argument("--n-gen-tokens", type=int, default=120)
    p.add_argument("--skip-ppl", action="store_true", help="Skip the full val-set perplexity pass")
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
        vloss, total_tok, nwin = full_val_perplexity(model, device)
        print(f"\nFULL VAL perplexity: {math.exp(vloss):.3f}  (loss {vloss:.4f}) "
              f"over {total_tok:,} tokens / {nwin:,} windows\n")

    for prompt in prompts:
        text = generate(model, tok, device, prompt, args.n_gen_tokens)
        print("=" * 78)
        print(f"PROMPT: {prompt}")
        print(f"-> {text}")


if __name__ == "__main__":
    main()
