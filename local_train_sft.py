"""On-prem port of train_sft.py (Phase 8, step 2): supervised fine-tuning of
our own pretrained 125M base model on our curated Q&A set. Same logic as
train_sft.py, ported from a Modal `@app.function(gpu="L4")` step to a plain
script -- no Modal Volume, no Modal Secret, no Modal image. Reads/writes
plain local disk under config.DATA_ROOT (set SLM_DATA_ROOT, same as
local_pipeline.py / train.py).

Single GPU, full fine-tune, bf16 autocast on fp32 master weights, loss only
on the answer tokens (labels were pre-masked to -100 by local_finetune.py's
curate step). No DDP / torch.compile -- the workload is tiny (a few thousand
short examples) and single-GPU keeps it simple and robust, same as the
original L4 design; this box's B200s are more than sufficient for one GPU.

    export SLM_DATA_ROOT=/raid/llm_sec/legal-slm-125M/data
    .venv/bin/python3 local_train_sft.py --epochs 2

Requires $SLM_DATA_ROOT/sft/dataset/{train,val}.jsonl to already exist
(local_finetune.py curate) and the Phase 5 base checkpoint at
config.BASE_CKPT_DIR.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

BASE_MODEL_DIR = config.BASE_CKPT_DIR       # our own Phase 5 pretrained model
SFT_DIR = f"{config.DATA_ROOT}/sft"
DATASET_DIR = f"{SFT_DIR}/dataset"
OUT_DIR = f"{SFT_DIR}/model"


def sft(epochs: float = 3.0, lr: float = 3e-5, batch_size: int = 32,
        weight_decay: float = 0.01, warmup_frac: float = 0.03, seed: int = 1337,
        device: str = "cuda") -> dict:
    torch.manual_seed(seed)

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    pad_id = tok.convert_tokens_to_ids("<|pad|>")
    eos_id = tok.convert_tokens_to_ids("<|eos|>")

    def load(split):
        rows = []
        with open(f"{DATASET_DIR}/{split}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                rows.append(json.loads(line))
        return rows

    train = load("train")
    val = load("val")
    print(f"train={len(train)} val={len(val)} | pad_id={pad_id} eos_id={eos_id}")

    # fp32 master weights; bf16 autocast for compute (stable, no loss scaling)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_DIR, torch_dtype=torch.float32).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                  weight_decay=weight_decay)

    steps_per_epoch = math.ceil(len(train) / batch_size)
    total_steps = int(steps_per_epoch * epochs)
    warmup = max(5, int(total_steps * warmup_frac))

    def lr_at(step):
        if step < warmup:
            return lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * lr * (1 + math.cos(math.pi * min(1.0, prog)))

    def collate(rows):
        maxlen = max(len(r["input_ids"]) for r in rows)
        ii, ll, am = [], [], []
        for r in rows:
            n = len(r["input_ids"])
            pad = maxlen - n
            ii.append(r["input_ids"] + [pad_id] * pad)
            ll.append(r["labels"] + [-100] * pad)
            am.append([1] * n + [0] * pad)
        return (torch.tensor(ii, device=device),
                torch.tensor(ll, device=device),
                torch.tensor(am, device=device))

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot, seen = 0.0, 0
        for i in range(0, len(val), batch_size):
            x, y, m = collate(val[i:i + batch_size])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, attention_mask=m, labels=y).loss
            tot += loss.item() * x.size(0)
            seen += x.size(0)
        model.train()
        return tot / max(1, seen)

    print(f"steps/epoch={steps_per_epoch} total_steps={total_steps} warmup={warmup}")
    print(f"init val_loss={evaluate():.4f}")

    rng = random.Random(seed)
    step = 0
    tokens_seen = 0
    t0 = time.time()
    for ep in range(math.ceil(epochs)):
        order = list(range(len(train)))
        rng.shuffle(order)
        for i in range(0, len(train), batch_size):
            if step >= total_steps:
                break
            batch = [train[j] for j in order[i:i + batch_size]]
            x, y, m = collate(batch)
            tokens_seen += int(m.sum().item())
            cur_lr = lr_at(step)
            for g in optimizer.param_groups:
                g["lr"] = cur_lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, attention_mask=m, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 25 == 0 or step == total_steps:
                print(f"step {step:>4}/{total_steps} | loss {loss.item():.4f} | lr {cur_lr:.2e} "
                      f"| tok_seen {tokens_seen/1e6:.2f}M")
        vloss = evaluate()
        print(f"== epoch {ep+1} done | val_loss {vloss:.4f} ==")

    dt = time.time() - t0
    final_val = evaluate()
    print(f"\nFINAL val_loss {final_val:.4f} | {dt:.0f}s | tokens_seen {tokens_seen/1e6:.2f}M")

    # ---- save fine-tuned model + tokenizer ----
    model.save_pretrained(OUT_DIR, safe_serialization=True)
    tok.save_pretrained(OUT_DIR)
    print(f"saved fine-tuned model -> {OUT_DIR}")

    # ---- sample generations to eyeball behavior ----
    model.eval()
    sys = "You are a knowledgeable legal and financial assistant. Answer accurately and concisely."
    tests = [
        "What is the purpose of a Form 10-K filing?",
        "In a breach of contract claim, what must the plaintiff prove?",
        "Summarize what an indemnification clause does.",
    ]
    for q in tests:
        ids = (tok("<|bos|>", add_special_tokens=False)["input_ids"]
               + [tok.convert_tokens_to_ids("<|system|>")] + tok(sys, add_special_tokens=False)["input_ids"]
               + [tok.convert_tokens_to_ids("<|user|>")] + tok(q, add_special_tokens=False)["input_ids"]
               + [tok.convert_tokens_to_ids("<|assistant|>")])
        inp = torch.tensor([ids], device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(inp, max_new_tokens=120, do_sample=True, temperature=0.7,
                                 top_p=0.9, top_k=50, eos_token_id=eos_id, pad_token_id=pad_id)
        ans = tok.decode(out[0][len(ids):], skip_special_tokens=True)
        print("\n" + "=" * 70)
        print(f"Q: {q}")
        print(f"A: {ans}")

    return {"final_val_loss": final_val, "tokens_seen": tokens_seen,
            "steps": step, "seconds": dt}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-frac", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    print(f"DATA_ROOT={config.DATA_ROOT}\n")
    sft(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        weight_decay=args.weight_decay, warmup_frac=args.warmup_frac,
        seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
