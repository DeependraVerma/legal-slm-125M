"""Phase 5: pretrain the 125M Llama SLM on the packed uint16 corpus.

Run under torchrun (one process per GPU); all hyperparameters come from config.py.
    torchrun --standalone --nproc_per_node=8 train.py
Controlled by env vars set by the Modal wrapper: EPOCHS, SMOKE_STEPS, COMPILE.
"""

from __future__ import annotations

import glob
import json
import math
import os
import time
from contextlib import nullcontext
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import LlamaConfig, LlamaForCausalLM

import config

SEQ_LEN = config.SEQ_LEN
T = config.TRAIN
H100_BF16_PEAK = 989e12  # per-GPU bf16 flops, for MFU


def is_master(rank: int) -> bool:
    return rank == 0


def log(rank: int, *a):
    if is_master(rank):
        print(*a, flush=True)


class WindowStore:
    """Random-access view over packed uint16 .bin files as (N, SEQ_LEN) windows."""

    def __init__(self, directory: str):
        self.files = sorted(glob.glob(f"{directory}/*.bin"))
        self.mmaps = [np.memmap(f, dtype=np.uint16, mode="r") for f in self.files]
        counts = [m.shape[0] // SEQ_LEN for m in self.mmaps]
        self.cum = np.cumsum([0] + counts)
        self.total = int(self.cum[-1])

    def gather(self, idxs) -> torch.Tensor:
        out = np.empty((len(idxs), SEQ_LEN), dtype=np.int64)
        for j, g in enumerate(idxs):
            i = int(np.searchsorted(self.cum, g, side="right") - 1)
            loc = int(g - self.cum[i])
            out[j] = self.mmaps[i][loc * SEQ_LEN:(loc + 1) * SEQ_LEN]
        return torch.from_numpy(out)


def lr_at(step: int, max_steps: int) -> float:
    """Linear warmup then cosine decay to min_lr, measured in tokens."""
    gbt = T.global_batch_tokens
    tokens = step * gbt
    if tokens < T.warmup_tokens:
        return T.lr * tokens / max(1, T.warmup_tokens)
    max_tokens = max_steps * gbt
    prog = (tokens - T.warmup_tokens) / max(1, max_tokens - T.warmup_tokens)
    prog = min(1.0, prog)
    return T.min_lr + 0.5 * (T.lr - T.min_lr) * (1.0 + math.cos(math.pi * prog))


def make_optimizer(model):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": T.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=T.lr, betas=(T.beta1, T.beta2), fused=True)


def commit_volume():
    try:
        import modal
        modal.Volume.from_name(config.VOLUME_NAME).commit()
    except Exception as e:  # exit-commit still persists on normal return
        print(f"  [warn] volume commit failed: {e}", flush=True)


@torch.no_grad()
def evaluate(model, val: WindowStore, device, micro: int, max_windows: int = 512) -> float:
    """Eval on the RAW module (never DDP/compiled) on rank 0 only.

    Do NOT call .eval()/.train() here: this Llama has no dropout/batchnorm so mode
    is a no-op, but toggling it forces a torch.compile recompile of the *training*
    graph, which desyncs DDP collectives and deadlocks NCCL. And never pass the
    compiled/DDP module — its embedded collectives would hang when run on one rank.
    """
    n = min(max_windows, val.total)
    total, seen = 0.0, 0
    for start in range(0, n, micro):
        idxs = list(range(start, min(start + micro, n)))
        x = val.gather(idxs).to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=x, labels=x).loss
        total += loss.item() * len(idxs)
        seen += len(idxs)
    return total / max(1, seen)


def main():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl", timeout=timedelta(minutes=5))
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
    else:
        rank, local_rank, world = 0, 0, 1
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    torch.manual_seed(T.seed + rank)
    torch.set_float32_matmul_precision("high")

    epochs = float(os.environ.get("EPOCHS", "2"))
    smoke = int(os.environ.get("SMOKE_STEPS", "0"))
    do_compile = os.environ.get("COMPILE", "1") == "1"

    # ---- data ----
    train = WindowStore(config.TRAIN_TOKENS_DIR)
    val = WindowStore(config.VAL_TOKENS_DIR)
    gbw = T.global_batch_tokens // SEQ_LEN                 # 512 windows / opt step
    per_rank = gbw // world                                # windows this rank handles / step
    grad_accum = per_rank // T.micro_batch_size            # micro-steps / opt step
    assert per_rank == grad_accum * T.micro_batch_size, (per_rank, grad_accum)
    steps_per_epoch = train.total // gbw
    max_steps = smoke if smoke > 0 else int(steps_per_epoch * epochs)
    ckpt_every = min(T.ckpt_every_steps, max_steps) if smoke == 0 else max_steps
    log_every = 5 if smoke else T.log_every_steps
    eval_every = T.eval_every_steps

    log(rank, f"world={world} grad_accum={grad_accum} per_rank={per_rank} "
              f"gbw={gbw} steps/epoch={steps_per_epoch} max_steps={max_steps} "
              f"train_windows={train.total} val_windows={val.total} smoke={smoke}")

    # ---- model ----
    cfg = LlamaConfig(**config.MODEL.to_llama_kwargs())
    cfg._attn_implementation = "sdpa"
    raw = LlamaForCausalLM(cfg).to(device)
    n_params = sum(p.numel() for p in raw.parameters())
    log(rank, f"model params: {n_params:,} (~{n_params/1e6:.1f}M)")
    model = DDP(raw, device_ids=[local_rank]) if ddp else raw
    step_model = torch.compile(model) if do_compile else model
    optimizer = make_optimizer(raw)

    ckpt_path = (f"{config.CKPT_DIR}/ckpt_smoke.pt" if smoke
                 else config.RESUME_CKPT_PATH)
    os.makedirs(config.CKPT_DIR, exist_ok=True)
    os.makedirs(config.BASE_CKPT_DIR, exist_ok=True)

    # ---- resume (full runs only) ----
    start_step = 0
    if smoke == 0 and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        raw.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])
        start_step = int(ck["step"])
        log(rank, f"resumed from step {start_step}")

    flops_per_tok = 6 * n_params
    perm_cache: dict[int, np.ndarray] = {}

    def epoch_perm(ep: int) -> np.ndarray:
        if ep not in perm_cache:
            perm_cache.clear()
            g = torch.Generator().manual_seed(T.seed + ep)
            perm_cache[ep] = torch.randperm(train.total, generator=g).numpy()
        return perm_cache[ep]

    model.train()
    t0 = time.time()
    for step in range(start_step, max_steps):
        lr = lr_at(step, max_steps if smoke == 0 else int(steps_per_epoch * epochs))
        for grp in optimizer.param_groups:
            grp["lr"] = lr

        ep = step // steps_per_epoch
        in_ep = step % steps_per_epoch
        perm = epoch_perm(ep)
        base = in_ep * gbw + rank * per_rank
        my = perm[base:base + per_rank]

        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=device)
        for m in range(grad_accum):
            sl = my[m * T.micro_batch_size:(m + 1) * T.micro_batch_size]
            x = train.gather(sl).to(device, non_blocking=True)
            sync_ctx = model.no_sync() if (ddp and m < grad_accum - 1) else nullcontext()
            with sync_ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = step_model(input_ids=x, labels=x).loss
                (loss / grad_accum).backward()
            loss_accum += loss.detach() / grad_accum
        norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), T.grad_clip)
        optimizer.step()

        # NOTE: log rank-0's local loss only. Do NOT add a manual all_reduce here:
        # under torch.compile it reorders collectives vs the compiled DDP graph and
        # deadlocks NCCL. Local loss is representative for logging.
        if step % log_every == 0 or step == max_steps - 1:
            torch.cuda.synchronize()
            dt = time.time() - t0
            t0 = time.time()
            done = max(1, step - start_step) if step == start_step else log_every
            tok_s = T.global_batch_tokens * (log_every if step else 1) / max(1e-6, dt)
            mfu = flops_per_tok * tok_s / (world * H100_BF16_PEAK)
            log(rank, f"step {step:>5}/{max_steps} | loss {loss_accum.item():.4f} | "
                      f"lr {lr:.2e} | grad_norm {norm.item():.2f} | "
                      f"{tok_s/1e3:.0f}k tok/s | mfu {mfu:.1%}")
            if is_master(rank):
                with open(config.METRICS_PATH, "a") as fh:
                    fh.write(json.dumps({"step": step, "loss": loss_accum.item(),
                                         "lr": lr, "tok_s": tok_s, "mfu": mfu}) + "\n")

        if step > 0 and step % eval_every == 0:
            if ddp:
                dist.barrier()
            if is_master(rank):
                vloss = evaluate(raw, val, device, T.micro_batch_size)
                log(rank, f"  [eval] step {step} val_loss {vloss:.4f} ppl {math.exp(vloss):.1f}")
                with open(config.METRICS_PATH, "a") as fh:
                    fh.write(json.dumps({"step": step, "val_loss": vloss}) + "\n")
            if ddp:
                dist.barrier()

        if is_master(rank) and step > 0 and step % ckpt_every == 0:
            torch.save({"model": raw.state_dict(), "optim": optimizer.state_dict(),
                        "step": step + 1, "model_cfg": config.MODEL.to_llama_kwargs()},
                       ckpt_path)
            commit_volume()
            log(rank, f"  [ckpt] saved step {step+1} -> {ckpt_path}")

    # ---- final eval + save ----
    if ddp:
        dist.barrier()
    if is_master(rank):
        vloss = evaluate(raw, val, device, T.micro_batch_size)
        log(rank, f"FINAL val_loss {vloss:.4f} ppl {math.exp(vloss):.1f}")
        torch.save({"model": raw.state_dict(), "optim": optimizer.state_dict(),
                    "step": max_steps, "model_cfg": config.MODEL.to_llama_kwargs()},
                   ckpt_path)
        if smoke == 0:
            raw.save_pretrained(config.BASE_CKPT_DIR)
            log(rank, f"saved HF model -> {config.BASE_CKPT_DIR}")
        commit_volume()
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
