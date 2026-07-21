"""Phase 5 for the 500M-class build. Identical logic to train.py (same
WindowStore, same LR schedule, same DDP/grad-accum/checkpoint machinery) --
the ONLY change is `import config_500m as config` instead of `import config`,
so every config.* reference (MODEL shape, TRAIN hyperparams, CKPT_DIR,
TRAIN_TOKENS_DIR, etc.) resolves against the 500M build's separate config and
data root instead of the live 125M pipeline's. train.py itself is untouched
and still trains the 125M model if invoked directly.

    torchrun --standalone --nproc_per_node=8 train_500m.py
Controlled by env vars: EPOCHS (default "2" in the underlying logic -- ALWAYS
pass EPOCHS=1 explicitly for this build, see config_500m.py's docstring on
why 1 epoch is correct here: abundant unique data, no need to repeat),
SMOKE_STEPS, COMPILE.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import time
from contextlib import nullcontext
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import LlamaConfig, LlamaForCausalLM

import config_500m as config

# --------------------------------------------------------------------------- #
# Staged (MiniCPM/SmolLM2-style Warmup-Stable-Decay) data schedule.
#
# Why: full-val eval on the flat-mix run showed the two bulk sources
# (sec-edgar-new 63%, caselaw-new 24% of tokens) diluted the original
# case-law/sec/fineweb-edu "identity" sources down to ~12.6% combined --
# down from 100% in the 125M model -- and fineweb-edu perplexity got WORSE
# (22.82 vs 20.73) as a direct result. Research (OLMo, SmolLM2, MiniCPM,
# FinPythia -- see project discussion) converges on: (a) a single bulk
# source dominating a mix is a known, fixable failure mode, not a neutral
# outcome, and (b) the fix most small successful models actually use is a
# STAGED schedule -- broad/balanced for most of training, then a short
# "decay" phase at the end that upweights the curated identity data so it's
# what the model sees last and retains most strongly (MiniCPM's WSD
# scheduler; SmolLM2's staged rebalancing).
#
# Implementation: rather than re-running Phases 1-4 with new token budgets
# (the static-mix approach), this reweights SAMPLING of the already-tokenized
# per-source .bin files at the training-loop level -- no new data engineering,
# same total step count/token budget/wall-clock time as the flat-mix run,
# just a different, phase-dependent draw from each source's existing pool.
# Small sources (fineweb-edu, case-law, sec) get sampled with replacement at
# a rate far above their natural size share -- exactly the "temperature
# sampling" mechanism (mT5-style) the research surfaced as the standard fix.
# --------------------------------------------------------------------------- #
SOURCES: tuple[str, ...] = ("case-law", "sec", "fineweb-edu", "sec-edgar-new", "caselaw-new")

STABLE_WEIGHTS: dict[str, float] = {
    "fineweb-edu": 0.15,
    "case-law": 0.13,
    "sec": 0.15,
    "sec-edgar-new": 0.30,
    "caselaw-new": 0.27,
}
DECAY_WEIGHTS: dict[str, float] = {
    "fineweb-edu": 0.30,
    "case-law": 0.25,
    "sec": 0.25,
    "sec-edgar-new": 0.12,
    "caselaw-new": 0.08,
}
DECAY_FRACTION = 0.15  # last 15% of steps switch to DECAY_WEIGHTS

assert abs(sum(STABLE_WEIGHTS.values()) - 1.0) < 1e-9
assert abs(sum(DECAY_WEIGHTS.values()) - 1.0) < 1e-9


def decay_threshold(max_steps: int) -> int:
    """Single source of truth for where the decay phase starts -- used both
    by weights_at()'s actual comparison and by the startup log line, so the
    logged 'decay_starts_at_step' always matches real behavior exactly
    (previously: the log used int() truncation but the comparison used the
    untruncated float, so the phase actually flipped one step later than
    what was logged -- harmless at 1-in-28,054 steps, but a real
    log/behavior mismatch worth not shipping)."""
    return int(max_steps * (1 - DECAY_FRACTION))


def weights_at(step: int, max_steps: int) -> dict[str, float]:
    return DECAY_WEIGHTS if step >= decay_threshold(max_steps) else STABLE_WEIGHTS


def split_counts(n: int, weights: dict[str, float]) -> dict[str, int]:
    """Largest-remainder rounding so per-source counts sum to exactly n
    (naive int(n*w) per source can under/overshoot n by a few due to
    rounding, which would silently shrink or pad the batch)."""
    raw = {s: n * w for s, w in weights.items()}
    floor = {s: int(v) for s, v in raw.items()}
    remainder = n - sum(floor.values())
    fracs = sorted(raw.items(), key=lambda kv: kv[1] - floor[kv[0]], reverse=True)
    for i in range(remainder):
        floor[fracs[i % len(fracs)][0]] += 1
    return floor


def _source_files(directory: str, src: str) -> list[str]:
    """Exact '{src}-<digits>.bin' match, NOT a shell glob -- glob(f'{src}-*.bin')
    for src='sec' also matches 'sec-edgar-new-*.bin' (shares the 'sec-'
    prefix), silently blending two sources together. Same bug already found
    and fixed in evaluate_500m.py's per-source breakdown."""
    pat = re.compile(rf"^{re.escape(src)}-\d+\.bin$")
    return sorted(f for f in glob.glob(f"{directory}/*.bin") if pat.match(f.split("/")[-1]))

SEQ_LEN = config.SEQ_LEN
T = config.TRAIN
B200_BF16_PEAK = 2_250e12  # per-GPU bf16 dense flops -- train.py's 125M script still
#  uses H100_BF16_PEAK (989e12), a known cosmetic-only MFU-logging artifact carried
#  over from the Modal-era config (confirmed in that build's own results: MFU read
#  ~57% instead of a lower, correctly-B200-relative number). Using the real B200
#  figure here so this build's own MFU% log line is meaningful, not inherited noise.


def is_master(rank: int) -> bool:
    return rank == 0


def log(rank: int, *a):
    if is_master(rank):
        print(*a, flush=True)


class WindowStore:
    """Random-access view over packed uint16 .bin files as (N, SEQ_LEN) windows."""

    def __init__(self, directory: str, files: list[str] | None = None):
        self.files = files if files is not None else sorted(glob.glob(f"{directory}/*.bin"))
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
    except Exception as e:
        print(f"  [warn] volume commit failed: {e}", flush=True)


@torch.no_grad()
def evaluate(model, val: WindowStore, device, micro: int, max_windows: int = 512) -> float:
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

    train_stores = {
        src: WindowStore(config.TRAIN_TOKENS_DIR, files=_source_files(config.TRAIN_TOKENS_DIR, src))
        for src in SOURCES
    }
    for src, store in train_stores.items():
        log(rank, f"  source '{src}': {store.total:,} train windows")
    total_train_windows = sum(s.total for s in train_stores.values())
    val = WindowStore(config.VAL_TOKENS_DIR)
    gbw = T.global_batch_tokens // SEQ_LEN
    per_rank = gbw // world
    grad_accum = per_rank // T.micro_batch_size
    assert per_rank == grad_accum * T.micro_batch_size, (per_rank, grad_accum)
    steps_per_epoch = total_train_windows // gbw
    max_steps = smoke if smoke > 0 else int(steps_per_epoch * epochs)
    ckpt_every = min(T.ckpt_every_steps, max_steps) if smoke == 0 else max_steps
    log_every = 5 if smoke else T.log_every_steps
    eval_every = T.eval_every_steps
    decay_start_step = decay_threshold(max_steps)

    log(rank, f"world={world} grad_accum={grad_accum} per_rank={per_rank} "
              f"gbw={gbw} steps/epoch={steps_per_epoch} max_steps={max_steps} "
              f"total_train_windows={total_train_windows} val_windows={val.total} "
              f"smoke={smoke} epochs={epochs} decay_starts_at_step={decay_start_step}")
    log(rank, f"  stable-phase weights: {STABLE_WEIGHTS}")
    log(rank, f"  decay-phase weights:  {DECAY_WEIGHTS}")

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

    start_step = 0
    if smoke == 0 and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        raw.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])
        start_step = int(ck["step"])
        log(rank, f"resumed from step {start_step}")

    flops_per_tok = 6 * n_params
    # Per-rank RNG for weighted-with-replacement sampling across the 5
    # per-source stores. Replaces the old single-permutation epoch_perm
    # scheme -- that only made sense over one flat pool; here small sources
    # are deliberately resampled far more often than their raw size would
    # give them under a plain permutation, which is the entire point of the
    # staged mix. Seeded per-rank (not per-rank-per-epoch) so each rank's
    # draws are reproducible across a resume but distinct from every other
    # rank's -- matches the original code's "seed + rank" convention.
    rng = np.random.default_rng(T.seed + rank)
    full_schedule_steps = max_steps if smoke == 0 else int(steps_per_epoch * epochs)
    logged_decay_start = False

    model.train()
    t0 = time.time()
    for step in range(start_step, max_steps):
        lr = lr_at(step, full_schedule_steps)
        for grp in optimizer.param_groups:
            grp["lr"] = lr

        weights = weights_at(step, full_schedule_steps)
        if weights is DECAY_WEIGHTS and not logged_decay_start:
            log(rank, f"  >>> entering DECAY phase at step {step} (upweighting curated core)")
            logged_decay_start = True

        counts = split_counts(per_rank, weights)
        parts = []
        for src, cnt in counts.items():
            if cnt == 0:
                continue
            store = train_stores[src]
            idxs = rng.integers(0, store.total, size=cnt)
            parts.append(store.gather(idxs))
        batch_x = torch.cat(parts, dim=0)
        batch_x = batch_x[rng.permutation(per_rank)]  # un-group by source before micro-batching

        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=device)
        for m in range(grad_accum):
            x = batch_x[m * T.micro_batch_size:(m + 1) * T.micro_batch_size].to(device, non_blocking=True)
            sync_ctx = model.no_sync() if (ddp and m < grad_accum - 1) else nullcontext()
            with sync_ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = step_model(input_ids=x, labels=x).loss
                (loss / grad_accum).backward()
            loss_accum += loss.detach() / grad_accum
        norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), T.grad_clip)
        optimizer.step()

        if step % log_every == 0 or step == max_steps - 1:
            torch.cuda.synchronize()
            dt = time.time() - t0
            t0 = time.time()
            tok_s = T.global_batch_tokens * (log_every if step else 1) / max(1e-6, dt)
            mfu = flops_per_tok * tok_s / (world * B200_BF16_PEAK)
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
