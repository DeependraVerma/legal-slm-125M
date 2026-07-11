# legal-slm-125 — a 125M legal & financial language model, from scratch

Build a 125-million-parameter Llama-style language model **from a random
initialization** — data pipeline, tokenizer, pretraining, evaluation, and a live
web demo — for legal and financial English. Everything is streamed, cleaned, and
trained reproducibly on [Modal](https://modal.com); the finished model lives on
Hugging Face and is served through a Vercel front end.

- 🤗 **Model:** https://huggingface.co/jonam-ai/slm-125m-base
- 🌐 **Live demo:** https://legal-slm-125.vercel.app
- 📊 **Held-out perplexity:** **9.13** on a 20.6M-token validation set

| | |
|---|---|
| Parameters | 125,848,320 (~125.8M, tied embeddings) |
| Architecture | Llama decoder · 12L · 768d · 12 heads · 1024 ctx |
| Tokenizer | 16,384 byte-level BPE, trained on this corpus |
| Training data | 2.04B unique tokens (US case law + SEC filings + educational web) |
| Pretraining | 2 epochs (7,778 steps) on 8×H100, bfloat16 |
| Held-out perplexity | 9.13 (val loss 2.211) |

> **This is a base (pretrained) model.** It continues text; it is not
> instruction-tuned and does not answer questions. It will confidently fabricate
> case names, citations, and figures. **Never** use its output as legal,
> financial, or factual advice.

---

## Table of contents

1. [How it works](#how-it-works)
2. [Repository layout](#repository-layout)
3. [Prerequisites](#prerequisites)
4. [The data — and why the mix is "legal-first"](#the-data)
5. [Replicate it, phase by phase](#replicate-it-phase-by-phase)
6. [Model architecture](#model-architecture)
7. [Design decisions (and why)](#design-decisions-and-why)
8. [Results](#results)
9. [Cost, honestly](#cost-honestly)
10. [Gotchas we already paid for](#gotchas-we-already-paid-for)
11. [Credits & license](#credits--license)

---

## How it works

The whole build is driven by a single config file (`config.py`) and runs as a
sequence of **phases**, each a Modal function fanned out one worker per shard:

```
Phase 0  smoke + measure   →  confirm the data streams and how many tokens exist
Phase 1  clean             →  stream + deterministic cleaning        → /data/clean
Phase 2  dedup + decontam  →  MinHash-LSH + exact + 13-gram strip    → /data/corpus
Phase 3  tokenizer         →  train a 16,384 byte-level BPE          → /data/tokenizer
Phase 4  tokenize + pack   →  uint16 1024-token windows, 99/1 split  → /data/tokens
Phase 5  pretrain          →  8×H100 DDP, 2 epochs                   → /data/checkpoints
Phase 6  evaluate + push   →  full-val perplexity + upload to HF
Phase 7  serve             →  Modal inference endpoint + Vercel site
```

All durable artifacts live on one Modal Volume (`slm-125m`) mounted at `/data`,
so any phase can be re-run or resumed independently.

## Repository layout

| File / dir | Role |
|---|---|
| `config.py` | **Single source of truth** — model, data mix, token budgets, cleaning thresholds, training hyperparameters, paths |
| `cleaning.py` | Deterministic, rule-based document cleaning (pure functions) |
| `dedup.py` | Hash / shingle / n-gram helpers for dedup + decontamination |
| `modal_app.py` | Modal app: images, Volume, and one function per phase (0–4, plus pretrain & evaluate) |
| `train.py` | Standalone DDP training loop, launched under `torchrun` on 8×H100 |
| `inference.py` | Modal scale-to-zero CPU endpoint that streams generations (Phase 7 backend) |
| `web/` | Next.js 16 front end deployed to Vercel — the live playground |

## Prerequisites

1. **Modal** — `pip install modal && modal token new` (free tier includes monthly
   compute credits). The GPU phase needs H100 access on your Modal plan.
2. **Hugging Face** — a token with the **write** role
   (huggingface.co/settings/tokens); only needed to push the model in Phase 6.
3. Copy the env template and fill in your own values (never commit it):
   ```bash
   cp .env.local.example .env.local
   # edit .env.local, then, before commands that need the tokens:
   source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET
   ```
4. **Node 18+** and the **Vercel CLI** (`npm i -g vercel`) for the web demo.

Sanity-check the config locally (no Modal needed):
```bash
python3 config.py
# slm-125m
# model: 125,847,552 params (~125.8M) | vocab 16384 | 12L/768d/12h kv=12
```

## The data

Three public, ungated datasets, streamed from Hugging Face (never fully
downloaded):

| Source | HF id | split | field |
|---|---|---|---|
| US case law | `HFforLegal/case-law` | `us` | `document` |
| SEC filings | `PleIAs/SEC` | `train` | `text` |
| Educational web | `HuggingFaceFW/fineweb-edu` (`sample-10BT`) | `train` | `text` |

**The mix is not 70/20/10.** The obvious plan — 70% case law, 20% SEC, 10% web at
~10B tokens — is impossible, because the legal sources are small. Measured clean
yields (run Phase 0's `measure` to confirm): case law ~0.8B tokens, SEC ~1.2B,
fineweb effectively unlimited. So the two legal sources cap at ~2B tokens total.

The strategy is **"legal-first"**: take *all* of case law (budget 1.0B) and *all*
of SEC (budget 1.3B), then add a small web slice (0.5B) for fluency. Budgets live
in `config.DATA_MIX`. The **realized** mix after real tokenization was:

| Source | Tokens | Share |
|---|---|---|
| US case law | 715M | 35% |
| SEC filings | 860M | 42% |
| Educational web | 464M | 23% |
| **Total (train)** | **2.04B** | 100% |

You get more "tokens seen" by running multiple **epochs** over this fixed corpus,
not by collecting more unique tokens.

## Replicate it, phase by phase

Run one phase, check the output, then continue. Do not chain them silently.

### Phase 0 — smoke + measure (CPU, ~$0)
```bash
modal run modal_app.py::main       # stream 10 docs/source, clean, print
modal run modal_app.py::measure    # project true clean-token yield per source
```
Expect ~9–10 of 10 docs kept per source, and `measure` to report roughly case-law
~0.8B, SEC ~1.2B, fineweb ~11B — the evidence for legal-first.

### Phase 1 — clean (CPU, a few minutes)
```bash
modal run modal_app.py::clean --fineweb-shards 5
```
Fans out ~20 workers (one per parquet shard), applies the deterministic chain
(line filter → boilerplate strip → length/repetition/language gates → an OCR gate
for scanned case law), and writes `/data/clean/<source>/shard-XX.txt`. Expect
~718k docs streamed, ~698k kept (~97%).

### Phase 2 — dedup + decontaminate (CPU, ~6 min)
```bash
modal run modal_app.py::dedup
```
MinHash signatures per case-law shard → one LSH near-duplicate pass → a per-shard
writer that also drops exact duplicates and **13-gram matches against the CaseHOLD
/ LexGLUE benchmarks** (so evaluation stays honest). Writes `/data/corpus/…`.
Expect ~24k CaseHOLD-contaminated docs removed, ~1.6k near-dups, ~2k SEC exact
dups; ~670k docs remain.

### Phase 3 — train the tokenizer (CPU, ~4 min)
```bash
modal run modal_app.py::tokenizer
```
Trains a fresh **16,384** byte-level BPE on the whole corpus and saves it to
`/data/tokenizer/`. Expect `vocab_size=16384` and two `roundtrip=True` checks.

### Phase 4 — tokenize + pack (CPU, ~10 min)
```bash
modal run modal_app.py::tokenize
```
14 workers encode the corpus, append `<|eos|>` after each document, pack into
**1024-token uint16 windows**, and route every 100th window to validation (a clean
99/1 split). Writes `/data/tokens/{train,val}/*.bin` and `index.json`. Expect
**train ≈ 2.04B tokens (1,991,282 windows), val ≈ 20.6M tokens (20,119 windows)**.

### Phase 5 — pretrain on 8×H100 (GPU)
```bash
modal run modal_app.py::pretrain_smoke      # 30-step sanity check
modal run modal_app.py::pretrain_run        # full 2-epoch run (compile ON)
```
`train.py` runs under `torchrun` with 8-way DDP, bfloat16, SDPA/flash attention,
and `torch.compile`. Hyperparameters come straight from `config.TRAIN`:

| | |
|---|---|
| Global batch | 524,288 tokens (512 windows/step) |
| Steps | 7,778 (2 epochs × 3,889) |
| Optimizer | AdamW β=(0.9, 0.95), wd 0.1 (2D params only), grad-clip 1.0 |
| LR schedule | 6e-4 → 6e-5 cosine, 200M-token linear warmup |
| Checkpoints | every 500 steps (resumable), metrics every 20, eval every 1000 |

Checkpoints (`/data/checkpoints/ckpt.pt`) make the run **resumable** — if it dies,
`pretrain_run` picks up from the last checkpoint. Throughput was ~3.19M tok/s at
~30% MFU; the useful compute is ~15–20 minutes.

### Phase 6 — evaluate + push to Hugging Face
```bash
modal run modal_app.py::evaluate           # full-val perplexity + sample generations (1×L4)
```
Then download the finished model + tokenizer from the Volume and upload them to
your HF repo (set `HF_REPO` in `config.py` first):
```bash
modal volume get slm-125m /checkpoints/base ./hf_export
modal volume get slm-125m /tokenizer        ./hf_export
huggingface-cli upload jonam-ai/slm-125m-base ./hf_export .
```
Expect a held-out perplexity around **9.13**.

### Phase 7 — serve the live demo
```bash
modal deploy inference.py                   # scale-to-zero CPU endpoint, streams tokens (SSE)
cd web
npm install
# set NEXT_PUBLIC_INFERENCE_URL to your Modal endpoint (or edit app/lib/model.ts)
npm run dev                                 # local preview
vercel deploy --prod                        # ship to Vercel
```
`inference.py` loads the HF model once per container and serves `/generate` with
token-by-token streaming; it scales to zero when idle (≈ $0). The Next.js site
calls it and renders the completion live.

## Model architecture

Maps 1:1 to `transformers.LlamaConfig`:

| Property | Value |
|---|---|
| Layers / hidden / heads | 12 / 768 / 12 (head dim 64, MHA) |
| Intermediate (SwiGLU) | 3,072 |
| Context length | 1,024 |
| Positional encoding | RoPE (θ = 10,000) |
| Normalization | RMSNorm (ε = 1e-5) |
| Activation | SwiGLU (silu) |
| Vocab | 16,384 byte-level BPE |
| Embeddings | tied input/output |
| Precision | bfloat16 (weights saved fp32) |

## Design decisions (and why)

None of these choices are arbitrary. The guiding principle for a **125M model on a
tight budget** is: *spend every parameter and every FLOP where it buys the most
quality, and copy the modern, battle-tested defaults (the Llama recipe) instead of
re-inventing them.* Here is the reasoning behind each one.

### Why RoPE for positional encoding (θ = 10,000)
Rotary Positional Embeddings encode position by **rotating** the query and key
vectors by an angle proportional to their position, so the *relative* distance
between two tokens falls directly out of their attention dot-product. We chose it
over learned absolute position embeddings because:
- **Zero extra parameters.** Learned absolute embeddings would add
  `max_position × hidden` weights; RoPE adds none — precious for a small model.
- **Relative positions generalize better** than absolute indices, which matters for
  long, repetitive legal/financial documents.
- **Graceful length extension.** RoPE can be interpolated to longer contexts later
  (NTK / linear scaling) without retraining from scratch.
- It is the proven standard in Llama / Mistral / Qwen. `θ = 10,000` is the classic
  base frequency and is well-matched to a 1,024 context (the very large `θ` values
  are only needed for 32k+ long-context models).

### Why SwiGLU for the MLP activation
SwiGLU computes `SiLU(x·W_gate) ⊙ (x·W_up)` before projecting back down — a
**gated** MLP whose multiplicative interaction lets the network modulate its own
signal. Shazeer's *"GLU Variants Improve Transformer"* showed it gives a
consistently lower loss than plain ReLU/GELU MLPs at the same budget. It costs a
third weight matrix (hence `intermediate = 3,072`, three matrices instead of two),
but the quality-per-parameter gain is worth it and it is the Llama-family default.

### Why a 16,384-token vocabulary
- **Parameter economics.** The embedding table (tied to the LM head) is
  `vocab × hidden = 16,384 × 768 ≈ 12.6M` params — already ~10% of the whole model.
  A 32k–128k vocab would let the embeddings *dominate* a 125M budget, starving the
  actual transformer layers. A lean 16k keeps parameters in the layers where the
  reasoning happens.
- **A power of two (2¹⁴)** aligns cleanly with GPU matmul tiling.
- **Byte-level BPE means zero out-of-vocabulary** — every byte is representable, so
  the odd characters, `§` symbols, citations, and numbers in legal/financial text
  never break.
- **Trained on our own corpus**, so frequent domain terms (`plaintiff`,
  `pursuant`, `10-K`) become single tokens → fewer tokens per document → cheaper
  training and more effective context.
- Trade-off: a smaller vocab means slightly *more* tokens per document, but for a
  small model the cheaper embeddings and better parameter allocation win.

### Why a 1,024-token context length
- **Attention is O(sequence²).** 1,024 keeps compute and memory modest — decisive
  when you are GPU-cost-constrained.
- For *pretraining a base model*, 1,024-token windows are more than enough to learn
  syntax, local coherence, and domain style; longer context gives diminishing
  returns at 125M while multiplying cost.
- We **pack** documents into contiguous 1,024-token windows (separated by `<|eos|>`),
  so no compute is wasted on padding.
- It can be **extended later** via RoPE interpolation if a long-context variant is
  ever needed — the choice is not permanent.

### Why 12 layers / 768 dim / 12 heads (head dim 64)
- This is the **canonical "base" transformer shape** (GPT-2 / BERT-base). It reliably
  lands at ~125.8M params with tied embeddings and is one of the most studied,
  stable-to-train configurations that exists — so results are comparable to the
  literature and we avoid hand-tuning the aspect ratio.
- **Depth vs. width:** 12 layers × 768 dims is a proven balance. Going deeper-and-
  thinner can help reasoning but is harder to train stably at small scale; going
  wider-and-shallower under-uses depth. 12/768 is the sweet spot.
- **768 ÷ 12 = 64**, the standard head dimension that flash-attention kernels are
  optimized for — expressive enough per head, cheap enough to compute.
- We did not invent these numbers; we adopted the known-good ratio so the parameter
  budget goes into a shape that is guaranteed to train well.

### Why full Multi-Head Attention (not GQA / MQA)
We set `num_key_value_heads == num_attention_heads` (= 12), i.e. **every head keeps
its own K and V**. Grouped-Query (GQA) and Multi-Query (MQA) attention *share* K/V
across heads to shrink the KV-cache and speed up autoregressive decoding — but that
is an **inference-memory optimization that trades away a little quality**. At 125M
params with a 1,024 context the KV-cache is tiny and inference is already cheap, so
there is nothing to buy: full MHA gives maximum representational capacity per head.
GQA/MQA earn their keep at 7B+ scale and long context; here it would be premature
optimization.

### Why bfloat16 compute but fp32 saved weights
- **bfloat16** keeps fp32's full 8-bit exponent (same dynamic range) with fewer
  mantissa bits, so — unlike fp16 — it needs **no loss-scaling** and won't
  overflow/underflow during training, while halving memory bandwidth and doubling
  throughput on H100 tensor cores. It is the modern default for LLM pretraining.
- We **compute** in bf16 (autocast) but keep the master weights and optimizer state
  in **fp32**, so the tiny gradient updates accumulate with full numerical precision.
- We **save** the final weights in **fp32** (a lossless ~480MB `safetensors`) so the
  published artifact is the highest-fidelity copy. Anyone downstream can downcast to
  bf16/fp16 or quantize to int8/int4 as they wish — you can always *lose* precision
  later, but you can never recover it if you ship a lossy checkpoint.

### Two more Llama-recipe choices
- **RMSNorm** (instead of LayerNorm) drops the mean-subtraction and bias term for a
  cheaper, equally effective normalization — fewer ops, one less thing to learn.
- **Tied embeddings** share the input embedding matrix with the output LM head,
  saving ~12.6M parameters (that ~10% of the model) with no measurable quality loss
  at this scale — parameters better spent inside the layers.

## Results

Held-out perplexity over training (20.6M-token validation set):

| Step | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | final |
|---|---|---|---|---|---|---|---|---|
| Perplexity | 16.4 | 12.5 | 11.2 | 10.5 | 10.0 | 9.6 | 9.4 | **9.13** |

Sample completions are coherent, on-domain legal/financial prose (see the live
demo) — while, being a base model, inventing all specifics.

## Cost, honestly

This project is *not* free — the GPU pretraining is the real expense, and being
honest about it helps you budget:

| Resource | Cost | What |
|---|---|---|
| H100 | ~$36 | 2-epoch pretraining (plus some avoidable debugging waste) |
| CPU | ~$2 | Phases 0–4 data pipeline |
| L4 | ~$0.07 | Phase 6 evaluation |
| Deployed inference | ~$0.06 | Phase 7 endpoint (scale-to-zero) |
| **Total usage** | **~$39** | on Modal's Starter plan |

Modal's free tier (~$30/month credits) absorbs most of it; out-of-pocket for this
run was ~$9. **The single biggest lever is the pretraining GPU spend** — fewer
epochs or a single-H100 run cost proportionally less. Everything up to Phase 5 is
cents.

## Gotchas we already paid for

1. **The mix is legal-first, not 70/20/10.** The legal sources only hold ~2B tokens.
2. **Modal image rule:** all `pip_install` / `apt_install` must come *before*
   `add_local_python_source`, or the image build errors.
3. **`is_english` is ASCII-first**, calling `langdetect` only on the ambiguous
   90–99% ASCII band — keep that ordering, `langdetect` is slow per document.
4. **The OCR gate needs the system wordlist** (`/usr/share/dict/words`), provided
   by the `wamerican` apt package in the base image.
5. **`torch.compile` + DDP is fragile.** A manual `all_reduce` for loss logging
   deadlocked NCCL against the compiled graph — log rank-0's local loss instead.
   And evaluate on the **raw** (un-DDP, uncompiled) module without toggling
   `.eval()/.train()`, or you force a recompile mid-run and desync the ranks.
6. **`modal deploy` won't swap a warm container's code.** Stop the app
   (`modal app stop --yes`) then redeploy; verify with a version marker.
7. **Avoid `from __future__ import annotations` in the FastAPI file** — it turns
   route type hints into strings FastAPI can't resolve for locally-imported classes.
8. **Keep heavy steps fanned out one-worker-per-shard.** Modal can preempt a long
   single container; sharded work is preemption-safe.

## Credits & license

Built from scratch as a hands-on study in end-to-end language-model engineering:
data → tokenizer → pretraining → evaluation → deployment. Inspired by the Vizuara
AI Lab "SLM from scratch" session.

Code is released under the [MIT License](LICENSE). The model weights on Hugging
Face carry their own card and disclaimers. This is a research artifact, **not**
a source of legal or financial advice.
