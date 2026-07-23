// Single source of truth for the 500M build's model facts.
// Parallel to lib/model.ts (the 125M build) — kept as a separate file
// rather than parameterizing one config, since the two models have
// genuinely different architectures, data mixes, and eval numbers.

export const HF_URL = "https://huggingface.co/DeependraVerma/slm-500m-base";
export const HF_SFT_URL = "https://huggingface.co/DeependraVerma/legal-slm-500m-sft";
export const HF_ONNX_URL = "https://huggingface.co/DeependraVerma/legal-slm-500m-sft-onnx";

export const CHAT_PRESETS = [
  "What must a plaintiff prove in a breach of contract claim?",
  "What does an indemnification clause do?",
  "This Agreement may be terminated by either party upon 30 days written notice. What are the termination terms?",
  "What is the purpose of a Form 10-K filing?",
  "Explain 'preponderance of the evidence'.",
] as const;

export const SFT_STATS = [
  { k: "Base", v: "slm-500m-base", note: "our own pretrained base" },
  { k: "Fine-tuned on", v: "21,186 Q&A", note: "same dataset as 125M, for a fair comparison" },
  { k: "SFT val loss", v: "0.1887", note: "epoch 2 — up from 0.1840 at epoch 1" },
  { k: "Fine-tune", v: "full fine-tune", note: "2 epochs, not LoRA" },
] as const;

export const HERO_STATS = [
  { value: "528.5M", label: "parameters" },
  { value: "5.07", label: "aggregate perplexity" },
  { value: "14.7B", label: "unique tokens" },
  { value: "16,384", label: "BPE vocabulary" },
] as const;

export const NUMBERS = [
  { k: "Trainable parameters", v: "528,538,752", note: "tied embeddings" },
  { k: "Unique training tokens", v: "14.7 billion", note: "5 sources, after dedup + decontam" },
  { k: "Training schedule", v: "Staged (WSD)", note: "1 epoch — abundant unique data, no repeat" },
  { k: "Aggregate held-out perplexity", v: "5.07", note: "full 5-source val set — see honesty note below" },
  { k: "Case-law / SEC / fineweb-edu ppl", v: "7.23 / 3.96 / 17.81", note: "all better than the 125M model" },
] as const;

export const INFRA_NOTE = "Pretrained on-prem, from a random initialization · bfloat16 compute · shares the 125M model's tokenizer";

export const ARCH = [
  { k: "Architecture", v: "Llama-style decoder" },
  { k: "Layers · dim · heads", v: "24 · 1152 · 18" },
  { k: "Head dimension", v: "64 (multi-head)" },
  { k: "Context length", v: "1,024 tokens" },
  { k: "Positional", v: "RoPE (θ = 10,000)" },
  { k: "Normalization", v: "RMSNorm (1e-5)" },
  { k: "Activation", v: "SwiGLU (silu), intermediate 4,608" },
  { k: "Vocabulary", v: "16,384 byte-level BPE (shared w/ 125M)" },
  { k: "Embeddings", v: "tied input / output" },
  { k: "Precision", v: "bfloat16" },
] as const;

// Time-averaged sampling weight across the staged schedule (85% stable-phase
// weights + 15% decay-phase weights, see train_500m.py) — the honest
// "what did the model actually see, on average" number, not just the
// stable-phase or decay-phase weights in isolation.
export const MIX = [
  { name: "SEC-EDGAR filings (bulk)", pct: 27, tone: "var(--brass)", src: "TeraflopAI/SEC-EDGAR" },
  { name: "US case law (bulk)", pct: 24, tone: "var(--green)", src: "common-pile/caselaw_access_project" },
  { name: "Educational web", pct: 17, tone: "var(--slate)", src: "fineweb-edu" },
  { name: "SEC 10-K filings", pct: 17, tone: "var(--brass-soft)", src: "PleIAs/SEC" },
  { name: "US case law", pct: 15, tone: "var(--green-deep)", src: "HFforLegal/case-law" },
] as const;

// Per-source held-out perplexity at the final checkpoint — shown in place of
// a training curve (the 500M run's periodic in-training checkpoints weren't
// logged at fixed intervals like the 125M run's were, so a step-by-step
// curve isn't available; per-source ppl is the real, honest comparison).
export const PERSOURCE_PPL = [
  { source: "SEC-EDGAR (bulk)", ppl500: 3.84, ppl125: null },
  { source: "SEC 10-K", ppl500: 3.96, ppl125: 4.36 },
  { source: "US case law (bulk)", ppl500: 8.23, ppl125: null },
  { source: "US case law", ppl500: 7.23, ppl125: 8.22 },
  { source: "Educational web", ppl500: 17.81, ppl125: 20.73 },
] as const;

// Direct, paired comparison against the 125M SFT model on the identical
// 1,059 held-out validation questions (see compare_125m_500m.py).
export const COMPARISON = [
  { task: "CUAD clause extraction (F1)", v125: "0.711", v500: "0.761", better: "500m" },
  { task: "CUAD refusal accuracy", v125: "90.1%", v500: "87.7%", better: "125m" },
  { task: "LEDGAR classification (exact-match)", v125: "74.1%", v500: "77.4%", better: "500m" },
  { task: "Case-law Q&A (closed-book)", v125: "35.6%", v500: "42.2%", better: "500m" },
  { task: "SEC filings Q&A (closed-book)", v125: "43.1%", v500: "46.9%", better: "500m" },
  { task: "General web Q&A (closed-book)", v125: "18.2%", v500: "30.3%", better: "500m" },
] as const;
