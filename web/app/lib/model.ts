// Single source of truth for the site's model facts.

// TODO: run `modal deploy` under your own Modal account and set
// NEXT_PUBLIC_INFERENCE_URL / NEXT_PUBLIC_CHAT_URL (or replace the
// fallbacks below) with your own *.modal.run URLs.
export const INFERENCE_URL =
  process.env.NEXT_PUBLIC_INFERENCE_URL ??
  "https://REPLACE_WITH_YOUR_MODAL_ACCOUNT--legal-slm-125-inference-slm-web.modal.run";

export const CHAT_URL =
  process.env.NEXT_PUBLIC_CHAT_URL ??
  "https://REPLACE_WITH_YOUR_MODAL_ACCOUNT--legal-slm-125m-chat-inference-chat-web.modal.run";

export const HF_URL = "https://huggingface.co/DeependraVerma/slm-125m-base";
export const HF_SFT_URL = "https://huggingface.co/DeependraVerma/legal-slm-125m-sft";

export const CHAT_PRESETS = [
  "What must a plaintiff prove in a breach of contract claim?",
  "What is the purpose of a Form 10-K filing?",
  "What does an indemnification clause do?",
  "Explain 'preponderance of the evidence'.",
  "What are the fiduciary duties of a corporate director?",
] as const;

export const SFT_STATS = [
  { k: "Base", v: "slm-125m-base", note: "our own pretrained base" },
  { k: "Fine-tuned on", v: "5,846 Q&A", note: "Gemini-distilled + judged" },
  { k: "SFT val loss", v: "2.06", note: "from 4.27" },
  { k: "Fine-tune", v: "1×L4 · ~80s", note: "full fine-tune" },
] as const;

export const HERO_STATS = [
  { value: "125.8M", label: "parameters" },
  { value: "7.76", label: "held-out perplexity" },
  { value: "2.04B", label: "unique tokens" },
  { value: "16,384", label: "BPE vocabulary" },
] as const;

export const NUMBERS = [
  { k: "Trainable parameters", v: "125,848,320", note: "tied embeddings" },
  { k: "Unique training tokens", v: "2.04 billion", note: "after dedup + decontam" },
  { k: "Tokens seen", v: "4.08 billion", note: "2 epochs" },
  { k: "Held-out perplexity", v: "7.76", note: "full 20.6M-token val set" },
  { k: "Final validation loss", v: "2.049", note: "cross-entropy" },
  { k: "Compute", v: "8 × B200 (on-prem)", note: "bfloat16" },
] as const;

export const ARCH = [
  { k: "Architecture", v: "Llama-style decoder" },
  { k: "Layers · dim · heads", v: "12 · 768 · 12" },
  { k: "Head dimension", v: "64 (multi-head)" },
  { k: "Context length", v: "1,024 tokens" },
  { k: "Positional", v: "RoPE (θ = 10,000)" },
  { k: "Normalization", v: "RMSNorm (1e-5)" },
  { k: "Activation", v: "SwiGLU (silu)" },
  { k: "Vocabulary", v: "16,384 byte-level BPE" },
  { k: "Embeddings", v: "tied input / output" },
  { k: "Precision", v: "bfloat16" },
] as const;

export const MIX = [
  { name: "US case law", pct: 35, tone: "var(--green)", src: "HFforLegal/case-law" },
  { name: "SEC filings", pct: 42, tone: "var(--brass)", src: "PleIAs/SEC" },
  { name: "Educational web", pct: 23, tone: "var(--slate)", src: "fineweb-edu" },
] as const;

// Real held-out perplexity at each eval checkpoint during pretraining
// (data/checkpoints/metrics.jsonl). The final point is the full 20,123-window
// val-set eval (evaluate_local.py); the rest are the periodic 512-window
// in-training checks, which is why the curve doesn't land exactly on 7.76.
export const CURVE: { step: number; ppl: number }[] = [
  { step: 1000, ppl: 16.36 },
  { step: 5000, ppl: 10.12 },
  { step: 10000, ppl: 10.02 },
  { step: 15000, ppl: 9.29 },
  { step: 20000, ppl: 8.91 },
  { step: 25000, ppl: 8.62 },
  { step: 30000, ppl: 8.41 },
  { step: 35000, ppl: 8.28 },
  { step: 38889, ppl: 7.76 },
];

export const PRESETS = [
  "The plaintiff shall bear the burden of",
  "Pursuant to the terms of this Agreement, the parties",
  "The Company's net revenues for the fiscal year",
  "IN THE UNITED STATES DISTRICT COURT FOR THE",
  "Notwithstanding any provision herein to the contrary,",
  "The defendant moved for summary judgment on the grounds that",
] as const;
