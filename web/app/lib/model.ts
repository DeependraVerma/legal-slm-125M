// Single source of truth for the site's model facts.

export const INFERENCE_URL =
  process.env.NEXT_PUBLIC_INFERENCE_URL ??
  "https://mcrao--legal-slm-125-inference-slm-web.modal.run";

export const HF_URL = "https://huggingface.co/jonam-ai/slm-125m-base";

export const HERO_STATS = [
  { value: "125.8M", label: "parameters" },
  { value: "9.13", label: "held-out perplexity" },
  { value: "2.04B", label: "unique tokens" },
  { value: "16,384", label: "BPE vocabulary" },
] as const;

export const NUMBERS = [
  { k: "Trainable parameters", v: "125,848,320", note: "tied embeddings" },
  { k: "Unique training tokens", v: "2.04 billion", note: "after dedup + decontam" },
  { k: "Tokens seen", v: "4.08 billion", note: "2 epochs" },
  { k: "Held-out perplexity", v: "9.13", note: "20.6M-token val set" },
  { k: "Final validation loss", v: "2.211", note: "cross-entropy" },
  { k: "Compute", v: "8 × H100", note: "bfloat16, ~30% MFU" },
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

// Real held-out perplexity at each eval checkpoint during pretraining.
export const CURVE: { step: number; ppl: number }[] = [
  { step: 1000, ppl: 16.4 },
  { step: 2000, ppl: 12.5 },
  { step: 3000, ppl: 11.2 },
  { step: 4000, ppl: 10.5 },
  { step: 5000, ppl: 10.0 },
  { step: 6000, ppl: 9.6 },
  { step: 7000, ppl: 9.4 },
  { step: 7778, ppl: 9.13 },
];

export const PRESETS = [
  "The plaintiff shall bear the burden of",
  "Pursuant to the terms of this Agreement, the parties",
  "The Company's net revenues for the fiscal year",
  "IN THE UNITED STATES DISTRICT COURT FOR THE",
  "Notwithstanding any provision herein to the contrary,",
  "The defendant moved for summary judgment on the grounds that",
] as const;
