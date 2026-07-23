"""Config for the 500M-class scale-up build. Mirrors config.py's structure
exactly (same dataclasses, same phase-directory layout) but is a SEPARATE
module with its own DATA_ROOT -- never imported by anything that touches the
live 125M pipeline (config.py, local_pipeline.py, train.py, local_finetune*.py
all stay untouched and continue to serve the deployed 125M model unaffected).

New data sources (added on top of the original 3), both verified directly
against live data before being wired in here -- not from dataset-card claims:
  - TeraflopAI/SEC-EDGAR (Apache-2.0): 10-K/10-Q/8-K filings only (the
    prose-rich disclosure types, not registration/insider-trading
    boilerplate). Measured real yield: 100% keep, 41.5B est. clean tokens
    from the full 2,849,722-doc pool. NOTE: several parquet shards in this
    repo are corrupt ("Couldn't deserialize thrift") -- confirmed
    reproducible, not a network blip. The real Phase 1 ingestion for this
    source (not yet written) needs the same per-file try/except skip logic
    used in measure_new_sources.py, not the standard clean_shard() path.
  - common-pile/caselaw_access_project (public domain): Harvard Caselaw
    Access Project + CourtListener, 6,919,240 docs. Measured real yield: 99%
    keep, 24.0B est. clean tokens.

Token budgets below are deliberately capped well under each source's real
ceiling (65.6B combined) to land around the ~20B target agreed for the 500M
build, not because more isn't available.

    export SLM_DATA_ROOT_500M=/raid/llm_sec/legal-slm-125M/data_500m
    .venv/bin/python3 config_500m.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

PROJECT = "slm-500m"
HF_REPO = "DeependraVerma/slm-500m-base"

VOLUME_NAME = "slm-500m"
DATA_ROOT = os.environ.get("SLM_DATA_ROOT_500M", "/data_500m")
CLEAN_DIR = f"{DATA_ROOT}/clean"
CORPUS_DIR = f"{DATA_ROOT}/corpus"
# Reuses the ORIGINAL project's tokenizer (same 16,384-vocab byte-level BPE)
# -- deliberate choice, not an oversight: the new sources are the same
# register of English legal/financial text the original tokenizer was
# already trained on, and reusing it skips a whole extra pipeline phase.
TOKENIZER_DIR = os.environ.get(
    "SLM_500M_TOKENIZER_DIR",
    f"{os.environ.get('SLM_DATA_ROOT', '/data')}/tokenizer",
)
TOKENS_DIR = f"{DATA_ROOT}/tokens"
TRAIN_TOKENS_DIR = f"{TOKENS_DIR}/train"
VAL_TOKENS_DIR = f"{TOKENS_DIR}/val"
CKPT_DIR = f"{DATA_ROOT}/checkpoints"
BASE_CKPT_DIR = f"{CKPT_DIR}/base"
RESUME_CKPT_PATH = f"{CKPT_DIR}/ckpt.pt"
METRICS_PATH = f"{CKPT_DIR}/metrics.jsonl"


@dataclass(frozen=True)
class ModelConfig:
    """Maps 1:1 to transformers.LlamaConfig. ~528.5M params with tied embeddings.
    Same architectural style as the 125M model (full MHA, SwiGLU 4x, RoPE,
    RMSNorm, tied embeddings) just scaled wider (768->1152) and deeper
    (12->24 layers)."""

    vocab_size: int = 16_384
    hidden_size: int = 1_152
    intermediate_size: int = 4_608        # SwiGLU inner, 4x hidden (same ratio as 125M)
    num_hidden_layers: int = 24
    num_attention_heads: int = 18         # head dim 64 (same as 125M)
    num_key_value_heads: int = 18         # == heads -> MHA
    max_position_embeddings: int = 1_024
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True
    attention_bias: bool = False

    def to_llama_kwargs(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_norm_eps,
            "hidden_act": self.hidden_act,
            "tie_word_embeddings": self.tie_word_embeddings,
            "attention_bias": self.attention_bias,
        }

    def approx_params(self) -> int:
        e = self.vocab_size * self.hidden_size
        h, i = self.hidden_size, self.intermediate_size
        kv = self.num_key_value_heads * (h // self.num_attention_heads)
        attn = h * h + 2 * (h * kv) + h * h
        mlp = 3 * h * i
        per_layer = attn + mlp + 2 * h
        return e + self.num_hidden_layers * per_layer


MODEL = ModelConfig()

SPECIAL_TOKENS: Mapping[str, str] = {
    "bos_token": "<|bos|>",
    "eos_token": "<|eos|>",
    "pad_token": "<|pad|>",
    "unk_token": "<|unk|>",
}
EXTRA_CHAT_TOKENS: tuple[str, ...] = ("<|user|>", "<|assistant|>", "<|system|>")


@dataclass(frozen=True)
class Source:
    name: str
    hf_id: str
    token_budget: int
    text_field: str
    split: str = "train"
    config_name: str | None = None
    strict_ocr: bool = False
    data_files: str | tuple[str, ...] | None = None  # NEW: optional glob(s), for sources
    #  organized in subfolders (e.g. TeraflopAI/SEC-EDGAR's per-filing-type layout)


# Original 3 sources kept at their existing budgets (already-cleaned files
# get copied over from the 125M model's /data/clean rather than re-streamed
# -- same source, same cleaning config, no reason to redo that CPU work).
# Two new sources added with budgets chosen to land near the ~20B combined
# target: 12B (sec-edgar-new) + 6B (caselaw-new) + ~2.04B (existing 3,
# realized) =~ 20B.
DATA_MIX: tuple[Source, ...] = (
    Source("case-law", "HFforLegal/case-law", 1_000_000_000, "document",
           split="us", strict_ocr=True),
    Source("sec", "PleIAs/SEC", 1_300_000_000, "text", split="train"),
    Source("fineweb-edu", "HuggingFaceFW/fineweb-edu", 500_000_000, "text",
           split="train", config_name="sample-10BT"),
    Source("sec-edgar-new", "TeraflopAI/SEC-EDGAR", 12_000_000_000, "text",
           split="train", data_files=("10-K/*.parquet", "10-Q/*.parquet", "8-K/*.parquet")),
    Source("caselaw-new", "common-pile/caselaw_access_project", 6_000_000_000, "text",
           split="train"),
)

TARGET_TOKENS: int = 20_000_000_000
CHARS_PER_TOKEN: float = 4.0

EVAL_HOLDOUT: tuple[str, ...] = ("coastalcph/lex_glue", "casehold/casehold")


@dataclass(frozen=True)
class CleanConfig:
    min_line_chars: int = 40
    max_nonalnum_ratio: float = 0.30
    min_doc_chars: int = 600
    repetition_top_k: int = 10
    max_repetition_ratio: float = 0.50
    ngram_n: int = 4
    lang_sample_chars: int = 5_000
    nonword_ratio_max: float = 0.20
    ocr_min_tokens: int = 50
    dict_path: str = "/usr/share/dict/words"


CLEAN = CleanConfig()

SEQ_LEN: int = 1_024
VAL_EVERY_N_WINDOWS: int = 100
TOKENS_DTYPE: str = "uint16"


@dataclass(frozen=True)
class TrainConfig:
    seq_len: int = SEQ_LEN
    micro_batch_size: int = 32
    global_batch_tokens: int = 524_288
    lr: float = 3e-4          # lower than 125M's 6e-4 -- standard practice, bigger models want a smaller peak LR
    min_lr: float = 3e-5
    warmup_tokens: int = 1_600_000_000  # ~8% of TARGET_TOKENS, same ratio as the 125M config
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    ckpt_every_steps: int = 500
    log_every_steps: int = 20
    eval_every_steps: int = 1_000
    seed: int = 1337


TRAIN = TrainConfig()

PRETRAIN_GPU_COUNT = 8
BUDGET_CAP_USD = 40.0

STAGES: tuple[str, ...] = (
    "setup", "clean", "dedup", "tokenizer", "tokenize", "pretrain", "deploy",
)


if __name__ == "__main__":
    p = MODEL.approx_params()
    print(f"{PROJECT}")
    print(f"model: {p:,} params (~{p/1e6:.1f}M) | vocab {MODEL.vocab_size} | "
          f"{MODEL.num_hidden_layers}L/{MODEL.hidden_size}d/"
          f"{MODEL.num_attention_heads}h kv={MODEL.num_key_value_heads}")
    print(f"target tokens: {TARGET_TOKENS/1e9:.1f}B (~{TARGET_TOKENS/p:.0f} tok/param)")
    print(f"stages: {' -> '.join(STAGES)}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"TOKENIZER_DIR (reused from 125M build): {TOKENIZER_DIR}")
