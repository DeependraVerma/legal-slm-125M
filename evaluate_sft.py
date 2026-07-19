"""Independent verification of the Phase 8 SFT model (data/sft/model/) --
same philosophy as evaluate_local.py's Phase 6 check: don't just trust the
training run's own numbers, re-measure and look at real outputs.

    .venv/bin/python3 evaluate_sft.py

Does two things:
1. Full val-set loss/perplexity, recomputed independently from the saved
   checkpoint (not reusing train_sft.py's in-training number).
2. Generates on a mix of (a) real held-out val questions (never trained on,
   checks generalization) and (b) fresh out-of-distribution questions with
   no matching training passage (checks it isn't just pattern-matching
   memorized phrasing) and (c) a deliberately unanswerable/adversarial
   question (checks it doesn't confidently hallucinate when it shouldn't
   know something).
"""

from __future__ import annotations

import json
import math
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

MODEL_DIR = f"{config.DATA_ROOT}/sft/model"
VAL_PATH = f"{config.DATA_ROOT}/sft/dataset/val.jsonl"
SYSTEM_PROMPT = "You are a knowledgeable legal and financial assistant. Answer accurately and concisely."

# Held-out-style but NOT drawn from the training corpus at all -- checks
# generalization beyond the specific SEC/case-law passages it was trained on.
OOD_QUESTIONS = [
    "What is the difference between a misdemeanor and a felony?",
    "What does it mean for a contract to be void versus voidable?",
    "Explain what a fiduciary duty is in simple terms.",
    "What is the purpose of an escrow account in a real estate transaction?",
    "What is the difference between common stock and preferred stock?",
]

# Deliberately unanswerable / adversarial -- checks it doesn't confidently
# invent a specific-sounding but fake answer.
ADVERSARIAL_QUESTIONS = [
    "What was the exact net revenue of Acme Fictional Corp in 2024?",
    "Cite the specific case that established the rule you just described.",
]


def full_val_loss(model, device) -> tuple[float, int]:
    rows = []
    with open(VAL_PATH, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for r in rows:
            ids = torch.tensor([r["input_ids"]], device=device)
            labels = torch.tensor([r["labels"]], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids, labels=labels)
            n_sup = sum(1 for x in r["labels"] if x != -100)
            total_loss += out.loss.item() * n_sup
            total_tok += n_sup
    return total_loss / total_tok, total_tok


def ask(model, tok, device, q: str, max_new: int = 150) -> str:
    ids = (tok("<|bos|>", add_special_tokens=False)["input_ids"]
           + [tok.convert_tokens_to_ids("<|system|>")] + tok(SYSTEM_PROMPT, add_special_tokens=False)["input_ids"]
           + [tok.convert_tokens_to_ids("<|user|>")] + tok(q, add_special_tokens=False)["input_ids"]
           + [tok.convert_tokens_to_ids("<|assistant|>")])
    inp = torch.tensor([ids], device=device)
    eos_id = tok.convert_tokens_to_ids("<|eos|>")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(inp, max_new_tokens=max_new, do_sample=True, temperature=0.7,
                              top_p=0.9, top_k=50, eos_token_id=eos_id, pad_token_id=eos_id)
    return tok.decode(out[0][len(ids):], skip_special_tokens=True)


def main() -> None:
    device = "cuda:0"
    print(f"loading SFT model from {MODEL_DIR}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16).to(device).eval()

    vloss, n_tok = full_val_loss(model, device)
    print(f"\nFULL VAL loss: {vloss:.4f}  ppl: {math.exp(vloss):.3f}  ({n_tok:,} supervised tokens)\n")

    # chat.jsonl is written from the same shuffled `kept` list, in the same
    # order, immediately after the train/val split -- so its first n_val
    # rows are exactly the val set (same order), not train examples.
    n_val = sum(1 for _ in open(VAL_PATH, encoding="utf-8"))
    chat_rows = [json.loads(l) for l in open(f"{config.DATA_ROOT}/sft/dataset/chat.jsonl", encoding="utf-8")]
    val_chat_rows = chat_rows[:n_val]
    rng = random.Random(7)
    val_questions = rng.sample(
        [r["messages"][1]["content"] for r in val_chat_rows if len(tok(r["messages"][1]["content"])["input_ids"]) < 40],
        4,
    )

    print("=" * 78)
    print("REAL HELD-OUT VAL QUESTIONS (never trained on, checks generalization)")
    print("=" * 78)
    for q in val_questions:
        print(f"\nQ: {q}")
        print(f"A: {ask(model, tok, device, q)}")

    print("\n" + "=" * 78)
    print("OUT-OF-DISTRIBUTION (general legal/financial knowledge, not from training corpus)")
    print("=" * 78)
    for q in OOD_QUESTIONS:
        print(f"\nQ: {q}")
        print(f"A: {ask(model, tok, device, q)}")

    print("\n" + "=" * 78)
    print("ADVERSARIAL (should NOT confidently hallucinate specifics)")
    print("=" * 78)
    for q in ADVERSARIAL_QUESTIONS:
        print(f"\nQ: {q}")
        print(f"A: {ask(model, tok, device, q)}")


if __name__ == "__main__":
    main()
