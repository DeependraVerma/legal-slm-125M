"""Local-only inference server — serves the base model (and the SFT chat
model, once Phase 8 exists) on this machine's own GPU, bound to localhost
only. No Modal, no public exposure.

    .venv/bin/python3 serve_local.py
    # then point web/.env.local at:
    #   NEXT_PUBLIC_INFERENCE_URL=http://localhost:8008
    #   NEXT_PUBLIC_CHAT_URL=http://localhost:8008

Endpoints mirror inference.py (/generate) and inference_chat.py (/chat) so
the existing Playground/Chat frontend code works unmodified against this.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

BASE_DIR = Path(__file__).parent / "hf_export"
BASE_MODEL_ID = os.environ.get("SLM_BASE_MODEL", str(BASE_DIR) if BASE_DIR.exists() else "DeependraVerma/slm-125m-base")
SFT_DIR = Path(os.environ.get("SLM_DATA_ROOT", str(Path(__file__).parent / "data"))) / "sft" / "model"
SFT_MODEL_ID = os.environ.get("SLM_SFT_MODEL", str(SFT_DIR) if SFT_DIR.exists() else "DeependraVerma/legal-slm-125m-sft")
SYSTEM_PROMPT = "You are a knowledgeable legal and financial assistant. Answer accurately and concisely."

DEVICE = os.environ.get("SLM_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if "cuda" in DEVICE else torch.float32

app = FastAPI(title="legal-slm-125 (local)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Served:
    def __init__(self, model_id: str):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=DTYPE).to(DEVICE)
        self.model.eval()
        self.eos_id = self.tok.convert_tokens_to_ids("<|eos|>")


print(f"[serve_local] loading base model from {BASE_MODEL_ID!r} onto {DEVICE} ({DTYPE})...")
base = Served(BASE_MODEL_ID)
print("[serve_local] base model ready.")

chat_model: Served | None = None
try:
    print(f"[serve_local] loading SFT chat model from {SFT_MODEL_ID!r}...")
    chat_model = Served(SFT_MODEL_ID)
    sid = chat_model.tok.convert_tokens_to_ids
    CHAT_BOS, CHAT_SYS, CHAT_USER, CHAT_ASST = sid("<|bos|>"), sid("<|system|>"), sid("<|user|>"), sid("<|assistant|>")
    CHAT_SYS_IDS = chat_model.tok(SYSTEM_PROMPT, add_special_tokens=False)["input_ids"]
    print("[serve_local] SFT chat model ready.")
except Exception as e:  # noqa: BLE001 - SFT model may not exist yet (Phase 8 not run)
    print(f"[serve_local] SFT chat model unavailable ({e}); /chat will return 503 until Phase 8 is done.")


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "base_model": BASE_MODEL_ID,
        "chat_model": SFT_MODEL_ID if chat_model else None,
    }


@app.post("/generate")
async def generate(req: Request):
    body = await req.json()
    prompt = (body.get("prompt") or "").strip() or "The plaintiff"
    max_new = max(8, min(256, int(body.get("max_new_tokens", 96))))
    temperature = max(0.1, min(1.5, float(body.get("temperature", 0.8))))

    ids = base.tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    streamer = TextIteratorStreamer(base.tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(
        input_ids=ids, max_new_tokens=max_new, do_sample=True,
        temperature=temperature, top_k=50, top_p=0.95,
        eos_token_id=base.eos_id, pad_token_id=base.eos_id, streamer=streamer,
    )

    def event_stream():
        thread = threading.Thread(target=base.model.generate, kwargs=kwargs)
        thread.start()
        n = 0
        for text in streamer:
            if text:
                n += 1
                yield f"data: {json.dumps({'token': text})}\n\n"
        thread.join()
        yield f"data: {json.dumps({'done': True, 'count': n})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/chat")
async def chat(req: Request):
    if chat_model is None:
        return {"error": "SFT chat model not available locally yet (Phase 8 hasn't been run/pushed)."}

    body = await req.json()
    message = (body.get("message") or "").strip() or "What is a Form 10-K?"
    max_new = max(16, min(256, int(body.get("max_new_tokens", 160))))
    temperature = max(0.1, min(1.5, float(body.get("temperature", 0.7))))

    q_ids = chat_model.tok(message, add_special_tokens=False)["input_ids"]
    prompt_ids = [CHAT_BOS, CHAT_SYS] + CHAT_SYS_IDS + [CHAT_USER] + q_ids + [CHAT_ASST]
    ids = torch.tensor([prompt_ids]).to(DEVICE)
    streamer = TextIteratorStreamer(chat_model.tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(
        input_ids=ids, max_new_tokens=max_new, do_sample=True,
        temperature=temperature, top_k=50, top_p=0.9,
        eos_token_id=chat_model.eos_id, pad_token_id=chat_model.eos_id, streamer=streamer,
    )

    def event_stream():
        thread = threading.Thread(target=chat_model.model.generate, kwargs=kwargs)
        thread.start()
        for text in streamer:
            if text:
                yield f"data: {json.dumps({'token': text})}\n\n"
        thread.join()
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    # Port 8000 is taken by another service on this box (vLLM) — don't use it.
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("SLM_PORT", 8008)))
