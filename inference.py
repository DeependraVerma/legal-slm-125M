"""Live inference endpoint for legal-slm-125.

Serves jonam-ai/slm-125m-base as a scale-to-zero CPU service with token-by-token
SSE streaming, so the Vercel frontend can show the base model completing text live.

    modal deploy inference.py
"""

# NOTE: intentionally NO `from __future__ import annotations` here — it would turn
# the FastAPI route hints (e.g. `req: Request`) into strings that FastAPI cannot
# resolve, since those classes are imported locally inside the container.

import modal

app = modal.App("legal-slm-125-inference")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "fastapi[standard]==0.115.4",
    )
)

MODEL_ID = "jonam-ai/slm-125m-base"
EOS = "<|eos|>"

# Persist the HF download so cold starts don't re-fetch the weights.
hf_cache = modal.Volume.from_name("legal-slm-hf-cache", create_if_missing=True)


@app.cls(
    image=image,
    cpu=4.0,
    memory=8192,
    volumes={"/cache": hf_cache},
    scaledown_window=300,   # stay warm 5 min between requests, then scale to zero
    min_containers=0,
)
class SLM:
    @modal.enter()
    def load(self):
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ["HF_HOME"] = "/cache"
        torch.set_num_threads(4)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir="/cache")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, cache_dir="/cache", torch_dtype=torch.float32)
        self.model.eval()
        self.eos_id = self.tok.convert_tokens_to_ids(EOS)
        hf_cache.commit()

    @modal.asgi_app()
    def web(self):
        import json
        import threading

        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
        from transformers import TextIteratorStreamer

        # Ensure FastAPI's get_type_hints() can always resolve the route hints,
        # regardless of local-scope imports or PEP 563 string annotations.
        globals()["Request"] = Request

        api = FastAPI(title="legal-slm-125")
        api.add_middleware(
            CORSMiddleware, allow_origins=["*"],
            allow_methods=["*"], allow_headers=["*"],
        )

        @api.get("/health")
        def health():
            return {"ok": True, "model": MODEL_ID, "v": 2}

        @api.post("/generate")
        async def generate(req: Request):
            body = await req.json()
            prompt = (body.get("prompt") or "").strip()
            if not prompt:
                prompt = "The plaintiff"
            max_new = max(8, min(256, int(body.get("max_new_tokens", 96))))
            temperature = max(0.1, min(1.5, float(body.get("temperature", 0.8))))

            ids = self.tok(prompt, return_tensors="pt").input_ids
            streamer = TextIteratorStreamer(
                self.tok, skip_prompt=True, skip_special_tokens=True)
            kwargs = dict(
                input_ids=ids, max_new_tokens=max_new, do_sample=True,
                temperature=temperature, top_k=50, top_p=0.95,
                eos_token_id=self.eos_id, pad_token_id=self.eos_id,
                streamer=streamer,
            )

            def event_stream():
                thread = threading.Thread(target=self.model.generate, kwargs=kwargs)
                thread.start()
                n = 0
                for text in streamer:
                    if text:
                        n += 1
                        yield f"data: {json.dumps({'token': text})}\n\n"
                thread.join()
                yield f"data: {json.dumps({'done': True, 'count': n})}\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        return api
