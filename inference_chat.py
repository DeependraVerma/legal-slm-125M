"""Live chat endpoint for the fine-tuned model (jonam-ai/legal-slm-125m-sft).

Scale-to-zero CPU service that renders the chat format and streams the assistant's
answer token-by-token (SSE), so the Vercel front end can show a live chatbot.

    modal deploy inference_chat.py
"""

# NOTE: no `from __future__ import annotations` — it breaks FastAPI route hint
# resolution for classes imported locally inside the container.

import modal

app = modal.App("legal-slm-125m-chat-inference")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "fastapi[standard]==0.115.4",
    )
)

MODEL_ID = "jonam-ai/legal-slm-125m-sft"
SYSTEM_PROMPT = "You are a knowledgeable legal and financial assistant. Answer accurately and concisely."

hf_cache = modal.Volume.from_name("legal-slm-hf-cache", create_if_missing=True)


@app.cls(
    image=image,
    cpu=4.0,
    memory=8192,
    volumes={"/cache": hf_cache},
    scaledown_window=300,
    min_containers=0,
)
class Chat:
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
        sid = self.tok.convert_tokens_to_ids
        self.bos, self.eos = sid("<|bos|>"), sid("<|eos|>")
        self.sys_t, self.user_t, self.asst_t = sid("<|system|>"), sid("<|user|>"), sid("<|assistant|>")
        self.sys_ids = self.tok(SYSTEM_PROMPT, add_special_tokens=False)["input_ids"]
        hf_cache.commit()

    def _prompt_ids(self, message: str):
        q = self.tok(message.strip(), add_special_tokens=False)["input_ids"]
        return ([self.bos, self.sys_t] + self.sys_ids
                + [self.user_t] + q + [self.asst_t])

    @modal.asgi_app()
    def web(self):
        import json
        import threading

        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
        from transformers import TextIteratorStreamer

        globals()["Request"] = Request  # ensure FastAPI can resolve the hint

        api = FastAPI(title="legal-slm-125m-sft chat")
        api.add_middleware(
            CORSMiddleware, allow_origins=["*"],
            allow_methods=["*"], allow_headers=["*"],
        )

        @api.get("/health")
        def health():
            return {"ok": True, "model": MODEL_ID}

        @api.post("/chat")
        async def chat(req: Request):
            body = await req.json()
            message = (body.get("message") or "").strip()
            if not message:
                message = "What is a Form 10-K?"
            max_new = max(16, min(256, int(body.get("max_new_tokens", 160))))
            temperature = max(0.1, min(1.5, float(body.get("temperature", 0.7))))

            ids = self.torch.tensor([self._prompt_ids(message)])
            streamer = TextIteratorStreamer(
                self.tok, skip_prompt=True, skip_special_tokens=True)
            kwargs = dict(
                input_ids=ids, max_new_tokens=max_new, do_sample=True,
                temperature=temperature, top_k=50, top_p=0.9,
                eos_token_id=self.eos, pad_token_id=self.eos, streamer=streamer,
            )

            def event_stream():
                thread = threading.Thread(target=self.model.generate, kwargs=kwargs)
                thread.start()
                for text in streamer:
                    if text:
                        yield f"data: {json.dumps({'token': text})}\n\n"
                thread.join()
                yield f"data: {json.dumps({'done': True})}\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        return api
