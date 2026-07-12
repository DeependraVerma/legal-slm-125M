// Browser-only in-browser inference via transformers.js (@huggingface/transformers).
// The model runs entirely on the visitor's device (no server).
// transformers.js is imported lazily so it never touches SSR or the initial bundle.

const ONNX_REPO = "jonam-ai/legal-slm-125m-sft-onnx";
const SYSTEM_PROMPT =
  "You are a knowledgeable legal and financial assistant. Answer accurately and concisely.";

/* eslint-disable @typescript-eslint/no-explicit-any */
let _mod: any = null;
let _tok: any = null;
let _model: any = null;
let _loading: Promise<void> | null = null;

export function isModelLoaded(): boolean {
  return !!(_model && _tok);
}

export async function ensureLoaded(onProgress?: (pct: number) => void): Promise<void> {
  if (isModelLoaded()) {
    onProgress?.(100);
    return;
  }
  if (!_loading) {
    _loading = (async () => {
      const t = await import("@huggingface/transformers");
      _mod = t;
      t.env.allowLocalModels = false;
      _tok = await t.AutoTokenizer.from_pretrained(ONNX_REPO);
      _model = await t.AutoModelForCausalLM.from_pretrained(ONNX_REPO, {
        dtype: "q8",
        device: "wasm",
        progress_callback: (p: any) => {
          if (p?.status === "progress" && p?.total) {
            onProgress?.(Math.min(99, Math.round((p.loaded / p.total) * 100)));
          }
        },
      });
      onProgress?.(100);
    })();
  }
  return _loading;
}

export async function generateChat(
  message: string,
  onToken: (token: string) => void,
): Promise<void> {
  await ensureLoaded();
  const t = _mod;
  const prompt = `<|bos|><|system|>${SYSTEM_PROMPT}<|user|>${message.trim()}<|assistant|>`;
  const inputs = _tok(prompt, { add_special_tokens: false });
  const eos = _tok.model.tokens_to_ids.get("<|eos|>");
  const streamer = new t.TextStreamer(_tok, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (tok: string) => onToken(tok),
  });
  await _model.generate({
    ...inputs,
    max_new_tokens: 128, // browser WASM is single-threaded; keep replies snappy
    do_sample: true,
    temperature: 0.7,
    top_k: 50,
    top_p: 0.9,
    eos_token_id: eos,
    streamer,
  });
}
