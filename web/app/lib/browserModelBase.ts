// Browser-only in-browser inference for the base (completion) model via
// transformers.js (@huggingface/transformers). Runs entirely on the
// visitor's device (no server) — mirrors browserModel.ts's chat-model
// pattern but without a chat template, since the base model only continues
// raw text.

const ONNX_BASE_REPO = "DeependraVerma/slm-125m-base-onnx";

/* eslint-disable @typescript-eslint/no-explicit-any */
let _mod: any = null;
let _tok: any = null;
let _model: any = null;
let _loading: Promise<void> | null = null;

export function isBaseModelLoaded(): boolean {
  return !!(_model && _tok);
}

export async function ensureBaseLoaded(onProgress?: (pct: number) => void): Promise<void> {
  if (isBaseModelLoaded()) {
    onProgress?.(100);
    return;
  }
  if (!_loading) {
    _loading = (async () => {
      const t = await import("@huggingface/transformers");
      _mod = t;
      t.env.allowLocalModels = false;
      _tok = await t.AutoTokenizer.from_pretrained(ONNX_BASE_REPO);
      _model = await t.AutoModelForCausalLM.from_pretrained(ONNX_BASE_REPO, {
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

export async function generateBase(
  prompt: string,
  maxNewTokens: number,
  temperature: number,
  onToken: (token: string) => void,
): Promise<void> {
  await ensureBaseLoaded();
  const t = _mod;
  const inputs = _tok(prompt, { add_special_tokens: false });
  const eos = _tok.model.tokens_to_ids.get("<|eos|>");
  const streamer = new t.TextStreamer(_tok, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (tok: string) => onToken(tok),
  });
  await _model.generate({
    ...inputs,
    max_new_tokens: maxNewTokens, // browser WASM is single-threaded; keep this modest
    do_sample: true,
    temperature,
    top_k: 50,
    top_p: 0.95,
    eos_token_id: eos,
    streamer,
  });
}
