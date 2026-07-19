"""Export the (locally trained) base model to ONNX + dynamic-int8-quantized
ONNX, laid out the way transformers.js (the in-browser runtime used by
web/app/lib/browserModel.ts) expects: `onnx/model.onnx` (fp32) and
`onnx/model_quantized.onnx` (q8) alongside the usual config/tokenizer files
at the repo root.

    .venv/bin/python3 export_onnx.py                 # exports hf_export/ -> onnx_export/
    huggingface-cli upload DeependraVerma/slm-125m-base-onnx ./onnx_export .

After uploading, point web/app/lib/model.ts's ONNX_BASE_REPO (or whichever
constant browserModelBase.ts reads) at that repo.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from optimum.exporters.onnx import main_export
from onnxruntime.quantization import QuantType, quantize_dynamic

SRC_DIR = Path(__file__).parent / "hf_export"
OUT_DIR = Path(__file__).parent / "onnx_export"
ONNX_SUBDIR = OUT_DIR / "onnx"


def main():
    if not SRC_DIR.exists():
        raise SystemExit(f"{SRC_DIR} not found — run Phase 6 (evaluate + push) first.")

    OUT_DIR.mkdir(exist_ok=True)

    print(f"[export_onnx] exporting {SRC_DIR} -> {OUT_DIR} (fp32 ONNX)...")
    main_export(
        model_name_or_path=str(SRC_DIR),
        output=str(OUT_DIR),
        task="text-generation-with-past",
        opset=17,
    )

    # optimum puts model.onnx (+ .onnx_data if >2GB, not the case here) at the
    # export root; transformers.js expects it under onnx/.
    ONNX_SUBDIR.mkdir(exist_ok=True)
    for f in OUT_DIR.glob("*.onnx"):
        shutil.move(str(f), ONNX_SUBDIR / f.name)

    fp32_path = ONNX_SUBDIR / "model.onnx"
    quant_path = ONNX_SUBDIR / "model_quantized.onnx"
    print(f"[export_onnx] dynamic int8 quantizing -> {quant_path.name}...")
    quantize_dynamic(str(fp32_path), str(quant_path), weight_type=QuantType.QInt8)

    print(f"[export_onnx] done. onnx/ contains: {sorted(p.name for p in ONNX_SUBDIR.iterdir())}")
    print(f"[export_onnx] root contains: {sorted(p.name for p in OUT_DIR.iterdir() if p.is_file())}")
    print("[export_onnx] next: huggingface-cli upload DeependraVerma/slm-125m-base-onnx ./onnx_export .")


if __name__ == "__main__":
    main()
