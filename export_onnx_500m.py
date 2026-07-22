"""Export the 500M-build's SFT model to ONNX + dynamic-int8-quantized ONNX,
same layout export_onnx.py produces for the 125M build (onnx/model.onnx +
onnx/model_quantized.onnx alongside config/tokenizer at the repo root, what
transformers.js expects).

Only the SFT model is exported here (that's what the in-browser chat needs) --
base-model ONNX export isn't part of this project's live demo.

One real difference from the 125M export worth calling out: this model's
fp32 weights are ~2.1GB, close enough to ONNX's 2GB single-file protobuf
limit that `optimum`'s exporter may emit a separate `model.onnx_data`
external-data file alongside `model.onnx`. export_onnx.py's file-moving step
only globs `*.onnx` -- it would silently leave a `.onnx_data` file behind at
the export root instead of moving it into onnx/ with its model file, which
would break loading (transformers.js loads model.onnx expecting its
external-data sibling in the same directory). This script globs both.

    .venv/bin/python3 export_onnx_500m.py
    huggingface-cli upload DeependraVerma/legal-slm-500m-sft-onnx ./onnx_export_500m_sft .
"""

from __future__ import annotations

import shutil
from pathlib import Path

from optimum.exporters.onnx import main_export
from onnxruntime.quantization import QuantType, quantize_dynamic

import config_500m as config

SRC_DIR = Path(f"{config.DATA_ROOT}/sft/model")
OUT_DIR = Path(__file__).parent / "onnx_export_500m_sft"
ONNX_SUBDIR = OUT_DIR / "onnx"


def main():
    if not SRC_DIR.exists():
        raise SystemExit(f"{SRC_DIR} not found — run local_train_sft_500m.py first.")

    OUT_DIR.mkdir(exist_ok=True)

    print(f"[export_onnx_500m] exporting {SRC_DIR} -> {OUT_DIR} (fp32 ONNX)...")
    main_export(
        model_name_or_path=str(SRC_DIR),
        output=str(OUT_DIR),
        task="text-generation-with-past",
        opset=17,
    )

    ONNX_SUBDIR.mkdir(exist_ok=True)
    # globs both *.onnx AND *.onnx_data -- see module docstring for why the
    # external-data file matters here specifically (2.1GB model, near the
    # 2GB single-protobuf-file limit)
    moved = []
    for pattern in ("*.onnx", "*.onnx_data"):
        for f in OUT_DIR.glob(pattern):
            shutil.move(str(f), ONNX_SUBDIR / f.name)
            moved.append(f.name)
    print(f"[export_onnx_500m] moved into onnx/: {moved}")

    fp32_path = ONNX_SUBDIR / "model.onnx"
    quant_path = ONNX_SUBDIR / "model_quantized.onnx"
    print(f"[export_onnx_500m] dynamic int8 quantizing -> {quant_path.name}...")
    quantize_dynamic(str(fp32_path), str(quant_path), weight_type=QuantType.QInt8)

    print(f"[export_onnx_500m] done. onnx/ contains: {sorted(p.name for p in ONNX_SUBDIR.iterdir())}")
    print(f"[export_onnx_500m] root contains: {sorted(p.name for p in OUT_DIR.iterdir() if p.is_file())}")
    print("[export_onnx_500m] next: huggingface-cli upload DeependraVerma/legal-slm-500m-sft-onnx "
          f"./{OUT_DIR.name} .")


if __name__ == "__main__":
    main()
