#!/usr/bin/env python3
"""Long-lived nnU-Net inference worker for the segmentation GUI.

The GUI must not import torch or reload the 504 checkpoint for every section.
This process loads the model once and then answers one JSON request per line:

    {"id": 7, "cmd": "predict", "image": "/path/section.JPG",
     "out": "/path/segmentations/UNet_output/group/section.png"}

Replies go to stdout, one JSON object per line:

    {"status": "ready", "device": "mps"}
    {"status": "ok", "id": 7, "mask": "...", "ms": 3600, "shape": [411, 576]}
    {"status": "error", "id": 7, "error": "..."}

Everything nnU-Net prints is redirected to stderr so stdout stays parseable.
Run it with the nnU-Net environment, not the GUI environment:

    .venv-nnunet/bin/python nnunet_tools/predict_worker.py \
        --model-dir bbdb_model_504
"""
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path


_PROTOCOL_OUT = sys.stdout


def reply(payload: dict) -> None:
    _PROTOCOL_OUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _PROTOCOL_OUT.flush()


def pick_device(requested: str, torch_module):
    if requested != "auto":
        return torch_module.device(requested)
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def run_one(predictor, image: Path, out_path: Path) -> tuple[int, tuple[int, int]]:
    """Predict one image and copy the mask to ``out_path``; returns ms and shape."""
    from PIL import Image

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="bbdb-unet-") as tmp:
        tmp_dir = Path(tmp)
        with Image.open(image) as img:
            gray = img.convert("L")
            native_hw = (gray.height, gray.width)
            # nnU-Net discovers cases by the ``_0000`` channel suffix.
            input_png = tmp_dir / "case_0000.png"
            gray.save(input_png)
        output_dir = tmp_dir / "out"
        output_dir.mkdir()
        with contextlib.redirect_stdout(sys.stderr):
            predictor.predict_from_files(
                [[str(input_png)]],
                str(output_dir),
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
            )
        produced = output_dir / "case.png"
        if not produced.is_file():
            candidates = [p for p in output_dir.glob("*.png") if not p.name.startswith(".")]
            if not candidates:
                raise RuntimeError(f"nnU-Net produced no mask in {output_dir}")
            produced = candidates[0]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(produced, out_path)
    return int((time.monotonic() - started) * 1000), native_hw


def main() -> int:
    # Keep stdout for the JSON protocol only; nnU-Net and torch like to print.
    sys.stdout = sys.stderr
    parser = argparse.ArgumentParser(description="Persistent nnU-Net worker for bbdb_seg")
    parser.add_argument("--model-dir", required=True,
                        help="folder containing fold_0/checkpoint_final.pth")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    # Accept either the trainer folder (contains fold_0/) or the package root
    # that HOWTO.txt describes (bbdb_model_504/nnUNet_results/...).
    if not (model_dir / "fold_0" / args.checkpoint).is_file():
        matches = sorted(model_dir.glob(
            f"nnUNet_results/*/*/fold_0/{args.checkpoint}"))
        if len(matches) == 1:
            model_dir = matches[0].parent.parent
    checkpoint = model_dir / "fold_0" / args.checkpoint
    if not checkpoint.is_file():
        reply({"status": "error", "error": f"checkpoint not found under: {model_dir}"})
        return 2

    try:
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        device = pick_device(args.device, torch)
        predictor = nnUNetPredictor(device=device, verbose=False, allow_tqdm=False)
        with contextlib.redirect_stdout(sys.stderr):
            predictor.initialize_from_trained_model_folder(
                str(model_dir), use_folds=(0,), checkpoint_name=args.checkpoint
            )
    except Exception as exc:  # pragma: no cover - only runs with the real model
        reply({"status": "error", "error": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc(limit=3)})
        return 2

    reply({"status": "ready", "device": str(device)})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            if request.get("cmd") == "quit":
                break
            if request.get("cmd") != "predict":
                raise ValueError(f"unknown command: {request.get('cmd')!r}")
            image = Path(str(request["image"])).expanduser()
            out_path = Path(str(request["out"])).expanduser()
            if not image.is_file():
                raise FileNotFoundError(f"image not found: {image}")
            ms, shape = run_one(predictor, image, out_path)
            reply({"status": "ok", "id": request_id, "mask": str(out_path),
                   "ms": ms, "shape": list(shape)})
        except Exception as exc:  # keep the worker alive for the next section
            reply({"status": "error", "id": request_id,
                   "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(limit=3)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
