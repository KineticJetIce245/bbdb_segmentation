"""Generate CAPS-style synthetic sections from the hand-drawn masks.

This follows the recipe in README "Stretch goal: bootstrap with synthetic data
and nnU-Net" (step 1): the model should learn the *shape* of the cortex, not one
particular stain, so each synthetic sample

1. builds a label map from the real mask (cortex) plus an Otsu tissue mask and,
   optionally, 1-6 random "pseudo layers" cut by distance from the mask edge;
2. deforms it with a random affine and/or smooth elastic warp (nearest
   interpolation, so labels stay labels);
3. paints random intensities per label, with the cortex randomly darker or
   brighter than its surroundings;
4. adds a bias field, blur, down/up-sampling and noise;
5. writes the deformed cortex mask as the 0/1 target.

Real pairs are copied next to the synthetic ones, so the training set is the
mix the README recommends.

Usage
-----
    python nnunet_tools/make_synthetic.py --per-mask 100
    python nnunet_tools/make_synthetic.py --per-mask 4 --preview preview.png
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "nnUNet" / "nnUNet_raw" / "Dataset501_BBDBCortex"
DEFAULT_OUT = REPO_ROOT / "nnUNet" / "nnUNet_raw" / "Dataset502_BBDBCortexSynth"

BACKGROUND, TISSUE, CORTEX = 0, 1, 2


def otsu_threshold(gray: np.ndarray) -> float:
    """Otsu's threshold without pulling in scikit-image."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * np.arange(256)) / total
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega))
    sigma_b = np.nan_to_num(sigma_b)
    return float(np.argmax(sigma_b))


def build_label_map(gray: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                    layers: int) -> tuple[np.ndarray, set[int]]:
    """0 background, 1 non-cortex tissue, 2..(2+layers) cortex layers."""
    tissue = gray > otsu_threshold(gray)
    tissue = ndimage.binary_closing(tissue, iterations=2)
    label = np.where(tissue, TISSUE, BACKGROUND).astype(np.uint8)
    cortex_values = {CORTEX}
    label[mask > 0] = CORTEX
    if layers > 0 and mask.any():
        depth = ndimage.distance_transform_edt(mask > 0)
        edges = np.quantile(depth[mask > 0], np.linspace(0, 1, layers + 2)[1:-1])
        for index, edge in enumerate(edges):
            value = CORTEX + 1 + index
            label[(mask > 0) & (depth > edge)] = value
            cortex_values.add(value)
    return label, cortex_values


def deform(label: np.ndarray, rng: np.random.Generator,
           use_affine: bool, use_elastic: bool) -> np.ndarray:
    """Random affine and/or smooth elastic warp, nearest neighbour only."""
    out = label
    shape = label.shape
    shape_f = np.array(shape, dtype=float)
    if use_affine:
        angle = rng.uniform(-25, 25)
        scale = rng.uniform(0.8, 1.2, size=2)
        shear = rng.uniform(-0.25, 0.25)
        theta = np.deg2rad(angle)
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        mat = rot @ np.diag(scale) @ np.array([[1, shear], [0, 1]])
        centre = shape_f / 2.0
        offset = centre - mat @ centre + rng.normal(0, 8, size=2)
        out = ndimage.affine_transform(out, mat, offset=offset, order=0,
                                       mode="nearest")
    if use_elastic:
        sigma = rng.uniform(8, 24)
        alpha = rng.uniform(6, 18)
        dy = ndimage.gaussian_filter(rng.normal(size=shape), sigma) * alpha
        dx = ndimage.gaussian_filter(rng.normal(size=shape), sigma) * alpha
        grid_y, grid_x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                                     indexing="ij")
        out = ndimage.map_coordinates(
            out, [grid_y + dy, grid_x + dx], order=0, mode="nearest")
    return out.astype(np.uint8)


def paint(label: np.ndarray, cortex_values: set[int],
          rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """Random intensity per label; cortex is randomly darker or brighter."""
    cortex_darker = bool(rng.integers(0, 2))
    if cortex_darker:
        cortex_level = rng.uniform(35, 120)
        tissue_level = rng.uniform(130, 225)
        background_level = rng.uniform(200, 255)
    else:
        cortex_level = rng.uniform(150, 240)
        tissue_level = rng.uniform(60, 150)
        background_level = rng.uniform(0, 60)
    image = np.full(label.shape, background_level, dtype=np.float32)
    image[label == TISSUE] = tissue_level
    for value in cortex_values:
        image[label == value] = cortex_level + rng.normal(0, 12)
    image += rng.normal(0, rng.uniform(3, 14), size=image.shape)
    return image, cortex_darker


def add_artefacts(image: np.ndarray, rng: np.random.Generator,
                  bias: bool, blur: bool, downscale: bool) -> np.ndarray:
    if bias:
        field = ndimage.gaussian_filter(rng.normal(size=image.shape), 40)
        field = field / (np.abs(field).max() + 1e-6)
        image = image * (1.0 + rng.uniform(0.1, 0.3) * field)
    if blur:
        image = ndimage.gaussian_filter(image, rng.uniform(0.4, 1.8))
    if downscale:
        factor = rng.uniform(1.2, 2.0)
        small = np.array(Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
                         .resize((max(8, int(image.shape[1] / factor)),
                                  max(8, int(image.shape[0] / factor))),
                                 Image.BILINEAR))
        image = np.array(Image.fromarray(small).resize(
            (image.shape[1], image.shape[0]), Image.BICUBIC), dtype=np.float32)
    return np.clip(image, 0, 255)


def make_one(gray: np.ndarray, mask: np.ndarray,
             rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    """One synthetic (image, 0/1 mask) pair plus a record of what was used."""
    layers = int(rng.integers(0, 7)) if rng.random() < 0.5 else 0
    label, cortex_values = build_label_map(gray, mask, rng, layers)
    use_affine = bool(rng.random() < 0.85)
    use_elastic = bool(rng.random() < 0.85)
    deformed = deform(label, rng, use_affine, use_elastic)
    image, cortex_darker = paint(deformed, cortex_values, rng)
    image = add_artefacts(image, rng, bias=bool(rng.random() < 0.7),
                          blur=bool(rng.random() < 0.6),
                          downscale=bool(rng.random() < 0.4))
    target = np.isin(deformed, list(cortex_values)).astype(np.uint8)
    record = {
        "layers": layers,
        "affine": use_affine,
        "elastic": use_elastic,
        "cortex_darker": cortex_darker,
        "cortex_fraction": round(float(target.mean()), 4),
    }
    return image.astype(np.uint8), target, record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--per-mask", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview", default=None,
                        help="write a montage of the first N synthetic pairs")
    args = parser.parse_args(argv)

    source = Path(args.source)
    out = Path(args.out)
    images_dir = out / "imagesTr"
    labels_dir = out / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    real_images = sorted((source / "imagesTr").glob("*.png"))
    if not real_images:
        print(f"no real pairs under {source}", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    records: list[dict] = []
    written = 0
    rejected = 0
    for image_path in real_images:
        case = image_path.name[: -len("_0000.png")]
        label_path = source / "labelsTr" / f"{case}.png"
        with Image.open(image_path) as raw:
            gray = np.array(raw.convert("L"))
        with Image.open(label_path) as raw:
            mask = (np.array(raw.convert("L")) > 0).astype(np.uint8)
        shutil.copy2(image_path, images_dir / image_path.name)
        shutil.copy2(label_path, labels_dir / label_path.name)
        for index in range(args.per_mask):
            # A strong warp can push the cortex out of frame; redraw instead of
            # silently producing fewer samples than requested.
            for _attempt in range(6):
                image, target, record = make_one(gray, mask, rng)
                if target.sum() >= 50:
                    break
            else:
                rejected += 1
                continue
            stem = f"{case}__syn{index:03d}"
            Image.fromarray(image, "L").save(images_dir / f"{stem}_0000.png")
            Image.fromarray(target, "L").save(labels_dir / f"{stem}.png")
            records.append(record)
            written += 1

    total = written + len(real_images)
    (out / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "gray"},
        "labels": {"background": 0, "cortex": 1},
        "numTraining": total,
        "file_ending": ".png",
    }, indent=2) + "\n", encoding="utf-8")

    print(f"synthetic   {written}   (redrawn after 6 failed warps: {rejected})")
    print(f"real        {len(real_images)}")
    print(f"numTraining {total}")
    print(f"out         {out}")
    if records:
        darker = sum(r["cortex_darker"] for r in records) / len(records)
        print(f"cortex darker than tissue: {darker:.0%} of samples")
        print(f"with pseudo layers:        "
              f"{sum(1 for r in records if r['layers']) / len(records):.0%}")
        print(f"mean cortex fraction:      "
              f"{np.mean([r['cortex_fraction'] for r in records]):.3f}")

    if args.preview and records:
        make_preview(out, args.preview, count=5)
    return 0 if written else 1


def make_preview(out: Path, path: str, count: int = 5) -> None:
    """Side-by-side montage: synthetic image above, its target mask below."""
    picks = sorted((out / "imagesTr").glob("*__syn*.png"))[:count]
    if not picks:
        return
    width, height = Image.open(picks[0]).size
    canvas = Image.new("L", (width * len(picks), height * 2), 0)
    for column, image_path in enumerate(picks):
        case = image_path.name[: -len("_0000.png")]
        canvas.paste(Image.open(image_path).convert("L"), (column * width, 0))
        mask = Image.open(out / "labelsTr" / f"{case}.png").convert("L")
        canvas.paste(mask.point(lambda v: 255 if v else 0),
                     (column * width, height))
    canvas.save(path)
    print(f"preview     {path}")


if __name__ == "__main__":
    sys.exit(main())
