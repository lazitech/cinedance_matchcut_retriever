"""Stage-3 segmentation bake-off CLI."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .config import load_paths
from .models import Detection, PanopticSegmenter, SamAutoSegmenter, filter_detections


def draw_detection_overlay(image: np.ndarray, detections: Sequence[Detection], alpha: float = 0.42) -> np.ndarray:
    """Draw deterministic coloured masks and thin outlines onto an RGB image."""

    base = np.asarray(image, dtype=np.uint8).copy()
    if base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("image must be an RGB array")
    overlay = base.astype(np.float32)
    colors = np.asarray([(230, 80, 60), (55, 150, 235), (80, 190, 105), (225, 165, 45), (175, 95, 210), (35, 185, 185)], dtype=np.float32)
    for index, det in enumerate(detections):
        mask = np.asarray(det.mask, dtype=bool)
        if mask.shape != base.shape[:2]:
            raise ValueError("detection mask shape does not match image")
        color = colors[index % len(colors)]
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color
        edge = mask.copy()
        edge[1:] &= ~mask[:-1]
        edge[:-1] |= mask[1:] & ~mask[:-1]
        edge[:, 1:] |= mask[:, :-1] & ~mask[:, 1:]
        edge[:, :-1] |= mask[:, 1:] & ~mask[:, :-1]
        overlay[edge] = color
    out = np.clip(overlay, 0, 255).astype(np.uint8)
    # Keep the count on the image itself so saved panels remain interpretable
    # even when separated from the sheet title.
    panel = Image.fromarray(out)
    ImageDraw.Draw(panel).text((4, 4), str(len(detections)), fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
    out = np.asarray(panel)
    return out


def render_sheet(image: np.ndarray, methods: Mapping[str, Sequence[Detection]]) -> np.ndarray:
    """Render original image beside one overlay per method."""

    panels = [np.asarray(image, dtype=np.uint8)]
    labels = ["original"]
    for name, detections in methods.items():
        panels.append(draw_detection_overlay(image, detections))
        labels.append(f"{name} ({len(detections)})")
    height = max(panel.shape[0] for panel in panels)
    width = sum(panel.shape[1] for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel, label in zip(panels, labels):
        pil_panel = Image.fromarray(panel)
        canvas.paste(pil_panel, (x, 0))
        x += panel.shape[1]
    return np.asarray(canvas)


# Public descriptive alias used by callers that want to make the intent clear.
render_bakeoff_sheet = render_sheet
draw_overlay = draw_detection_overlay


def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _area_stats(detections: list[list[Detection]], frame_hw: tuple[int, int]) -> dict[str, float]:
    h, w = frame_hw
    fractions = np.asarray([det.mask.mean() for image_dets in detections for det in image_dets], dtype=float)
    if not len(fractions):
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0}
    return {key: float(value) for key, value in zip(("min", "p25", "median", "p75", "max"), np.quantile(fractions, [0, .25, .5, .75, 1]))}


def _method_summary(detections: list[list[Detection]], times: list[float], area_stats: dict[str, float], peak_gpu_memory: int | None) -> dict:
    counts = np.asarray([len(x) for x in detections], dtype=float)
    return {
        "images": len(detections),
        "total_seconds": float(sum(times)),
        "seconds_per_image": float(np.mean(times)) if times else 0.0,
        "objects_per_image": {"mean": float(counts.mean()) if len(counts) else 0.0, "median": float(np.median(counts)) if len(counts) else 0.0, "p90": float(np.quantile(counts, .9)) if len(counts) else 0.0},
        "area_fraction": area_stats,
        "peak_gpu_memory_bytes": peak_gpu_memory,
        "peak_gpu_memory": peak_gpu_memory,
    }


def run(args: argparse.Namespace) -> None:
    paths = load_paths(args.config)
    library_root = paths.library_root
    keyframes = pd.read_parquet(library_root / "keyframes.parquet")
    if args.n < len(keyframes):
        rng = np.random.default_rng(args.seed)
        rows = keyframes.iloc[rng.choice(len(keyframes), size=args.n, replace=False)].reset_index(drop=True)
    else:
        rows = keyframes.reset_index(drop=True)
    images = [_load_image(library_root / str(row.path)) for row in rows.itertuples()]
    segmenters = {"mask2former": PanopticSegmenter(device=args.device, cache_dir=paths.hf_cache), "sam2_auto": SamAutoSegmenter(device=args.device, cache_dir=paths.hf_cache)}
    out_dir = Path(args.out) if args.out else library_root / "bakeoff"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sheets").mkdir(exist_ok=True)
    all_detections: dict[str, list[list[Detection]]] = {}
    summary: dict[str, dict] = {}
    for method, segmenter in segmenters.items():
        if images:
            segmenter.segment([images[0]])  # warm-up, excluded from timings
        try:
            import torch
            torch.cuda.reset_peak_memory_stats(args.device)
        except (ImportError, RuntimeError):
            torch = None
        detections: list[list[Detection]] = []
        times: list[float] = []
        for image in images:
            start = time.perf_counter()
            raw = segmenter.segment([image])[0]
            times.append(time.perf_counter() - start)
            kept = filter_detections(raw, image.shape[:2], max_objects=args.max_objects)
            detections.append(kept)
        all_detections[method] = detections
        peak = int(torch.cuda.max_memory_allocated(args.device)) if torch is not None and torch.cuda.is_available() else None
        summary[method] = _method_summary(detections, times, _area_stats(detections, images[0].shape[:2]) if images else {}, peak)
        object_rows = []
        for row, image_dets in zip(rows.itertuples(), detections):
            for det in image_dets:
                ys, xs = np.nonzero(det.mask)
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else [0, 0, 0, 0]
                object_rows.append({"keyframe_id": str(row.keyframe_id), "method": method, "score": float(det.score), "label": det.label, "area_frac": float(det.mask.mean()), "bbox": bbox})
        pd.DataFrame(object_rows, columns=["keyframe_id", "method", "score", "label", "area_frac", "bbox"]).to_parquet(out_dir / f"objects_{method}.parquet", index=False)
    for index in range(min(args.sheets, len(images))):
        methods = {name: detections[index] for name, detections in all_detections.items()}
        Image.fromarray(render_sheet(images[index], methods)).save(out_dir / "sheets" / f"{index:04d}.jpg", quality=92)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out")
    parser.add_argument("--sheets", type=int, default=40)
    parser.add_argument("--max-objects", type=int, default=None)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
