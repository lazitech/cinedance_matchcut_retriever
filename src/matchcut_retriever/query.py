"""Command line query interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import load_paths
from .render import render_results
from .retrieval import Library, QueryObject, rank
from .retrieval_config import load_retrieval_config, override_config


def _read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def _prepare(image, x, y):
    """Prepare an image to the library's long-side convention."""
    try:
        from .framing import prepare_image  # optional parallel card
        prepared, transform = prepare_image(image)
        # Support either a callable coordinate mapper or a transform object.
        if hasattr(transform, "map_point"):
            x, y = transform.map_point(float(x), float(y))
        elif callable(transform):
            x, y = transform(float(x), float(y))
        return np.asarray(prepared), float(x), float(y)
    except (ImportError, ModuleNotFoundError):
        h, w = image.shape[:2]
        scale = min(1.0, 640.0 / max(h, w))
        if scale == 1.0:
            return image, float(x), float(y)
        out = np.asarray(Image.fromarray(image).resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS))
        return out, float(x * scale), float(y * scale)


def _candidate_preview(image, detections, out_path):
    thumbs = []
    font = ImageFont.load_default()
    for index, det in enumerate(detections):
        panel = image.copy()
        contours, _ = cv2.findContours(det.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.polylines(panel, contours, True, (255, 70, 40), 3)
        pil = Image.fromarray(panel)
        ImageDraw.Draw(pil).text((8, 8), str(index), fill=(255, 255, 0), font=font)
        thumbs.append(np.asarray(pil))
    if not thumbs:
        Image.fromarray(image).save(out_path)
        return
    h, w = image.shape[:2]
    tile_w = 320
    tile_h = max(1, round(h * tile_w / w))
    cols = min(4, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        tile = cv2.resize(thumb, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        canvas[(i // cols) * tile_h:(i // cols + 1) * tile_h, (i % cols) * tile_w:(i % cols + 1) * tile_w] = tile
    Image.fromarray(canvas).save(out_path, quality=92)


def build_parser():
    parser = argparse.ArgumentParser(description="Rank library objects for a clicked query subject")
    parser.add_argument("--image", required=True)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--prepared", action="store_true")
    parser.add_argument("--mask-index", type=int, default=None)
    parser.add_argument("--mask")
    parser.add_argument("--mode", choices=("hard", "soft"), default="soft")
    parser.add_argument("--semantic", type=float, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--min-scale", type=float)
    parser.add_argument("--max-scale", type=float)
    parser.add_argument("--max-translate", type=float)
    parser.add_argument("--max-rot", type=float)
    parser.add_argument("--coarse-n", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--config")
    parser.add_argument("--library-root")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    image = _read_rgb(args.image)
    if args.prepared:
        prepared, x, y = image, args.x, args.y
    else:
        prepared, x, y = _prepare(image, args.x, args.y)
    detections = []
    if args.mask:
        mask = np.asarray(Image.open(args.mask).convert("L")) > 0
        if mask.shape != prepared.shape[:2]:
            raise ValueError(f"mask shape {mask.shape} does not match image {prepared.shape[:2]}")
    else:
        from .models import SamPointSegmenter
        detections = SamPointSegmenter(device=args.device).masks_at_point(prepared, x, y)
        if not detections:
            raise RuntimeError("SAM returned no masks at the requested point")
        _candidate_preview(prepared, detections, out / "candidates.jpg")
        index = args.mask_index if args.mask_index is not None else 0
        if index < 0 or index >= len(detections):
            raise IndexError(f"mask index {index} out of range (0..{len(detections)-1})")
        mask = detections[index].mask
    Image.fromarray((mask.astype(np.uint8) * 255)).save(out / "query_mask.png")
    clip_vector = None
    semantic = 0.0 if args.semantic is None else float(args.semantic)
    if semantic != 0.0:
        from .models import ClipEmbedder
        clip_vector = ClipEmbedder(device=args.device).embed_masked([prepared], [mask])[0]
    config = load_retrieval_config(args.config if args.config and Path(args.config).name == "retrieval.toml" else None)
    config = override_config(config, min_scale=args.min_scale, max_scale=args.max_scale, max_translate_frac=args.max_translate, max_rot_deg=args.max_rot, coarse_n=args.coarse_n, k=args.k, semantic_weight=semantic)
    if args.coarse_n is not None:
        from dataclasses import replace
        config = replace(config, hard_coarse_n=int(args.coarse_n))
    if args.library_root:
        library_root = Path(args.library_root)
    else:
        path_cfg = args.config if args.config and "paths" in Path(args.config).read_text() else None
        library_root = load_paths(path_cfg).library_root
    library = Library.load(library_root)
    query = QueryObject.from_mask(prepared, mask, clip_vector)
    results = rank(library, query, args.mode, semantic_weight=semantic, k=config.k, params=config)
    render_results(prepared, mask, results, library, out / "grid.jpg")
    payload = {
        "query": {"image": str(args.image), "prepared": bool(args.prepared), "x": float(x), "y": float(y), "mask_index": args.mask_index, "height": int(prepared.shape[0]), "width": int(prepared.shape[1])},
        "parameters": {"mode": args.mode, "semantic_weight": semantic, "k": config.k, "coarse_n": config.coarse_n, "hard_coarse_n": config.hard_coarse_n, "workers": config.workers, "hard_limits": config.hard_limits.__dict__, "soft_params": config.soft_params.__dict__},
        "timings": results.timings,
        "results": [item.to_dict() for item in results],
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
