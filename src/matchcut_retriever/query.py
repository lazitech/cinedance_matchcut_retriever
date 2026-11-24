"""Command line query interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import load_paths
from .render import render_results
from .retrieval import Library, QueryObject, rank
from .retrieval_config import load_retrieval_config, override_config
from .shape import MAX_AREA_FRAC, MIN_AREA_FRAC


def _read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def _prepare(image, x, y):
    """Prepare an image to the library's long-side convention."""
    try:
        from .framing import prepare_image  # optional parallel card
        prepared, transform = prepare_image(image, long_side=640)
        x, y = transform.to_prepared(float(x), float(y))
        return np.asarray(prepared), float(x), float(y), transform
    except (ImportError, ModuleNotFoundError):
        h, w = image.shape[:2]
        scale = min(1.0, 640.0 / max(h, w))
        if scale == 1.0:
            return image, float(x), float(y), None
        out = np.asarray(Image.fromarray(image).resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS))
        return out, float(x * scale), float(y * scale), scale


def _transform_mask(mask, prepared_shape, transform):
    """Apply the image preparation transform to a mask with nearest-neighbour sampling."""
    if transform is None:
        return mask
    if isinstance(transform, (int, float)):
        return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((prepared_shape[1], prepared_shape[0]), Image.Resampling.NEAREST), dtype=np.uint8) > 0
    x0, y0, x1, y1 = (int(value) for value in transform.crop)
    cropped = np.asarray(mask, dtype=np.uint8)[y0:y1, x0:x1]
    return np.asarray(Image.fromarray(cropped).resize((prepared_shape[1], prepared_shape[0]), Image.Resampling.NEAREST), dtype=np.uint8) > 0


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


def query_warnings(mask, mode, limits, n_results):
    """Explain query masks the library cannot match well, and empty result lists."""
    area = float(np.asarray(mask, dtype=bool).mean())
    warnings = []
    if mode == "hard" and (area * limits.max_scale ** 2 < MIN_AREA_FRAC or area * limits.min_scale ** 2 > MAX_AREA_FRAC):
        warnings.append(
            f"query object covers {area:.2%} of the frame; library objects cover {MIN_AREA_FRAC:.1%}-{MAX_AREA_FRAC:.0%}, "
            f"so no candidate fits the hard scale limits {limits.min_scale}-{limits.max_scale}; pick another --mask-index or use --mode soft"
        )
    elif area < MIN_AREA_FRAC or area > MAX_AREA_FRAC:
        warnings.append(
            f"query object covers {area:.2%} of the frame, outside the library's {MIN_AREA_FRAC:.1%}-{MAX_AREA_FRAC:.0%}; "
            "consider another --mask-index"
        )
    if n_results == 0:
        warnings.append(f"no results in {mode} mode" + (" within the hard limits" if mode == "hard" else ""))
    return warnings


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
    parser.add_argument("--hard-coarse-n", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", help="local paths TOML")
    parser.add_argument("--retrieval-config", default=None, help="retrieval TOML")
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--library-root")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    image = _read_rgb(args.image)
    try:
        paths = load_paths(args.config)
    except FileNotFoundError:
        if args.config is not None or not args.library_root:
            raise
        paths = None
    config = load_retrieval_config(args.retrieval_config)
    semantic = config.semantic_weight if args.semantic is None else float(args.semantic)
    if args.no_clip and semantic != 0.0:
        raise ValueError("--no-clip cannot be combined with a non-zero semantic weight")
    if args.prepared:
        prepared, x, y, prep_transform = image, args.x, args.y, None
    else:
        prepared, x, y, prep_transform = _prepare(image, args.x, args.y)
    detections = []
    if args.mask:
        mask = np.asarray(Image.open(args.mask).convert("L")) > 0
        mask = _transform_mask(mask, prepared.shape, prep_transform)
    else:
        from .models import SamPointSegmenter
        detections = SamPointSegmenter(device=args.device, cache_dir=(paths.hf_cache if paths else None)).masks_at_point(prepared, x, y)
        if not detections:
            raise RuntimeError("SAM returned no masks at the requested point")
        _candidate_preview(prepared, detections, out / "candidates.jpg")
        index = args.mask_index if args.mask_index is not None else 0
        if index < 0 or index >= len(detections):
            raise IndexError(f"mask index {index} out of range (0..{len(detections)-1})")
        mask = detections[index].mask
    Image.fromarray((mask.astype(np.uint8) * 255)).save(out / "query_mask.png")
    clip_vector = None
    if not args.no_clip:
        from .models import ClipEmbedder
        clip_vector = ClipEmbedder(device=args.device, cache_dir=(paths.hf_cache if paths else None)).embed_masked([prepared], [mask])[0]
    config = override_config(config, min_scale=args.min_scale, max_scale=args.max_scale, max_translate_frac=args.max_translate, max_rot_deg=args.max_rot, coarse_n=args.coarse_n, hard_coarse_n=args.hard_coarse_n, workers=args.workers, k=args.k, semantic_weight=semantic)
    if args.library_root:
        library_root = Path(args.library_root)
    else:
        library_root = paths.library_root
    library = Library.load(library_root)
    query = QueryObject.from_mask(prepared, mask, clip_vector)
    results = rank(library, query, args.mode, semantic_weight=semantic, k=config.k, params=config)
    render_results(prepared, mask, results, library, out / "grid.jpg")
    warnings = query_warnings(mask, args.mode, config.hard_limits, len(results))
    for message in warnings:
        print(f"warning: {message}", file=sys.stderr)
    candidates = [{"index": i, "area_frac": float(np.asarray(det.mask, dtype=bool).mean()), "score": float(det.score)} for i, det in enumerate(detections)]
    payload = {
        "query": {"image": str(args.image), "prepared": bool(args.prepared), "x": float(x), "y": float(y), "mask_index": args.mask_index, "height": int(prepared.shape[0]), "width": int(prepared.shape[1]), "area_frac": float(mask.mean()), "candidates": candidates},
        "warnings": warnings,
        "parameters": {"mode": args.mode, "semantic_weight": semantic, "k": config.k, "coarse_n": config.coarse_n, "hard_coarse_n": config.hard_coarse_n, "workers": config.workers, "hard_axis_ecc_threshold": config.hard_axis_ecc_threshold, "hard_limits": config.hard_limits.__dict__, "soft_params": config.soft_params.__dict__},
        "timings": results.timings,
        "results": [item.to_dict() for item in results],
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
