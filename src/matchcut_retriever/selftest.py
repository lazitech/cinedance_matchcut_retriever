"""Synthetic-friendly Recall@K self-test for the match-cut retriever.

The module intentionally keeps model construction lazy.  This makes the CPU
test suite usable without SAM/CLIP weights while the CLI can run the complete
query pipeline on a configured library.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from .config import load_paths
from .framing import prepare_image
from .query import _transform_mask
from .render import render_results
from .retrieval import Library, QueryObject, rank
from .retrieval_config import load_retrieval_config, override_config


def sample_object_ids(objects, n: int = 200, seed: int = 0) -> list[int]:
    """Return a deterministic uniform sample of object ids without replacement."""
    size = len(objects)
    if size == 0 or n <= 0:
        return []
    count = min(int(n), size)
    rng = np.random.default_rng(int(seed))
    ids = np.arange(size, dtype=np.int64)
    return [int(value) for value in rng.choice(ids, size=count, replace=False)]


def _warp(image: np.ndarray, mask: np.ndarray, scale: float, angle_deg: float, tx: float, ty: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle_deg), float(scale))
    matrix[:, 2] += (float(tx), float(ty))
    warped_image = cv2.warpAffine(np.asarray(image), matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    warped_mask = cv2.warpAffine(np.asarray(mask, dtype=np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return warped_image, warped_mask


def perturb_image_mask(image: np.ndarray, mask: np.ndarray, params: dict[str, float | int] | None = None, rng: np.random.Generator | None = None, max_retries: int = 12) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Perturb an analysis-resolution frame and its mask.

    ``params`` may specify any of scale, angle_deg, tx_frac, ty_frac,
    brightness, contrast, saturation and jpeg_quality.  Fractions are relative
    to the frame dimensions.  The returned parameter dictionary is JSON-safe.
    """
    arr = np.asarray(image, dtype=np.uint8)
    original = np.asarray(mask, dtype=bool)
    if arr.ndim != 3 or arr.shape[:2] != original.shape:
        raise ValueError("image must be HxWx3 and mask must match its dimensions")
    if not np.any(original):
        raise ValueError("mask must be non-empty")
    rng = rng or np.random.default_rng()
    base_area = int(original.sum())
    requested = dict(params or {})
    deterministic = params is not None
    for attempt in range(max(1, int(max_retries))):
        values = {
            "scale": float(requested.get("scale", 1.0 if deterministic else rng.uniform(0.9, 1.1))),
            "angle_deg": float(requested.get("angle_deg", 0.0 if deterministic else rng.uniform(-3.0, 3.0))),
            "tx_frac": float(requested.get("tx_frac", 0.0 if deterministic else rng.uniform(-0.05, 0.05))),
            "ty_frac": float(requested.get("ty_frac", 0.0 if deterministic else rng.uniform(-0.05, 0.05))),
            "brightness": float(requested.get("brightness", 1.0 if deterministic else rng.uniform(0.85, 1.15))),
            "contrast": float(requested.get("contrast", 1.0 if deterministic else rng.uniform(0.85, 1.15))),
            "saturation": float(requested.get("saturation", 1.0 if deterministic else rng.uniform(0.85, 1.15))),
            "jpeg_quality": int(requested.get("jpeg_quality", 100 if deterministic else rng.integers(60, 91))),
            "attempt": attempt,
        }
        warped, warped_mask = _warp(arr, original, values["scale"], values["angle_deg"], values["tx_frac"] * arr.shape[1], values["ty_frac"] * arr.shape[0])
        if int(warped_mask.sum()) >= 0.5 * base_area:
            # Colour operations are done in PIL for consistent RGB semantics.
            pil = Image.fromarray(warped, mode="RGB")
            pil = ImageEnhance.Brightness(pil).enhance(values["brightness"])
            pil = ImageEnhance.Contrast(pil).enhance(values["contrast"])
            pil = ImageEnhance.Color(pil).enhance(values["saturation"])
            if (values["scale"] == 1.0 and values["angle_deg"] == 0.0 and values["tx_frac"] == values["ty_frac"] == 0.0 and values["brightness"] == values["contrast"] == values["saturation"] == 1.0):
                # Preserve an explicitly requested identity perturbation exactly;
                # stochastic runs virtually never land on this branch.
                out = arr.copy()
            else:
                encoded = io.BytesIO()
                pil.save(encoded, format="JPEG", quality=values["jpeg_quality"])
                encoded.seek(0)
                out = np.asarray(Image.open(encoded).convert("RGB"), dtype=np.uint8)
            values["mask_area"] = int(warped_mask.sum())
            values["area_ratio"] = float(warped_mask.sum() / base_area)
            values["retries"] = attempt
            return out, warped_mask, values
    raise RuntimeError("unable to draw a valid perturbation")


def choose_interior_click(mask: np.ndarray, rng: np.random.Generator | None = None) -> tuple[int, int, float]:
    """Draw an (x, y) pixel from the inner half of a mask's distance transform."""
    arr = np.asarray(mask, dtype=bool)
    if not np.any(arr):
        raise ValueError("mask must be non-empty")
    distance = cv2.distanceTransform(arr.astype(np.uint8), cv2.DIST_L2, 3)
    threshold = float(distance.max()) * 0.5
    ys, xs = np.where(arr & (distance >= threshold - 1e-8))
    if len(xs) == 0:
        ys, xs = np.where(arr)
    rng = rng or np.random.default_rng()
    index = int(rng.integers(0, len(xs)))
    return int(xs[index]), int(ys[index]), float(distance[ys[index], xs[index]])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = np.asarray(a, bool), np.asarray(b, bool)
    union = np.logical_or(aa, bb).sum()
    return float(np.logical_and(aa, bb).sum() / union) if union else 0.0


def _record_files(out: Path) -> list[Path]:
    return sorted(out.glob("records*.jsonl"))


def _load_records(out: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _record_files(out):
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # A killed process can leave a truncated final line; the
                    # completed samples remain valid and resumable.
                    continue
    return records


def _hit_for_source(results, clip_id: str, shot_no: int, object_id: int) -> dict[str, Any]:
    for index, result in enumerate(results, 1):
        if (str(result.clip_id), int(result.shot_no)) == (str(clip_id), int(shot_no)):
            return {"rank": index, "exact_object": int(result.object_id) == int(object_id), "iou": float(result.iou), "object_id": int(result.object_id)}
    return {"rank": None, "exact_object": False, "iou": None, "object_id": None}


def _bucket(area_frac: float) -> str:
    return "<2%" if area_frac < 0.02 else ("2-10%" if area_frac < 0.10 else ">=10%")


def summarize_records(records: Iterable[dict[str, Any]], k_values: tuple[int, ...] = (1, 10, 50)) -> dict[str, Any]:
    records = list(records)
    summary: dict[str, Any] = {"n": len(records), "modes": {}, "sam": {}}
    recorded_k = [int(record["k_max"]) for record in records if record.get("k_max") is not None]
    supported_max = min(recorded_k) if recorded_k else max(k_values, default=0)
    cutoffs = tuple(sorted(set(k_values) | ({supported_max} if recorded_k else set())))
    keys = sorted({(source, mode) for record in records for source, value in record.get("results", {}).items() for mode in value})
    for source, mode in keys:
        rows = [r for r in records if r.get("results", {}).get(source, {}).get(mode) is not None]
        hits = [r["results"][source][mode] for r in rows]
        ranks = [h["rank"] for h in hits]
        times = [float(h.get("timing_s", h.get("rank_time_s", h.get("time_s", 0.0)))) for h in hits]
        recall = {
            str(k): (sum(rank is not None and rank <= k for rank in ranks) / len(rows) if rows else 0.0)
            if k <= supported_max else None
            for k in cutoffs
        }
        exact_hits = sum(h.get("rank") is not None and bool(h.get("exact_object", False)) for h in hits)
        hit_count = sum(h.get("rank") is not None for h in hits)
        entry: dict[str, Any] = {"n": len(rows), "recall": recall, "recall_at_1": recall.get("1"), "recall_at_10": recall.get("10"), "recall_at_50": recall.get("50"), "exact_object_share": exact_hits / hit_count if hit_count else 0.0, "median_rank_time_s": float(np.median(times)) if times else None}
        entry.update({f"recall@{k}": value for k, value in recall.items()})
        for label, predicate in (("area", lambda r: _bucket(float(r.get("area_frac", 0.0)))), ("segmenter", lambda r: str(r.get("segmenter", "unknown")))):
            groups: dict[str, list[int | None]] = {}
            for r, h in zip(rows, hits):
                groups.setdefault(predicate(r), []).append(h["rank"])
            entry[label] = {
                name: {
                    "n": len(vals),
                    "recall": {
                        str(k): (sum(v is not None and v <= k for v in vals) / len(vals) if vals else 0.0)
                        if k <= supported_max else None
                        for k in cutoffs
                    },
                }
                for name, vals in sorted(groups.items())
            }
        summary["modes"].setdefault(source, {})[mode] = entry
    sam_rows = [r.get("sam") for r in records if r.get("sam")]
    if sam_rows:
        best = [float(r["best_iou"]) for r in sam_rows if r.get("best_iou") is not None]
        summary["sam"] = {"n": len(sam_rows), "median_best_iou": float(np.median(best)) if best else None, "candidate_0_best_share": sum(int(r.get("best_index", -1)) == 0 for r in sam_rows) / len(sam_rows)}
    failures = [r for r in records if r.get("failure")]
    summary["failures"] = [{"sample_index": r.get("sample_index"), "reason": r.get("failure")} for r in failures]
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    lines = ["# Match-cut retriever self-test", "", "| Mask source | Mode | n | Recall@1 | Recall@10 | Recall@50 | Exact object | Median rank time (s) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for source, modes in summary.get("modes", {}).items():
        for mode, item in modes.items():
            recall = item["recall"]
            fmt = lambda value: "—" if value is None else f"{value:.3f}"
            median = "—" if item["median_rank_time_s"] is None else f"{item['median_rank_time_s']:.2f}"
            lines.append(f"| {source} | {mode} | {item['n']} | {fmt(recall.get('1'))} | {fmt(recall.get('10'))} | {fmt(recall.get('50'))} | {item['exact_object_share']:.3f} | {median} |")
    for source, modes in summary.get("modes", {}).items():
        for mode, item in modes.items():
            lines += ["", f"### {source} / {mode} breakdown", "", "| Group | n | Recall@1 | Recall@10 | Recall@50 |", "|---|---:|---:|---:|---:|"]
            for label in ("area", "segmenter"):
                for group, values in item.get(label, {}).items():
                    r = values["recall"]
                    lines.append(f"| {label}={group} | {values['n']} | {fmt(r.get('1'))} | {fmt(r.get('10'))} | {fmt(r.get('50'))} |")
    if summary.get("sam", {}).get("n"):
        lines += ["", f"SAM: median best IoU {summary['sam']['median_best_iou']:.3f}; candidate 0 best share {summary['sam']['candidate_0_best_share']:.3f}."]
    if summary.get("failures"):
        lines += ["", f"Failures: {len(summary['failures'])}."]
    return "\n".join(lines) + "\n"


def run_selftest(library_root: str | Path, out: str | Path, n: int = 200, seed: int = 0, mask_source: str = "both", modes: Iterable[str] = ("soft", "hard"), k_max: int = 50, semantic: float = 0.0, config_path: str | Path | None = None, retrieval_config: str | Path | None = None, device: str = "cuda:0", workers: int | None = None, coarse_n: int | None = None, hard_coarse_n: int | None = None, shard: str | None = None, point_segmenter: Any = None, embedder: Any = None, render_failures: int = 0) -> dict[str, Any]:
    """Run or resume the self-test and write summary.json/summary.md."""
    root, out = Path(library_root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    library = Library.load(root)
    shard_index, shard_count = 0, 1
    if shard:
        left, right = str(shard).split("/", 1); shard_index, shard_count = int(left), int(right)
        if shard_count <= 0 or not 0 <= shard_index < shard_count: raise ValueError("invalid --shard I/N")
    records_path = out / (f"records-{shard_index:03d}-of-{shard_count:03d}.jsonl" if shard_count > 1 else "records.jsonl")
    existing = {int(r.get("sample_index")) for r in _load_records(out)}
    config = load_retrieval_config(retrieval_config)
    config = override_config(config, semantic_weight=float(semantic), workers=workers, coarse_n=coarse_n, hard_coarse_n=hard_coarse_n, k=int(k_max))
    modes = [str(m).strip().lower() for m in modes if str(m).strip()]
    if any(mode not in {"soft", "hard"} for mode in modes):
        raise ValueError("modes must contain only soft and hard")
    if mask_source not in {"sam", "gt", "both"}:
        raise ValueError("mask_source must be sam, gt, or both")
    sources = [mask_source] if mask_source in {"gt", "sam"} else ["gt", "sam"]
    sam_model = point_segmenter
    if "sam" in sources and sam_model is None:
        from .models import SamPointSegmenter
        paths = load_paths(config_path)
        sam_model = SamPointSegmenter(device=device, cache_dir=paths.hf_cache)
    if semantic and embedder is None:
        from .models import ClipEmbedder
        paths = load_paths(config_path)
        embedder = ClipEmbedder(device=device, cache_dir=paths.hf_cache)
    with records_path.open("a", encoding="utf-8") as stream:
        for sample_index, object_id in enumerate(sample_object_ids(library.objects, n, seed)):
            if sample_index % shard_count != shard_index or sample_index in existing:
                continue
            row = library.objects.iloc[int(object_id)]
            record: dict[str, Any] = {"sample_index": sample_index, "object_id": int(object_id), "clip_id": str(row.clip_id), "shot_no": int(row.shot_no), "keyframe_id": str(row.keyframe_id), "area_frac": float(row.area_frac), "segmenter": str(row.segmenter), "k_max": int(k_max), "results": {}}
            try:
                image = np.asarray(Image.open(library.keyframe_path(row.keyframe_id)).convert("RGB"), dtype=np.uint8)
                source_mask = library.mask(object_id)
                rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(sample_index), int(object_id)]))
                perturbed, perturbed_mask, params = perturb_image_mask(image, source_mask, rng=rng)
                prepared, transform = prepare_image(perturbed, long_side=640)
                prepared_mask = _transform_mask(perturbed_mask, prepared.shape, transform)
                click_x, click_y, click_distance = choose_interior_click(prepared_mask, rng)
                record.update({"perturbation": params, "prepared_shape": list(prepared.shape), "click": {"x": click_x, "y": click_y, "distance": click_distance}})
                masks = {"gt": prepared_mask}
                if "sam" in sources:
                    detections = sam_model.masks_at_point(prepared, click_x, click_y)
                    if not detections:
                        raise RuntimeError("SAM returned no masks")
                    ious = [_iou(det.mask, prepared_mask) for det in detections]
                    best_index = int(np.argmax(ious))
                    masks["sam"] = np.asarray(detections[best_index].mask, bool)
                    record["sam"] = {"best_iou": float(ious[best_index]), "best_index": best_index, "candidate_0_iou": float(ious[0]), "candidate_count": len(detections)}
                for source in sources:
                    record["results"][source] = {}
                    query_mask = masks[source]
                    clip_vector = None
                    if semantic:
                        clip_vector = embedder.embed_masked([prepared], [query_mask])[0]
                    query = QueryObject.from_mask(prepared, query_mask, clip_vector)
                    for mode in modes:
                        started = time.perf_counter()
                        results = rank(library, query, mode, semantic_weight=semantic, k=k_max, params=config)
                        elapsed = time.perf_counter() - started
                        hit = _hit_for_source(results, row.clip_id, row.shot_no, object_id)
                        hit["timing_s"] = float(elapsed)
                        hit["rank_timings"] = dict(getattr(results, "timings", {}))
                        record["results"][source][mode] = hit
                        if hit["rank"] is None and render_failures:
                            failure_dir = out / "failures" / source / mode
                            count = len(list(failure_dir.glob("*.jpg")))
                            if count < render_failures:
                                render_results(prepared, query_mask, results, library, failure_dir / f"{sample_index:06d}.jpg")
            except Exception as exc:  # failures count as misses and remain resumable
                record["failure"] = f"{type(exc).__name__}: {exc}"
                for source in sources:
                    record["results"].setdefault(source, {})
                    for mode in modes:
                        record["results"][source].setdefault(mode, {"rank": None, "exact_object": False, "iou": None, "timing_s": 0.0})
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
    records = _load_records(out)
    summary = summarize_records(records)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / "summary.md").write_text(_markdown(summary))
    print((out / "summary.md").read_text(), end="")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure Recall@K of the match-cut query pipeline")
    parser.add_argument("--config"); parser.add_argument("--retrieval-config"); parser.add_argument("--library-root")
    parser.add_argument("--out", required=True); parser.add_argument("--n", type=int, default=200); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask-source", choices=("sam", "gt", "both"), default="both"); parser.add_argument("--modes", default="soft,hard"); parser.add_argument("--k-max", type=int, default=50)
    parser.add_argument("--semantic", type=float, default=0.0); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--workers", type=int); parser.add_argument("--coarse-n", type=int); parser.add_argument("--hard-coarse-n", type=int)
    parser.add_argument("--shard"); parser.add_argument("--summarize-only", action="store_true"); parser.add_argument("--render-failures", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    if args.summarize_only:
        summary = summarize_records(_load_records(out)); (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n"); (out / "summary.md").write_text(_markdown(summary)); print((out / "summary.md").read_text(), end=""); return 0
    if not args.library_root:
        args.library_root = str(load_paths(args.config).library_root)
    run_selftest(args.library_root, out, n=args.n, seed=args.seed, mask_source=args.mask_source, modes=args.modes.split(","), k_max=args.k_max, semantic=args.semantic, config_path=args.config, retrieval_config=args.retrieval_config, device=args.device, workers=args.workers, coarse_n=args.coarse_n, hard_coarse_n=args.hard_coarse_n, shard=args.shard, render_failures=args.render_failures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
