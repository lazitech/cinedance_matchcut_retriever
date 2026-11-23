"""Build and maintain the object library used by the ranking stage."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd

from .config import load_paths
from .framing import frame_quality
from .masks import rle_decode, rle_encode
from .models import Detection, ClipEmbedder, PanopticSegmenter, SamAutoSegmenter
from .models.common import filter_detections
from .shape import FINGERPRINT_DIM, object_geometry, shape_fingerprint

LOG = logging.getLogger(__name__)

OBJECT_COLUMNS = [
    "object_id", "keyframe_id", "clip_id", "shot_no", "k", "t_s", "frame_w", "frame_h",
    "cx", "cy", "area", "area_frac", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
    "angle", "eccentricity", "n_components", "solidity", "score", "label", "segmenter",
    "mask_offset", "mask_nbytes",
]
PART_COLUMNS = [c for c in OBJECT_COLUMNS if c not in {"object_id", "mask_offset"}]


def clean_mask(mask: np.ndarray, min_component_frac: float = 0.02) -> np.ndarray | None:
    """Remove small connected components and fill small enclosed holes."""
    if not 0 <= min_component_frac <= 1:
        raise ValueError("min_component_frac must be in [0, 1]")
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2 or not arr.any():
        return None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(arr.astype(np.uint8), 8)
    # A one-pixel island is a segmentation speck even when it is the only
    # foreground pixel, so retain the fractional rule's useful absolute floor.
    threshold = max(2.0, float(arr.sum()) * float(min_component_frac))
    kept = np.zeros_like(arr)
    for i in range(1, n):
        if float(stats[i, cv2.CC_STAT_AREA]) >= threshold:
            kept[labels == i] = True
    if not kept.any():
        return None

    # Components of the complement touching the border are outside the object;
    # the other components are holes.
    inv = (~kept).astype(np.uint8)
    nh, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    hole_limit = float(kept.sum()) * 0.01
    border_labels = set(np.unique(np.r_[hole_labels[0], hole_labels[-1],
                                        hole_labels[:, 0], hole_labels[:, -1]]))
    for i in range(1, nh):
        if i not in border_labels and float(hole_stats[i, cv2.CC_STAT_AREA]) < hole_limit:
            kept[hole_labels == i] = True
    return kept if kept.any() else None


def merge_detections(
    primary: list[Detection], secondary: list[Detection], iou_threshold: float = 0.8
) -> list[Detection]:
    """Keep primary detections and secondary masks not duplicated by a primary."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    result = list(primary)
    if not secondary:
        return result
    if not primary:
        return result + list(secondary)
    shape = primary[0].mask.shape
    if any(d.mask.shape != shape for d in [*primary, *secondary]):
        raise ValueError("all detection masks must have the same shape")
    p = np.asarray([d.mask.reshape(-1) for d in primary], dtype=np.uint8)
    p_area = p.sum(axis=1, dtype=np.int64)
    for det in secondary:
        s = det.mask.reshape(-1).astype(np.uint8, copy=False)
        s_area = int(s.sum())
        inter = p @ s
        union = p_area + s_area - inter
        ious = np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)
        if not ious.size or float(ious.max()) <= iou_threshold:
            result.append(det)
    return result


def _solidity(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return 1.0
    hull = cv2.convexHull(np.column_stack((xs, ys)).astype(np.int32))
    hull_area = float(cv2.contourArea(hull))
    return min(1.0, float(mask.sum() / hull_area)) if hull_area > 0 else 1.0


def build_objects(
    image: np.ndarray,
    detections: list[Detection],
    *,
    keyframe_id: str = "",
    clip_id: str = "",
    shot_no: int = 0,
    k: int = 0,
    t_s: float = 0.0,
    max_objects: int = 40,
    min_component_frac: float = 0.02,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    """Clean, filter and describe detections from one analysis-resolution frame."""
    arr = np.asarray(image)
    if arr.ndim < 2:
        raise ValueError("image must have at least two dimensions")
    h, w = arr.shape[:2]
    cleaned: list[Detection] = []
    for det in detections:
        cleaned_mask = clean_mask(det.mask, min_component_frac)
        if cleaned_mask is not None:
            cleaned.append(Detection(cleaned_mask, det.score, det.label, det.source))
    kept = filter_detections(cleaned, (h, w), 0.005, 0.60, max_objects)
    rows: list[dict[str, Any]] = []
    masks: list[np.ndarray] = []
    for det in kept:
        mask = np.asarray(det.mask, dtype=bool)
        geom = object_geometry(mask)
        x0, y0, x1, y1 = geom.bbox
        rows.append({
            "keyframe_id": str(keyframe_id), "clip_id": str(clip_id), "shot_no": int(shot_no),
            "k": int(k), "t_s": float(t_s), "frame_w": int(w), "frame_h": int(h),
            "cx": np.float32(geom.cx), "cy": np.float32(geom.cy), "area": np.float32(geom.area),
            "area_frac": np.float32(geom.area_frac), "bbox_x0": int(x0), "bbox_y0": int(y0),
            "bbox_x1": int(x1), "bbox_y1": int(y1), "angle": np.float32(geom.angle),
            "eccentricity": np.float32(geom.eccentricity),
            "n_components": np.int16(cv2.connectedComponents(mask.astype(np.uint8), 8)[0] - 1),
            "solidity": np.float32(_solidity(mask)), "score": np.float32(det.score),
            "label": det.label, "segmenter": str(det.source),
        })
        masks.append(mask)
    return rows, masks


def _temp_dir(parent: Path, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))


def _clip_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _append_marker(path: Path, clip_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as fh:
        try:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover
            pass
        fh.seek(0)
        existing = [line.strip() for line in fh.read().splitlines() if line.strip()]
        if clip_id not in existing:
            existing.append(clip_id)
        # When clips.txt is present, retain its canonical order.  This makes
        # independent shard runs consolidate byte-for-byte like one run.
        clips_path = path.parent / "clips.txt"
        if clips_path.is_file():
            order = {cid: i for i, cid in enumerate(_clip_list(clips_path))}
            existing.sort(key=lambda cid: order.get(cid, len(order)))
        fh.seek(0)
        fh.truncate()
        fh.write("".join(f"{cid}\n" for cid in existing))
        fh.flush()
        os.fsync(fh.fileno())
        try:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_UN)
        except ImportError:  # pragma: no cover
            pass


class LibraryBuilder:
    """Incrementally build one atomic part per clip."""

    def __init__(self, root, segmenters, embedder, max_objects: int = 40, batch_size: int = 32):
        self.root = Path(root)
        self.segmenters = list(segmenters)
        self.embedder = embedder
        self.max_objects = int(max_objects)
        self.batch_size = max(1, int(batch_size))
        if self.max_objects < 0:
            raise ValueError("max_objects must be non-negative")
        self.root.mkdir(parents=True, exist_ok=True)

    def _processed(self) -> set[str]:
        marker = self.root / "processed_clips.txt"
        return set(_clip_list(marker)) if marker.exists() else set()

    def _segment(self, images: list[np.ndarray]) -> list[list[Detection]]:
        if not images:
            return []
        per_image: list[list[list[Detection]]] = []
        for segmenter in self.segmenters:
            outputs: list[list[Detection]] = []
            for start in range(0, len(images), self.batch_size):
                outputs.extend(segmenter.segment(images[start:start + self.batch_size]))
            if len(outputs) != len(images):
                raise ValueError("segmenter returned the wrong number of detection lists")
            per_image.append(outputs)
        merged: list[list[Detection]] = []
        for i in range(len(images)):
            if not per_image:
                merged.append([])
            elif len(per_image) == 1:
                merged.append(per_image[0][i])
            else:
                merged.append(merge_detections(per_image[0][i], [d for outputs in per_image[1:] for d in outputs[i]]))
        return merged

    def _write_part(self, clip_id: str, rows: list[dict[str, Any]], masks: list[np.ndarray], images: list[np.ndarray]) -> None:
        parts = self.root / "parts"
        final = parts / str(clip_id)
        if final.exists():
            return
        temp = _temp_dir(parts, f".{clip_id}.")
        try:
            blob = bytearray()
            local_offset = 0
            for row, mask in zip(rows, masks):
                encoded = rle_encode(mask)
                row["mask_nbytes"] = len(encoded)
                blob.extend(encoded)
                local_offset += len(encoded)
            part = pd.DataFrame(rows)
            for col in PART_COLUMNS:
                if col not in part:
                    part[col] = pd.Series(dtype="object")
            part = part[PART_COLUMNS]
            part.to_parquet(temp / "objects.parquet", index=False)
            (temp / "masks.bin").write_bytes(blob)
            fps = np.asarray([shape_fingerprint(mask) for mask in masks], dtype=np.float16)
            if fps.size == 0:
                fps = np.empty((0, FINGERPRINT_DIM), dtype=np.float16)
            else:
                fps = fps.reshape(len(masks), FINGERPRINT_DIM)
            np.save(temp / "shape_fp.npy", fps)
            if images:
                embeds = []
                for start in range(0, len(images), self.batch_size):
                    embeds.append(np.asarray(self.embedder.embed_masked(images[start:start + self.batch_size], masks[start:start + self.batch_size]), dtype=np.float32))
                emb = np.concatenate(embeds, axis=0) if embeds else np.empty((0, 0), dtype=np.float32)
            else:
                emb = np.empty((0, 0), dtype=np.float32)
            if emb.ndim != 2 or emb.shape[0] != len(masks):
                raise ValueError("embedder returned an array with the wrong shape")
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = np.divide(emb, norms, out=np.zeros_like(emb), where=norms > 0)
            np.save(temp / "clip_emb.npy", emb.astype(np.float16))
            os.replace(temp, final)
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise

    def process_clips(self, clip_ids: Iterable[str]) -> dict[str, int]:
        keyframes_path = self.root / "keyframes.parquet"
        if not keyframes_path.is_file():
            raise FileNotFoundError(keyframes_path)
        table = pd.read_parquet(keyframes_path)
        parts = self.root / "parts"
        parts.mkdir(parents=True, exist_ok=True)
        done = self._processed()
        summary = {"clips": 0, "keyframes": 0, "skipped_keyframes": 0, "objects": 0}
        for clip_id in clip_ids:
            cid = str(clip_id)
            if cid in done or (parts / cid / "objects.parquet").is_file():
                continue
            rows = table[table["clip_id"].astype(str) == cid].sort_values("keyframe_id")
            usable_images: list[np.ndarray] = []
            usable_meta: list[Any] = []
            for record in rows.itertuples(index=False):
                image = cv2.imread(str(self.root / str(record.path)), cv2.IMREAD_COLOR)
                if image is None:
                    summary["skipped_keyframes"] += 1
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if not bool(frame_quality(image)["usable"]):
                    summary["skipped_keyframes"] += 1
                    continue
                usable_images.append(image)
                usable_meta.append(record)
            detections = self._segment(usable_images)
            object_rows: list[dict[str, Any]] = []
            object_masks: list[np.ndarray] = []
            object_images: list[np.ndarray] = []
            for image, record, dets in zip(usable_images, usable_meta, detections):
                built_rows, built_masks = build_objects(
                    image, dets, keyframe_id=str(record.keyframe_id), clip_id=cid,
                    shot_no=int(record.shot_no), k=int(record.k), t_s=float(record.t_s),
                    max_objects=self.max_objects,
                )
                object_rows.extend(built_rows)
                object_masks.extend(built_masks)
                object_images.extend([image] * len(built_masks))
            self._write_part(cid, object_rows, object_masks, object_images)
            _append_marker(self.root / "processed_clips.txt", cid)
            summary["clips"] += 1
            summary["keyframes"] += len(usable_images)
            summary["objects"] += len(object_rows)
        return summary


def _cast_objects(frame: pd.DataFrame) -> pd.DataFrame:
    for column in OBJECT_COLUMNS:
        if column not in frame:
            frame[column] = pd.Series(index=frame.index, dtype="object")
    frame = frame[OBJECT_COLUMNS].copy()
    casts = {
        "object_id": "int64", "shot_no": "int32", "k": "int8", "t_s": "float64",
        "frame_w": "int32", "frame_h": "int32", "cx": "float32", "cy": "float32",
        "area": "float32", "area_frac": "float32", "bbox_x0": "int32", "bbox_y0": "int32",
        "bbox_x1": "int32", "bbox_y1": "int32", "angle": "float32", "eccentricity": "float32",
        "n_components": "int16", "solidity": "float32", "score": "float32",
        "mask_offset": "int64", "mask_nbytes": "int32",
    }
    for column, dtype in casts.items():
        frame[column] = frame[column].fillna(0).astype(dtype)
    frame["keyframe_id"] = frame["keyframe_id"].astype(str)
    frame["clip_id"] = frame["clip_id"].astype(str)
    return frame


def consolidate(root) -> pd.DataFrame:
    """Concatenate committed clip parts and validate the resulting index."""
    root = Path(root)
    marker = root / "processed_clips.txt"
    ids: list[str] = []
    seen: set[str] = set()
    if marker.exists():
        for cid in _clip_list(marker):
            if cid not in seen:
                ids.append(cid)
                seen.add(cid)
    all_rows: list[dict[str, Any]] = []
    all_fp: list[np.ndarray] = []
    all_emb: list[np.ndarray] = []
    blob = bytearray()
    segments: set[str] = set()
    for cid in ids:
        part_dir = root / "parts" / cid
        if not (part_dir / "objects.parquet").is_file():
            continue
        part = pd.read_parquet(part_dir / "objects.parquet")
        data = (part_dir / "masks.bin").read_bytes()
        fp = np.asarray(np.load(part_dir / "shape_fp.npy"))
        emb = np.asarray(np.load(part_dir / "clip_emb.npy"))
        if len(part) != len(fp) or len(part) != len(emb):
            raise ValueError(f"part {cid} array lengths disagree")
        local_offset = 0
        for _, record in part.iterrows():
            nbytes = int(record["mask_nbytes"])
            if local_offset < 0 or nbytes < 0 or local_offset + nbytes > len(data):
                raise ValueError(f"part {cid} has invalid mask offsets")
            row = record.to_dict()
            row["object_id"] = len(all_rows)
            row["mask_offset"] = len(blob)
            blob.extend(data[local_offset:local_offset + nbytes])
            local_offset += nbytes
            segments.add(str(row.get("segmenter", "unknown")))
            all_rows.append(row)
        all_fp.append(fp.astype(np.float16, copy=False))
        all_emb.append(emb.astype(np.float16, copy=False))
    output = _cast_objects(pd.DataFrame(all_rows))
    fingerprints = np.concatenate(all_fp, axis=0) if all_fp else np.empty((0, FINGERPRINT_DIM), dtype=np.float16)
    if fingerprints.ndim != 2 or fingerprints.shape != (len(output), FINGERPRINT_DIM):
        raise ValueError("shape fingerprint array has the wrong shape")
    if all_emb:
        dimensions = {array.shape[1] for array in all_emb if array.ndim == 2 and array.shape[0]}
        if len(dimensions) > 1 or any(array.ndim != 2 for array in all_emb):
            raise ValueError("CLIP arrays have inconsistent dimensions")
        clip_dim = next(iter(dimensions), 0)
        normalized = [array.reshape(0, clip_dim) if array.shape[0] == 0 else array for array in all_emb]
        clip_embeddings = np.concatenate(normalized, axis=0).astype(np.float16, copy=False)
    else:
        clip_dim = 0
        clip_embeddings = np.empty((0, 0), dtype=np.float16)
    if len(clip_embeddings) != len(output):
        raise ValueError("CLIP array length disagrees with objects")
    for record in output.itertuples(index=False):
        start = int(record.mask_offset)
        encoded = bytes(blob[start:start + int(record.mask_nbytes)])
        decoded = rle_decode(encoded)
        if decoded.shape != (int(record.frame_h), int(record.frame_w)):
            raise ValueError(f"mask shape mismatch for object {record.object_id}")

    temp = _temp_dir(root, ".consolidate-")
    try:
        index = temp / "index"
        index.mkdir()
        output.to_parquet(temp / "objects.parquet", index=False)
        (index / "masks.bin").write_bytes(blob)
        np.save(index / "shape_fp.npy", fingerprints)
        np.save(index / "clip_emb.npy", clip_embeddings)
        meta = {
            "n_objects": int(len(output)), "segmenter": ",".join(sorted(segments)),
            "segmenter_params": {}, "clip_model": "clip", "clip_dim": int(clip_dim),
            "fingerprint_dim": int(FINGERPRINT_DIM), "fingerprint_version": 1,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "clips": ids,
        }
        (index / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
        root.mkdir(parents=True, exist_ok=True)
        os.replace(temp / "objects.parquet", root / "objects.parquet")
        final_index = root / "index"
        final_index.mkdir(exist_ok=True)
        for name in ("masks.bin", "shape_fp.npy", "clip_emb.npy", "meta.json"):
            os.replace(index / name, final_index / name)
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    return output


def _outline(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
    return out


def neighbours(root, n: int = 12, k: int = 8, seed: int = 0, out=None) -> list[Path]:
    """Write contact sheets for random objects and their fingerprint neighbours."""
    root = Path(root)
    out_dir = Path(out) if out is not None else root / "neighbours"
    out_dir.mkdir(parents=True, exist_ok=True)
    objects = pd.read_parquet(root / "objects.parquet")
    fingerprints = np.asarray(np.load(root / "index" / "shape_fp.npy"), dtype=np.float32)
    if len(objects) == 0:
        return []
    rng = np.random.default_rng(seed)
    queries = rng.choice(len(objects), size=min(int(n), len(objects)), replace=False)
    blob = (root / "index" / "masks.bin").read_bytes()
    written: list[Path] = []
    for query in map(int, queries):
        q = objects.iloc[query]
        scores = fingerprints @ fingerprints[query]
        order = np.argsort(-scores, kind="stable")
        candidates = [int(i) for i in order if int(i) != query and not (str(objects.iloc[int(i)].clip_id) == str(q.clip_id) and int(objects.iloc[int(i)].shot_no) == int(q.shot_no))][: int(k)]
        panels: list[np.ndarray] = []
        for idx in [query] + candidates:
            row = objects.iloc[idx]
            key = str(row.keyframe_id).split(":")
            if len(key) < 3:
                continue
            path = root / "keyframes" / key[0] / f"{int(key[1]):03d}_{int(key[2])}.jpg"
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            start = int(row.mask_offset); end = start + int(row.mask_nbytes)
            mask = rle_decode(blob[start:end])
            image = _outline(image, mask)
            if idx != query:
                ys, xs = np.nonzero(mask)
                if len(xs):
                    pad = max(4, int(0.1 * max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))
                    x0, x1 = max(0, int(xs.min()) - pad), min(image.shape[1], int(xs.max()) + 1 + pad)
                    y0, y1 = max(0, int(ys.min()) - pad), min(image.shape[0], int(ys.max()) + 1 + pad)
                    image = image[y0:y1, x0:x1]
            cv2.putText(image, f"{idx} {scores[idx]:.3f}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            panels.append(image)
        if not panels:
            continue
        target_h = min(240, max(panel.shape[0] for panel in panels))
        resized = [cv2.resize(panel, (max(1, round(panel.shape[1] * target_h / panel.shape[0])), target_h), interpolation=cv2.INTER_AREA) for panel in panels]
        sheet = cv2.hconcat(resized)
        path = out_dir / f"object_{query}.jpg"
        cv2.imwrite(str(path), sheet)
        written.append(path)
    return written


def _summary(root: Path) -> dict[str, Any]:
    objects_path = root / "objects.parquet"
    if not objects_path.is_file():
        return {"objects": 0, "clips": 0, "keyframes": 0, "skipped_keyframes": 0, "bytes": {}}
    frame = pd.read_parquet(objects_path)
    keyframes = int(frame["keyframe_id"].nunique()) if len(frame) else 0
    clips = int(frame["clip_id"].nunique()) if len(frame) else 0
    bytes_map = {}
    for path in [objects_path, root / "index" / "masks.bin", root / "index" / "shape_fp.npy", root / "index" / "clip_emb.npy", root / "index" / "meta.json"]:
        if path.is_file():
            bytes_map[str(path.relative_to(root))] = path.stat().st_size
    return {"objects": len(frame), "clips": clips, "keyframes": keyframes, "skipped_keyframes": 0, "objects_per_keyframe": (len(frame) / keyframes if keyframes else 0.0), "bytes": bytes_map}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m matchcut_retriever.library")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--segmenter", choices=("both", "mask2former", "sam2_auto"), default="both")
    build.add_argument("--device", default="cuda:0")
    build.add_argument("--max-objects", type=int, default=40)
    build.add_argument("--limit-clips", type=int)
    build.add_argument("--shard")
    build.add_argument("--no-consolidate", action="store_true")
    build.add_argument("--config")
    con = sub.add_parser("consolidate"); con.add_argument("--config")
    stats = sub.add_parser("stats"); stats.add_argument("--config")
    neigh = sub.add_parser("neighbours"); neigh.add_argument("--n", type=int, default=12); neigh.add_argument("--k", type=int, default=8); neigh.add_argument("--seed", type=int, default=0); neigh.add_argument("--out"); neigh.add_argument("--config")
    args = parser.parse_args(argv)
    paths = load_paths(getattr(args, "config", None))
    root = paths.library_root
    if args.command == "consolidate":
        consolidate(root)
    elif args.command == "stats":
        print(json.dumps(_summary(root), indent=2, sort_keys=True))
    elif args.command == "neighbours":
        neighbours(root, args.n, args.k, args.seed, args.out)
    else:
        clips = _clip_list(root / "clips.txt")
        processed = set(_clip_list(root / "processed_clips.txt")) if (root / "processed_clips.txt").is_file() else set()
        clips = [cid for cid in clips if cid not in processed]
        if args.shard:
            try:
                shard_i, shard_n = (int(x) for x in args.shard.split("/", 1))
                if shard_n <= 0 or shard_i < 0 or shard_i >= shard_n:
                    raise ValueError
            except ValueError as exc:
                raise SystemExit("--shard must be I/N with 0 <= I < N") from exc
            clips = [cid for pos, cid in enumerate(clips) if pos % shard_n == shard_i]
        if args.limit_clips is not None:
            clips = clips[: max(0, args.limit_clips)]
        segmenters = []
        if args.segmenter in ("both", "mask2former"):
            segmenters.append(PanopticSegmenter(device=args.device, cache_dir=paths.hf_cache))
        if args.segmenter in ("both", "sam2_auto"):
            segmenters.append(SamAutoSegmenter(device=args.device, cache_dir=paths.hf_cache))
        embedder = ClipEmbedder(device=args.device, cache_dir=paths.hf_cache)
        summary = LibraryBuilder(root, segmenters, embedder, args.max_objects).process_clips(clips)
        if not args.no_consolidate:
            consolidate(root)
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

