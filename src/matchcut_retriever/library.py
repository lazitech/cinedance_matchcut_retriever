"""Build and maintain the object library used by the ranking stage."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
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
from .shape import (
    FINGERPRINT_DIM,
    FINGERPRINT_VERSION,
    MAX_AREA_FRAC,
    MIN_AREA_FRAC,
    object_geometry,
    shape_fingerprint,
)

LOG = logging.getLogger(__name__)

LEGACY_FINGERPRINT_VERSION = "r4"


def _fingerprint_version(metadata: dict[str, Any]) -> str:
    """Return a normalized fingerprint version, including legacy metadata."""

    value = metadata.get("fingerprint_version", LEGACY_FINGERPRINT_VERSION)
    return LEGACY_FINGERPRINT_VERSION if value is None else str(value)


def _version_error(subject: str, version: str) -> ValueError:
    return ValueError(
        f"{subject} uses fingerprint version {version!r}, but the current version is "
        f"{FINGERPRINT_VERSION!r}; run `python -m matchcut_retriever.library refingerprint`"
    )

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
    kept = filter_detections(cleaned, (h, w), MIN_AREA_FRAC, MAX_AREA_FRAC, max_objects)
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


def _model_id(model: Any) -> str:
    return str(getattr(model, "model_id", getattr(model, "name", model.__class__.__name__)))


def _model_params(model: Any) -> dict[str, Any]:
    params = getattr(model, "params", None)
    if isinstance(params, dict):
        return dict(params)
    keys = ("device", "batch_size", "points_per_side", "pred_iou_thresh", "stability_thresh", "nms_iou", "points_per_batch")
    return {key: getattr(model, key) for key in keys if hasattr(model, key)}


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
        marker = self.root / "objects_processed.txt"
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

    def _metadata(self) -> dict[str, Any]:
        return {
            "clip_model": _model_id(self.embedder),
            "segmenter_ids": [_model_id(segmenter) for segmenter in self.segmenters],
            "segmenter_params": {_model_id(segmenter): _model_params(segmenter) for segmenter in self.segmenters},
            "fingerprint_version": FINGERPRINT_VERSION,
        }

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
            (temp / "meta.json").write_text(json.dumps(self._metadata(), sort_keys=True))
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
        clip_ids = [str(clip_id) for clip_id in clip_ids]
        done = self._processed()
        started = time.monotonic()
        summary = {"clips": 0, "keyframes": 0, "skipped_keyframes": 0, "objects": 0}
        for clip_id in clip_ids:
            cid = str(clip_id)
            if cid in done:
                continue
            if (parts / cid / "objects.parquet").is_file():
                # Recover a complete part if a process crashed after its
                # atomic rename but before updating the marker file.
                _append_marker(self.root / "objects_processed.txt", cid)
                done.add(cid)
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
            _append_marker(self.root / "objects_processed.txt", cid)
            summary["clips"] += 1
            summary["keyframes"] += len(usable_images)
            summary["objects"] += len(object_rows)
            elapsed = max(time.monotonic() - started, 1e-9)
            LOG.info(
                "library clips %d/%d (%.2f clips/s): %s objects=%d objects/keyframe=%.2f skipped_keyframes=%d",
                summary["clips"], len(clip_ids), summary["clips"] / elapsed, cid,
                len(object_rows), len(object_rows) / len(usable_images) if usable_images else 0.0,
                summary["skipped_keyframes"],
            )
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


def consolidate(root) -> dict[str, Any]:
    """Publish a consolidated index with bounded memory.

    Builds and queries must not run concurrently.  An interrupted publish is
    detected by :func:`verify_library` and repaired by rerunning consolidate.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    marker = root / "objects_processed.txt"
    ids: list[str] = []
    seen: set[str] = set()
    if marker.exists():
        for cid in _clip_list(marker):
            if cid not in seen:
                ids.append(cid)
                seen.add(cid)

    part_root = root / "parts"
    # Hidden directories are in-flight or interrupted temporaries (builder,
    # refingerprint); clip ids never start with a dot.
    orphan_parts = {
        path.name for path in part_root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and (path / "objects.parquet").is_file() and path.name not in seen
    } if part_root.is_dir() else set()
    if orphan_parts:
        clip_order = _clip_list(root / "clips.txt") if (root / "clips.txt").is_file() else []
        ordered_orphans = [cid for cid in clip_order if cid in orphan_parts]
        ordered_orphans.extend(sorted(orphan_parts.difference(ordered_orphans)))
        ids.extend(ordered_orphans)
        seen.update(ordered_orphans)

    # Pass 1 only reads parquet/numpy headers and sidecar metadata.
    descriptors: list[dict[str, Any]] = []
    total_objects = 0
    clip_dimensions: set[int] = set()
    segments: set[str] = set()
    segmenter_params: dict[str, Any] = {}
    clip_models: set[str] = set()
    fingerprint_versions: set[str] = set()
    for cid in ids:
        part_dir = root / "parts" / cid
        parquet_path = part_dir / "objects.parquet"
        if not parquet_path.is_file():
            continue
        row_count = int(pq.ParquetFile(parquet_path).metadata.num_rows)
        fp_shape = tuple(np.load(part_dir / "shape_fp.npy", mmap_mode="r").shape)
        emb_shape = tuple(np.load(part_dir / "clip_emb.npy", mmap_mode="r").shape)
        if fp_shape != (row_count, FINGERPRINT_DIM):
            raise ValueError(f"part {cid} fingerprint shape disagrees")
        if len(emb_shape) != 2 or emb_shape[0] != row_count:
            raise ValueError(f"part {cid} embedding shape disagrees")
        if row_count:
            clip_dimensions.add(int(emb_shape[1]))
        descriptors.append({
            "cid": cid,
            "dir": part_dir,
            "rows": row_count,
            "mask_bytes": int((part_dir / "masks.bin").stat().st_size),
            "global_start": total_objects,
        })
        total_objects += row_count
        sidecar = part_dir / "meta.json"
        if sidecar.is_file():
            info = json.loads(sidecar.read_text())
            clip_models.add(str(info.get("clip_model", "clip")))
            segments.update(str(value) for value in info.get("segmenter_ids", []))
            segmenter_params.update(info.get("segmenter_params", {}))
        else:
            info = {}
            columns = pd.read_parquet(parquet_path, columns=["segmenter"])
            segments.update(str(value) for value in columns["segmenter"].dropna().unique())
        part_version = _fingerprint_version(info)
        fingerprint_versions.add(part_version)
        if part_version != FINGERPRINT_VERSION:
            raise _version_error(f"part {cid}", part_version)
    if len(fingerprint_versions) > 1:
        versions = ", ".join(sorted(fingerprint_versions))
        raise ValueError(
            f"parts have mixed fingerprint versions ({versions}); run `python -m "
            "matchcut_retriever.library refingerprint`"
        )
    if len(clip_dimensions) > 1:
        raise ValueError("CLIP arrays have inconsistent dimensions")
    clip_dim = next(iter(clip_dimensions), 0)
    sample_rng = np.random.default_rng(0)
    sample_indices = set(
        map(int, sample_rng.choice(total_objects, size=min(1000, total_objects), replace=False))
    ) if total_objects else set()

    temp = _temp_dir(root, ".consolidate-")
    writer = None
    shape_mem = clip_mem = None
    orphan_markers = set(orphan_parts)
    try:
        index = temp / "index"
        index.mkdir()
        shape_path = index / "shape_fp.npy"
        clip_path = index / "clip_emb.npy"
        shape_mem = np.lib.format.open_memmap(
            shape_path, mode="w+", dtype=np.float16, shape=(total_objects, FINGERPRINT_DIM)
        )
        clip_mem = np.lib.format.open_memmap(
            clip_path, mode="w+", dtype=np.float16, shape=(total_objects, clip_dim)
        )
        schema = pa.schema([
            pa.field("object_id", pa.int64()),
            pa.field("keyframe_id", pa.string()),
            pa.field("clip_id", pa.string()),
            pa.field("shot_no", pa.int32()),
            pa.field("k", pa.int8()),
            pa.field("t_s", pa.float64()),
            pa.field("frame_w", pa.int32()),
            pa.field("frame_h", pa.int32()),
            pa.field("cx", pa.float32()),
            pa.field("cy", pa.float32()),
            pa.field("area", pa.float32()),
            pa.field("area_frac", pa.float32()),
            pa.field("bbox_x0", pa.int32()),
            pa.field("bbox_y0", pa.int32()),
            pa.field("bbox_x1", pa.int32()),
            pa.field("bbox_y1", pa.int32()),
            pa.field("angle", pa.float32()),
            pa.field("eccentricity", pa.float32()),
            pa.field("n_components", pa.int16()),
            pa.field("solidity", pa.float32()),
            pa.field("score", pa.float32()),
            pa.field("label", pa.string()),
            pa.field("segmenter", pa.string()),
            pa.field("mask_offset", pa.int64()),
            pa.field("mask_nbytes", pa.int32()),
        ])
        writer = pq.ParquetWriter(temp / "objects.parquet", schema=schema)
        mask_output = index / "masks.bin"
        sample_rows: list[tuple[int, int, int, int, int]] = []
        mask_base = 0
        for descriptor in descriptors:
            cid = descriptor["cid"]
            part_dir = descriptor["dir"]
            part_path = part_dir / "objects.parquet"
            part = pd.read_parquet(part_path)
            nbytes = (
                part["mask_nbytes"].to_numpy(dtype=np.int64, copy=False)
                if len(part) else np.empty(0, dtype=np.int64)
            )
            if np.any(nbytes < 0):
                raise ValueError(f"part {cid} has negative mask sizes")
            local_offsets = np.zeros(len(part), dtype=np.int64)
            if len(part) > 1:
                local_offsets[1:] = np.cumsum(nbytes[:-1], dtype=np.int64)
            if int(nbytes.sum(dtype=np.int64)) != descriptor["mask_bytes"]:
                raise ValueError(f"part {cid} mask lengths disagree with masks.bin")
            part.insert(0, "object_id", np.arange(
                descriptor["global_start"], descriptor["global_start"] + len(part), dtype=np.int64
            ))
            part["mask_offset"] = mask_base + local_offsets
            part = _cast_objects(part)
            if len(part):
                shape_values = np.asarray(np.load(part_dir / "shape_fp.npy", mmap_mode="r"))
                clip_values = np.asarray(np.load(part_dir / "clip_emb.npy", mmap_mode="r"))
                shape_mem[descriptor["global_start"]:descriptor["global_start"] + len(part)] = shape_values
                clip_mem[descriptor["global_start"]:descriptor["global_start"] + len(part)] = clip_values
                local_indices = np.flatnonzero(
                    np.isin(
                        np.arange(descriptor["global_start"], descriptor["global_start"] + len(part), dtype=np.int64),
                        list(sample_indices),
                    )
                )
                for local_index in local_indices:
                    record = part.iloc[int(local_index)]
                    sample_rows.append((
                        int(record.object_id), int(record.mask_offset), int(record.mask_nbytes),
                        int(record.frame_h), int(record.frame_w),
                    ))
            table = pa.Table.from_pandas(part[OBJECT_COLUMNS], schema=schema, preserve_index=False)
            writer.write_table(table)
            with (part_dir / "masks.bin").open("rb") as source, mask_output.open("ab") as destination:
                while True:
                    chunk = source.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)
            mask_base += descriptor["mask_bytes"]
        shape_mem.flush()
        clip_mem.flush()
        del shape_mem, clip_mem
        shape_mem = clip_mem = None
        writer.close()
        writer = None

        with mask_output.open("rb") as source:
            for _, offset, nbytes, frame_h, frame_w in sample_rows:
                source.seek(offset)
                decoded = rle_decode(source.read(nbytes))
                if decoded.shape != (frame_h, frame_w):
                    raise ValueError("sampled mask shape disagrees with frame dimensions")

        generation = uuid.uuid4().hex
        data_paths = {
            "objects.parquet": temp / "objects.parquet",
            "index/masks.bin": index / "masks.bin",
            "index/shape_fp.npy": index / "shape_fp.npy",
            "index/clip_emb.npy": index / "clip_emb.npy",
        }
        byte_sizes = {name: int(path.stat().st_size) for name, path in data_paths.items()}
        meta = {
            "generation": generation,
            "n_objects": total_objects,
            "segmenter": ",".join(sorted(segments)),
            "segmenter_params": segmenter_params,
            "clip_model": ",".join(sorted(clip_models)) if clip_models else "clip",
            "clip_dim": int(clip_dim),
            "fingerprint_dim": int(FINGERPRINT_DIM),
            "fingerprint_version": FINGERPRINT_VERSION,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "clips": ids,
            "bytes": byte_sizes,
        }
        (index / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
        root.mkdir(parents=True, exist_ok=True)
        # Data files are published first.  meta.json is the final consistency
        # marker and is replaced atomically after all data files are visible.
        os.replace(temp / "objects.parquet", root / "objects.parquet")
        final_index = root / "index"
        final_index.mkdir(exist_ok=True)
        for name in ("masks.bin", "shape_fp.npy", "clip_emb.npy"):
            os.replace(index / name, final_index / name)
        os.replace(index / "meta.json", final_index / "meta.json")
        for cid in orphan_markers:
            _append_marker(root / "objects_processed.txt", cid)
    finally:
        if writer is not None:
            writer.close()
        if shape_mem is not None:
            shape_mem.flush()
        if clip_mem is not None:
            clip_mem.flush()
        shutil.rmtree(temp, ignore_errors=True)
    verify_library(root)
    return {
        "n_objects": total_objects,
        "n_clips": len(ids),
        "clip_dim": int(clip_dim),
        "fingerprint_version": FINGERPRINT_VERSION,
        "generation": generation,
        "bytes": byte_sizes,
    }


def _refingerprint_part_worker(part_path: str) -> dict[str, Any]:
    """Recompute one part while keeping all non-fingerprint files untouched."""

    part_dir = Path(part_path)
    sidecar = part_dir / "meta.json"
    metadata = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    version = _fingerprint_version(metadata)
    if version == FINGERPRINT_VERSION:
        return {"part": part_dir.name, "status": "skipped", "objects": 0}

    table = pd.read_parquet(part_dir / "objects.parquet", columns=["mask_nbytes"])
    lengths = table["mask_nbytes"].to_numpy(dtype=np.int64, copy=False) if len(table) else np.empty(0, dtype=np.int64)
    blob = (part_dir / "masks.bin").read_bytes()
    fingerprints = np.empty((len(lengths), FINGERPRINT_DIM), dtype=np.float16)
    offset = 0
    for index, length in enumerate(lengths):
        length = int(length)
        if length < 0 or offset + length > len(blob):
            raise ValueError(f"part {part_dir.name} has invalid mask lengths")
        mask = rle_decode(blob[offset:offset + length])
        fingerprints[index] = shape_fingerprint(mask).astype(np.float16)
        offset += length
    if offset != len(blob):
        raise ValueError(f"part {part_dir.name} mask lengths disagree with masks.bin")

    temp = _temp_dir(part_dir.parent, f".{part_dir.name}.refingerprint-")
    backup = part_dir.parent / f".{part_dir.name}.refingerprint-backup-{uuid.uuid4().hex}"
    try:
        # Hard links keep unchanged files byte-identical without copying a
        # potentially large part into the temporary directory.
        for child in part_dir.iterdir():
            if child.name in {"shape_fp.npy", "meta.json"}:
                continue
            target = temp / child.name
            if child.is_file():
                os.link(child, target)
            elif child.is_dir():
                shutil.copytree(child, target)
        np.save(temp / "shape_fp.npy", fingerprints)
        metadata["fingerprint_version"] = FINGERPRINT_VERSION
        (temp / "meta.json").write_text(json.dumps(metadata, sort_keys=True))

        # The directory swap makes the part appear wholly old or wholly new
        # to readers.  Restore the original if the second rename fails.
        os.replace(part_dir, backup)
        try:
            os.replace(temp, part_dir)
        except Exception:
            os.replace(backup, part_dir)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        if backup.exists() and not part_dir.exists():
            os.replace(backup, part_dir)
        raise
    return {"part": part_dir.name, "status": "rewritten", "objects": len(lengths)}


def refingerprint(root, workers: int = 1) -> dict[str, Any]:
    """Recompute stale part fingerprints from their stored masks and publish the index."""

    root = Path(root)
    workers = max(1, int(workers))
    part_root = root / "parts"
    if part_root.is_dir():
        # Complete or roll back a directory swap interrupted by a process
        # crash before discovering parts for this run.
        for backup in part_root.glob(".*.refingerprint-backup-*"):
            name = backup.name.split(".refingerprint-backup-", 1)[0].lstrip(".")
            final = part_root / name
            if final.exists():
                shutil.rmtree(backup, ignore_errors=True)
            else:
                os.replace(backup, final)
        # A crash before the swap leaves a half-written temporary; the original
        # part is still in place, so the temporary is simply discarded.
        for temp in part_root.glob(".*.refingerprint-*"):
            if temp.is_dir() and ".refingerprint-backup-" not in temp.name:
                shutil.rmtree(temp, ignore_errors=True)
    part_dirs = sorted(
        (path for path in part_root.iterdir()
         if path.is_dir() and not path.name.startswith(".") and (path / "objects.parquet").is_file()),
        key=lambda path: path.name,
    ) if part_root.is_dir() else []
    LOG.info("refingerprint: %d parts, workers=%d", len(part_dirs), workers)
    results: list[dict[str, Any]] = []
    if workers > 1 and len(part_dirs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            iterator = pool.map(_refingerprint_part_worker, [str(path) for path in part_dirs])
            for result in iterator:
                results.append(result)
                LOG.info("refingerprint %s part %s (%d objects)", result["status"], result["part"], result["objects"])
    else:
        for part_dir in part_dirs:
            result = _refingerprint_part_worker(str(part_dir))
            results.append(result)
            LOG.info("refingerprint %s part %s (%d objects)", result["status"], result["part"], result["objects"])
    summary = consolidate(root)
    rewritten = sum(result["status"] == "rewritten" for result in results)
    skipped = sum(result["status"] == "skipped" for result in results)
    summary.update({"parts": len(part_dirs), "rewritten_parts": rewritten, "skipped_parts": skipped})
    LOG.info("refingerprint complete: rewritten=%d skipped=%d objects=%d", rewritten, skipped, summary["n_objects"])
    return summary


def verify_library(root) -> None:
    """Verify published data files against the final index metadata marker."""
    import pyarrow.parquet as pq

    root = Path(root)
    message = "library files are inconsistent with index/meta.json; rerun consolidate"
    try:
        meta = json.loads((root / "index" / "meta.json").read_text())
        if not {"generation", "n_objects", "clip_dim", "bytes"}.issubset(meta):
            raise ValueError("metadata keys")
        n_objects = int(meta["n_objects"])
        clip_dim = int(meta["clip_dim"])
        fingerprint_version = _fingerprint_version(meta)
        expected_bytes = dict(meta["bytes"])
        paths = {
            "objects.parquet": root / "objects.parquet",
            "index/masks.bin": root / "index" / "masks.bin",
            "index/shape_fp.npy": root / "index" / "shape_fp.npy",
            "index/clip_emb.npy": root / "index" / "clip_emb.npy",
        }
        if any(not path.is_file() for path in paths.values()):
            raise ValueError("missing data file")
        if any(int(path.stat().st_size) != int(expected_bytes[name]) for name, path in paths.items()):
            raise ValueError("byte size")
        if int(pq.ParquetFile(paths["objects.parquet"]).metadata.num_rows) != n_objects:
            raise ValueError("parquet rows")
        shape = tuple(np.load(paths["index/shape_fp.npy"], mmap_mode="r").shape)
        clip = tuple(np.load(paths["index/clip_emb.npy"], mmap_mode="r").shape)
        if shape != (n_objects, FINGERPRINT_DIM) or clip != (n_objects, clip_dim):
            raise ValueError("array shape")
    except Exception as exc:
        if isinstance(exc, RuntimeError) and str(exc) == message:
            raise
        raise RuntimeError(message) from exc
    # A consistent index built with another fingerprint is not loadable either;
    # report it with the migration hint rather than as an inconsistency.
    if fingerprint_version != FINGERPRINT_VERSION:
        raise _version_error("library index", fingerprint_version)
    LOG.info("verified library: objects=%d fingerprint_version=%s", n_objects, fingerprint_version)


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
        meta_path = root / "index" / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        return {"objects": 0, "clips": 0, "keyframes": 0, "skipped_keyframes": 0,
                "fingerprint_version": _fingerprint_version(meta), "bytes": {}}
    frame = pd.read_parquet(objects_path)
    keyframes = int(frame["keyframe_id"].nunique()) if len(frame) else 0
    clips = int(frame["clip_id"].nunique()) if len(frame) else 0
    bytes_map = {}
    for path in [objects_path, root / "index" / "masks.bin", root / "index" / "shape_fp.npy", root / "index" / "clip_emb.npy", root / "index" / "meta.json"]:
        if path.is_file():
            bytes_map[str(path.relative_to(root))] = path.stat().st_size
    meta_path = root / "index" / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    return {"objects": len(frame), "clips": clips, "keyframes": keyframes, "skipped_keyframes": 0,
            "objects_per_keyframe": (len(frame) / keyframes if keyframes else 0.0),
            "fingerprint_version": _fingerprint_version(meta), "bytes": bytes_map}


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
    ref = sub.add_parser("refingerprint"); ref.add_argument("--config"); ref.add_argument("--workers", type=int, default=1)
    stats = sub.add_parser("stats"); stats.add_argument("--config")
    neigh = sub.add_parser("neighbours"); neigh.add_argument("--n", type=int, default=12); neigh.add_argument("--k", type=int, default=8); neigh.add_argument("--seed", type=int, default=0); neigh.add_argument("--out"); neigh.add_argument("--config")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = load_paths(getattr(args, "config", None))
    root = paths.library_root
    if args.command == "consolidate":
        print(json.dumps(consolidate(root), indent=2, sort_keys=True))
    elif args.command == "refingerprint":
        if args.workers < 1:
            raise SystemExit("--workers must be >= 1")
        print(json.dumps(refingerprint(root, args.workers), indent=2, sort_keys=True))
    elif args.command == "stats":
        verify_library(root)
        print(json.dumps(_summary(root), indent=2, sort_keys=True))
    elif args.command == "neighbours":
        neighbours(root, args.n, args.k, args.seed, args.out)
    else:
        clips = _clip_list(root / "clips.txt")
        processed = set(_clip_list(root / "objects_processed.txt")) if (root / "objects_processed.txt").is_file() else set()
        if args.shard:
            try:
                shard_i, shard_n = (int(x) for x in args.shard.split("/", 1))
                if shard_n <= 0 or shard_i < 0 or shard_i >= shard_n:
                    raise ValueError
            except ValueError as exc:
                raise SystemExit("--shard must be I/N with 0 <= I < N") from exc
            # Shard positions are defined by the complete clips.txt list, so
            # already-processed clips do not shift the assignment of later
            # clips between reruns.
            clips = [cid for pos, cid in enumerate(clips) if pos % shard_n == shard_i]
        clips = [cid for cid in clips if cid not in processed]
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
            summary.update(consolidate(root))
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

