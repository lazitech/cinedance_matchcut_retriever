"""Library loading and shape/semantic ranking."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, asdict, replace
import logging
import math
import multiprocessing
import os
import threading
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from .masks import rle_decode
from .memory import disable_transparent_hugepages
from .parquet_io import read_parquet_buffered
from .library import _pca_status, pca_files
from .retrieval_config import RetrievalConfig, load_retrieval_config
from .shape import (
    FINGERPRINT_DIM,
    FINGERPRINT_VERSION,
    HardLimits,
    MatchResult,
    Similarity,
    SoftParams,
    hard_match,
    hard_prefilter,
    object_geometry,
    shape_fingerprint,
    soft_match,
)

LOGGER = logging.getLogger(__name__)


class RankedResults(list):
    """List of :class:`Result` objects with coarse/rerank timing metadata."""

    def __init__(self, values=(), timings=None, coarse_method="exact", subject_filter="off"):
        super().__init__(values)
        self.timings = timings or {}
        self.coarse_method = coarse_method
        self.subject_filter = subject_filter


@dataclass(frozen=True)
class QueryObject:
    image: np.ndarray
    mask: np.ndarray
    geometry: Any
    fingerprint: np.ndarray
    clip_vector: np.ndarray | None = None

    @classmethod
    def from_mask(cls, image, mask, clip_vector=None):
        image_arr = np.asarray(image)
        mask_arr = np.asarray(mask, dtype=bool)
        if image_arr.ndim != 3 or image_arr.shape[:2] != mask_arr.shape:
            raise ValueError("image must be HxWx3 and mask must match its first two dimensions")
        if not np.any(mask_arr):
            raise ValueError("query mask must be non-empty")
        vector = None
        if clip_vector is not None:
            vector = np.asarray(clip_vector, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vector))
            if norm == 0:
                raise ValueError("clip_vector must be non-zero")
            vector = vector / norm
        return cls(image_arr, mask_arr, object_geometry(mask_arr), shape_fingerprint(mask_arr), vector)


@dataclass(frozen=True)
class Result:
    object_id: int
    clip_id: str
    shot_no: int
    start_s: float | None
    end_s: float | None
    keyframe_id: str
    t_s: float
    bbox: tuple[int, int, int, int]
    fingerprint_cosine: float
    iou: float
    transform_cost: float
    clip_cosine: float | None
    geometric_score: float
    final_score: float
    transform: Similarity
    transform_angle_deg: float
    transform_tx_px: float
    transform_ty_px: float
    transform_tx_frac: float
    transform_ty_frac: float
    within_limits: bool
    mode: str
    subject_score: float | None = None

    @property
    def scale(self) -> float:
        return float(self.transform.scale)

    @property
    def angle_deg(self) -> float:
        return self.transform_angle_deg

    @property
    def tx_frac(self) -> float:
        """X translation as a fraction of the query frame *width*."""
        return self.transform_tx_frac

    @property
    def ty_frac(self) -> float:
        """Y translation as a fraction of the query frame *width* (not height).

        Both axes use the width so the values are on the same scale as
        ``HardLimits.max_translate_frac`` (a radius in units of the width).
        """
        return self.transform_ty_frac

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["transform"] = {
            "scale": self.transform.scale,
            "angle": self.transform_angle_deg,
            "angle_deg": self.transform_angle_deg,
            "tx": self.transform_tx_frac,
            "ty": self.transform_ty_frac,
            "tx_px": self.transform_tx_px,
            "ty_px": self.transform_ty_px,
            "tx_frac": self.transform_tx_frac,
            "ty_frac": self.transform_ty_frac,
        }
        value["bbox"] = list(self.bbox)
        return value


class Library:
    """Memory-mapped retrieval library following the shared card contract."""

    def __init__(self, root: Path, objects: pd.DataFrame, masks_bin, shape_fp, clip_emb, meta, keyframes, shots,
                 shape_pca=None, pca_basis=None, pca_status="absent", pca_status_reason=None,
                 subject_score=None, subject_status="absent", subject_status_reason=None):
        self.root = Path(root)
        self.objects = objects.reset_index(drop=True)
        self.masks_bin = masks_bin
        self.shape_fp = shape_fp
        self.clip_emb = clip_emb
        self.meta = meta
        self.keyframes = keyframes
        self.shots = shots
        self.shape_pca = shape_pca
        self.pca_basis = pca_basis
        self.pca_status = pca_status
        self.pca_status_reason = pca_status_reason
        self.subject_score = subject_score
        self.subject_status = subject_status
        self.subject_status_reason = subject_status_reason
        self._geometry_cache = {
            name: self.objects[name].to_numpy(copy=True)
            for name in ("frame_w", "frame_h", "cx", "cy", "area", "eccentricity", "angle")
            if name in self.objects
        }
        has_offsets = "mask_offset" in self.objects and "mask_nbytes" in self.objects
        self._mask_offset = self.objects.mask_offset.to_numpy(dtype=np.int64) if has_offsets else None
        self._mask_nbytes = self.objects.mask_nbytes.to_numpy(dtype=np.int64) if has_offsets else None
        self._keyframe_by_id = {str(row.keyframe_id): row for row in keyframes.itertuples(index=False)}
        self._shot_by_id = {(str(row.clip_id), int(row.shot_no)): row for row in shots.itertuples(index=False)}
        # Public aliases make per-shot/keyframe metadata convenient for callers.
        self.keyframe_lookup = self._keyframe_by_id
        self.shot_lookup = self._shot_by_id

    @classmethod
    def load(cls, root: str | Path, preload: bool = False) -> "Library":
        root = Path(root)
        # Must precede the multi-GB allocations below (see memory.py).
        disable_transparent_hugepages()
        objects = read_parquet_buffered(root / "objects.parquet")
        meta_path = root / "index" / "meta.json"
        import json
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        value = meta.get("fingerprint_version", "r4")
        fingerprint_version = "r4" if value is None else str(value)
        if fingerprint_version != FINGERPRINT_VERSION:
            raise ValueError(
                f"library fingerprint version {fingerprint_version!r} does not match current "
                f"{FINGERPRINT_VERSION!r}; run `python -m matchcut_retriever.library refingerprint`"
            )
        n = len(objects)
        if "object_id" not in objects or not np.array_equal(objects.object_id.to_numpy(), np.arange(n, dtype=np.int64)):
            raise ValueError("objects.parquet object_id must be contiguous 0..N-1")
        def mmap_npy(path, expected_dim=None):
            arr = np.load(path, mmap_mode="r")
            if len(arr) != n:
                raise ValueError(f"{path.name} length does not match objects.parquet")
            if expected_dim is not None and (arr.ndim != 2 or arr.shape[1] != expected_dim):
                raise ValueError(f"{path.name} shape does not match metadata")
            return arr
        shape_fp = mmap_npy(root / "index" / "shape_fp.npy")
        if preload:
            shape_fp = np.asarray(np.load(root / "index" / "shape_fp.npy"), dtype=np.float16)
        clip_emb = mmap_npy(root / "index" / "clip_emb.npy")
        if shape_fp.ndim != 2 or shape_fp.shape[1] != int(meta.get("fingerprint_dim", shape_fp.shape[1])):
            raise ValueError("shape fingerprint dimension does not match meta.json")
        if clip_emb.ndim != 2 or clip_emb.shape[1] != int(meta.get("clip_dim", clip_emb.shape[1])):
            raise ValueError("CLIP dimension does not match meta.json")
        if "n_objects" in meta and int(meta["n_objects"]) != n:
            raise ValueError("meta.json n_objects does not match objects.parquet")
        masks_path = root / "index" / "masks.bin"
        # np.memmap cannot map an empty file (a library whose build kept no objects).
        has_masks = masks_path.is_file() and masks_path.stat().st_size > 0
        masks_bin = np.memmap(masks_path, mode="r", dtype=np.uint8) if has_masks else np.empty(0, dtype=np.uint8)
        if preload and has_masks:
            masks_bin = np.fromfile(masks_path, dtype=np.uint8)
        if "mask_offset" in objects and "mask_nbytes" in objects:
            ends = objects.mask_offset.to_numpy(dtype=np.int64) + objects.mask_nbytes.to_numpy(dtype=np.int64)
            if len(ends) and (np.any(objects.mask_offset.to_numpy(dtype=np.int64) < 0) or np.max(ends) > len(masks_bin)):
                raise ValueError("mask offsets exceed masks.bin")
        keyframes = read_parquet_buffered(root / "keyframes.parquet")
        shots_path = root / "shots.parquet"
        shots = read_parquet_buffered(shots_path) if shots_path.is_file() else pd.DataFrame(columns=["clip_id", "shot_no", "start_s", "end_s"])
        shape_pca = basis = None
        pca_status = "absent"
        pca_reason = "metadata missing"
        pca_meta_path = root / "index" / "shape_pca.json"
        pca_state, pca_reason = _pca_status(root, meta)
        if pca_state == "valid":
            try:
                pmeta = json.loads(pca_meta_path.read_text())
                dim = int(pmeta["dim"])
                pca_path, basis_path = pca_files(root / "index", pmeta)
                shape_pca = np.asarray(np.load(pca_path), dtype=np.float32)
                basis = np.asarray(np.load(basis_path), dtype=np.float32)
                shape_pca = np.ascontiguousarray(shape_pca)
                basis = np.ascontiguousarray(basis)
                pca_status = "valid"
                pca_reason = "ok"
            except Exception as exc:
                pca_status = "stale"
                pca_reason = str(exc)
                LOGGER.warning("stale PCA index at %s: %s", root, exc)
        elif pca_state == "stale":
            pca_status, pca_reason = "stale", pca_reason
            LOGGER.warning("stale PCA index at %s: %s", root, pca_reason)
        subject_score = None
        subject_status, subject_reason = "absent", "subject index metadata missing"
        subject_meta_path = root / "index" / "subject.json"
        subject_data_path = root / "index" / "subject.npy"
        if subject_meta_path.is_file() != subject_data_path.is_file():
            subject_status, subject_reason = "stale", "subject index data and metadata are incomplete"
            LOGGER.warning("subject index unavailable at %s: %s", root, subject_reason)
        elif subject_meta_path.is_file() and subject_data_path.is_file():
            try:
                smeta = json.loads(subject_meta_path.read_text())
                if not smeta.get("generation") or not meta.get("generation"):
                    subject_status, subject_reason = "stale", "generation metadata is missing"
                elif smeta.get("generation") != meta.get("generation"):
                    subject_status, subject_reason = "stale", "generation differs from index/meta.json"
                elif int(smeta.get("n_objects", -1)) != n:
                    subject_status, subject_reason = "stale", "n_objects does not match objects.parquet"
                else:
                    arr = np.load(subject_data_path, mmap_mode="r")
                    if arr.ndim != 1 or len(arr) != n:
                        subject_status, subject_reason = "stale", "subject.npy length does not match objects.parquet"
                    else:
                        subject_score = np.asarray(arr, dtype=np.float32) if preload else arr.astype(np.float32, copy=False)
                        subject_status, subject_reason = "valid", "ok"
            except Exception as exc:
                subject_status, subject_reason = "stale", str(exc)
            if subject_status != "valid":
                LOGGER.warning("subject index unavailable at %s: %s", root, subject_reason)
        return cls(root, objects, masks_bin, shape_fp, clip_emb, meta, keyframes, shots,
                   shape_pca, basis, pca_status, pca_reason, subject_score, subject_status, subject_reason)

    def mask_bytes(self, object_id: int) -> bytes:
        """Return the object's RLE-encoded mask (decode with ``rle_decode``)."""
        if self._mask_offset is None:
            row = self.objects.iloc[int(object_id)]
            start, length = int(row.mask_offset), int(row.mask_nbytes)
        else:
            start, length = int(self._mask_offset[int(object_id)]), int(self._mask_nbytes[int(object_id)])
        return bytes(memoryview(self.masks_bin)[start:start + length])

    def mask(self, object_id: int) -> np.ndarray:
        return rle_decode(self.mask_bytes(object_id))

    def geometry_arrays(self) -> dict[str, np.ndarray]:
        return {name: self._geometry_cache[name] for name in ("frame_w", "frame_h", "cx", "cy", "area")}

    def keyframe_path(self, keyframe_id: str) -> Path:
        row = self._keyframe_by_id.get(str(keyframe_id))
        if row is None:
            raise KeyError(keyframe_id)
        return self.root / str(row.path)

    def shot(self, clip_id: str, shot_no: int):
        return self._shot_by_id.get((str(clip_id), int(shot_no)))

    def keyframe(self, keyframe_id: str):
        return self._keyframe_by_id.get(str(keyframe_id))


def _match_chunk(args):
    """Decode RLE candidate masks and match each against the query mask."""
    query_mask, blobs, mode, hard_limits, soft_params = args
    results = []
    for blob in blobs:
        candidate = rle_decode(blob)
        results.append(hard_match(query_mask, candidate, hard_limits) if mode == "hard"
                       else soft_match(query_mask, candidate, soft_params))
    return results


# A query used to create (and fork) a fresh process pool, which for a library
# preloaded into tens of GB of RAM cost seconds per query. Keep one pool of
# spawned workers per process instead; they import only the matching code.
_POOL: ProcessPoolExecutor | None = None
_POOL_WORKERS = 0
_POOL_LOCK = threading.Lock()


def _get_pool(workers: int) -> ProcessPoolExecutor:
    global _POOL, _POOL_WORKERS
    with _POOL_LOCK:
        if _POOL is None or _POOL_WORKERS != workers:
            if _POOL is not None:
                _POOL.shutdown(wait=False, cancel_futures=True)
            _POOL = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
            _POOL_WORKERS = workers
        return _POOL


def _reset_pool(pool: ProcessPoolExecutor) -> None:
    global _POOL, _POOL_WORKERS
    with _POOL_LOCK:
        if _POOL is pool:
            _POOL, _POOL_WORKERS = None, 0
    pool.shutdown(wait=False, cancel_futures=True)


def _rerank(query_mask, candidate_blobs, mode, limits, soft_params, workers):
    """Match RLE-encoded candidates in input order, in parallel when worthwhile."""
    blobs = list(candidate_blobs)
    workers = int(workers)
    if workers <= 1 or len(blobs) < 8:
        return _match_chunk((query_mask, blobs, mode, limits, soft_params))
    # A few chunks per worker balance load while pickling the query mask only
    # once per chunk; RLE blobs are a few KB each instead of full masks.
    size = max(1, math.ceil(len(blobs) / (workers * 4)))
    tasks = [(query_mask, blobs[i:i + size], mode, limits, soft_params) for i in range(0, len(blobs), size)]
    pool = _get_pool(workers)
    try:
        chunks = list(pool.map(_match_chunk, tasks))
    except BrokenProcessPool:
        LOGGER.warning("rerank worker pool broke; retrying this query in-process")
        _reset_pool(pool)
        return _match_chunk((query_mask, blobs, mode, limits, soft_params))
    return [match for chunk in chunks for match in chunk]


def _coarse_cosines(query_fp, matrix, chunk=65536, threads=None):
    """Dot every (fp16) fingerprint row with the query, chunked across threads.

    The fp16->fp32 conversion and matrix-vector product release the GIL, so
    threads scale this memory-bound scan (one thread took ~70 s over 11M rows).
    """
    query_fp = np.asarray(query_fp, dtype=np.float32)
    values = np.empty(len(matrix), dtype=np.float32)

    def run(start):
        values[start:start + chunk] = np.asarray(matrix[start:start + chunk], dtype=np.float32) @ query_fp

    starts = range(0, len(matrix), chunk)
    threads = min(16, os.cpu_count() or 1) if threads is None else int(threads)
    if threads <= 1 or len(starts) <= 1:
        for start in starts:
            run(start)
    else:
        with ThreadPoolExecutor(max_workers=min(threads, len(starts))) as pool:
            list(pool.map(run, starts))
    return values


def _top_cosine_ids(ids: np.ndarray, scores: np.ndarray, limit: int) -> np.ndarray:
    """Select top scores with deterministic descending-score/id tie order."""
    ids = np.asarray(ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    limit = max(0, min(int(limit), len(ids)))
    if limit == 0:
        return ids[:0]
    if limit < len(ids):
        part = np.argpartition(-scores, limit - 1)[:limit]
        cutoff = float(np.min(scores[part]))
        selected = np.flatnonzero(scores >= cutoff)
    else:
        selected = np.arange(len(ids))
    order = np.lexsort((ids[selected], -scores[selected]))
    return ids[selected[order[:limit]]]


def rank(library: Library, query: QueryObject, mode: str = "soft", semantic_weight: float | None = None,
         k: int | None = None, params: RetrievalConfig | HardLimits | SoftParams | None = None) -> RankedResults:
    """Rank library objects by shape, optional semantic similarity, and merge per shot."""
    mode = str(mode).lower()
    if mode not in {"soft", "hard"}:
        raise ValueError("mode must be 'soft' or 'hard'")
    if isinstance(params, dict):
        base = load_retrieval_config()
        updates = {key: value for key, value in params.items() if key in {"coarse_n", "hard_coarse_n", "workers", "k", "semantic_weight", "coarse_method", "pca_shortlist_factor", "min_subject_score"}}
        if isinstance(params.get("hard_limits"), dict):
            updates["hard_limits"] = HardLimits(**params["hard_limits"])
        if isinstance(params.get("soft_params"), dict):
            updates["soft_params"] = SoftParams(**params["soft_params"])
        if "coarse_method" in updates:
            updates["coarse_method"] = str(updates["coarse_method"]).lower()
        if "pca_shortlist_factor" in updates:
            updates["pca_shortlist_factor"] = int(updates["pca_shortlist_factor"])
        config = replace(base, **updates)
        if config.coarse_method not in {"auto", "exact", "pca"}:
            raise ValueError("coarse_method must be one of auto, exact, pca")
        if int(config.pca_shortlist_factor) < 1:
            raise ValueError("pca_shortlist_factor must be >= 1")
    else:
        config = params if isinstance(params, RetrievalConfig) else load_retrieval_config()
    limits = params if isinstance(params, HardLimits) else config.hard_limits
    soft_params = params if isinstance(params, SoftParams) else config.soft_params
    weight = config.semantic_weight if semantic_weight is None else float(semantic_weight)
    if weight != 0 and query.clip_vector is None:
        raise ValueError("semantic ranking requires query.clip_vector when semantic_weight is non-zero")
    top_k = int(k if k is not None else config.k)
    started = time.perf_counter()
    subject_threshold = float(config.min_subject_score)
    subject_filter = "off"
    if subject_threshold > 0:
        if library.subject_status == "valid" and library.subject_score is not None:
            subject_filter = "applied"
        else:
            subject_filter = f"unavailable: {library.subject_status_reason or library.subject_status}"
            LOGGER.warning("subject filter unavailable; proceeding without filtering: %s", subject_filter)
    if mode == "hard":
        pre_start = time.perf_counter()
        feasible = hard_prefilter(query.geometry, library.geometry_arrays(), limits)
        # hard_prefilter is deliberately conservative about geometry. For
        # clearly elongated objects, principal-axis rotation is also a useful
        # hard exclusion; near-circular objects skip this check because their
        # axis is arbitrary (hard_match's within_limits flag is informational).
        # Axis estimates move by several degrees under small mask changes, so
        # only differences beyond max_rot_deg + hard_axis_margin_deg exclude.
        if "angle" in library.objects and "eccentricity" in library.objects:
            q_ecc = float(query.geometry.eccentricity)
            c_ecc = library._geometry_cache["eccentricity"]
            c_angle = library._geometry_cache["angle"]
            delta = np.abs((c_angle - float(query.geometry.angle) + math.pi / 2) % math.pi - math.pi / 2)
            axis_threshold = float(config.hard_axis_ecc_threshold)
            feasible &= ~((q_ecc > axis_threshold) & (c_ecc > axis_threshold) & (delta > math.radians(limits.max_rot_deg + config.hard_axis_margin_deg) + 1e-9))
        survivor_ids = np.flatnonzero(feasible)
        prefilter_time = time.perf_counter() - pre_start
        coarse_n = config.hard_coarse_n
    else:
        survivor_ids = np.arange(len(library.objects), dtype=np.int64)
        prefilter_time = 0.0
        coarse_n = config.coarse_n
    if subject_filter == "applied":
        survivor_ids = survivor_ids[np.asarray(library.subject_score[survivor_ids]) >= subject_threshold]
    method = str(config.coarse_method).lower()
    if method not in {"auto", "exact", "pca"}:
        raise ValueError("coarse_method must be one of auto, exact, pca")
    if method == "auto":
        method = "pca" if library.pca_status == "valid" else "exact"
    if method == "pca" and library.pca_status != "valid":
        reason = library.pca_status_reason or "PCA index is absent"
        raise ValueError(f"coarse_method='pca' requires a valid PCA index ({reason})")
    pca_time = 0.0
    rescore_time = 0.0
    if method == "pca":
        pca_started = time.perf_counter()
        # Raw reduced dots estimate the exact cosine; normalizing would rank by projected cosine.
        qproj = np.asarray(query.fingerprint, dtype=np.float32) @ library.pca_basis
        pca_scores = np.asarray(library.shape_pca @ qproj, dtype=np.float32)
        shortlist_n = min(len(survivor_ids), int(config.pca_shortlist_factor) * int(coarse_n))
        shortlist_ids = _top_cosine_ids(survivor_ids, pca_scores[survivor_ids], shortlist_n)
        pca_time = time.perf_counter() - pca_started
        rescore_started = time.perf_counter()
        # Read the memmap in ascending object-id order for locality, then restore score order.
        exact_ids = np.sort(shortlist_ids)
        exact_scores = _coarse_cosines(query.fingerprint, library.shape_fp[exact_ids])
        fp_cos = np.zeros(len(library.objects), dtype=np.float32)
        fp_cos[exact_ids] = exact_scores
        candidate_ids = _top_cosine_ids(exact_ids, exact_scores, int(coarse_n))
        rescore_time = time.perf_counter() - rescore_started
        coarse_time = pca_time + rescore_time
    else:
        exact_started = time.perf_counter()
        all_fp_cos = _coarse_cosines(query.fingerprint, library.shape_fp)
        candidate_ids = _top_cosine_ids(survivor_ids, all_fp_cos[survivor_ids], int(coarse_n))
        fp_cos = all_fp_cos
        coarse_time = time.perf_counter() - exact_started
    rerank_start = time.perf_counter()
    candidate_blobs = [library.mask_bytes(int(object_id)) for object_id in candidate_ids]
    matches = _rerank(query.mask, candidate_blobs, mode, limits, soft_params, config.workers)
    rerank_time = time.perf_counter() - rerank_start
    rows: list[Result] = []
    query_w = query.mask.shape[1]
    # One vectorized row selection instead of a per-candidate iloc lookup.
    candidate_rows = library.objects.iloc[np.asarray(candidate_ids, dtype=np.int64)].itertuples(index=False)
    for object_id, match, row in zip(candidate_ids, matches, candidate_rows):
        clip_id, shot_no = str(row.clip_id), int(row.shot_no)
        clip_cos = None
        if query.clip_vector is not None:
            clip_cos = float(np.asarray(query.clip_vector, dtype=np.float32) @ np.asarray(library.clip_emb[int(object_id)], dtype=np.float32))
        geometric = float(match.score if mode == "soft" else match.iou)
        final = geometric + weight * clip_cos if clip_cos is not None else geometric
        transform = match.transform
        shot = library.shot(clip_id, shot_no)
        start_s = float(getattr(shot, "start_s", np.nan)) if shot is not None else None
        end_s = float(getattr(shot, "end_s", np.nan)) if shot is not None else None
        if start_s is not None and not np.isfinite(start_s): start_s = None
        if end_s is not None and not np.isfinite(end_s): end_s = None
        rows.append(Result(
            object_id=int(object_id), clip_id=clip_id, shot_no=shot_no,
            start_s=start_s, end_s=end_s, keyframe_id=str(row.keyframe_id), t_s=float(row.t_s),
            bbox=tuple(int(v) for v in (row.bbox_x0, row.bbox_y0, row.bbox_x1, row.bbox_y1)),
            fingerprint_cosine=float(fp_cos[int(object_id)]), iou=float(match.iou), transform_cost=float(match.cost),
            clip_cosine=clip_cos, geometric_score=geometric, final_score=float(final), transform=transform,
            transform_angle_deg=float(np.degrees(transform.angle)), transform_tx_px=float(transform.tx),
            transform_ty_px=float(transform.ty), transform_tx_frac=float(transform.tx / query_w),
            transform_ty_frac=float(transform.ty / query_w), within_limits=bool(match.within_limits), mode=mode,
            subject_score=(float(library.subject_score[int(object_id)]) if library.subject_score is not None else None),
        ))
    rows.sort(key=lambda item: (-item.final_score, -item.geometric_score, item.object_id))
    merged: dict[tuple[str, int], Result] = {}
    for item in rows:
        key = (item.clip_id, item.shot_no)
        if key not in merged:
            merged[key] = item
    results = sorted(merged.values(), key=lambda item: (-item.final_score, -item.geometric_score, item.object_id))[:top_k]
    timings = {"coarse_search_s": float(coarse_time), "prefilter_s": float(prefilter_time), "rerank_s": float(rerank_time), "pca_s": float(pca_time), "rescore_s": float(rescore_time), "total_s": float(time.perf_counter() - started)}
    LOGGER.info("retrieval timings: %s", timings)
    return RankedResults(results, timings, method, subject_filter)


__all__ = ["Library", "QueryObject", "Result", "RankedResults", "rank"]
