"""Dependency-free segmentation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """A single segmentation result in input-image coordinates."""

    mask: np.ndarray
    score: float
    label: str | None = None
    source: str = "unknown"

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.mask, dtype=bool)
        if self.mask.ndim != 2:
            raise ValueError("Detection.mask must be a 2-D array")
        self.score = float(self.score)
        if self.label is not None:
            self.label = str(self.label)
        self.source = str(self.source)


def filter_detections(
    dets: list[Detection],
    frame_hw: tuple[int, int],
    min_area_frac: float = 0.005,
    max_area_frac: float = 0.6,
    max_objects: int | None = None,
    min_score: float = 0.0,
) -> list[Detection]:
    """Filter masks by score and frame-relative area, largest first."""

    h, w = (int(frame_hw[0]), int(frame_hw[1]))
    if h <= 0 or w <= 0:
        raise ValueError("frame_hw must contain positive dimensions")
    if min_area_frac < 0 or max_area_frac < min_area_frac:
        raise ValueError("invalid area fraction bounds")
    frame_area = h * w
    kept: list[tuple[int, Detection]] = []
    for order, det in enumerate(dets):
        mask = np.asarray(det.mask, dtype=bool)
        if mask.shape != (h, w):
            raise ValueError(f"detection mask has shape {mask.shape}, expected {(h, w)}")
        area = int(mask.sum())
        frac = area / frame_area
        if area and min_area_frac <= frac <= max_area_frac and float(det.score) >= min_score:
            kept.append((order, det))
    kept.sort(key=lambda item: (-int(item[1].mask.sum()), item[0]))
    if max_objects is not None:
        if max_objects < 0:
            raise ValueError("max_objects must be non-negative or None")
        kept = kept[: int(max_objects)]
    return [det for _, det in kept]


def point_grid(points_per_side: int) -> np.ndarray:
    """Return SAM-style grid points as normalized ``(x, y)`` coordinates."""

    n = int(points_per_side)
    if n <= 0:
        raise ValueError("points_per_side must be positive")
    centers = (np.arange(n, dtype=np.float32) + 0.5) / n
    xx, yy = np.meshgrid(centers, centers, indexing="xy")
    return np.stack((xx.ravel(), yy.ravel()), axis=1)


def stability_score(
    logits: np.ndarray,
    threshold: float = 0.0,
    offset: float = 1.0,
) -> np.ndarray:
    """Compute SAM's stability score over the leading mask dimensions."""

    values = np.asarray(logits)
    if values.ndim < 2:
        raise ValueError("logits must have at least two spatial dimensions")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    high = values > (threshold + offset)
    low = values > (threshold - offset)
    intersection = np.logical_and(high, low).sum(axis=(-2, -1), dtype=np.float64)
    union = np.logical_or(high, low).sum(axis=(-2, -1), dtype=np.float64)
    return np.divide(intersection, union, out=np.ones_like(intersection, dtype=np.float64), where=union != 0)


def mask_nms(masks: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy mask NMS; returned indices are in descending score order."""

    masks_arr = np.asarray(masks, dtype=bool)
    scores_arr = np.asarray(scores, dtype=float).reshape(-1)
    if masks_arr.ndim != 3:
        raise ValueError("masks must have shape (N, H, W)")
    if len(masks_arr) != len(scores_arr):
        raise ValueError("masks and scores must have the same length")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    order = np.argsort(-scores_arr, kind="stable")
    areas = masks_arr.reshape(len(masks_arr), -1).sum(axis=1).astype(np.float64)
    kept: list[int] = []
    for idx in order:
        if all(
            (np.logical_and(masks_arr[idx], masks_arr[other]).sum() /
             max(areas[idx] + areas[other] - np.logical_and(masks_arr[idx], masks_arr[other]).sum(), 1.0))
            <= iou_threshold
            for other in kept
        ):
            kept.append(int(idx))
    return np.asarray(kept, dtype=np.int64)
