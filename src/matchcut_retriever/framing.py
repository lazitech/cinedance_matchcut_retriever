"""Frame preparation and simple image quality checks.

The library and query paths share these helpers so that click coordinates and
object masks use exactly the same crop and resize convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# A bar is expected to be very dark and substantially more uniform than a
# naturally dark (textured) scene.  The thresholds tolerate JPEG ringing.
BAR_LUMA_MAX = 24.0
BAR_STD_MAX = 6.0
BAR_DARK_FRACTION = 0.94
MIN_ACTIVE_FRACTION = 0.30

QUALITY_BLACK_LUMA = 8.0
QUALITY_FLAT_STD = 3.0
QUALITY_MIN_STD = 4.0
QUALITY_MIN_SHARPNESS = 2.0


def _luma(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("image must be a grayscale or HxWx3 array")
    # OpenCV images are conventionally BGR.  The luminance coefficients are
    # symmetric enough for bar detection, and this also works for RGB input.
    return cv2.cvtColor(arr[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32)


def _bar_line(line: np.ndarray) -> bool:
    """Return whether a row/column is plausibly a letterbox bar."""
    line = np.asarray(line, dtype=np.float32)
    if line.size == 0:
        return False
    dark = line <= BAR_LUMA_MAX
    return (
        float(line.mean()) <= BAR_LUMA_MAX
        and float(line.std()) <= BAR_STD_MAX
        and float(dark.mean()) >= BAR_DARK_FRACTION
    )


def detect_active_area(image: np.ndarray) -> tuple[int, int, int, int]:
    """Find contiguous near-black bars and return the active half-open box.

    Only lines connected to an image border are considered, which prevents a
    dark band inside a scene from being mistaken for letterboxing.
    """
    lum = _luma(image)
    h, w = lum.shape
    # A frame that is dark everywhere but has real texture is a scene, not a
    # letterbox.  This guard prevents a few unusually quiet border rows from
    # being peeled off in night scenes.
    if float(lum.mean()) < 35.0 and float(lum.std()) > BAR_STD_MAX * 0.8:
        return (0, 0, w, h)
    top = 0
    while top < h and _bar_line(lum[top, :]):
        top += 1
    bottom = h
    while bottom > top and _bar_line(lum[bottom - 1, :]):
        bottom -= 1
    left = 0
    while left < w and _bar_line(lum[:, left]):
        left += 1
    right = w
    while right > left and _bar_line(lum[:, right - 1]):
        right -= 1

    # A full-frame dark scene can satisfy the line test.  Avoid returning an
    # implausibly tiny active region; this also handles an all-black frame.
    area_fraction = max(0, right - left) * max(0, bottom - top) / float(h * w)
    if area_fraction < MIN_ACTIVE_FRACTION or right <= left or bottom <= top:
        return (0, 0, w, h)
    return (int(left), int(top), int(right), int(bottom))


@dataclass(frozen=True)
class ImageTransform:
    """Coordinate transform produced by :func:`prepare_image`."""

    crop: tuple[int, int, int, int]
    scale: float

    def to_prepared(self, x: float, y: float) -> tuple[float, float]:
        x0, y0, _, _ = self.crop
        return ((float(x) - x0) * self.scale, (float(y) - y0) * self.scale)


def prepare_image(
    image: np.ndarray, long_side: int = 640
) -> tuple[np.ndarray, ImageTransform]:
    """Crop bars and resize an image so its longest side is ``long_side``."""
    if long_side <= 0:
        raise ValueError("long_side must be positive")
    arr = np.asarray(image)
    if arr.ndim not in (2, 3) or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("image must be a non-empty 2-D or 3-D array")
    crop = detect_active_area(arr)
    x0, y0, x1, y1 = crop
    cropped = arr[y0:y1, x0:x1]
    h, w = cropped.shape[:2]
    scale = float(long_side) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    # INTER_AREA is appropriate when reducing; interpolation is otherwise
    # chosen for smooth enlargement without introducing block artifacts.
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    prepared = cv2.resize(cropped, (new_w, new_h), interpolation=interp)
    # Keep the conceptual uniform scale in the transform.  The rounded output
    # dimensions can differ by one pixel on the short axis, but click mapping
    # should remain tied to the requested long-side scale.
    return prepared, ImageTransform(crop=crop, scale=scale)


def frame_quality(image: np.ndarray) -> dict[str, float | bool]:
    """Return luminance, texture, sharpness and an ingestion usability flag."""
    gray = _luma(image)
    mean_luma = float(gray.mean())
    std_luma = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    usable = bool(
        mean_luma > QUALITY_BLACK_LUMA
        and std_luma > QUALITY_MIN_STD
        and sharpness > QUALITY_MIN_SHARPNESS
        and not (mean_luma < QUALITY_BLACK_LUMA * 1.5 and std_luma < QUALITY_FLAT_STD)
    )
    return {
        "mean_luma": mean_luma,
        "std_luma": std_luma,
        "sharpness": sharpness,
        "usable": usable,
    }


__all__ = ["ImageTransform", "detect_active_area", "prepare_image", "frame_quality"]
