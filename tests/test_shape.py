import math

import cv2
import numpy as np

from matchcut_retriever.shape import (
    HardLimits,
    Similarity,
    hard_match,
    object_geometry,
    shape_fingerprint,
    hard_prefilter,
)


def _ellipse(axes, center=(80, 80), shape=(160, 160)):
    mask = np.zeros(shape, dtype=bool)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
    return mask


def test_fingerprint_invariance_and_compact_shape_discrimination():
    disc = _ellipse((30, 30))
    translated_scaled = _ellipse((60, 60), center=(120, 90), shape=(240, 240))
    square = np.zeros((160, 160), dtype=bool)
    square[54:107, 54:107] = True
    triangle = np.zeros((160, 160), dtype=bool)
    cv2.fillPoly(triangle, [np.array([[80, 45], [113, 102], [47, 102]])], 1)
    disc_fp = shape_fingerprint(disc)
    same = float(disc_fp @ shape_fingerprint(translated_scaled))
    disc_square = float(disc_fp @ shape_fingerprint(square))
    disc_triangle = float(disc_fp @ shape_fingerprint(triangle))
    assert same >= 0.95
    assert disc_square <= same - 0.05
    assert disc_triangle <= same - 0.05


def test_hard_match_searches_identity_when_closed_form_is_outside():
    query = np.zeros((180, 320), dtype=bool)
    query[55:125, 120:200] = True
    candidate = np.zeros_like(query)
    candidate[70:110, 140:180] = True
    result = hard_match(query, candidate)
    assert result.transform.scale <= HardLimits().max_scale
    assert not result.within_limits


def test_hard_match_recovers_in_limit_transform():
    candidate = _ellipse((30, 12), center=(140, 90), shape=(180, 320))
    transform = Similarity(1.1, math.radians(4), 20, -10)
    query = cv2.warpAffine(
        candidate.astype(np.uint8), transform.matrix((320, 180)), (320, 180),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    result = hard_match(query, candidate)
    assert result.within_limits
    assert result.iou >= 0.95
    assert abs(result.transform.tx - 20) <= 2
    assert abs(result.transform.ty + 10) <= 2
    assert abs(result.transform.angle) <= math.radians(6)


def test_geometry_disc_eccentricity_is_near_zero():
    geometry = object_geometry(_ellipse((30, 30)))
    assert geometry.eccentricity < 0.1


def test_prefilter_keeps_random_closed_form_feasible_cases():
    rng = np.random.default_rng(7)
    limits = HardLimits()
    for _ in range(25):
        query = _ellipse(
            (int(rng.integers(10, 30)), int(rng.integers(8, 24))),
            center=(int(rng.integers(45, 275)), int(rng.integers(35, 145))),
            shape=(180, 320),
        )
        candidate = _ellipse(
            (int(rng.integers(10, 30)), int(rng.integers(8, 24))),
            center=(int(rng.integers(45, 275)), int(rng.integers(35, 145))),
            shape=(180, 320),
        )
        result = hard_match(query, candidate, limits)
        geometry = object_geometry(candidate)
        arrays = {
            name: np.array([getattr(geometry, name)])
            for name in ("frame_w", "frame_h", "cx", "cy", "area")
        }
        kept = hard_prefilter(object_geometry(query), arrays, limits)[0]
        assert kept or not result.within_limits


def test_hard_match_respects_non_unit_and_zero_rotation_limits():
    mask = _ellipse((30, 12), center=(140, 90), shape=(180, 320))
    for limits in (
        HardLimits(min_scale=1.1, max_scale=1.3),
        HardLimits(min_scale=0.8, max_scale=0.9),
        HardLimits(max_rot_deg=0.0),
    ):
        transform = hard_match(mask, mask, limits).transform
        assert limits.min_scale <= transform.scale <= limits.max_scale
        assert abs(transform.angle) <= math.radians(limits.max_rot_deg) + 1e-9
        assert math.hypot(transform.tx, transform.ty) <= (
            limits.max_translate_frac * mask.shape[1] + 1e-9
        )
