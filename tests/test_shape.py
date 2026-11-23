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
    cover_fit_matrix,
    _iou_from_small,
    _warp_iou,
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


def test_hard_match_deterministic_polygon_alignment():
    candidate = np.zeros((180, 320), dtype=bool)
    cv2.fillPoly(candidate, [np.array([[90, 45], [150, 55], [125, 125], [70, 110]])], 1)
    query = np.zeros_like(candidate)
    cv2.fillPoly(query, [np.array([[125, 50], [185, 60], [160, 130], [105, 115]])], 1)
    result = hard_match(query, candidate)
    assert result.iou >= 0.85


def test_cached_iou_matches_warp_iou_for_different_aspect_ratios():
    rng = np.random.default_rng(11)
    query = rng.random((241, 377)) > 0.88
    candidate = rng.random((199, 521)) > 0.91
    query[120, 188] = True
    candidate[99, 260] = True
    query_wh = (query.shape[1], query.shape[0])
    candidate_wh = (candidate.shape[1], candidate.shape[0])
    cover_fit = cover_fit_matrix(candidate_wh, query_wh)
    resize_scale = min(1.0, 160.0 / max(query.shape))
    query_size = (
        round(query.shape[1] * resize_scale),
        round(query.shape[0] * resize_scale),
    )
    candidate_size = (
        round(candidate.shape[1] * resize_scale),
        round(candidate.shape[0] * resize_scale),
    )
    query_small = cv2.resize(query.astype(np.uint8), query_size, interpolation=cv2.INTER_NEAREST)
    candidate_small = cv2.resize(
        candidate.astype(np.uint8), candidate_size, interpolation=cv2.INTER_NEAREST
    )
    for transform in (
        Similarity(0.82, -0.03, 12.0, -7.0),
        Similarity(1.18, 0.04, -9.0, 8.0),
    ):
        full = _warp_iou(query, candidate, transform, cover_fit)
        cached = _iou_from_small(
            query_small,
            candidate_small,
            transform,
            cover_fit,
            resize_scale,
            query_size,
            query_wh,
            candidate_wh,
        )
        assert abs(full - cached) <= 0.005
