"""NumPy/OpenCV geometry and shape matching primitives.

Masks use row-major ``(y, x)`` array indexing. Geometric coordinates use
``(x, y)`` pixels, with the origin at the top-left of the frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class ObjectGeometry:
    """Geometry of one non-empty mask in its frame's pixel coordinates."""

    frame_w: int
    frame_h: int
    cx: float
    cy: float
    area: float
    bbox: tuple[int, int, int, int]
    angle: float
    eccentricity: float

    @property
    def area_frac(self) -> float:
        """Return mask area divided by frame area."""

        return self.area / (self.frame_w * self.frame_h)

    @property
    def bbox_norm(self) -> tuple[float, float, float, float]:
        """Return the half-open ``(x0, y0, x1, y1)`` box normalized by frame size."""

        x0, y0, x1, y1 = self.bbox
        return (x0 / self.frame_w, y0 / self.frame_h, x1 / self.frame_w, y1 / self.frame_h)


def object_geometry(mask: np.ndarray) -> ObjectGeometry:
    """Compute centroid, area, half-open box, principal angle and eccentricity.

    The input is a ``(H, W)`` mask; returned ``cx, cy`` and box coordinates
    are in ``(x, y)`` pixels. The principal angle is in ``(-pi/2, pi/2]``.
    """

    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2 or not np.any(array):
        raise ValueError("mask must be a non-empty 2-D array")
    frame_h, frame_w = array.shape
    ys, xs = np.nonzero(array)
    area = float(xs.size)
    cx = float(xs.mean())
    cy = float(ys.mean())
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    dx = xs.astype(np.float64) - cx
    dy = ys.astype(np.float64) - cy
    covariance = np.array([
        [np.mean(dx * dx), np.mean(dx * dy)],
        [np.mean(dx * dy), np.mean(dy * dy)],
    ])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    principal = eigenvectors[:, order[0]]
    largest = float(eigenvalues[0])
    smallest = float(max(eigenvalues[1], 0.0))
    eccentricity = 0.0 if largest <= 1e-12 else float(
        math.sqrt(max(0.0, 1.0 - smallest / largest))
    )
    if largest <= 1e-12 or abs(largest - smallest) <= 1e-10 * max(1.0, largest):
        angle = 0.0
    else:
        angle = (math.atan2(principal[1], principal[0]) + math.pi / 2) % math.pi - math.pi / 2
        if angle <= -math.pi / 2 + 1e-12:
            angle = math.pi / 2
    return ObjectGeometry(frame_w, frame_h, cx, cy, area, (x0, y0, x1, y1), angle, eccentricity)


FINGERPRINT_SIZE = 32
FINGERPRINT_DIM = FINGERPRINT_SIZE * FINGERPRINT_SIZE


def shape_fingerprint(mask: np.ndarray) -> np.ndarray:
    """Rasterize a mask into a translation/scale-invariant 1024-D fingerprint.

    Coordinates are normalized around the centroid in ``(x, y)`` pixels. The
    normalized square spans approximately ``+-2*sqrt(area)`` and preserves
    rotation because no principal-axis alignment is applied.
    """

    array = np.asarray(mask, dtype=np.uint8)
    geometry = object_geometry(array)
    scale = max(math.sqrt(geometry.area), 1e-6)
    window = 4.0 * scale
    factor = (FINGERPRINT_SIZE - 1) / window
    left = geometry.cx - 2.0 * scale
    top = geometry.cy - 2.0 * scale
    # Map the actual mask crop into the normalized frame using area sampling.
    affine = np.array(
        [[factor, 0.0, -left * factor], [0.0, factor, -top * factor]],
        dtype=np.float32,
    )
    raster = cv2.warpAffine(
        array.astype(np.float32), affine, (FINGERPRINT_SIZE, FINGERPRINT_SIZE),
        flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    raster = cv2.GaussianBlur(raster, (3, 3), 0)
    # Radial weighting preserves the compact raster's smoothness while making
    # silhouettes with different corner structure distinguishable at 32x32.
    yy, xx = np.mgrid[:FINGERPRINT_SIZE, :FINGERPRINT_SIZE]
    radius = np.sqrt((xx - (FINGERPRINT_SIZE - 1) / 2.0) ** 2
                     + (yy - (FINGERPRINT_SIZE - 1) / 2.0) ** 2)
    raster *= (radius / (FINGERPRINT_SIZE / 2.0)) ** 4
    vector = raster.reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector


def fingerprint_matrix(masks) -> np.ndarray:
    """Return one ``float32`` fingerprint row for each ``(y, x)`` mask."""

    rows = [shape_fingerprint(mask) for mask in masks]
    if not rows:
        return np.empty((0, FINGERPRINT_DIM), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32).reshape(-1, FINGERPRINT_DIM)


@dataclass(frozen=True)
class Similarity:
    """Similarity transform applied after candidate cover-fitting."""

    scale: float = 1.0
    angle: float = 0.0
    tx: float = 0.0
    ty: float = 0.0

    def matrix(self, canvas_wh=None) -> np.ndarray:
        """Return a 2x3 affine matrix for ``(x, y)`` pixel coordinates.

        When ``canvas_wh`` is given, the matrix implements
        ``s*R*(p-c)+c+(tx, ty)`` around the canvas center ``c``. Without it,
        the matrix uses the origin as the rotation center.
        """

        cosine, sine = math.cos(self.angle), math.sin(self.angle)
        linear = self.scale * np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
        if canvas_wh is None:
            translation = np.array([self.tx, self.ty], dtype=np.float64)
        else:
            width, height = map(float, canvas_wh)
            center = np.array([width / 2.0, height / 2.0])
            translation = center + np.array([self.tx, self.ty]) - linear @ center
        return np.column_stack((linear, translation))

    def apply(self, points, canvas_wh=None) -> np.ndarray:
        """Apply the transform to an ``(..., 2)`` array of ``(x, y)`` points."""

        coordinates = np.asarray(points, dtype=float)
        if canvas_wh is None:
            matrix = self.matrix()
            return coordinates @ matrix[:, :2].T + matrix[:, 2]
        width, height = map(float, canvas_wh)
        center = np.array([width / 2.0, height / 2.0])
        linear = self.scale * np.array([
            [math.cos(self.angle), -math.sin(self.angle)],
            [math.sin(self.angle), math.cos(self.angle)],
        ])
        return (coordinates - center) @ linear.T + center + np.array([self.tx, self.ty])


def cover_fit_matrix(cand_wh, query_wh) -> np.ndarray:
    """Return candidate-to-query matrix for CSS ``object-fit: cover``.

    Width and height arguments are ``(W, H)``; the returned 2x3 matrix maps
    candidate ``(x, y)`` pixels to centered query-canvas pixels.
    """

    candidate_width, candidate_height = map(float, cand_wh)
    query_width, query_height = map(float, query_wh)
    scale = max(query_width / candidate_width, query_height / candidate_height)
    return np.array([
        [scale, 0.0, (query_width - scale * candidate_width) / 2.0],
        [0.0, scale, (query_height - scale * candidate_height) / 2.0],
    ], dtype=float)


@dataclass(frozen=True)
class HardLimits:
    """Bounds for additional scale, canvas translation and rotation."""

    min_scale: float = 0.8
    max_scale: float = 1.25
    max_translate_frac: float = 0.10
    max_rot_deg: float = 6.0


@dataclass(frozen=True)
class SoftParams:
    """Controls soft matching and its dimensionless transform cost."""

    allow_rotation: bool = True
    ecc_threshold: float = 0.3
    cost_weight: float = 0.1
    scale_norm: float = 0.25
    translate_norm: float = 0.25
    rotation_norm_deg: float = 15.0


@dataclass
class MatchResult:
    """IoU, transform cost, score, transform and hard-limit status."""

    iou: float
    cost: float
    score: float
    transform: Similarity
    within_limits: bool


def _normalize_axis_angle(angle: float) -> float:
    return (angle + math.pi / 2) % math.pi - math.pi / 2


def _rotation(angle: float) -> np.ndarray:
    return np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ], dtype=float)


def _centroid_translation(query_geometry, candidate_geometry, cover_fit, scale, angle):
    query_center = np.array([query_geometry.frame_w / 2.0, query_geometry.frame_h / 2.0])
    query_point = np.array([query_geometry.cx, query_geometry.cy])
    candidate_point = cover_fit[:, :2] @ np.array([
        candidate_geometry.cx, candidate_geometry.cy,
    ]) + cover_fit[:, 2]
    return query_point - (
        query_center + scale * _rotation(angle) @ (candidate_point - query_center)
    )


def _closed(query_mask, candidate_mask, params=SoftParams(), max_rotation=None):
    query_geometry = object_geometry(query_mask)
    candidate_geometry = object_geometry(candidate_mask)
    query_height, query_width = query_mask.shape
    cover_fit = cover_fit_matrix(
        (candidate_mask.shape[1], candidate_mask.shape[0]), (query_width, query_height)
    )
    cover_scale = float(cover_fit[0, 0])
    scale = math.sqrt(query_geometry.area / (candidate_geometry.area * cover_scale ** 2))
    angle = 0.0
    if (
        params.allow_rotation
        and query_geometry.eccentricity > params.ecc_threshold
        and candidate_geometry.eccentricity > params.ecc_threshold
    ):
        angle = _normalize_axis_angle(query_geometry.angle - candidate_geometry.angle)
        if max_rotation is not None and abs(angle) > max_rotation:
            # A difference beyond the hard tilt bound describes different
            # shapes, rather than a transform that should be clamped.
            angle = 0.0
    translation = _centroid_translation(
        query_geometry, candidate_geometry, cover_fit, scale, angle
    )
    return Similarity(scale, angle, float(translation[0]), float(translation[1])), cover_fit


def _warp_iou(query_mask, candidate_mask, transform, cover_fit, long_side=160):
    query_height, query_width = query_mask.shape
    resize_scale = min(1.0, float(long_side) / max(query_height, query_width))
    output_size = (
        max(1, int(round(query_width * resize_scale))),
        max(1, int(round(query_height * resize_scale))),
    )
    query_small = cv2.resize(
        query_mask.astype(np.uint8), output_size, interpolation=cv2.INTER_NEAREST
    )
    transform_matrix = transform.matrix((query_width, query_height))
    homogeneous_cover = np.vstack((cover_fit, [0.0, 0.0, 1.0]))
    homogeneous_transform = np.vstack((transform_matrix, [0.0, 0.0, 1.0]))
    matrix = (homogeneous_transform @ homogeneous_cover)[:2]
    matrix[:, :2] *= resize_scale
    matrix[:, 2] *= resize_scale
    candidate_small = cv2.warpAffine(
        candidate_mask.astype(np.uint8), matrix, output_size,
        flags=cv2.INTER_NEAREST, borderValue=0,
    )
    intersection = np.count_nonzero(query_small & candidate_small)
    union = np.count_nonzero(query_small | candidate_small)
    return float(intersection / union) if union else 1.0


def _iou_from_small(
    query_small, candidate_small, transform, cover_fit, resize_scale, output_size, query_wh
):
    """Compute IoU from masks already resized by the same coordinate factor."""

    transform_matrix = transform.matrix(query_wh)
    homogeneous_cover = np.vstack((cover_fit, [0.0, 0.0, 1.0]))
    homogeneous_transform = np.vstack((transform_matrix, [0.0, 0.0, 1.0]))
    matrix = (homogeneous_transform @ homogeneous_cover)[:2]
    # Source coordinates are candidate_small pixels, so the linear part is
    # unchanged; only destination translation is expressed at small scale.
    matrix[:, 2] *= resize_scale
    candidate_small = cv2.warpAffine(
        candidate_small, matrix, output_size,
        flags=cv2.INTER_NEAREST, borderValue=0,
    )
    intersection = np.count_nonzero(query_small & candidate_small)
    union = np.count_nonzero(query_small | candidate_small)
    return float(intersection / union) if union else 1.0


def _cost(transform: Similarity, frame_wh, params=SoftParams()) -> float:
    width, _ = frame_wh
    translation_term = math.hypot(transform.tx, transform.ty) / (
        params.translate_norm * width + 1e-9
    )
    return float(
        abs(math.log(max(transform.scale, 1e-9))) / params.scale_norm
        + abs(transform.angle) / math.radians(params.rotation_norm_deg)
        + translation_term
    )


def soft_match(q_mask, c_mask, params=SoftParams()) -> MatchResult:
    """Compare masks with centroid alignment and a soft transform penalty.

    Masks are ``(H, W)`` arrays and transform translations are query-canvas
    ``(x, y)`` pixels. IoU is measured after candidate cover-fitting.
    """

    closed, cover_fit = _closed(q_mask, c_mask, params)
    candidates = [closed]
    query_geometry = object_geometry(q_mask)
    candidate_geometry = object_geometry(c_mask)
    if (
        params.allow_rotation
        and query_geometry.eccentricity > params.ecc_threshold
        and candidate_geometry.eccentricity > params.ecc_threshold
    ):
        for angle in (closed.angle + math.pi, closed.angle - math.pi):
            translation = _centroid_translation(
                query_geometry, candidate_geometry, cover_fit, closed.scale, angle
            )
            candidates.append(
                Similarity(closed.scale, angle, float(translation[0]), float(translation[1]))
            )
    scored = []
    for candidate in candidates:
        iou = _warp_iou(q_mask, c_mask, candidate, cover_fit)
        cost = _cost(candidate, (q_mask.shape[1], q_mask.shape[0]), params)
        scored.append((iou - params.cost_weight * cost, iou, candidate, cost))
    # The axis-direction alternatives are ranked by the same adjusted score
    # returned to callers, so a tiny IoU gain cannot hide a large transform cost.
    score, iou, transform, cost = max(scored, key=lambda item: item[0])
    defaults = HardLimits()
    within_limits = (
        defaults.min_scale <= transform.scale <= defaults.max_scale
        and math.hypot(transform.tx, transform.ty) <= defaults.max_translate_frac * q_mask.shape[1]
        and abs(transform.angle) <= math.radians(defaults.max_rot_deg)
    )
    return MatchResult(iou, cost, score, transform, within_limits)


def _clip_translation(translation, maximum) -> np.ndarray:
    vector = np.asarray(translation, dtype=float).copy()
    norm = float(np.linalg.norm(vector))
    if norm > maximum and norm > 0.0:
        vector *= maximum / norm
    return vector


def _clip_transform(transform: Similarity, limits: HardLimits, width: int) -> Similarity:
    scale = min(max(transform.scale, limits.min_scale), limits.max_scale)
    rotation = min(
        max(transform.angle, -math.radians(limits.max_rot_deg)),
        math.radians(limits.max_rot_deg),
    )
    translation = _clip_translation(
        [transform.tx, transform.ty], limits.max_translate_frac * width
    )
    return Similarity(scale, rotation, float(translation[0]), float(translation[1]))


def _coarse_translations(centroid_translation, maximum) -> list[np.ndarray]:
    values = np.linspace(-maximum, maximum, 3)
    points = [
        np.array([x, y], dtype=float)
        for x in values
        for y in values
        if math.hypot(x, y) <= maximum + 1e-9
    ]
    points.extend((np.zeros(2), _clip_translation(centroid_translation, maximum)))
    unique = {}
    for point in points:
        unique[(round(float(point[0]), 7), round(float(point[1]), 7))] = point
    return list(unique.values())


def hard_match(q_mask, c_mask, limits=HardLimits()) -> MatchResult:
    """Find best in-limit query-canvas IoU and report closed-form feasibility.

    The returned transform is always projected into ``limits``. The
    ``within_limits`` flag refers to the unconstrained centroid/area/axis
    solution; candidates to exclude are the responsibility of
    :func:`hard_prefilter`.
    """

    _, query_width = q_mask.shape
    maximum_translation = limits.max_translate_frac * query_width
    maximum_rotation = math.radians(limits.max_rot_deg)
    unrestricted, cover_fit = _closed(q_mask, c_mask, SoftParams())
    within_limits = (
        limits.min_scale <= unrestricted.scale <= limits.max_scale
        and math.hypot(unrestricted.tx, unrestricted.ty) <= maximum_translation
        and abs(unrestricted.angle) <= maximum_rotation
    )
    projected_closed, _ = _closed(
        q_mask, c_mask, SoftParams(), max_rotation=maximum_rotation
    )
    projected = _clip_transform(projected_closed, limits, query_width)

    scale_values = np.unique(np.array([
        limits.min_scale, 1.0, limits.max_scale, projected.scale,
    ]))
    angle_values = np.unique(np.array([
        -maximum_rotation, 0.0, maximum_rotation, projected.angle,
    ]))
    query_geometry = object_geometry(q_mask)
    candidate_geometry = object_geometry(c_mask)
    hypotheses = []
    for scale in scale_values:
        for angle in angle_values:
            centroid_translation = _centroid_translation(
                query_geometry, candidate_geometry, cover_fit, float(scale), float(angle)
            )
            for translation in _coarse_translations(
                centroid_translation, maximum_translation
            ):
                hypotheses.append(Similarity(
                    float(scale), float(angle), float(translation[0]), float(translation[1])
                ))
    coarse_scale = min(1.0, 64.0 / max(q_mask.shape))
    coarse_output = (
        max(1, int(round(q_mask.shape[1] * coarse_scale))),
        max(1, int(round(q_mask.shape[0] * coarse_scale))),
    )
    coarse_query = cv2.resize(
        q_mask.astype(np.uint8), coarse_output, interpolation=cv2.INTER_NEAREST
    )
    coarse_candidate_size = (
        max(1, int(round(c_mask.shape[1] * coarse_scale))),
        max(1, int(round(c_mask.shape[0] * coarse_scale))),
    )
    coarse_candidate = cv2.resize(
        c_mask.astype(np.uint8), coarse_candidate_size, interpolation=cv2.INTER_NEAREST
    )
    coarse = [
        (
            _iou_from_small(
                coarse_query, coarse_candidate, hypothesis, cover_fit,
                coarse_scale, coarse_output, (query_width, q_mask.shape[0])
            ),
            hypothesis,
        )
        for hypothesis in hypotheses
    ]
    coarse.sort(key=lambda item: item[0], reverse=True)
    best_iou, best_transform = coarse[0]
    best_iou = _warp_iou(q_mask, c_mask, best_transform, cover_fit)
    projected_iou = _warp_iou(q_mask, c_mask, projected, cover_fit)
    if projected_iou > best_iou:
        best_iou, best_transform = projected_iou, projected

    scale_step = max(0.01, (limits.max_scale - limits.min_scale) / 20.0)
    angle_step = max(math.radians(0.25), maximum_rotation / 6.0)
    # The projected closed form is refined explicitly; the coarse winner was
    # already evaluated at full resolution above.
    seeds = [projected]
    for seed in seeds:
        for scale_delta in (-scale_step, 0.0, scale_step):
            for angle_delta in (-angle_step, 0.0, angle_step):
                candidate = _clip_transform(
                    Similarity(
                        seed.scale + scale_delta, seed.angle + angle_delta,
                        seed.tx, seed.ty,
                    ), limits, query_width
                )
                iou = _warp_iou(q_mask, c_mask, candidate, cover_fit)
                if iou > best_iou:
                    best_iou = iou
                    best_transform = candidate
    return MatchResult(
        iou=float(best_iou),
        cost=_cost(best_transform, (query_width, q_mask.shape[0])),
        score=float(best_iou), transform=best_transform, within_limits=within_limits,
    )


def hard_prefilter(q_geom: ObjectGeometry, cand: dict[str, np.ndarray], limits) -> np.ndarray:
    """Conservatively filter candidate geometry by feasible area and centroid.

    Candidate arrays contain ``frame_w, frame_h, cx, cy, area`` in each
    candidate frame's ``(x, y)`` pixels. Extra candidates may be retained, but
    one with a feasible centroid/area transform is never removed.
    """

    frame_w = np.asarray(cand["frame_w"], dtype=float)
    frame_h = np.asarray(cand["frame_h"], dtype=float)
    cx = np.asarray(cand["cx"], dtype=float)
    cy = np.asarray(cand["cy"], dtype=float)
    area = np.asarray(cand["area"], dtype=float)
    query_width, query_height = q_geom.frame_w, q_geom.frame_h
    maximum_translation = limits.max_translate_frac * query_width
    maximum_rotation = math.radians(limits.max_rot_deg)
    cover_scale = np.maximum(query_width / frame_w, query_height / frame_h)
    candidate_points = np.stack([
        cover_scale * cx + (query_width - cover_scale * frame_w) / 2.0,
        cover_scale * cy + (query_height - cover_scale * frame_h) / 2.0,
    ], axis=1)
    required_scale = np.sqrt(q_geom.area / (area * cover_scale ** 2))
    feasible = (required_scale >= limits.min_scale) & (required_scale <= limits.max_scale)

    # Rotating around the canvas center makes the centroid trace a circular
    # arc. Compute its exact closest distance instead of angular sampling.
    canvas_center = np.array([query_width / 2.0, query_height / 2.0])
    query_vector = np.array([q_geom.cx, q_geom.cy]) - canvas_center
    query_radius = float(np.linalg.norm(query_vector))
    candidate_vectors = candidate_points - canvas_center
    candidate_radius = np.linalg.norm(candidate_vectors, axis=1) * required_scale
    candidate_angle = np.arctan2(candidate_vectors[:, 1], candidate_vectors[:, 0])
    query_angle = math.atan2(float(query_vector[1]), float(query_vector[0]))
    angle_delta = (query_angle - candidate_angle + math.pi) % (2.0 * math.pi) - math.pi
    on_arc = np.abs(angle_delta) <= maximum_rotation
    radial_distance = np.abs(candidate_radius - query_radius)

    def endpoint_distance(sign):
        endpoint_angle = candidate_angle + sign * maximum_rotation
        return np.sqrt(np.maximum(
            0.0,
            candidate_radius ** 2 + query_radius ** 2
            - 2.0 * candidate_radius * query_radius
            * np.cos(endpoint_angle - query_angle),
        ))

    closest_distance = np.where(
        on_arc, radial_distance,
        np.minimum(endpoint_distance(-1), endpoint_distance(1)),
    )
    feasible &= closest_distance <= maximum_translation + 1e-9
    return feasible


__all__ = [
    "ObjectGeometry", "FINGERPRINT_DIM", "shape_fingerprint", "fingerprint_matrix",
    "object_geometry", "Similarity", "cover_fit_matrix", "HardLimits", "SoftParams",
    "MatchResult", "soft_match", "hard_match", "hard_prefilter",
]
