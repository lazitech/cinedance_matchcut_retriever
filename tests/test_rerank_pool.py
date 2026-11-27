import math
from types import SimpleNamespace

import numpy as np

from matchcut_retriever import retrieval
from matchcut_retriever.masks import rle_encode
from matchcut_retriever.retrieval_config import load_retrieval_config
from matchcut_retriever.shape import hard_prefilter


def _ellipse(h, w, cx, cy, a, b):
    yy, xx = np.mgrid[:h, :w]
    return ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1


def test_parallel_rerank_matches_serial_and_reuses_pool():
    cfg = load_retrieval_config()
    rng = np.random.default_rng(0)
    query = _ellipse(90, 160, 80, 45, 30, 18)
    masks = [_ellipse(90, 160, 80 + rng.uniform(-15, 15), 45 + rng.uniform(-10, 10), rng.uniform(15, 40), rng.uniform(10, 25))
             for _ in range(20)]
    blobs = [rle_encode(m) for m in masks]
    for mode in ("soft", "hard"):
        serial = retrieval._rerank(query, blobs, mode, cfg.hard_limits, cfg.soft_params, workers=1)
        parallel = retrieval._rerank(query, blobs, mode, cfg.hard_limits, cfg.soft_params, workers=2)
        assert len(parallel) == len(blobs)
        for a, b in zip(serial, parallel):
            assert math.isclose(a.iou, b.iou) and math.isclose(a.cost, b.cost)
            assert a.within_limits == b.within_limits
    pool = retrieval._get_pool(2)
    retrieval._rerank(query, blobs, "soft", cfg.hard_limits, cfg.soft_params, workers=2)
    assert retrieval._get_pool(2) is pool


def _reference_prefilter(q, cand, limits):
    # The pre-optimization implementation, evaluated on every candidate.
    fw, fh = cand["frame_w"].astype(float), cand["frame_h"].astype(float)
    cx, cy, area = cand["cx"].astype(float), cand["cy"].astype(float), cand["area"].astype(float)
    qw, qh = q.frame_w, q.frame_h
    max_t, max_r = limits.max_translate_frac * qw, math.radians(limits.max_rot_deg)
    cover = np.maximum(qw / fw, qh / fh)
    pts = np.stack([cover * cx + (qw - cover * fw) / 2, cover * cy + (qh - cover * fh) / 2], axis=1)
    req = np.sqrt(q.area / (area * cover ** 2))
    ok = (req >= limits.min_scale) & (req <= limits.max_scale)
    center = np.array([qw / 2, qh / 2])
    qv = np.array([q.cx, q.cy]) - center
    qr = float(np.linalg.norm(qv))
    cv = pts - center
    cr = np.linalg.norm(cv, axis=1) * req
    ca = np.arctan2(cv[:, 1], cv[:, 0])
    qa = math.atan2(qv[1], qv[0])
    on_arc = np.abs((qa - ca + math.pi) % (2 * math.pi) - math.pi) <= max_r
    end = lambda s: np.sqrt(np.maximum(0, cr ** 2 + qr ** 2 - 2 * cr * qr * np.cos(ca + s * max_r - qa)))
    dist = np.where(on_arc, np.abs(cr - qr), np.minimum(end(-1), end(1)))
    return ok & (dist <= max_t + 1e-9)


def test_hard_prefilter_matches_full_evaluation():
    limits = load_retrieval_config().hard_limits
    rng = np.random.default_rng(1)
    n = 5000
    cand = {
        "frame_w": rng.choice([640, 640, 480], n).astype(np.int32),
        "frame_h": rng.choice([272, 360, 640], n).astype(np.int32),
        "cx": rng.uniform(0, 640, n).astype(np.float32),
        "cy": rng.uniform(0, 360, n).astype(np.float32),
        "area": rng.uniform(500, 40000, n).astype(np.float32),
    }
    for _ in range(5):
        q = SimpleNamespace(frame_w=640, frame_h=360, cx=float(rng.uniform(100, 540)), cy=float(rng.uniform(80, 280)),
                            area=float(rng.uniform(2000, 30000)))
        got = hard_prefilter(q, cand, limits)
        want = _reference_prefilter(q, cand, limits)
        assert got.dtype == bool and np.array_equal(got, want)
        assert 0 < got.sum() < n
