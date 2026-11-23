import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from matchcut_retriever.masks import rle_encode
from matchcut_retriever.retrieval import Library, QueryObject, rank
from matchcut_retriever.retrieval_config import RetrievalConfig
from matchcut_retriever.shape import HardLimits, object_geometry, shape_fingerprint
from matchcut_retriever.query import _prepare, _transform_mask, build_parser, main as query_main


def _mask(kind, shape=(80, 120), center=(60, 40), size=12, angle=0):
    h, w = shape
    out = np.zeros((h, w), bool)
    if kind == "disc":
        cv2.circle(out, center, size, 1, -1)
    elif kind == "square":
        x, y = center
        out[y - size:y + size, x - size:x + size] = 1
    elif kind == "triangle":
        x, y = center
        cv2.fillPoly(out, [np.array([[x, y - size], [x + size, y + size], [x - size, y + size]])], 1)
    elif kind == "bar":
        x, y = center
        cv2.rectangle(out, (x - size * 2, y - size // 2), (x + size * 2, y + size // 2), 1, -1)
    elif kind == "ellipse":
        axes = tuple(size) if isinstance(size, (tuple, list)) else (int(size), int(size))
        cv2.ellipse(out, center, axes, 0, 0, 360, 1, -1)
    if angle:
        matrix = cv2.getRotationMatrix2D(center, angle, 1)
        out = cv2.warpAffine(out.astype(np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST).astype(bool)
    return out


def make_library(tmp_path, specs):
    root = Path(tmp_path)
    (root / "index").mkdir(parents=True)
    objects, keyframes, shots, blob = [], [], [], bytearray()
    fps, embs = [], []
    for oid, spec in enumerate(specs):
        clip, shot, kind = spec[:3]
        shape = spec[3] if len(spec) > 3 else (80, 120)
        center = spec[4] if len(spec) > 4 else (60, 40)
        size = spec[5] if len(spec) > 5 else 12
        angle = spec[6] if len(spec) > 6 else 0
        mask = _mask(kind, shape, center, size, angle)
        geom = object_geometry(mask)
        encoded = rle_encode(mask)
        offset = len(blob)
        blob.extend(encoded)
        keyframe_id = f"{clip}:{shot}:0"
        path = f"keyframe_{oid}.jpg"
        Image.fromarray(np.full((*shape, 3), 160, np.uint8)).save(root / path)
        objects.append(dict(object_id=oid, keyframe_id=keyframe_id, clip_id=clip, shot_no=shot, k=0,
            t_s=1.0, frame_w=shape[1], frame_h=shape[0], cx=geom.cx, cy=geom.cy,
            area=geom.area, area_frac=geom.area_frac, bbox_x0=geom.bbox[0], bbox_y0=geom.bbox[1],
            bbox_x1=geom.bbox[2], bbox_y1=geom.bbox[3], angle=geom.angle,
            eccentricity=geom.eccentricity, n_components=1, solidity=1.0, score=1.0,
            label=kind, segmenter="synthetic", mask_offset=offset, mask_nbytes=len(encoded)))
        keyframes.append(dict(keyframe_id=keyframe_id, clip_id=clip, shot_no=shot, k=0,
            t_s=1.0, frame_idx=1, path=path, width=shape[1], height=shape[0]))
        shots.append(dict(clip_id=clip, shot_no=shot, start_s=0.0, end_s=2.0))
        fps.append(shape_fingerprint(mask))
        embs.append(np.array([1.0, 0.0] if kind == "disc" else [0.0, 1.0], np.float32))
    pd.DataFrame(objects).to_parquet(root / "objects.parquet")
    pd.DataFrame(keyframes).to_parquet(root / "keyframes.parquet")
    pd.DataFrame(shots).drop_duplicates().to_parquet(root / "shots.parquet")
    (root / "index" / "masks.bin").write_bytes(bytes(blob))
    np.save(root / "index" / "shape_fp.npy", np.asarray(fps, np.float16))
    np.save(root / "index" / "clip_emb.npy", np.asarray(embs, np.float16))
    (root / "index" / "meta.json").write_text(json.dumps({"n_objects": len(objects), "fingerprint_dim": 1024, "clip_dim": 2}))
    return root


@pytest.fixture
def library(tmp_path):
    specs = [("disc", 0, "disc"), ("square", 0, "square"), ("tri", 0, "triangle"),
             ("same", 0, "disc", (80, 120), (60, 40), 12), ("same", 1, "disc", (80, 120), (60, 40), 12)]
    return make_library(tmp_path, specs)


def query_for(lib, object_id=0, clip=None):
    library = Library.load(lib)
    image = np.zeros((80, 120, 3), np.uint8)
    return library, QueryObject.from_mask(image, library.mask(object_id), clip)


def test_library_load_and_soft_shape_ranking(library):
    lib, query = query_for(library)
    results = rank(lib, query, "soft", k=5, params=RetrievalConfig(workers=1, coarse_n=20))
    assert results[0].clip_id == "disc"
    assert results[0].geometric_score >= results[1].geometric_score
    assert results.timings["coarse_search_s"] >= 0


def test_hard_prefilter_limits_and_changed_limits(library):
    lib, query = query_for(library)
    shifted = make_library(library / "shifted", [("inside", 0, "disc", (80, 120), (72, 40), 12),
        ("outside", 0, "disc", (80, 120), (95, 40), 12)])
    lib2, q2 = query_for(shifted)
    strict = RetrievalConfig(workers=1, hard_coarse_n=20, hard_limits=HardLimits(max_translate_frac=.1))
    ids = {r.clip_id for r in rank(lib2, q2, "hard", params=strict, k=20)}
    assert "inside" in ids and "outside" not in ids
    loose = RetrievalConfig(workers=1, hard_coarse_n=20, hard_limits=HardLimits(max_translate_frac=.4))
    assert "outside" in {r.clip_id for r in rank(lib2, q2, "hard", params=loose, k=20)}


def test_hard_mode_keeps_prefilter_survivors_even_if_within_limits_false(library):
    root = make_library(library / "axis", [("q", 0, "disc"), ("near_circle", 0, "disc", (80, 120), (60, 40), 12, 40)])
    lib, query = query_for(root)
    results = rank(lib, query, "hard", params=RetrievalConfig(workers=1, hard_coarse_n=20), k=20)
    assert "near_circle" in {r.clip_id for r in results}


def test_hard_cover_fit_wide_frame(tmp_path):
    root = make_library(tmp_path / "wide", [("wide", 0, "disc", (120, 80), (40, 60), 7)])
    lib = Library.load(root)
    query_mask = _mask("disc", (80, 120), (60, 40), 10)
    query = QueryObject.from_mask(np.zeros((80, 120, 3), np.uint8), query_mask)
    results = rank(lib, query, "hard", params=RetrievalConfig(workers=1, hard_coarse_n=10), k=5)
    assert results


def test_hard_scale_rotation_exclusions_and_inside_limits(tmp_path):
    root = make_library(tmp_path / "limits", [
        ("query", 0, "bar", (80, 120), (60, 40), 10, 0),
        ("inside_scale", 0, "bar", (80, 120), (60, 40), 11, 0),
        ("outside_scale", 0, "bar", (80, 120), (60, 40), 24, 0),
        ("inside_rot", 0, "bar", (80, 120), (60, 40), 10, 4),
        ("outside_rot", 0, "bar", (80, 120), (60, 40), 10, 20),
    ])
    lib, query = query_for(root, 0)
    cfg = RetrievalConfig(workers=1, hard_coarse_n=20)
    ids = {r.clip_id for r in rank(lib, query, "hard", params=cfg, k=20)}
    assert "inside_scale" in ids and "outside_scale" not in ids
    assert "inside_rot" in ids and "outside_rot" not in ids


def test_hard_cover_fit_different_aspect_ratio(tmp_path):
    root = make_library(tmp_path / "cover", [
        ("query", 0, "disc", (80, 120), (60, 40), 12),
        ("cover", 0, "disc", (120, 80), (40, 60), 8),
    ])
    lib, query = query_for(root, 0)
    results = rank(lib, query, "hard", params=RetrievalConfig(workers=1, hard_coarse_n=20), k=20)
    assert "cover" in {result.clip_id for result in results}


def test_hard_near_circular_axis_is_informational(tmp_path):
    root = make_library(tmp_path / "round", [("query", 0, "disc"), ("round", 0, "disc")])
    objects = pd.read_parquet(root / "objects.parquet")
    objects.loc[1, "angle"] = math.radians(40)
    objects.loc[1, "eccentricity"] = 0.0
    objects.to_parquet(root / "objects.parquet")
    lib, query = query_for(root)
    results = rank(lib, query, "hard", params=RetrievalConfig(workers=1, hard_coarse_n=20), k=20)
    assert "round" in {result.clip_id for result in results}


def test_merge_and_semantic_weight(library):
    lib, query = query_for(library, clip=np.array([1., 0.]))
    cfg = RetrievalConfig(workers=1, coarse_n=20)
    results = rank(lib, query, "soft", semantic_weight=0, params=cfg, k=20)
    assert len([(r.clip_id, r.shot_no) for r in results]) == len(set((r.clip_id, r.shot_no) for r in results))
    promoted = rank(lib, query, "soft", semantic_weight=1, params=cfg, k=20)
    assert promoted[0].clip_cosine is not None
    demoted = rank(lib, query, "soft", semantic_weight=-1, params=cfg, k=20)
    assert demoted[0].clip_id != "disc" or demoted[0].final_score <= promoted[0].final_score
    with pytest.raises(ValueError, match="query.clip_vector"):
        rank(lib, QueryObject.from_mask(np.zeros((80, 120, 3), np.uint8), lib.mask(0)), "soft", semantic_weight=1, params=cfg)


def test_semantic_positive_and_negative_change_order(tmp_path):
    root = make_library(tmp_path / "semantic", [("same", 0, "disc"), ("other", 0, "disc")])
    embeddings = np.asarray([[1., 0.], [-1., 0.]], np.float16)
    np.save(root / "index" / "clip_emb.npy", embeddings)
    lib, query = query_for(root, clip=np.array([1., 0.]))
    cfg = RetrievalConfig(workers=1, coarse_n=20)
    positive = rank(lib, query, "soft", semantic_weight=1, params=cfg, k=20)
    negative = rank(lib, query, "soft", semantic_weight=-1, params=cfg, k=20)
    assert positive[0].clip_id == "same"
    assert negative[0].clip_id == "other"


def test_library_rejects_length_mismatch(library):
    arr = np.load(library / "index" / "shape_fp.npy")
    np.save(library / "index" / "shape_fp.npy", arr[:-1])
    with pytest.raises(ValueError, match="length"):
        Library.load(library)


def test_cli_mask_prepared_end_to_end(library, tmp_path):
    lib = Library.load(library)
    image = np.zeros((80, 120, 3), np.uint8)
    image_path, mask_path = tmp_path / "query.jpg", tmp_path / "mask.png"
    Image.fromarray(image).save(image_path)
    Image.fromarray((lib.mask(0) * 255).astype(np.uint8)).save(mask_path)
    out = tmp_path / "out"
    env = dict(os.environ, PYTHONPATH="src")
    subprocess.run([".venv/bin/python", "-m", "matchcut_retriever.query", "--image", str(image_path), "--x", "60", "--y", "40", "--prepared", "--mask", str(mask_path), "--no-clip", "--out", str(out), "--library-root", str(library)], check=True, env=env)
    assert (out / "grid.jpg").is_file() and (out / "results.json").is_file() and (out / "query_mask.png").is_file()
    assert len(json.loads((out / "results.json").read_text())["results"]) <= 20


def test_unprepared_mask_uses_nearest_image_transform():
    image = np.zeros((100, 1200, 3), np.uint8)
    original = np.zeros((100, 1200), bool)
    original[20:40, 200:240] = True
    prepared, _, _, transform = _prepare(image, 220, 30)
    transformed = _transform_mask(original, prepared.shape, transform)
    assert transformed.shape == prepared.shape[:2]
    assert transformed.sum() > 0
    assert transformed.sum() < original.sum()


def test_cli_separate_configs_and_configured_semantic_weight(library, tmp_path, monkeypatch):
    image_path = tmp_path / "query.jpg"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.zeros((80, 120, 3), np.uint8)).save(image_path)
    Image.fromarray((Library.load(library).mask(0) * 255).astype(np.uint8)).save(mask_path)
    local = tmp_path / "paths.toml"
    local.write_text("[paths]\nvideo_root='/v'\nmanifest_dir='/m'\nannotation_dir='/a'\nlibrary_root='" + str(library) + "'\nhf_cache='/cache'\n")
    retrieval = tmp_path / "custom-name.toml"
    retrieval.write_text("[retrieval]\nsemantic_weight=0.5\nworkers=1\nk=3\ncoarse_n=20\n")

    class FakeClip:
        def __init__(self, **kwargs):
            assert kwargs["cache_dir"] == Path("/cache")
        def embed_masked(self, images, masks):
            return np.asarray([[1.0, 0.0]], np.float32)

    monkeypatch.setattr("matchcut_retriever.models.ClipEmbedder", FakeClip)
    out = tmp_path / "out"
    query_main(["--image", str(image_path), "--x", "60", "--y", "40", "--prepared", "--mask", str(mask_path), "--config", str(local), "--retrieval-config", str(retrieval), "--out", str(out)])
    payload = json.loads((out / "results.json").read_text())
    assert payload["parameters"]["semantic_weight"] == 0.5
    assert payload["results"][0]["clip_cosine"] is not None


def test_cli_pool_flags_are_independent():
    args = build_parser().parse_args(["--image", "x", "--x", "1", "--y", "2", "--out", "o", "--coarse-n", "7", "--hard-coarse-n", "9", "--workers", "2"])
    assert args.coarse_n == 7 and args.hard_coarse_n == 9 and args.workers == 2


def test_cli_explicit_missing_paths_config_raises(library, tmp_path):
    image_path = tmp_path / "query.jpg"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.zeros((80, 120, 3), np.uint8)).save(image_path)
    Image.fromarray((Library.load(library).mask(0) * 255).astype(np.uint8)).save(mask_path)
    with pytest.raises(FileNotFoundError):
        query_main([
            "--image", str(image_path), "--x", "60", "--y", "40", "--prepared",
            "--mask", str(mask_path), "--no-clip", "--config", str(tmp_path / "missing.toml"),
            "--out", str(tmp_path / "out"), "--library-root", str(library),
        ])


def test_framing_transform_maps_click_and_mask(monkeypatch):
    class Transform:
        crop = (20, 10, 180, 90)
        scale = 4.0

        def to_prepared(self, x, y):
            return (x - self.crop[0]) * self.scale, (y - self.crop[1]) * self.scale

    def prepare_image(image, long_side=640):
        assert long_side == 640
        x0, y0, x1, y1 = Transform.crop
        cropped = image[y0:y1, x0:x1]
        prepared = cv2.resize(cropped, (640, 320), interpolation=cv2.INTER_LINEAR)
        return prepared, Transform()

    module = types.ModuleType("matchcut_retriever.framing")
    module.prepare_image = prepare_image
    monkeypatch.setitem(sys.modules, "matchcut_retriever.framing", module)

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    original_mask = np.zeros((100, 200), dtype=bool)
    original_mask[30:40, 60:70] = True
    prepared, x, y, transform = _prepare(image, 65, 35)
    transformed = _transform_mask(original_mask, prepared.shape, transform)
    assert prepared.shape[:2] == (320, 640)
    assert (x, y) == (180.0, 100.0)
    ys, xs = np.nonzero(transformed)
    assert xs.min() >= 160 and xs.max() < 200
    assert ys.min() >= 80 and ys.max() < 120


def test_hard_axis_threshold_keeps_mild_elongation_and_rejects_strong(tmp_path):
    root = make_library(tmp_path / "axis-threshold", [
        ("query", 0, "ellipse", (100, 140), (70, 50), (24, 20), 0),
        ("mild", 0, "ellipse", (100, 140), (70, 50), (24, 20), 40),
    ])
    lib = Library.load(root)
    query = QueryObject.from_mask(np.zeros((100, 140, 3), dtype=np.uint8), lib.mask(0))
    results = rank(lib, query, "hard", params=RetrievalConfig(workers=1, hard_coarse_n=20), k=20)
    ids = {result.clip_id for result in results}
    assert "mild" in ids
    strong_root = make_library(tmp_path / "axis-threshold-strong", [
        ("query", 0, "ellipse", (100, 140), (70, 50), (36, 12), 0),
        ("strong", 0, "ellipse", (100, 140), (70, 50), (36, 12), 40),
    ])
    strong_lib = Library.load(strong_root)
    strong_query = QueryObject.from_mask(np.zeros((100, 140, 3), dtype=np.uint8), strong_lib.mask(0))
    strong_results = rank(strong_lib, strong_query, "hard", params=RetrievalConfig(workers=1, hard_coarse_n=20), k=20)
    assert "strong" not in {result.clip_id for result in strong_results}
