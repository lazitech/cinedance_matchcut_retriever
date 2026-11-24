import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from matchcut_retriever.library import LibraryBuilder, consolidate, refingerprint
from matchcut_retriever.masks import rle_decode
from matchcut_retriever.retrieval import Library
from matchcut_retriever.shape import FINGERPRINT_DIM, FINGERPRINT_VERSION, object_geometry, shape_fingerprint

from test_library import FakeEmbedder, FakeSegmenter, _make_root


def _r4_fingerprint(mask):
    geometry = object_geometry(mask)
    scale = max(math.sqrt(geometry.area), 1e-6)
    factor = 31.0 / (4.0 * scale)
    affine = np.array([
        [factor, 0.0, -(geometry.cx - 2.0 * scale) * factor],
        [0.0, factor, -(geometry.cy - 2.0 * scale) * factor],
    ], dtype=np.float32)
    raster = cv2.warpAffine(mask.astype(np.float32), affine, (32, 32), flags=cv2.INTER_AREA,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    raster = cv2.GaussianBlur(raster, (3, 3), 0)
    yy, xx = np.mgrid[:32, :32]
    radius = np.sqrt((xx - 15.5) ** 2 + (yy - 15.5) ** 2)
    raster *= (radius / 16.0) ** 4
    vector = raster.reshape(-1).astype(np.float32)
    vector /= np.linalg.norm(vector)
    return vector


def _make_legacy_parts(tmp_path):
    root = _make_root(tmp_path, clips=("a", "b"), frames_per_clip=1)
    LibraryBuilder(root, [FakeSegmenter("mask2former")], FakeEmbedder()).process_clips(["a", "b"])
    for part_dir in (root / "parts").iterdir():
        metadata = json.loads((part_dir / "meta.json").read_text())
        metadata.pop("fingerprint_version")
        (part_dir / "meta.json").write_text(json.dumps(metadata))
        objects = pd.read_parquet(part_dir / "objects.parquet")
        blob = (part_dir / "masks.bin").read_bytes()
        fingerprints = []
        offset = 0
        for length in objects["mask_nbytes"]:
            length = int(length)
            fingerprints.append(_r4_fingerprint(rle_decode(blob[offset:offset + length])))
            offset += length
        np.save(part_dir / "shape_fp.npy", np.asarray(fingerprints, dtype=np.float16))
    return root


def test_refingerprint_migrates_legacy_parts_and_is_idempotent(tmp_path):
    root = _make_legacy_parts(tmp_path)
    with pytest.raises(ValueError, match="refingerprint"):
        consolidate(root)
    before = {
        part.name: {
            name: (part / name).read_bytes()
            for name in ("objects.parquet", "masks.bin", "clip_emb.npy")
        }
        for part in (root / "parts").iterdir()
    }
    result = refingerprint(root, workers=2)
    assert result["rewritten_parts"] == 2
    meta = json.loads((root / "index" / "meta.json").read_text())
    assert meta["fingerprint_version"] == FINGERPRINT_VERSION
    library = Library.load(root)
    for object_id in range(len(library.objects)):
        assert np.allclose(library.shape_fp[object_id], shape_fingerprint(library.mask(object_id)), atol=2e-3)
    after = {
        part.name: {
            name: (part / name).read_bytes()
            for name in ("objects.parquet", "masks.bin", "clip_emb.npy")
        }
        for part in (root / "parts").iterdir()
    }
    assert before == after
    mtimes = {part.name: (part / "shape_fp.npy").stat().st_mtime_ns for part in (root / "parts").iterdir()}
    second = refingerprint(root, workers=2)
    assert second["rewritten_parts"] == 0 and second["skipped_parts"] == 2
    assert mtimes == {part.name: (part / "shape_fp.npy").stat().st_mtime_ns for part in (root / "parts").iterdir()}


def test_consolidate_rejects_mixed_versions(tmp_path):
    root = _make_legacy_parts(tmp_path)
    first = next((root / "parts").iterdir())
    metadata = json.loads((first / "meta.json").read_text())
    metadata["fingerprint_version"] = FINGERPRINT_VERSION
    (first / "meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="refingerprint"):
        consolidate(root)


def test_interrupted_refingerprint_temp_is_not_a_part(tmp_path):
    import shutil

    root = _make_legacy_parts(tmp_path)
    # Simulate a crash after the temporary copy was written but before the swap.
    shutil.copytree(root / "parts" / "a", root / "parts" / ".a.refingerprint-crashed")
    result = refingerprint(root, workers=1)
    assert result["n_clips"] == 2 and result["rewritten_parts"] == 2
    assert not list((root / "parts").glob(".*"))
    library = Library.load(root)
    assert set(library.objects.clip_id) == {"a", "b"}
    assert len(library.objects) == len(library.objects.drop_duplicates(["keyframe_id", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]))


def test_verify_library_rejects_a_consistent_legacy_index(tmp_path):
    from matchcut_retriever.library import verify_library

    root = _make_root(tmp_path, clips=("a",), frames_per_clip=1)
    LibraryBuilder(root, [FakeSegmenter("mask2former")], FakeEmbedder()).process_clips(["a"])
    consolidate(root)
    verify_library(root)
    meta_path = root / "index" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["fingerprint_version"] = "r4"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="refingerprint"):
        verify_library(root)


def test_empty_current_library_loads(tmp_path):
    class EmptySegmenter(FakeSegmenter):
        def segment(self, images):
            return [[] for _ in images]

    root = _make_root(tmp_path, clips=("empty",), frames_per_clip=1)
    LibraryBuilder(root, [EmptySegmenter("mask2former")], FakeEmbedder()).process_clips(["empty"])
    consolidate(root)
    library = Library.load(root)
    assert len(library.objects) == 0
    assert library.shape_fp.shape == (0, FINGERPRINT_DIM)
