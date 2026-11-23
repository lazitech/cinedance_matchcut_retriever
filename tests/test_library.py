import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from matchcut_retriever.library import (
    LibraryBuilder,
    build_objects,
    clean_mask,
    consolidate,
    merge_detections,
    neighbours,
    verify_library,
)
from matchcut_retriever.masks import rle_decode
from matchcut_retriever.models import Detection
from matchcut_retriever.shape import FINGERPRINT_DIM


class FakeSegmenter:
    def __init__(self, source, label=None):
        self.name = source
        self.model_id = f"fake/{source}"
        self.source = source
        self.label = label
        self.calls = 0

    def segment(self, images):
        self.calls += 1
        result = []
        for image in images:
            h, w = image.shape[:2]
            masks = []
            disc = np.zeros((h, w), np.uint8)
            cv2.circle(disc, (w // 4, h // 2), max(4, min(h, w) // 8), 1, thickness=-1)
            masks.append(Detection(disc.astype(bool), 0.9, self.label, self.source))
            if self.source == "sam2_auto":
                part = np.zeros((h, w), bool)
                part[h // 2:h // 2 + h // 5, w // 2:w // 2 + w // 5] = True
                masks.append(Detection(part, 0.8, None, self.source))
                tiny = np.zeros((h, w), bool)
                tiny[0, 0] = True
                masks.append(Detection(tiny, 1.0, None, self.source))
            result.append(masks)
        return result


class FakeEmbedder:
    name = "fake-clip"
    model_id = "fake/clip"

    def embed_masked(self, images, masks):
        values = []
        for mask in masks:
            v = np.array([float(mask.sum()), 1.0, 2.0, 3.0], np.float32)
            values.append(v / np.linalg.norm(v))
        return np.asarray(values, np.float32).reshape(len(masks), 4)


class FailingEmbedder(FakeEmbedder):
    def __init__(self):
        self.failed = False

    def embed_masked(self, images, masks):
        if self.failed:
            raise RuntimeError("injected crash")
        self.failed = True
        return super().embed_masked(images, masks)


def _make_root(tmp_path, clips=("a", "b"), frames_per_clip=2):
    root = tmp_path / "library"
    rows = []
    for ci, clip in enumerate(clips):
        for k in range(frames_per_clip):
            directory = root / "keyframes" / clip
            directory.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(ci * 100 + k)
            image = rng.integers(30, 220, (80, 100, 3), dtype=np.uint8)
            rel = f"keyframes/{clip}/001_{k}.jpg"
            cv2.imwrite(str(root / rel), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            rows.append(dict(keyframe_id=f"{clip}:1:{k}", clip_id=clip, shot_no=1,
                             k=k, t_s=float(k), frame_idx=k, path=rel, width=100, height=80))
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(root / "keyframes.parquet", index=False)
    (root / "clips.txt").write_text("".join(f"{c}\n" for c in clips))
    return root


def test_clean_mask_components_holes_and_empty():
    mask = np.zeros((50, 50), bool)
    mask[5:25, 5:25] = True
    mask[12, 12] = False
    mask[0, 0] = True
    cleaned = clean_mask(mask)
    assert cleaned is not None and cleaned[12, 12] and not cleaned[0, 0]
    two = np.zeros((50, 50), bool)
    two[3:18, 3:18] = True
    two[30:45, 30:45] = True
    result = clean_mask(two)
    assert result is not None
    assert cv2.connectedComponents(result.astype(np.uint8), 8)[0] - 1 == 2
    specks = np.zeros((30, 30), bool)
    specks[::3, ::3] = True
    assert clean_mask(specks) is None


def test_merge_detections_preserves_primary_and_drops_duplicate():
    p = np.zeros((20, 20), bool); p[2:12, 2:12] = 1
    dup = np.zeros((20, 20), bool); dup[2:12, 2:12] = 1
    part = np.zeros((20, 20), bool); part[2:7, 2:12] = 1
    disjoint = np.zeros((20, 20), bool); disjoint[14:19, 14:19] = 1
    primary = [Detection(p, .9, "person", "mask2former")]
    merged = merge_detections(primary, [Detection(dup, .8, None, "sam2_auto"), Detection(part, .7, None, "sam2_auto"), Detection(disjoint, .6, None, "sam2_auto")])
    assert len(merged) == 3
    assert merged[0].label == "person" and merged[0].source == "mask2former"


def test_build_objects_filters_area_and_limits():
    image = np.zeros((100, 100, 3), np.uint8)
    masks = []
    for x, size in [(5, 10), (30, 20), (60, 30), (0, 90)]:
        m = np.zeros((100, 100), bool); m[x:x + size, 10:10 + size] = True
        masks.append(Detection(m, 1.0, None, "sam2_auto"))
    rows, cleaned = build_objects(image, masks, max_objects=2)
    assert len(rows) == len(cleaned) == 2
    assert rows[0]["area"] >= rows[1]["area"]
    assert all(r["segmenter"] == "sam2_auto" for r in rows)


def test_end_to_end_format_and_incremental_build(tmp_path):
    root = _make_root(tmp_path)
    builder = LibraryBuilder(root, [FakeSegmenter("mask2former", "thing"), FakeSegmenter("sam2_auto")], FakeEmbedder(), max_objects=4, batch_size=2)
    summary = builder.process_clips(["a", "b"])
    assert summary["clips"] == 2 and summary["objects"] > 0
    summary = consolidate(root)
    objects = pd.read_parquet(root / "objects.parquet")
    assert objects.columns.tolist() == ["object_id", "keyframe_id", "clip_id", "shot_no", "k", "t_s", "frame_w", "frame_h", "cx", "cy", "area", "area_frac", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "angle", "eccentricity", "n_components", "solidity", "score", "label", "segmenter", "mask_offset", "mask_nbytes"]
    assert objects.object_id.tolist() == list(range(len(objects)))
    fp = np.load(root / "index/shape_fp.npy"); emb = np.load(root / "index/clip_emb.npy")
    assert fp.dtype == np.float16 and fp.shape == (len(objects), FINGERPRINT_DIM)
    assert emb.dtype == np.float16 and emb.shape[0] == len(objects)
    assert np.allclose(np.linalg.norm(fp.astype(np.float32), axis=1), 1, atol=.02)
    assert np.allclose(np.linalg.norm(emb.astype(np.float32), axis=1), 1, atol=.02)
    expected_dtypes = {
        "object_id": "int64", "shot_no": "int32", "k": "int8", "t_s": "float64",
        "frame_w": "int32", "frame_h": "int32", "cx": "float32", "cy": "float32",
        "area": "float32", "area_frac": "float32", "bbox_x0": "int32", "bbox_y0": "int32",
        "bbox_x1": "int32", "bbox_y1": "int32", "angle": "float32", "eccentricity": "float32",
        "n_components": "int16", "solidity": "float32", "score": "float32",
        "mask_offset": "int64", "mask_nbytes": "int32",
    }
    assert {key: str(objects[key].dtype) for key in expected_dtypes} == expected_dtypes
    blob = (root / "index/masks.bin").read_bytes()
    for row in objects.itertuples():
        mask = rle_decode(blob[row.mask_offset:row.mask_offset + row.mask_nbytes])
        assert mask.shape == (row.frame_h, row.frame_w)
        assert int(mask.sum()) == int(row.area)
    meta = json.loads((root / "index/meta.json").read_text())
    assert {"n_objects", "segmenter", "segmenter_params", "clip_model", "clip_dim", "fingerprint_dim", "fingerprint_version", "built_at", "clips"} <= meta.keys()
    assert meta["n_objects"] == len(objects)
    assert summary["n_objects"] == len(objects)
    assert summary["generation"] == meta["generation"]
    assert summary["bytes"] == meta["bytes"]
    assert "fake/clip" in meta["clip_model"]
    assert "fake/mask2former" in meta["segmenter"]
    assert builder.process_clips(["a", "b"])["clips"] == 0


def test_keyframe_processed_marker_does_not_skip_objects(tmp_path):
    root = _make_root(tmp_path, clips=("a",))
    (root / "processed_clips.txt").write_text("a\n")
    builder = LibraryBuilder(root, [FakeSegmenter("mask2former")], FakeEmbedder())
    result = builder.process_clips(["a"])
    assert result["clips"] == 1
    assert (root / "objects_processed.txt").read_text().strip() == "a"


def test_shards_and_orphan_recovery(tmp_path):
    root = _make_root(tmp_path, clips=("a", "b", "c", "d"), frames_per_clip=1)
    s1, s2 = FakeSegmenter("mask2former"), FakeSegmenter("mask2former")
    b1 = LibraryBuilder(root, [s1], FakeEmbedder()); b2 = LibraryBuilder(root, [s2], FakeEmbedder())
    b1.process_clips(["a", "c"])
    b2.process_clips(["b", "d"])
    consolidate(root)
    assert sorted((root / "objects_processed.txt").read_text().split()) == ["a", "b", "c", "d"]
    calls = s1.calls + s2.calls
    b1.process_clips(["a", "b", "c", "d"])
    assert s1.calls + s2.calls == calls

    fresh = _make_root(tmp_path / "fresh", clips=("a", "b", "c", "d"), frames_per_clip=1)
    LibraryBuilder(fresh, [FakeSegmenter("mask2former")], FakeEmbedder()).process_clips(["a", "b", "c", "d"])
    consolidate(fresh)
    assert pd.read_parquet(root / "objects.parquet").equals(pd.read_parquet(fresh / "objects.parquet"))
    assert np.array_equal(np.load(root / "index/shape_fp.npy"), np.load(fresh / "index/shape_fp.npy"))
    assert np.array_equal(np.load(root / "index/clip_emb.npy"), np.load(fresh / "index/clip_emb.npy"))


def test_neighbours_writes_sheets(tmp_path):
    root = _make_root(tmp_path, clips=("a", "b", "c"), frames_per_clip=1)
    LibraryBuilder(root, [FakeSegmenter("mask2former")], FakeEmbedder()).process_clips(["a", "b", "c"])
    consolidate(root)
    output = neighbours(root, n=2, k=1, seed=0, out=tmp_path / "sheets")
    assert len(output) == 2 and all(path.is_file() for path in output)


def test_crash_leaves_no_half_written_next_part_and_rerun_completes(tmp_path):
    root = _make_root(tmp_path, clips=("a", "b"), frames_per_clip=1)
    failing = FailingEmbedder()
    builder = LibraryBuilder(root, [FakeSegmenter("mask2former")], failing)
    builder.process_clips(["a"])
    # A fresh failing embedder crashes while processing b; a complete a part
    # remains and no b part or marker is published.
    failing.failed = True
    try:
        builder.process_clips(["b"])
    except RuntimeError:
        pass
    assert not (root / "parts/b").exists()
    assert "b" not in (root / "objects_processed.txt").read_text() if (root / "objects_processed.txt").exists() else True
    builder = LibraryBuilder(root, [FakeSegmenter("mask2former")], FakeEmbedder())
    assert builder.process_clips(["b"])["clips"] == 1


def test_incremental_new_clip_matches_fresh_build(tmp_path):
    incremental = _make_root(tmp_path / "inc", clips=("a",), frames_per_clip=1)
    builder = LibraryBuilder(incremental, [FakeSegmenter("mask2former")], FakeEmbedder())
    builder.process_clips(["a"])
    # Add a new keyframe/clip and update the source list.
    fresh_extra = _make_root(tmp_path / "fresh", clips=("a", "b"), frames_per_clip=1)
    (incremental / "keyframes/b").mkdir(parents=True)
    image = np.random.default_rng(100).integers(30, 220, (80, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(incremental / "keyframes/b/001_0.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    extra = pd.DataFrame([dict(keyframe_id="b:1:0", clip_id="b", shot_no=1, k=0, t_s=0.0, frame_idx=0, path="keyframes/b/001_0.jpg", width=100, height=80)])
    pd.concat([pd.read_parquet(incremental / "keyframes.parquet"), extra], ignore_index=True).to_parquet(incremental / "keyframes.parquet", index=False)
    (incremental / "clips.txt").write_text("a\nb\n")
    builder.process_clips(["b"])
    consolidate(incremental)
    fresh_builder = LibraryBuilder(fresh_extra, [FakeSegmenter("mask2former")], FakeEmbedder())
    fresh_builder.process_clips(["a", "b"])
    consolidate(fresh_extra)
    assert pd.read_parquet(incremental / "objects.parquet").equals(pd.read_parquet(fresh_extra / "objects.parquet"))
    assert np.array_equal(np.load(incremental / "index/shape_fp.npy"), np.load(fresh_extra / "index/shape_fp.npy"))


def test_orphan_part_is_consolidated_without_resegmentation(tmp_path):
    root = _make_root(tmp_path, clips=("a",))
    segmenter = FakeSegmenter("mask2former")
    LibraryBuilder(root, [segmenter], FakeEmbedder()).process_clips(["a"])
    calls = segmenter.calls
    (root / "objects_processed.txt").unlink()
    consolidate(root)
    objects = pd.read_parquet(root / "objects.parquet")
    assert len(objects) > 0
    assert segmenter.calls == calls
    assert (root / "objects_processed.txt").read_text().strip() == "a"


def test_progress_log_reports_clip_metrics(tmp_path, caplog):
    import logging
    root = _make_root(tmp_path, clips=("a",), frames_per_clip=1)
    with caplog.at_level(logging.INFO, logger="matchcut_retriever.library"):
        LibraryBuilder(root, [FakeSegmenter("mask2former")], FakeEmbedder()).process_clips(["a"])
    assert "clips 1/1" in caplog.text
    assert "objects/keyframe" in caplog.text


def test_verify_detects_corruption_and_consolidate_repairs(tmp_path):
    root = _make_root(tmp_path, clips=("a", "b"), frames_per_clip=1)
    builder = LibraryBuilder(root, [FakeSegmenter("mask2former")], FakeEmbedder())
    builder.process_clips(["a", "b"])
    first = consolidate(root)
    clip_path = root / "index" / "clip_emb.npy"
    original = clip_path.read_bytes()
    clip_path.write_bytes(original[:-64])
    with pytest.raises(RuntimeError, match="library files are inconsistent"):
        verify_library(root)
    repaired = consolidate(root)
    verify_library(root)
    assert repaired["generation"] != first["generation"]
