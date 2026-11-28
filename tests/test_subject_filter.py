import json

import cv2
import numpy as np
import pandas as pd
import pytest

from matchcut_retriever.library import (
    LibraryBuilder, build_pca, build_subject_index, build_subject_scores,
    consolidate,
)
from matchcut_retriever.masks import rle_encode
from matchcut_retriever.retrieval import Library, QueryObject, rank
from matchcut_retriever.retrieval_config import RetrievalConfig
from matchcut_retriever.app import QueryEngine, build_app
from matchcut_retriever.selftest import run_selftest, sample_object_ids

from test_library import FakeEmbedder, FakeSegmenter, _make_root
from test_retrieval import make_library, query_for


class FakeSaliency:
    model_id = "fake/saliency"
    revision = "fake-revision"
    input_size = 64

    def __init__(self, value=0.75, fail_after=None):
        self.value = value
        self.fail_after = fail_after
        self.calls = 0

    def predict(self, images):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("fake saliency failure")
        return [np.full(image.shape[:2], self.value, np.float32) for image in images]


class EmptySegmenter(FakeSegmenter):
    def segment(self, images):
        self.calls += 1
        return [[] for _ in images]


def _build_parts(tmp_path, clips=("a", "b"), frames_per_clip=1):
    root = _make_root(tmp_path, clips=clips, frames_per_clip=frames_per_clip)
    LibraryBuilder(root, [FakeSegmenter("mask2former")], FakeEmbedder()).process_clips(list(clips))
    return root


def test_subject_score_math_and_empty_part(tmp_path):
    root = _build_parts(tmp_path / "filled", clips=("a",))
    model = FakeSaliency(0.73)
    build_subject_scores(root, model=model)
    scores = np.load(root / "parts/a/subject.npy")
    assert len(scores) > 0 and np.allclose(scores, 0.73, atol=1e-3)

    # A row with an empty mask is retained by the synthetic fixture to check
    # the stage's explicit zero score handling.
    part = root / "parts/a"
    rows = pd.read_parquet(part / "objects.parquet")
    empty_encoded = rle_encode(np.zeros((80, 100), bool))
    rows.loc[:, "mask_nbytes"] = len(empty_encoded)
    rows.to_parquet(part / "objects.parquet", index=False)
    (part / "masks.bin").write_bytes(empty_encoded)
    (part / "subject.npy").unlink()
    build_subject_scores(root, model=FakeSaliency(0.73))
    assert np.all(np.load(part / "subject.npy") == 0)

    empty = _make_root(tmp_path / "empty", clips=("z",), frames_per_clip=1)
    LibraryBuilder(empty, [EmptySegmenter("empty")], FakeEmbedder()).process_clips(["z"])
    build_subject_scores(empty, model=FakeSaliency())
    assert np.load(empty / "parts/z/subject.npy").shape == (0,)

    mask = np.array([[True, False], [False, True]])
    sal = np.array([[0.2, 0.9], [0.6, 0.1]], np.float32)
    assert np.isclose(float(sal[mask].mean()), 0.15)


def test_saliency_outputs_skip_shard_failure_and_inode_safety(tmp_path):
    root = _build_parts(tmp_path / "main", clips=("a", "b", "c"), frames_per_clip=1)
    model = FakeSaliency(0.8)
    result = build_subject_scores(root, model=model, shard=(1, 2))
    assert result["clips"] == 1
    assert not (root / "parts/a/subject.npy").exists()
    assert (root / "parts/b/subject.npy").exists()
    assert not (root / "parts/c/subject.npy").exists()
    npz = np.load(root / "parts/b/saliency.npz")
    assert npz.files and all(npz[k].dtype == np.uint8 and max(npz[k].shape) <= 160 for k in npz.files)
    metadata = json.loads((root / "parts/b/saliency.json").read_text())
    assert metadata["version"] == "s1" and metadata["score"] == "mask_mean"

    before_calls = model.calls
    build_subject_scores(root, model=model, shard=(1, 2))
    assert model.calls == before_calls

    objects = root / "parts/b/objects.parquet"
    masks = root / "parts/b/masks.bin"
    object_inode, masks_inode = objects.stat().st_ino, masks.stat().st_ino
    assert objects.stat().st_ino == object_inode and masks.stat().st_ino == masks_inode

    failing_root = _build_parts(tmp_path / "failure", clips=("a", "b"), frames_per_clip=1)
    with pytest.raises(RuntimeError, match="fake saliency failure"):
        build_subject_scores(failing_root, model=FakeSaliency(fail_after=0))
    assert not list((failing_root / "parts").glob(".*/"))
    assert not (failing_root / "parts/a/subject.npy").exists()
    assert not (failing_root / "parts/a/saliency.npz").exists()


def test_subject_index_alignment_missing_and_reconsolidate(tmp_path):
    root = _build_parts(tmp_path, clips=("a", "b"), frames_per_clip=1)
    build_subject_scores(root, model=FakeSaliency(0.6))
    first = consolidate(root)
    info = build_subject_index(root)
    assert info["n_objects"] == len(pd.read_parquet(root / "objects.parquet"))
    objects = pd.read_parquet(root / "objects.parquet")
    values = np.load(root / "index/subject.npy")
    expected = []
    for row in objects.itertuples():
        expected.append(float(np.load(root / "parts" / row.clip_id / "subject.npy")[0]))
    assert np.allclose(values, expected)

    meta_path = root / "index/meta.json"
    meta = json.loads(meta_path.read_text()); meta["generation"] = "stale-generation"; meta_path.write_text(json.dumps(meta))
    assert Library.load(root).subject_status == "stale"
    meta["generation"] = first["generation"]; meta_path.write_text(json.dumps(meta))
    (root / "parts/a/subject.npy").unlink()
    with pytest.raises(ValueError, match="1 parts lack"):
        build_subject_index(root)
    build_subject_scores(root, model=FakeSaliency(0.6))
    rebuilt = consolidate(root)
    loaded = Library.load(root)
    assert rebuilt["generation"] != first["generation"]
    assert loaded.subject_status == "valid"


def test_library_subject_statuses(tmp_path):
    root = make_library(tmp_path / "lib", [("a", 0, "disc"), ("b", 0, "square")])
    lib = Library.load(root)
    assert lib.subject_status == "absent"
    meta_path = root / "index/meta.json"; meta = json.loads(meta_path.read_text()); meta["generation"] = "g"; meta_path.write_text(json.dumps(meta))
    np.save(root / "index/subject.npy", np.array([0.8, 0.2], np.float16))
    (root / "index/subject.json").write_text(json.dumps({"generation": "g", "n_objects": 2}))
    assert Library.load(root).subject_status == "valid"
    np.save(root / "index/subject.npy", np.array([0.8], np.float16))
    assert Library.load(root).subject_status == "stale"
    (root / "index/subject.npy").unlink()
    assert Library.load(root).subject_status == "stale"


@pytest.mark.parametrize("mode", ["soft", "hard"])
@pytest.mark.parametrize("coarse_method", ["exact", "pca"])
def test_rank_subject_filter_before_shortlist(tmp_path, mode, coarse_method):
    root = make_library(tmp_path / f"{mode}-{coarse_method}", [("bad", 0, "disc"), ("good", 0, "square")])
    meta_path = root / "index/meta.json"; meta = json.loads(meta_path.read_text())
    meta.update(generation="g", clips=["bad", "good"], bytes={
        "objects.parquet": (root / "objects.parquet").stat().st_size,
        "index/masks.bin": (root / "index/masks.bin").stat().st_size,
        "index/shape_fp.npy": (root / "index/shape_fp.npy").stat().st_size,
        "index/clip_emb.npy": (root / "index/clip_emb.npy").stat().st_size,
    }, clip_dim=2, n_objects=2)
    meta_path.write_text(json.dumps(meta))
    np.save(root / "index/subject.npy", np.array([0.1, 0.9], np.float16))
    (root / "index/subject.json").write_text(json.dumps({"generation": "g", "n_objects": 2}))
    if coarse_method == "pca":
        build_pca(root, dim=2)
    lib, query = query_for(root, 0)
    cfg = RetrievalConfig(workers=1, coarse_n=1, hard_coarse_n=1, coarse_method=coarse_method, min_subject_score=0.5)
    results = rank(lib, query, mode, params=cfg, k=1)
    assert results and results[0].object_id == 1 and results.subject_filter == "applied"
    off = rank(lib, query, mode, params=RetrievalConfig(workers=1, coarse_n=1, hard_coarse_n=1, coarse_method=coarse_method, min_subject_score=0), k=1)
    assert off and off[0].object_id == 0
    assert off[0].to_dict()["subject_score"] == pytest.approx(0.1, abs=0.01)


def test_rank_unavailable_is_unfiltered(tmp_path):
    root = make_library(tmp_path, [("best", 0, "disc"), ("other", 0, "square")])
    lib, query = query_for(root, 0)
    results = rank(lib, query, "soft", params=RetrievalConfig(workers=1, coarse_n=1, min_subject_score=0.5), k=1)
    assert results[0].object_id == 0 and results.subject_filter.startswith("unavailable")


def test_app_search_passes_subject_threshold_and_blocks_build(tmp_path, monkeypatch):
    root = _build_parts(tmp_path, clips=("a",), frames_per_clip=1)
    consolidate(root)
    engine = QueryEngine(library_root=root, sam=object(), clip=object())
    image = np.full((80, 100, 3), 128, np.uint8)
    mask = np.zeros((80, 100), bool); mask[20:50, 30:60] = True
    captured = {}

    def fake_rank(library, query, mode, semantic_weight=None, k=None, params=None):
        captured["threshold"] = params.min_subject_score
        return []

    monkeypatch.setattr("matchcut_retriever.app.rank", fake_rank)
    engine.search(image, mask, min_subject=0.85)
    assert captured["threshold"] == pytest.approx(0.85)
    pytest.importorskip("gradio")
    assert build_app(engine) is not None


def test_selftest_samples_only_eligible_subjects_and_records_pool(tmp_path):
    root = _build_parts(tmp_path, clips=("a", "b"), frames_per_clip=1)
    build_subject_scores(root, model=FakeSaliency(0.8))
    consolidate(root)
    out = tmp_path / "selftest"
    summary = run_selftest(root, out, n=2, seed=1, mask_source="gt", modes=("soft",), k_max=2, device="cpu", workers=1)
    assert summary["min_subject_score"] == pytest.approx(0.5)
    assert summary["eligible_pool_size"] == len(pd.read_parquet(root / "objects.parquet"))
    records = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
    scores = np.load(root / "index/subject.npy")
    assert all(scores[int(record["object_id"])] >= 0.5 for record in records)

    absent = Library.load(root)
    (root / "index/subject.npy").unlink(); (root / "index/subject.json").unlink()
    assert sample_object_ids(absent.objects, n=4, seed=2) == sample_object_ids(absent.objects, n=4, seed=2)
