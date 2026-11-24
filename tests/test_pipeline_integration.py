"""Cross-card check: a library built by LibraryBuilder is loadable and queryable by retrieval."""

import json

import numpy as np
from PIL import Image

from matchcut_retriever.library import LibraryBuilder
from matchcut_retriever.query import main as query_main
from matchcut_retriever.retrieval import Library, QueryObject, rank
from matchcut_retriever.retrieval_config import load_retrieval_config

from test_library import FakeEmbedder, FakeSegmenter, _make_root


def _build(tmp_path):
    root = _make_root(tmp_path, clips=("a", "b", "c"), frames_per_clip=2)
    builder = LibraryBuilder(root, [FakeSegmenter("mask2former", "thing"), FakeSegmenter("sam2_auto")], FakeEmbedder())
    builder.process_clips(["a", "b", "c"])
    from matchcut_retriever.library import consolidate

    consolidate(root)
    return root


def _rect_query(h=80, w=100):
    # Same rectangle FakeSegmenter("sam2_auto") emits for every keyframe.
    image = np.full((h, w, 3), 128, np.uint8)
    mask = np.zeros((h, w), bool)
    mask[h // 2:h // 2 + h // 5, w // 2:w // 2 + w // 5] = True
    return image, mask


def test_built_library_ranks_in_both_modes(tmp_path):
    root = _build(tmp_path)
    library = Library.load(root)
    assert len(library.objects) > 0
    image, mask = _rect_query()
    query = QueryObject.from_mask(image, mask)
    config = load_retrieval_config()
    for mode in ("soft", "hard"):
        results = rank(library, query, mode, semantic_weight=0.0, k=5, params=config)
        assert results, mode
        # One result per (clip, shot); every clip has a single shot here.
        assert len({(r.clip_id, r.shot_no) for r in results}) == len(results)
        best = results[0]
        assert library.objects.iloc[best.object_id].segmenter == "sam2_auto"
        assert best.iou > 0.9


def test_query_cli_on_built_library(tmp_path):
    root = _build(tmp_path)
    image, mask = _rect_query()
    image_path, mask_path, out = tmp_path / "q.png", tmp_path / "m.png", tmp_path / "out"
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
    query_main([
        "--image", str(image_path), "--x", "55", "--y", "45", "--prepared", "--mask", str(mask_path),
        "--no-clip", "--mode", "hard", "--k", "3", "--library-root", str(root), "--out", str(out),
    ])
    payload = json.loads((out / "results.json").read_text())
    assert 0 < len(payload["results"]) <= 3
    assert (out / "grid.jpg").is_file()


def test_query_cli_warns_when_hard_mode_cannot_match_a_tiny_mask(tmp_path):
    root = _build(tmp_path)
    image = np.full((80, 100, 3), 128, np.uint8)
    mask = np.zeros((80, 100), bool)
    mask[40:42, 50:52] = True  # 0.05% of the frame, below the library's 0.5% floor
    image_path, mask_path, out = tmp_path / "q.png", tmp_path / "m.png", tmp_path / "out"
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
    query_main([
        "--image", str(image_path), "--x", "50", "--y", "40", "--prepared", "--mask", str(mask_path),
        "--no-clip", "--mode", "hard", "--k", "3", "--library-root", str(root), "--out", str(out),
    ])
    payload = json.loads((out / "results.json").read_text())
    assert payload["results"] == []
    assert any("hard scale limits" in message for message in payload["warnings"])
    assert any("no results" in message for message in payload["warnings"])
    assert payload["query"]["area_frac"] < 0.005


def test_query_warnings_quiet_for_a_normal_mask():
    from matchcut_retriever.query import query_warnings
    from matchcut_retriever.shape import HardLimits

    mask = np.zeros((80, 100), bool)
    mask[20:60, 30:70] = True
    assert query_warnings(mask, "hard", HardLimits(), 5) == []
    assert query_warnings(mask, "soft", HardLimits(), 5) == []


def test_library_with_zero_objects_loads(tmp_path):
    class EmptySegmenter(FakeSegmenter):
        def segment(self, images):
            return [[] for _ in images]

    from matchcut_retriever.library import consolidate

    root = _make_root(tmp_path, clips=("a",), frames_per_clip=1)
    LibraryBuilder(root, [EmptySegmenter("mask2former")], FakeEmbedder()).process_clips(["a"])
    consolidate(root)
    library = Library.load(root)
    assert len(library.objects) == 0
