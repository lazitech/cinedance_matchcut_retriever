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
