"""Focused tests for the human evaluation workflow."""

import hashlib
import json

import numpy as np
from PIL import Image

from matchcut_retriever.human_eval import (
    build_annotation_app,
    load_labels,
    pool_results,
    prepare_queries,
    report_results,
    save_labels,
)
from matchcut_retriever.library import LibraryBuilder, consolidate
from test_library import FakeEmbedder, FakeSegmenter, _make_root


class PointSegmenter:
    """Small deterministic point segmenter for prepare tests."""

    def __init__(self):
        self.calls = 0

    def masks_at_point(self, image, x, y):
        self.calls += 1
        h, w = image.shape[:2]
        masks = []
        for offset in (0, 1):
            mask = np.zeros((h, w), dtype=bool)
            y0 = min(h - 10, max(0, int(round(y)) - 10 + offset * 3))
            x0 = min(w - 12, max(0, int(round(x)) - 12 + offset * 3))
            mask[y0:y0 + 10, x0:x0 + 12] = True
            masks.append(type("Detection", (), {"mask": mask, "score": 1.0})())
        return masks


def _seed_output(tmp_path):
    out = tmp_path / "eval"
    (out / "queries" / "q1").mkdir(parents=True)
    (out / "panels" / "q1").mkdir(parents=True)
    Image.fromarray(np.zeros((12, 16, 3), np.uint8)).save(out / "queries" / "q1" / "prepared.png")
    for item_id in ("a", "b", "c"):
        Image.fromarray(np.zeros((20, 20, 3), np.uint8)).save(out / "panels" / "q1" / f"{item_id}.jpg")
    (out / "queries.json").write_text(json.dumps({"queries": [{"id": "q1", "group": "campus", "note": "tower"}]}))
    def item_id(clip, object_id):
        return hashlib.sha1(f"q1|{clip}|1|{object_id}".encode()).hexdigest()[:16]
    ids = [item_id("c1", 1), item_id("c2", 2), item_id("c3", 3)]
    (out / "items.json").write_text(json.dumps({"queries": {"q1": [
        {"item_id": ids[0], "panel": "panels/q1/a.jpg", "clip_id": "c1", "shot_no": 1, "object_id": 1, "start_s": 0, "end_s": 1, "keyframe_id": "c1:1:0", "t_s": 0},
        {"item_id": ids[1], "panel": "panels/q1/b.jpg", "clip_id": "c2", "shot_no": 1, "object_id": 2, "start_s": 0, "end_s": 1, "keyframe_id": "c2:1:0", "t_s": 0},
        {"item_id": ids[2], "panel": "panels/q1/c.jpg", "clip_id": "c3", "shot_no": 1, "object_id": 3, "start_s": 0, "end_s": 1, "keyframe_id": "c3:1:0", "t_s": 0},
    ]}}))
    (out / "rankings.json").write_text(json.dumps({"q1": {
        "soft": {"results": [{"clip_id": "c1", "shot_no": 1, "object_id": 1}, {"clip_id": "c2", "shot_no": 1, "object_id": 2}]},
        "hard": {"results": [{"clip_id": "c2", "shot_no": 1, "object_id": 2}]},
    }}))
    return out, ids


def test_label_save_load_roundtrip_and_unticked_items(tmp_path):
    out, ids = _seed_output(tmp_path)
    saved = save_labels(out, "q1", [ids[0]], ids, "usable tower")
    assert saved["usable"] == [ids[0]]
    assert load_labels(out)["q1"]["judged"] == ids
    report = report_results(out, k=2)
    assert report["modes"]["soft"]["P@K"] == 0.5
    assert report["modes"]["hard"]["Hit@K"] == 0.0
    assert report["unjudged"] == []


def test_annotation_blocks_builds_without_launching(tmp_path):
    out, _ = _seed_output(tmp_path)
    app = build_annotation_app(out)
    assert app is not None
    assert str(out) in str(getattr(app, "matchcut_allowed_paths", []))


def test_prepare_incremental_and_metadata_refresh(tmp_path):
    image_path = tmp_path / "query.png"
    Image.fromarray(np.full((80, 100, 3), 128, dtype=np.uint8)).save(image_path)
    spec = tmp_path / "queries.toml"
    spec.write_text(
        '[[query]]\n'
        'id = "q1"\nimage = "query.png"\nx = 50\ny = 40\n'
        'mask_index = 0\ngroup = "campus"\nnote = "first"\nexclude_clips = ["a"]\n'
    )
    out = tmp_path / "eval"
    segmenter = PointSegmenter()
    first = prepare_queries(spec, out, segmenter=segmenter)
    assert segmenter.calls == 1 and first[0]["mask_index"] == 0
    second = prepare_queries(spec, out, segmenter=segmenter)
    assert segmenter.calls == 1, "unchanged inputs should reuse prepared artifacts"
    assert second[0]["note"] == "first"
    spec.write_text(
        '[[query]]\n'
        'id = "q1"\nimage = "query.png"\nx = 50\ny = 40\n'
        'mask_index = 1\ngroup = "movie"\nnote = "updated"\nexclude_clips = ["b"]\n'
    )
    third = prepare_queries(spec, out, segmenter=segmenter)
    assert segmenter.calls == 2
    assert third[0]["mask_index"] == 1
    assert third[0]["group"] == "movie" and third[0]["note"] == "updated"
    assert third[0]["exclude_clips"] == ["b"]
    spec.write_text(spec.read_text().replace("mask_index = 1", "mask_index = 9"))
    try:
        prepare_queries(spec, out, segmenter=segmenter)
    except IndexError as exc:
        assert "out of range" in str(exc)
    else:
        raise AssertionError("invalid mask_index should fail clearly")


def test_pool_pipeline_deduplicates_and_is_stable(tmp_path):
    root = _make_root(tmp_path, clips=("a", "b", "c"), frames_per_clip=1)
    LibraryBuilder(root, [FakeSegmenter("sam2_auto")], FakeEmbedder()).process_clips(["a", "b", "c"])
    consolidate(root)
    image_path = tmp_path / "query.png"
    Image.fromarray(np.full((80, 100, 3), 128, dtype=np.uint8)).save(image_path)
    spec = tmp_path / "queries.toml"
    spec.write_text('[[query]]\nid="q1"\nimage="query.png"\nx=55\ny=45\nexclude_clips=["a"]\n')
    out = tmp_path / "eval"
    prepare_queries(spec, out, segmenter=PointSegmenter())
    pool_results(out, library_root=root, k=2, modes=("soft", "hard"), seed=7)
    first = json.loads((out / "items.json").read_text())
    entries = first["queries"]["q1"]
    assert entries
    assert all(item["clip_id"] != "a" for item in entries)
    assert all("object_id" in item for item in entries)
    assert all(not ({"mode", "rank", "score"} & set(item)) for item in entries)
    keys = {(item["clip_id"], item["shot_no"], item["object_id"]) for item in entries}
    assert len(keys) == len(entries)
    pool_results(out, library_root=root, k=2, modes=("soft", "hard"), seed=7)
    assert json.loads((out / "items.json").read_text())["queries"]["q1"] == entries


def test_report_lists_unlabeled_queries_and_unjudged_items(tmp_path):
    out, ids = _seed_output(tmp_path)
    queries = json.loads((out / "queries.json").read_text())["queries"]
    queries.append({"id": "q2", "group": "movie", "note": "second"})
    (out / "queries.json").write_text(json.dumps({"queries": queries}))
    items = json.loads((out / "items.json").read_text())["queries"]
    items["q2"] = [{"item_id": "q2-item", "clip_id": "c9", "shot_no": 1, "object_id": 9}]
    (out / "items.json").write_text(json.dumps({"queries": items}))
    save_labels(out, "q1", [ids[0]], [ids[0]])
    report = report_results(out, k=2)
    assert report["unlabeled"] == ["q2"]
    assert "q2-item" in report["unjudged"]
    assert "Unlabeled queries" in (out / "report.md").read_text()

