"""Focused tests for the human evaluation workflow."""

import hashlib
import json

import numpy as np
from PIL import Image

from matchcut_retriever.human_eval import (
    build_annotation_app,
    load_labels,
    report_results,
    save_labels,
)


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
        {"item_id": ids[0], "panel": "panels/q1/a.jpg", "clip_id": "c1", "shot_no": 1, "start_s": 0, "end_s": 1, "keyframe_id": "c1:1:0", "t_s": 0},
        {"item_id": ids[1], "panel": "panels/q1/b.jpg", "clip_id": "c2", "shot_no": 1, "start_s": 0, "end_s": 1, "keyframe_id": "c2:1:0", "t_s": 0},
        {"item_id": ids[2], "panel": "panels/q1/c.jpg", "clip_id": "c3", "shot_no": 1, "start_s": 0, "end_s": 1, "keyframe_id": "c3:1:0", "t_s": 0},
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

