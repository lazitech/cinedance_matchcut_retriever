import json

import numpy as np
import pytest

from matchcut_retriever.models import Detection
from matchcut_retriever.selftest import (
    choose_interior_click,
    perturb_image_mask,
    run_selftest,
    sample_object_ids,
    summarize_records,
)
from matchcut_retriever.library import LibraryBuilder, consolidate
from test_library import FakeEmbedder, FakeSegmenter, _make_root


def _build(tmp_path):
    root = _make_root(tmp_path, clips=("a", "b"), frames_per_clip=1)
    LibraryBuilder(root, [FakeSegmenter("mask2former", "thing"), FakeSegmenter("sam2_auto")], FakeEmbedder()).process_clips(["a", "b"])
    consolidate(root)
    return root


def test_perturb_identity_and_consistent_transform():
    image = np.zeros((64, 80, 3), np.uint8)
    image[20:34, 24:40] = (220, 20, 10)
    mask = np.zeros((64, 80), bool)
    mask[20:34, 24:40] = True
    out, out_mask, params = perturb_image_mask(
        image, mask,
        params={"scale": 1, "angle_deg": 0, "tx_frac": 0, "ty_frac": 0,
                "brightness": 1, "contrast": 1, "saturation": 1, "jpeg_quality": 90},
    )
    assert np.array_equal(out_mask, mask)
    assert np.mean(np.abs(out.astype(int) - image.astype(int))) < 8
    moved, moved_mask, _ = perturb_image_mask(
        image, mask,
        params={"scale": 1, "angle_deg": 0, "tx_frac": 0.10, "ty_frac": -0.05,
                "brightness": 1, "contrast": 1, "saturation": 1, "jpeg_quality": 90},
    )
    assert not np.array_equal(moved_mask, mask)
    # The transformed image and mask still overlap in the same colored object.
    assert moved[moved_mask].mean() > 30
    assert params["area_ratio"] == pytest.approx(1.0)


def test_click_is_interior():
    mask = np.zeros((80, 100), bool)
    mask[15:65, 25:75] = True
    x, y, distance = choose_interior_click(mask, np.random.default_rng(2))
    assert mask[y, x]
    assert distance > 0
    # The helper promises at least half the maximum distance transform value.
    import cv2
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    assert distance >= float(dt.max()) * 0.5 - 1e-6


def test_sample_ids_are_deterministic():
    objects = list(range(20))
    assert sample_object_ids(objects, n=8, seed=17) == sample_object_ids(objects, n=8, seed=17)
    assert sample_object_ids(objects, n=8, seed=17) != sample_object_ids(objects, n=8, seed=18)
    assert len(sample_object_ids(objects, n=100, seed=0)) == 20


def test_summarize_records_recall_arithmetic():
    records = [
        {"sample_index": 0, "area_frac": .01, "segmenter": "x", "results": {"gt": {"soft": {"rank": 1, "timing_s": .2}}}},
        {"sample_index": 1, "area_frac": .05, "segmenter": "x", "results": {"gt": {"soft": {"rank": 11, "timing_s": .4}}}},
        {"sample_index": 2, "area_frac": .2, "segmenter": "y", "results": {"gt": {"soft": {"rank": None, "timing_s": .6}}}},
    ]
    summary = summarize_records(records)
    entry = summary["modes"]["gt"]["soft"]
    assert entry["n"] == 3
    assert entry["recall"] == {"1": pytest.approx(1 / 3), "10": pytest.approx(1 / 3), "50": pytest.approx(2 / 3)}
    assert entry["median_rank_time_s"] == pytest.approx(.4)
    assert summary["failures"] == []


def test_end_to_end_resume_and_shards(tmp_path):
    root = _build(tmp_path / "lib")
    one = tmp_path / "one"
    result = run_selftest(root, one, n=3, seed=4, mask_source="gt", modes=("soft",), k_max=5, device="cpu", workers=1)
    assert result["n"] == 3
    lines = (one / "records.jsonl").read_text().splitlines()
    assert len(lines) == 3
    # Resume must not append duplicate sample records.
    run_selftest(root, one, n=3, seed=4, mask_source="gt", modes=("soft",), k_max=5, device="cpu", workers=1)
    assert len((one / "records.jsonl").read_text().splitlines()) == 3
    shards = tmp_path / "shards"
    run_selftest(root, shards, n=3, seed=4, mask_source="gt", modes=("soft",), k_max=5, device="cpu", workers=1, shard="0/2")
    run_selftest(root, shards, n=3, seed=4, mask_source="gt", modes=("soft",), k_max=5, device="cpu", workers=1, shard="1/2")
    assert len(list(shards.glob("records-*.jsonl"))) == 2
    sh_summary = json.loads((shards / "summary.json").read_text())
    assert sh_summary["n"] == result["n"]
    assert sh_summary["modes"]["gt"]["soft"]["recall"] == result["modes"]["gt"]["soft"]["recall"]


def test_kmax_limits_supported_recall_cutoffs(tmp_path):
    root = _build(tmp_path / "lib")
    out = tmp_path / "k5"
    summary = run_selftest(root, out, n=2, seed=9, mask_source="gt", modes=("soft",), k_max=5, device="cpu", workers=1)
    records = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
    assert records and all(record["k_max"] == 5 for record in records)
    recall = summary["modes"]["gt"]["soft"]["recall"]
    assert recall["1"] is not None and recall["5"] is not None
    assert recall["10"] is None and recall["50"] is None
    markdown = (out / "summary.md").read_text()
    assert "—" in markdown


def test_fake_sam_candidate_selection(tmp_path):
    root = _build(tmp_path / "lib")

    class FakePoint:
        def masks_at_point(self, image, x, y):
            h, w = image.shape[:2]
            good = np.ones((h, w), bool)
            bad = np.zeros((h, w), bool); bad[: max(1, h // 8), : max(1, w // 8)] = True
            return [Detection(bad, .9), Detection(good, .8)]

    out = tmp_path / "sam"
    summary = run_selftest(root, out, n=1, seed=0, mask_source="sam", modes=("soft",), k_max=5, device="cpu", workers=1, point_segmenter=FakePoint())
    rec = json.loads((out / "records.jsonl").read_text().splitlines()[0])
    assert rec["sam"]["best_index"] == 1
    assert rec["sam"]["best_iou"] > rec["sam"]["candidate_0_iou"]
    assert summary["sam"]["n"] == 1
