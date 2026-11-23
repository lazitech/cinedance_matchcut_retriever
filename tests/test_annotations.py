from matchcut_retriever.annotations import (
    keyframe_times,
    list_clip_ids,
    select_clips,
    shot_rows,
)


def test_clip_listing_selection_and_rows(tmp_path):
    (tmp_path / "b.mp4").touch()
    (tmp_path / "a.mp4").touch()
    (tmp_path / "partial.mp4.part").touch()
    assert list_clip_ids(tmp_path) == ["a", "b"]
    ann = tmp_path / "ann"
    ann.mkdir()
    for clip in ("a", "b"):
        (ann / f"{clip}.json").write_text("{}")
    assert select_clips(["b", "a"], ann, 1, 3) == ["a"]
    data = {
        "video_id": "a",
        "media": {"fps": 24, "width": 64, "height": 48, "duration_sec": 2},
        "shots": [
            {"shot_no": 0, "start_sec": 0, "end_sec": 0.5},
            {"shot_no": 1, "start_sec": 0.5, "end_sec": 2,
             "shot_type": {"scale": "Wide"}, "brief_caption": "x"},
        ],
    }
    rows = shot_rows(data)
    assert rows[0]["usable"] is False
    assert rows[1]["usable"] is True and rows[1]["scale"] == "Wide"
    assert keyframe_times(2, 4) == [2.4, 3.0, 3.6]

