import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from matchcut_retriever.data import main


def _fixture(tmp_path: Path) -> Path:
    video_root = tmp_path / "videos"
    annotation_dir = tmp_path / "annotations"
    library_root = tmp_path / "library"
    video_root.mkdir()
    annotation_dir.mkdir()
    library_root.mkdir()
    writer = cv2.VideoWriter(str(video_root / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for index in range(40):
        color = (20, 20, 20) if index < 10 else ((220, 220, 220) if index < 20 else (30, 30, 200))
        writer.write(np.full((48, 64, 3), color, dtype=np.uint8))
    writer.release()
    annotation = {
        "video_id": "clip",
        "media": {"fps": 10, "width": 64, "height": 48, "duration_sec": 4.0},
        "shots": [
            {"shot_no": 0, "start_sec": 0.0, "end_sec": 1.3},
            {"shot_no": 1, "start_sec": 1.3, "end_sec": 2.3},
            {"shot_no": 2, "start_sec": 2.3, "end_sec": 4.0},
        ],
    }
    (annotation_dir / "clip.json").write_text(json.dumps(annotation))
    config = tmp_path / "config.toml"
    config.write_text(
        "[paths]\n"
        f'video_root="{video_root}"\n'
        f'manifest_dir="{tmp_path / "manifest"}"\n'
        f'annotation_dir="{annotation_dir}"\n'
        f'library_root="{library_root}"\n'
        f'hf_cache="{tmp_path / "hf"}"\n'
    )
    return config


def test_data_pipeline_cli_is_resumable_and_reports_offset(tmp_path):
    config = _fixture(tmp_path)
    main(["select", "--config", str(config), "--n", "1"])
    main(["shots", "--config", str(config)])
    main(["keyframes", "--config", str(config), "--workers", "1", "--long-side", "32"])
    keyframes = pd.read_parquet(tmp_path / "library" / "keyframes.parquet")
    assert list(keyframes.columns) == ["keyframe_id", "clip_id", "shot_no", "k", "t_s", "frame_idx", "path", "width", "height"]
    assert len(keyframes) == 9
    before = {path: path.stat().st_mtime_ns for path in (tmp_path / "library" / "keyframes" / "clip").glob("*.jpg")}
    main(["keyframes", "--config", str(config), "--workers", "1", "--long-side", "32"])
    after = {path: path.stat().st_mtime_ns for path in before}
    assert before == after

    # Move annotations three frames after the actual cuts, then inspect the
    # nearest-cut histogram even though the default tolerance is two frames.
    annotation_path = tmp_path / "annotations" / "clip.json"
    annotation = json.loads(annotation_path.read_text())
    annotation["shots"][1]["start_sec"] = 1.3
    annotation["shots"][2]["start_sec"] = 2.3
    annotation_path.write_text(json.dumps(annotation))
    main(["check-boundaries", "--config", str(config), "--n", "1"])
    report = json.loads((tmp_path / "library" / "eval" / "boundary_check" / "report.json").read_text())
    assert report["offset_histogram"].get("-3") == 2
