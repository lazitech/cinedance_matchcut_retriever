import cv2
import numpy as np

from matchcut_retriever.keyframes import extract_frames


def test_extract_frames_unsorted_repeated_and_no_upscale(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (8, 6))
    for i in range(5):
        writer.write(np.full((6, 8, 3), i * 35, dtype=np.uint8))
    writer.release()
    got = extract_frames(path, [0.31, 0.0, 0.31, 99.0], long_side=100)
    assert [item[0] for item in got] == [3, 0, 3, 4]
    assert all(frame.dtype == np.uint8 for _, frame in got)
    assert all(max(frame.shape[:2]) == 8 for _, frame in got)

