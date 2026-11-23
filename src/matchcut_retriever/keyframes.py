"""Sequential video decoding for analysis keyframes."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _resize_rgb(frame_bgr: np.ndarray, long_side: int) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if long_side <= 0:
        raise ValueError("long_side must be positive")
    scale = min(1.0, float(long_side) / max(h, w))
    if scale < 1.0:
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def extract_frames(
    video_path: str | Path, times_s, long_side: int = 640
) -> list[tuple[int, np.ndarray]]:
    """Decode *video_path* sequentially and return nearest frames for times.

    The returned list follows the order of ``times_s`` and preserves duplicate
    requests. Each image is RGB uint8 and is never enlarged.
    """
    times = [float(t) for t in times_s]
    if not times:
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        cap.release()
        raise ValueError(f"video has invalid FPS: {video_path}")
    targets = [max(0.0, t * fps) for t in times]
    unique_targets = sorted(set(targets))
    resolved: dict[float, tuple[int, np.ndarray]] = {}
    next_pos = 0
    prev_idx = -1
    prev_frame: np.ndarray | None = None
    idx = 0
    last_idx = -1
    last_frame: np.ndarray | None = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            last_idx, last_frame = idx, frame
            while next_pos < len(unique_targets) and idx >= unique_targets[next_pos]:
                target = unique_targets[next_pos]
                # At the first frame on/after target, compare with previous.
                if prev_frame is None or target - prev_idx > idx - target:
                    chosen_idx, chosen = idx, frame
                else:
                    chosen_idx, chosen = prev_idx, prev_frame
                resolved[target] = (chosen_idx, _resize_rgb(chosen, long_side))
                next_pos += 1
            prev_idx, prev_frame = idx, frame
            idx += 1
    finally:
        cap.release()
    if last_frame is None:
        raise ValueError(f"video contains no decodable frames: {video_path}")
    # Requests beyond EOF clamp to the final decodable frame. This also covers
    # targets that were skipped because no frame reached them.
    final = (last_idx, _resize_rgb(last_frame, long_side))
    for target in unique_targets:
        resolved.setdefault(target, final)
    return [resolved[target] for target in targets]
