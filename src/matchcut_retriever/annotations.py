"""Helpers for reading CineDance shot annotations."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable


def load_annotation(path: str | Path) -> dict:
    """Load one annotation JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"annotation must be an object: {path}")
    return value


def list_clip_ids(video_root: str | Path) -> list[str]:
    """Return sorted clip ids for complete MP4 files in *video_root*."""
    root = Path(video_root)
    return sorted(p.stem for p in root.glob("*.mp4") if p.is_file() and not p.name.endswith(".part"))


def select_clips(video_ids: Iterable[str], annotation_dir: str | Path, n: int, seed: int) -> list[str]:
    """Sample up to ``n`` ids that have an annotation, deterministically."""
    if n < 0:
        raise ValueError("n must be non-negative")
    ann_dir = Path(annotation_dir)
    available = sorted(str(cid) for cid in video_ids if (ann_dir / f"{cid}.json").is_file())
    rng = random.Random(seed)
    chosen = rng.sample(available, min(n, len(available)))
    return sorted(chosen)


def _first(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def shot_rows(annotation: dict, min_shot_s: float = 1.0) -> list[dict]:
    """Flatten an annotation into one row per shot."""
    media = annotation.get("media") or {}
    clip_id = annotation.get("video_id")
    rows: list[dict] = []
    for shot in annotation.get("shots") or []:
        if not isinstance(shot, dict):
            continue
        start = float(shot.get("start_sec", 0.0) or 0.0)
        end = float(shot.get("end_sec", start) or start)
        # Annotation timestamps are decimal values; rounding avoids marking a
        # nominal one-second shot unusable due to binary floating-point noise.
        duration = round(end - start, 9)
        shot_type = shot.get("shot_type") or {}
        if not isinstance(shot_type, dict):
            shot_type = {}
        rows.append({
            "clip_id": clip_id,
            "shot_no": int(shot.get("shot_no", len(rows))),
            "start_s": start,
            "end_s": end,
            "duration_s": duration,
            "usable": duration >= min_shot_s,
            "scale": shot_type.get("scale"),
            "angle": shot_type.get("angle"),
            "movement": shot_type.get("movement"),
            "transition_type": shot.get("shot_transition_type"),
            "shot_description": shot.get("shot_description") or "",
            "brief_caption": shot.get("brief_caption") or "",
            "media_fps": media.get("fps"),
            "media_width": media.get("width"),
            "media_height": media.get("height"),
            "media_duration_s": media.get("duration_sec"),
        })
    return rows


def keyframe_times(start_s: float, end_s: float, fracs: Iterable[float] = (0.2, 0.5, 0.8)) -> list[float]:
    """Return times at fractional positions within a shot."""
    return [float(start_s + (end_s - start_s) * frac) for frac in fracs]
