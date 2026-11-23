"""Command line data preparation utilities for the match-cut library."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .annotations import keyframe_times, list_clip_ids, load_annotation, select_clips, shot_rows
from .config import Paths, load_paths
from .keyframes import extract_frames

LOG = logging.getLogger(__name__)
KEYFRAME_COLUMNS = [
    "keyframe_id",
    "clip_id",
    "shot_no",
    "k",
    "t_s",
    "frame_idx",
    "path",
    "width",
    "height",
]


def _read_clip_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"clip list not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _write_parquet(rows: list[dict[str, Any]], path: Path, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if columns is not None:
        for column in columns:
            if column not in frame:
                frame[column] = pd.Series(dtype="object")
        frame = frame[columns]
    frame.to_parquet(path, index=False)


def command_select(paths: Paths, n: int, seed: int, force: bool) -> None:
    ids = select_clips(list_clip_ids(paths.video_root), paths.annotation_dir, n, seed)
    target = paths.library_root / "clips.txt"
    content = "".join(f"{clip_id}\n" for clip_id in ids)
    if target.exists() and target.read_text() != content and not force:
        raise FileExistsError(f"refusing to overwrite a different clip list: {target} (use --force)")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    print(f"selected {len(ids)} clips -> {target}")


def command_shots(paths: Paths, min_shot_s: float) -> None:
    rows: list[dict[str, Any]] = []
    for clip_id in _read_clip_list(paths.library_root / "clips.txt"):
        annotation_path = paths.annotation_dir / f"{clip_id}.json"
        if not annotation_path.is_file():
            LOG.warning("missing annotation for %s", clip_id)
            continue
        rows.extend(shot_rows(load_annotation(annotation_path), min_shot_s=min_shot_s))
    _write_parquet(rows, paths.library_root / "shots.parquet")
    print(f"wrote {len(rows)} shots -> {paths.library_root / 'shots.parquet'}")


def _usable_shots(annotation: dict[str, Any], min_shot_s: float = 1.0) -> list[dict[str, Any]]:
    return [row for row in shot_rows(annotation, min_shot_s=min_shot_s) if row["usable"]]


def _clip_expected_paths(clip_id: str, shots: list[dict[str, Any]]) -> list[Path]:
    return [
        Path("keyframes") / clip_id / f"{int(shot['shot_no']):03d}_{k}.jpg"
        for shot in shots
        for k in range(3)
    ]


def _extract_clip_job(
    clip_id: str,
    video_path: str,
    library_root: str,
    shots: list[dict[str, Any]],
    long_side: int,
) -> tuple[str, list[dict[str, Any]], str | None]:
    """Extract all keyframes for one clip; this function is process-pool safe."""
    root = Path(library_root)
    try:
        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        if not fps or fps <= 0:
            raise RuntimeError("video has no usable FPS")
        requests: list[float] = []
        request_meta: list[tuple[dict[str, Any], int]] = []
        for shot in shots:
            times = keyframe_times(float(shot["start_s"]), float(shot["end_s"]))
            for k, t_s in enumerate(times):
                requests.append(t_s)
                request_meta.append((shot, k))
        if not requests:
            return clip_id, [], None
        decoded = extract_frames(video_path, requests, long_side=long_side)
        if len(decoded) != len(requests):
            raise RuntimeError(f"decoded {len(decoded)} of {len(requests)} requested frames")
        rows: list[dict[str, Any]] = []
        for (shot, k), (frame_idx, frame) in zip(request_meta, decoded):
            rel = Path("keyframes") / clip_id / f"{int(shot['shot_no']):03d}_{k}.jpg"
            destination = root / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(destination), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"failed to write {destination}")
            height, width = frame.shape[:2]
            rows.append(
                {
                    "keyframe_id": f"{clip_id}:{int(shot['shot_no'])}:{k}",
                    "clip_id": clip_id,
                    "shot_no": int(shot["shot_no"]),
                    "k": k,
                    "t_s": float(frame_idx / fps),
                    "frame_idx": int(frame_idx),
                    "path": rel.as_posix(),
                    "width": int(width),
                    "height": int(height),
                }
            )
        return clip_id, rows, None
    except Exception as exc:  # pragma: no cover - exercised through CLI integration
        return clip_id, [], f"{type(exc).__name__}: {exc}"


def _load_existing_keyframes(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=KEYFRAME_COLUMNS)
    frame = pd.read_parquet(path)
    for column in KEYFRAME_COLUMNS:
        if column not in frame:
            frame[column] = pd.Series(dtype="object")
    return frame[KEYFRAME_COLUMNS]


def command_keyframes(paths: Paths, workers: int, long_side: int) -> None:
    started = time.monotonic()
    clips = _read_clip_list(paths.library_root / "clips.txt")
    existing_path = paths.library_root / "keyframes.parquet"
    existing = _load_existing_keyframes(existing_path)
    processed_path = paths.library_root / "processed_clips.txt"
    processed = {line.strip() for line in processed_path.read_text().splitlines() if line.strip()} if processed_path.exists() else set()
    shot_table = pd.read_parquet(paths.library_root / "shots.parquet") if (paths.library_root / "shots.parquet").is_file() else None

    jobs: list[tuple[str, str, list[dict[str, Any]]]] = []
    skipped: set[str] = set()
    existing_ids = set(existing["keyframe_id"].astype(str)) if not existing.empty else set()
    for clip_id in clips:
        annotation_path = paths.annotation_dir / f"{clip_id}.json"
        if not annotation_path.is_file():
            LOG.warning("missing annotation for %s", clip_id)
            continue
        if shot_table is not None and "clip_id" in shot_table.columns:
            clip_rows = shot_table[shot_table["clip_id"] == clip_id]
            shots = clip_rows[clip_rows["usable"].astype(bool)].to_dict("records") if "usable" in clip_rows else []
        else:
            shots = _usable_shots(load_annotation(annotation_path))
        expected = _clip_expected_paths(clip_id, shots)
        expected_ids = {f"{clip_id}:{int(shot['shot_no'])}:{k}" for shot in shots for k in range(3)}
        if all((paths.library_root / path).is_file() for path in expected) and expected_ids.issubset(existing_ids):
            skipped.add(clip_id)
            processed.add(clip_id)
            continue
        video_path = paths.video_root / f"{clip_id}.mp4"
        jobs.append((clip_id, str(video_path), shots))

    results: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    if workers <= 1 or len(jobs) <= 1:
        job_results = [_extract_clip_job(cid, vp, str(paths.library_root), shots, long_side) for cid, vp, shots in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = [pool.submit(_extract_clip_job, cid, vp, str(paths.library_root), shots, long_side) for cid, vp, shots in jobs]
            job_results = [future.result() for future in as_completed(futures)]
    for clip_id, rows, error in job_results:
        if error:
            failures.append((clip_id, error))
            LOG.warning("failed to decode %s: %s", clip_id, error)
        else:
            results.extend(rows)
            processed.add(clip_id)

    if results:
        new_frame = pd.DataFrame(results, columns=KEYFRAME_COLUMNS)
        existing = existing[~existing["clip_id"].isin({row["clip_id"] for row in results})]
        existing = pd.concat([existing, new_frame], ignore_index=True)
    _write_parquet(existing.to_dict("records"), existing_path, KEYFRAME_COLUMNS)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.write_text("".join(f"{clip_id}\n" for clip_id in sorted(processed)))

    bytes_on_disk = sum(path.stat().st_size for path in (paths.library_root / "keyframes").rglob("*.jpg")) if (paths.library_root / "keyframes").exists() else 0
    if shot_table is not None and "usable" in shot_table.columns:
        shot_count = int(shot_table[shot_table["usable"].astype(bool)].shape[0])
    else:
        shot_count = sum(len(_usable_shots(load_annotation(paths.annotation_dir / f"{cid}.json"))) for cid in clips if (paths.annotation_dir / f"{cid}.json").is_file())
    print(
        f"clips={len(clips)} shots={shot_count} "
        f"keyframes={len(existing)} failures={len(failures)} elapsed={time.monotonic() - started:.2f}s bytes={bytes_on_disk}"
    )


def _video_metrics(video_path: Path) -> tuple[float, int, int, float, list[float], list[np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    metrics: list[float] = []
    samples: list[np.ndarray] = []
    previous: np.ndarray | None = None
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if previous is None:
            metrics.append(0.0)
        else:
            metrics.append(float(np.mean(np.abs(gray - previous)) / 255.0))
        previous = gray
        # Keep only small previews for contact sheets.  Full-resolution clips can
        # contain thousands of frames and must not all be retained in memory.
        preview_h = 96
        preview_w = max(1, round(frame.shape[1] * preview_h / frame.shape[0]))
        samples.append(cv2.resize(frame, (preview_w, preview_h), interpolation=cv2.INTER_AREA))
        frame_count += 1
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return fps, width, height, duration, metrics, samples


def _detect_cuts(metrics: list[float]) -> list[int]:
    if len(metrics) < 2:
        return []
    values = np.asarray(metrics[1:], dtype=np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(median + 6.0 * mad, median * 3.0 + 0.02, 0.20)
    return [index + 1 for index, value in enumerate(values) if float(value) >= threshold]


def _write_contact_sheet(path: Path, frames: list[np.ndarray | None], labels: list[str]) -> None:
    tiles: list[np.ndarray] = []
    for frame, label in zip(frames, labels):
        if frame is None:
            tile = np.zeros((96, 128, 3), dtype=np.uint8)
        else:
            tile = cv2.resize(frame, (128, 96), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, label, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if tiles:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])


def command_check_boundaries(paths: Paths, n: int, seed: int, tolerance_frames: int) -> None:
    clips = _read_clip_list(paths.library_root / "clips.txt")
    selected = select_clips(clips, paths.annotation_dir, min(n, len(clips)), seed)
    out_dir = paths.library_root / "eval" / "boundary_check"
    sheets_dir = out_dir / "sheets"
    offsets: dict[str, int] = {}
    mismatches: list[dict[str, Any]] = []
    annotated_total = matched_total = unmatched_detected = 0
    sheet_count = 0

    for clip_id in selected:
        annotation_path = paths.annotation_dir / f"{clip_id}.json"
        video_path = paths.video_root / f"{clip_id}.mp4"
        if not annotation_path.is_file() or not video_path.is_file():
            continue
        annotation = load_annotation(annotation_path)
        fps, width, height, duration, metrics, frames = _video_metrics(video_path)
        detected = _detect_cuts(metrics)
        media = annotation.get("media") or {}
        clip_mismatch: dict[str, Any] = {"clip_id": clip_id}
        for name, actual, expected in [
            ("fps", fps, media.get("fps")),
            ("width", width, media.get("width")),
            ("height", height, media.get("height")),
            ("duration_s", duration, media.get("duration_sec")),
        ]:
            if expected is None:
                continue
            tolerance = 0.05 if name in {"fps", "duration_s"} else 0
            if abs(float(actual) - float(expected)) > tolerance:
                clip_mismatch[name] = {"annotation": expected, "actual": actual}
        if len(clip_mismatch) > 1:
            mismatches.append(clip_mismatch)
        boundaries = [
            int(round(float(shot.get("start_sec", 0.0)) * fps))
            for shot in annotation.get("shots", [])
            if int(shot.get("shot_no", 0)) > 0
        ]
        annotated_total += len(boundaries)
        used_detected: set[int] = set()
        for boundary in boundaries:
            nearest = [(abs(cut - boundary), cut) for cut in detected]
            if nearest:
                _, nearest_cut = min(nearest)
                offset = nearest_cut - boundary
                offsets[str(offset)] = offsets.get(str(offset), 0) + 1
                if abs(offset) <= tolerance_frames:
                    used_detected.add(nearest_cut)
                    matched_total += 1
            if sheet_count < 8:
                indices = [boundary + delta for delta in range(-2, 3)]
                chosen = [frames[i] if 0 <= i < len(frames) else None for i in indices]
                _write_contact_sheet(sheets_dir / f"{clip_id}_{boundary:06d}.jpg", chosen, [str(i - boundary) for i in indices])
                sheet_count += 1
        unmatched_detected += sum(1 for cut in detected if not any(abs(cut - boundary) <= tolerance_frames for boundary in boundaries))

    report = {
        "clips": selected,
        "fraction_annotated_boundaries_with_detected_cut": (matched_total / annotated_total if annotated_total else 0.0),
        "fraction_annotated_boundaries_with_cut": (matched_total / annotated_total if annotated_total else 0.0),
        "annotated_boundaries": annotated_total,
        "matched_boundaries": matched_total,
        "offset_histogram": offsets,
        "detected_cuts_not_near_annotation": unmatched_detected,
        "detected_cuts_not_near_any_annotated_boundary": unmatched_detected,
        "per_clip_mismatches": mismatches,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m matchcut_retriever.data")
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--config")
    select.add_argument("--n", type=int, default=500)
    select.add_argument("--seed", type=int, default=0)
    select.add_argument("--force", action="store_true")
    shots = sub.add_parser("shots")
    shots.add_argument("--config")
    shots.add_argument("--min-shot-s", type=float, default=1.0)
    keyframes = sub.add_parser("keyframes")
    keyframes.add_argument("--config")
    keyframes.add_argument("--workers", type=int, default=16)
    keyframes.add_argument("--long-side", type=int, default=640)
    boundaries = sub.add_parser("check-boundaries")
    boundaries.add_argument("--config")
    boundaries.add_argument("--n", type=int, default=30)
    boundaries.add_argument("--seed", type=int, default=0)
    boundaries.add_argument("--tolerance-frames", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = load_paths(args.config)
    if args.command == "select":
        command_select(paths, args.n, args.seed, args.force)
    elif args.command == "shots":
        command_shots(paths, args.min_shot_s)
    elif args.command == "keyframes":
        command_keyframes(paths, args.workers, args.long_side)
    elif args.command == "check-boundaries":
        command_check_boundaries(paths, args.n, args.seed, args.tolerance_frames)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
