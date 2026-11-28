"""Gradio application and UI-independent query engine."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import load_paths
from .framing import prepare_image
from .query import query_warnings
from .retrieval import Library, QueryObject, rank
from .retrieval_config import load_retrieval_config, override_config


def cut_shot_clip(source, start_s: float, end_s: float, target, error_out=None) -> bool:
    """Cut and encode a shot clip atomically with ffmpeg."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=target.stem + ".", suffix=".mp4", dir=target.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        cmd = ["ffmpeg", "-y", "-ss", str(max(0.0, float(start_s))), "-i", str(source),
               "-t", str(max(0.01, float(end_s) - float(start_s))), "-vf",
               "scale=w='if(gt(iw,ih),min(640,iw),-2)':h='if(gt(iw,ih),-2,min(640,ih))'",
               "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-movflags", "+faststart", str(temp)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0 or not temp.is_file() or temp.stat().st_size == 0:
            if error_out is not None:
                error_out.append(proc.stderr.decode(errors="replace")[-300:])
            return False
        os.replace(temp, target)
        return True
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class Candidate:
    """A SAM candidate in prepared-image coordinates."""
    mask: np.ndarray
    score: float
    area_frac: float

    def __getitem__(self, key: str):
        """Allow the small record to be used as either an object or mapping."""
        if key not in {"mask", "score", "area_frac"}:
            raise KeyError(key)
        return getattr(self, key)


class QueryEngine:
    """Thread-safe, lazy model wrapper used by the Gradio front end."""

    def __init__(self, library_root=None, config=None, retrieval_config=None,
                 device: str = "cpu", sam=None, clip=None, paths=None, preload: bool = False):
        self.device = device
        self._path_message: str | None = None
        self.paths = paths
        if self.paths is None:
            try:
                # Resolve machine paths even when a library root override is
                # supplied: video extraction and model caches still need them.
                self.paths = load_paths(config)
            except FileNotFoundError:
                # Synthetic/test deployments may intentionally omit the local
                # machine config. Keep the explicit library root usable.
                self._path_message = "machine paths are not configured; video clips and model cache paths are unavailable"
        if library_root is None:
            if self.paths is None:
                self.paths = load_paths(config)
            library_root = self.paths.library_root
        self.library_root = Path(library_root)
        self.library = Library.load(self.library_root, preload=preload)
        self.config = load_retrieval_config(retrieval_config)
        self._sam = sam
        self._clip = clip
        self._lock = threading.RLock()
        self._clip_cache: dict[tuple[tuple[int, ...], str, str, tuple[int, ...], str, str], np.ndarray] = {}
        self.last_message: str | None = self._path_message

    @property
    def sam(self):
        with self._lock:
            if self._sam is None:
                from .models import SamPointSegmenter
                cache = self.paths.hf_cache if self.paths is not None else None
                self._sam = SamPointSegmenter(device=self.device, cache_dir=cache)
        return self._sam

    @property
    def clip(self):
        with self._lock:
            if self._clip is None:
                from .models import ClipEmbedder
                cache = self.paths.hf_cache if self.paths is not None else None
                self._clip = ClipEmbedder(device=self.device, cache_dir=cache)
        return self._clip

    def warm_up(self) -> None:
        """Load SAM and CLIP now; a cold load on the first click can take
        over 40 s on CPU and made the first request fail through the share link."""
        _ = self.sam
        _ = self.clip

    @staticmethod
    def _array(image) -> np.ndarray:
        if isinstance(image, (str, os.PathLike)):
            from PIL import Image
            return np.asarray(Image.open(image).convert("RGB"))
        if hasattr(image, "convert"):
            image = image.convert("RGB")
            return np.asarray(image)
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError("image must be an RGB image")
        return np.asarray(arr[..., :3], dtype=np.uint8)

    def candidates(self, image, x: float, y: float):
        """Prepare *image*, map its pixel click, and run point-SAM.

        Returns ``(prepared_rgb, candidates)`` where each candidate exposes
        ``mask``, ``score`` and ``area_frac`` (and supports mapping-style
        access for UI callbacks).
        """
        arr = self._array(image)
        prepared, transform = prepare_image(arr, long_side=640)
        px, py = transform.to_prepared(float(x), float(y))
        with self._lock:
            detections = self.sam.masks_at_point(prepared, px, py)
        candidates = [
            Candidate(mask=np.asarray(det.mask, dtype=bool), score=float(det.score),
                      area_frac=float(np.asarray(det.mask, dtype=bool).mean()))
            for det in detections
        ]
        return np.asarray(prepared), candidates

    def _clip_vector(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        mask = np.asarray(mask, dtype=bool)
        key = (
            tuple(image.shape), image.dtype.str, hashlib.sha1(image.tobytes()).hexdigest(),
            tuple(mask.shape), mask.dtype.str, hashlib.sha1(mask.tobytes()).hexdigest(),
        )
        with self._lock:
            value = self._clip_cache.get(key)
            if value is None:
                value = np.asarray(self.clip.embed_masked([image], [mask])[0], dtype=np.float32)
                self._clip_cache[key] = value
            return value

    def search(self, prepared, mask, mode=None, semantic=None, k=None,
               min_scale=None, max_scale=None, max_translate=None, max_rot=None,
               min_subject=None):
        image = self._array(prepared)
        mask = np.asarray(mask, dtype=bool)
        mode = (mode or "soft").lower()
        weight = self.config.semantic_weight if semantic is None else float(semantic)
        clip_vector = self._clip_vector(image, mask) if weight != 0 else None
        params = override_config(self.config, min_scale=min_scale, max_scale=max_scale,
                                 max_translate_frac=max_translate, max_rot_deg=max_rot,
                                 k=k, semantic_weight=weight,
                                 min_subject_score=min_subject)
        query = QueryObject.from_mask(image, mask, clip_vector)
        with self._lock:
            results = rank(self.library, query, mode, semantic_weight=weight,
                           k=params.k, params=params)
        warnings = query_warnings(mask, mode, params.hard_limits, len(results))
        return results, dict(getattr(results, "timings", {})), warnings

    @property
    def video_cache_dir(self) -> Path:
        """Where shot clips are cached; the Gradio app must list it in ``allowed_paths``."""
        return Path(self.library_root) / "cache" / "shot_clips"

    def shot_video(self, clip_id: str, shot_no: int):
        """Create/cache a small H.264 clip for a library shot."""
        self.last_message = None
        if self.paths is None:
            self.last_message = "video root is not configured"
            return None
        source = Path(self.paths.video_root) / f"{clip_id}.mp4"
        if not source.is_file():
            self.last_message = f"source video is missing: {source}"
            return None
        shot = self.library.shot(clip_id, shot_no)
        if shot is None:
            self.last_message = f"shot not found: {clip_id} shot {shot_no}"
            return None
        start = float(getattr(shot, "start_s"))
        end = float(getattr(shot, "end_s"))
        cache_dir = self.video_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"{clip_id}_{int(shot_no)}.mp4"
        if target.is_file() and target.stat().st_size > 0:
            return str(target)
        errors = []
        if not cut_shot_clip(source, start, end, target, errors):
            detail = errors[0] if errors else ""
            self.last_message = f"ffmpeg failed for {clip_id} shot {shot_no}: {detail}".rstrip()
            return None
        return str(target)


__all__ = ["Candidate", "QueryEngine", "cut_shot_clip"]


def _result_gallery_item(engine, prepared, query_mask, result, rank: int):
    """Build a compact candidate/overlay panel for the Gradio gallery."""
    from .render import result_panel
    panel = result_panel(engine.library, prepared, query_mask, result)
    caption = f"#{rank} {result.clip_id[:12]}… shot {result.shot_no}"
    if result.start_s is not None and result.end_s is not None:
        caption += f" [{result.start_s:.2f}-{result.end_s:.2f}s]"
    caption += f" score={result.final_score:.3f} IoU={result.iou:.3f} cost={result.transform_cost:.3f}"
    if getattr(result, "subject_score", None) is not None:
        caption += f" subject={result.subject_score:.3f}"
    if result.clip_cosine is not None:
        caption += f" CLIP={result.clip_cosine:.3f}"
    return panel, caption


def _ensure_gradio_temp_dir() -> str:
    """Point Gradio's upload dir at a per-user folder unless GRADIO_TEMP_DIR is set.

    Gradio defaults to ``/tmp/gradio``; on a shared host another user may own it and every
    upload then fails with a permission error. Gradio reads the variable at import time, so
    this must run before ``import gradio``.
    """
    current = os.environ.get("GRADIO_TEMP_DIR")
    if current:
        return current
    try:
        user = str(os.getuid())
    except AttributeError:  # pragma: no cover - non-POSIX
        user = os.environ.get("USERNAME", "user")
    path = Path(tempfile.gettempdir()) / f"gradio-{user}"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = str(path)
    return str(path)


def build_app(engine: QueryEngine | None = None, **engine_kwargs):
    """Construct the Gradio Blocks application without launching a server."""
    _ensure_gradio_temp_dir()
    import gradio as gr
    if engine is None:
        engine = QueryEngine(**engine_kwargs)
    cfg = engine.config
    with gr.Blocks(title="Match-Cut Retriever") as app:
        gr.Markdown("# CineDance Match-Cut Retriever\nUpload an image and click a subject to search matching shots.")
        with gr.Row():
            image = gr.Image(label="Query image", type="numpy")
            prepared_view = gr.Image(label="Prepared image / selected mask", type="numpy", interactive=False)
        candidates = gr.Radio(label="SAM candidates", choices=[], value=None)
        warnings = gr.Markdown()
        with gr.Row():
            mode = gr.Radio(["soft", "hard"], value="soft", label="Mode")
            semantic = gr.Slider(-1, 1, value=float(cfg.semantic_weight), step=0.05, label="Semantic preference (prefer different class ← ignore → prefer same class)")
            min_subject = gr.Slider(0, 1, value=float(getattr(cfg, "min_subject_score", 0.5)), step=0.05, label="Minimum subject score")
            k = gr.Slider(5, 50, value=min(20, max(5, cfg.k)), step=1, label="K")
        with gr.Accordion("Hard-mode limits", open=False):
            with gr.Row():
                min_scale = gr.Number(value=cfg.min_scale, label="Minimum scale")
                max_scale = gr.Number(value=cfg.max_scale, label="Maximum scale")
                max_translate = gr.Number(value=cfg.max_translate_frac, label="Max translate (frame width fraction)")
                max_rot = gr.Number(value=cfg.max_rot_deg, label="Max rotation (degrees)")
        search_btn = gr.Button("Search", variant="primary")
        timings = gr.Markdown()
        gallery = gr.Gallery(label="Results (top: candidate shot, bottom: aligned overlay)", columns=4, height=900, object_fit="contain", show_label=True)
        breakdown = gr.JSON(label="Result breakdown")
        video = gr.Video(label="Shot clip", interactive=False)
        state = gr.State({"prepared": None, "items": []})
        result_state = gr.State([])

        def on_select(img, evt: gr.SelectData = None):
            if img is None or evt is None or evt.index is None:
                return gr.update(), gr.update(), "Upload an image and click a subject.", {"prepared": None, "items": []}
            try:
                x, y = evt.index[:2]
                prepared, items = engine.candidates(img, x, y)
                if not items:
                    return prepared, gr.update(choices=[], value=None), "SAM returned no masks at this point.", {"prepared": prepared, "items": []}
                choices = [f"{i}: {item['area_frac']:.1%} (score {item['score']:.3f})" for i, item in enumerate(items)]
                # Show candidate zero by default.
                from .render import _outline
                preview = _outline(prepared, items[0]["mask"], (255, 70, 40))
                return preview, gr.update(choices=choices, value=choices[0]), "Select a candidate mask, then Search.", {"prepared": prepared, "items": items}
            except Exception as exc:
                return gr.update(), gr.update(), f"Error: {exc}", {"prepared": None, "items": []}

        def on_candidate(choice, st):
            if not st or st.get("prepared") is None or not choice:
                return gr.update(), st
            idx = int(str(choice).split(":", 1)[0])
            item = st["items"][idx]
            from .render import _outline
            return _outline(st["prepared"], item["mask"], (255, 70, 40)), st

        def do_search(choice, st, mode_value, semantic_value, min_subject_value, k_value, min_s, max_s, trans, rot):
            if not st or st.get("prepared") is None or not choice:
                return [], "Click the query image first.", {}, None
            idx = int(str(choice).split(":", 1)[0]); item = st["items"][idx]
            results, t, warns = engine.search(st["prepared"], item["mask"], mode_value, semantic_value, int(k_value), min_s, max_s, trans, rot, min_subject_value)
            gallery_items = [_result_gallery_item(engine, st["prepared"], item["mask"], r, rank)
                             for rank, r in enumerate(results, 1)]
            msg = "\n".join(f"⚠️ {w}" for w in warns) if warns else ""
            subject_filter = getattr(results, "subject_filter", None)
            if isinstance(subject_filter, str) and subject_filter.startswith("unavailable"):
                msg += ("\n" if msg else "") + f"⚠️ Subject filter unavailable: {subject_filter}"
            if t:
                msg += ("\n" if msg else "") + "Timings: " + ", ".join(f"{key}={value:.3f}s" for key, value in t.items())
                msg += f" (coarse_method={getattr(results, 'coarse_method', 'exact')})"
            return gallery_items, msg, {"results": [r.to_dict() for r in results]}, results

        def on_result(result_list, evt: gr.SelectData = None):
            if evt is None or evt.index is None or not result_list:
                return {}, None
            idx = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
            try:
                result = result_list[int(idx)]
            except (IndexError, TypeError, ValueError):
                return {"error": "selected result is unavailable"}, None
            breakdown = result.to_dict()
            try:
                path = engine.shot_video(result.clip_id, result.shot_no)
            except Exception as exc:
                path = None
                breakdown["error"] = str(exc)
            if path is None:
                breakdown["error"] = (
                    getattr(engine, "last_message", None)
                    or "unable to create the shot video"
                )
            return breakdown, path

        # This module uses postponed annotations, so `gr.SelectData` above is only a string that
        # Gradio cannot resolve (gr is a local import); without the real class it never injects
        # the event and every click looks like "no click".
        on_select.__annotations__["evt"] = gr.SelectData
        on_result.__annotations__["evt"] = gr.SelectData

        # Keep the click handler synchronous so Gradio preserves SelectData's
        # pixel index before the longer SAM call is queued.
        image.select(on_select, [image], [prepared_view, candidates, warnings, state], queue=False)
        candidates.change(on_candidate, [candidates, state], [prepared_view, state])
        search_btn.click(do_search, [candidates, state, mode, semantic, min_subject, k, min_scale, max_scale, max_translate, max_rot], [gallery, timings, breakdown, result_state])
        gallery.select(on_result, [result_state], [breakdown, video])
    # Gradio only serves files from cwd/temp unless the directory is allowed at launch.
    app.matchcut_allowed_paths = [str(engine.video_cache_dir)]
    app.matchcut_engine = engine
    return app


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Launch the Gradio match-cut query UI")
    parser.add_argument("--config")
    parser.add_argument("--library-root")
    parser.add_argument("--retrieval-config")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--preload", action="store_true", help="load shape fingerprints and masks into RAM")
    parser.add_argument("--no-warmup", action="store_true", help="load SAM/CLIP lazily on the first request instead of at startup")
    args = parser.parse_args(argv)
    # `kill -USR1 <pid>` dumps every thread's stack to stderr (ptrace/py-spy is
    # unavailable on the shared server) for diagnosing a hung request.
    import faulthandler
    import signal
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    app = build_app(library_root=args.library_root, config=args.config, retrieval_config=args.retrieval_config, device=args.device, preload=args.preload)
    if not args.no_warmup:
        app.matchcut_engine.warm_up()
    launch_kwargs = {"server_name": args.host, "server_port": args.port, "share": args.share}
    allowed = getattr(app, "matchcut_allowed_paths", None)
    if allowed:
        launch_kwargs["allowed_paths"] = list(allowed)
    app.queue().launch(**launch_kwargs)
    return 0


__all__ = ["Candidate", "QueryEngine", "build_app", "cut_shot_clip", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
