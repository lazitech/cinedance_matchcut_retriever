"""CPU-only checks for the Gradio query engine."""

import numpy as np
import cv2
import pandas as pd
from pathlib import Path

from matchcut_retriever.app import QueryEngine
import matchcut_retriever.app as app_module
from matchcut_retriever.models import Detection
from matchcut_retriever.library import LibraryBuilder, consolidate
from matchcut_retriever.config import Paths
from matchcut_retriever.app import build_app

from test_library import FakeEmbedder, FakeSegmenter, _make_root


class FakeSam:
    def __init__(self):
        self.calls = []

    def masks_at_point(self, image, x, y):
        self.calls.append((image.shape, float(x), float(y)))
        mask = np.zeros(image.shape[:2], dtype=bool)
        yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
        mask[(xx - int(round(x))) ** 2 + (yy - int(round(y))) ** 2 < 20**2] = True
        return [Detection(mask, 0.9, None, "fake-sam")]


class CountingClip(FakeEmbedder):
    def __init__(self):
        self.calls = 0

    def embed_masked(self, images, masks):
        self.calls += 1
        return super().embed_masked(images, masks)


def _engine(tmp_path, clip=None):
    root = _make_root(tmp_path, clips=("a", "b"), frames_per_clip=1)
    LibraryBuilder(root, [FakeSegmenter("sam2_auto")], FakeEmbedder()).process_clips(["a", "b"])
    consolidate(root)
    return QueryEngine(library_root=root, sam=FakeSam(), clip=clip or CountingClip())


def test_candidates_maps_click_after_black_bar_crop(tmp_path):
    engine = _engine(tmp_path)
    image = np.full((200, 400, 3), 120, dtype=np.uint8)
    image[:40] = 0
    image[160:] = 0
    prepared, candidates = engine.candidates(image, 200, 100)
    assert prepared.shape[1] == 640
    assert len(candidates) == 1
    # Active region is 400x120, so the click maps to (320, 96).
    _, x, y = engine.sam.calls[-1]
    assert abs(x - 320) < 2 and abs(y - 96) < 2
    assert candidates[0]["mask"].shape == prepared.shape[:2]


def test_search_merges_per_shot_and_caches_semantic_query(tmp_path):
    clip = CountingClip()
    engine = _engine(tmp_path, clip=clip)
    image = np.full((80, 100, 3), 128, dtype=np.uint8)
    mask = np.zeros((80, 100), dtype=bool)
    mask[40:56, 50:70] = True
    results, _, warnings = engine.search(image, mask, mode="soft", semantic=0.5, k=5)
    assert results
    assert len({(r.clip_id, r.shot_no) for r in results}) == len(results)
    assert warnings == []
    engine.search(image, mask, mode="soft", semantic=0.5, k=5)
    assert clip.calls == 1


def test_clip_cache_key_includes_shape_and_dtype(tmp_path):
    clip = CountingClip()
    engine = _engine(tmp_path, clip=clip)
    image = np.full((8, 12, 3), 128, dtype=np.uint8)
    mask = np.zeros((8, 12), dtype=bool)
    mask[2:6, 3:8] = True
    engine._clip_vector(image, mask)
    # Same bytes with a different image shape must not reuse the first entry.
    image_reshaped = image.reshape(4, 24, 3)
    mask_reshaped = mask.reshape(4, 24)
    engine._clip_vector(image_reshaped, mask_reshaped)
    assert clip.calls == 2


def test_library_root_override_still_loads_machine_paths(tmp_path, monkeypatch):
    root = _make_root(tmp_path, clips=("a",), frames_per_clip=1)
    LibraryBuilder(root, [FakeSegmenter("sam2_auto")], FakeEmbedder()).process_clips(["a"])
    consolidate(root)
    expected = Paths(tmp_path / "videos", tmp_path, tmp_path, root, tmp_path / "hf")
    monkeypatch.setattr("matchcut_retriever.app.load_paths", lambda config=None: expected)
    engine = QueryEngine(library_root=root, sam=FakeSam(), clip=CountingClip())
    assert engine.paths == expected
    assert engine.last_message is None


def test_search_reports_tiny_mask_warning(tmp_path):
    engine = _engine(tmp_path)
    image = np.full((80, 100, 3), 128, dtype=np.uint8)
    mask = np.zeros((80, 100), dtype=bool)
    mask[40:42, 50:52] = True
    _, _, warnings = engine.search(image, mask, mode="hard", semantic=0.0, k=3)
    assert any("outside the library" in warning or "hard scale limits" in warning for warning in warnings)


def test_shot_video_is_cached_and_missing_source_is_reported(tmp_path):
    root = _make_root(tmp_path, clips=("a",), frames_per_clip=1)
    LibraryBuilder(root, [FakeSegmenter("sam2_auto")], FakeEmbedder()).process_clips(["a"])
    consolidate(root)
    pd.DataFrame([{"clip_id": "a", "shot_no": 1, "start_s": 0.2, "end_s": 0.8}]).to_parquet(root / "shots.parquet", index=False)
    video_root = tmp_path / "videos"
    video_root.mkdir()
    writer = cv2.VideoWriter(str(video_root / "a.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (80, 120))
    for value in range(20):
        writer.write(np.full((120, 80, 3), value * 8, dtype=np.uint8))
    writer.release()
    paths = Paths(video_root, tmp_path, tmp_path, root, tmp_path)
    engine = QueryEngine(library_root=root, paths=paths, sam=FakeSam(), clip=CountingClip())
    first = engine.shot_video("a", 1)
    assert first is not None and Path(first).is_file()
    second = engine.shot_video("a", 1)
    assert second == first
    capture = cv2.VideoCapture(first)
    assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) <= 640
    assert 0.3 <= capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS) <= 1.1
    capture.release()
    assert engine.shot_video("missing", 1) is None
    assert "source video is missing" in (engine.last_message or "")


def test_build_app_with_injected_engine_does_not_launch(tmp_path):
    import pytest

    pytest.importorskip("gradio")  # optional `app` extra
    engine = _engine(tmp_path)
    app = build_app(engine)
    assert app is not None


def test_main_configures_and_launches_app(monkeypatch):
    calls = {}

    class FakeApp:
        def queue(self):
            calls["queued"] = True
            return self

        def launch(self, **kwargs):
            calls["launch"] = kwargs

    def fake_build_app(**kwargs):
        calls["build"] = kwargs
        return FakeApp()

    monkeypatch.setattr(app_module, "build_app", fake_build_app)
    assert app_module.main(["--library-root", "/tmp/library", "--host", "0.0.0.0", "--port", "8123", "--share"]) == 0
    assert calls["build"]["library_root"] == "/tmp/library"
    assert calls["queued"] is True
    assert calls["launch"] == {"server_name": "0.0.0.0", "server_port": 8123, "share": True}
