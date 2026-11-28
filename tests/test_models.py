from __future__ import annotations

import builtins
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest


def test_optional_modules_import_without_torch():
    import matchcut_retriever.bakeoff as bakeoff
    import matchcut_retriever.models as models

    assert models.Detection is not None
    assert callable(bakeoff.render_sheet)


def _det(area: int, shape=(10, 10), score=1.0):
    mask = np.zeros(shape, dtype=bool)
    mask.flat[:area] = True
    from matchcut_retriever.models import Detection

    return Detection(mask, score)


def test_filter_detections_bounds_sort_limit_and_score():
    from matchcut_retriever.models import filter_detections

    dets = [_det(10, score=0.9), _det(40, score=0.8), _det(1, score=0.99), _det(80, score=0.99)]
    kept = filter_detections(dets, (10, 10), min_area_frac=0.05, max_area_frac=0.6, max_objects=2, min_score=0.7)
    assert [d.mask.sum() for d in kept] == [40, 10]
    assert filter_detections([_det(0)], (10, 10), min_area_frac=0.0) == []
    assert filter_detections([_det(20, score=0.1)], (10, 10), min_score=0.2) == []


def test_point_grid_shape_and_range():
    from matchcut_retriever.models import point_grid

    points = point_grid(4)
    assert points.shape == (16, 2)
    assert np.all(points > 0) and np.all(points < 1)
    assert np.allclose(points.min(axis=0), 0.125)
    assert np.allclose(points.max(axis=0), 0.875)


def test_stability_score_hand_built_logits():
    from matchcut_retriever.models import stability_score

    logits = np.array([[[2.0, 0.5], [-2.0, 0.0]], [[2.0, 2.0], [2.0, 2.0]]])
    # high (>1) and low (>-1): first mask has one pixel in intersection and
    # three in union; second mask is stable everywhere.
    assert np.allclose(stability_score(logits), [1 / 3, 1.0])


def test_mask_nms_overlap_disjoint_and_empty():
    from matchcut_retriever.models import mask_nms

    first = np.zeros((6, 6), dtype=bool)
    first[:3, :3] = True
    overlap = first.copy()
    overlap[3, 3] = True
    disjoint = np.zeros((6, 6), dtype=bool)
    disjoint[4:, 4:] = True
    masks = np.stack([first, overlap, disjoint])
    assert np.array_equal(mask_nms(masks, np.array([0.7, 0.9, 0.4]), 0.5), [1, 2])
    empty = mask_nms(np.empty((0, 6, 6), dtype=bool), np.empty(0), 0.5)
    assert empty.dtype == np.int64 and empty.shape == (0,)


def test_prepare_masked_crop_fill_letterbox_and_empty():
    from matchcut_retriever.models import prepare_masked_crop

    image = np.full((10, 20, 3), 10, dtype=np.uint8)
    mask = np.zeros((10, 20), dtype=bool)
    mask[2:8, 7:13] = True
    crop = prepare_masked_crop(image, mask, pad=0, fill=(127, 127, 127), size=20)
    assert crop.shape == (20, 20, 3) and crop.dtype == np.uint8
    # The 6x6 crop is square and entirely masked, while the image itself has
    # no mask pixels outside it. An empty mask is an all-fill crop.
    assert np.all(crop == 10)
    empty = prepare_masked_crop(image, np.zeros_like(mask), fill=(7, 8, 9), size=12)
    assert np.all(empty == np.array([7, 8, 9], dtype=np.uint8))

    # A wide non-square masked bbox must be letterboxed vertically, preserving
    # its aspect ratio rather than stretching it to the output square.
    wide = np.zeros((10, 20), dtype=bool)
    wide[3:7, 4:16] = True
    wide_image = np.full((10, 20, 3), 200, dtype=np.uint8)
    wide_crop = prepare_masked_crop(wide_image, wide, pad=0, fill=(1, 2, 3), size=20)
    assert np.all(wide_crop[:6] == np.array([1, 2, 3], dtype=np.uint8))
    assert np.all(wide_crop[-6:] == np.array([1, 2, 3], dtype=np.uint8))
    assert np.all(wide_crop[6:13] == 200)


def test_sheet_rendering_synthetic_image_and_detections():
    from matchcut_retriever.bakeoff import render_sheet
    from matchcut_retriever.models import Detection

    image = np.zeros((12, 16, 3), dtype=np.uint8)
    mask = np.zeros((12, 16), dtype=bool)
    mask[2:8, 3:9] = True
    sheet = render_sheet(image, {"mask2former": [Detection(mask, 0.8)]})
    assert sheet.shape[1:] == (32, 3)
    assert sheet.shape[0] > image.shape[0]
    assert np.any(sheet[:, 16:] != 0)


@pytest.mark.parametrize("wrapper", ["PanopticSegmenter", "SamAutoSegmenter", "SamPointSegmenter", "ClipEmbedder"])
def test_model_wrappers_require_gpu_extra(monkeypatch, wrapper):
    import matchcut_retriever.models as models

    original_import = builtins.__import__

    def deny_gpu(name, *args, **kwargs):
        if name in {"torch", "transformers"}:
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny_gpu)
    with pytest.raises(ImportError, match="gpu extra"):
        getattr(models, wrapper)()


def test_projected_image_features_handles_tensor_and_output_objects():
    from types import SimpleNamespace

    from matchcut_retriever.models.clip import _projected_image_features

    class FakeTensor:
        pass

    fake_torch = SimpleNamespace(Tensor=FakeTensor)
    tensor = FakeTensor()
    assert _projected_image_features(tensor, fake_torch) is tensor
    pooled = object()
    assert _projected_image_features(SimpleNamespace(pooler_output=pooled), fake_torch) is pooled
    embeds = object()
    assert _projected_image_features(SimpleNamespace(image_embeds=embeds, pooler_output=pooled), fake_torch) is embeds
    with pytest.raises(RuntimeError):
        _projected_image_features(SimpleNamespace(), fake_torch)


def test_saliency_model_predict_instantiates_transforms_and_batches(monkeypatch):
    import matchcut_retriever.models.saliency as saliency

    class FakeBatch:
        def __init__(self, value):
            self.value = value

        def to(self, device):
            return self

        def float(self):
            return self

    class FakePred:
        def __init__(self, value):
            self.value = value

        def sigmoid(self):
            return self

        def __getitem__(self, key):
            return FakePred(self.value[key])

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class FakeTorch:
        @staticmethod
        def stack(values):
            return FakeBatch(np.stack(values))

        @staticmethod
        @contextmanager
        def inference_mode():
            yield

    class FakeModel:
        def __init__(self):
            self.calls = []

        def float(self):
            return self

        def eval(self):
            return self

        def __call__(self, batch):
            self.calls.append(batch.value.shape[0])
            # The wrapper expects (B, 1, S, S) before selecting channel 0.
            return [FakePred(np.full((batch.value.shape[0], 1, 8, 8), 0.5, np.float32))]

    fake_model = FakeModel()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return fake_model

    class FakeNormalize:
        def __init__(self, mean, std):
            self.mean, self.std = mean, std

        def __call__(self, tensor):
            return tensor

    class FakeToTensor:
        def __init__(self):
            pass

        def __call__(self, image):
            return np.asarray(image, dtype=np.float32).transpose(2, 0, 1)

    monkeypatch.setattr(
        saliency,
        "_require_gpu",
        lambda: (FakeTorch, FakeAutoModel, (FakeNormalize, FakeToTensor)),
    )
    model = saliency.SaliencyModel(device="cpu", input_size=8, batch_size=2)
    images = [
        np.zeros((3, 5, 3), dtype=np.uint8),
        np.zeros((7, 2, 3), dtype=np.uint8),
        np.zeros((4, 9, 3), dtype=np.uint8),
    ]
    outputs = model.predict(images)
    assert [output.shape for output in outputs] == [(3, 5), (7, 2), (4, 9)]
    assert all(output.dtype == np.float32 and np.all((output >= 0) & (output <= 1)) for output in outputs)
    assert fake_model.calls == [2, 1]
