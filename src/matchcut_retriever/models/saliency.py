"""Lazy BiRefNet saliency model wrapper."""
from __future__ import annotations
from typing import Any
import numpy as np


def _require_gpu() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForImageSegmentation
        from torchvision.transforms import Normalize, ToTensor
    except ImportError as exc:
        raise ImportError("SaliencyModel requires the optional 'gpu' extra") from exc
    return torch, AutoModelForImageSegmentation, (Normalize, ToTensor)


class SaliencyModel:
    """Run BiRefNet and return one float32 saliency map per RGB image."""
    def __init__(self, model_id: str = "ZhengPeng7/BiRefNet", revision: str = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4", device: str = "cuda", cache_dir=None, input_size: int = 1024, batch_size: int = 8):
        torch, AutoModelForImageSegmentation, transforms = _require_gpu()
        self._torch = torch
        self.model_id, self.revision = str(model_id), str(revision)
        self.device, self.input_size, self.batch_size = str(device), int(input_size), int(batch_size)
        if self.input_size <= 0 or self.batch_size <= 0:
            raise ValueError("input_size and batch_size must be positive")
        self.params = {"device": self.device, "input_size": self.input_size, "batch_size": self.batch_size, "revision": self.revision}
        self.model = AutoModelForImageSegmentation.from_pretrained(self.model_id, trust_remote_code=True, revision=self.revision, cache_dir=cache_dir).float().eval()
        if self.device.startswith("cuda"):
            self.model = self.model.to(self.device).half()
        self._normalize, self._to_tensor = transforms
        self._norm = self._normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def predict(self, images: list[np.ndarray]) -> list[np.ndarray]:
        from PIL import Image
        if not images:
            return []
        outputs: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch_images = images[start:start + self.batch_size]
            tensors = []
            for image in batch_images:
                arr = np.asarray(image, dtype=np.uint8)
                if arr.ndim != 3 or arr.shape[2] != 3:
                    raise ValueError("saliency inputs must be RGB HxWx3 uint8 arrays")
                tensor = self._norm(self._to_tensor(Image.fromarray(arr).resize((self.input_size, self.input_size))))
                tensors.append(tensor)
            batch = self._torch.stack(tensors)
            if self.device.startswith("cuda"):
                batch = batch.to(self.device).half()
            else:
                batch = batch.to(self.device).float()
            with self._torch.inference_mode():
                pred = self.model(batch)[-1].sigmoid()[:, 0]
            pred = pred.detach().float().cpu().numpy()
            for image, sal in zip(batch_images, pred):
                h, w = image.shape[:2]
                restored = np.asarray(Image.fromarray(sal).resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)
                outputs.append(np.clip(restored, 0.0, 1.0))
        return outputs
