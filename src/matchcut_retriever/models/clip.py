"""CLIP wrapper and dependency-free masked-crop preparation."""

from __future__ import annotations

from typing import Any

import numpy as np


def _require_gpu() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ImportError(
            "Model wrappers require the optional 'gpu' extra; install the gpu extra on the GPU server"
        ) from exc
    return torch, transformers


class ClipEmbedder:
    name = "clip"

    def __init__(self, model_id: str = "openai/clip-vit-large-patch14", device: str = "cuda", cache_dir=None, batch_size: int = 64):
        torch, transformers = _require_gpu()
        self._torch = torch
        self.device = device
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.processor = transformers.CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = transformers.CLIPModel.from_pretrained(model_id, cache_dir=cache_dir).to(device).eval()

    @staticmethod
    def prepare_masked_crop(image: np.ndarray, mask: np.ndarray, pad: float = 0.1, fill: tuple[int, int, int] = (127, 127, 127), size: int = 224) -> np.ndarray:
        from PIL import Image

        image_arr = np.asarray(image, dtype=np.uint8)
        mask_arr = np.asarray(mask, dtype=bool)
        if image_arr.ndim != 3 or image_arr.shape[2] != 3 or mask_arr.shape != image_arr.shape[:2]:
            raise ValueError("image must be HxWx3 and mask must match its first two dimensions")
        if size <= 0 or pad < 0:
            raise ValueError("size must be positive and pad non-negative")
        ys, xs = np.nonzero(mask_arr)
        canvas = np.empty_like(image_arr)
        canvas[...] = np.asarray(fill, dtype=np.uint8)
        canvas[mask_arr] = image_arr[mask_arr]
        if len(xs) == 0:
            return np.full((size, size, 3), np.asarray(fill, dtype=np.uint8), dtype=np.uint8)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        bw, bh = x1 - x0, y1 - y0
        dx, dy = bw * float(pad), bh * float(pad)
        left, top = max(0, int(np.floor(x0 - dx))), max(0, int(np.floor(y0 - dy)))
        right, bottom = min(image_arr.shape[1], int(np.ceil(x1 + dx))), min(image_arr.shape[0], int(np.ceil(y1 + dy)))
        crop = canvas[top:bottom, left:right]
        ch, cw = crop.shape[:2]
        scale = min(size / cw, size / ch)
        nw, nh = max(1, round(cw * scale)), max(1, round(ch * scale))
        resized = np.asarray(Image.fromarray(crop).resize((nw, nh), Image.Resampling.BILINEAR), dtype=np.uint8)
        out = np.full((size, size, 3), np.asarray(fill, dtype=np.uint8), dtype=np.uint8)
        ox, oy = (size - nw) // 2, (size - nh) // 2
        out[oy:oy + nh, ox:ox + nw] = resized
        return out

    def _features(self, images: list[np.ndarray]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with self._torch.inference_mode():
            if hasattr(self.model, "get_image_features"):
                output = self.model.get_image_features(**inputs)
            else:
                output = self.model(**inputs)
            features = _projected_image_features(output, self._torch)
            features = self._torch.nn.functional.normalize(features, p=2, dim=-1)
        return features.detach().float().cpu().numpy().astype(np.float32)

    def embed_masked(self, images: list[np.ndarray], masks: list[np.ndarray]) -> np.ndarray:
        if len(images) != len(masks):
            raise ValueError("images and masks must have the same length")
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        crops = [self.prepare_masked_crop(image, mask) for image, mask in zip(images, masks)]
        return np.concatenate([self._features(crops[start:start + self.batch_size]) for start in range(0, len(crops), self.batch_size)], axis=0)

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate([self._features(images[start:start + self.batch_size]) for start in range(0, len(images), self.batch_size)], axis=0)


def _projected_image_features(output: Any, torch_module: Any) -> Any:
    """Return projected image embeddings from a CLIP output.

    transformers 4.x ``get_image_features`` returns a tensor; 5.x returns a
    ``BaseModelOutputWithPooling`` whose ``pooler_output`` holds the projected
    embedding. A full ``CLIPModel`` forward exposes ``image_embeds``.
    """
    if isinstance(output, torch_module.Tensor):
        return output
    for name in ("image_embeds", "pooler_output"):
        features = getattr(output, name, None)
        if features is not None:
            return features
    raise RuntimeError("CLIP output has no image_embeds/pooler_output field")


# Also expose the pure helper at module level for data-pipeline callers.
prepare_masked_crop = ClipEmbedder.prepare_masked_crop
