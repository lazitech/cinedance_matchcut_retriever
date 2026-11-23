"""Lazy wrappers around Mask2Former and SAM2 from transformers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .common import Detection, mask_nms, point_grid


def _require_gpu() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ImportError(
            "Model wrappers require the optional 'gpu' extra; install the gpu extra on the GPU server"
        ) from exc
    return torch, transformers


def _move_inputs(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    return inputs


class PanopticSegmenter:
    name = "mask2former"

    def __init__(self, model_id: str = "facebook/mask2former-swin-large-coco-panoptic", device: str = "cuda", cache_dir=None, batch_size: int = 8):
        torch, transformers = _require_gpu()
        self._torch = torch
        self.model_id = str(model_id)
        self.device = device
        self.batch_size = int(batch_size)
        self.params = {"device": self.device, "batch_size": self.batch_size}
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.processor = transformers.AutoImageProcessor.from_pretrained(self.model_id, cache_dir=cache_dir)
        self.model = transformers.Mask2FormerForUniversalSegmentation.from_pretrained(self.model_id, cache_dir=cache_dir).to(device).eval()

    def _segment_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        inputs = _move_inputs(self.processor(images=images, return_tensors="pt"), self.device)
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = [(int(image.shape[0]), int(image.shape[1])) for image in images]
        results = self.processor.post_process_panoptic_segmentation(outputs, target_sizes=target_sizes)
        detections: list[list[Detection]] = []
        for result, image in zip(results, images):
            segmentation = result["segmentation"].detach().cpu().numpy() if hasattr(result["segmentation"], "detach") else np.asarray(result["segmentation"])
            image_dets: list[Detection] = []
            for info in result.get("segments_info", []):
                segment_id = int(info["id"])
                label_id = int(info.get("label_id", -1))
                id2label = getattr(self.model.config, "id2label", {})
                label = id2label.get(label_id, id2label.get(str(label_id), str(label_id)))
                image_dets.append(Detection(segmentation == segment_id, float(info.get("score", 1.0)), label, self.name))
            detections.append(image_dets)
        return detections

    def segment(self, images: list[np.ndarray]) -> list[list[Detection]]:
        return [det for start in range(0, len(images), self.batch_size) for det in self._segment_batch(images[start:start + self.batch_size])]


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


class _SamBase:
    def __init__(self, model_id: str, device: str, cache_dir):
        torch, transformers = _require_gpu()
        self._torch = torch
        self.model_id = str(model_id)
        self.device = device
        self.processor = transformers.Sam2Processor.from_pretrained(self.model_id, cache_dir=cache_dir)
        self.model = transformers.Sam2Model.from_pretrained(self.model_id, cache_dir=cache_dir).to(device).eval()

    def _embedding(self, image: np.ndarray):
        inputs = _move_inputs(self.processor(images=image, return_tensors="pt"), self.device)
        with self._torch.inference_mode():
            return self.model.get_image_embeddings(pixel_values=inputs["pixel_values"]), inputs

    def _predict(self, embedding: Any, image: np.ndarray, points_xy: np.ndarray, labels: np.ndarray | None = None):
        # The processor performs the model's resize/coordinate transform.  Its
        # pixel values are deliberately ignored here: the image embedding is
        # computed once and reused for every prompt batch.
        prompt_inputs = _move_inputs(
            self.processor(images=image, input_points=points_xy.tolist(), input_labels=(labels.tolist() if labels is not None else None), return_tensors="pt"),
            self.device,
        )
        batch = int(prompt_inputs["input_points"].shape[0])
        emb = embedding.repeat(batch, 1, 1, 1) if hasattr(embedding, "repeat") else embedding
        with self._torch.inference_mode():
            try:
                outputs = self.model(image_embeddings=emb, input_points=prompt_inputs["input_points"], input_labels=prompt_inputs.get("input_labels"), multimask_output=True)
            except TypeError:
                outputs = self.model(image_embeddings=emb, input_points=prompt_inputs["input_points"], input_labels=prompt_inputs.get("input_labels"))
        logits = getattr(outputs, "pred_masks", None)
        if logits is None:
            logits = getattr(outputs, "masks", None)
        if logits is None:
            raise RuntimeError("Sam2Model output has no pred_masks/masks field")
        logits_np = _as_numpy(logits)
        try:
            # This removes the processor's square-image padding before
            # upscaling, so masks retain the original aspect ratio.  Keeping
            # logits (binarize=False) also makes stability filtering faithful
            # to SAM's low/high-threshold definition.
            processed = self.processor.post_process_masks(
                logits.detach().cpu() if hasattr(logits, "detach") else logits,
                prompt_inputs["original_sizes"],
                binarize=False,
            )
            if isinstance(processed, (list, tuple)):
                logits_np = _as_numpy(processed[0])
            else:
                logits_np = _as_numpy(processed)
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
            pass
        if logits_np.ndim == 5:
            if logits_np.shape[0] == 1:
                logits_np = logits_np[0]
            elif logits_np.shape[1] == 1:
                logits_np = logits_np[:, 0]
            else:
                logits_np = logits_np.reshape((-1,) + logits_np.shape[2:])
        if logits_np.ndim == 3:
            logits_np = logits_np[:, None]
        scores = getattr(outputs, "iou_scores", None)
        if scores is None:
            scores = getattr(outputs, "predicted_iou_scores", None)
        scores_np = np.ones(logits_np.shape[:2], dtype=np.float32) if scores is None else _as_numpy(scores)
        scores_np = np.asarray(scores_np).reshape(logits_np.shape[0], -1)
        height, width = image.shape[:2]
        masks = np.empty((logits_np.shape[0], logits_np.shape[1], height, width), dtype=bool)
        from PIL import Image
        for i in range(logits_np.shape[0]):
            for j in range(logits_np.shape[1]):
                small = (logits_np[i, j] > 0).astype(np.uint8) * 255
                masks[i, j] = np.asarray(Image.fromarray(small).resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8) > 0
        return masks, scores_np, logits_np


class SamAutoSegmenter(_SamBase):
    name = "sam2_auto"

    def __init__(self, model_id: str = "facebook/sam2.1-hiera-large", device: str = "cuda", cache_dir=None, points_per_side: int = 16, pred_iou_thresh: float = 0.8, stability_thresh: float = 0.9, nms_iou: float = 0.7, points_per_batch: int = 64):
        super().__init__(model_id, device, cache_dir)
        self.points_per_side = int(points_per_side)
        self.pred_iou_thresh = float(pred_iou_thresh)
        self.stability_thresh = float(stability_thresh)
        self.nms_iou = float(nms_iou)
        self.points_per_batch = int(points_per_batch)
        self.params = {
            "points_per_side": self.points_per_side,
            "pred_iou_thresh": self.pred_iou_thresh,
            "stability_thresh": self.stability_thresh,
            "nms_iou": self.nms_iou,
            "points_per_batch": self.points_per_batch,
        }
        if self.points_per_batch <= 0:
            raise ValueError("points_per_batch must be positive")

    def _filtered_batch(self, embedding: Any, image: np.ndarray, batch_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict one prompt batch and return (bool masks, scores) passing both thresholds.

        Thresholds are applied on the GPU to the low-resolution logits, and
        only the surviving masks are upsampled to the image size, so the CPU
        never touches the (points x 3) full-resolution logit stack.
        """
        torch = self._torch
        # Sam2Processor expects (image, object, point, coordinate) and
        # (image, object, point) dimensions when batching independent point
        # prompts for one image embedding.
        labels = np.ones((1, len(batch_points), 1), dtype=np.int64)
        prompt_inputs = _move_inputs(
            self.processor(images=image, input_points=batch_points[None, :, None, :].tolist(), input_labels=labels.tolist(), return_tensors="pt"),
            self.device,
        )
        with torch.inference_mode():
            outputs = self.model(
                image_embeddings=embedding,
                input_points=prompt_inputs["input_points"],
                input_labels=prompt_inputs["input_labels"],
                multimask_output=True,
            )
            logits = outputs.pred_masks  # (1, points, 3, h, w) low-resolution logits
            scores = outputs.iou_scores  # (1, points, 3)
            logits = logits.reshape(-1, *logits.shape[-2:])
            scores = scores.reshape(-1).float()
            # SAM stability score at threshold 0 with offset 1: the mask at
            # logit > 1 is a subset of the mask at logit > -1, so IoU = high / low.
            high = (logits > 1.0).flatten(1).sum(1).float()
            low = (logits > -1.0).flatten(1).sum(1).float()
            stability = torch.where(low > 0, high / low.clamp_min(1.0), torch.ones_like(low))
            keep = (scores >= self.pred_iou_thresh) & (stability >= self.stability_thresh)
            if not bool(keep.any()):
                return np.empty((0,) + image.shape[:2], dtype=bool), np.empty(0, dtype=np.float32)
            # post_process_masks interpolates masks[i] per image, so it needs (image, object, mask, h, w).
            kept_logits = logits[keep][None, :, None]
            kept_scores = scores[keep]
            original_sizes = prompt_inputs["original_sizes"]
            # post_process_masks removes the processor's resize and returns
            # logits at the original image size; it runs on the logits' device.
            processed = self.processor.post_process_masks(kept_logits, original_sizes, binarize=False)
            full = processed[0] if isinstance(processed, (list, tuple)) else processed
            full = full.reshape(-1, *full.shape[-2:])
            masks = (full > 0).cpu().numpy()
        height, width = image.shape[:2]
        if masks.shape[-2:] != (height, width):
            raise RuntimeError(f"post-processed SAM masks have shape {masks.shape[-2:]}, expected {(height, width)}")
        return masks, kept_scores.cpu().numpy().astype(np.float32)

    def _segment_one(self, image: np.ndarray) -> list[Detection]:
        embedding, _ = self._embedding(image)
        h, w = image.shape[:2]
        grid = point_grid(self.points_per_side)
        points = grid * np.array([w, h], dtype=np.float32)
        all_masks: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        for start in range(0, len(points), self.points_per_batch):
            masks, scores = self._filtered_batch(embedding, image, points[start:start + self.points_per_batch])
            all_masks.append(masks)
            all_scores.append(scores)
        mask_arr = np.concatenate(all_masks) if all_masks else np.empty((0, h, w), dtype=bool)
        score_arr = np.concatenate(all_scores) if all_scores else np.empty(0, dtype=np.float32)
        if len(mask_arr) == 0:
            return []
        kept = mask_nms(mask_arr, score_arr, self.nms_iou)
        return [Detection(mask_arr[i], float(score_arr[i]), None, self.name) for i in kept]

    def segment(self, images: list[np.ndarray]) -> list[list[Detection]]:
        return [self._segment_one(image) for image in images]


class SamPointSegmenter(_SamBase):
    name = "sam2_point"

    def __init__(self, model_id: str = "facebook/sam2.1-hiera-large", device: str = "cuda", cache_dir=None):
        super().__init__(model_id, device, cache_dir)

    def masks_at_point(self, image: np.ndarray, x: float, y: float) -> list[Detection]:
        embedding, _ = self._embedding(image)
        points = np.asarray([[[[float(x), float(y)]]]], dtype=np.float32)
        labels = np.ones((1, 1, 1), dtype=np.int64)
        masks, scores, _ = self._predict(embedding, image, points, labels)
        order = np.argsort(-scores[0], kind="stable")
        return [Detection(masks[0, i], scores[0, i], None, self.name) for i in order]
