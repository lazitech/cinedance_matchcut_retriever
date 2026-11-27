"""Result-grid rendering utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .shape import cover_fit_matrix


def _outline(image: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    out = np.asarray(image, dtype=np.uint8).copy()
    contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.polylines(out, contours, True, colour, 3, cv2.LINE_AA)
    return out


def _fit(image: np.ndarray, size: tuple[int, int], background=(30, 30, 30)) -> np.ndarray:
    width, height = size
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _font(size=20):
    for name in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def result_panel(library, prepared, query_mask, result) -> np.ndarray:
    """Render one candidate keyframe above its aligned overlay.

    This is the compact panel used by the Gradio query UI and human-evaluation
    tool.  It deliberately contains no captions so callers can choose whether
    metadata is appropriate for their audience.
    """
    candidate = np.asarray(Image.open(library.keyframe_path(result.keyframe_id)).convert("RGB"))
    candidate_mask = np.asarray(library.mask(result.object_id), dtype=bool)
    query = np.asarray(prepared, dtype=np.uint8)
    query_mask = np.asarray(query_mask, dtype=bool)
    if query.ndim != 3 or query_mask.shape != query.shape[:2]:
        raise ValueError("prepared and query_mask dimensions do not match")
    cand = _outline(candidate, candidate_mask, (255, 170, 40))
    cover = cover_fit_matrix((candidate.shape[1], candidate.shape[0]), (query.shape[1], query.shape[0]))
    cover_h = np.vstack([cover, [0, 0, 1]])
    transform_h = np.vstack([result.transform.matrix((query.shape[1], query.shape[0])), [0, 0, 1]])
    mat = (transform_h @ cover_h)[:2]
    aligned = cv2.warpAffine(candidate, mat, (query.shape[1], query.shape[0]), borderValue=(25, 25, 25))
    warped = cv2.warpAffine(candidate_mask.astype(np.uint8), mat, (query.shape[1], query.shape[0]), flags=cv2.INTER_NEAREST).astype(bool)
    overlay = cv2.addWeighted(query, 0.5, aligned, 0.5, 0)
    overlay = _outline(_outline(overlay, query_mask, (50, 220, 80)), warped, (255, 70, 60))
    height, width = 220, 320

    def fit(image):
        scale = min(width / image.shape[1], height / image.shape[0])
        small = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
        out = np.full((height, width, 3), 25, dtype=np.uint8)
        y, x = (height - small.shape[0]) // 2, (width - small.shape[1]) // 2
        out[y:y + small.shape[0], x:x + small.shape[1]] = small
        return out

    return np.concatenate([fit(cand), fit(overlay)], axis=0)


def render_results(query_image, query_mask, results, library, out_path):
    """Write a readable query/candidate/aligned-overlay result grid."""
    query = np.asarray(query_image, dtype=np.uint8)
    mask = np.asarray(query_mask, dtype=bool)
    if query.ndim != 3 or query.shape[:2] != mask.shape:
        raise ValueError("query_image and query_mask dimensions do not match")
    panel_w, panel_h = 320, 240
    rows = max(1, len(results))
    canvas = np.full((rows * (panel_h + 62) + 40, panel_w * 3, 3), 25, dtype=np.uint8)
    # Query is shown once at the top-left and repeated down the first column for scanning.
    query_panel = _fit(_outline(query, mask, (50, 220, 80)), (panel_w, panel_h))
    canvas[20:20 + panel_h, :panel_w] = query_panel
    labels = []
    for rank, result in enumerate(results, 1):
        y0 = 20 + (rank - 1) * (panel_h + 62)
        candidate = np.asarray(Image.open(library.keyframe_path(result.keyframe_id)).convert("RGB"))
        candidate_mask = library.mask(result.object_id)
        candidate_panel = _fit(_outline(candidate, candidate_mask, (255, 170, 40)), (panel_w, panel_h))
        query_wh = (query.shape[1], query.shape[0])
        cover = cover_fit_matrix((candidate.shape[1], candidate.shape[0]), query_wh)
        aligned = cv2.warpAffine(candidate, (result.transform.matrix(query_wh) @ np.vstack([cover, [0, 0, 1]]))[:2], (query.shape[1], query.shape[0]), flags=cv2.INTER_LINEAR, borderValue=(25, 25, 25))
        warped_mask = cv2.warpAffine(candidate_mask.astype(np.uint8), (result.transform.matrix(query_wh) @ np.vstack([cover, [0, 0, 1]]))[:2], (query.shape[1], query.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0).astype(bool)
        overlay = cv2.addWeighted(query, 0.5, aligned, 0.5, 0)
        overlay = _outline(overlay, mask, (50, 220, 80))
        overlay = _outline(overlay, warped_mask, (255, 70, 60))
        canvas[y0:y0 + panel_h, panel_w:panel_w * 2] = candidate_panel
        canvas[y0:y0 + panel_h, panel_w * 2:] = _fit(overlay, (panel_w, panel_h))
        # Two caption lines so the text fits the two result columns; the full
        # clip id is in results.json.
        where = f"#{rank} {result.clip_id[:13]}… shot {result.shot_no}"
        if result.start_s is not None and result.end_s is not None:
            where += f" [{result.start_s:.2f}-{result.end_s:.2f}s]"
        scores = f"score {result.final_score:.3f}  IoU {result.iou:.3f}  cost {result.transform_cost:.3f}"
        if result.clip_cosine is not None:
            scores += f"  CLIP {result.clip_cosine:.3f}"
        labels.append((y0, where, scores))
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)
    for y0, where, scores in labels:
        draw.text((panel_w + 8, y0 + panel_h + 6), where, fill=(240, 240, 240), font=_font(16))
        draw.text((panel_w + 8, y0 + panel_h + 30), scores, fill=(200, 200, 200), font=_font(16))
    draw.text((10, 3), "Query", fill=(240, 240, 240), font=_font(20))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pil_canvas.save(out, quality=94)
    return out


__all__ = ["render_results", "result_panel"]
