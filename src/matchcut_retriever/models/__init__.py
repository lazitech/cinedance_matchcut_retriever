"""Model wrappers and pure helpers for the match-cut retriever.

The optional GPU dependencies are imported only when a model wrapper is
constructed.  This keeps data preparation and unit tests usable on machines
without torch or transformers installed.
"""

from .common import Detection, filter_detections, mask_nms, point_grid, stability_score
from .segmentation import PanopticSegmenter, SamAutoSegmenter, SamPointSegmenter
from .clip import ClipEmbedder, prepare_masked_crop
from .saliency import SaliencyModel

__all__ = [
    "Detection",
    "filter_detections",
    "point_grid",
    "stability_score",
    "mask_nms",
    "PanopticSegmenter",
    "SamAutoSegmenter",
    "SamPointSegmenter",
    "ClipEmbedder",
    "prepare_masked_crop",
    "SaliencyModel",
]
