"""Configuration for query ranking and result rendering."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path

from .shape import HardLimits, SoftParams


@dataclass(frozen=True)
class RetrievalConfig:
    hard_limits: HardLimits = HardLimits()
    soft_params: SoftParams = SoftParams()
    coarse_n: int = 2000
    hard_coarse_n: int = 500
    workers: int = 8
    k: int = 20
    semantic_weight: float = 0.0
    hard_axis_ecc_threshold: float = 0.8
    # Principal-axis estimates are noisy (several degrees under small edits),
    # so the axis exclusion only drops candidates beyond max_rot_deg + margin.
    hard_axis_margin_deg: float = 10.0
    coarse_method: str = "auto"
    pca_shortlist_factor: int = 20

    def __post_init__(self):
        method = str(self.coarse_method).lower()
        if method not in {"auto", "exact", "pca"}:
            raise ValueError("coarse_method must be one of auto, exact, pca")
        if int(self.pca_shortlist_factor) < 1:
            raise ValueError("pca_shortlist_factor must be >= 1")
        object.__setattr__(self, "coarse_method", method)
        object.__setattr__(self, "pca_shortlist_factor", int(self.pca_shortlist_factor))

    @property
    def min_scale(self): return self.hard_limits.min_scale
    @property
    def max_scale(self): return self.hard_limits.max_scale
    @property
    def max_translate_frac(self): return self.hard_limits.max_translate_frac
    @property
    def max_rot_deg(self): return self.hard_limits.max_rot_deg


def load_retrieval_config(path: str | Path | None = None) -> RetrievalConfig:
    """Load TOML config, falling back to the tracked defaults."""

    target = Path(path) if path else Path(__file__).resolve().parents[2] / "configs" / "retrieval.toml"
    values: dict = {}
    if target.is_file():
        with target.open("rb") as handle:
            raw = tomllib.load(handle)
            values = raw.get("retrieval", raw)
    hard_values = dict(values.get("hard_limits", values.get("hard", {})))
    soft_values = dict(values.get("soft_params", values.get("soft", {})))
    hard_names = {field.name for field in fields(HardLimits)}
    soft_names = {field.name for field in fields(SoftParams)}
    hard = HardLimits(**{name: hard_values.get(name, values.get(name)) for name in hard_names if name in hard_values or name in values})
    soft = SoftParams(**{name: soft_values.get(name, values.get(name)) for name in soft_names if name in soft_values or name in values})
    method = str(values.get("coarse_method", "auto")).lower()
    if method not in {"auto", "exact", "pca"}:
        raise ValueError("coarse_method must be one of auto, exact, pca")
    factor = int(values.get("pca_shortlist_factor", 20))
    if factor < 1:
        raise ValueError("pca_shortlist_factor must be >= 1")
    return RetrievalConfig(
        hard_limits=hard,
        soft_params=soft,
        coarse_n=max(1, int(values.get("coarse_n", 2000))),
        hard_coarse_n=max(1, int(values.get("hard_coarse_n", 500))),
        workers=max(1, int(values.get("workers", 8))),
        k=max(1, int(values.get("k", 20))),
        semantic_weight=float(values.get("semantic_weight", 0.0)),
        hard_axis_ecc_threshold=float(values.get("hard_axis_ecc_threshold", 0.8)),
        hard_axis_margin_deg=max(0.0, float(values.get("hard_axis_margin_deg", 10.0))),
        coarse_method=method,
        pca_shortlist_factor=factor,
    )


def override_config(config: RetrievalConfig, **kwargs) -> RetrievalConfig:
    """Apply CLI style scalar overrides to a config."""
    hard = config.hard_limits
    soft = config.soft_params
    hard_updates = {name: kwargs[name] for name in ("min_scale", "max_scale", "max_translate_frac", "max_rot_deg") if kwargs.get(name) is not None}
    soft_updates = {name: kwargs[name] for name in ("allow_rotation", "ecc_threshold", "cost_weight", "scale_norm", "translate_norm", "rotation_norm_deg") if kwargs.get(name) is not None}
    if hard_updates:
        hard = replace(hard, **hard_updates)
    if soft_updates:
        soft = replace(soft, **soft_updates)
    updates = {name: kwargs[name] for name in ("coarse_n", "hard_coarse_n", "workers", "k", "semantic_weight", "hard_axis_ecc_threshold", "hard_axis_margin_deg", "coarse_method", "pca_shortlist_factor") if kwargs.get(name) is not None}
    if "coarse_method" in updates:
        updates["coarse_method"] = str(updates["coarse_method"]).lower()
        if updates["coarse_method"] not in {"auto", "exact", "pca"}:
            raise ValueError("coarse_method must be one of auto, exact, pca")
    if "pca_shortlist_factor" in updates:
        updates["pca_shortlist_factor"] = int(updates["pca_shortlist_factor"])
        if updates["pca_shortlist_factor"] < 1:
            raise ValueError("pca_shortlist_factor must be >= 1")
    return replace(config, hard_limits=hard, soft_params=soft, **updates)


load_config = load_retrieval_config


__all__ = ["RetrievalConfig", "load_retrieval_config", "load_config", "override_config"]
