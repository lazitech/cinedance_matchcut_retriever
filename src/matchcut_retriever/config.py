"""Machine-specific paths, read from configs/local.toml (not tracked by git)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "local.toml"


@dataclass(frozen=True)
class Paths:
    video_root: Path
    manifest_dir: Path
    library_root: Path
    hf_cache: Path


def load_paths(config_path: str | os.PathLike | None = None) -> Paths:
    """Load paths from `config_path`, $MCR_CONFIG, or configs/local.toml, in that order."""
    path = Path(config_path or os.environ.get("MCR_CONFIG") or DEFAULT_CONFIG)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; copy configs/local.example.toml to configs/local.toml"
        )
    with path.open("rb") as f:
        raw = tomllib.load(f)["paths"]
    return Paths(**{name: Path(raw[name]).expanduser() for name in Paths.__dataclass_fields__})
