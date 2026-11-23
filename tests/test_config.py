from pathlib import Path

import pytest

from matchcut_retriever.config import REPO_ROOT, load_paths


def test_example_config_loads():
    paths = load_paths(REPO_ROOT / "configs" / "local.example.toml")
    assert paths.library_root == Path("/data2/lvanzhe/matchcut_library/dev500")
    assert paths.annotation_dir.name == "annotations"
    assert paths.video_root.name == "video_live"


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_paths(tmp_path / "nope.toml")


def test_env_var_overrides_default(tmp_path, monkeypatch):
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        '[paths]\nvideo_root="/v"\nmanifest_dir="/m"\nannotation_dir="/a"\n'
        'library_root="/l"\nhf_cache="/h"\n'
    )
    monkeypatch.setenv("MCR_CONFIG", str(cfg))
    assert load_paths().library_root == Path("/l")
