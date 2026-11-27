import json
from pathlib import Path

import numpy as np
import pytest

from matchcut_retriever.library import (
    LibraryBuilder,
    _summary,
    build_pca,
    consolidate,
    pca_files,
    verify_library,
)
from matchcut_retriever.retrieval import Library, QueryObject, rank
from matchcut_retriever.retrieval_config import (
    RetrievalConfig,
    load_retrieval_config,
    override_config,
)
from matchcut_retriever.shape import FINGERPRINT_DIM, FINGERPRINT_VERSION, HardLimits

from test_library import FakeEmbedder, FakeSegmenter, _make_root
from test_retrieval import make_library


def _built_root(tmp_path, clips=("a", "b")):
    root = _make_root(tmp_path, clips=clips, frames_per_clip=1)
    LibraryBuilder(root, [FakeSegmenter("mask2former")], FakeEmbedder()).process_clips(list(clips))
    consolidate(root)
    return root


def _query(root, object_id=0, preload=False):
    library = Library.load(root, preload=preload)
    mask = library.mask(object_id)
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    return library, QueryObject.from_mask(image, mask)


def _build_pca(root, dim=8):
    return build_pca(root, dim=dim, sample=200_000, seed=17)


def test_build_pca_files_shapes_dtypes_metadata_orthonormal_and_deterministic(tmp_path):
    root = _built_root(tmp_path)
    first = _build_pca(root, dim=8)
    index = root / "index"
    metadata = json.loads((index / "shape_pca.json").read_text())
    projections_path, basis_path = pca_files(index, metadata)
    projections = np.load(projections_path)
    basis = np.load(basis_path)
    assert projections.shape == (first["n_objects"], 8)
    assert projections.dtype == np.float16
    assert basis.shape == (FINGERPRINT_DIM, 8)
    assert basis.dtype == np.float32
    assert np.allclose(basis.T @ basis, np.eye(8), atol=1e-5)
    assert metadata["dim"] == 8
    assert metadata["sample"] == 200_000
    assert metadata["seed"] == 17
    assert metadata["fingerprint_version"] == FINGERPRINT_VERSION
    assert metadata["generation"] == first["generation"]
    assert 0.0 <= metadata["explained_energy_fraction"] <= 1.0
    before = basis.copy()
    _build_pca(root, dim=8)
    second_meta = json.loads((index / "shape_pca.json").read_text())
    assert np.array_equal(before, np.load(pca_files(index, second_meta)[1]))
    # The rebuild published new arrays and removed the first build's files.
    assert second_meta["projections_file"] != metadata["projections_file"]
    assert not projections_path.exists() and not basis_path.exists()
    assert sorted(p.name for p in index.glob("shape_pca*")) == sorted(
        ["shape_pca.json", second_meta["projections_file"], second_meta["basis_file"]])


@pytest.mark.parametrize("mode", ["soft", "hard"])
def test_pca_and_exact_match_when_shortlist_covers_all(tmp_path, mode):
    root = make_library(tmp_path, [
        ("a", 0, "disc"), ("b", 0, "square"), ("c", 0, "triangle"),
        ("d", 0, "bar"),
    ])
    # The hand-built fixture predates consolidated metadata; add a valid PCA
    # index through the normal build path's metadata contract.
    meta_path = root / "index" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update(generation="fixture", bytes={
        "objects.parquet": (root / "objects.parquet").stat().st_size,
        "index/masks.bin": (root / "index/masks.bin").stat().st_size,
        "index/shape_fp.npy": (root / "index/shape_fp.npy").stat().st_size,
        "index/clip_emb.npy": (root / "index/clip_emb.npy").stat().st_size,
    })
    meta_path.write_text(json.dumps(meta))
    _build_pca(root, dim=16)
    library, query = _query(root)
    config = RetrievalConfig(
        workers=1, coarse_n=100, hard_coarse_n=100,
        coarse_method="exact", hard_limits=HardLimits(max_translate_frac=1.0),
    )
    exact = rank(library, query, mode, params=config, k=20)
    pca = rank(library, query, mode, params=override_config(config, coarse_method="pca", pca_shortlist_factor=1), k=20)
    assert [(r.object_id, r.fingerprint_cosine, r.final_score) for r in pca] == [
        (r.object_id, r.fingerprint_cosine, r.final_score) for r in exact
    ]


def test_small_pca_shortlist_reports_exact_fingerprint_cosine(tmp_path):
    root = make_library(tmp_path, [
        ("a", 0, "disc"), ("b", 0, "square"), ("c", 0, "triangle"),
    ])
    meta_path = root / "index" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update(generation="fixture", bytes={
        "objects.parquet": (root / "objects.parquet").stat().st_size,
        "index/masks.bin": (root / "index/masks.bin").stat().st_size,
        "index/shape_fp.npy": (root / "index/shape_fp.npy").stat().st_size,
        "index/clip_emb.npy": (root / "index/clip_emb.npy").stat().st_size,
    })
    meta_path.write_text(json.dumps(meta))
    _build_pca(root, dim=4)
    library, query = _query(root)
    result = rank(library, query, params=RetrievalConfig(
        workers=1, coarse_n=1, pca_shortlist_factor=1, coarse_method="pca",
    ), k=1)[0]
    expected = float(np.asarray(library.shape_fp[result.object_id], dtype=np.float32) @ query.fingerprint)
    assert result.fingerprint_cosine == pytest.approx(expected)


def test_consolidate_makes_pca_stale_and_auto_falls_back(tmp_path):
    root = _built_root(tmp_path)
    _build_pca(root, dim=4)
    consolidate(root)
    library, query = _query(root)
    assert library.pca_status == "stale"
    exact = rank(library, query, params=RetrievalConfig(workers=1, coarse_method="auto"), k=2)
    assert exact.coarse_method == "exact"
    with pytest.raises(ValueError, match="valid PCA"):
        rank(library, query, params=RetrievalConfig(workers=1, coarse_method="pca"), k=2)
    assert _summary(root)["pca_status"] == "stale"
    verify_library(root)


@pytest.mark.parametrize("kind", ["truncated", "mismatched"])
def test_bad_pca_arrays_are_stale_and_load_does_not_crash(tmp_path, kind):
    root = _built_root(tmp_path)
    _build_pca(root, dim=4)
    projections_path, basis_path = pca_files(root / "index", json.loads((root / "index" / "shape_pca.json").read_text()))
    path = projections_path if kind == "truncated" else basis_path
    if kind == "truncated":
        path.write_bytes(path.read_bytes()[:-32])
    else:
        np.save(path, np.zeros((1, 1), dtype=np.float32))
    library = Library.load(root)
    assert library.pca_status == "stale"
    assert library.shape_pca is None


def test_config_parsing_override_and_dict_validation(tmp_path):
    config_path = tmp_path / "retrieval.toml"
    config_path.write_text("[retrieval]\ncoarse_method='exact'\npca_shortlist_factor=3\n")
    config = load_retrieval_config(config_path)
    assert config.coarse_method == "exact" and config.pca_shortlist_factor == 3
    assert override_config(config, coarse_method="PCA", pca_shortlist_factor=5).coarse_method == "pca"
    bad_method = tmp_path / "bad-method.toml"
    bad_method.write_text("[retrieval]\ncoarse_method='bad'\n")
    with pytest.raises(ValueError, match="coarse_method"):
        load_retrieval_config(bad_method)
    bad_factor = tmp_path / "bad-factor.toml"
    bad_factor.write_text("[retrieval]\npca_shortlist_factor=0\n")
    with pytest.raises(ValueError, match="pca_shortlist_factor"):
        load_retrieval_config(bad_factor)
    with pytest.raises(ValueError):
        RetrievalConfig(pca_shortlist_factor=0)
    root = make_library(tmp_path / "fixture", [("a", 0, "disc")])
    library, query = _query(root)
    with pytest.raises(ValueError, match="coarse_method"):
        rank(library, query, params={"coarse_method": "bad"})


def test_exact_duplicate_fingerprint_tie_order_and_preload_equality(tmp_path):
    root = make_library(tmp_path / "duplicates", [
        ("a", 0, "disc"), ("b", 0, "disc"), ("c", 0, "square"),
    ])
    cold, query = _query(root, preload=False)
    warm, warm_query = _query(root, preload=True)
    config = RetrievalConfig(workers=1, coarse_n=3, coarse_method="exact")
    cold_results = rank(cold, query, params=config, k=3)
    warm_results = rank(warm, warm_query, params=config, k=3)
    assert [r.object_id for r in cold_results][:2] == [0, 1]
    assert [(r.object_id, r.fingerprint_cosine, r.final_score) for r in warm_results] == [
        (r.object_id, r.fingerprint_cosine, r.final_score) for r in cold_results
    ]


def test_failed_pca_rebuild_keeps_previous_index_consistent(tmp_path, monkeypatch):
    root = _built_root(tmp_path)
    _build_pca(root, dim=4)
    index = root / "index"
    before_meta = (index / "shape_pca.json").read_text()
    real_replace = __import__("os").replace

    def failing_replace(src, dst):
        # Fail when publishing the metadata of the second build.
        if str(dst).endswith("shape_pca.json"):
            raise OSError("simulated crash before publish")
        return real_replace(src, dst)

    monkeypatch.setattr("matchcut_retriever.library.os.replace", failing_replace)
    with pytest.raises(OSError):
        _build_pca(root, dim=6)
    monkeypatch.undo()
    assert (index / "shape_pca.json").read_text() == before_meta
    library = Library.load(root)
    assert library.pca_status == "valid" and library.shape_pca.shape[1] == 4
    # No orphaned arrays from the failed build remain.
    meta = json.loads(before_meta)
    assert sorted(p.name for p in index.glob("shape_pca*")) == sorted(
        ["shape_pca.json", meta["projections_file"], meta["basis_file"]])
    assert not list(index.glob(".shape_pca*"))


def test_pca_metadata_rejects_path_like_file_names(tmp_path):
    with pytest.raises(ValueError):
        pca_files(tmp_path, {"projections_file": "../x.npy", "basis_file": "b.npy"})
