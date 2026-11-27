import pandas as pd
import pyarrow.parquet as pq

import matchcut_retriever.library as library_module
from matchcut_retriever.library import LibraryBuilder, consolidate
from matchcut_retriever.parquet_io import read_parquet_buffered

from test_library import FakeEmbedder, FakeSegmenter, _make_root


def _build(tmp_path):
    root = _make_root(tmp_path, clips=("a", "b", "c"), frames_per_clip=2)
    LibraryBuilder(root, [FakeSegmenter("mask2former")], FakeEmbedder()).process_clips(["a", "b", "c"])
    return root


def test_consolidate_writes_large_row_groups_not_one_per_clip(tmp_path):
    root = _build(tmp_path)
    consolidate(root)
    path = root / "objects.parquet"
    assert pq.ParquetFile(path).metadata.num_row_groups == 1
    assert read_parquet_buffered(path).equals(pd.read_parquet(path))


def test_row_group_limit_splits_and_preserves_rows(tmp_path, monkeypatch):
    root = _build(tmp_path)
    consolidate(root)
    expected = pd.read_parquet(root / "objects.parquet")
    monkeypatch.setattr(library_module, "ROW_GROUP_ROWS", 2)
    consolidate(root)
    path = root / "objects.parquet"
    meta = pq.ParquetFile(path).metadata
    assert meta.num_row_groups > 1
    assert all(meta.row_group(i).num_rows <= 2 for i in range(meta.num_row_groups))
    assert read_parquet_buffered(path).equals(expected)


def test_buffered_reader_supports_column_selection(tmp_path):
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "t.parquet"
    frame.to_parquet(path, index=False)
    assert read_parquet_buffered(path, columns=["b"]).equals(frame[["b"]])
