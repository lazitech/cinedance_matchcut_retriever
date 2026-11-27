"""Parquet helpers tuned for libraries on spinning disks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Consolidated tables are written in large row groups; one row group per clip (18k+ groups)
# turns a full read into hundreds of thousands of small scattered reads.
ROW_GROUP_ROWS = 1_000_000


def read_parquet_buffered(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a parquet file with one sequential read, then parse it from memory.

    Letting the parquet reader seek per column chunk is extremely slow on a spinning disk when a
    file has many row groups (the full library's objects.parquet took over 15 minutes; reading the
    bytes first takes seconds).
    """
    data = Path(path).read_bytes()
    return pq.read_table(pa.BufferReader(data), columns=columns).to_pandas()


__all__ = ["ROW_GROUP_ROWS", "read_parquet_buffered"]
