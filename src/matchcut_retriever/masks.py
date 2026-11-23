"""Compact, lossless run-length encoding for binary masks.

The wire format is intentionally small and self contained.  It starts with
``MKR1`` followed by little-endian uint32 height and width and one byte for
the value of the first run.  The remainder is an unsigned varint for each
run length, in row-major (C) order.  Run lengths are encoded as varints so
large uniform regions remain very compact while arbitrary masks incur only a
small overhead at transitions.
"""

from __future__ import annotations

import struct

import numpy as np


_MAGIC = b"MKR1"
_HEADER = struct.Struct("<4sIIB")


def _put_varint(value: int) -> bytes:
    """Encode a non-negative integer as an unsigned LEB128 varint."""
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def rle_encode(mask: np.ndarray) -> bytes:
    """Return a compact lossless encoding of a 2-D boolean mask.

    The shape is stored in the encoding, so :func:`rle_decode` reconstructs
    the original ``H x W`` array.  Run detection is vectorised with NumPy;
    the only Python iteration is over runs while writing their varints.
    """
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError("mask must be a 2-D array")
    if arr.shape[0] > 0xFFFFFFFF or arr.shape[1] > 0xFFFFFFFF:
        raise ValueError("mask dimensions exceed encoding limits")
    bits = arr.astype(bool, copy=False).ravel(order="C")
    n = int(bits.size)
    first = int(bits[0]) if n else 0
    if n:
        # Transition indices delimit runs; diff on uint8 avoids object loops.
        transitions = np.flatnonzero(bits[1:] != bits[:-1]) + 1
        ends = np.concatenate((transitions, np.array([n], dtype=np.int64)))
        starts = np.concatenate((np.array([0], dtype=np.int64), transitions))
        lengths = (ends - starts).astype(np.int64, copy=False)
    else:
        lengths = np.empty(0, dtype=np.int64)
    payload = b"".join(_put_varint(int(length)) for length in lengths)
    return _HEADER.pack(_MAGIC, int(arr.shape[0]), int(arr.shape[1]), first) + payload


def _get_varint(data: memoryview, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    # A uint32-sized mask needs at most five bytes per run length.
    for _ in range(5):
        if offset >= len(data):
            raise ValueError("truncated RLE run length")
        byte = int(data[offset])
        offset += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, offset
        shift += 7
    raise ValueError("invalid RLE varint")


def rle_decode(data: bytes | bytearray | memoryview) -> np.ndarray:
    """Decode bytes produced by :func:`rle_encode` into a bool ``H x W`` mask."""
    raw = memoryview(data)
    if len(raw) < _HEADER.size:
        raise ValueError("truncated RLE header")
    magic, height, width, first = _HEADER.unpack(raw[: _HEADER.size])
    if magic != _MAGIC:
        raise ValueError("invalid RLE magic")
    if first not in (0, 1):
        raise ValueError("invalid initial RLE value")
    total = int(height) * int(width)
    if total == 0:
        if len(raw) != _HEADER.size:
            raise ValueError("unexpected data for empty mask")
        return np.zeros((height, width), dtype=bool)

    runs: list[int] = []
    offset = _HEADER.size
    consumed = 0
    while consumed < total:
        length, offset = _get_varint(raw, offset)
        if length <= 0 or length > total - consumed:
            raise ValueError("invalid RLE run length")
        runs.append(length)
        consumed += length
    if offset != len(raw):
        raise ValueError("trailing bytes in RLE data")

    lengths = np.asarray(runs, dtype=np.int64)
    values = (np.arange(lengths.size, dtype=np.int8) & 1) ^ np.int8(first)
    flat = np.repeat(values.astype(bool), lengths)
    return flat.reshape((height, width), order="C")


__all__ = ["rle_encode", "rle_decode"]
