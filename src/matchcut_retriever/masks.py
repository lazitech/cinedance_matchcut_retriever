"""Compact, lossless run-length encoding for binary masks.

The wire format starts with ``MKR1``, little-endian uint32 height and width,
and one byte for the value of the first run. The remainder contains unsigned
varints for each run length in row-major (C) order. This keeps uniform masks
very small while retaining exact boolean pixels.
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
    """Encode a 2-D mask in row-major ``(y, x)`` pixel order.

    ``mask`` is interpreted as a boolean array with rows indexed by ``y`` and
    columns indexed by ``x``. The original ``(height, width)`` shape is stored
    in the returned bytes, so :func:`rle_decode` reconstructs the exact mask.
    Run detection is vectorised with NumPy; Python iteration is only used to
    write one varint per run.
    """
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("mask must be a 2-D array")
    if array.shape[0] > 0xFFFFFFFF or array.shape[1] > 0xFFFFFFFF:
        raise ValueError("mask dimensions exceed encoding limits")

    bits = array.astype(bool, copy=False).ravel(order="C")
    total = int(bits.size)
    first_value = int(bits[0]) if total else 0

    if total:
        # Transition indices delimit runs; this avoids a Python loop over pixels.
        transitions = np.flatnonzero(bits[1:] != bits[:-1]) + 1
        ends = np.concatenate(
            (transitions, np.array([total], dtype=np.int64))
        )
        starts = np.concatenate(
            (np.array([0], dtype=np.int64), transitions)
        )
        run_lengths = (ends - starts).astype(np.int64, copy=False)
    else:
        run_lengths = np.empty(0, dtype=np.int64)

    payload = b"".join(
        _put_varint(int(length)) for length in run_lengths
    )
    header = _HEADER.pack(
        _MAGIC,
        int(array.shape[0]),
        int(array.shape[1]),
        first_value,
    )
    return header + payload


def _get_varint(data: memoryview, offset: int) -> tuple[int, int]:
    """Read one uint32-sized unsigned varint from ``data``."""
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
    """Decode RLE bytes into a boolean mask with row-major ``(y, x)`` pixels.

    The returned array has the encoded ``(height, width)`` shape. Pixel row
    ``y`` and column ``x`` correspond to the same coordinates used by
    :func:`rle_encode`; malformed or truncated encodings raise ``ValueError``.
    """
    raw = memoryview(data)
    if len(raw) < _HEADER.size:
        raise ValueError("truncated RLE header")

    magic, height, width, first_value = _HEADER.unpack(raw[: _HEADER.size])
    if magic != _MAGIC:
        raise ValueError("invalid RLE magic")
    if first_value not in (0, 1):
        raise ValueError("invalid initial RLE value")

    total = int(height) * int(width)
    if total == 0:
        if len(raw) != _HEADER.size:
            raise ValueError("unexpected data for empty mask")
        return np.zeros((height, width), dtype=bool)

    run_lengths: list[int] = []
    offset = _HEADER.size
    consumed = 0
    while consumed < total:
        length, offset = _get_varint(raw, offset)
        if length <= 0 or length > total - consumed:
            raise ValueError("invalid RLE run length")
        run_lengths.append(length)
        consumed += length

    if offset != len(raw):
        raise ValueError("trailing bytes in RLE data")

    lengths = np.asarray(run_lengths, dtype=np.int64)
    values = (np.arange(lengths.size, dtype=np.int8) & 1) ^ np.int8(
        first_value
    )
    flat = np.repeat(values.astype(bool), lengths)
    return flat.reshape((height, width), order="C")


__all__ = ["rle_encode", "rle_decode"]
