"""Process-level memory settings for loading large libraries."""

from __future__ import annotations

import ctypes
import logging
import sys

LOGGER = logging.getLogger(__name__)

PR_SET_THP_DISABLE = 41
PR_GET_THP_DISABLE = 42


def disable_transparent_hugepages() -> bool:
    """Opt this process (and its children) out of transparent huge pages.

    numpy madvises huge pages for large arrays.  On a long-running host with
    fragmented memory, loading a multi-GB library then stalls in kernel direct
    compaction (observed: two concurrent preloads spent >1000 s in sys time and
    never finished; with THP disabled each took ~113 s).  ``prctl`` needs no
    root and is inherited across fork/exec.  Returns True when the flag is set;
    a no-op returning False on non-Linux systems or if the call fails.
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0:
            LOGGER.debug("prctl(PR_SET_THP_DISABLE) failed: errno %d", ctypes.get_errno())
            return False
    except (OSError, AttributeError) as exc:
        LOGGER.debug("prctl unavailable: %s", exc)
        return False
    return True


def transparent_hugepages_disabled() -> bool | None:
    """Return the process THP-disable flag, or None when it cannot be read."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        value = ctypes.CDLL(None, use_errno=True).prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0)
    except (OSError, AttributeError):
        return None
    return None if value < 0 else bool(value)
