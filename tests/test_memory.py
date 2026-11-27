import subprocess
import sys

import pytest


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="prctl is Linux-only")
def test_disable_transparent_hugepages_sets_flag_and_is_inherited():
    # Run in a subprocess; the flag is inherited, and earlier tests that call
    # Library.load may already have set it on the pytest process, so clear it first.
    code = (
        "import ctypes, subprocess, sys\n"
        "from matchcut_retriever.memory import disable_transparent_hugepages, transparent_hugepages_disabled\n"
        "ctypes.CDLL(None).prctl(41, 0, 0, 0, 0)\n"
        "assert transparent_hugepages_disabled() is False\n"
        "assert disable_transparent_hugepages() is True\n"
        "assert transparent_hugepages_disabled() is True\n"
        "child = subprocess.run([sys.executable, '-c', 'from matchcut_retriever.memory import "
        "transparent_hugepages_disabled as f; print(f())'], capture_output=True, text=True, check=True)\n"
        "assert child.stdout.strip() == 'True', child.stdout\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_noop_off_linux(monkeypatch):
    from matchcut_retriever import memory

    monkeypatch.setattr(memory.sys, "platform", "darwin")
    assert memory.disable_transparent_hugepages() is False
    assert memory.transparent_hugepages_disabled() is None
