#!/usr/bin/env python3
"""Run one command behind a temporary, private Mihomo proxy (for hosts that block GitHub etc.).

    python3 scripts/proxy_run.py [--node N] -- git push
    python3 scripts/proxy_run.py -- uv python install 3.11

Starts ~/clashctl/bin/mihomo with a single node from ~/clashctl/resources/profiles/2.yaml on a
free localhost port, exports http(s)_proxy / all_proxy, sets GIT_SSH_COMMAND to tunnel ssh through
the proxy, runs the command, and always stops the proxy. Nothing about the node is printed or
written outside a temp dir. Works with the system python3 (3.8) as long as PyYAML is installed.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

CLASH = Path(os.environ.get("CLASHCTL_HOME", Path.home() / "clashctl"))
PROFILE = CLASH / "resources/profiles/2.yaml"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start(node_index, work):
    with PROFILE.open() as f:
        nodes = yaml.safe_load(f)["proxies"]
    if not 1 <= node_index <= len(nodes):
        sys.exit("--node must be between 1 and %d" % len(nodes))
    node = nodes[node_index - 1]
    port = free_port()
    conf = {
        "allow-lan": False, "mode": "rule", "log-level": "silent",
        "proxies": [node],
        "listeners": [{"name": "mcr", "type": "mixed", "listen": "127.0.0.1",
                       "port": port, "proxy": node["name"]}],
        "rules": ["MATCH,REJECT"],
    }
    path = Path(work) / "config.json"
    path.write_text(json.dumps(conf))
    proc = subprocess.Popen([str(CLASH / "bin/mihomo"), "-d", work, "-f", str(path)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        if proc.poll() is not None:
            sys.exit("mihomo exited during startup (node %d)" % node_index)
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return proc, port
        except OSError:
            time.sleep(0.1)
    proc.kill()
    sys.exit("timed out starting mihomo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=int, default=1, help="1-based node index in profile 2")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command given")

    with tempfile.TemporaryDirectory(prefix="mcr-mihomo-") as work:
        proc, port = start(a.node, work)
        url = "http://127.0.0.1:%d" % port
        env = dict(os.environ)
        for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            env[k] = url
        env["no_proxy"] = env["NO_PROXY"] = "localhost,127.0.0.1"
        env["GIT_SSH_COMMAND"] = "ssh -o ProxyCommand='nc -X connect -x 127.0.0.1:%d %%h %%p'" % port
        try:
            rc = subprocess.call(cmd, env=env)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    sys.exit(rc)


if __name__ == "__main__":
    main()
