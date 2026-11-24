# CineDance Match-Cut Retriever

Click a subject in any image; get back movie shots from CineDance whose objects have a similar
shape, so the two can be joined with a match cut. Part of the CS180 final project
*Berkeley and World Cinema Through Match Cuts* (see `docs/proposal.md`).

Status: stage 0 (repo and environment). The design and roadmap are in `CLAUDE.md`.

## Setup

```bash
uv sync                      # CPU deps + pytest
uv sync --extra gpu          # on the GPU server
cp configs/local.example.toml configs/local.toml   # then edit paths
uv run pytest
```

On a host that cannot reach GitHub, wrap network commands with the temporary proxy:
`python3 scripts/proxy_run.py -- git push`.

## Data

Uses CineDance-1M (gated, CC-BY-NC-SA). No video, frames, masks, annotations or features are
committed to this repository.

## Gradio UI

After building or obtaining a retrieval library, launch the local query interface with:

```bash
.venv/bin/python -m matchcut_retriever.app --config configs/local.toml
```

The interface opens at `http://127.0.0.1:7860`. Upload an image, click the subject, choose a
SAM mask, adjust ranking controls, and press **Search**. To expose a temporary public share link
from the server, run the proxy wrapper:

```bash
python3 scripts/proxy_run.py -- .venv/bin/python -m matchcut_retriever.app --config configs/local.toml --share
```
