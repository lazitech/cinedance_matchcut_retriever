# CineDance Match-Cut Retriever

Click a subject in any image; get back movie shots from CineDance whose objects have a similar
shape, so the two can be joined with a match cut. Part of the CS180 final project
*Berkeley and World Cinema Through Match Cuts* (see `docs/proposal.md`).

Status: stage 7 (Gradio UI) on the 500-clip development library. The design and roadmap are in
`CLAUDE.md`.

## Setup

```bash
uv sync                              # core deps + pytest (unit tests only)
uv sync --extra gpu --extra app      # models (SAM, CLIP, Mask2Former) + Gradio UI
cp configs/local.example.toml configs/local.toml   # then edit paths
uv run pytest
```

The `gpu` extra holds the model dependencies (torch, transformers); they also run on CPU, so the
UI and the query CLI work without a GPU (`--device cpu`, slower). Building the library needs a GPU.

On a host that cannot reach GitHub, wrap network commands with the temporary proxy:
`python3 scripts/proxy_run.py -- git push`.

## Data

Uses CineDance-1M (gated, CC-BY-NC-SA). No video, frames, masks, annotations or features are
committed to this repository.

## Gradio UI

Queries load SAM and CLIP, so install both extras (`uv sync --extra gpu --extra app`). After
building or obtaining a retrieval library, launch the query interface with:

```bash
.venv/bin/python -m matchcut_retriever.app --config configs/local.toml   # --device cuda:0 for a GPU
```

The interface opens at `http://127.0.0.1:7860`. Upload an image, click the subject, choose a
SAM mask, adjust ranking controls, and press **Search**. To expose a temporary public share link
from the server, run the proxy wrapper:

```bash
python3 scripts/proxy_run.py -- .venv/bin/python -m matchcut_retriever.app --config configs/local.toml --share
```

The proxy is temporary and stays up only while the wrapped app process is running; stop the app
to stop the proxy.
