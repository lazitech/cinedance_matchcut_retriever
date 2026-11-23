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
