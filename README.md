# CineDance Match-Cut Retriever

Click a subject in any image; get back movie shots from CineDance whose objects have a similar
shape, so the two can be joined with a match cut. Part of the CS180 final project
*Berkeley and World Cinema Through Match Cuts* (see `docs/proposal.md`).

Status: stage 7 (Gradio UI) on the 500-clip development library. The design and roadmap are in
`CLAUDE.md`.

## Setup

```bash
uv sync                              # dev machine: core deps + pytest (unit tests only)
uv sync --extra gpu --extra app      # GPU server: models (SAM, CLIP, Mask2Former) + Gradio UI
cp configs/local.example.toml configs/local.toml   # then edit paths
uv run pytest
```

Model inference (library builds, queries, the UI) runs on the GPU server only; the dev machine
runs unit tests on synthetic data and views results. On the server, the UI and the query CLI can
use the CPU (`--device cpu`, the default for the UI) so they need no GPU; building the library does.

On a host that cannot reach GitHub, wrap network commands with the temporary proxy:
`python3 scripts/proxy_run.py -- git push`.

## Data

Uses CineDance-1M (gated, CC-BY-NC-SA). No video, frames, masks, annotations or features are
committed to this repository.

## Building a retrieval library

After object extraction, consolidate the parts and build the PCA shortlist index:

```bash
.venv/bin/python -m matchcut_retriever.library consolidate --config configs/local.toml
.venv/bin/python -m matchcut_retriever.library pca --config configs/local.toml
```

Every `consolidate` creates a new index generation, which makes an older PCA index stale. Re-run
the `pca` command after consolidating before using PCA coarse search.

For the full library, pass `--preload` to the query, app, or self-test command to keep fingerprints
and masks in RAM (about 29 GB, plus PCA projections).

### Subject filter

The subject filter uses BiRefNet saliency to keep library objects that are a main subject of their
frame. It is a hard candidate filter in both soft and hard ranking modes; the query mask is never
filtered. The default threshold is `min_subject_score = 0.5` (set it to `0` to disable filtering).

Run the GPU saliency pass, then publish the consolidated subject index:

```bash
.venv/bin/python -m matchcut_retriever.library saliency --config configs/local.toml --device cuda
.venv/bin/python -m matchcut_retriever.library subject-index --config configs/local.toml
```

The saliency pass writes each part's `saliency.npz` (uint8 maps downsampled to long side 160),
`saliency.json` (model and scoring metadata), and `subject.npy` (float16 per-object mask means).
`subject-index` writes `index/subject.npy` and `index/subject.json`, aligned to `object_id` and tied
to the current index generation. Use the CLI override `query --min-subject SCORE`, or adjust the
**Minimum subject score** slider in the app. If the subject index is absent or stale, retrieval
continues without filtering and reports that state in its results.

## Gradio UI

Queries load SAM and CLIP, so run the UI on the GPU server with both extras installed
(`uv sync --extra gpu --extra app`). After building a retrieval library, launch it with:

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

## Human evaluation

The human-evaluation tool compares the soft and hard ranking modes with a blind Gradio labeling
workflow. Write a TOML query spec containing one `[[query]]` entry per subject (image path, click
coordinates, optional `mask_index`, `group`, `exclude_clips`, and `note`), then run:

```bash
.venv/bin/python -m matchcut_retriever.human_eval prepare --spec queries.toml --out outputs/eval --config configs/local.toml --device cpu
.venv/bin/python -m matchcut_retriever.human_eval pool --out outputs/eval --config configs/local.toml --k 10 --modes soft,hard
.venv/bin/python -m matchcut_retriever.human_eval annotate --out outputs/eval --config configs/local.toml
.venv/bin/python -m matchcut_retriever.human_eval report --out outputs/eval --k 10
```

`prepare` stores the resized query image, selected mask, candidate preview, and `queries.json`.
`pool` writes mode-specific rankings to `rankings.json`, and a shuffled blind candidate list plus
rendered panels to `items.json` and `panels/`. The annotation page stores labels atomically in
`labels.json`; each query records `usable`, `judged`, `note`, and `saved_at`. `report` reads only
saved labels and writes `report.json` and `report.md`, including P@K, Hit@K, group breakdowns,
returned-count precision, per-query counts, mode overlap, and unjudged items.
