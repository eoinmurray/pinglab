---
name: new-experiment
description: Scaffold a new pinglab experiment (post + notebook) with standard files. Use when creating or forking a new experiment, or when asked to make a new post/notebook pair in posts/ and notebooks/.
---

# New Experiment

## Overview

Create a full experiment scaffold (post + notebook) using the standard pinglab layout.

## Workflow

1) Collect inputs
- slug (required): e.g. `exp.4-new-idea`
- title (optional, defaults to slug)
- description (optional, defaults to `Experiment: <slug>`)

2) Run scaffold script from repo root
- Command:
  - `uv run python .codex/skills/new-experiment/scripts/scaffold_experiment.py <slug> --title "..." --description "..."`
- The script writes:
  - `posts/<slug>.mdx`
  - `notebooks/<slug>/config.yaml`
  - `notebooks/<slug>/run.py`
  - `notebooks/<slug>/lib/model.py`
  - `notebooks/<slug>/lib/experiment.py`
  - `notebooks/<slug>/lib/plots.py`

3) Verify
- Confirm the files exist and the post renders (`VeslxGallery` points at `_artifacts/<slug>`).
- Optionally run the notebook: `uv run python notebooks/<slug>/run.py`.
- Ensure `run.py` copies `config.yaml` into the artifacts directory before parsing it.

## Assets

- Templates live in `assets/templates/`:
  - `post.mdx`
  - `config.yaml`
  - `run.py`
  - `lib/model.py`
  - `lib/experiment.py`
  - `lib/plots.py`

## Notes

- Plots use square figures (6x6) by default.
- Defaults are safe placeholders; update config and experiment logic per project needs.
