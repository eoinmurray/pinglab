# Pinglab

Site: [pl.eoinmurray.info](https://pl.eoinmurray.info)

Spiking neural networks with explicit excitatory/inhibitory populations and PING (pyramidal–interneuron gamma) dynamics, trained with surrogate gradients and measured against simpler recurrent baselines.

The shared diagnostic is **Δt-stability**: train at one integration timestep, evaluate at another. Models grounded in continuous-time dynamics generalise across Δt; models that overfit their training step do not.

## Requirements

- [uv](https://docs.astral.sh/uv/) for the Python side
- [bun](https://bun.sh) for the Astro docs site

## Running experiments

The training/inference entry point is `src/pinglab/oscilloscope.py`.

```sh
# See all flags
uv run python src/pinglab/oscilloscope.py --help

# Quick smoke run: single-layer snnTorch on MNIST
uv run python src/pinglab/oscilloscope.py train \
  --model standard-snn --n-hidden 256 --dataset mnist \
  --max-samples 1000 --epochs 3

# PING network with explicit E/I
uv run python src/pinglab/oscilloscope.py train \
  --model ping --n-hidden 256 --ei-strength 0.5 \
  --dataset mnist --max-samples 1000 --epochs 3
```

Run outputs land under `src/artifacts/` (gitignored — reproducible from code + config + seed).

## Notebooks

Each published result has a dedicated runner under `src/pinglab/notebooks/nbNNN.py` and a matching entry under `src/docs/src/pages/notebooks/nbNNN.mdx`. Runners accept only two CLI args — `--tier {extra small, small, medium, large, extra large}` for run size, and `--modal-gpu {none, T4, L4, A10G, A100, H100}` for remote dispatch; every other hyperparameter is a hardcoded literal in the runner so the file *is* the recipe.

```sh
# Run a notebook locally at its default tier
uv run src/pinglab/notebooks/nb005.py

# Faster pass at the smallest tier
uv run src/pinglab/notebooks/nb005.py --tier "extra small"

# Remote dispatch (costs real money — confirm before using)
uv run src/pinglab/notebooks/nb005.py --tier large --modal-gpu L4
```

The runner wipes `src/artifacts/notebooks/<slug>/` and `src/docs/public/figures/notebooks/<slug>/` before each run, then writes figures, a training video, and a `numbers.json` with machine-checked success criteria. The gate's pass/fail badge shows up in the notebook header and on the site homepage.

## Tests

```sh
uv run pytest                         # full unit suite
uv run pytest -k lif                  # filter by name
uv run pytest -m "not slow"           # skip slow end-to-end tests
```

Unit tests live in `src/pinglab/tests/unit/` and cover model forward passes, LIF integrators, metrics, and CLI flag propagation. Slow tests (subprocess or GPU) are marked with `@pytest.mark.slow`.

## Docs site

Findings, method notes, and archived drafts live in the notebook inside `src/docs/`.

```sh
cd src/docs
bun install
bun run dev      # local preview
bun run build    # static build to dist/
```

Deployed via GitHub Pages (see `.github/workflows/pages.yml`).

## Layout

- `src/pinglab/` — Python package: models, training harness, plotting, notebook repro scripts
- `src/docs/` — Astro site (notebook entries, method reference, LLM conventions)
- `src/papers/` — third-party reading list (PDFs gitignored; see `src/papers/README.md` for citations)
- `src/scripts/` — misc one-off scripts

Before making any edits, read `src/docs/src/pages/styleguide.md` — it holds the repo layout, glossary, invariants, and conventions every edit must respect.

## License

MIT — see `LICENSE`.
