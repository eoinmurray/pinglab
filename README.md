# Pinglab

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
  --model snntorch --n-hidden 256 --dataset mnist \
  --max-samples 1000 --epochs 3

# PING network with explicit E/I
uv run python src/pinglab/oscilloscope.py train \
  --model ping --n-hidden 256 --ei-strength 0.5 \
  --dataset mnist --max-samples 1000 --epochs 3
```

Run outputs land under `src/artifacts/` (gitignored — reproducible from code + config + seed).

## Docs site

Findings, method notes, and archived drafts live in the journal inside `src/docs/`.

```sh
cd src/docs
bun install
bun run dev      # local preview
bun run build    # static build to dist/
```

Deployed via GitHub Pages (see `.github/workflows/pages.yml`).

## Layout

- `src/pinglab/` — Python package: models, training harness, plotting, journal repro scripts
- `src/docs/` — Astro site (journal entries, method reference, LLM context)
- `src/papers/` — third-party reading list (PDFs gitignored; see `src/papers/README.md` for citations)
- `src/scripts/` — misc one-off scripts

Before making any edits, read `src/docs/src/pages/llm-context.md` — it holds the repo layout, glossary, invariants, and conventions every edit must respect.

## License

MIT — see `LICENSE`.
