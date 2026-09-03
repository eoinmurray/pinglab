# Pinglab

Site: [pl.eoinmurray.info](https://pl.eoinmurray.info)

Conductance-based spiking neural networks with explicit excitatory/inhibitory populations and PING (pyramidal–interneuron gamma) dynamics. The site is the project's notebook — the manuscript, method reference, and per-result entries all live there.

## Architecture

[`AGENTS.md`](AGENTS.md) contains repository instructions for coding agents.
Subsystem guidance lives beside the files it governs in three independently
versioned guides: the [Experiment Runner Guide](experiments/README.md),
the [Storage Guide](tools/pingstore/README.md), and the
[Writing Guide](writings/README.md).

## Installation

Prerequisites: [uv](https://docs.astral.sh/uv/) (Python), [typst](https://typst.app) and [task](https://taskfile.dev) (publishing).

```sh
git clone https://github.com/eoinmurray/pinglab.git
cd pinglab

# Python environment + dependencies
uv sync --dev
```

## Publishing

The site is built from Typst sources (`writings/`) with the installed Demolab engine:

```sh
uv run demolab build    # → .demolab/site/ (PDFs are disabled in demolab.yaml)
uv run demolab dev      # live preview with article-scoped run selectors
```

Experiment reports read explicit Demolab inputs through `writings/dataset-template.typ`.
Without a selected input, they show an unavailable-data notice. A production
build does not inherit preview selections or run experiments; the current
configuration supplies no fixed publication inputs.

## Tests

```sh
task test:fast                 # lint + typecheck + quick tests
uv run pytest                  # full suite
uv run pytest -m "not slow"    # skip slow end-to-end tests
```

## License

MIT — see `LICENSE`.
