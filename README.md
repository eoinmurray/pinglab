# Pinglab

Site: [pl.eoinmurray.info](https://pl.eoinmurray.info)

Conductance-based spiking neural networks with explicit excitatory/inhibitory populations and PING (pyramidal–interneuron gamma) dynamics. The site is the project's notebook — the manuscript, method reference, and per-result entries all live there.

## Installation

Prerequisites: [uv](https://docs.astral.sh/uv/) (Python), [typst](https://typst.app) and [task](https://taskfile.dev) (publishing).

```sh
git clone https://github.com/eoinmurray/pinglab.git
cd pinglab

# Python environment + dependencies
uv sync --dev
```

## Publishing

The site is built from Typst sources (`writings/`) with the vendored demolab engine:

```sh
task build    # → artifacts/site/ (web) + artifacts/pdfs/ (per-entry PDFs + book.pdf)
task dev      # hot-reloading preview on :3000
```

## Tests

```sh
task test:fast                 # lint + typecheck + quick tests
uv run pytest                  # full suite
uv run pytest -m "not slow"    # skip slow end-to-end tests
```

## License

MIT — see `LICENSE`.
