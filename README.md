# Pinglab

Site: [pl.eoinmurray.info](https://pl.eoinmurray.info)

Conductance-based spiking neural networks with explicit excitatory/inhibitory populations and PING (pyramidal–interneuron gamma) dynamics. The site is the project's notebook — the manuscript, method reference, and per-result entries all live there.

## Installation

Prerequisites: [uv](https://docs.astral.sh/uv/) (Python) and [bun](https://bun.sh) (docs site).

```sh
git clone https://github.com/eoinmurray/pinglab.git
cd pinglab

# Python environment + dependencies
uv sync

# Docs site dependencies
cd src/docs && bun install
```

## Tests

```sh
uv run pytest                  # full suite
uv run pytest -m "not slow"    # skip slow end-to-end tests
```

## License

MIT — see `LICENSE`.
