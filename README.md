# Pinglab

Site: [pinglab.eoinmurray.info](https://pinglab.eoinmurray.info)

Conductance-based spiking neural networks with explicit excitatory/inhibitory populations and PING (pyramidal–interneuron gamma) dynamics. The site is the project's notebook — the manuscript, method reference, and per-result entries all live there.

## Architecture

[`AGENTS.md`](AGENTS.md) contains repository instructions for coding agents.
Subsystem guidance lives beside the files it governs in three independently
versioned guides: the [Experiment Runner Guide](experiments/README.md),
the [Storage Guide](tools/pingstore/README.md), and the
[Writing Guide](WRITING_GUIDE.md).

## Installation

Prerequisites: [uv](https://docs.astral.sh/uv/) (Python), and [Bun](https://bun.sh) (Astro publishing).

```sh
git clone https://github.com/eoinmurray/pinglab.git
cd pinglab

# Python environment + dependencies
uv sync --dev
```

## Publishing

Reports are MDX files in `md/experiments/expXXX/`, separate from experiment code
in `experiments/expXXX/`. The Astro site uses Bun:

```sh
cd site
bun install
bun run dev --background   # http://localhost:3000
bun run build              # site/dist
```

For frozen production output and GitHub Pages publication, see the
[local Astro publisher](tools/publishing/README.md). Numerical bindings read
selected Pingstore presentations in TypeScript; runs and experiments are never
launched by a site build. Original Typst writings remain as historical authored
sources.

## Tests

```sh
task test:fast                 # lint + typecheck + quick tests
uv run pytest                  # full suite
uv run pytest -m "not slow"    # skip slow end-to-end tests
```

## License

MIT — see `LICENSE`.

## SNN library

The authoring, simulator and visualisation implementation lives in [snnlab](https://pypi.org/project/snnlab/). `uv sync` installs the PyPI release pinned in `uv.lock`, subject to the version requirement in `pyproject.toml`.

```python
from snnlab import lang, sim, viz
```

For local library development, clone it alongside Pinglab and run `uv add --editable ../snnlab`. Revert that local source override before sharing reproducible experiment configurations. Invoke the simulator with `uv run python -m snnlab.sim` or `uv run snnsim`; the old tool directories have been removed.
