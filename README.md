# Pinglab

Conductance-based spiking neural networks with excitatory/inhibitory populations
and PING (pyramidal–interneuron gamma) dynamics.

Site: [pinglab.eoinmurray.info](https://pinglab.eoinmurray.info).

Main documentation lives in [`notes/`](notes/):

1. [Experiment Rules](notes/rules/experiment.md): experiment structure, stages and scientific inputs.
2. [Compute Rules](notes/rules/compute.mdx): compute providers, access and execution workflows.
3. [Pingstore Rules](notes/rules/pingstore.md): run storage, provenance, validation, backups and pruning.
4. [Writing Rules](notes/rules/writing.md): scientific writing, article metadata and editorial conventions.
5. [Canonical Parameters](notes/canonical-parameters.mdx): reference network and training settings for COBA and PING.
6. [COBA and PING: Dynamics and Learning](notes/coba-ping-dynamics.mdx): model and training derivations (formerly exp113).
7. [Mean-field E/I dynamics](notes/mean-field-model.mdx): exp117/exp148 closures, four recurrent pathways, shared input and private drive.
8. [CTC and attention](notes/ctc-attention/index.mdx): communication through coherence and selective routing from two populations into a shared receiver.
9. [Research workflow](notes/workflow.mdx): repository layout and a blueprint for computational research and publishing.
10. [Manuscript](notes/manuscript/index.mdx): synthesis of PING experiments, formerly exp110.

See [Astro setup and GitHub Pages deployment](site/README.md)
for site development and deployment. Coding agents must read
[`AGENTS.md`](AGENTS.md) for task-specific instruction routing.

Use the [Modal dispatcher](libraries/modal_dispatcher/README.md) to launch experiment
compute on Modal and collect validated Pingstore results.

Pingstore has three source modules: [`core.py`](libraries/pingstore/core.py) owns the
storage implementation, [`library.py`](libraries/pingstore/library.py) exposes the
experiment API, and [`cli.py`](libraries/pingstore/cli.py) provides the commands.
Import experiment helpers from `pingstore.library`; run commands with
`uv run pingstore` or `uv run python -m pingstore.cli`.

For generic CSD3 / Wilkes3 experiment deployment, see [HPC deployment](libraries/hpc/README.md).

Python formatting uses [Ruff](https://docs.astral.sh/ruff/formatter/) with an
88-character line length. Function signatures that fit stay on one line; longer
signatures put each parameter on its own indented line, with the closing `):`
on a separate line. A trailing comma keeps an expanded signature multiline.
The formatter also applies its standard layout to calls and collections.

```sh
uv run ruff format         # Apply formatting.
uv run ruff format --check # Check formatting without editing files.
uv run ruff check          # Check lint rules separately.
```

CI checks both lint and formatting.

Install the local Git hook once per checkout after installing development
dependencies:

```sh
uv sync --dev
uv run pre-commit install
```
