# Pinglab

Conductance-based spiking neural networks with excitatory/inhibitory populations
and PING (pyramidal–interneuron gamma) dynamics.

Site: [pinglab.eoinmurray.info](https://pinglab.eoinmurray.info).

Main documentation lives in [`md/notes/`](md/notes/):

1. [Experiment Rules](md/notes/experiment-rules.md): experiment structure, stages and scientific inputs.
2. [Compute Rules](md/notes/compute-rules.mdx): compute providers, access and execution workflows.
3. [Pingstore Rules](md/notes/pingstore-rules.md): run storage, provenance, validation, backups and pruning.
4. [Writing Rules](md/notes/writing-rules.md): scientific writing, article metadata and editorial conventions.
5. [Canonical Parameters](md/notes/canonical-parameters.mdx): reference network and training settings for COBA and PING.
6. [COBA and PING: Dynamics and Learning](md/notes/coba-ping-dynamics.mdx): model and training derivations (formerly exp113).
7. [Mean-field E/I dynamics](md/notes/mean-field-model.mdx): exp117/exp148 closures, four recurrent pathways, shared input and private drive.

See [Astro setup and GitHub Pages deployment](site/README.md)
for site development and deployment. Coding agents must read
[`AGENTS.md`](AGENTS.md) for task-specific instruction routing.

Use the [Modal dispatcher](tools/modal_dispatcher/README.md) to launch experiment
compute on Modal and collect validated Pingstore results.

For generic CSD3 / Wilkes3 experiment deployment, see [HPC deployment](tools/hpc/README.md).

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

Each commit checks staged Python files for lint and formatting; either failure
rejects the commit. Partially staged files are checked using their staged content,
and excluded generated directories follow the Ruff configuration. The hook checks
without automatically fixing or staging files. Run `uv run pre-commit run --all-files`
to check every tracked Python file manually. New clones need the installation
command too; Git does not copy installed hooks.

Readability checks enable modern Python syntax (`UP`), simpler control flow
(`SIM`), comprehensions (`C4`), consistent returns (`RET`), common Python pitfalls
(`B`), redundant constructs (`PIE`), and cyclomatic complexity (`C901`, maximum 15).
Complexity counts decision
paths, rather than lines. `C408` is excluded to preserve the `dict(key=value, ...)`
configuration style required by Experiment Rules. Review findings before applying
fixes: `zip(strict=True)` can change behavior, and scientific routines need
numerical validation when refactored. Keep stage orchestration separate from
input validation, execution and measurement helpers within the owning module.
Use named arguments for long internal APIs. Pair aligned arrays with
`zip(..., strict=True)`, use `itertools.pairwise()` for adjacent samples, and spell
out `strict=False` when a partially filled display grid intentionally leaves
unused axes. Bind callback inputs when creating them inside a loop.
