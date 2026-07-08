# Runbook: Migrating existing code

Bring an existing codebase into demolab on three principles: **one experiment at a time**, **wrap don't rewrite** (the new `tool.py` is a thin adapter that imports and calls their functions — never reimplement their science), and **one environment** (their deps fold into the root `pyproject.toml`). Get one experiment publishing end to end before touching the next.

**Triggers** — say any of these, or just `MIGRATE-CODE`: **"migrate my code"**, "import my repo", "bring my existing code in".

1. **Inventory.** Read their repo (local path or clone). List candidate experiments — each script/function that ends in a figure or a few numbers. With the user, pick the single simplest first.
2. **Bring their code in.** Installable package → `uv add <name>` (PyPI, or git/local-path dep) + `uv sync`, then `import` it. Loose scripts → copy only the modules the experiment needs next to `tool.py` or into a shared package under `tools/`.
3. **Merge deps.** `uv add` only what the chosen experiment needs (never `pip install`); surface version conflicts and resolve with the user.
4. **Wrap as a tool command.** Create `tools/<tool>/tool.py` modeled on `neuron/tool.py`: copy `setup_run_dir`/`write_output` verbatim (adjust `TEMP_DIR` + logger), add an `argparse` subcommand for the experiment's params, and in the handler call `setup_run_dir` → **their function** → write the data (a CSV, JSON, `.npz` — whatever suits) → `write_output` with a manifest (headline metrics; the runner draws the figure, so the tool emits no plot — a rendering/video is the exception). Keep it thin; if you're porting their math, stop and import instead. The tool must stay generic — it must **not** import runner code (`experiments/`), and a runner reaches it by *running* it, never importing (a `tools` ↔ `experiments` import boundary kept by convention). Add a test (`task test`). Verify `temp/<tool>/<cmd>/` has the full set.
5. **Runner + writing.** As in the [Getting started](GETTING-STARTED.md) runbook, step 4. Confirm the published figure and numbers match their original code's output.
6. **Repeat** for the next experiment; stop when the ones that matter are done.

Notes: thread a `--seed` through anything random (see `neuron`'s `net` command); bring and run their relevant tests with `uv run`.
