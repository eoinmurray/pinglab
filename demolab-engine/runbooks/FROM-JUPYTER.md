# Runbook: Convert a Jupyter notebook into an experiment

Triggers: **"from jupyter"**, "convert my notebook", "import this notebook", "make my notebook reproducible", **FROM-JUPYTER**. Goal: turn a `.ipynb` into a proper demolab experiment — a runner + writeup — with the randomness seeded, the numbers wired to `numbers.json` so they can't drift, and the page reproducing (or correcting) the notebook's output.

Notebooks are where most computational science lives, and they're the opposite of reproducible: out-of-order execution, hidden global state, unseeded randomness, hardcoded numbers in the prose, no pinned environment. This runbook **launders** one into something that runs top-to-bottom, deterministically, with every published number traceable to the run. Drive it interactively — don't silently rewrite; **propose the mapping and confirm before building**.

## 0. Prerequisites
The repo is scaffolded (a working lab, or a bare `task scaffold`) and `task build` is green. Ask **which `.ipynb`** to convert and pick the experiment id (`expNNN`). The notebook can live anywhere — it isn't part of the contract and won't be published.

## 1. Run it first — capture ground truth
Execute the notebook top-to-bottom (`uv run jupyter nbconvert --to notebook --execute`) and keep its outputs (figures, values) to check parity against later. This also surfaces finding #1: **if it won't run clean top-to-bottom, its results depend on hidden execution order** — exactly the sin you're here to fix. Note it; don't paper over it.

## 2. Map cells by kind — propose, then confirm
Read the notebook and lay out a plan for the user before touching anything:
- **Computation** → the runner `experiments/expNNN.py`. **Default inline** — a converted notebook is a one-off until it's reused across experiments (RULES §4). Lift into a `tools/<name>/tool.py` **only** if the user says they'll reuse the model; never manufacture a tool to satisfy the shape.
- **Plots** → the runner renders the figure(s) from the data into `artifacts/data/expNNN/`, using the shared figure style (`experiments/helpers/style.py`).
- **Prose / markdown** → the writeup `writings/expNNN.typ` (a `#let meta` + `#let body` pair).
- **Parameters** (the constants at the top) → the run's config.
- **Results** (the values the prose reports) → `numbers.json`.
- **Scratch** (dead exploratory cells, debugging) → dropped — **name what you're dropping** so nothing important goes silently.

If the notebook is really several results, **propose a split** into multiple experiments (one id each) and confirm before proceeding.

## 3. Fix the reproducibility sins as you port
- **Seed** every source of randomness through an explicit `--seed`, captured in the config, so the run reproduces (DOCTOR checks this).
- Make it run **top-to-bottom with no hidden state** — no reliance on cell order, no mutable globals carried between cells.
- **Pin** any new dependency with `uv add` (into `pyproject.toml`) — never a bare `pip`/unpinned import.

## 4. Wire the numbers so they can't drift — the point of the whole thing
Find every hardcoded number in the prose ("a firing rate of 90 Hz", "improved by 12%") and replace it with a `numbers.json` reference (`#run.<command>.<metric>`, or a `#numbers-table(...)`) — never a typed literal (RULES §6.2, HOUSESTYLE H9). **If a number in the prose has no source in the run, that's a flag:** either compute it into the run, or cut the claim.

## 5. Build and check parity
`task run -- expNNN`, then `task build` (or the running `task dev`). Open the new page beside the notebook's original output and confirm the figures + numbers match. If they differ, **explain why** — usually the notebook's numbers were stale or from out-of-order runs, and the reproducible version is the correct one. That's a feature, not a regression.

## 6. Hand off
Show the page + its PDF. State plainly **what was dropped** (scratch cells) and **what was fixed** (seeds added, drifting numbers wired to the run, deps pinned). Then offer to run **RED-TEAM** on the fresh experiment — a just-laundered notebook is exactly the thing to adversarially check before anyone trusts it.

Note: the original `.ipynb` is yours to keep or delete — it's not part of the demolab record once the experiment reproduces.
