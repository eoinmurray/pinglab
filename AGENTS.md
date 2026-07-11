# AGENTS.md

This is a **demolab** lab (an agent-operated lab notebook). **Before doing anything else,
run `demolab docs` and follow what it prints** — the full agent manual plus the menu of
runbooks and guides, always in step with the installed engine. (No venv yet? `uv sync`
provides the `demolab` command, or run `uvx demolab-cli docs`.)

If the user's message is a NAME in CAPS (`HELP`, `LINT`, `DOCTOR`, …), that **is** the
command — the manual explains. Two rules worth stating even before you've read it: commits
are authored as the human only (never an agent trailer or co-author), and results are never
hand-typed (writings read their run's data).

## This lab's own rules

pinglab — spiking E/I (PING) networks, trained with surrogate gradients and diagnosed via
Δt-stability. The `tools/snn` engine emits data → `experiments/` runners render figures →
`writings/*.typ` publish via Typst.

- **Toolchain here also includes `task` (go-task)** for this repo's own lanes — lint,
  typecheck, tests, sim/train (`task` lists them). Publishing tasks delegate to `demolab`.
- **No RunPod fan-outs without explicit permission** — `--runpod --live` (and any
  pod-creating call) spends real money; default local. Same for anything Modal-dispatching.
- **Don't write to GitHub issues/PRs without explicit permission** — with explicit
  permission, creating/posting an issue is fine; reading is always fine. Don't open
  branches/PRs unless asked — "commit and push" means commit + push to the current branch.
- **Free reign on notebooks; editing the cli (`tools/snn`) needs explicit permission.**

## Session hygiene

Eoin tends to let sessions grow far too large. When the conversation **changes topic** (a new
experiment, a pivot from writing to debugging, a finished runbook, an unrelated question), the
agent should proactively say so and suggest the right cleanup — don't wait to be asked:

- **`/clear`** when the new topic shares nothing with the old one — carry-over context is pure
  cost. Offer a one-paragraph handoff summary to paste into the fresh session if useful.
- **`/compact`** when the thread continues but the history is mostly spent (long tool output,
  finished sub-tasks, resolved debugging).

Suggest at natural boundaries (task done, topic pivot) — not mid-task, and at most once per
boundary; if declined, drop it until the next one.
