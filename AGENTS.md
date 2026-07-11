# AGENTS.md

pinglab — spiking E/I (PING) networks, trained with surrogate gradients and diagnosed via
Δt-stability. Structured as the **demolab framework** (agent-operated lab notebook): the
`tools/snn` engine emits data → `experiments/` runners render figures → `writings/*.typ`
publish via Typst to web + PDFs + a book.

**Conventions** → `demolab docs RULES` (prints the file's path — read it): toolchain, what
the package owns vs what's ours, the tool ↔ experiment contract + schemas, and how to add
a tool / experiment / writing; authoring style → `demolab docs HOUSESTYLE`; slide-deck
conventions → `demolab docs SLIDES`. The engine ships in the **`demolab-cli` package**
(installed by `uv sync`; the gitignored `.demolab/` staging dir at the root is machine-managed
— never edit it); "update demolab" is a dependency bump (`demolab docs UPDATE`).

## Non-negotiables (both worlds)

- **Toolchain:** `uv` (Python), `typst` (publishing) via the `demolab` CLI, `task` (go-task)
  for this repo's own lanes (lint, tests, sim/train). Never call `pip` / `python` /
  `python3` / `bun`-build directly for the new structure.
- **Commits:** author every commit as the human only — never a `Co-Authored-By:` / agent
  trailer, never an agent in the author/committer fields.
- **No RunPod fan-outs without explicit permission** — `--runpod --live` (and any pod-creating call) spends real money; default local.
- **Don't write to GitHub issues/PRs without explicit permission** — with explicit permission, creating/posting an issue is fine; reading is always fine. Don't open branches/PRs unless asked.

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

## Commands — type a NAME

demolab is driven by typing a **name in CAPS** (SCREAMING-KEBAB). Three commands:

- **`HELP`** — run `demolab docs` and present its menu, one line each.
- **`<RUNBOOK>`** — a runbook name → run `demolab docs <NAME>`, read the file, then **start
  it** and drive it step by step: run a step, show the result, confirm before the next.
  Never dump the whole runbook at once. E.g. `LINT`, `DOCTOR`, `NEXT`.
- **`<GUIDE>`** — a guide name → read it the same way, then **walk the user through it**:
  summarise it, go section by section, answer questions — don't just paste the file.
  E.g. `RULES`, `SLIDES`.

**The NAME is the command.** If the user's message is (or starts with) one of these names —
`LINT`, `RULES`, `HELP` — that *is* the request: act on it, don't ask what they mean.
Natural phrasings route too ("lint the writings" → `LINT`); the `demolab docs` menu's
one-liners tell you which doc covers what. **Docs are the source of truth** — when the user
asks "how do I…" about operating demolab, check the menu before improvising an answer.
