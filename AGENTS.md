> **demolab lab** — before working here, run `demolab docs` and follow what it prints
> (the agent manual + runbook menu; no venv yet? `uvx demolab-cli docs`). A user message
> that is just a NAME in CAPS (`HELP`, `LINT`, `DOCTOR`, …) is a command — the manual explains.

# pinglab

Spiking E/I (PING) networks, trained with surrogate gradients and diagnosed via
Δt-stability. The `tools/snn` engine emits data → `experiments/` runners render figures →
`writings/*.typ` publish via Typst.

## This lab's rules

- **No RunPod fan-outs without explicit permission** — `--runpod --live` (and any
  pod-creating call) spends real money; default local. Same for anything Modal-dispatching.
- **Don't write to GitHub issues/PRs without explicit permission** — with explicit
  permission, creating/posting an issue is fine; reading is always fine. Don't open
  branches/PRs unless asked — "commit and push" means commit + push to the current branch.
- **Free reign on notebooks; editing the cli (`tools/snn`) needs explicit permission.**
- NEVER add Co-Authored-By, "Generated with Claude Code", or any other AI-attribution trailers to commit messages or PR descriptions. Commits are authored by Eoin alone.

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

## Development workflow

Use judgment about whether work should go directly to `main` or through a PR. Do not apply the rule mechanically.

### Small changes

Small, obvious, low-risk changes may be committed directly to `main`. Examples include trivial fixes, small configuration changes, typo/documentation fixes, and other changes where a PR would add ceremony without useful context.

### Substantial work

For features, refactors, investigations, or any change involving meaningful design decisions, multiple steps, uncertainty, or useful implementation context:

1. Create a branch.
2. If there is not yet code worth committing, create an empty commit:
   ```bash
   git commit --allow-empty -m "Start <feature>"
   ```
3. Push the branch and immediately open a **draft PR**.
4. Use the PR description as the primary working document for the change: motivation, design decisions, TODOs, discoveries, trade-offs, and implementation status should live there.
5. Update the PR description as understanding of the problem evolves.
6. Mark the PR ready once the work is coherent and ready for final review/merge.

Prefer starting a draft PR over creating a separate issue when the work is already going to be implemented and the issue would merely duplicate or fragment the same documentation. Use issues when something genuinely needs to exist independently of an implementation PR.

### Use judgment and intervene

Actively flag when the current workflow appears inappropriate rather than blindly following the immediate instruction.

In particular:

- If substantial work is starting directly on `main`, suggest creating a branch and draft PR before proceeding.
- If a growing change initially treated as "small" has accumulated complexity, suggest moving it to a branch/PR.
- If an issue and PR are duplicating the same evolving feature documentation, suggest consolidating the implementation discussion into the PR.
- Conversely, do not suggest a PR for trivial changes where it would provide little value.
- If uncertain, favor preserving useful context without introducing unnecessary process.

The goal is **good engineering history and useful documentation, not process for its own sake**.
