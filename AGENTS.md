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
- **For writings-only changes, do not run the software test suite.** Build the affected
  Demolab entries (and the complete publication only when the change can affect shared
  rendering, collection structure, or the book). Run code tests only when code, data
  contracts, runners, or executable examples change.
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

## Document delivery

- Do not open, embed, or display generated PDFs in the Codex app.
- When handing off a document, provide clickable links to both its generated PDF and its
  generated HTML webpage. Do not substitute a PDF attachment or preview for those links.

## Experiment drafts

Use `status: "draft"` for an experiment whose design is still being refined and which has not
been run. The canonical draft format is deliberately minimal:

1. A short `Abstract` stating the question and intended measurement.
2. An enumerated `Methods` section describing the planned experiment.
3. A separate enumerated `Results` section. Keep its items in the same order as the methods.
   Give each result separate list items for axes, traces or marks, why, and expectation or
   observation.

Keep a network or experimental-design diagram beside the method that defines it. Do not repeat
that diagram as a result. State near the start of Methods which parts have been run. Clearly
distinguish planned outputs from observations.

For every plotted result, define the horizontal and vertical axes, the traces or marks, why the
plot is needed, and what it is expected to show. Define a third axis when needed. For a non-plot
result, state that axes and traces do not apply, then define the displayed elements, why the
output is needed, and the expected output.

Use that structure only for planned results. Once a result exists, remove the planning fields.
Show only its title, figure, and concise interpretation.

Write drafts in direct, human-readable language. Introduce technical terms only when they
are needed, explain them briefly, and do not add speculative theory or extra sections unless
the experiment requires them or the user asks for them.

### Methods-to-results mirror figures

Suggest the **exp086 figure pattern** when a planned mechanism would be easier to understand
visually:

1. Put a clean, hand-authored SVG schematic in Methods. Use plot-like axes and illustrative
   curves to explain the proposed mechanism, and label it clearly as a design schematic rather
   than simulated or measured data.
2. Plan a Results figure with the same panel layout and visual language, generated
   reproducibly from the real experimental data.

Use direct SVG for the conceptual figure because precise explanatory layout matters. Use the
experiment runner and its plotting library for the measured result because provenance and
reproducibility matter. The user can trigger this pattern by saying **“use the exp086 figure
pattern”** or **“make a Methods-to-Results mirror figure.”**

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

The PR description is the scientific decision record, not release-note decoration. Keep it current with the question being tested, competing interpretations, provenance or data implications, decisions made, rejected alternatives, and unresolved limitations. Commits should remain small enough to explain one coherent change, but the PR is where the reasoning across those commits lives.

### Campaign and publication worktrees

Campaign execution, campaign activation, artifact promotion, and publication-view rebuilding must happen on the campaign's own branch in a dedicated worktree. Do not activate a campaign in a general development worktree: activation intentionally replaces tracked publication artifacts and makes unrelated changes dangerously easy to commit together.

- One campaign or publication view per worktree and branch.
- Open its draft PR before substantial execution or promotion work begins.
- Treat `runs/` and R2 as the raw execution record; Git tracks only the selected publication view.
- New raw arrays, checkpoints, caches, and duplicated inputs do not belong in `artifacts/data/`. Archive them through `runstore`; promote compact results, final figures, and provenance metadata.
- Review `git status` and the artifact diff before every campaign commit. A large generated diff is evidence to inspect, not permission to use `git add -A` blindly.

### Use judgment and intervene

Actively flag when the current workflow appears inappropriate rather than blindly following the immediate instruction.

In particular:

- If substantial work is starting directly on `main`, suggest creating a branch and draft PR before proceeding.
- If a growing change initially treated as "small" has accumulated complexity, suggest moving it to a branch/PR.
- If an issue and PR are duplicating the same evolving feature documentation, suggest consolidating the implementation discussion into the PR.
- Conversely, do not suggest a PR for trivial changes where it would provide little value.
- If uncertain, favor preserving useful context without introducing unnecessary process.

The goal is **good engineering history and useful documentation, not process for its own sake**.
