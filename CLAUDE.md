The working conventions for this repo live in the demolab guides (shipped in the
`demolab-cli` package — `demolab docs <NAME>` prints a guide's path; read it):

- `demolab docs RULES` — the framework invariants (repo structure, the tool↔experiment boundary, provenance, how to add a tool / experiment / writing).
- `demolab docs HOUSESTYLE` — authoring style, rules H1–H13 (prose, math with every term defined, 16:9 figures, `≈` not `~`, native Typst math, structure).

Read them before making any edits.

## Commands — type a NAME

demolab is driven by names in CAPS. **If the user's message is just one of these names, that IS the command — do it, don't ask what they mean:**

- **`HELP`** — run `demolab docs` and present the menu of runbooks and guides.
- **A runbook name** (e.g. `TOUR`, `LINT`, `DOCTOR`, `RED-TEAM`, `STEELMAN`, `NEXT`, `AUTORESEARCH`, `PLAN`, `NIGHT-SHIFT`, `UPDATE`) — run `demolab docs <NAME>`, read the file, then **start that runbook** and drive it step by step.
- **A guide name** (e.g. `RULES`, `HOUSESTYLE`, `SLIDES`, `STRUCTURE`, `GLOSSARY`, `SUPPORT`, `AUTORESEARCH-RULES`) — read it the same way, then **walk the user through that guide** interactively.

The full, current menu comes from `demolab docs` — run it rather than relying on a memorised list. Interaction rules are in [AGENTS.md](AGENTS.md).

Non-negotiables kept here so they are always in context (full detail in the article):

- Do NOT dispatch RunPod fan-outs (`exp022.py --runpod --live`, or any pod-creating call) without explicit permission — RunPod spends real money. Default to local runs.
- Do NOT write to GitHub issues without explicit permission — with explicit permission, creating/posting an issue is fine. Never comment on existing issues without explicit permission. Reading is always fine.
- Do NOT create branches or open PRs unless explicitly asked — commit to the current branch (usually `main`). "Commit and push" means commit + push, not branch + PR.
- You have free reign when making notebooks, but editing the cli needs explicit permission.
- NEVER add Co-Authored-By, "Generated with Claude Code", or any other AI-attribution trailers to commit messages or PR descriptions. Commits are authored by Eoin alone.
- Session hygiene: at topic pivots or when a task wraps up, proactively suggest `/clear` (unrelated new topic; offer a handoff summary) or `/compact` (same thread, spent history). Full rule in [AGENTS.md](AGENTS.md) → "Session hygiene".
