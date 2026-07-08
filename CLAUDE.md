The working conventions for this repo live in the demolab guides:

- `demolab-engine/guides/RULES.md` — the framework invariants (repo structure, the tool↔experiment boundary, provenance, how to add a tool / experiment / writing).
- `demolab-engine/guides/HOUSESTYLE.md` — authoring style, rules H1–H13 (prose, math with every term defined, 16:9 figures, `≈` not `~`, native Typst math, structure).

Read them before making any edits.

## Commands — type a NAME

demolab is driven by names in CAPS. **If the user's message is just one of these names, that IS the command — do it, don't ask what they mean:**

- **`HELP`** — list the runbooks and guides (the menu).
- **A runbook name** (`GETTING-STARTED`, `TOUR`, `LINT`, `DOCTOR`, `RED-TEAM`, `STEELMAN`, `NEXT`, `UPDATE`, `MIGRATE-CODE`, `MIGRATE-STACK`, `FROM-JUPYTER`, `FROM-PAPER`, `EMBED-DOCS`, `GROUND-CLAIMS`) — **start that runbook** and drive it step by step.
- **A guide name** (`RULES`, `HOUSESTYLE`, `SLIDES`, `STRUCTURE`, `GLOSSARY`, `SUPPORT`) — **walk the user through that guide** interactively.

The full table (with what each does and its lower-case aliases) is in [AGENTS.md](AGENTS.md).

Non-negotiables kept here so they are always in context (full detail in the article):

- Do NOT dispatch RunPod fan-outs (`exp022.py --runpod --live`, or any pod-creating call) without explicit permission — RunPod spends real money. Default to local runs.
- NEVER comment on or otherwise write to GitHub issues. Reading is fine.
- Do NOT create branches or open PRs unless explicitly asked — commit to the current branch (usually `main`). "Commit and push" means commit + push, not branch + PR.
- You have free reign when making notebooks, but editing the cli needs explicit permission.
- NEVER add Co-Authored-By, "Generated with Claude Code", or any other AI-attribution trailers to commit messages or PR descriptions. Commits are authored by Eoin alone.
