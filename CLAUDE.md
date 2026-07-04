The working conventions for this repo live in the demolab guides:

- `demolab-engine/guides/RULES.md` — the framework invariants (repo structure, the tool↔experiment boundary, provenance, how to add a tool / experiment / writing).
- `demolab-engine/guides/HOUSE-STYLE.md` — authoring style, rules H1–H13 (prose, math with every term defined, 16:9 figures, `≈` not `~`, native Typst math, structure).

Read them before making any edits.

Non-negotiables kept here so they are always in context (full detail in the article):

- Do NOT dispatch Modal jobs (`--modal-gpu ...`) without explicit permission — Modal spends real money. Default to local runs.
- NEVER comment on or otherwise write to GitHub issues. Reading is fine.
- Do NOT create branches or open PRs unless explicitly asked — commit to the current branch (usually `main`). "Commit and push" means commit + push, not branch + PR.
- You have free reign when making notebooks, but editing the cli needs explicit permission.
- NEVER add Co-Authored-By, "Generated with Claude Code", or any other AI-attribution trailers to commit messages or PR descriptions. Commits are authored by Eoin alone.
