The working conventions for this repo — the "house rules" — now live in the House Rules article:

- on disk: `writings/ar016.typ`
- on the site: `/ar016/`

Read it before making any edits. It covers tooling (uv / typst / task), running experiments and reporting timing/cost, the notebook-is-the-recipe rule, figure and prose style (16:9 figures, `≈` not `~`, no backticks in docs), version control, and working style. The demolab authoring style rules (H1–H13) live in `demolab-engine/guides/HOUSE-STYLE.md`.

Non-negotiables kept here so they are always in context (full detail in the article):

- Do NOT dispatch Modal jobs (`--modal-gpu ...`) without explicit permission — Modal spends real money. Default to local runs.
- NEVER comment on or otherwise write to GitHub issues. Reading is fine.
- Do NOT create branches or open PRs unless explicitly asked — commit to the current branch (usually `main`). "Commit and push" means commit + push, not branch + PR.
- You have free reign when making notebooks, but editing the cli needs explicit permission.
- NEVER add Co-Authored-By, "Generated with Claude Code", or any other AI-attribution trailers to commit messages or PR descriptions. Commits are authored by Eoin alone.
