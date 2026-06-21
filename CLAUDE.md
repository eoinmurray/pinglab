The working conventions for this repo — the "house rules" — now live in the House Rules article:

- on disk: `src/docs/src/pages/articles/ar016.mdx`
- on the site: `/articles/ar016/`

Read it before making any edits. It covers tooling (uv / bun), running experiments and reporting timing/cost, the notebook-is-the-recipe rule, figure and prose style (16:9 figures, `≈` not `~`, no backticks in docs), version control, and working style.

Non-negotiables kept here so they are always in context (full detail in the article):

- Do NOT dispatch Modal jobs (`--modal-gpu ...`) without explicit permission — Modal spends real money. Default to local runs.
- NEVER comment on or otherwise write to GitHub issues. Reading is fine.
- Do NOT create branches or open PRs unless explicitly asked — commit to the current branch (usually `main`). "Commit and push" means commit + push, not branch + PR.
