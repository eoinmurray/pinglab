# CLAUDE.md

Read [AGENTS.md](AGENTS.md), then run `demolab docs` and follow what it prints — it is the
agent manual for working in this lab. A user message that is just a NAME in CAPS (`HELP`,
`LINT`, `DOCTOR`, …) is a command; the manual explains how each routes.

Non-negotiables kept here so they are always in context (detail in [AGENTS.md](AGENTS.md)):

- Do NOT dispatch RunPod fan-outs (`exp022.py --runpod --live`, or any pod-creating call) without explicit permission — RunPod spends real money. Default to local runs.
- Do NOT write to GitHub issues without explicit permission — with explicit permission, creating/posting an issue is fine. Never comment on existing issues without explicit permission. Reading is always fine.
- Do NOT create branches or open PRs unless explicitly asked — commit to the current branch (usually `main`). "Commit and push" means commit + push, not branch + PR.
- You have free reign when making notebooks, but editing the cli needs explicit permission.
- NEVER add Co-Authored-By, "Generated with Claude Code", or any other AI-attribution trailers to commit messages or PR descriptions. Commits are authored by Eoin alone.
- Session hygiene: at topic pivots or when a task wraps up, proactively suggest `/clear` (unrelated new topic; offer a handoff summary) or `/compact` (same thread, spent history). Full rule in [AGENTS.md](AGENTS.md) → "Session hygiene".
