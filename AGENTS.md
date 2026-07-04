# AGENTS.md

pinglab — spiking E/I (PING) networks, trained with surrogate gradients and diagnosed via
Δt-stability. **Migrating in-place to the demolab framework** (agent-operated lab notebook:
`tools/` emit data → `experiments/` runners render figures → `writings/*.typ` publish via
Typst). During the migration the old `src/` (cli, notebooks, Astro docs) and the new
`tools/ experiments/ writings/` coexist — see [MIGRATE-CODE](demolab-engine/runbooks/MIGRATE-CODE.md).

**Conventions (the new structure)** → [`demolab-engine/guides/RULES.md`](demolab-engine/guides/RULES.md):
toolchain, the framework/content firewall, the tool ↔ experiment contract + schemas, and
how to add a tool / experiment / writing. The `demolab-engine/` dir is vendored **upstream
from `../demolab`** — never edit it here; "update demolab" re-pulls it wholesale.

**Legacy conventions (still governing the un-migrated `src/`)** → the House Rules article,
`writings/ar016.typ`.

## Non-negotiables (both worlds)

- **Toolchain:** `uv` (Python), `typst` (publishing), `task` (go-task). Never call `pip` /
  `python` / `python3` / `bun`-build directly for the new structure.
- **Commits:** author every commit as the human only — never a `Co-Authored-By:` / agent
  trailer, never an agent in the author/committer fields.
- **No Modal without explicit permission** — `--modal-gpu` spends real money; default local.
- **Never write to GitHub issues/PRs** (reading is fine). Don't open branches/PRs unless asked.

## Runbooks

Say the trigger; open the file in [`demolab-engine/runbooks/`](demolab-engine/runbooks/) and
drive it **interactively** (run each step, show the result, confirm before moving on).

| Trigger | Runbook |
|---------|---------|
| *"migrate my code"* | [MIGRATE-CODE.md](demolab-engine/runbooks/MIGRATE-CODE.md) |
| *"how do I get started"* | [GETTING-STARTED.md](demolab-engine/runbooks/GETTING-STARTED.md) |
| *"embed demolab as a docs site"* | [EMBED-DOCS.md](demolab-engine/runbooks/EMBED-DOCS.md) |
| *"migrate the stack to MATLAB / Julia"* | [MIGRATE-STACK.md](demolab-engine/runbooks/MIGRATE-STACK.md) |
| *"doctor the repo"* | [DOCTOR.md](demolab-engine/runbooks/DOCTOR.md) |
| *"update demolab"* | [UPDATE.md](demolab-engine/runbooks/UPDATE.md) |
| *"ground my claims"* | [GROUND-CLAIMS.md](demolab-engine/runbooks/GROUND-CLAIMS.md) |
