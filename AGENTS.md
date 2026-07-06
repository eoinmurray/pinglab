# AGENTS.md

pinglab — spiking E/I (PING) networks, trained with surrogate gradients and diagnosed via
Δt-stability. Structured as the **demolab framework** (agent-operated lab notebook): the
`tools/snn` engine emits data → `experiments/` runners render figures → `writings/*.typ`
publish via Typst to web + PDFs + a book.

**Conventions** → [`demolab-engine/guides/RULES.md`](demolab-engine/guides/RULES.md):
toolchain, the framework/content firewall, the tool ↔ experiment contract + schemas, and
how to add a tool / experiment / writing; authoring style is in
[`HOUSESTYLE.md`](demolab-engine/guides/HOUSESTYLE.md); slide-deck conventions are in
[`SLIDES.md`](demolab-engine/guides/SLIDES.md). The `demolab-engine/` dir is vendored
**upstream from `../demolab`** — never edit it here; "update demolab" re-pulls it wholesale.

## Non-negotiables (both worlds)

- **Toolchain:** `uv` (Python), `typst` (publishing), `task` (go-task). Never call `pip` /
  `python` / `python3` / `bun`-build directly for the new structure.
- **Commits:** author every commit as the human only — never a `Co-Authored-By:` / agent
  trailer, never an agent in the author/committer fields.
- **No Modal without explicit permission** — `--modal-gpu` spends real money; default local.
- **Never write to GitHub issues/PRs** (reading is fine). Don't open branches/PRs unless asked.

## Commands — type a NAME

demolab is driven by typing a **name in CAPS** (SCREAMING-KEBAB). Three commands:

- **`HELP`** — list the runbooks and guides below, one line each. The menu.
- **`<RUNBOOK>`** — a runbook name → **start it** and drive it step by step: run a step, show the result, confirm before the next. Never dump the whole runbook at once. E.g. `LINT`, `DOCTOR`, `GETTING-STARTED`.
- **`<GUIDE>`** — a guide name → **walk the user through it**: summarise it, go section by section, answer questions — don't just paste the file. E.g. `RULES`, `SLIDES`.

**The NAME is the command.** If the user's message is (or starts with) one of these names — `LINT`, `RULES`, `HELP` — that *is* the request: act on it, don't ask what they mean. The lower-case phrasings in the "also triggers on" column still work as aliases, but the CAPS name is canonical and always routes.

> **`GETTING-STARTED`** ("set up my lab", "how do I get started") means **following the runbook as a conversation** — orient the user, then ask the gated questions *in order* and wait for answers. **Do not autonomously clone, scaffold, install, run the demo, and report back** — that races past every choice the user is supposed to make (fresh-or-migrate, demo-or-clean, stack, branding, publish, what to compute). Read the runbook first; run nothing before its step-0 orient + ready-check.

### Runbooks — `NAME` starts it

| Name | Does | Also triggers on |
| ---- | ---- | ---------------- |
| [`GETTING-STARTED`](demolab-engine/runbooks/GETTING-STARTED.md) | set up a fresh lab, interactively | "set up my lab", "how do I get started" |
| [`TOUR`](demolab-engine/runbooks/TOUR.md) | walk through this repo | "tour", "walk me through this repo" |
| [`MIGRATE-CODE`](demolab-engine/runbooks/MIGRATE-CODE.md) | wrap an existing codebase | "migrate my code" |
| [`FROM-JUPYTER`](demolab-engine/runbooks/FROM-JUPYTER.md) | convert a notebook | "from jupyter", "convert my notebook" |
| [`FROM-PAPER`](demolab-engine/runbooks/FROM-PAPER.md) | reproduce a paper | "from paper", "reproduce this paper" |
| [`EMBED-DOCS`](demolab-engine/runbooks/EMBED-DOCS.md) | use demolab as a docs site | "embed demolab as a docs site" |
| [`MIGRATE-STACK`](demolab-engine/runbooks/MIGRATE-STACK.md) | switch language (MATLAB / R / Julia / …) | "migrate the stack to MATLAB" |
| [`GROUND-CLAIMS`](demolab-engine/runbooks/GROUND-CLAIMS.md) | back every claim with a run or citation | "ground my claims" |
| [`NEXT`](demolab-engine/runbooks/NEXT.md) | suggest what to run next | "what next", "what should I run next" |
| [`UPDATE`](demolab-engine/runbooks/UPDATE.md) | vendor the latest engine | "update demolab" |
| [`DOCTOR`](demolab-engine/runbooks/DOCTOR.md) | audit the structure against RULES | "doctor the repo" |
| [`LINT`](demolab-engine/runbooks/LINT.md) | audit the prose + figures vs the house style | "lint the writings" |
| [`RED-TEAM`](demolab-engine/runbooks/RED-TEAM.md) | attack the result's validity | "red-team", "critique this experiment" |
| [`STEELMAN`](demolab-engine/runbooks/STEELMAN.md) | make the strongest case for it | "steelman", "make the case for this" |

### Guides — `NAME` walks you through it

| Name | Covers |
| ---- | ------ |
| [`RULES`](demolab-engine/guides/RULES.md) | the contract, toolchain, firewall, how-tos |
| [`HOUSESTYLE`](demolab-engine/guides/HOUSESTYLE.md) | prose / math / figure style (the H-rules) |
| [`SLIDES`](demolab-engine/guides/SLIDES.md) | deck conventions + the layout catalog |
| [`STRUCTURE`](demolab-engine/guides/STRUCTURE.md) | the annotated file tree |
| [`GLOSSARY`](demolab-engine/guides/GLOSSARY.md) | the vocabulary |
| [`SUPPORT`](demolab-engine/guides/SUPPORT.md) | getting a human (issues / email) |
