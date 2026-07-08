# Runbook: Getting started

Set the user's lab up end to end — scaffolded, branded, building (and optionally published) — until they understand the loop and have landed their first experiment: a fresh one, or the first piece of a migrated codebase.

**Triggers** — say any of these, or just `GETTING-STARTED`: **"how do I get started"**, "help me set up", "onboard me", "walk me through this repo".

## Decisions the agent surfaces

Onboarding is a conversation, not an interrogation. **Orient first, then ask only what you must, in order.** Bracketed `[n]` points at the step where each decision lands.

**Number the options.** Whenever you present a choice, lay the options out as a numbered list (1, 2, 3…) with the recommended default first and marked — so the user can answer with a single digit instead of typing a sentence.

**Order matters:** fresh-or-migrate → demo-or-clean → branding → stack → (verify) → clear-demo → publish → first experiment. Migration and stack are foundational (they decide *what* gets built and *in what language*), so they're settled up front; branding is quick and motivating, so it leads the config; and **nothing of theirs gets built until the scaffold verifies green.**

**Must-ask** (wait for an answer):
- **Ready to start?** — after the orient. `[0]`
- **Fresh, or migrate an existing codebase?** — fresh → a new experiment; migrate → `MIGRATE-CODE.md`, incremental. `[1]`
- **Install missing toolchain?** — `uv`/`typst`/`go-task`, default yes. Door B already did this. `[2]`
- **Demo or clean?** — `task add-demo-content` vs bare `task scaffold`, default demo. `[2]`
- **Stack?** — Python (default, `uv`) or MATLAB/Julia/R/Octave (`MIGRATE-STACK.md`). If migrating, confirm from their code — don't ask blind. `[4]`
- **Clear the demo?** — `task clear-demo-content`, default yes (only if they took the demo path). `[6]`
- **Publish to GitHub Pages?** — free unless the repo is private; default yes, if no stop (it works locally). `[7]`
- **What do you want to compute?** — open-ended, required, no default — *only after the scaffold verifies* (step 5). `[8]`

**Offer-with-a-default** (state it, move on):
- **Branding** (`demolab.yaml`, one pass): site name (default "demolab") · tagline · book/PDF title (from the name) · author + contact (offer git config). `[3]`
- **Custom domain, or `*.github.io`?** — default github.io; if custom, write `CNAME` and give the DNS records. `[7]`

**Surface, then default to inline** (do *not* silently build a tool): **tool-vs-inline** `[8]` — a first experiment computes **inline in the runner**. A `tools/` CLI is only worth it for science **reused across multiple experiments/writeups**. Say this to the user, and build a tool *only if they confirm reuse* — never manufacture one to satisfy the shape (RULES §4).

**Defer** (only if the user raises it): collections (§6.5), house style (`HOUSESTYLE.local.md`), license (ships MIT).

**Instruction, not a choice:** enabling Pages (*Settings → Pages → Source: GitHub Actions*) is a GitHub-UI click you can't do — tell the user. And commits never record agent authorship (a rule, not a prompt).

**Two things you drive, not ask:** the **dev server** (start it at step 2 and leave it up, so branding and the first experiment render live) and the **editor** (offer once at step 2 — `code .` / `cursor .` — unless they're already in an editor-agent).

**Two doors, one runbook.** Users arrive either by pasting a prompt into an agent, or via `curl … install.sh | sh`. Door B has already installed the toolchain and laid down a bare scaffold, so **detect tools-present + tree-already-scaffolded and resume at step 2**, acknowledging it ("your lab's already fetched and scaffolded — now let's shape it") rather than re-running install.

## The flow

**Read this before running anything — this is a strict, ordered procedure, not a narrative to adapt.** Execute steps **0 → 9 in order**. Do **not** reorder, merge, or skip steps, and do **not** pull a later step forward — the most common mistake is asking about **publishing early; that is step 7, never step 3–4.** Complete each step — *including waiting for the user's answer to its question* — before starting the next. Run **no** command (`clone`, `task install`, `scaffold`/`add-demo-content`, `run`, `dev`) before the step-0 orient and the user's "ready". If you catch yourself doing several things then reporting back, stop: you are freestyling, not following this runbook.

**Ground rules (self-contained — you need nothing else to run this):**
- **The repo must be the user's own copy.** If you arrived with just a URL and an empty folder, clone + *degit* it first, so there's **no `origin` remote** pointing at upstream and the demolab-project files are gone: `git clone --depth 1 https://github.com/eoinmurray/demolab . && rm -rf .git landing .github/workflows/landing.yml && git init && git add -A && git commit -m "Start my lab from demolab"`. If the tree is **already here and scaffolded** (an `install.sh` run brought you), don't re-clone — resume at step 2.
- **Toolchain:** drive everything through **`task`** (it wraps `uv` + `typst`). Never call `pip` / `python` / `python3` directly.
- **Commits:** author every commit as the **human only** — never an agent or `Co-Authored-By:` trailer.
- Deeper conventions are in [`../guides/RULES.md`](../guides/RULES.md); the other runbooks (migrate, lint, doctor, update…) are indexed in [`../../AGENTS.md`](../../AGENTS.md).

0. **Orient, then get the go-ahead.** Before touching anything, in a few sentences:
   - **The arc** — "I'll set your lab up end to end: scaffold → brand → pick your stack → get it building and, if you want, published → then your first experiment. Mostly me working while you answer a handful of questions."
   - **What to have handy** — a rough idea of a first thing to compute; a GitHub account if you'd like it online; or the repo/path if you're bringing in existing code.
   - **How long** — about 5–10 minutes to a built, live lab; the first experiment takes as long as it takes. Less if `install.sh` already ran, or if you skip publishing.
   - Then **"Ready?"** — and wait. For a Door-B arrival, say the fetch + install are already done and the estimate is shorter.

1. **Fresh or migrate?** "Starting fresh, or bringing in an existing codebase?" *Fresh* → continue. *Migrate* → it settles the stack (step 4) from their code and turns the finale (step 8) into [`MIGRATE-CODE.md`](MIGRATE-CODE.md); set the expectation now that migration is **incremental** — one experiment at a time, wrapping functions, not rewriting the science.

2. **Get the tree standing.** *(Door B already did the install + bare scaffold — skip to the dev server + editor below.)* The repo needs `uv`, `typst`, `go-task`; if any are missing, **offer to install them and, once approved, do it** (macOS: `brew install uv typst go-task`; `uv` via `curl -LsSf https://astral.sh/uv/install.sh | sh`; confirm each is on PATH). Then `task install`. Now ask **demo or clean**: most people learn best from the worked example, so unless they choose clean (bare `task scaffold`), run `task add-demo-content` (the `neuron`/`mujoco` tools, `exp00*` runners + writeups, a deck), then `task run -- exp000` and show in a sentence how the artifact became a page.
   - **Start the dev server, and get them into it before going further:** `task dev` in the background, then present the live URL **prominently** (<http://localhost:3000>, or the next free port — read it from the output, don't assume). Ask them to open it and click into the exp000 page, and **confirm they've actually looked before you continue** — the whole point is that they *see* a run become a page; don't proceed on a URL they haven't opened. It stays running, so branding and the first experiment render live.

3. **Brand it** (one pass). Gather, then write the optional root `demolab.yaml`: site name (default "Demolab"), tagline, book/PDF title (defaults from the name), and **author + contact** (offer to pull from git config — they render as a byline under the homepage title and an `<meta name="author">`; contact, if given, links the byline). The engine defaults any key you omit and updates never touch it; `task dev` hot-reloads, so they watch it change. Deeper theming (`style.css`, `favicon.svg`) lives inside the black box `demolab-engine/build/` — leave it as advanced. **Do not raise publishing here** — that's step 7; branding does not lead into "and shall we publish?"

4. **Stack.** Confirm the language their *tools* will be written in: **Python (`uv`) by default**, or MATLAB/Julia/R/Octave. If migrating, this is dictated by their code — confirm it, don't ask blind. Python → note it and move on. Otherwise follow [`MIGRATE-STACK.md`](MIGRATE-STACK.md) **now** to wire the language's runner glue + figure helpers, so the first experiment lands on the right stack instead of being ported later.

5. **Verify the scaffold, and orient them in it** (a gate, not a choice). Before any of *their* content: the structure is laid down (`writings/ experiments/ tools/ artifacts/` + `demolab.yaml`), `task build` is green (the empty-state homepage on a clean tree, or the demo site), and `task test` passes. **Have them open their editor** (`code .` / `cursor .` / their `$EDITOR`, unless they're already in one) and look at the tree while you walk it in a sentence — `writings/` (prose), `experiments/` (runners), `tools/` (reusable science), `artifacts/` (the record). **Confirm they've opened it and seen the layout** before moving on. Only once the build is green *and* they've oriented do you ask what to compute — don't build a first experiment on a half-set-up tree they haven't looked at.

6. **Clear the demo?** Only if they took the demo path. `task clear-demo-content` removes exactly the paths in `demolab-engine/scaffold/demo-manifest.json` (the demo tools, `exp00*`/`ar00*` writeups + deck, `playground.py`, their staged `artifacts/`). Their content and the whole framework zone aren't in that manifest, so they're untouched. The demo **source persists** in `demolab-engine/scaffold/demo/`, so step 8 still has models to copy. Then `task clean && task build` to confirm the site still stands.

7. **Publish to GitHub Pages?** *"Free, unless the repo is private."* Default yes; if no, stop — it all works locally. If yes: create + push a GitHub repo (`gh`), run `task deploy-setup` (drops `.github/workflows/deploy.yml` — one supported path, no options), **offer a custom domain** (default `*.github.io`; if custom, write `CNAME` and give the DNS records), then **tell the user to enable Pages** (*Settings → Pages → Source: GitHub Actions* — the one UI click you can't do), and push. Run `task build` first to confirm it compiles; confirm the Action succeeds. The site uses relative links, so it works under any Pages path with no base config.

8. **First experiment — or first migration** (the finale).
   - *Fresh:* now ask what they want to compute — keep it small. **Default to computing it inline in the runner — do not build a tool.** Say so, and why, in a sentence: *"I'll compute this directly in the experiment. A reusable `tools/` module is only worth it once you're running the same model across several experiments — if that happens, we'll promote it into one then."* Build a `tools/<name>/tool.py` **only if the user confirms the science is genuinely reusable** (a solver/model they'll call repeatedly); never manufacture one to satisfy the shape (RULES §4). Scaffold by **modeling on `demolab-engine/scaffold/demo/`** (persists after clear-demo), following [`../guides/RULES.md`](../guides/RULES.md):
     - **Runner `experiments/expNNN.py`** (model `scaffold/demo/experiments/exp000.py`): compute the result **inline**, then render the figure(s) into `artifacts/data/expNNN/` and stage a `numbers.json` of the headline metrics + config.
     - **Writeup `writings/expNNN.typ`** (model `scaffold/demo/writings/exp000.typ`): a `#let meta` + `#let body` pair; read the run with `#let run = json("/artifacts/data/expNNN/numbers.json")`, embed figures with `#image(...)`, tables via `#numbers-table(...)` — **never hand-type numbers**. Video via `#video(...)` (HTML only).
     - **Only if reuse was confirmed**, add the tool first: `tools/<name>/tool.py` (model `scaffold/demo/tools/neuron/tool.py`) — `setup_run_dir`/`write_output`, the data (CSV/JSON/`.npz`/… — its choice) + a `manifest` of metrics, **data not plots** (a physics video is the one exception, `headline_video`), plus `test_<tool>.py` (`task test` green). The runner then calls its CLI instead of computing inline.
     - `task run -- expNNN` — it rebuilds live in the dev server. Present the new page's URL **prominently**, have them open it + its PDF, and **confirm they've read their published result** — this is the payoff of the whole flow. Then commit + push (auto-redeploys if published).
   - *Migrate:* run [`MIGRATE-CODE.md`](MIGRATE-CODE.md) to land the **first** migrated experiment — wrap one function as a tool, one runner, one writeup — same live-site payoff. The rest comes in incrementally, later.

9. **Sign off** — short and warm. Their editor's open (step 5) and the dev server's running; tell them in your own words:
   - **It works** — their first entry is built (and live at `<url>` if they published); the dev server's up for live preview.
   - **Guides** (`demolab-engine/guides/`) are the reference: RULES · GLOSSARY · HOUSESTYLE · STRUCTURE.
   - **Runbooks** (`demolab-engine/runbooks/`) do the common jobs — trigger any by just its name (**LINT**, **DOCTOR**, **RED-TEAM**, **NEXT**, **TOUR**, **FROM-PAPER**, …), and I'll run it. Say **HELP** anytime to see the full menu.
   - **Ask me anything about the repo** — how it works, where something lives, why a convention is the way it is.

Notes: provenance is automatic — each run stamps its git commit into `numbers.json` and the page/PDF footer. Commit the tool code *before* a run you intend to publish, so the footer reads clean (an uncommitted run stamps *dirty*).
