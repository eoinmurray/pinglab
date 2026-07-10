# Runbook: Getting started

> Take a new user from an empty folder to a lab that is *theirs*: scaffolded, building, with
> their first experiment — their own science, not a demo — live on a page. Then brand it and,
> if they want, publish it.

## When to use
When a user is new to demolab and wants their lab set up. **Onboarding is a conversation, not
an interrogation** — orient first, ask only what you must, and **wait for each answer before
moving on**; do *not* autonomously scaffold, run, or publish past a choice the user hasn't
made. And it is a *short* conversation: three questions (ready? · what to compute? · publish?),
with everything else offered-with-a-default. The goal is their own result on a page they can
touch, inside ten minutes.

**No demo content.** The lab starts clean and the first thing in it is theirs. The finished,
polished example they may want to see is the landing site — <https://demolab.eoinmurray.info>
is the shipped demo, built straight from `demolab-engine/scaffold/demo/` — so point there
rather than overlaying demo content into their repo. (`task add-demo-content` exists as an
escape hatch if they explicitly ask to explore a worked example locally; `task
clear-demo-content` removes it again. Don't offer it unprompted.)

Have handy before starting: a rough idea of a first thing to compute (or the repo/path/notebook
/paper you're bringing in — starters are on offer if you have neither), and a GitHub account if
you'd like it online.

## What it does

**Read this before running anything — this is a strict, ordered procedure, not a narrative to
adapt.** Execute steps **0 → 6 in order**. Do **not** reorder, merge, or skip steps, and do
**not** pull a later step forward — the most common mistake is raising **publishing or branding
early; they are steps 4–5, after the first experiment, never before it.** Complete each step —
*including waiting for the user's answer to its question* — before starting the next. Run
**no** command (`clone`, `task install`, `task scaffold`, `run`, `dev`) before the step-0
orient and the user's "ready". If you catch yourself doing several things then reporting back,
stop: you are freestyling, not following this runbook.

**Ground rules (self-contained — you need nothing else to run this):**
- **Signpost every step.** As you start each step, tell the user where they are and what's next
  in a short clause, so a ten-minute flow never feels open-ended — e.g. *"Lab's up (1/6) — next
  we build your first experiment."* The six steps are: **1** stand the lab up · **2** first
  experiment · **3** make it permanent · **4** brand · **5** publish · **6** sign off. Don't
  turn it into ceremony; one clause at each transition is enough.
- **The repo must be the user's own copy.** If you arrived with just a URL and an empty folder,
  clone + *degit* it first, so there's **no `origin` remote** pointing at upstream and the
  upstream Pages workflow is gone: `git clone --depth 1 https://github.com/eoinmurray/demolab .
  && rm -rf .git .github/workflows/landing.yml && git init && git add -A && git commit -m
  "Start my lab from demolab"`. If the tree is **already here and scaffolded**, don't re-clone —
  resume at step 2.
- **Toolchain:** drive everything through **`task`** (it wraps `uv` + `typst`). Never call `pip`
  / `python` / `python3` directly.
- **Paths are repo-root-relative.** Every path in this runbook (and the guides) starts at the
  repo root — `cd` into the clone before following one; if you started a directory above it,
  prefix the clone's folder name.
- **Commits:** author every commit as the **human only** — never an agent or `Co-Authored-By:`
  trailer.
- Deeper conventions are in [`../guides/RULES.md`](../guides/RULES.md); the other runbooks
  (migrate, lint, doctor, update…) are indexed in [`../../AGENTS.md`](../../AGENTS.md).

0. **Orient, then get the go-ahead.** Before touching anything, in a few sentences:
   - **The arc** — "I'll stand your lab up, then we build your first experiment together and you
     watch it become a page — figures, numbers, PDF, all computed from the run. Then a quick
     brand, and publishing if you want it. Mostly me working — you make three real calls (ready,
     what to compute, whether to publish); everything else I default and tell you what I picked."
   - **What to have handy** — a first thing to compute (or code/a notebook/a paper to bring in —
     and if you have neither, I'll suggest starters); a GitHub account if you'd like it online.
   - **How long** — about ten minutes to your first result on a live page; less if the toolchain
     is already installed or you skip publishing.
   - Then **"Ready?"** — and wait.

1. **Stand the lab up** (you drive; no questions except the install approval). *(Toolchain
   already present and tree already scaffolded? Acknowledge it and skip to the dev server.)*
   - **Toolchain:** the repo needs `uv`, `typst`, `go-task`; if any are missing, **offer to
     install them and, once approved, do it** — macOS: `brew install uv typst go-task`; Linux:
     `uv` via `curl -LsSf https://astral.sh/uv/install.sh | sh`, typst + task from the distro or
     their release pages; Windows: `winget install astral-sh.uv Typst.Typst Task.Task`.
     **No package manager** (locked-down or sandboxed machine)? Download the `typst` and `task`
     release binaries into a repo-local `.tools/bin/` and put it on PATH — the build prefers
     `.tools/bin/` over PATH automatically. Confirm each tool resolves.
   - **Scaffold:** `task install`, then `task scaffold` — the bare structure (`writings/
     experiments/ tools/ artifacts/` + `demolab.yaml`), nothing else. Verify quietly: `task
     build` green (the friendly empty-state homepage) and `task test` passes — report the result
     in one line, don't make a ceremony of it. If either fails, fix it before going on: nothing
     of theirs gets built on a broken scaffold.
   - **Dev server:** `task dev` in the background, then present the live URL **prominently**
     (<http://localhost:3000>, or the next free port — read it from the output, don't assume).
     Ask them to open it — they should see the empty-state homepage ("your lab is ready"). It
     stays up for the whole session, so their experiment and branding render live.
   - **Editor:** offer once — `code .` / `cursor .` / their `$EDITOR` — unless they're already
     in one. Walk the tree in a sentence: `writings/` (prose), `experiments/` (runners),
     `tools/` (reusable science), `artifacts/` (the record). Don't gate on it.
   - **Sandboxed / locked-down environments:** if `uv` can't write its default cache (e.g.
     `AppData\Local\uv` on Windows), set `UV_CACHE_DIR=.uv-cache`; if pytest can't reach the
     system temp dir, point `TMP`/`TEMP` at a repo-local folder. `task install` and the first
     `task build` download packages (PyPI + Typst universe), so a network-restricted agent
     needs those runs approved.

2. **First experiment — theirs** (the heart of the flow; one open question, shaped so no one
   freezes on it). Ask what they want to compute, and offer the three doors in one breath:
   *"What should your first experiment be? Name anything small, or tell me your field and I'll
   suggest a few starters — or point me at existing code, a notebook, or a paper and we'll bring
   that in instead. No idea? I've got a safe default ready."*
   - **Starters** (if they want suggestions): ask their field, then offer **three numbered
     one-liners, smallest first** — each an afternoon-small classic with one obvious parameter
     they can vary on their own later. Calibrate to the field; canonical examples: a damped
     pendulum's amplitude decay (physics), a logistic-map bifurcation sweep (dynamics), an
     integrate-and-fire neuron's f–I curve (neuro), an SIR epidemic peak vs `R₀` (epi/bio), a
     random walk's mean-squared displacement (stats/CS). Whichever they pick is built as **their
     experiment, in their repo** — a starter is a first experiment, not a demo.
   - **Universal backup** (no field, "not sure", or nothing above lands): offer a **Monte Carlo
     estimate of π** — throw `N` random points at the unit square, count how many fall inside the
     quarter-circle, `π ≈ 4 · inside / N`. It's field-agnostic, genuinely small, has one obvious
     knob (`N` → a tighter estimate), and shows off the seed + provenance machinery cleanly.
     Always keep it in your back pocket as the fourth, numbered option so no one leaves step 2
     empty-handed.
   - **Bringing something in?** Existing codebase → [`MIGRATE-CODE.md`](MIGRATE-CODE.md); a
     notebook → [`FROM-JUPYTER.md`](FROM-JUPYTER.md); a paper → [`FROM-PAPER.md`](FROM-PAPER.md).
     Land the **first** experiment via that runbook (same live-page payoff as below), and set
     the expectation that the rest migrates incrementally, later.
   - **Stack, in passing:** Python (`uv`) is the default — don't ask, just note it. If their
     idea or their code is MATLAB/Julia/R/Octave, follow [`MIGRATE-STACK.md`](MIGRATE-STACK.md)
     **now**, before building, so the experiment lands on the right stack.
   - **Build it — inline in the runner; do not build a tool.** Say so, and why, in a sentence:
     *"I'll compute this directly in the experiment. A reusable `tools/` module is only worth it
     once you're running the same model across several experiments — if that happens, we'll
     promote it then."* Build a `tools/<name>/tool.py` **only if the user confirms the science
     is genuinely reusable** (RULES §4); never manufacture one to satisfy the shape. Model the
     file shapes on `demolab-engine/scaffold/demo/` (engine reference data — read it, never
     overlay it), following [`../guides/RULES.md`](../guides/RULES.md):
     - **Runner `experiments/expNNN.py`** (model `scaffold/demo/experiments/exp000.py`): compute
       the result **inline**, render the figure(s) into `artifacts/data/expNNN/`, and stage a
       `numbers.json` of the headline metrics + config.
     - **Writeup `writings/expNNN.typ`** (model `scaffold/demo/writings/exp000.typ`): a
       `#let meta` + `#let body` pair; read the run with
       `#let run = json("/artifacts/data/expNNN/numbers.json")`, embed figures with `#image(...)`,
       tables via `#numbers-table(...)` — **never hand-type numbers**. Video via `#video(...)`
       (HTML only).
     - **Only if reuse was confirmed**, add the tool first: `tools/<name>/tool.py` (model
       `scaffold/demo/tools/neuron/tool.py`) — `setup_run_dir`/`write_output`, the data + a
       `manifest` of metrics, **data not plots**, plus `test_<tool>.py` (`task test` green). The
       runner then calls its CLI.
   - `task run -- expNNN` — it rebuilds live in the dev server. Present the new page's URL
     **prominently**, have them open the page and its PDF. Their science, on a published page,
     minutes in.

3. **Make it permanent** (don't skip it — this is where the core contract lands). Their page is
   already on screen from step 2. Say the rule out loud, once: *nothing on that page is typed by
   hand — the write-up reads the run's `numbers.json`, so the prose, tables, and captions can't
   drift from what the code computed.* **Don't ask them to change a parameter and rerun** —
   building the experiment already earned the payoff, and an interactive tweak is friction, not
   teaching; they'll vary parameters on their own once it's theirs. Then make it permanent:
   commit (as them), and run once more so the provenance footer on the page reads **clean** —
   point it out: every result carries the exact commit that produced it.

4. **Brand it** (one pass, offer-with-defaults — their result is on screen; now put their name
   on it). Gather, then write the optional root `demolab.yaml`: site name (default "Demolab"),
   tagline, book/PDF title (defaults from the name), and **author + contact** (offer to pull
   from git config — they render as a byline under the homepage title and an
   `<meta name="author">`; contact, if given, links the byline). The engine defaults any key
   you omit and updates never touch it; `task dev` hot-reloads, so they watch it change. Deeper
   theming (`style.css`, `favicon.svg`) lives inside the black box `demolab-engine/build/` —
   leave it as advanced.

5. **Publish to GitHub Pages?** *"Free, unless the repo is private."* Default yes; if no, skip —
   it all works locally. If yes: create + push a GitHub repo (`gh`), run `task deploy-setup`
   (drops the deploy workflows — one supported path, no options), **offer a custom domain**
   (default `*.github.io`; if custom, write `CNAME` and give the DNS records), then **tell the
   user to flip the Pages setting** (the one UI click you can't do — `deploy-setup` prints the
   exact setting), and push. Run `task build` first to confirm it compiles; confirm the Action
   succeeds and give them the live URL — their experiment, on the internet, citable.

6. **Sign off** — short and warm. The dev server's running and their first result is on a page;
   tell them in your own words:
   - **It works** — their first experiment is built (and live at `<url>` if they published);
     the dev server's up for live preview.
   - **Open the files behind it** — get them into their editor (`code .` / `cursor .` / their
     `$EDITOR`, unless they're already in one) and **link the exact pair their result came
     from**: `experiments/expNNN.py` (the runner — its `CONFIG` block holds every parameter,
     and every metric on the page is computed here) and `writings/expNNN.typ` (the write-up —
     point at the `json("/artifacts/data/expNNN/numbers.json")` line, so they see the page
     really is read from the run, not typed). Invite them to poke: edit either file, `task run
     -- expNNN`, and the page follows.
   - **Show the full menu** — close by running **HELP**: present *every* runbook and then every
     guide as a **numbered list, one per line with its one-line description**, drawn from the
     tables in [`../../AGENTS.md`](../../AGENTS.md) (don't hand-summarise from memory or trim to
     a favourite few — the whole point is they see everything demolab can do next). Runbooks
     first (**NEXT** to pick the next experiment, **TOUR**, **MIGRATE-CODE**, **FROM-JUPYTER**,
     **FROM-PAPER**, **LINT**, **DOCTOR**, **RED-TEAM**, **UPDATE**, …), then the guides
     (**RULES**, **HOUSESTYLE**, **STRUCTURE**, **GLOSSARY**, …). Tell them: **HELP** re-shows
     this menu anytime, and typing any name in it starts that job.
   - **Ask me anything about the repo** — how it works, where something lives, why a convention
     is the way it is.

Notes: provenance is automatic — each run stamps its git commit into `numbers.json` and the
page/PDF footer (an uncommitted run stamps *dirty*; step 3's commit + clean-provenance rerun
turns that from a surprise into the point). The demo under `demolab-engine/scaffold/demo/` is engine reference data and
the landing site's source — read it for file shapes, never overlay it during onboarding.

---

## Agent contract
- **Triggers** — `GETTING-STARTED`, "how do I get started", "help me set up", "onboard me",
  "walk me through this repo".
- **Gates** — the step-0 orient + the user's "ready" must land before **any** command (`clone`,
  `task install`, `task scaffold`, `run`, `dev`). Step 1's quiet verify is a hard gate:
  **nothing of theirs gets built until `task build` and `task test` are green.** Never pull
  branding (step 4) or publishing (step 5) forward — the first experiment comes first.
- **Report & apply** — **conversation-driven, not autonomous.** Orient first, ask the gated
  questions **in order**, wait for each answer before the next, and never scaffold/run/publish
  past a choice the user hasn't made. **Number the options** — present every choice as a
  numbered list (1, 2, 3…) with the recommended default first and marked, so the user can
  answer with a single digit. Bracketed `[n]` below points at the step where each decision
  lands.
- **Order of decisions** — ready → (install approval) → what to compute (which settles
  fresh-vs-migrate and stack as branches, not questions) → brand → publish. The
  user's own experiment is the spine; config and publishing hang off it afterwards, when
  there's something worth naming and shipping.
- **Must-ask** (wait for an answer):
  - **Ready to start?** — after the orient. `[0]`
  - **Install missing toolchain?** — `uv`/`typst`/`go-task`, default yes; skip entirely if all
    three already resolve. `[1]`
  - **What should your first experiment be?** — open but shaped: their idea · field → three
    numbered starters · bring in code/notebook/paper (→ `MIGRATE-CODE` / `FROM-JUPYTER` /
    `FROM-PAPER`). No idea / no field? Fall back to the **Monte Carlo π** starter so no one
    freezes here — this is the one question with a safety net, not a hard no-default. `[2]`
  - **Publish to GitHub Pages?** — free unless the repo is private; default yes, if no skip (it
    works locally). `[5]`
- **Offer-with-a-default** (state it, move on):
  - **Editor** — offer once at step 1 (`code .` / `cursor .`), unless they're already in one;
    at sign-off, prompt again with the concrete pair to explore (`experiments/expNNN.py` +
    `writings/expNNN.typ`). `[1][6]`
  - **Branding** (`demolab.yaml`, one pass): site name (default "Demolab") · tagline · book/PDF
    title (from the name) · author + contact (offer git config). `[4]`
  - **Custom domain, or `*.github.io`?** — default github.io; if custom, write `CNAME` and give
    the DNS records. `[5]`
- **Branch, don't ask** — fresh-vs-migrate and stack are **branches of the what-to-compute
  answer**, not standalone questions: code/notebook/paper offered → the matching runbook;
  MATLAB/Julia/R/Octave named → `MIGRATE-STACK` before building. Python + fresh needs no
  question at all. `[2]`
- **Surface, then default to inline** (do *not* silently build a tool): **tool-vs-inline** `[2]`
  — a first experiment computes **inline in the runner**. A `tools/` CLI is only worth it for
  science **reused across multiple experiments/writeups**. Say this to the user, and build a
  tool *only if they confirm reuse* — never manufacture one to satisfy the shape (RULES §4).
- **Defer** (only if the user raises it): demo content (`task add-demo-content` — the worked
  example, also viewable at <https://demolab.eoinmurray.info>; `task clear-demo-content`
  undoes it), collections (§6.5), house style (`HOUSESTYLE.local.md`), license (ships MIT).
- **Instruction, not a choice:** flipping the Pages setting is a GitHub-UI click you can't do —
  tell the user. And commits never record agent authorship (a rule, not a prompt).
- **Two things you drive, not ask:** the **dev server** (start it at step 1 and leave it up, so
  the experiment and branding render live) and the **clean-provenance commit** (step 3 — commit
  as them, rerun so the footer reads clean, and state the no-hand-typing contract; **don't** ask
  them to vary a parameter and rerun).
- **Meet the tree where it is.** If the toolchain is already present and the tree already
  scaffolded, **resume at step 2**, acknowledging it ("your lab's already standing — let's get
  your first experiment in") rather than re-running install.
