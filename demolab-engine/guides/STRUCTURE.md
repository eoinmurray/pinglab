# STRUCTURE — the repo's filesystem and layout

A map of what every part of a demolab repo is and where it lives. For the *why* behind the
zones (what updates wholesale, what's yours) see the firewall in [`RULES.md`](RULES.md) §3;
for term definitions see [`GLOSSARY.md`](GLOSSARY.md). Bracketed `[§x]` / `[Gn]` / `[Hn]`
tags point at the rule that governs each path.

demolab ships **engine-only** — a clean tree with just `demolab-engine/` and the reconciled
root files, no content dirs. `task scaffold` lays down the empty structure below; `task
add-demo-content` also overlays the worked demo. The demo + skeleton live *inside the engine* at
`demolab-engine/scaffold/` (so they version with it), and doubles as the engine's smoke test
([`test_engine_build.py`](../build/test_engine_build.py)). See [Scaffolding](#scaffolding) below.

## The tree

The working tree **once scaffolded** (a fresh clone has only the `demolab-engine/` + root-file
rows):

```
demolab/
├── tools/                  the science — one directory per tool          [§3.4, §4, §8, G23]
│   ├── neuron/               tool.py (the CLI) + test_neuron.py
│   └── mujoco/               tool.py + test_mujoco.py
├── experiments/            the runners — one expNNN.py per experiment     [§3.4, §7, G22]
│   ├── expNNN.py             runs a tool's CLI, renders figures, stages artifacts/data/expNNN/
│   ├── helpers/              shared runner code (not experiments) — style.py (figure style, H15), provenance.py (§4.7)
│   └── playground.py         the interactive Streamlit demo (exempt from the contract, §8.5, G16)
├── writings/               the writeups — one .typ per entry, by id       [§6, G24, HOUSESTYLE]
│   ├── expNNN.typ            an experiment writeup (#let meta + #let body)
│   ├── arNNN.typ             an article — prose-only, no runner            [G1]
│   └── arNNN.slide.typ       a deck — Touying slides → standalone PDF      [G9]
├── artifacts/              the committed record of every run              [§5]
│   ├── data/<id>/            figures + numbers.json + run.sh (+ any mp4) — the publisher-neutral record  [§5.1, G18]
│   ├── pdfs/                 compiled PDFs (per entry + book.pdf) — shareable                    [§5.3]
│   └── site/                 the built web site — GITIGNORED (CI rebuilds + deploys it)          [§5.3]
├── demolab.yaml            optional branding + collections config          [§3.3, §6.5, G4]
├── HOUSESTYLE.local.md     optional — your house-style overrides (extend/replace)  [§3.3]
├── demolab-engine/         the engine — the BLACK BOX, swapped wholesale on update  [§3.1, G3]
│   ├── VERSION               the engine's SemVer (`task version`); compared on update
│   ├── CHANGELOG.md          notable engine changes per version (Keep a Changelog)
│   ├── build/                the Typst publisher                           [§5.2, G9]
│   │   ├── main.typ            the bundle root — reads the manifest, emits every document
│   │   ├── lib.typ            shared helpers (pages, numbers-table, video, collections)
│   │   ├── build.py           discovers writings → manifest, then compiles main.typ
│   │   ├── style.css          the web theme
│   │   ├── cite-popover.js    hover popovers for inline citations (web-only)
│   │   └── favicon.svg        the tab icon
│   ├── scaffold/             what `task scaffold`/`add-demo-content` lay down (+ smoke-test fixture)
│   │   ├── skeleton/           the bare structure: writings/ experiments/ tools/ artifacts/ + config
│   │   ├── demo/               the worked demo overlaid by add-demo-content (tools, exp00*, writeups)
│   │   ├── demo-manifest.json  the exact paths `task clear-demo-content` removes
│   │   └── ../build/test_engine_build.py  builds these fixtures end-to-end as `task test`
│   ├── runbooks/             the 17 agent runbooks (one file each)         [AGENTS.md, G21]
│   └── guides/               RULES.md · GLOSSARY.md · HOUSESTYLE.md · SLIDES.md · AUTORESEARCH-RULES.md · STRUCTURE.md (this file)
├── AGENTS.md               the thin entry point — agents read this first   [§3.2]
├── CLAUDE.md               a thin pointer to AGENTS.md (for Claude Code)    [§3.2]
├── README.md               the human-facing overview                       [§3.2]
├── Taskfile.yml            the task wrapper (task install/run/dev/build/test/…)   [§1.3]
├── pyproject.toml          pinned Python deps (uv)                          [§1.1]
├── uv.lock                 the resolved lockfile
├── .github/workflows/      CI — builds the bundle, deploys artifacts/site/ to Pages   [§3.2, §5.3]
├── .gitignore              ignores temp/, artifacts/site/, .venv/, .claude/, …
├── LICENSE
└── temp/                   short-lived run scratch — GITIGNORED             [§5.1]
    ├── <tool>/<cmd>/          a tool run's raw output (config/output/manifest/log/run.sh/data/mp4)  [§4.3]
    └── bundle/               build scratch: index.json manifest + compiled deck PDFs
```

## Reading the tree

**S1 — Split by kind, not by pipeline stage.** Code (`tools/`), runners (`experiments/`), prose (`writings/`), and the record (`artifacts/`) each have a home — a result isn't scattered across a per-experiment folder. What ties an experiment together is its **id**, not a directory.

**S2 — One id threads through, by name.** `experiments/exp000.py` runs it → `artifacts/data/exp000/` holds its figures + `numbers.json` → `writings/exp000.typ` writes it up. Ids: `expNNN` (experiment), `arNNN` (article), `arNNN.slide` (deck). Same id, three places.

**S3 — Three kinds of writing.** `expNNN.typ` (an experiment's writeup, reads its run), `arNNN.typ` (a prose-only article), and `arNNN.slide.typ` (a deck, compiled to a standalone PDF). `build.py` discovers the first two as bundle entries (`#let meta` + `#let body`); a `.slide.typ` is listed but rendered only as PDF (§6.1, G24).

**S4 — Committed vs regenerable.** *Committed* (the record CI can't reproduce, so it must be in git): `artifacts/data/` and `artifacts/pdfs/`. *Gitignored* (rebuilt on demand): `temp/` and `artifacts/site/`. Never commit scratch; never rely on it surviving (§5.1, §5.3).

**S5 — The black box vs your stuff.** Everything under `demolab-engine/` is pure upstream — you never hand-edit it, and *"update demolab"* overwrites it wholesale (§3.1). Everything you own lives *outside* it: `tools/`, `experiments/`, `writings/`, `artifacts/`, and your `demolab.yaml`. Root files (`AGENTS.md`, `README.md`, `Taskfile.yml`, `pyproject.toml`, CI) are framework-but-reconciled — updated by diff, not swap (§3.2).

**S6 — Where the build goes.** `task build` runs `demolab-engine/build/build.py`, which globs `writings/*.typ` into `temp/bundle/index.json`, then compiles `demolab-engine/build/main.typ` to three targets in one pass: the web site → `artifacts/site/`, per-entry PDFs + `book.pdf` → `artifacts/site/pdfs/` (mirrored to `artifacts/pdfs/`). On an empty (freshly-scaffolded) tree the build still succeeds — it emits a single `index.html` with a friendly empty state, and skips `all.html`/collection pages/`book.pdf`. CI deploys `artifacts/site/` to GitHub Pages (§5.2, §5.3).

## Scaffolding

demolab ships engine-only; the content tree is materialised on demand from `demolab-engine/scaffold/`.

**S7 — Two overlays, one command each.** `skeleton/` is the bare structure (empty `writings/` `experiments/` `tools/` `artifacts/` + the config templates `demolab.yaml`, `HOUSESTYLE.local.md`, `experiments/helpers/style.py` + `experiments/helpers/provenance.py`); `task scaffold` copies it into the repo root non-destructively (`rsync --ignore-existing`, so re-running never clobbers your work). `demo/` is the worked example; `task add-demo-content` runs `scaffold` then overlays it. `task clear-demo-content` deletes exactly the paths in `demo-manifest.json` — nothing you authored is listed there, so it can't touch your content.

**S8 — The demo is the engine's test.** Because `demo/` lives inside the swapped-wholesale `demolab-engine/`, it versions with the engine and can't drift from it. [`test_engine_build.py`](../build/test_engine_build.py) (run by `task test`) assembles skeleton + demo in a throwaway tree via `DEMOLAB_ROOT` and builds it end-to-end — so the shipped example is also the integration smoke test. It also asserts the empty (skeleton-only) tree builds its empty-state homepage.
