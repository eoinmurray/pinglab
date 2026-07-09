# Changelog

Notable changes to the **demolab engine** (`demolab-engine/`). Format follows
[Keep a Changelog](https://keepachangelog.com); demolab uses [SemVer](https://semver.org) while
`0.x` signals the contract is still stabilising:

- **major** — a break that may need edits to *your* content (the tool↔experiment contract, the
  firewall, or the `meta` schema). *"update demolab"* flags it for review.
- **minor** — a new backward-compatible capability.
- **patch** — fixes and tweaks.

The current version lives in [`VERSION`](VERSION); `task version` prints it. On *"update demolab"*
the runbook shows the entries between your version and the latest.

## [Unreleased]

### Added
- **Semi-autonomous research programs (AUTORESEARCH).** A new flow for steered-by-day,
  run-overnight research: a program is one collection with a pre-registered `plan` article
  (hypothesis + kill criteria + a machine-readable experiment queue), an append-only `log`
  article (decisions/anomalies, digest at top), and a night-shift contract (budgets, scope,
  stop conditions). Three runbooks — **AUTORESEARCH** (start/steer a program), **PLAN** (triage
  last night + queue the next pre-registered experiments), **NIGHT-SHIFT** (work the queue on a
  branch, open a PR whose description is the digest) — plus the **AUTORESEARCH-RULES** guide
  they cite. DOCTOR gains a one-plan-one-log / every-queued-entry-has-a-kill check; LINT exempts
  the `log` from the prose rules; NEXT reads the `log`'s decision arc. Opt-in and additive:
  nothing changes for a lab that doesn't create a `plan` article.
- **Per-PR site previews (`deploy.yml` rewrite + new `preview.yml`).** `task deploy-setup` now
  installs two workflows. Production (`deploy.yml`) switched from the GitHub-Actions Pages flow
  to the **branch-based** flow (build main → commit `artifacts/site` to a `gh-pages` branch, via
  `JamesIves/github-pages-deploy-action`, `single-commit`, `clean-exclude: [pr-preview/, CNAME]`).
  Previews (`preview.yml`, via `rossjrw/pr-preview-action`) build every PR to
  `pr-preview/pr-<N>/`, post a sticky comment with the URL, and tear down on close/merge — so a
  night-shift PR is a clickable doc set, not a raw diff. A new build test asserts the emitted
  HTML carries no root-absolute URLs, since subpath-served previews depend on relative links.

  > **Migration — action required if you already publish (`.github/workflows/deploy.yml` exists).**
  > The new deploy uses a `gh-pages` branch instead of the GitHub-Actions Pages source, so an
  > *"update demolab"* that refreshes the template will not publish until you flip one setting.
  > After updating:
  > 1. Re-run `task deploy-setup` (rewrites `deploy.yml`, adds `preview.yml`), commit, push to main.
  > 2. **Settings → Pages → Source: "Deploy from a branch" → `gh-pages` / `(root)`.** The first
  >    push to main creates the branch; until you switch the source, the site keeps serving the
  >    last GitHub-Actions deploy.
  > 3. (Recommended) **Settings → General → Pull Requests → "Automatically delete head branches"**,
  >    and protect `main`.
  > On a **custom domain**, your `CNAME` is preserved by `clean-exclude` — no action. A lab that
  > never opted into Pages is unaffected until it runs `task deploy-setup`.

### Changed
- **`task dev:demo-site` reads and serves directly from `demolab-engine/scaffold/demo/`.** Sets
  `DEMOLAB_ROOT` there (no root symlinks, no `temp/demo-site/` staging). A `content-prefix`
  Typst input + `data-file()` in `lib.typ` let demo writings resolve `/artifacts/data/…` while
  `--root` stays at the repo checkout for the engine.
- **Install scripts moved into the shipped demo.** `install.sh`, `install.ps1`, and `CNAME` now live
  at `demolab-engine/scaffold/demo/site/` (served at the Pages root alongside the demo). The
  top-level `landing/` directory is gone. `task add-demo-content` skips `site/` so installers
  don't land in a user's lab root.

## [0.5.2] — 2026-07-08

### Added
- **Homepage welcome block (`demolab.yaml` `welcome:`).** An optional hero on the index page —
  pitch prose, quick links, install commands, and an agent prompt — rendered above the
  collection directory. Absent on a normal lab; the shipped demo sets it so
  demolab.eoinmurray.info is the example site with a welcome, not a separate landing page.

### Changed
- **Upstream Pages deploy publishes the demo at `/`.** `.github/workflows/landing.yml` now uploads
  `artifacts/site/` at the site root (plus `install.sh`, `install.ps1`, and `CNAME` from
  `landing/`). The old `/example/` prefix and standalone `landing/index.html` are gone.

## [0.5.1] — 2026-07-08

### Added
- **Staged-experiment flow (RULES §7.5) + a DOCTOR check for it.** An optional, opt-in
  contract for expensive runners: split the runner into `compute → analyse → plot` so a run
  can re-enter at any stage (full / skip-compute / plot-only) without repeating the costly
  prefix. The one boundary the contract enforces is that the **plot** stage reads only from the
  committed `artifacts/data/<id>/` record — which is what makes a plot-only pass reproducible
  from a clean clone with no `temp/` (§5.1, §5.3). The rule deliberately defines the *flow and
  the boundary, not the mechanism*: no mandated flag names, stage harness, or function
  signatures, and most runners stay one-shot. DOCTOR §3 adds a matching **behavioural** check —
  re-run a staged runner's plot-only mode with `temp/` hidden and confirm the figures still
  render; a pass that only fails with scratch hidden is reaching into `temp/`. Advisory unless
  the experiment claims clone-and-replot, since plotting from warm `temp/` while iterating is fine.

## [0.5.0] — 2026-07-08

### Added
- **Experiment runners now carry provenance like tools do**, via a new
  `experiments/helpers/provenance.py`. Two helpers: `stamp(config)` adds the same
  `_provenance` block (git commit, `dirty` flag, UTC timestamp) that a tool's
  `setup_run_dir` writes — so an **inline** runner (no tool to inherit from) produces a
  `numbers.json` indistinguishable to `numbers-table` / `provenance-footer` / DOCTOR; and
  `write_run_sh(ARTIFACTS)` drops a `run.sh` reproducer into the committed
  `artifacts/data/<id>/` record — the committed twin of the `run.sh` tools write into
  scratch `temp/`. The demo runners (`exp000`–`exp003`) now emit `run.sh`, DOCTOR flags a
  record missing one, and the stamp is kept as a separate copy from `tools/*/tool.py`
  because the firewall (§4.5) forbids a tool importing `experiments/`. (RULES §4.1, §4.7,
  §7.2; FROM-JUPYTER step 4–5.)

## [0.4.6] — 2026-07-07

### Changed
- **Runbooks lead with a human overview, then a labeled `Triggers` line.** Each runbook opened with
  its agent-routing `Triggers:` line, which reads as machinery to a person. Every runbook now opens
  with a plain-language description of what it does and when to use it, followed by a
  `**Triggers** — say any of these, or just \`NAME\`:` line. The trigger phrases are unchanged and
  the step-by-step bodies are untouched, so agent routing is unaffected; the files just read as
  documentation for a human too. (The guides already led with a human summary.)

## [0.4.5] — 2026-07-07

### Changed
- **The dev server's rebuild log is now one terse line.** `task dev` reprinted the full output paths
  and every entry/deck id on *every* rebuild, so an active editing session buried the terminal in
  wrapping dumps. `build.py` now prints a concise summary last
  (`built 9 entries + 2 decks -> artifacts/site/`) — which is all the watch loop echoes per rebuild —
  and keeps the full id list on the line above, shown only by a one-shot `task build`.

## [0.4.4] — 2026-07-06

### Fixed
- **Dev server hardening (round two).** Closed a path-traversal hole — the `.html` serving path read
  `SITE / <request path>` directly, so a crafted `..%2f…​.html` could read files outside the served
  dir; it's now confined to the site (verified a traversal request returns 404). Also bounded the
  per-tab SSE queue (a stalled tab can't grow it without limit) and unified benign-disconnect
  handling across every socket write.

### Changed
- **A broken entry no longer takes down the whole site.** One entry that referenced a missing figure
  (or had any Typst error) aborted the single bundle compile, so the *entire* site failed to build.
  `build.py` now flags the failing entry and retries, and `main.typ` renders it as a visible stub
  page at its own URL — a red "this page failed to build" notice with the error — kept out of the
  listings and the book, while every other page builds. Decks that fail to compile are skipped the
  same way. The build reports what it stubbed and still exits 0, so the dev server serves the good
  site with the broken page visibly flagged rather than an all-or-nothing error overlay.

## [0.4.3] — 2026-07-06

### Fixed
- **`task dev:demo-site` previews the demo's config, not a stale root copy.** It symlinked the demo's
  content but not its `demolab.yaml` / `HOUSESTYLE.local.md` (which live in the skeleton), so the
  preview built against whatever config happened to sit at the root — losing branding and collection
  labels/descriptions. It now symlinks those too (and tears them down on exit), so the preview
  matches what ships, and editing the demo's `demolab.yaml` hot-reloads.

## [0.4.2] — 2026-07-06

### Fixed
- **Dev server no longer spews tracebacks or dies quietly.** A browser resetting a connection
  (closing an SSE stream, reloading, navigating away) used to dump a `ConnectionResetError` stack —
  harmless, but it reads as a crash. The server now swallows those benign disconnects (a custom
  `handle_error`), and hardens against the genuinely bad case: the watch loop survives a transient
  error instead of silently stopping all rebuilds, a hung compile times out (120s) instead of
  freezing the server, and a raced port falls through to the next.

## [0.4.1] — 2026-07-06

### Changed
- **Command grammar: the NAME is the trigger.** `HELP` lists the runbooks + guides; a runbook's
  SCREAMING-KEBAB name (`LINT`, `DOCTOR`, …) starts it, a guide's name (`RULES`, `SLIDES`, …) walks
  the user through it. AGENTS.md is restructured around the three commands with the name made
  primary (was "a phrase, or the name"), and the grammar is mirrored into the always-loaded
  `CLAUDE.md` so a bare name routes without the agent having to have read AGENTS.md first — the
  reason a bare `LINT` misfired in older repos. `test_command_catalog.py` keeps the AGENTS tables in
  step with the actual runbook/guide files.

## [0.4.0] — 2026-07-06

### Added
- **Slide layouts are a named, liftable catalog.** Each layout is a block in the gallery deck
  (`ar005.slide.typ`) marked `// layout: <name>`; SLIDES.md D11 is the index (name → when-to-use), and
  you build a slide by copying the named block out of the gallery rather than from a Typst component
  library. One tested source of truth — `test_slide_catalog.py` asserts the gallery's marker names
  match the D11 catalog, so the two can't drift.
- **`task dev:demo-site`.** Serves the shipped demo (`demolab-engine/scaffold/demo/`) through the
  live engine by symlinking its content into the root — one source of truth, no duplicated files —
  and tears the links down on exit. Handy for previewing the reference lab or developing engine
  features against it without a full `add-demo-content` sandbox.
- **Demo ships the layout-gallery deck (`ar005`).** `SLIDES.md` D12 points authors at a gallery
  deck to copy layouts from; the demo now actually includes it — one slide per D11 layout, driven by
  real run data. Expanded to cover five more layouts (three-column, quote, section divider, big
  number, diagram) and to match demolab's two-ink web palette instead of touying's teal accent (the
  D4 skeleton now carries the palette recipe). The slide's title labels its layout (no separate tag).
  Decks now default to `header: none` — touying's running section header just reprinted the title on
  every slide; turn it back on for a long, multi-section talk. Slide content is also vertically
  centred by default (`#set align(horizon)`) — adaptive, so sparse slides balance and full ones keep
  their title near the top.

## [0.3.0] — 2026-07-06

### Changed
- **New dev server (`task dev`).** Replaced `typst watch`'s built-in server with a small Python
  dev server (`demolab-engine/build/devserver.py`) that rebuilds via `build.py` on any source
  change and serves the site with live-reload. It fixes two long-standing dev annoyances: a **new
  entry or deck now appears without restarting** (the build re-globs the filesystem each time, which
  `typst watch` couldn't), and a **failed compile shows up in the browser** as a full-screen overlay
  carrying the Typst error — clearing on the next good build — instead of silently serving the stale
  site with the error buried in the terminal. To keep saves snappy it skips deck recompilation
  when the change touched no `.slide.typ` or data asset, so a prose/CSS/lib edit rebuilds in ~0.4s
  instead of ~1.1s. Trade-off vs `typst watch`: a full bundle compile per save rather than Typst's
  incremental recompile. No Node, no new dependencies.

## [0.2.5] — 2026-07-06

### Added
- **`guides/SLIDES.md` — deck authoring guide.** Conventions for `writings/*.slide.typ` decks,
  numbered `D1–D13`: the `.slide.typ` marker and skeleton, sizing in absolute `pt` against the
  842 × 474 pt canvas (~350 pt usable under a title), per-aspect figure rows, the
  silent-pagination overflow trap and the page-count check, the layout vocabulary (bullets,
  two-column, code, equation + terms, four figure layouts, table, focus, closer), and the
  dev-server caveat for decks created mid-`task dev`. Indexed from `AGENTS.md`, RULES §3.1,
  STRUCTURE's tree, and GLOSSARY G9.
- **`pending-figure` — placeholder for an unrendered figure.** A `#pending-figure(caption: …,
  note: …, ratio: …)` helper (and the `#pending` body it wraps) stands in for a figure whose asset
  isn't ready yet — a re-run in flight, data not cleared for release. It numbers as a normal
  "Figure N" and reserves the figure's footprint (a tinted, dashed, rounded panel with a small
  framed-image mark over the muted reason) so the page doesn't reflow when the real plot lands.
  Replaces the bare floating text that a missing asset used to leave on the web. RULES §6.2.
- **Entry pdf link moved inline.** On an entry page the `pdf` link now sits in the meta strip next
  to the status (`ar000 · 30 May 2026 · Revising · pdf`) instead of being flexed to the right edge
  of the title row.
- **Inline citations render demibold.** A `#cite(…)` (brackets included) now stands out from the
  body at weight 600 on both targets. CMU ships only Roman + Bold, so the web loads a real
  mid-weight face — Latin Modern Roman Demi, the CM-lineage demibold — for weight 600; it's fetched
  only on pages that actually carry a citation.

### Changed
- **Status ordering reversed.** Listings now sort `final → revising → building → draft` (settled
  work first) instead of leading with work-in-progress.
- **LINT now audits the figures, not just the prose.** The lint runbook gained a Figures pass
  (H10–H15): mechanical greps over the rendered assets (`artifacts/data/<id>/*.svg`, `*.png`) for
  palette, white background, format, and `alt:` text, plus a **required vision pass** (§2b) that
  opens each figure to check labelled axes with units, no baked-in title, legibility, aspect,
  grayscale-safety, and central style. Documents the SVG trap — matplotlib writes axis labels as
  glyph paths, not `<text>`, so plots can't be grep-linted — the reason agents were linting prose
  and silently skipping the plots.

## [0.2.4] — 2026-07-06

### Fixed
- **`#cite` spacing ([#1](https://github.com/eoinmurray/demolab/issues/1)).** The inline citation
  was set flush against the preceding word (`runs[2]`). The helper now owns a thin gap before the
  bracket — a weak `h()` in the PDF, a `margin-left` on the web span — so authors attach `#cite`
  directly to the word (`runs#cite(2)`) and the bracket keeps its space without ever orphaning onto
  the next line. Documented the convention in HOUSESTYLE H24.

## [0.2.3] — 2026-07-06

### Added
- **Heading anchors.** Every heading on a web page now carries a slug `id` (its text lowercased,
  non-alphanumerics collapsed to hyphens), so any section is directly linkable as
  `entry.html#the-slug`. A quiet `#` permalink fades in on hover to grab that URL. Applies to entry
  titles, section/subsection headings, and the auto-built References heading.

## [0.2.2] — 2026-07-06

### Fixed
- **Figure numbering restarts per entry.** The whole bundle compiles in one pass, so Typst's
  global figure counter was carrying across every document — a standalone entry PDF could open at
  "Figure 7". Each entry (its page + standalone PDF) now numbers figures from 1; the book keeps
  numbering continuously 1…N across chapters.

## [0.2.1] — 2026-07-06

### Changed
- **LINT enforces the references system.** The lint runbook now flags any hand-rolled citation —
  typed `[1]` brackets, a manual `== References` section, literal `doi.org` links, or author–year
  cites like "(Smith 2020)" — as an H24 violation; references must go through `#cite` +
  `#reference-list`. Also documented the system fully in RULES §6.6.

## [0.2.0] — 2026-07-06

### Changed
- **Listing layout stacked.** Entry rows now put the `id` + title on top with a quiet
  `date · status · pdf` sub-line beneath the title, instead of right-aligning the meta — so long
  titles wrap cleanly without orphaning the metadata. Applies to every listing (collection pages +
  `all.html`).

## [0.1.0] — 2026-07-06

Initial versioned release — the engine after its foundational build-out.

### Added
- **Engine-only distribution.** The repo ships with no content; `task scaffold` lays down the bare
  structure, `task add-demo-content` overlays a worked demo, `task clear-demo-content` removes it.
  The demo lives in `demolab-engine/scaffold/` and doubles as the smoke-test fixture.
- **One-line installers** — `install.sh` (macOS/Linux) and `install.ps1` (Windows), served from the
  project landing page at demolab.eoinmurray.info.
- **14 agent runbooks** — getting-started, tour, migrate-code, from-jupyter, from-paper,
  migrate-stack, embed-docs, next, ground-claims, lint, doctor, red-team, steelman, update — plus a
  `HELP` index, each triggerable by bare name.
- **Citations** — `#cite(...)` inline numbered cites + `#reference-list(...)` with DOI links, and
  Wikipedia-style hover popovers on the web (DOIs open in a new tab).
- **Entry status** — a free-form `meta.status` (`draft`/`revising`/`final`), shown as plain text
  across every listing and entry page, and driving listing order (Articles → Experiments → Slides,
  then status, then id).
- **Author/contact branding** — a byline under the homepage title + `<meta name="author">`.
- **Friendly empty-state homepage** on a freshly-scaffolded repo.

### Changed
- The figure-data format is the author's choice (CSV, JSON, `.npz`, …); only the contract files
  (`config`/`output`/`manifest`/`numbers.json`) stay JSON.
