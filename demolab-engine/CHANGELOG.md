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
