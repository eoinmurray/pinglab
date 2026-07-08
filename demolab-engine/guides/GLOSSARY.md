# GLOSSARY — demolab's vocabulary

The words demolab uses for its parts, defined once. Terms are numbered `G<n>` (citable, like
RULES' `§N.M`) and listed alphabetically. Section references like `§4.3` point into
[`RULES.md`](RULES.md), where the mechanics live.

**G1 — Article.** A prose-only writeup, `writings/ar<NNN>.typ`. No runner, no tool; it uses no data pipeline (§4.1). Contrast **experiment** (G12).

**G2 — Artifacts.** `artifacts/`: the committed record. `artifacts/data/<id>/` holds a run's figures + `numbers.json`; `artifacts/pdfs/` holds the shareable PDFs; `artifacts/site/` is the gitignored web build (§5.1, §5.3).

**G3 — Black box.** `demolab-engine/` (the `build/` engine, `runbooks/`, `guides/`). Pure upstream, never hand-edited; *"update demolab"* overwrites it wholesale (§3.1). Your customisation lives *outside* it, in `demolab.yaml` and your content.

**G4 — Brand config.** The optional root `demolab.yaml`: wordmark, PDF titles, and the collection registry. Absent ⇒ engine defaults (§3.3, §6.5).

**G5 — Bundle.** The Typst multi-document output from one compile: the site, per-entry PDFs, and the book, all emitted together by `main.typ` (§5.2).

**G6 — Coding agent.** The AI assistant (Claude Code, Cursor, aider, …) that reads this repo and operates it: runs the toolchain, wires files, follows runbooks.

**G7 — Collection.** A group of entries sharing a `collection:` slug in their `meta`. Entries are grouped by collection on the homepage; a slug title-cases by default, and `demolab.yaml` can give it a label/description/order (§6.5).

**G8 — Contract.** The file-based interface between a tool and an experiment: a tool writes a fixed set of files (§4.3), a runner reads them by running the tool's CLI (§4.5). It's language-neutral (§1.4).

**G9 — Deck.** A `writings/<id>.slide.typ` Touying slide deck. Paged-only, so it's compiled to a standalone PDF and grouped under the `slides` collection — listed, but not rendered as an HTML page and not an **entry** (G11). Authoring guide: [`SLIDES.md`](SLIDES.md).

**G10 — Engine.** `demolab-engine/build/`: the Typst publishing code (`main.typ`, `lib.typ`, `build.py`, `style.css`, `favicon.svg`). The heart of the black box (G3).

**G11 — Entry.** A published item that becomes its own page: an **experiment** (G12) or an **article** (G1). Decks (G9) are listed but aren't entries.

**G12 — Experiment.** A runner + writeup pair sharing an id: `experiments/exp<NNN>.py` (runs a tool, stages results) + `writings/exp<NNN>.typ` (the writeup). Contrast **article** (G1) (§4.1, §7).

**G13 — Headline metrics / video.** The fields (and optional mp4) a tool's `manifest.json` declares as the run's surfaced results; `write_output` validates them (§4.2, §8.3).

**G14 — Lab notebook.** The whole repo. *Not* the Jupyter kind — a published, citable record of computational work.

**G15 — Manifest.** Two things: (a) `manifest.json`, which a tool writes to declare its headline metrics/video (§4.3); (b) `temp/bundle/index.json`, the build manifest `build.py` writes for `main.typ` to read (§5.2).

**G16 — `numbers.json`.** The aggregated, committed record of a run: each command's `config.json` + headline metrics + a `_provenance` stamp, in `artifacts/data/<id>/` (alongside a `run.sh` reproducer the runner writes). Writings read it so numbers can't drift (§4.6, §5.4).

**G17 — Playground.** `experiments/playground.py`, the interactive Streamlit demo. Not an `exp*` runner; exempt from the contract, but still drives a tool's CLI (§8.5).

**G18 — Provenance.** The `_provenance` block stamped into every run: git commit SHA, a `dirty` flag, a UTC timestamp. Stamped by a tool's `setup_run_dir` or, for an inline runner, `helpers/provenance.stamp`. Surfaced as a page/PDF footer (§4.7).

**G19 — Record.** See **artifacts** (G2), `artifacts/data/<id>/`: the publisher-neutral, committed output of a run.

**G20 — Rendering.** A tool-produced video (e.g. `mujoco` → `.mp4`) — the one non-tabular output a tool emits itself, since it isn't a plot of data (§4.2).

**G21 — Runbook.** An agent procedure in `demolab-engine/runbooks/`, fired by a trigger phrase (e.g. *"how do I get started"*). The source of truth for operating the repo.

**G22 — Runner.** `experiments/exp<NNN>.py`: runs a tool's CLI (or computes inline), renders figures from the data, and stages `artifacts/data/<id>/`. Never imports a tool (§4.5).

**G23 — Static site.** The published output: plain HTML + assets, no server to run, free to host (GitHub Pages).

**G24 — Tool.** A small program (Python by default) holding *reusable* science — a model or solver behind a small CLI that writes the contract files. Reuse is the bar for making one (§4.1); a tool emits data, not plots (§4.2).

**G25 — Writing.** Any `writings/<id>.typ`: an experiment's writeup, an article, or (with `.slide.typ`) a deck. A `meta` + `body` pair the engine discovers and publishes (§6.1).
