# RULES — demolab's conventions

The single source of truth for how a demolab repo is structured and the conventions to
follow — principles *and* the tool ↔ experiment contract. It lives inside the engine
(`demolab-engine/guides/`), so it updates wholesale on *"update demolab"*; the root
`AGENTS.md` / `CLAUDE.md` are thin pointers to it. For a user-facing overview and run
instructions, see [`README.md`](../../README.md).

Rules are numbered `§<section>.<rule>` so other docs (e.g. the *Doctor the repo* runbook)
can cite them precisely. Terms (tool, experiment, deck, collection, provenance…) are
defined in [`GLOSSARY.md`](GLOSSARY.md).

## 1. Toolchain

**1.1 — Python via `uv`.** Never call `python` / `python3` directly. Deps are pinned in the root `pyproject.toml` / `uv.lock`; run scripts with `uv run python <script>` (e.g. `uv run python tools/neuron/tool.py lif`); `uv sync` after pulling.

**1.2 — Publishing via `typst`.** Use the `typst` CLI (an installed prerequisite, like `go-task`). It compiles the site + PDFs (`task build` / `task dev`); the bundle build passes `--features bundle,html` (experimental, deliberately used here). No Node/`bun` — demolab publishes entirely with Typst.

**1.3 — Prefer `task`.** Common commands are wrapped in a `Taskfile` — prefer `task <name>` (run `task` to list them). To check the toolchain *and* that the repo obeys these conventions, run the *Doctor the repo* runbook (*"doctor the repo"*).

**1.4 — Python is the default, not a requirement.** The contract (§4) is *file-based and language-neutral* — a tool is reached by running its CLI (a subprocess), never by importing it, and speaks only through files; Typst publishes from `numbers.json` + PNGs regardless of what produced them. To move the tool layer to MATLAB / R / Julia / Octave, follow the *Migrating the stack* runbook ([MIGRATE-STACK.md](../runbooks/MIGRATE-STACK.md)).

## 2. Commits

**2.1 — Human-only authorship.** Author every commit as the human only. **Never** record an agent as author or co-author: no `Co-Authored-By:` trailer naming Claude or any AI, no agent name in the author/committer fields, no "🤖 Generated with …" line. The history reads as the human's own work.

## 3. Repo layout — the framework/content firewall

**3.1 — Black box** (pure upstream; never edited here, swapped wholesale on update): `demolab-engine/build/` (the Typst engine: `main.typ`, `lib.typ`, `build.py`, `style.css`, `favicon.svg`), `demolab-engine/runbooks/` (the runbooks), and `demolab-engine/guides/` (this file + `GLOSSARY.md` + `HOUSE-STYLE.md`). Updates cleanly and survives any deletion of example content.

**3.2 — Reconciled** (framework, but kept thin or pinned to root by tooling; updated by diff, not swap): `AGENTS.md` + its `CLAUDE.md` pointer (both thin — they just point here), `README.md`, the `Taskfile`, `pyproject.toml`, and `.github/` CI.

**3.3 — Branding, yours, optional** (a root override the engine reads; never overwritten by updates): `demolab.yaml` — the wordmark + PDF titles. Absent ⇒ engine defaults. Deeper theming (`style.css`, `favicon.svg`) currently lives inside the black box, so editing it is possible but gets overwritten on update — treat as advanced.

**3.4 — User content** (100% the user's — freely deletable and replaceable): `tools/*`, `experiments/*` (runners, plus `playground.py` — the Streamlit demo, exempt from the contract), `writings/*` (`.typ` writeups), `artifacts/*` (`data/` per-run figures + `numbers.json`, `pdfs/` compiled PDFs; `artifacts/site/` is a gitignored build), `temp/*` (regenerable scratch).

## 4. The tool ↔ experiment contract

**4.1 — Reuse is the bar.** A tool exists to hold *reusable* science — a model or solver run across more than one experiment, or the same one re-run with swept parameters. Using a tool is a choice, not a requirement: a genuine one-off can compute inline in its runner and stage its own `artifacts/data/expNNN/` directly; articles (`ar*`) use no tool at all. **Don't manufacture a tiny tool to satisfy the contract** — the CLI, manifest, tests, and import firewall earn their ceremony by being *shared*. Going inline trades away the manifest validation, provenance stamp, and easy unit-testing — fine for a throwaway. When reuse actually appears, **promote** the code into `tools/<tool>/tool.py` then and point the runner at its CLI — not in anticipation.

**4.2 — Tools emit data, not plots.** A tool writes the CSV/JSON a figure is drawn *from*; drawing the figure is the runner's job. The one exception is a **rendering** (a physics video, `mujoco` → `.mp4`), which a tool *does* produce. `write_output` validates that a declared `headline_video` exists on disk and that every `headline_metrics` key is in `output.json` — so a manifest can never lie about a run.

**4.3 — The file set.** Each tool subcommand `<cmd>` writes a fixed set of files into `temp/<tool>/<cmd>/`, overwriting the previous run:

| File | Schema |
|------|--------|
| `config.json` | flat object of argparse args |
| `output.json` | flat object of metrics, command-specific field names |
| `manifest.json` | `{ headline_video?: str, headline_metrics: [str, …] }` |
| `output.log` | timestamped log lines |
| `run.sh` | executable script that re-invokes the tool with the same args |
| `<cmd>.csv`, … | the run's data — the numbers a figure would be drawn from |
| `<cmd>.mp4` | *rendering tools only* — the canonical video (`manifest.headline_video`) |

**4.4 — The runner reads, doesn't hardcode.** Subcommand name maps 1:1 to the directory under `temp/<tool>/`. The runner reads `manifest.json` to discover the headline metrics (and a headline video, if any) — it never hardcodes metric field names. It *renders* the figures itself from the tool's CSV data. It only chooses *which* commands an experiment bundles (`COMMANDS` in `expNNN.py`).

**4.5 — Import boundary.** A runner reaches a tool by *running its CLI* (subprocess), never by `import`ing it, and tools never import runner code. They communicate only through the files in §4.3 — which is what keeps tools generic and the contract language-neutral (§1.4).

**4.6 — `numbers.json` aggregation.** The runner aggregates each command's `config.json` + its headline metric fields into a single `numbers.json` in `artifacts/data/expNNN/`:

```json
{
  "lif": { "config": { "current": 2.5, "duration": 100.0, ... }, "firing_rate_hz": 90.0 },
  "net": { "config": { "n": 200, ... }, "mean_firing_rate_hz": 104.2 }
}
```

**4.7 — Provenance.** `setup_run_dir` stamps a `_provenance` block into `config.json` — the git commit SHA, a `dirty` flag (uncommitted changes at run time), and a UTC timestamp — which flows into the committed `numbers.json`. Every published result records exactly which code produced it; the publisher surfaces it as a page/PDF footer. Degrades gracefully outside a git repo (`commit: null`).

## 5. Publishing

**5.1 — Scratch vs record.** `temp/<tool>/<cmd>/` is scratch — gitignored, overwritten every run. The runner writes the rendered figure(s) + aggregated `numbers.json` (and any video) into **`artifacts/data/<id>/`**, which *is* committed. That folder is the publisher-neutral record: the single place the publisher reads from.

**5.2 — Typst is the publisher.** `task build` runs `demolab-engine/build/build.py`, which globs `writings/*.typ` (and each entry's mp4s) into a JSON manifest (`temp/bundle/index.json`), then compiles the committed, static `demolab-engine/build/main.typ` (which reads the manifest) to **three targets in one pass** — no generated Typst source:
- **Web** — `artifacts/site/`: `index.html` (entries grouped by collection, §6.5), `all.html` (every entry, newest first), and an HTML page per entry (figures inline, videos play, math as MathML, styled by `demolab-engine/build/style.css`).
- **Per-entry PDFs** — `artifacts/site/pdfs/<id>.pdf`.
- **Book** — `artifacts/site/pdfs/book.pdf`: every entry, with a table of contents.

**5.3 — What's committed.** PDFs are mirrored to the committed, shareable `artifacts/pdfs/`. `artifacts/site/` is a gitignored build output (CI regenerates + deploys it to Pages). CI does **not** run the experiments, so `artifacts/data/` **must** be committed — that record, not the ephemeral `temp/`, is what reaches the site.

**5.4 — Numbers can't drift.** Each `writings/<id>.typ` reads its own bundle natively — `json("/artifacts/data/<id>/numbers.json")`, `#image("/artifacts/data/<id>/fig.png")` (compiled with `--root` at the repo root) — so the numbers and figures come straight from the run.

## 6. Authoring writings

For *how a writing should read* — prose, math, figures, structure — see [`HOUSE-STYLE.md`](HOUSE-STYLE.md). This section is the mechanics.

**6.1 — `meta` + `body`.** A writing is `writings/<id>.typ`: a `#let meta = (title, date, description?, collection?, status?)` block and a `#let body = [ … ]` block. `build.py` discovers entries by those two top-level definitions. Model a new one on `exp000.typ`.

**6.2 — Use `lib.typ` helpers; never hand-type numbers.** Import with `#import "/demolab-engine/build/lib.typ": …`:
- `numbers-table(entry, title: "…")` — a parameter/metric table straight from a `numbers.json` command entry.
- `video("<file>.mp4", caption: […])` — plays as HTML `<video>`, omitted from the PDF. `build.py` auto-emits every mp4 as a bundle asset.
- `provenance-footer(run.<cmd>.config)` — the git-commit footer.

Numbers must come from the run (§5.4) — never hand-type a literal that could disagree with `numbers.json`.

**6.3 — Figures.** A data figure is a tool-rendered PNG staged by the runner — `#image("/artifacts/data/<id>/fig.png", width: 100%)`. A *drawing* (a schematic, not a simulation result) can be drawn directly in Typst — native graphics scale crisply, no image file. For something a reader should *explore*, point them at the Streamlit playground (`task playground`); in-browser interactivity is deliberately not part of the static site.

**6.4 — The `status` field.** Optional `meta` field for lifecycle — `draft → building → revising → final`, and back freely. It renders next to the date on the entry's page and in the book. Free-form; pick a convention and stick to it.

**6.5 — Collections.** Set `collection: <slug>` in an entry's `meta` to group it on the homepage; the slug title-cases by default (`neuron-models` → "Neuron models"). An optional `collections` map + `collection-order` list in the root `demolab.yaml` give each collection a `label` / `description` and set the display order — a collection with no registry entry still works. Decks are grouped under `slides`. Uncollected entries fall under `uncategorized`. `all.html` lists everything flat, newest first, tagged with each entry's collection.

## 7. Adding an experiment

**7.1 — Tool subcommand.** Add a subcommand (or reuse one) in the relevant `tools/<tool>/tool.py`. Pass a `manifest` to `write_output` declaring the headline metrics (and a video, for a rendering tool).

**7.2 — Runner.** Create `experiments/expNNN.py` modeled on an existing runner; declare `COMMANDS`; render the figure(s) from the CSV data into `artifacts/data/expNNN/`. Single-tool runners use bare strings (`COMMANDS = ("lif", "net")`); multi-tool runners use `(tool, command)` pairs (`COMMANDS = (("mujoco", "cartpole"),)`).

**7.3 — Writeup.** Create `writings/expNNN.typ` as a `meta` + `body` pair (§6.1); read the run with `json(...)`, embed figures with `#image(...)`, render tables with `#numbers-table(...)`.

**7.4 — Run + build.** `uv run python experiments/expNNN.py`, then `task build` (or the running `task dev`).

## 8. Adding a tool

**8.1 — One directory, the file set.** Each tool lives in its own directory under `tools/` and writes its run artifacts under `temp/<tool>/<cmd>/` (§4.3). A rendering tool also writes its video and declares `headline_video`. It does **not** write plots (§4.2).

**8.2 — Reuse the `setup_run_dir` / `write_output` pattern.** From an existing `tool.py`: `setup_run_dir(command, args)` creates the run dir, configures a logger to `output.log` + stdout, dumps `config.json`, writes the `run.sh` reproducer. `write_output(run_dir, metrics, manifest)` validates the manifest and writes `output.json` + `manifest.json` last. Subcommands are wired via `argparse` `set_defaults(func=...)`; `main()` calls `args.func(args)`.

**8.3 — `write_output` validation.** `headline_metrics` is required and validated against `output.json`; any declared `headline_video`/`headline_figure` must exist on disk. Data tools declare no asset (`{"headline_metrics": [...]}`); a rendering tool adds `headline_video`.

**8.4 — Every tool ships tests.** Put them in `tools/<tool>/test_<tool>.py` (run with `task test`). Unit-test the science functions directly — shapes, known properties, determinism (seeded) — and the manifest contract (`write_output` rejects a metric absent from `output.json`, or a missing figure). Keep tests off the filesystem where possible — call the sim functions, not the CLI.

**8.5 — The playground is exempt.** The Streamlit playground (`experiments/playground.py`) is an interactive demo, not an `exp*` runner: it authors no committed artifacts and is exempt from *producing* a manifest. But it still **runs the `neuron lif` CLI** on each slider change and reads back `temp/neuron/lif/lif.csv` + `output.json`, rather than reimplementing the simulation. **Don't duplicate the science** — reach the tool through its CLI.
