# Presentation inputs and run links

This is Pinglab's presentation layer, not a Pingstore catalogue or management CLI.
Stored runs remain immutable. No experiment is executed or materialized.

Generate its JSON with the lab-owned presentation hook:

```sh
uv run python -m writings.prepare
```

The hook uses this checkout's declared experiment graph and writes only the
generated `.demolab/pinglab-inputs.json`, reporting validation results on stderr.
Demolab invokes it using:

```yaml
prepare: [uv, run, python, -m, writings.prepare]
```

`DEMOLAB_INPUTS` supplies the JSON object of URL input values; `DEMOLAB_ARTICLE`
identifies the requested article. Both are optional for ordinary builds.

While the Demolab change is unreleased, run from the Pinglab checkout:

```sh
PYTHONPATH=../demolab uv run --no-sync demolab dev 3010
```

Open an article to see a bordered Dataset table: run name, readable creation date and
time (timezone label omitted), duration, export size (decimal bytes/KB/MB/GB/TB) and execution origin (`slurm`,
`modal`, `runpod`, `local`, `mixed` or `unknown`). A single Upstream/Downstream
block above the run table links the current page's experiment dependencies;
it does not repeat the experiment title or add columns to each run.
Compute and analyse run names are
non-clickable; compute names are underlined. Only eligible present runs have selection links.
Each run link opens a new tab with explicit URL parameters. The selected paths
apply to the entire article, including figures and numerical prose outside the panel.
Every parameterized request gets independent output; it does not change another
tab, a default selection, or stored data. Reopen the URL to get a fresh rendering.

Every experiment article includes the table. Data-backed reports place `run-view`
explicitly after the complete abstract, preserving surrounding Typst style rules;
`with-datasets(..., placed: inputs-ready(...))` supplies the empty-report fallback
without duplicating an already-placed table. Pages without an abstract retain any
opening introduction before the table; reference pages starting with a heading
show it before that heading. Unavailable-data notices remain before the table.
Articles without declared inputs show an empty table with “No datasets declared.”
The table remains web-only; article text and authored dates are unchanged.

The local source override avoids changing Pinglab's dependency lock or publishing
an engine release. Once an engine containing this feature is installed, ordinary
`uv run demolab dev` is sufficient. An already-running older server must be
restarted with the source override; a page reload cannot upgrade its Python code.

## What to edit

- `writings/run-view.typ`: the complete web-only panel, labels, layout, links,
  run links and displayed metadata. The technical panel is separate from the
  scientific article; PDFs do not include it.
- `writings/prepare.py`: the lab-owned hook that supplies the collection graph,
  plus direct inputs not represented by that scheduling graph (exp048's bank
  and exp046's direct bank input). It imports declarations only, never runners.
- `tools/pingstore/presentation_inputs.py`: the JSON projection called by the hook
  before each build/render. It merges the supplied computation dependencies with
  article input declarations and validates v3
  payloads and upstream identity pins, then atomically writes
  `.demolab/pinglab-inputs.json`. It never modifies Pingstore.
- `writings/run-inputs.typ`: directory resolution, defaults and media attachments.
  Existing figures still call the same article-scoped `data-file()` helper.
- `writings/run-defaults.json`: optional explicit article/input defaults. An empty
  object retains Pinglab's prior Latest behavior, now implemented in user code.
- `demolab.yaml`: allowed URL inputs and the preparation command. The inactive
  `legacy_preview` block is retained for the later retirement/rollback decision.

For example, a committed default can be expressed as:

```json
{"exp022": {"exp022": "exp022-r003-present"}}
```

Only pin a run that exists locally and whose scientific contents you intend to
show. Missing pins fail rather than silently selecting another run. URL choices
override defaults only for their article. Multi-input links carry all currently
available selected inputs into the new tab. Missing inputs remain empty until
data is available; dependent reports retain their existing readiness checks.

## Metadata meanings

- **Collection**: authoritative `run.json.collection`, not the article's category.
- **Views**: current membership in optional `.pingstore/collections.json`.
- **Duration**: `execution.completed_at − execution.started_at` from authoritative
  `run.json`, projected as `duration_seconds` for every stage. This measures only
  the recorded operation, not upstream runs or total campaign compute time.
  Import operations are explicitly labelled `(import)` and exclude the original
  training/simulation. Whole-second timestamps with equal endpoints display
  `<1 s`; missing endpoints display a dash. Invalid, timezone-free or reversed
  timestamp pairs fail preparation. Hover for the exact recorded difference.
- **Retained scientific timing**: when `run.json.scientific_execution.record`
  explicitly identifies compact historical timing in this validated v4 run's `run.json`,
  the table instead shows that execution's start-to-finish span, labelled
  `(HPC span)` for Slurm. The tooltip distinguishes this elapsed span (including
  gaps) from summed job-hours of retained completed cell attempts, and keeps the
  import duration separate. The job total includes inherited and retrained cells
  retained in the bank, not unretained attempts or all historical campaign work.
  Creation date and origin still describe the recorded storage operation.
  No historical inputs are traversed, no legacy run becomes operational, and no
  stored manifest is rewritten. Evidence bytes remain covered by the enclosing
  v3 payload checksum; unsafe references or invalid/duplicate attempts fail.
- **Export** / **Files**: bytes and regular-file count in this present run's export.
- **Display-only stages**: compute/analyse rows sum regular files recursively in
  the validated scientific export directory, including an explicit `export_root`.
  `display_runs` is separate from selectable `runs`; discovery, URL inputs and
  default pins still accept only eligible present runs. Hover over a size for
  the exact byte count, or a date for the full recorded instant normalized to UTC.
- **Run payload**: all immutable bytes under `export/`, excluding root
  `run.json`.
- **Upstream / Downstream**: direct declared experiment dependencies and their
  reverse, including article inputs for comparisons and syntheses. These belong
  to the current page, independent of stored runs or selected datasets. IDs are
  sorted, deduplicated, linked to articles, and exclude the experiment itself.
  A dash means no declared dependency in that direction. The block remains
  visible without local data, and links remain clickable in static builds.
- **Upstream payload**: payload bytes of all transitively referenced runs, counting
  each identity once. This is not a separate download size or a count of samples.

The preparation command preserves prior generated JSON on failure, but the engine
aborts that build/render rather than consuming stale data. It validates all visible
runs, as the previous adapter did, so large stores can make each request slower.
No unvalidated fast path is introduced.

The lower-level `pingstore presentation-inputs` command remains available for
standalone projection, but includes only article-declared dependencies unless a
caller supplies computation declarations to the Python API. Use the lab hook
above for Pinglab pages so the collection dependencies are included.

Static builds show the selected present runs and display-only compute/analyse
rows with the same compact metadata, without selection links.
Parameterized links require the local rendering server. User-declared `meta.assets`
keeps selected videos self-contained in both ordinary and URL-rendered output.

The old engine implementation and compatibility branches are not retired yet.
