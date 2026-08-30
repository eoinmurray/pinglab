# exp025 — accuracy–rate frontier

Contract migration: runner 4.0.0 and Pingstore `pingstore.run/v4`.
The migration retained historical measurements, model choices and plotting
conventions. The author-approved article corrections are recorded below;
stored figures and runs remain unchanged.

## Independent stages

```sh
uv run python -m experiments.exp025.compute --source <exp022-compute-run>
uv run python -m experiments.exp025.analyse --source <exp025-compute-run>
uv run python -m experiments.exp025.present --source <exp025-analyse-run>
```

Each command completes one immutable run. Analyse does not simulate; present
only renders saved analysis. No stage selects a latest run, falls back to old
artifacts, launches upstream work or publishes. The collection explicitly
orchestrates all three stages with reserved IDs and pinned references.
Legacy flat entrypoints and monolithic campaign completion are rejected.

The bank input is an explicitly selected v4 exp022 compute run. All 36 TR-02
cells and 12 TR-07 cells are validated against their final-epoch checkpoint
hashes and training histories. Recursive lineage is validated before use and
checked again before completion.

Production compute executes 98 inference jobs: 36 frontier evaluations,
12 representative-seed PFG evaluations, 48 input-weight-scale evaluations,
and two digit-0/sample-0, 400 ms raster snapshots. Quantitative evaluations
use all 1,000 selected official MNIST test images. The PFG calculation retains
all those trials, despite the old `PFG_MAX_TRIALS=256` constant's misleading
name: the original cap was `min(256 * 5, 1000)`.

`PINGLAB_SMOKE=1` uses 100 evaluation images, three scale factors, and seed 42
for the low-input-weight history summary (56 inference jobs). It still requires
the full trained bank; smoke does not silently retrain or substitute checkpoints.

## Scientific boundaries

- Compute retains raw metrics, PING population traces and full sparse E/I
  spikes, scale-sweep sample-wise E rates, and full E/I illustrative snapshots.
- Analyse owns frontier means/SEMs, Welch frequency, inhibitory-cycle
  participation, float32 sample-wise rate penalties, history summaries,
  scale-crossing midpoint and raster coordinates.
- Present consumes those saved results and emits the existing fourteen figure
  files plus `numbers.json`, preserving downstream article input names.
- COBA PFG uses accuracy and rate metrics only; its undefined frequency and
  participation remain `null`. Unused COBA arrays are not imported.
- The exp092/093 consumers use presentation figures/numbers, not the omitted
  raw members. The staged input migration of downstream consumers is separate.

The contract migration itself did not change the science. The retained bank
uses gradient damping 1 for COBA and 1000 for PING; the revised article now
states this limitation. Historical plotting labels and inference choices
have not been rewritten.

## Selective Gold-2 import

The completed import was an explicit, offline historical operation. Its one-off
entrypoint has been retired; the executed code and plan remain in run provenance.
It did not download,
train, simulate, analyse, present or publish. Planning verifies archive identity,
checksums, the 48 checkpoint hashes and the selected configurations. Import
rebuilds the plan, validates exact equality, copies selected NPY bytes with
lossless ZIP compression, validates recordings, and checks the source again
before completion. Checkpoints are pinned, not duplicated.

Gold-2 selects the `ggs-fr-repair-20260820-ac6f4988` branch for exp025. Its
Slurm log identifies job 34111989. Both raster snapshots reside in
`state/checkpoints/current-repair-exp022/cells/*__off__seed42/infer/` rather
than the exp025 state directory. Their configurations select final weights,
digit 0/sample 0 and 400 ms. These directories are read only.

Retained command paths name a `73f0883e` checkout, while the campaign and
per-job configs declare `ac6f4988`. Both records are preserved; no executed
revision is inferred from a checkout directory name. The operational run's
origin is local and operation is historical-import; its HPC producer is
recorded separately under `historical_import`.

Original metadata, training records, simulator configs/scripts/logs, the import
plan and a per-file/per-NPY checksum mapping are retained in `provenance/`.
Unused array members, duplicated weights and old plots stay in Gold-2.
All selected trials and E/I spikes are retained without subsampling.

## Completed import and verification (2026-08-28)

- `exp025-r001-compute`: local historical import, pinned to `exp022-r001-compute`.
- `exp025-r002-analyse`: separate analysis of that evidence and bank.
- `exp025-r003-present`: fourteen figure files plus `numbers.json`.

Live R2 metadata matched the cache exactly. All 653 selected source files,
627,674,150 bytes before selection/compression, passed inventory checks before
and after import. The compute export is 33,907,869 bytes; provenance is
11,796,774 bytes. All three completed runs occupy 48,697,000 bytes. These are
file-byte totals, not filesystem allocated sizes. All 132 selected NPY members
retain their exact bytes and dtypes. No simulation, training, remote write or
full-archive download was performed.

The original scientific functions were replayed read-only on the imported
arrays. Every scientific value and training-source record matches the new
analysis exactly; checkpoint lists are compared by cell identity because their
ordering changed. Against the archived Gold-2 summary, all accuracies, firing
rates, participation fractions, losses, penalties and seed summaries match
exactly. Only six Welch gamma-frequency values differ, by at most
0.000003558 Hz. The same differences arise from the original code on this
runtime; this is not evidence that the stage refactor or selective import
changed the measurement. The historical numerical cause remains unresolved.

The expanded storage/experiment suite passed 438 tests, with 20 existing
exp041/exp044 Matplotlib layout warnings. The final scoped suite passed 65
tests, including corrupt-but-resigned payload rejection and normalization of
current simulator metrics: compute retains original metrics
and adds seed/decay only from a validated companion configuration. No simulation
numbers are filled from historical outputs or substituted to match the archive.

All 27 pre-existing runs and their authoritative manifests validate unchanged.
Discovery exposes only the new present stage. The unchanged article compiles
against that explicit presentation into five preview pages; all five article
figures were inspected. No files were materialized into `.artifacts/exp025`.
The audit, original runner, comparison, logs and preview are retained under
`.r2/exp025-contract-ksvo4kux/`; original import evidence is also inside the
immutable compute run's provenance.

### Review items identified during migration

The article and scientific claims are deliberately not marked finished:
its PFG caption says 256 trials while the retained computation uses 1,000;
low-input-weight validation histories are labelled as test measurements;
matched-recipe wording omits the gradient-damping difference; and the
order-of-magnitude claim needs reconciliation with the retained frontier.
These are review items, not changes silently applied during migration.
The original plot labels and axis conventions remain. The article's structure
and `updated_at` have not been changed in this migration. Publication, commit
and push have not been performed.

Metadata follow-up: the article status is now `Results available`, reflecting
the verified retained evidence (Writing Guide §3.5). Its scientific text remains
unchanged. At the author's explicit request, `updated_at` is `2026-08-28`;
the original creation date is preserved.

### Author-approved writing corrections (2026-08-28)

The subsequent writing pass follows Writing Guide 9.0.0, with Results before
Methods, concise captions, numbered scientific operations and equations,
supporting appendices, and a cited Welch reference. It supersedes the earlier
notes that scientific text was unchanged. The original creation date and
`updated_at: 2026-08-28` are retained.

The article now reports the 6.86-fold unpenalised baseline rate ratio (computed
from unrounded means), explains the gradient-damping confound, and withdraws
unsupported structural-floor, constant-participation and isolated gamma-benefit
claims. It corrects PFG coverage to 1,000 test images, identifies the low-input
histories as validation measurements, and describes the actual output-membrane
readout, trainable initial zeros, and final-epoch checkpoint selection. The
participation-frequency product is an approximation, and the input-scale
crossing is an empirical midpoint, not a fitted bifurcation.

All five figure inputs and all four pinned lineage runs are unchanged. The
captions explain two retained figure-label problems: “Test accuracy” on the
validation-history panel and `f*` for an input-scale crossing. Correcting those
labels requires a separate presentation revision; the article remains at
`Results available`, pending that work and author review. No simulations,
publication, commit or push were performed during this writing pass.

Verification: all six rendered article pages were inspected; unselected inputs
show the unavailable-data notice, while missing selected files and corrupt JSON
fail compilation. The existing writing-input and writing-status suites passed
89 tests, and `git diff --check` passed. The writing audit is retained under
`.r2/exp025-writing-approved/`.

## Future-run data retention — 2026-08-28

Snapshot jobs record E/I spikes only. Scale jobs accumulate only per-sample E rates. PING frequency jobs retain E population traces and E/I events; unused I traces, other rate arrays and output events are not emitted. Native compute no longer writes full NPZs and then extracts selected fields.

These changes affect future execution only. Existing immutable runs and R2
archives are unchanged. Required arrays keep their original numerical values;
selected NPZ outputs use lossless compression. No production rerun or new
publication was performed for this cleanup.
