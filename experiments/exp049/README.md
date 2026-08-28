# exp049 — trainable recurrent PING loop

## Contract migration

Runner Guide 3.0.0 and Storage Guide 3.0.0. The four TR-05 conditions and
seeds 42–44 remain unchanged. Training belongs to exp022; exp049 never trains.
The legacy flat runner and bare package entrypoint now reject execution.

Each command is independent and requires an explicit completed v3 source:

```sh
uv run python -m experiments.exp049.compute --source <exp022-compute-run-id>
uv run python -m experiments.exp049.analyse --source <exp049-compute-run-id>
uv run python -m experiments.exp049.present --source <exp049-analysis-run-id>
```

`--run-id` accepts an explicit stage identity/reservation. New IDs end in
`-compute`, `-analyse` or `-present`, without an execution-location suffix.
`PINGLAB_SMOKE=1` reduces quantitative evaluation from 1,000 to 100 images;
it does not change the bank, condition/seed grid or snapshot selection.
The collection adapter reserves and dispatches the three stages explicitly.
Neither the stages nor the adapter publish into `.artifacts`.

- **Compute:** 12 official-test inference jobs, 12 recurrent-weight dumps and
  four snapshots, all from `weights_final.pth` at epoch 50. It pins the exp022
  bank and records the exact recipe, checkpoint hashes and simulator commands.
- **Analyse:** validates recordings and their configurations; reads the pinned
  bank's 50-epoch histories; saves all numerical results, histogram counts,
  trajectory aggregates and raster display coordinates.
- **Present:** draws only saved analysis, yielding 24 figure files and
  `numbers.json` in a flat export. Existing names and the `summary`,
  `rhythmicity`, `config` and checkpoint fields remain available to writings.

Inputs and their complete ancestry are validated before use and checked again
before completion. Runs are written under hidden temporary directories and
become visible only after successful completion. Historical v2 evidence is not
an operational input or fallback.

## Preserved measurements

Endpoint accuracy and E/I rates come from the final-checkpoint official-test
metrics. PSDs use demeaned, nonconstant E-population trials, full-trial Welch
density, `detrend=False`, then the mean PSD and maximum raw frequency bin in
5–150 Hz. There is no new peak interpolation or rhythmicity estimator.

Weight means retain the original array dtype. Zero fractions count entries
`<= 0`; positive means exclude zeros. Histograms pool three seeds before
binning positive entries into the original 49 bins.

Condition cards use the fixed reference-image diagnostic `rate_e`/`rate_i`
histories, not dataset-wide training rates. The trajectory figures
prefer the validation `test_rate_e`/`test_rate_i` keys (including a present null), retain
complete seed curves, and smooth the mean and min/max envelopes with the
original five-epoch edge-padded moving average. Phase trajectories are unsmoothed.
Rhythmicity retains the frozen final mean and trainable first-available/final
contrast summaries. A first-available value is not proof of an epoch-0 value.
The historical `acc` field is validation accuracy; `contrast` comes from the
fixed digit-0 Poisson realization. Rate–contrast trajectories therefore combine
different evaluation samples. These label corrections do not reinterpret or
rewrite the existing analysis schema or its immutable measurement metadata.

Raster selection uses RNG(42), sampling up to 200 E then 50 I cells without
replacement, from seed 42 / sample index 0. Analysis retains the recorded class.
Card spacing was enlarged to prevent panel-label overlap. For a degenerate
constant-trace seed, its zero PSD is placed on the other seeds' frequency grid;
this avoids the legacy mixed-shape plotting failure without changing any
archived result. Undefined endpoint frequencies remain null.

## Gold-2 audit and completed selective import

The 2026-08-28 audit verified the live R2 manifest/inventory against the cached
`r2:pinglab/campaigns/gold-2` records. Inventory SHA-256:
`c7b9455968e34ac3be2df46a57ab4fc0ffcd94dc88799bf11b36c9b673a88f68`.
All 181 exp049 state/derived files matched their inventory checksums.
The study state is **616,778,669 bytes across 152 files**.

The local `exp022-r001-compute` bank contains all required TR-05 final
checkpoints. All 12 checkpoint records agree with Gold-2; the 24 retained trained
E/I matrices also match those checkpoint tensors exactly. Replaying the
migrated calculations reproduced all 12 archived summaries, their full original
PSD arrays and the rhythmicity summary exactly. This was a read-only audit,
not an exp049 operational import or a new simulation.

The approved selective import retains:

| Retain | Purpose |
| --- | --- |
| 12 `metrics.json` files | Official-test accuracy and population rates |
| `dt`, `pop_e` from 12 population archives | Exact endpoint PSD replay |
| Initial and trained E→I / I→E arrays from 12 weight dumps | Weight summaries and pooled histograms |
| `dt`, population sizes, label and full E/I spikes from four snapshots | Reproducible raster selection |
| Configurations, commands, logs, archive inventory/lineage and verification records | Historical producer and import provenance |

The selected payload contains **187,314,147 uncompressed bytes**. A lossless
ZIP compression audit measured **43,366,316 bytes**, plus **120,850 bytes** of
per-job sidecars and small archive/provenance records. Preserve exact NPY entry
bytes: no dtype reduction, resampling or replacement simulation. Unused I
population traces, voltage/conductance recordings, input spikes and unrelated
weight matrices remain in R2. Training histories/checkpoints are reused through
the bank pin, not copied again.

exp092 and exp109 consume report summaries/rhythmicity and, where
applicable, `training_curves.svg`; these interfaces are retained. Migrating those
consumers' execution is separate work.

The authorized local import completed on 2026-08-28:

- `exp049-r001-compute`: imported recordings, **43,592,492 bytes** in export
  (including the recipe/bank evidence), **46,786,640 bytes** for the complete run.
- `exp049-r002-analyse`: measurements from the imported compute run and pinned bank.
- `exp049-r003-present`: 24 figure files plus `numbers.json` and its metadata projection.

The original producer was base-campaign Slurm job **33913526**, host **gpu-q-22**,
with an NVIDIA A100-SXM4-80GB. The compute manifest records this under
`historical_import.producer`, separately from `origin: local` and
`execution.operation: historical-import`. No simulation or training was launched.
All 96 selected NPY entries and 12 metric files match their originals exactly.
The new analysis/presentation reproduce the 12 archived summaries, checkpoint
records, rhythmicity summary and previously audited plot data. The 35 pre-existing
run manifests and the article were unchanged at that import checkpoint;
publication has not occurred. The subsequent article review is recorded below.

`import_gold2.py` has explicit `plan` and `import` actions. Planning requires
`--archive`, `--source` (the v3 exp022 bank) and a new `--plan` path; import requires
`--archive` and that approved `--plan`. It revalidates the plan, checksums,
producer lineage, configurations, checkpoint matrices and archived results
before allocating a run, then rechecks sources before atomic completion.
It never launches analyse/present or publishes. Re-importing is a new operation,
not an implicit resume or update of the completed import.

The executable plan, fresh R2 metadata check, import verification and independent
stage verification are retained under `.r2/exp049-import-f1_srzrv/`. The completed
compute run also retains the plan, original records, file/array mappings and a
human-readable history. Author acceptance and publication remain separate
human approval gates.

## Science and writing review — 2026-08-28

`writings/exp049.typ` now follows Writing Guide 9.0.0, with `updated_at` set to
2026-08-28 and its original creation date retained. Results contain five numbered
figure subsections; Methods are five scientific operations, with a numbered
contrast equation. The appendix retains initialization, diagnostic and
interpretation details. No Discussion section is needed.

The review checked retained results and the original producer implementation at
`4ad223d32620dd9f03698b89f28aedfe944d43ac`, including training, initialization and
the autocorrelation metric. Corrections include:

- The 7,000-image subset split was 6,300 training / 700 validation; endpoint
  evaluation used 1,000 official-test images, with final-epoch checkpoints.
- Optimization used AdamW with zero weight decay. Input sparsity was an initial
  zeroing operation with compensating scaling, not a fixed connectivity mask.
- Frozen test E/I means were 16.6/108.0 Hz. Only zero initialization left I silent;
  the other trainable conditions retained mean I rates of 5.5 and 8.7 Hz.
- Trainable test accuracies ranged from 84.4% to 91.1%; condition means differed.
  The evidence does not establish equivalent accuracy or measured energy savings.
- Contrast is a fixed reference-image diagnostic, not a test-population measure
  or a probability of PING. There is no logged epoch 0 in these trajectories.
  The observed gap does not establish attractors, basins, a separatrix, an instant
  transition, or universal failure to learn PING.

`exp049-r006-present` renders the existing `exp049-r002-analyse` results with
correct evaluation labels, unclipped rate axes and no inferred basin boundary.
Card spectra label the retained mean of seed-wise peak bins, not a guaranteed
gamma frequency. `exp049-r004-present` and `exp049-r005-present` are intermediate
review renderings; r006 incorporates the card annotation spacing fixes.
All remain immutable, as do all
38 runs that existed before this review. Numerical results and plotted data are
unchanged; no compute or analyse stage was rerun. No presentation was published.

The article is `Ready for review`, not marked `Reviewed`. Review evidence,
original producer source extracts, numerical/coordinate checks and the explicit
four-page article preview are under `.r2/exp049-review-ll4vk3du/`.

## Verification

Synthetic tests cover all three stages without real simulations, explicit
lineage, final-checkpoint selection, corruption and v2 rejection, atomic failure,
collection reservations/resumption, missing-data semantics, raster RNG selection,
and rendering the Typst article from an explicitly selected present
run. Numerical and figure-coordinate replay against retained Gold-2 evidence is
recorded locally under `.r2/exp049-contract-25a868hu/`; these audit artifacts are
not operational runs or publication inputs. The 520-test broader regression suite
passed, followed by an 82-test focused pass after final refinements. Ruff and
`ty check experiments/exp049` also pass. Plotted line, scatter and segment
coordinates agree with the original runner across all 12 figure families.

The import extension passed all 29 exp049 tests and the broader 532-test suite
(20 existing plotting-layout warnings from other studies). The actual imported
run passed complete v3 layout/checksum validation, exact NPY/metric comparisons
and end-to-end result checks. A five-page Typst preview was compiled against
`exp049-r003-present`; the writing itself was not revised or published.

The science review adds a regression check for evaluation labels, unclipped rate
envelopes and preserved trajectory coordinates without an inferred basin divider.
All 12 figure families were also compared against the pre-review renderer using
retained production analysis: lines, scatter points, histogram/raster coordinates
and segment colours are unchanged, apart from removing the unsupported 17 Hz
reference line. The complete numerical export is unchanged apart from execution
metadata identifying the new presentation.

The review passed 533 broader regression tests (20 existing layout warnings from
other studies), followed by 120 focused tests after the final label adjustment.
Ruff, the exp049 type check and `git diff --check` pass. All 41 completed runs pass
v3 layout and full payload-checksum validation; all 38 pre-review manifests are
unchanged. The final four-page preview was compiled against `exp049-r006-present`
and visually inspected, including all four condition cards.
