# exp048 — streaming duration and input rate

## Contract migration — 2026-08-28

This migration follows Experiment Runner Guide 3.0.0 and Storage Guide 3.0.0.
It changes execution contracts, not the scientific recipe. The retired flat
runner now exits with guidance; importing the package performs no experiment.

Each stage requires an explicitly selected completed v3 run:

```sh
uv run python -m experiments.exp048.compute --source <exp022-compute-run-id>
uv run python -m experiments.exp048.analyse --source <exp048-compute-run-id>
uv run python -m experiments.exp048.present --source <exp048-analysis-run-id>
```

The equivalent `python experiments/exp048/<stage>.py` entrypoints work too.
There is no latest-run selection, training, automatic upstream work, cache
fallback, materialization or publication. `--run-id` accepts only an unused
source-neutral reservation created through the shared Pingstore allocator.
No independent run-number guessing or experiment-specific allocator is used.
Local and HPC execution use the same contract. This experiment does not add
a scheduler adapter or join the collection execution graph.

### Stage responsibilities

- **Compute:** validate the three canonical PING cells and their
  `best_validation` checkpoints; construct the original deterministic stimuli;
  invoke the current shared simulator CLI; retain full sparse E/I/output spike
  recordings, exact stimulus pixels and labels, lossless input coordinates,
  input-array hashes, random seeds and the extracted trained readout matrices.
  It does not decode classifications or aggregate results.
- **Analyse:** reconstruct input bytes to check their hashes, validate recording
  dimensions and coordinates, replay the original output integrator and matched
  sliding mean, retain per-stream segment predictions, and compute the original
  per-seed rows and across-seed summaries. It reads no live dataset and launches
  no simulator. The two illustrative streams retain their complete arrays for
  independent rendering.
- **Present:** validate the saved grids and illustrative arrays, then draw all
  four original figure families. Its flat export contains eight figure files
  and `numbers.json`, plus shared presentation bookkeeping. Missing analysis
  never triggers computation. Existing figure names and numerical interfaces
  remain available to exp092 and exp109.

Bank payloads are referenced, not copied. Extracted output matrices are small
compute products, not duplicate checkpoint banks. Compute retains all input and
recorded output spikes without subsampling; sparse input coordinates reconstruct
the exact float32 simulator input. Native raster arrays are compressed without
changing their values or dtypes. Scratch dense inputs and unused weight-dump
arrays are not retained. Original simulator configurations, commands and logs
belong in `provenance/`; shared stage helpers capture execution, code and lockfile
provenance. This retention policy describes future computation, not an approved
historical import.

All stage inputs and their complete pinned ancestry are validated before use
and rechecked before completion. Payload or manifest changes, v2 inputs, wrong
stages, invalid checkpoint roles and incomplete recordings fail closed.
The shared stage helper reserves hidden directories and atomically exposes a
run only after validation. Failed runs remain hidden; rerunning uses a fresh
identity, and resuming an interrupted reservation is rejected.

## Preserved scientific recipe

The baseline has 1,024 E cells, 256 I cells, 784 input channels and ten classes.
It uses 0.1 ms timesteps, 200 ms training presentations, 25 Hz full-intensity
pixel encoding and training seeds 42–44. Selected `weights.pth` files have the
registered `best_validation` role; they are not replaced with final checkpoints.
The checkpoint helper verifies each role, epoch and hash.

The unchanged production recipe contains **197 jobs, 6,512 streams and 65,110
segments**:

| Component | Conditions and sampling |
| --- | --- |
| Fixed illustrative stream | Five digits, 50 ms each, seed 42 |
| Varying illustrative stream | (200 ms, 10 Hz), (50, 100), (100, 25), (25, 200), (75, 15) |
| Duration sweep | Four durations × two rate protocols × three seeds; 20 streams of ten digits per cell |
| Duration/rate grid | Eight durations × six rates × three seeds; 40 streams of ten digits per cell |
| Low-rate extension | Nine rates × three seeds; ten streams of ten digits per cell, fixed 200 ms windows |

The duration sweep retains 25, 50, 100 and 200 ms. The grid retains 10, 15, 25,
40, 50, 75, 100 and 200 ms crossed with 5, 10, 25, 50, 100 and 200 Hz. The
low-rate extension retains 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2 and 3 Hz.
Rate compensation remains `25 * (200 / duration)` Hz.

RNG initialization, loop order and reset boundaries are preserved, including
paired constant/compensated streams, sequential grid sample selection and
per-rate resets for the low-rate extension. The output integrator retains its
one-step E-spike delay and configured time constant, defaulting to 2 ms.
The fixed headline selects softmax argmax; the varying headline and quantitative
sweeps select logits argmax. Startup windows divide by their available length.
Readout windows match each segment's duration; they are not independently varied.

Grid and duration aggregation retain float32 seed arrays and sample SEM
(`ddof=1`). The low-rate extension retains its original float64 aggregation;
the psychometric curve combines those points with the grid's 200 ms row.
No production smoke profile or arbitrary recipe overrides were introduced.
Test fixtures shorten streams only within isolated test processes.

## Historical evidence — non-Gold-2 summary import

The cached Gold-2 inventory has **zero exp048 entries**, including state,
derived outputs and experiment-specific producer records. No operational local
exp048 run was found during the initial audit. The archive's base and repaired shared exp022 banks contain
1,133 files / 2,258,472,251 bytes and 1,162 files / 2,260,887,263 bytes respectively;
these are upstream bank totals, not exp048's footprint.

The existing `exp022-r001-compute` bank's three selected checkpoint hashes match
both archived banks, and its selected cells pass the new training contract.
This establishes compatibility only. It does not establish which models,
simulator revision or dataset implementation produced historical exp048 results.
There is no evidence here establishing an HPC producer; one must not be invented.

On 2026-08-28, read-only `rclone cat` checks of live
`r2:pinglab/campaigns/gold-2/{run.json,inventory.json,lineage.json}` matched the
cached files byte for byte. The live inventory also contains zero exp048 entries.
The verified inventory SHA-256 is
`c7b9455968e34ac3be2df46a57ab4fc0ffcd94dc88799bf11b36c9b673a88f68`.

The subsequent whole-store search found the approved historical source at
`r2:pinglab/datasets/gamma-gated-sparsity/baseline-20260826/experiment-runs/exp048/exp048--r003/`.
It contains 13 payload files (734,494 bytes), plus the archive run record and
inventory: **15 files / 738,066 bytes**. All original files are retained without
subsampling. These include 195 per-seed rows (24 duration, 144 grid, 27 low-rate),
aggregates, eight figures and execution evidence. No raw streams were found.
The archive's July 24 local manifest predates Gold-2's August campaign; August 26
is the inventory-generation date. Gold-2 deliberately excluded exp048 in favor
of exp082. Fifteen identical HPC checkout copies and one alternate-rendering
copy add no independent numerical evidence and are not imported.

Anomancer approved a summary-level historical import and independent rebuild on
2026-08-28. The explicit entrypoint is:

```sh
uv run python -m experiments.exp048.import_historical --source r2:pinglab/datasets/gamma-gated-sparsity/baseline-20260826/experiment-runs/exp048/exp048--r003
uv run python -m experiments.exp048.analyse --source <historical-import-analyse-id>
uv run python -m experiments.exp048.present --source <reaggregated-analyse-id>
```

The importer pins both original metadata hashes, verifies every payload checksum
against the approved inventory, and creates an `analyse` run with operation
`historical-import`. Its export contains unchanged numbers; `provenance/archive/`
preserves all 15 source files. No compute run or upstream checkpoint pin is
invented. Original local execution, empty command record, missing completion
timestamp, r003/r001 discrepancy, and the low-rate attribution to "exp065 initial
computation" remain explicit. `historical.gold_2` is false in every new manifest
and in derived numerical metadata. Existing Gold-2 banks are neither copied nor
asserted as ancestors: checkpoint compatibility does not establish identity.

The separate analysis stage validates the historical recipe and seed/count
coverage, then rebuilds all means and sample SEMs with the original float32
duration/grid and float64 low-rate arithmetic. It compares every saved field
against the archive at relative/absolute tolerance 1e-12. No segment predictions
are regenerated. Presentation reads the validated analysis and its pinned
historical ancestor; it renders the quantitative figures and carries both raster
families unchanged, with per-file source hashes and explicit carry-forward
metadata. Historical images are retained as evidence outside the import export,
not mislabeled as newly generated analysis products. New exports remain flat.

This is a distinct, validated historical-summary path, not a fallback for missing
native streams. Native compute/analyse/present requirements remain unchanged.
No raw replay, new simulation, training, archive mutation, materialization or
publication is authorized by these commands. Source-neutral IDs are reserved
through the shared helpers; completed runs are immutable. Failed work remains
hidden and requires a fresh reservation.

## Science, writing and shared-file boundaries

The collection registry tags exp048, but its execution plan deliberately excludes
it in favor of exp082. That exclusion is preserved and tested. exp092 and exp109
consume its presentation outputs; exp082 is a scientific successor, not an
operational consumer of exp048 data. No shared collection, simulator or test
files were edited. Concurrent changes to those files remain owned elsewhere.

The article's title is prefixed `DEPRECATED` at the author's request. Its creation
date remains `2026-06-08`; substantive clarification sets `updated_at` to
`2026-08-28`. Results precede Methods, with numbered result headings and concise
captions absorbing the former concluding discussion. Equations are preserved;
the startup window denominator now states the implemented available-length rule.
Known segment timing and uncalibrated softmax evidence are explicit. No `Reviewed`
status is assigned; unresolved producer lineage prevents a full reproduction claim.
The old abstract's universal failure floor below 15 ms was unsupported: the saved
10 ms grid spans about 17–63% accuracy. It now describes rate-dependent losses.

Existing scientific issues to review, not silently redesign:

- The variable-duration decoder uses known segment durations and endpoint times.
  The network receives continuous input, but that is not a fully blind timing
  decoder. Cross-article claims of no segmentation cue need this distinction.
- The duration plot marks approximately 28 ms as one gamma cycle, without a
  separately retained frequency measurement here. The current compatible bank
  records 6 ms GABA decay, whereas the old runner comment referenced 9 ms.
- The original plot labels raster sample ranks using full-population endpoint
  labels. Historical images remain unchanged; the caption explains those labels.
  This is not a claim that the displayed sample contains the full population.

## Verification

The final verification run passed **131 tests**,
covering exp048, shared Pingstore behavior and the collection graph/plan exclusion.
Dedicated probes cover stage isolation, source-neutral reservations, checkpoint
roles, full ancestry changes, corrupt and re-signed malformed evidence, exact
input reconstruction, incomplete analysis, atomic failures, parser compatibility,
decoder timing, variable windows, RNG pairing, aggregation precision, historical
summary isolation, archive checksums, provenance flags and plot/caption regressions.
The simulator backend is synthetic in stage tests; the real CLI parser is checked
without launching inference. Ruff and the exp048 type check pass.

At initial extraction, all **13 stimulus, decoder, aggregation and plotting
functions** had identical Python ASTs to the pre-migration source. The subsequent
visual review changed only the rate-curve legend's opaque background, preventing
the chance line from crossing its text without obscuring the inset label. A separate
replay of the original orchestration on the same short synthetic streams matched
all per-seed rows, aggregates, the complete psychometric curve and both complete
illustrative-stream arrays exactly. This is code-preservation evidence, not a
historical result comparison. Synthetic test runs live in temporary directories,
not this checkout's Pingstore.

The authorized historical import and independent reaggregation/presentation are
complete. All 195 saved per-seed rows are retained; aggregate replay matches the
archive at 1e-12 tolerance. Four carried raster files have identical source hashes.
The final article was checked in browser HTML and all five PDF pages: both figures
load, ranges and labels agree with the captions, and all eight equations render.
See [the import report](IMPORT_REPORT.md) for exact run lineage and retained bytes.

No production simulation, training, archive mutation, publication, materialization,
commit or push was performed by this task. Real simulator execution, raw decoder
replay and exact historical checkpoint identity remain unverified.
