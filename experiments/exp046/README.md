# exp046 — spikes per excitatory cell per gamma cycle

Execution target: **Experiment Runner Guide 4.3.0 / Storage Guide 4.3.0**,
using `pingstore.run/v4` and source-neutral stage IDs. This migration preserves
the numerical definitions. The author deferred scientific and article review
until after migration and historical-data handling; the article is unchanged
and remains `Implemented`. Writing Guide 9.0.0 review is **not complete**.

## Independent stages

```sh
uv run python experiments/exp046/compute.py --source exp022-r001-compute
uv run python experiments/exp046/analyse.py --source <exp046-compute-id> --frequency-source exp041-r002-analyse
uv run python experiments/exp046/present.py --source <exp046-analyse-id>
```

These are commands for later explicit execution, not actions performed during
the migration. Compute runs inference, not training. Analyse requires a selected
exp041 analysis using exactly the same exp022 bank, final-epoch checkpoints and
evaluation profile. Present renders saved results; it never counts cycles,
fits a model or starts upstream work. None of these stages publishes outputs.

Every stage records exact upstream manifest and payload checksums. Ancestors
are validated recursively before consumption and checked again before completion.
Outputs become visible atomically only when complete. Failed work stays hidden;
retry with a fresh identity. `--run-id` accepts an unused reservation, including
one allocated before Slurm submission. Execution origin remains in `run.json`,
not in the run ID. Existing completed runs are not rewritten.

The retired combined runner and package entry point fail before creating outputs.
Importing the package exposes recipe definitions without resolving storage paths.
New collection plans reserve and dispatch the three stages explicitly and pin
the completed exp041 analysis. Legacy campaign plans cannot resume exp046.
There is no implicit latest selection, historical fallback, upstream execution
inside a stage, or automatic materialization.

## Preserved measurements and outputs

- Use the 18 TR-03 final-epoch checkpoints from the selected exp022 bank:
  inhibitory decay 4.5, 6, 9, 12, 18 and 27 ms, each with seeds 42–44.
  Preserve the 0.1 ms timestep, 200 ms trial and fixed 1,000-image official
  MNIST test subset. The smoke profile uses 100 images.
- Compute retains each network's full-trial sparse E/I spikes, per-cell E
  rates and metrics. It references checkpoints rather than copying them.
  Commands, configurations and execution logs are retained in provenance.
- Analyse detects inhibitory population peaks with the existing 1 ms Gaussian,
  5% height threshold and half-period minimum distance derived from the selected
  exp041 frequency. Integer midpoint boundaries and trial-edge intervals remain
  unchanged; trials without detected peaks are skipped, as before.
- Count every E cell in every detected interval, using buckets 0, 1, 2 and >=3.
  Pool cell-cycle counts across seeds and conditions, rather than averaging
  percentages. Preserve the through-origin fit of all 18 network maximum rates,
  centred R squared and the existing denominator floors.
- Reject incomplete grids, mismatched bank/profile evidence, invalid indices,
  duplicate sparse spikes, nonfinite values and disagreement between retained
  per-cell rates and full spike counts.

Analysis exports `results.json`. Presentation exports `numbers.json`,
`spikes_per_cycle_distribution.{svg,pdf}` and `ceiling_vs_fgamma.{svg,pdf}`.
Existing `results`, `global_fracs`, `per_tau`, `ceiling` and
`n_cell_cycle_pairs` fields remain available. This preserves the outputs used
by exp092 and exp109; their own execution migrations are separate work.
Repository run IDs are no longer stamped onto scientific figures.

## Deferred scientific review

Do not interpret contract conformance as validation of the existing claims.
The unchanged article contains hardcoded counts, proportions and fit values
that differ from the previously audited Gold-2 summary. The retained 0.20
reference line and its exp041 attribution also require an author decision;
presentation provenance records that this review is deferred. Its axis limits,
the one-spike ceiling interpretation, pooling weights and zero-peak exclusions
must be considered in that final review. No scientific claim was silently fixed
or discarded during this execution migration.

## Verification and next gate

The targeted regression suite passed **465 tests** in 52.87 seconds, with 20
existing Matplotlib layout warnings from exp041/exp044. Scoped lint and
`git diff --check` passed. All 24 existing operational runs were revalidated
unchanged, as were all 18 selected bank checkpoints and the article bytes.

Synthetic tests cover independent stages, explicit frequency/checkpoint lineage,
v2 rejection, malformed recordings, changed ancestors, incomplete runs, collection
reservation/resume and unchanged-article input compatibility. No production
inference or training was run.

An offline comparison with the saved pre-migration runner gives exact equality
for all 18 synthetic measurement records, per-condition and global fractions,
cell-cycle totals and fit outputs. The detector and cycle-counter syntax trees
are identical. Synthetic plot previews were inspected; the scatter layout
preview uses illustrative rates and is explicitly labelled as non-scientific.

Local source snapshots, comparison script/results and test logs are retained
under `.r2/exp046-contract-gnt65xky/`. That code migration imported no Gold-2
data, created no operational exp046 run, and did not update `.artifacts`,
publish, commit or push. The separately authorized import is recorded below.
Scientific and article review remains last.

## Completed selective Gold-2 import

On 2026-08-27 the author authorized selective import and independent analysis
and presentation, while retaining the deferred scientific-review boundary.

| Stage | Run | Operation |
| --- | --- | --- |
| Compute | `exp046-r001-compute` | Local historical import |
| Analyse | `exp046-r002-analyse` | Count cycles and fit retained measurements |
| Present | `exp046-r003-present` | Render saved analysis |

All three have `origin: local` and source-neutral IDs. The compute run pins
`exp022-r001-compute`; analysis also pins `exp041-r002-analyse`. The original
producer is Slurm job **33913630**, campaign
`ggs-production-20260818-4ad223d3`, recorded separately from the import.
Its retained job log identifies host `gpu-q-24` and an NVIDIA A100-SXM4-80GB.

Live R2 `run.json`, `inventory.json` and `lineage.json` matched the cache byte
for byte. All **135 selected source files**, totalling **1,347,364,524 bytes**,
passed inventory checks before and after import. No full archive download,
remote write, production simulation or training was needed.

The compute export is **101,488,250 bytes**, with **3,025,642 bytes** of
provenance. All three new runs together occupy **105,861,899 bytes**. This
reduction comes from lossless ZIP compression and omission of unused arrays,
not trial or neuron subsampling. All **216 selected NPY entries** retain their
exact bytes and dtypes: full E/I spike indices, timestep, trial/population
dimensions and per-cell E rates for all 18 networks and 1,000 trials each.
Unused output-spike indices, per-cell I rates and per-sample E rates remain
in the unchanged Gold-2 originals. Checkpoints are referenced, not duplicated.

Original metrics, configs, logs, historical summaries and producer records are
retained verbatim under compute provenance. Operational metrics add only seed
and inhibitory decay recovered from the verified sibling configs. The importer
records the selected file inventory, per-array checksums, destination mappings,
its source and the shared extraction helper. It never downloads or publishes.

The one-off importer was retired after completion. The executed plan, code and verification records
are retained under `.r2/exp046-import-sasnxdvi/` and in compute provenance.

### Preservation results and unresolved historical difference

- All 18 accuracy records and per-cell rate arrays match Gold-2 exactly.
  The pre-migration detector and counter reproduce the new measurements exactly
  on the imported arrays, including when supplied the archived frequencies.
  No numerical definition changed in this migration.
- The **archived summary is not reproduced exactly**. Current analysis observes
  **167,178,240** cell-cycle pairs versus **167,177,216** archived: one additional
  cycle for the 12 ms, seed-43 network. Bucket counts differ in 11 networks;
  every pooled fraction changes by less than **0.0001 percentage points**.
  The historical cause remains unresolved. The matching pre-migration replay
  rules out this stage refactor and the small upstream-frequency change as its
  cause; it does not establish the historical runtime or code responsible.
- Current exp041 frequencies differ from the archive by at most
  **0.000002546 Hz**, leaving all integer peak-separation thresholds unchanged.
  Maximum-rate slope and R squared differ by less than **6 × 10⁻⁹**.
- The expanded suite passed **473 tests** with 20 existing exp041/exp044
  Matplotlib layout warnings. All 24 prior runs and their manifests validate
  unchanged. The three new runs and their lineage validate; discovery lists
  only the new present stage. Scoped lint and whitespace checks passed.
- Both real figure previews were inspected with explicit review warnings.
  The preserved axis limit clips the highest maximum-rate markers, and the
  small-probability bar labels are crowded. These remain presentation-review
  items alongside the disputed reference line; no claim of finished scientific
  presentation is made.

`writings/exp046.typ` and `.artifacts/exp046` remain unchanged. The article is
still `Implemented`, not `Ready for review`. Its hardcoded claims, reference
line, interpretation and the historical counting difference remain for the
author's final science review. No publication, commit or push was performed.

## Future-run data retention — 2026-08-28

Cycle-counting jobs retain sparse E/I events and per-cell E rates only, with lossless compression at the first write. They no longer record voltage, conductance, input or readout trajectories, or accumulate unused I and per-sample rate arrays.

These changes affect future execution only. Existing immutable runs and R2
archives are unchanged. Required arrays keep their original numerical values;
selected NPZ outputs use lossless compression. No production rerun or new
publication was performed for this cleanup.
