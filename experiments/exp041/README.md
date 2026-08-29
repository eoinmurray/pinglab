# exp041 — inhibitory decay, gamma frequency and firing rate

Execution target: **Experiment Runner Guide 3.0.0 / Storage Guide 3.0.0**, using
`pingstore.run/v3` and source-neutral stage IDs. The article follows **Writing
Guide 9.0.0**, with the approved scientific-account correction described below.

## Independent stages

```sh
uv run python experiments/exp041/compute.py --source exp022-r001-compute
uv run python experiments/exp041/analyse.py --source <exp041-compute-id>
uv run python experiments/exp041/present.py --source <exp041-analyse-id>
```

These are instructions for later explicit execution, not actions performed during
this migration. Compute runs inference, not training. Each command prints one new
run ID, records exact upstream pins, and completes independently. Use `--run-id`
only for an unused reservation created before dispatch. Failed work stays hidden;
retry with a fresh identity. There is no implicit latest bank, flat-run fallback,
automatic upstream execution, materialization or publication.

The retired combined runner and module entry point fail before creating outputs.
Importing the package exposes recipe definitions without resolving storage paths.
New collection plans reserve and dispatch all three stages explicitly, including
Slurm origin metadata. Existing legacy campaign plans cannot resume this study.

## Preserved science

- Use the 18 TR-03 final-epoch checkpoints from the explicitly selected exp022
  bank: inhibitory decay 4.5, 6, 9, 12, 18 and 27 ms, each with seeds 42–44.
  Training remains owned by exp022. Retain the 0.1 ms timestep, 200 ms trial and
  fixed 1,000-image official-test subset; the smoke profile uses 100 images.
- Compute retains per-trial population traces and metrics for all 18 cells,
  plus sample-0 snapshots for seed 42 at each decay. Checkpoints are referenced,
  not copied. Simulation commands, configurations and logs belong in provenance.
- Analyse computes full-trial, demeaned Welch density estimates, averages PSDs
  across trials, and finds the 5–150 Hz peak with the existing parabolic
  interpolation and half-bin clamp. Per-trial peak distributions are diagnostics;
  their medians are not the fitted frequencies.
- Preserve both least-squares fits, affine and through-origin, across the six
  condition means over seeds. Both use centred total sum of squares for R².
  Seed error bars are SEMs with ddof=1. Undefined spectra/fits fail explicitly.
- Raster selection uses RNG seed 0, selecting 200 E and 64 I cells without
  replacement. Illustrative rates use the entire snapshot populations over the
  full trial, before subsampling or the 100 ms display window.
- Analyse retains training trajectories from the bank; present reads only saved
  analysis outputs. Those trajectories are validation measurements, so their
  axis labels now say validation. Presentation does not fit, infer or select
  checkpoints and does not stamp repository run IDs onto scientific figures.

Analysis exports `results.json`, `spectra.json` and selected `rasters.npz`.
Presentation exports five figure families (SVG/PDF, with PNG/PDF for the dense
raster) and compact `numbers.json`. The existing per-cell `results` fields and
both `fit` definitions remain available for downstream consumers.

## Writing correction

The author approved correcting the article to match the preserved calculation:

1. The fitted frequency is the interpolated peak of the trial-mean PSD, not a
   median of per-trial peaks.
2. The fit uses six seed-averaged condition points, not 18 individual networks.

The four main plots and both fitted models remain. Numbered Results now precede
Methods; captions distinguish validation trajectories, illustrative rasters,
binned PSD markers and seed SEMs. The cycle-participation argument and equations
remain in an appendix, explicitly conditional rather than established by the
regression. The unsupported universal interpolation-error bound was replaced
with a cited bias caveat. Methods cite the spectral estimators.

Read-only checks of the existing inventory-verified Gold-2 summary found that
the old metadata coefficients did not match retained results; the description
now avoids hardcoded coefficients. All reported values still come only from an
explicitly selected present run. The retained training histories also do not
support a universal accuracy plateau at epoch 15 or rising rates for every
network, so the caption reports the trajectories without those blanket claims.

The original creation date remains; the substantive correction is dated
2026-08-27. The article stayed `Implemented` during synthetic verification.
After the separately authorized import below, its real inputs, claims, figures,
references and limitations were checked and the status advanced to `Ready for
review`. This is not author acceptance or publication.

## Downstream boundary

Exp033 and exp054 still consume legacy exp041 paths. New campaigns stop those
consumers explicitly rather than reading stale or missing files after exp041
finishes. Their own staged input migrations are separate work. Exp046 now pins
the exp041 analysis measurements and the same exp022 training bank directly;
no automatic publication bridge was introduced. Its selective historical-data
import is recorded in [the exp046 notes](../exp046/README.md); scientific article
review remains separate.

The code migration itself imported no Gold-2 data and rewrote no existing runs.
The later, separately authorized import is recorded below. Exp033 and exp054 use
the median of the three per-network frequencies at each decay time; exp046 uses
all 18 frequencies and the same final-epoch training checkpoints. Both forms of
aggregation remain possible from the retained analysis rows. No extra voltage,
conductance or input/output arrays are needed by those existing consumers.

## Verification

The final targeted suite passed **434 tests**, with 17 Matplotlib layout
warnings. Scoped lint and `git diff --check` passed. All 21 existing operational
runs were revalidated unchanged; no operational exp041 run was created.

- The existing bank passes read-only checks for all 18 final checkpoints and
  complete training histories. No production simulation was run.
- A comparison against the preserved pre-migration functions, using synthetic
  evidence, gives exact equality for all 18 measurement records, both fitted
  outputs, six raster selections and their full-population rates.
- Targeted regression coverage includes independent stages, strict ancestor
  pins, v2 rejection, corrupted data, missing measurements, failed reservations,
  profile consistency, collection dispatch and legacy downstream rejection.
- The revised article compiles against synthetic staged presentation outputs;
  corrupt selected input fails instead of producing an empty report. Synthetic
  pages are visibly labelled and are not scientific results. All four rendered
  pages were visually inspected after the writing correction.

Working evidence, source snapshots, test logs and the synthetic comparison are
retained locally under `.r2/exp041-contract-z1qf_9b3/`. No commit, push, R2 write or
publication was performed as part of this migration pass.

The post-writing suite passed in 30.56 s; its log is
`writing-tests-external-temp.log`. An earlier run placed test campaigns inside
the repository and correctly hit the external-campaign-root guard; the final
run used pytest's normal external temporary directory. `writing-preservation.json`
records the unchanged operational runs and synthetic render inspection.

## Completed selective Gold-2 import

On 2026-08-27 the author authorized the audit, local import, independent analysis
and presentation, and real-output article review. No simulation, publication,
remote write, commit or push was performed.

| Stage | Completed run | Operation |
| --- | --- | --- |
| Compute | `exp041-r001-compute` | Local historical import |
| Analyse | `exp041-r002-analyse` | Recompute measurements from imported traces |
| Present | `exp041-r003-present` | Render saved analysis |

All three have `origin: local` and source-neutral IDs. The imported compute run
pins `exp022-r001-compute`; all 18 checkpoint hashes and final-epoch roles match
Gold-2 exactly. The scientific producer remains the original Slurm job 33913524,
campaign `ggs-production-20260818-4ad223d3`, recorded separately from the import.
The original job log identifies host `gpu-q-29` and an NVIDIA A100-SXM4-80GB.

The live R2 `run.json`, `inventory.json` and `lineage.json` matched the local
cache byte for byte. All 148 selected source files were verified against that
inventory before and after import. No full archive download was needed.

The compute export is **7,717,966 bytes**, with **3,528,995 bytes** of provenance:
18 metric/trace pairs covering 1,000 test images each and six seed-42 sample-0
snapshots. Full E/I population traces and the snapshot E/I spikes, labels,
population sizes and timestep retain exact NPY bytes, including dtypes; ZIP
compression alone changes. Unused snapshot voltages, conductances and input/output
recordings remain in the unchanged Gold-2 source. Checkpoints are not copied.
The original metrics are retained verbatim in provenance; operational metrics
add only seed and inhibitory decay, recovered from verified sibling configs.
The run's import plan, file mapping and retained importer record every source
checksum, selected array checksum, destination and transformation.

The one-off offline importer was retired after completion. The executed plan is retained under
`.r2/exp041-import-c12xbg6s/import-plan.json` and in the compute provenance.
It did not download, simulate or publish. Changed plans, wrong checkpoint
roles/hashes, configuration conflicts and checksum mismatches fail closed.

### Scientific and operational verification

- Accuracy, firing rate and sample counts match Gold-2 exactly for all 18
  networks. Gamma frequencies differ by at most **0.000002546 Hz**; the largest
  PSD difference is **1.85 × 10⁻⁷** of its spectrum's peak. Both fitted models
  agree at displayed precision. These differences are consistent with float32
  numerical precision; the specific historical runtime cause was not established.
- Replaying the preserved pre-migration measurement code on the same imported
  arrays reproduces all current spectra and scalar measurements exactly, as well
  as the six raster selections and full-population rates. No estimator changed.
- The affine fit is `a = -0.6991512193 Hz`, `p = 0.2848483063`,
  `R² = 0.9973947097`; the through-origin fit has `p₀ = 0.2712347848`,
  `R² = 0.9947298973`. A negative fitted intercept is not a physical background
  firing rate; the article keeps the participation interpretation conditional.
- Four real article pages were visually inspected using an explicit binding to
  `exp041-r003-present`. The public artifact view was not changed. No author
  review or acceptance is implied by `Ready for review`.
- The expanded regression suite passed **439 tests** (20 Matplotlib layout
  warnings); the five added import cases cover lossless extraction, unchanged
  metrics, independent stages and rejection of inconsistent evidence. Scoped lint
  and whitespace checks passed. Existing 21 runs remain unchanged; all new runs
  and their input lineage validate, and discovery lists only the present stage.

The audit, comparison, live-R2 metadata checks, real rendered pages and test log
are retained under `.r2/exp041-import-c12xbg6s/`. The run-local `README.md` explains
the original HPC execution, local import and retained scientific scope in plain
English. Removing unused source arrays from the local import does not authorize
deleting or altering the Gold-2 originals.

## Future-run data retention — 2026-08-28

Frequency jobs accumulate and store only E population traces; raster jobs retain only E/I spikes and metadata. Both use the E/I spike recorder, without voltage, conductance, input or readout trajectories.

These changes affect future execution only. Existing immutable runs and R2
archives are unchanged. Required arrays keep their original numerical values;
selected NPZ outputs use lossless compression. No production rerun or new
publication was performed for this cleanup.
