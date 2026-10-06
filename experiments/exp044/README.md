# Exp044: integration-timestep audit

Execution follows Experiment Runner Guide 4.9.0 and Storage Guide 4.9.0.
The title and matching article links were edited under Writing Guide 36.0.0.
Training remains owned by exp022. Exp044 never launches it.

```sh
uv run python experiments/exp044/compute.py --source <exp022-compute-bank>
uv run python experiments/exp044/analyse.py --source <exp044-compute-run>
uv run python experiments/exp044/present.py --source <exp044-analyse-run>
```

Each command prints one new completed run ID. All stages require explicit v4
sources and validate their pinned dependencies up to and including the explicitly
selected exp022 bank before consuming evidence and again before completion.
Missing stage inputs or banks, v2, changed manifests/payloads,
wrong experiments/stages and incomplete evidence are errors. No latest-run or
mutable training-root fallback is used. The former combined launcher fails with
stage directions, including for `--plot-only` and `--skip-training`.

## Preserved science and outputs

- The current v3 native-graph recipe uses timesteps 0.05, 0.1, 0.2, 0.3 and
  0.6 ms, with seeds 42–44, explicit E/I refractories of 1.2/0.6 ms, a 7,000-image training
  pool and 50 training epochs. Trials last nominally 200 ms; whole-step
  truncation gives 666/333 steps and 199.8 ms at 0.3/0.6 ms. Final-epoch
  checkpoints are used for both evaluation and raster probes. The historical
  v1 recipe and its 0.05/0.1/0.25/0.5/1-ms grid remain historical evidence;
  new stages accept only the v3 recipe.
- Compute retains 15 official-test evaluations and five seed-42 raw spike probes.
  The default evaluation uses 1,000 images. `PINGLAB_SMOKE=1` retains the existing
  100-image diagnostic cap; it is recorded in compute provenance. Downstream
  stages use the saved profile, not their environment.
- Inference explicitly requests automatic local device selection instead of
  inheriting the training host's CUDA device from the bank configuration.
- Analysis validates the training settings and histories, measures E/I rates,
  retains test accuracy and computes means and SEM across seeds. It prepares
  the same deterministic 200-E/64-I neuron samples, preserving full-population,
  full-trial raster rates and a 100 ms display window. No gamma-period estimator
  or new convergence criterion is introduced.
- Presentation renders saved analysis and exports the same six SVG/PNG/PDF
  filenames and `numbers.json`. Per-cell `results` and the prior numerical
  configuration fields remain available to downstream articles. Histories are
  correctly labelled validation measurements; figures have no run-ID stamps.
  Claims about monotonicity, convergence and period invariance are not inferred
  from an earlier article's hardcoded values.

## Execution and storage boundary

Use `--run-id` only for an unused v4 reservation. Local and scheduler executions
reserve fresh stage identities; failures leave hidden incomplete runs. Source
checkpoints remain in the bank. Graph execution binds bank checkpoints in
memory; no CLI commands, copied training configurations or simulator logs are
generated. Scientific data live
in compute `export/`; graph digests, parameter roles, devices and timings live
in `run.json`. Raw spike probes use `spikes.npz`.
Analysis and presentation never simulate. None of the stages materializes or
publishes. Preview/publication requires a separately selected present run.

Collection plans reserve all three stage IDs before dispatch, read the explicit
bank ID from the exp022 campaign manifest, and record checksum-pinned references.
Completed stages may be reused only with matching bank, profile and lineage.
Legacy campaign plans are rejected; staged outputs are excluded from v2 capture.

## Explicit source boundary

The subsequent [exp022 ancestry repair](../exp022/ANCESTRY.md) verified the R2
ancestor and updated this chain's pins without changing scientific outputs.
The source-boundary policy below remains unchanged; its historical references
record the circumstances of the original execution.

The user explicitly selected `.pingstore/runs/exp022-r001-compute` as the
new source data for exp044. This is a scoped source-boundary instruction: exp044
validates and pins the selected v4 bank's complete payload and authoritative
manifest, but does not recursively require its older import sources. Its 15
timestep cells contain the configurations, final checkpoints and histories used
here. Neither the bank nor its historical references are changed or migrated.

Each new stage records `source_boundary` in run.json, naming the validated bank
and preserving its untraversed historical input references. In particular,
`exp022-gold-2-repaired-slurm` remains an unresolved historical reference in the
bank; these runs do not claim to have verified that earlier source. The boundary
does not authorize v2 consumption or relax validation of exp044's own stage
inputs, and it is not a repository-wide change to the guides.

## Historical verification

The checks below predate the native-graph migration and do not verify it.

The unit tests use synthetic temporary banks and mocked inference, not scientific
runs. They exercise stage separation, checkpoint policy, measurements, failure
handling, v4 lineage, collection dispatch and selected-input Typst rendering.

```sh
uv run pytest experiments/exp044/test.py
```

The conformance verification passed 214 tests, including checkpoint/smoke rules,
collection dispatch, writing inputs and exp023/024 stage regressions. The selected
article was compiled and visually checked using both synthetic and newly executed
production results. Ruff and `git diff --check` passed. The inherited Matplotlib
`tight_layout` warnings remain; the inspected figures were legible and unclipped.

## Production execution

The full production recipe completed on 2026-08-27 with `PINGLAB_SMOKE=0`.
The table uses the current source-neutral identities:

| Stage | Completed run |
| --- | --- |
| Compute | `exp044-r002-compute` |
| Analyse | `exp044-r003-analyse` |
| Present | `exp044-r004-present` |

Compute evaluated 1,000 official-test images for each of 15 final checkpoints and
retained five raster probes. Mean E rates across seeds ranged from 13.84 to
20.15 Hz over the five timesteps; mean test accuracy ranged from 88.8 to 90.1%.
The presentation export contains six figure files, `numbers.json` and the
bookkeeping projection `_manifest.json`. All three completed runs passed layout,
payload and pinned-source validation. The production article compiled to four
preview pages, all visually inspected. No training, publication or materialization
into `.artifacts/` was performed.

The bank's before/after references matched exactly during that production
execution. These historical hashes precede the later ancestry and ID migrations:

- Payload: `sha256:2513137c209022d1c68308c1705cae2c45c4c461f1315ea58e02fb7d4600d881`
- Authoritative manifest SHA-256: `fa83345874e50809363b92c4970e9045bcac530a27ac04c89246625371dcee7c`

An earlier attempt, before the explicit source-boundary instruction, stopped
before reservation because `exp022-gold-2-repaired-slurm` was absent. After that
instruction, `exp044-r001-compute-local` failed because the imported configuration
requested unavailable CUDA; its hidden incomplete directory is retained for
inspection and is not scientific evidence. The successful retry used a fresh
identity after fixing device selection. No source evidence was rewritten.


## Bank origin correction

The selected bank was previously named `exp022-r001-compute-slurm`. The subsequent
`exp022-r001-compute-local` identity described the local import; historical
training remains Slurm. The separately authorized
[origin correction](../exp022/README.md#local-import-origin-correction) updated
all three exp044 runs' pins and selected-bank references without changing their
scientific outputs, original execution records or selected-bank boundary.

The current bank is `exp022-r001-compute`. The
[source-neutral naming migration](../../tools/pingstore/SOURCE_NEUTRAL_IDS.md)
removed execution-source suffixes from all completed runs and updated their pins.
Local import origin and historical Slurm training remain explicit in `run.json`.

## Future-run data retention — 2026-08-28

Raster probes retain only E/I spikes and metadata, using the E/I spike recorder. The official-test evaluations remain metrics-only. All timesteps, sample choices and full-trial rates are unchanged.

These changes affect future execution only. Existing immutable runs and R2
archives are unchanged. Required arrays keep their original numerical values;
selected NPZ outputs use lossless compression. No production rerun or new
publication was performed for this cleanup.

## Corrected refractory sweep — 2026-09-09

PLAN.md step 5.6 completed the production v2 recipe against `exp022-r007-compute`:
`exp044-r007-compute` → `exp044-r008-analyse` → `exp044-r009-present`.
Compute ran locally in **2 min 25 s**, completing all fifteen 1,000-image
official-test evaluations and five seed-42 probes. Mean test accuracy spans
88.37–89.63%, E firing 14.27–17.02 Hz and I firing 87.29–153.90 Hz.

All stages passed v4 payload and explicit-lineage validation. Eighty relevant
tests passed; the existing article-render test was excluded. All recorded
neurons respected the configured refractory intervals, and coarse-probe rates
use the actual 199.8-ms duration. The unchanged 0.1-ms control reproduces all
three old evaluation rows and both raw probe spike arrays exactly. The old
coarse conditions remain distinct historical evidence.

The three figure sets were inspected and are legible and unclipped despite the
existing `tight_layout` warnings. Full digests, seed summaries and comparisons
are recorded in PLAN.md. Article adoption, publication and consumer repinning
remain separate work; no training, materialization or push occurred.

## Native graph migration — 2026-10-06

Compute declares the circuit with `snnlab.lang`, reuses
`experiments.helpers.ping.build_ping` unchanged, and executes `GraphExecutor`
directly. No removed PING component, CLI builder, subprocess, legacy model or
parameter-interchange adapter is used. Scientific settings and checkpoint
parameter roles belong to the recipe. Compute/analyse/present remain independent.

The verified final-epoch bank matrices bind directly in runtime [source, target]
orientation without transposition or fan-in rescaling. Feedforward matrices use
the trained forward pass's nonnegative clamp; reciprocal matrices retain their
stored values. Checkpoint keys, dtype, shapes, finiteness, nonnegative reciprocal
weights and zero E→E/I→I matrices are checked before inference. Unsupported
signed, adaptive, learned-leak or alternative-readout configurations are rejected.
The output LIF retains its 2-ms decay, threshold-one subtractive reset and mean
pre-reset voltage over the full trial. Each presentation starts from reset state.

Official-test evaluations keep the seed-42 subset, 64-image batches and
20260415 encoder stream, including mean cross-entropy and E/I firing rates.
Raster probes use the same image and network seed,
but their native encoder stream starts from that seed rather than consuming
historical CLI weight-initialization draws. Exact historical raster identity is
therefore not claimed. New v3 stages reject the former v1/v2 execution recipes;
all historical runs, checkpoint files, measurements and figures remain unchanged.

Validation for this migration is limited to source inspection, syntax parsing,
Ruff and diff review. Existing stage-separation fixtures were updated to mock
the native cell-evaluation boundary; they were not run. No experiment stage,
training, simulation, run creation, result regeneration, commit or push occurred.
Checkpoint loading against the real bank, graph execution, device behaviour,
numerical parity, memory use and rendering remain unverified. The article's
legacy title marker and matching links were removed without changing its results.
Article tags and local-data availability remain unassessed: discovery and a full
Writing Guide conformance pass are outside this static-only migration.


The metadata cleanup removes `TRAINING_COMMON_FIELDS` and its package export.
The bank reader explicitly extracts the scientific quantities needed for graph
execution, training-history interpretation and the saved article interface.
Initializer specifications, unused parameter bounds, optimizer replay settings
and generic simulator feature inventories are not carried into the native
recipe. Unsupported forward dynamics are rejected at the immutable-bank input
boundary. The stored bank's field names and checkpoint tensor keys remain data
schema requirements; they do not select or reconstruct a legacy executor.
New metric exports use `accuracy_pct`, `cross_entropy` and E/I rate keys rather
than the CLI's best-accuracy and hidden/inhibitory aliases. Analysis retains its
existing article-facing result names. This follow-up received static checks only.
