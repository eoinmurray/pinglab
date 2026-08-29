# Exp044: integration-timestep audit

Conformance target: Experiment Runner Guide 2.0.0, Storage Guide 2.0.0 and
Writing Guide 8.0.0. Training remains owned by exp022. Exp044 never launches it.

```sh
uv run python experiments/exp044/compute.py --source <exp022-compute-bank>
uv run python experiments/exp044/analyse.py --source <exp044-compute-run>
uv run python experiments/exp044/present.py --source <exp044-analyse-run>
```

Each command prints one new completed run ID. All stages require explicit v3
sources and validate their pinned dependencies up to and including the explicitly
selected exp022 bank before consuming evidence and again before completion.
Missing stage inputs or banks, v2, changed manifests/payloads,
wrong experiments/stages and incomplete evidence are errors. No latest-run or
mutable training-root fallback is used. The former combined launcher fails with
stage directions, including for `--plot-only` and `--skip-training`.

## Preserved science and outputs

- The five timesteps remain 0.05, 0.1, 0.25, 0.5 and 1 ms, with seeds 42–44,
  200 ms trials, a 7,000-image training pool and 50 training epochs. Final-epoch
  checkpoints are used for both evaluation and raster probes.
- Compute retains 15 official-test evaluations and five seed-42 raw snapshots.
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

Use `--run-id` only for an unused v3 reservation. Local and scheduler executions
reserve fresh stage identities; failures leave hidden incomplete runs. Source
checkpoints remain in the bank. Commands, logs and training configurations are
retained under `provenance/`, while raw outputs live in compute `export/`.
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
validates and pins the selected v3 bank's complete payload and authoritative
manifest, but does not recursively require its older import sources. Its 15
timestep cells contain the configurations, final checkpoints and histories used
here. Neither the bank nor its historical references are changed or migrated.

Each new stage records `source_boundary` in run.json, naming the validated bank
and preserving its untraversed historical input references. In particular,
`exp022-gold-2-repaired-slurm` remains an unresolved historical reference in the
bank; these runs do not claim to have verified that earlier source. The boundary
does not authorize v2 consumption or relax validation of exp044's own stage
inputs, and it is not a repository-wide change to the guides.

## Verification

The unit tests use synthetic temporary banks and mocked inference, not scientific
runs. They exercise stage separation, checkpoint policy, measurements, failure
handling, v3 lineage, collection dispatch and selected-input Typst rendering.

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
