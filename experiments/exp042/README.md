# Exp042: independent timing-intervention stages

Conformance targets: Experiment Runner Guide 2.0.0, Storage Guide 2.0.0 and
Writing Guide 8.0.0. This is a code/writing migration, not a new scientific run.
The initial code migration imported no Gold-2 data. A separately authorized
selective import is recorded below; no existing archive or completed run changed.

## Explicit commands

```sh
uv run python experiments/exp042/compute.py --source <exp022-compute-id>
uv run python experiments/exp042/analyse.py --source <exp042-compute-id>
uv run python experiments/exp042/present.py --source <exp042-analyse-id>
```

Each command completes one v3 run and prints its ID. All inputs and their complete
ancestry must be validated v3 runs with exact manifest and payload pins. There is
no latest-run, historical-directory or missing-cache fallback. The bank is an
explicit human choice; the migration does not choose between historical banks.
The three TR-02 PING baseline cells use final-epoch checkpoints, never deployment
checkpoints. No stage trains upstream models or publishes its output.

The reduced scientific scope is unchanged: 66 production condition jobs (nine
controls, 30 cycle-jitter, 27 per-cell-jitter), or 39 smoke jobs with
`PINGLAB_SMOKE=1`. Evaluation uses 1,000 or 100 test images respectively. Two
illustrative perturbation recordings at 14 ms are additional compute work, not
part of the condition-job count. Shared baseline caching avoids duplicate baseline
simulations across shards: the successful fresh-run launch budgets are 69 production
and 42 smoke (condition jobs plus three illustrative launches). Retries can add work.
No production runtime or retained-size measurement exists for this staged version.

## Retention and stage boundaries

- Compute retains per-condition simulator metrics and two losslessly compressed
  single-trial spike recordings, with configurations, checkpoint hashes and
  execution provenance. Large baseline rasters, overrides and unused snapshot
  channels remain scratch. Overrides are removed immediately after inference;
  sharded baselines are shared within the compute reservation until completion.
- Analyse selects the same display cells, measures full-population illustrative
  rates, and computes the same per-seed rows, means and standard errors. Its
  compact raster selections let presentation run without simulation.
- Present reads only retained analysis and renders `rhythm_compound`,
  `cell_jitter_sweep` and `jitter_sweep`, plus report numbers. Exports are flat.
  Obsolete figures are never created; existing outputs are never cleaned in place.

The original spike-time transforms are preserved, including rounded offsets,
boundary clamping and merging coincident spikes. Realised rates remain measured;
exact count preservation is not claimed. Results prose is retained in Discussion.
The compound-panel caption now reflects its actual mean curves without error bars;
the standalone sweep figures retain standard errors.

## HPC and recovery

The collection adapter retains eight round-robin compute shards. It reserves the
compute, analyse and present IDs before live Slurm submission. Shards receive an
explicit completed bank and write only their allocated compute reservation.
Each shard has private override scratch, an exclusive worker lock, code/bank/recipe
record, and checksummed completion record. Distributed execution requires frozen,
committed execution code. A retry reuses only a matching complete shard; stale
locks require explicit recovery rather than automatic removal.

A shared reservation lock prevents collection racing workers; per-cell locks
prevent duplicate baseline simulations. After all shards succeed, the compute
collector verifies them, produces the two
illustrative recordings, removes the shared baseline scratch, and completes the run atomically. Only then does the
collection orchestrator invoke analyse and present explicitly. Neither downstream
stage launches compute. A failed collector remains hidden and needs a fresh
identity or separately reviewed recovery, not automatic reuse.

Legacy monolithic commands, `--skip-training`, `--plot-only` and the old RunPod
dispatcher are not operational interfaces. The simulator can execute on the
current host, including an explicitly provisioned GPU host; automated RunPod
dispatch has not been ported to the v3 reservation protocol. Historical campaigns
require their original checkout and are not accepted by the new adapter.

## Review and publication

The writing consumes an explicitly selected v3 presentation. Without one, it
shows the shared unavailable-data notice. Fixture-based tests are not scientific
evidence or published results. Preview selection, production execution and
publication remain separate human decisions. Keep all pinned input runs when
transferring a completed result.

## Migration verification

On 2026-08-27, 329 related tests passed (eight existing exp044 layout warnings).
Checks covered v3 rejection, complete lineage pins, independent stages, shard
resume and collection locking, temporary-array cleanup, retained measurements,
and collection integration. All three spike-transform algorithms matched the
pre-migration syntax trees after excluding docstrings. Ruff and `git diff --check`
passed for the changed execution code and tests.

The four-page article was compiled and visually reviewed using explicitly labelled
synthetic fixtures; the unavailable-data state and corrupt-input failure were also
tested. This verifies rendering and interfaces, not scientific conclusions or
production runtime. That initial code migration created no operational exp042 runs.


## Selective Gold-2 import: 2026-08-27

The separately authorized import retained the 66 conditions in the current recipe
and the seed-42, sample-0, sigma-14 ms cycle/per-cell snapshots. This is a subset of
the historical Wilkes campaign, not a rerun of the current compute implementation.
All three original final-epoch checkpoint SHA-256 values match the explicitly
pinned `exp022-r001-compute-slurm` bank. No checkpoint was copied downstream.

| Stage | Completed run | Total retained bytes |
|---|---|---:|
| Historical compute import | `exp042-r001-compute-local` | 100,753,155 |
| New analysis | `exp042-r002-analyse-local` | 249,869 |
| New presentation | `exp042-r003-present-local` | 584,010 |

The three runs total **101.6 MB** including byte-preserved source evidence,
archive inventories, historical execution logs and code provenance. The compute
scientific export is only **201,657 bytes**: wrapped metrics and losslessly
compressed original E/I spike arrays and labels. The two complete original
snapshots remain under compute `provenance/gold-2/`, including their unused
channels, to preserve original scientific bytes. Their presence does not make
those unused channels operational inputs.

The allowlist contains 361 payload files, 98,050,094 bytes, plus the archive's
root `run.json` and `inventory.json`. Overrides, full evaluation rasters, other
snapshots, Pareto and cross-tau outputs were not imported. Baseline caches contain
metrics with embedded settings, rather than separate config files; original
shard logs retain their execution history. Historical figures and numbers are
comparison evidence only, outside the new presentation export.

`import_gold2.py` is a scoped, explicit historical operation, not an operational
legacy fallback or general storage CLI. It plans from archive metadata without
fetching, then validates the entire selected subset before reserving a new v3
compute identity. It never runs simulation, analysis, presentation or publication.
Missing files, wrong hashes, changed plans, scientific mismatches and ambiguous
checkpoint identities fail closed.

Commands used (paths relative to the repository):

```sh
uv run python experiments/exp042/import_gold2.py plan \
  --archive .r2/exp042-import-20260827 \
  --source exp022-r001-compute-slurm \
  --plan .r2/exp042-import-20260827/import-plan.json
rclone copy r2:pinglab/campaigns/gold-2 .r2/exp042-import-20260827 \
  --files-from-raw .r2/exp042-import-20260827/import-plan.files.txt
uv run python experiments/exp042/import_gold2.py import \
  --archive .r2/exp042-import-20260827 \
  --plan .r2/exp042-import-20260827/import-plan.json
uv run python experiments/exp042/analyse.py --source exp042-r001-compute-local
uv run python experiments/exp042/present.py --source exp042-r002-analyse-local
```

The root archive metadata was fetched before planning and rechecked against R2
afterward. The existing plan path is deliberately not overwritten; do not rerun
these commands as a publication or automatic resume operation. The exact file
mapping and hashes are also retained immutably in compute
`provenance/import-plan.json`. Local working evidence and review pages are in
`.r2/exp042-import-20260827/`.

Verification: **337 tests passed**, with eight existing exp044 layout warnings.
All 66 regenerated result rows exactly match the original numbers. Raster display
selections and full-population illustrative rates match the original snapshots;
compaction preserves array dtypes and values. All three runs and their recursive
input pins validate under v3. The current four-page writing was rendered and
visually reviewed with the imported presentation, without editing its text.
Existing run manifests, published files and user guide edits remain unchanged.

Exp092 needs `numbers.json` and `rhythm_compound.png`; exp093 uses the compound
figure. No collection compute experiment depends on exp042 raw arrays. Those
outputs are present, but no downstream experiment was migrated or executed here.
Keep the pinned exp022 bank and its complete ancestry with any transferred run.
R2 remains unchanged. No simulation, publication, commit or push was performed.
