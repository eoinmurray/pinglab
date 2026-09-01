# Exp042: independent timing-intervention stages

Conformance targets: Experiment Runner Guide 4.3.0, Storage Guide 4.3.0 and
Writing Guide 8.0.0. This is a code/writing migration, not a new scientific run.
The initial code migration imported no Gold-2 data. A separately authorized
selective import is recorded below; no existing archive or completed run changed.

## Reflecting-boundary sensitivity: 2026-09-01

Recipe v4 removes the circular connection between the start and end of the
finite presentation. In the independent-spike arm, each proposed event time is
reflected individually into the recorded interval. In the fixed-window arm, the
single shared displacement is reflected against the range allowed by the first
and last actual spike in that source window, then the same corrected displacement
is applied to every event in the group. The latter deliberately avoids splitting
or reversing a group at an edge.

Same-cell/time collisions are still count-preserving, but their nearest-free
search is now bounded: `+1, -1, +2, -2, ...` candidates outside the presentation
are skipped rather than wrapped to the opposite edge. Diagnostics record
reflected events and collision repairs, and the exact per-trial/per-cell count
invariant remains mandatory. Recipe v3 circular results remain immutable
historical evidence but cannot be used as recipe v4 results. A fresh production
compute, analyse and present chain is therefore required.

### Recipe-v4 local verification

All 28 focused exp042 tests pass, including multiple-bounce reflection, both
finite edges, exact count preservation, bounded collision repair and rigid
fixed-window displacement at a boundary. The 27 collection integration tests
and four exp042 downstream-cap tests also pass. Ruff and `git diff --check`
pass. The broader downstream-cap module currently has one unrelated pre-existing
exp049 argument-order failure; exp042 is not implicated.

## Count-preserving jitter repair: 2026-08-31

Issue #167 identified that the retained independent-spike transform clamped
displaced events to the finite presentation boundaries and wrote them into a
binary tensor. Boundary pile-ups and same-cell/time collisions therefore merged
events. The fixed-window group transform used the same mechanism and had the same
defect. Existing retained results honestly measure the delivered inhibitory rate,
but they are not results from a fixed-count timing intervention.

Recipe v3 replaces clamping with circular wrapping for both jitter arms. If two
events from the same trial and inhibitory cell propose the same timestep, the
later event searches `+1, -1, +2, -2, ...` around its proposed circular time and
uses the nearest free timestep. Output remains binary. The transform asserts that
the complete per-trial, per-cell count matrix exactly equals the baseline matrix;
override serialization repeats the total-count check. Every condition records
the number of wrapped events, collision-resolved events and maximum resolution
distance, and analysis rejects missing, false or inconsistent invariants.

This repair does not reinterpret fixed clock windows as detected gamma cycles or
intact biological volleys. The arm remains a **fixed-window group jitter**
intervention; events on opposite sides of a window boundary may receive different
proposed offsets. Detecting and shifting empirical volleys would be a new protocol.

The transform change alters injected activity and therefore invalidates existing
exp042 compute outputs as evidence for recipe v3. A fresh production compute run,
followed by new analyse and present runs, is required before updating scientific
claims or publication inputs. The pinned exp022 checkpoint bank does not require
retraining. No HPC run was launched as part of this implementation.

### Local pre-HPC verification

The 2026-08-31 preflight passed all 25 focused exp042 tests. These cover both
complete production sigma grids, exact per-trial/per-cell counts, deterministic
binary output, zero jitter, temporal wrapping, a fully collided cell, shared
fixed-window population shifts, sparse serialization, malformed baseline
rejection, analysis fail-closed behavior, stage lineage, sharding and rendering.
The 27 collection integration tests, four exp042 downstream-cap tests and six
simulator override/snapshot tests also passed. Ruff and `git diff --check` passed.
A production simulation was deliberately not run; these checks establish code
and contract readiness, not scientific results.

## Explicit commands

```sh
uv run python experiments/exp042/compute.py --source <exp022-compute-id>
uv run python experiments/exp042/analyse.py --source <exp042-compute-id>
uv run python experiments/exp042/present.py --source <exp042-analyse-id>
```

Each command completes one v4 run and prints its ID. All inputs and their complete
ancestry must be validated v4 runs with exact payload pins. There is
no latest-run, historical-directory or missing-cache fallback. The bank is an
explicit human choice; the migration does not choose between historical banks.
The three TR-02 PING baseline cells use final-epoch checkpoints, never deployment
checkpoints. No stage trains upstream models or publishes its output.

The reduced scientific scope contains 57 production condition jobs (30
fixed-window group-jitter and 27 independent-spike-jitter), or 30 smoke jobs
with `PINGLAB_SMOKE=1`. Evaluation uses 1,000 or 100 test images respectively. Two
illustrative perturbation recordings at 14 ms are additional compute work, not
part of the condition-job count. Shared baseline caching avoids duplicate baseline
simulations across shards. The two zero-zero arms share one canonical replay per
seed, retaining separate logical rows and explicit `replay_of` provenance. The
successful fresh-run launch budgets are 60 production and 33 smoke (three fewer
distinct sweep evaluations than condition rows, plus three baseline recordings
and three illustrative launches).
Retries can add work.
No production runtime or retained-size measurement exists for this staged version.

## Retention and stage boundaries

- Compute requests I-only baseline recordings and E/I-only illustrative snapshots;
  voltage, conductance, input and readout recording buffers are not allocated.
  Metrics-only overrides do not record trajectories. Inference loads only the
  MNIST test partition, preserving test selection, normalization and RNG behavior.
  Compute retains per-condition simulator metrics and two losslessly compressed
  single-trial spike recordings, with configurations, checkpoint hashes and
  execution provenance. Baseline I rasters and overrides remain scratch;
  unused snapshot channels are not generated. Overrides are removed immediately after inference;
  sharded baselines are shared within the compute reservation until completion.
- Analyse selects the same display cells, measures full-population illustrative
  rates, and computes the same per-seed rows, means and standard errors. Its
  compact raster selections let presentation run without simulation.
- Present reads only retained analysis and renders `rhythm_compound.png` plus
  report numbers. Exports are flat. Existing runs stay intact; obsolete figures
  are never created, and existing outputs are never cleaned in place.

The two figure-producing spike-time transforms retain rounded Gaussian offsets,
then apply the recipe-v4 reflection and bounded collision policy documented
above. Realised rates remain measured in addition to the exact count invariant.
The compound-panel caption reflects its mean curves without error bars. Standard
errors remain retained in the analysis for the scientific uncertainty account.

## HPC and recovery

The collection adapter retains eight round-robin compute shards. It reserves the
compute, analyse and present IDs before live Slurm submission. Shards receive an
explicit completed bank and write only their allocated compute reservation.
Each shard has private override scratch, an exclusive worker lock, code/bank/recipe
record, and checksummed completion record. Distributed execution requires frozen,
committed execution code. A retry reuses only a matching complete shard; stale
locks require explicit recovery rather than automatic removal.

A shared reservation lock prevents collection racing workers; per-cell locks
prevent duplicate baseline and zero-zero replay simulations. Either zero-zero arm
may acquire the lock first; it always executes the canonical cycle-zero job and
the other arm reuses those metrics. The replay is not replaced with unperturbed
baseline metrics. After all shards succeed, the compute
collector verifies them, produces the two
illustrative recordings, removes the shared baseline scratch, and completes the run atomically. Only then does the
collection orchestrator invoke analyse and present explicitly. Neither downstream
stage launches compute. A failed collector remains hidden and needs a fresh
identity or separately reviewed recovery, not automatic reuse.

Legacy monolithic commands, `--skip-training`, `--plot-only` and the old RunPod
dispatcher are not operational interfaces. The simulator can execute on the
current host, including an explicitly provisioned GPU host; automated RunPod
dispatch has not been ported to the v4 reservation protocol. Historical campaigns
require their original checkout and are not accepted by the new adapter.

## Review and publication

The writing consumes an explicitly selected v4 presentation. Without one, it
shows the shared unavailable-data notice. Fixture-based tests are not scientific
evidence or published results. Preview selection, production execution and
publication remain separate human decisions. Keep all pinned input runs when
transferring a completed result.

## Performance changes verified: 2026-08-28

The simulator fast suite passed 382 tests (two existing expected failures; slow
tests excluded). The focused experiment, collection, checkpoint and writing-input
suite passed 121 tests. New checks compare metrics-only overrides with full
recording on a real tiny simulator, compare I-only raster events and selected
snapshot arrays exactly, verify test-only loading, exercise concurrent zero
replay reuse, and retained all 66 rows in the then-current recipe with 66 mocked
CLI launches.
The existing writing fixture compiles with only PNG/SVG presentation files.
Ruff and whitespace checks passed. The three existing exp042 runs and their
exp022 bank still validate against their pre-edit manifest and payload pins.

These are code/fixture checks, not a production rerun or a measured GPU wall-time
speedup. No existing run, archive or published output was modified.

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

The current bank is `exp022-r001-compute`; current exp042 IDs are
`exp042-r001-compute`, `exp042-r002-analyse` and `exp042-r003-present`.
The [source-neutral naming migration](../../tools/pingstore/SOURCE_NEUTRAL_IDS.md)
updated all pins while preserving local import origin and Wilkes ancestry.
The commands, suffixed IDs and sizes below describe the original import, before
the origin correction and naming migration; retained migration evidence adds bytes.
Original import plans remain historical evidence; any future import must create
a fresh plan against the current bank ID and checksums.

The separately authorized import retained the 66 conditions in the then-current recipe
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

The completed import was a scoped historical operation, not an operational
legacy fallback. Its retired entrypoint planned from archive metadata and
validated the selected subset before reserving a new v3 compute identity. It did
not run simulation, analysis, presentation or publication.
Missing files, wrong hashes, changed plans, scientific mismatches and ambiguous
checkpoint identities fail closed.

The root archive metadata was fetched before planning and rechecked against R2
afterward. The existing plan path is deliberately not overwritten. The exact file
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

Exp092 needs `numbers.json` and `rhythm_compound.png`.
No collection compute experiment depends on exp042 raw arrays. Those
outputs are present, but no downstream experiment was migrated or executed here.
Keep the pinned exp022 bank and its complete ancestry with any transferred run.
R2 remains unchanged. No simulation, publication, commit or push was performed.
