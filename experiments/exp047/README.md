# exp047 — inhibitory pool-size controls

## Contract migration

Experiment Runner Guide 4.3.0 and Storage Guide 4.3.0. The scientific recipe remains the
untrained PING comparison of fixed summed I→E coupling and fixed expected individual
synaptic strength. No training bank, dataset or upstream compute input is needed.
The old combined runner is retired and fails before creating output. Importing
the package does not run code or allocate storage.

Each stage is explicit and independent:

```sh
uv run python -m experiments.exp047.compute
uv run python -m experiments.exp047.analyse --source <exp047-compute-run-id>
uv run python -m experiments.exp047.present --source <exp047-analysis-run-id>
```

- **Compute:** runs each unique simulation once. Per-simulation metrics belong
  in `export/probe/<condition>/metrics.json`; configuration, commands and logs
  use discarded `.scratch/simulations/<condition>/`. `export/evidence.json` records
  the recipe and complete job list. No aggregate or figure is generated.
- **Analyse:** validates the compute recipe, grid and retained configurations;
  maps shared measurements into both controls and computes the seed mean and
  sample standard deviation (`ddof=1`). It retains `results.json` and pins compute.
- **Present:** validates and draws saved analysis without recomputing estimators.
  The flat export contains `pool_size_controls.svg`, `pool_size_controls.pdf`,
  `numbers.json` and the shared publication metadata projection. It pins analysis;
  compute remains reachable through the validated ancestry.

Stages use the shared allocator, source-neutral identities and hidden temporary
directories. Inputs require v4 layout, checksums and exact payload pins.
Complete ancestry is checked before consumption and again before completion.
Failed executions remain hidden and require a new identity or separately
authorized recovery. No stage launches another or materializes/publishes output.

## Preserved scientific recipe

There are 1,024 excitatory cells, 784 synthetic-input channels and inhibitory
pools of 16, 64 and 256. Input is 25 Hz, input-weight parent mean is 1.2 with
initial zero fraction 0.95, and nominal summed E→I coupling is 1.0 µS.
Fixed-total controls use nominal I→E couplings 1, 2 and 4 µS. Fixed-synapse
controls use those values divided by the reference inhibitory pool of 256;
their nominal summed coupling grows with pool size. The existing simulator CLI
arguments and defaults are preserved; no simulator code was changed.

Production uses 500 ms, 0.1 ms timesteps, eight trials and seeds 40, 41 and 42.
`PINGLAB_SMOKE=1` preserves the existing 200 ms/two-trial profile with seeds 40
and 41. E/I rates are the simulator's population- and trial-averaged full-window
rates, not gamma-frequency measurements. Analysis does not pretend to recover
spike trains from these rates.

The 18 control conditions produce 54 reported seed rows but only **42 unique
simulations** (28 in smoke). Both arms share the reference-pool simulations;
they also share the 64-cell, 1 µS condition. This reuse and its order are preserved,
not replaced by independent repeated measurements.

The original `config`, `definition`, `raw` and `summary` interfaces and both
figure filenames are preserved. The four panels, axes, coupling labels, means
and sample-SD error bars are unchanged. The right-column title now says
"Fixed mean synapse", correcting the historical implication that each realised
weight is held identical. Run-ID stamps are omitted to keep repository identifiers
out of rendered scientific figures.

## Collection integration

`collection.py` implements explicit compute/analyse/present orchestration,
reservation before dispatch, exact stage-reference checking, profile checks,
failure handling and reuse of a completed chain. It does not publish.

The shared collection planner, execution dispatcher and Slurm reservation gate
register exp047. New plans use explicit stage references instead of legacy
derived paths; Slurm reserves all three stage identities before dispatch. The
shared regression test rejects legacy composite exp047 output as operational
evidence while preserving historical composition checks.

Concurrent exp033 edits initially changed `plan.py` after this task's baseline
check. Editing stopped, and the author subsequently approved the shared-file
handoff. Exp047-only changes were then applied to the current planner, dispatcher,
Slurm module and shared collection test, preserving the existing exp033 edits.
The isolated diff is `.r2/exp047-contract-vpggkoz_/exp047-integration.patch`.
These files contain mixed ownership and require coordinated hunk separation
before any commit. No simulator or other shared helper was changed.

## Gold-2 audit and approved selective import

The 2026-08-28 audit verified **219 experiment-specific files / 343,525 bytes**
against the cached archive inventory. This includes 210 simulation files /
151,685 bytes, six derived files / 150,844 bytes and three
experiment-specific provenance files / 40,996 bytes. Shared archive
metadata is additional; required shared upstream banks are **zero**.

Fresh R2 `run.json` and `inventory.json` were retrieved into the task's audit
directory and matched the cached bytes exactly. The retired read-only planner
checked those copies, selected source hashes, scientific configurations,
numerical replay and producer lineage without allocating a run.

The original producer is base campaign `ggs-production-20260818-4ad223d3`, commit
`4ad223d32620dd9f03698b89f28aedfe944d43ac`, Slurm job **33913459** on
`gpu-q-44`. Its retained status records completion at `2026-08-19T09:33:31Z`.
The run is not part of the repaired exp022 training branch. The collection-wide
exp022 manifest hash is incidental metadata, not a scientific input pin.
Original per-probe timestamps have no timezone and are preserved without
inventing one.

The author approved this selection, and it was imported on 2026-08-28:

| Selection | Treatment |
| --- | --- |
| 42 metrics, 16,388 bytes | Retain unchanged as compute evidence |
| 168 simulation configurations, commands and logs | Retain unchanged as provenance |
| Historical numbers and runner command | Retain for numerical replay and lineage |
| HPC logs, completion status, base manifest/plan and composite lineage | Retain original producer evidence |
| Archive manifest and entire archive inventory | Retain; losslessly gzip the inventory |
| Old SVG/PDF and two publishing bookkeeping files, 133,654 bytes | Exclude; leave unchanged in Gold-2 |
| Upstream banks | None required or copied |

The plan selects **220 source files / 2,336,236 bytes**, including shared archive
metadata. Exact selected-source retention is **618,144 bytes** after compressing
the complete 2,101,788-byte inventory to 383,696 bytes. No observations, seeds,
trials or metrics are subsampled. These sizes exclude new manifests, recipe,
selection/mapping records, code evidence and independently derived runs. This
small study is not a major storage-saving opportunity: full provenance can exceed
the experiment-specific historical footprint.

No historical spike arrays or realised weight matrices exist in this selection.
Rates can be reaggregated exactly but cannot be remeasured from original spikes.
All 54 historical rows and every summary were reproduced exactly, including the
shared conditions. No new simulation was run to establish those comparisons.

Original audit records are in `.r2/exp047-contract-vpggkoz_/`; import-time
metadata, plan, byte verification and rendering checks are in
`.r2/exp047-import-lk3aeibd/`. These audit directories are not operational runs.

## Completed historical import and independent rebuild

The retired one-off importer rederived the approved plan before allocation and
after copying, checked every source byte/hash, and verified lossless decompression.
Its executed code remains in run provenance. The historical producer remains
Slurm job 33913459; the new compute record explicitly identifies a **local
historical import**, not a newly executed simulation. Its immutable README,
retained commands/logs, producer records, original hashes and mapping preserve
that distinction. There are no scientific upstream inputs or checkpoint roles.

R2 metadata was refreshed immediately before import and matched both the cached
archive and approved plan. All **220 retained files** were verified against their
original hashes (decompressing the inventory). The import's recipe/evidence and
historical numerical replay were checked before atomic completion.

| Run | Role | Entire run bytes | Export bytes |
| --- | --- | ---: | ---: |
| `exp047-r001-compute` | Local historical import of original HPC measurements | 1,274,814 | 21,247 |
| `exp047-r002-analyse` | New local aggregation of retained measurements | 476,936 | 17,103 |
| `exp047-r003-present` | Initial local presentation, retained unchanged | 599,196 | 138,217 |
| `exp047-r004-present` | Corrected mean-synapse label; article review input | 608,786 | 137,856 |

Analysis pins `exp047-r001-compute`; both presentations pin `exp047-r002-analyse`.
Every pin includes the authoritative-manifest hash and payload checksum. The
reviewed chain occupies **2,360,536 bytes**; all four runs occupy **2,959,732 bytes**.
Source retention falls by **1,718,092 bytes (73.5%)** through lossless inventory
compression, and old derived output exclusions avoid another 133,654 bytes.
These are selective-copy savings, not deletion from Gold-2. Complete provenance
and the extra immutable presentation make total new storage larger than the
experiment-specific original footprint.

The initial presentation remains available as historical evidence of the first
rebuild; it was not modified when its label was corrected. The reviewed article
uses only `exp047-r004-present` through an explicit, task-local preview binding.
No published selection or `.artifacts` directory was changed.

All original `config`, `definition`, `raw` and `summary` values match exactly in
the analysis and both presentations: all 54 seed rows, shared conditions, means
and sample SDs are preserved. No production or smoke simulation was launched.
All four new runs validate, and the 48 pre-existing authoritative manifests
remain unchanged. Selected archive sources also remain hash-identical.

## Science and writing review

The article follows Writing Guide 9.0.0, preserving creation date 2026-07-14 and
setting `updated_at` to 2026-08-28. Its milestone is **Ready for review**, not
Reviewed. Results precede the flat numbered Methods; useful interpretation is in
the concise caption and abstract, with no Discussion or repeated Conclusion.
Both the expected-weight equation and active-volley equation are preserved.

The scientific review made these corrections without changing the experiment:

- The inverse-scaling path supports approximate rate invariance for the tested
  grid; two controls do not establish that this compensation is uniquely necessary.
- Expected individual weight and expected summed weight are controlled;
  finite random matrices do not hold every realised synapse identical. The
  expectation identities are exact by definition; nominal Gaussian means only
  approximate their lower-clamped expectations.
- Input spikes use independent Bernoulli draws per timestep, a discrete Poisson
  approximation. Rates average the full simulation window, cells and trials.
- Rate evidence does not measure gamma frequency, phase coherence or volley
  participation. The cited gamma review supports this measurement distinction,
  not the experiment's numerical results or a claim of biological scaling.

The review checked all four figure panels, three coupling levels, inhibitory
sizes 16/64/256, rate axes, legends and seed-SD conventions. Typst's rounding of
15.625 differed from the figure's half-way rounding; Methods now states exact
synapse levels and the legend retains its original rounded labels. Browser HTML
loads the complete figure and both display equations, without horizontal
overflow. Both A4 pages were visually inspected. HTML compilation retains
Typst's standard experimental-export warning; no publication was attempted.

## Verification and remaining gates

The dedicated suite uses synthetic metrics in isolated temporary stores. It
covers stage isolation, exact grid/reuse and sample SD, v4 and scientific-input
validation, payload/manifest/ancestry changes, hidden failures, explicit
reservations, campaign profile/reuse, import-plan checks, byte-preserving imports,
source mutation during import and retired launchers. Rendering regressions cover
the integer seed-list failure, both equations, exact coupling values, the corrected
figure label, missing explicit input and corrupt selected numbers.

**45 dedicated tests and 189 combined regression tests passed.** The combined
suite includes exp023, exp081, the shared collection and Pingstore, including
planner/adapter registration and reservation before mocked Slurm submission.
Ruff, formatting checks and the exp047/package test type check passed with `ty`.
The shared collection test's pre-existing exp033 import-order lint was left
untouched at the ownership handoff. Commands:

```sh
uv run --no-sync pytest -q experiments/exp047/test.py experiments/exp023/test.py experiments/exp081/test.py experiments/collections/gamma_gated_sparsity/test.py tools/pingstore/tests
uv run --no-sync ruff check experiments/exp047
uv run --no-sync ruff format --check experiments/exp047
uv run --no-sync ty check experiments/exp047/test.py
```

Not performed: real simulator execution, Slurm submission, spike-level
remeasurement (spikes unavailable), publication, materialization, or commit/push
by this task. Author acceptance remains required for Reviewed. Commit approval
and coordinated exclusive staging remain required; never include exp033 or other
tasks' changes. The exp033 task was explicitly given the exclusive Git window
while exp047 continued only package/article/test work, plus ownership of the
narrow exp033 addition to Slurm's staged dispatch set after its commit.
