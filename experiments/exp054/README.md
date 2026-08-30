# exp054 — rhythmicity across the coupling plane

## Contract migration

Experiment Runner Guide 4.3.0, Storage Guide 4.3.0 and Writing Guide 17.0.0. Independent
contracts, approved selective import, analysis, presentation and scientific
corrections are complete. Shared collection registration was authorized and
coordinated with exp080. The author approved committing exp054-owned files
and its separated shared registration/test hunks. Push, publication and
materialization are not authorized by that approval.

```sh
uv run python -m experiments.exp054.compute
uv run python -m experiments.exp054.analyse --source <exp054-compute-id> \
  --frequency-source <exp041-analysis-id>
uv run python -m experiments.exp054.present --source <exp054-analysis-id>
```

Each command independently completes one source-neutral v4 run. `--run-id` is
only for an unused identity allocated by the shared reservation helper; HPC
execution requires reservation before submission. All sources and their complete
ancestry are validated and rechecked before completion. Failures remain hidden
and cannot be resumed implicitly. Nothing selects a latest run, launches an
upstream stage, materializes output or publishes. The flat runner and combined
package invocation now fail explicitly.

- **Compute:** retain every sparse spike record, including pre-burn activity and
  output spikes, for each unique untrained-network probe. Retain the mean-field
  reference and frequency continuations plus complete up/down solver trajectories.
  Exp033 numerical functions are reused as functions; no exp033 stage is launched.
  Native compute has no upstream inputs. Configurations, commands and logs belong
  in provenance. ZIP compression preserves every original NPY member byte.
- **Analyse:** reconstruct full E/I rasters, apply the original burn-in, measure
  rates and rhythmicity, measure retained mean-field ramps, and calculate the
  exp041 three-seed median overlay. Save the complete numerical summary and all
  plotting coordinates. Inputs are explicit compute and exp041 analysis runs.
- **Present:** draw the existing eight PNGs and compound PDF from saved analysis,
  with `numbers.json` in a flat export. No estimator or solver runs here.

## Preserved scientific recipe

- Untrained PING: 256 E and 256 I cells, seed 42, one trial, 0.25 ms timestep,
  1,000 ms recording and 100 ms discarded burn-in. Private input is 256
  independent channels with identity weight 0.5 at 100 Hz.
- Sweep all 121 points of the 11-by-11 coupling grid: W_EI 0–3, W_IE 0–6.
  Private null rates remain 1, 2, 5, 10, 20, 40, 70, 100 Hz. Shared null rates
  remain 8, 12, 16, 20, 28, 40, 60, 100 Hz, with 200 channels, weight 0.2 and
  initial zero fraction 0.95. There are 136 unique recordings: the grid origin
  at 100 Hz is also the final private null. This reuse removes a duplicate
  invocation of the same seeded scientific condition, not a trial or condition.
- The existing smoke profile retains a 6-by-6 grid and 400 ms recordings; all
  16 null conditions and mean-field settings remain unchanged. There are 51
  unique smoke probes. Analysis/presentation take the profile from the source,
  never the current environment.
- Rates use all cells over the complete post-burn recording. Autocorrelation
  retains 1 ms bins, 100 ms maximum lag, overlap/chance normalization, excluded
  zero lag, smoothing `[0.25, 0.5, 0.25]`, first local trough and preceding lobe.
  The original missing-contrast behavior is preserved, not replaced by zero.
- Display selection remains the first 200 ms, first 160 E/48 I cells, every
  second grid position (all positions in smoke), and the original three diagonal
  callouts. Display reduction does not discard retained scientific recordings.
- Theory retains the 4 mV effective-noise reference, 401 drives over 0–4 nA,
  six inhibitory decays, Brent refinement, 25-point up/down ramps, endpoint
  continuation, LSODA tolerances and original amplitude/regression criteria.
  Native compute now retains trajectories; analysis measures amplitudes after
  1,500 ms in each 2,000 ms integration. No sensitivity sweep or model redesign
  was added to exp054.

The extracted figure functions preserve the scientific coordinates and geometry.
Approved display corrections repair the null lag glyph, replace the unsupported
“Supercritical, reversible onset” title with “Mean-field amplitude”, qualify the
frequency comparison, remove an internal experiment identifier from its legend,
and supply inverse-millisecond units for mean-field rates and eigenvalues.
These changes generated a new immutable presentation; earlier runs are unchanged.

## Shared ownership and integration

The local `collection.py` adapter supports explicit exp041 analysis input,
reservation, independent stage dispatch and validated reuse. Shared registration
now routes exp054 through it in `plan.py`, `execution.py` and `slurm.py`:

- New plans require `stage-refs.json`, with no combined runner command.
- Dispatch and reuse require staged v4 runs and explicit frequency ancestry.
- Scheduler dispatch reserves all stage identities first. No shard/simulator
  change was necessary. Tests mock scheduler submission; no real job was sent.
- Finalization excludes exp054 from legacy capture. Separately authorized
  publication can resolve its validated presentation through the existing route;
  this task did not invoke publication or materialization.

Only exp054 registration hunks and its necessary shared test expectations were
changed. The shared tests affected are `test_gamma_gated_sparsity_collection.py`,
`exp041/test.py` and `exp033/test.py`. Exp080's previously
uncommitted registration and fixture hunks were preserved, not adopted as ours.
The old exp054 source-text assertion now checks the explicit recipe; dedicated
numerical tests verify sigma propagation through the retained compute functions.

A baseline and task-only shared patch are retained in
`.r2/exp054-final-m9bwak6v/before/` and `exp054-shared-only.patch`. They document
ownership, not a command to restore old files. Coordinate the shared index before
staging reviewed hunks; never broadly stage these mixed-ownership files.
No other experiment package, guide, simulator, shared renderer, dependent article
or publication configuration was edited by this task.

## Approved selective Gold-2 import

Read-only audit: `.r2/exp054-contract-4qgrfydx/`. The precise selection and source
hashes are in `import-plan.json`; the audit script and live metadata checks are
retained beside it. The author approved this exact selection before execution.
The pinned plan retains its original proposal status as historical evidence;
authorization and actual execution are separately recorded in the imported run.
Plan SHA-256: `41cbfb4cd2c73c64f8ee82c063d61f33ca7e37ace69fa6be15aa1ae94c1db16d`.

All 833 exp054-related archive files (68,787,686 bytes) match the cached inventory.
Live R2 `run.json`, `inventory.json` and `lineage.json` match the cache byte for
byte. Original producer: campaign `ggs-production-20260818-4ad223d3`, commit
`4ad223d32620dd9f03698b89f28aedfe944d43ac`, Slurm job **33913631**, host `gpu-q-35`.
The derived manifest's `host=local` conflicts with scheduler evidence and must
remain recorded as a historical inconsistency, distinct from the new local import.

Approved inclusion: **828 source files / 49,141,471 bytes**, comprising all 136
full recordings and their configurations/metrics/logs/commands, historical numbers
and manifest, experiment scheduler/completion evidence and shared archive/campaign
metadata. Retain original exp054/exp033/helper producer source and lockfile as
additional provenance. Preserve every field and NPY byte; do not subsample.

The 46,293,216 recording bytes losslessly compress to **3,199,735 bytes**. Selected
source evidence, repacked recordings and producer code total **6,822,238 bytes**.
Estimate **8,396,699–8,896,699 bytes** after generated manifests, mappings, notes,
validation records and the actual checkout patch. The initial smaller estimate
was revised after measuring the full shared-checkout source patch (879,566 bytes)
and detailed plan/mapping (544,895 bytes). This is approximately 87–88%
less retained space than copying exp054's full archive footprint. The source
archive remains unchanged; no disk space is reclaimed. Later analysis/presentation
run sizes are additional; actual completed sizes are recorded below.

Exclude the duplicate 20,419,748-byte compound cache, regenerable figures,
`_run.txt`, every checkpoint bank, and copies of other experiments' payloads.
All cached grid rasters and coordinates can be recovered from the full recordings;
all five mean-field cache components exactly match already-retained exp033 evidence.

Explicit operational references:

| Role | Existing run |
| --- | --- |
| Historical exp054 mean-field evidence retained by exp033 | `exp033-r001-compute` |
| Spiking frequency measurements | `exp041-r002-analyse` |
| Transitive frequency ancestry | `exp041-r001-compute` → `exp022-r001-compute` |

All references and complete ancestry validate. The base and repaired archive
banks remain distinct evidence; this import copies neither. Historical mean-field
trajectories are missing and must not be fabricated. The pinned exp033 import
preserves exp054's own producer identity for that cache.

Replay matches rates and recipe exactly. Differences in contrast are at most
**4.440892098500626e-16**; autocorrelation differences fit the stated FFT tolerance
(`rtol=1e-12`, `atol=1e-15`). Import analysis preserves original scalar values and
records its independent numerical recheck. The historical frequency overlay is
also preserved, with the existing explicit upstream precision deltas. Missing
trajectories are not reconstructed and no historical analysis is called new compute.

Live metadata and selected source hashes were rechecked immediately before the
import. Exp080 and exp082 confirmed disjoint store ownership. Identities were
allocated through the validated stage helper; source evidence and complete
ancestry were rechecked before atomic completion. No identity was guessed.

## Completed local operations

Validation evidence: `.r2/exp054-import-ch71rm5x/`. These are newly allocated
operations on historical evidence, not new simulations or training.

| Stage | Run | Complete bytes | Export bytes |
| --- | --- | ---: | ---: |
| Historical import | `exp054-r001-compute` | 8,706,249 | 3,233,495 |
| Independent analysis | `exp054-r002-analyse` | 1,684,463 | 785,677 |
| Initial rebuilt presentation | `exp054-r003-present` | 2,082,955 | 1,196,026 |
| Corrected presentation | `exp054-r004-present` | 1,825,030 | 1,195,221 |

The import has 843 files. Its 136 recordings occupy exactly **3,199,735 bytes**;
every original NPY member is unchanged. Total import size is within the approved
estimate. It is **87.34% smaller** than the full 68,787,686-byte exp054 footprint;
the initial three operations total 12,473,667 bytes (81.87% smaller). Including
the additional corrected presentation, all four retained operations occupy
14,298,697 bytes. The current chain alone occupies 12,215,742 bytes, but the prior
presentation is retained and must not be omitted from actual disk accounting.
These comparisons exclude the already-retained shared upstream banks. No archive was deleted and
no disk space was reclaimed by pruning. Future runs also retain their own code
patches; size comparisons must include that provenance overhead.

The one-off importer was retired after completion; its code and invocation remain
in run provenance. It checked the exact selection, source sizes/hashes, live metadata,
producer identity/configuration, full raster inventory, every NPY member and
upstream pins. It retains source-to-target mappings, original producer code,
commands and evidence, with the local import distinct from the Slurm producer.
Original configurations, grid and null-control scalar sections compare exactly
in the new analysis and presentation. Independent contrast recalculation differs
by at most 4.440892098500626e-16, within the recorded FFT tolerance.

Lineage is explicit: present → analyse → historical compute, with compute inputs
`exp033-r001-compute` and `exp041-r002-analyse`; the frequency chain continues to
`exp041-r001-compute` and `exp022-r001-compute`. No bank or upstream payload was
copied. All 74 run manifests present before the import remained unchanged;
all 79 run manifests present before the corrected presentation also remained
unchanged. Discovery and the presentation projection validate the visible store.
Both presentations pin the same `exp054-r002-analyse`; the corrected run changes
only the compound PNG/PDF and run-specific bookkeeping, not numerical results.

## Verification and remaining issues

Final review evidence: `.r2/exp054-final-m9bwak6v/`.

- **37 dedicated tests pass**, including stage isolation, invalid inputs and
  ancestry, checksum failures, atomic completion/failure, HPC reservations,
  lossless historical import, sigma propagation, actual figure rendering,
  shared adapter dispatch and mocked scheduler reservation-before-dispatch.
- The integrated Pingstore / exp054 / collection / exp041 / noise-sensitivity
  suite passes **232 tests** (12 existing exp041 plotting-layout warnings).
  All collection tests pass with exp080's and exp054's coordinated fixture fixes.
- **5 exp054 writing checks pass**, covering status, data-free rendering and
  contents. The dedicated render regression also compiles both HTML and PDF,
  verifies the first image is the maps-plus-rasters file, checks visible equation
  numbers, and confirms absent input does not display scientific figures.
- The separate exp033 stage regression has **37 passes and 1 failure**: its
  existing article fixture omits `contents.typ`. This unrelated fixture was not
  edited. This is not a claim that the entire repository test suite is green.
- Ruff, exp054 type checks and task-scoped diff whitespace checks pass.
- No production simulation, training, scheduler submission, publication,
  materialization, archive alteration or completed-run mutation was performed.

The revised article's six figures load in desktop and 390 px mobile browser HTML.
Both numbered equations render, the Methods contents link works and neither
viewport has page-level horizontal overflow. The five-page A4 PDF and compound
PDF were visually inspected: labels/units, coupling ranges, raster windows,
figure-caption pairing and equation layout are correct. Methods begins on its
own page; the first equation is no longer detached from its explanation.

The review uses an isolated snapshot of the live article/helpers and a freshly
validated projection, with only its file path redirected to the audit directory.
No shared `.demolab` projection, default selection or `.artifacts` was changed.
Full application URL routing was not tested. The legacy preview-only binding
loads images but does not populate the current shared Datasets view; the prepared
snapshot does populate the rows.

One shared display limitation remains outside the approved registration scope:
the reviewed snapshot's Datasets row shows the local import's 8-second duration
without an import suffix or the retained 40-minute-2-second Slurm span. Exp080
subsequently added the suffix in its uncommitted shared view changes; those
changes are not part of this migration. Original-producer timing remains absent.
The authoritative imported producer record is correct. Shared projection/view support requires coordinated
ownership; do not alter the completed import to accommodate the display.

The article retains creation date 2026-06-15 and `[▦ DATA]`, with substantive
revision date 2026-08-28. Dependent exp092 and exp109 already have `[▦ DATA]` and
need no status edits. The approved scientific corrections, limitations and
preserved equations are recorded in [SCIENCE_REVIEW.md](SCIENCE_REVIEW.md).
No Reviewed status was assigned. The article is validated against the full
historical production evidence; synthetic/smoke runs are contract tests, not
new evidence for its numerical claims.

## Isolated commit verification

Commit review used committed `c28904af` plus only the 24 exp054-owned paths or
separated shared-file hunks. The isolated candidate passes **202 integration
tests and five writing checks**; the larger shared-checkout total above includes
other tasks' uncommitted tests. The dedicated render fixture now obtains
`lib.typ` from the installed Demolab package, avoiding dependence on generated
checkout files. The standalone writing-status check was given the same installed
library in the disposable candidate; no shared generated files were changed.
Evidence: `.r2/exp054-commit-8yevmopc/`.

## Future-run data retention — 2026-08-28

Probes record E/I spikes without other trajectories and emit only events at or after the registered burn-in. Explicit recording_start_step metadata preserves absolute timestep indices and the full simulation duration. Full-window rates and dynamics are unchanged. Analysis constructs only post-burn-in dense arrays; native compute moves the compressed payload without repacking it.

These changes affect future execution only. Existing immutable runs and R2
archives are unchanged. Required arrays keep their original numerical values;
selected NPZ outputs use lossless compression. No production rerun or new
publication was performed for this cleanup.
