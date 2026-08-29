# exp085 — pathway-specific synchronization of two PING networks

## Contract migration

This is a `demo` experiment, not a member of the gamma-gated-sparsity execution
collection. The migration follows Experiment Runner Guide 3.0.0 and Storage
Guide 3.0.0; exp037 was an implementation example. No shared collection,
simulator, registry or shared-test edits are required.

```sh
uv run python -m experiments.exp085.compute
uv run python -m experiments.exp085.analyse --source <exp085-compute-run-id>
uv run python -m experiments.exp085.present --source <exp085-analyse-run-id>
```

The corresponding `experiments/exp085/{compute,analyse,present}.py` scripts also
work directly. Each command completes one immutable `pingstore.run/v4` run and
prints its source-neutral ID. `--run-id` accepts only an unused reservation
allocated through the existing shared stage helper. Flat execution and
`python -m experiments.exp085` stop with explicit-stage instructions.

There is no latest-run fallback, upstream dispatch, training, import,
materialization, publication or automatic selection. Failed work remains hidden
and requires a fresh identity; downstream stages never resume it. Inputs pin
both payload and authoritative manifest hashes, validate full ancestry before
use, and recheck that ancestry before atomic completion. V2, wrong-stage inputs,
recipe mismatches, altered payloads, altered manifests and unsupported layouts
are rejected.

## Independent stages and retained evidence

- **Compute:** 49 simulation calls: one uncoupled trajectory, one phase-response
  baseline, 42 probes (21 phases for each target population), one uncoupled
  prefix, and four coupling branches. It retains every returned recording and
  initialized parameter tensor without dtype conversion or subsampling, plus
  exact input trains, five graph definitions, requests, acquisition schedule and
  the shared prefix runtime state. Each branch receives a detached copy of the
  same state and the same input suffix. The state is a dynamical checkpoint, not
  a trained or validation-selected model. Parameters are retained separately
  because the runtime-state format deliberately excludes weights.
  Matrices retain the executor's source-by-target orientation; graph declarations
  use the transposed target-by-source shape. Retention never transposes the data.
- **Analyse:** consumes one explicit standalone compute run, verifies its
  acquisition grid and evidence, and independently computes all numerical
  results and plot traces. It retains `results.json`, a JSON/NPZ plot-data pair
  and the small network definition, referencing rather than copying raw
  recordings. The complete phase-response and four-condition results remain
  available. Per-timestep means of illustrative voltage and conductance are
  calculated here, rather than during presentation.
- **Present:** consumes one explicit analysis run and validates its compute
  ancestry. It draws the same network SVG and five PNG figures, and writes
  `protocol.json` and `numbers.json` into a flat export. Presentation metadata
  is a projection; `run.json` remains authoritative. Plot normalization and
  display geometry stay in presentation; scientific estimators do not.

Compute detects one baseline cycle near 700 ms to schedule phase-specific
probes. This is an acquisition decision, not an invocation of the analysis
stage. Analysis independently checks that the retained cycle matches the raw
baseline before measuring responses. Configuration, code/lock provenance,
actual stage commands, timestamps and per-simulation execution records are
retained by the stage helper and the experiment.

## Preserved scientific choices

The committed recipe retains 80 E and 20 I cells per network, 128 input channels,
0.1 ms timesteps, 300/260 Hz input drives, network seed 85 and input seeds
8501/8502. The local E-to-I weight is 0.5 with a 1 ms AMPA decay; inhibitory
decay is 9 ms. Both nominal coupling strengths are 0.08 with exact fan-in 8
and 2 ms delay. No parameter sweep, extra repetition or new operating point
has been introduced.

The uncoupled trajectory lasts 2 s with 300 ms discarded. Population rates use
1 ms Gaussian smoothing; excitatory volley detection retains the 15 ms minimum
separation and 10% prominence criterion. Phase is interpolated between volleys.
The original regularity, gamma-range, detuning, wrapping and one-I-spike-per-cycle
checks remain in analysis, rather than being mistaken for newly observed results.

Phase-response trials last 900 ms. Sampling remains 0.02–0.30 in increments of
0.02, then 0.40–0.90 in increments of 0.10. Positive shifts mean advances of the
next excitatory volley. Representative cases remain E/0.70, I/0.08 and I/0.12;
the strongest inhibitory delay is selected with the original strict comparison.
Reported phase fractions use rounded acquisition timesteps, not nominal inputs.

Coupling begins at 500 ms. The final 500 ms of valid interpolated phase defines
locking: absolute fitted drift below 0.25 cycles/s and circular concentration
above 0.95. The mechanism window remains −5 to +17 ms around the first arriving
source volley with a complete window. There is no across-seed aggregation or
uncertainty estimate: this remains a single seeded demonstration.

`experiments.exp085` continues to export the scientific constants, network
authoring, input construction, phase and rhythm helpers used by exp086 and its
tests. Those imports do not execute a stage or import the old publishing path.
The concurrently migrated exp086 package remains outside this task's ownership.

## Historical import and writing gate

The cached Gold-2 inventory dated 2026-08-23 contains 8,407 files, but **zero
exp085 files / zero exp085 bytes**. At the initial audit, local Pingstore had no
exp085 run. No shared upstream bank is required. This does not establish that no other
archive contains the experiment. Live R2 metadata has not been verified for an
exp085 source; no selective import can be proposed until a source is identified.

The former runner retained the authored network, an uncoupled summary trace,
figures and numerical summaries, but did not persist the full probe/pathway
recordings or the branching state. Archived figures alone cannot support
complete numerical reanalysis. Any later import must identify actual retained
evidence and original producer identity; a local historical run must not be
labelled as an HPC production run without evidence. There is no importer or
synthetic replacement for missing historical observations in this package.

The initial contract migration left the article and its creation date unchanged,
with `[≡ TXT]` matching successful discovery without exp085 data. The subsequent
authorized fresh execution and writing pass are recorded below. No historical
result or storage saving is claimed.

## Verification

Dedicated tests use synthetic recordings in temporary stores, not scientific
Pingstore runs. They exercise the full 49-call acquisition schedule, detached
branching state, exact input reuse, initialized weights, stage isolation,
source-neutral reservations, v2 rejection, full ancestry, payload/manifest
tampering, invalid layouts, malformed evidence, failure retention, refused
implicit resume, source mutation and atomic completion. Real plotting is tested
against synthetic analysis, including the network SVG and all five PNGs.

A separate three-timestep (0.3 ms) CPU executor interface test for each network/probe graph checks actual
recording/parameter names, JSON execution metadata and runtime-state roundtrip.
It is not a production simulation. The pre-migration runner was also replayed
against the same synthetic backend: the complete numerical record and all
five PNG figures matched exactly. This is a preservation check, not historical
scientific validation. Audit records are in `/tmp/pinglab-exp085-migration/`.

The contract-migration regression passed **240 tests** covering exp085, the existing exp086
scientific interface, Pingstore and writing inputs/status. A separate pass of
the concurrently migrated exp086 stage suite passed **25 tests**. The exp085
package passes Ruff and type checking. The tiny executor test caught and now
guards the runtime-weight orientation mismatch described above. Successful
local discovery then returned 33 present runs, none for exp085. At that gate,
the original article and existing scientific test file were byte-identical.

No production experiment or import was performed during that contract-only pass.

## Authorized fresh execution — 2026-08-28

Following the explicit request to rerun, the three commands above completed
independently with sources supplied explicitly. Existing validated helpers
allocated all IDs. These are **new local scientific runs**, not historical
imports or HPC runs. No upstream bank is needed.

| Run | Input | Files | Retained bytes including provenance |
| --- | --- | ---: | ---: |
| `exp085-r001-compute` | None | 258 | 525,288,907 |
| `exp085-r002-analyse` | `exp085-r001-compute` | 9 | 1,926,649 |
| `exp085-r003-present` | `exp085-r002-analyse` | 14 | 1,715,058 |

Total: **528,930,614 bytes**. Compute took 162 seconds, analysis 4 seconds and
presentation 2 seconds, measured from their recorded stage timestamps. All 49
simulation calls, full recordings, exact inputs and the shared runtime state
were retained. No subsampling or dtype conversion was introduced. Each run's
immutable `run.json` contains command, execution identity, source provenance,
timestamps and checksum-pinned inputs. Execution source hashes remained stable
through the run despite concurrent checkout edits.

The fresh numerical results are:

- Uncoupled frequencies: A **50.716 Hz**, B **46.707 Hz**, with seven phase wraps.
- Final drift (cycles/s): none **4.733829**, E-to-E **0.016489**, E-to-I
  **3.570827**, both **−0.004346**. Only E-to-E and both satisfy the preserved
  drift/concentration locking criteria.
- The illustrated first E-to-E correction advances the target volley by
  **0.9 ms**. I doublets occur at sampled phases approximately **0.102–0.141**.

Validated all three v3 layouts, payload checksums and complete ancestry. Present
numbers equal analysis results; discovery successfully exposes the present run.
There is no historical retained numerical source for comparison: synthetic
old/new parity and this new execution must not be described as an historical
reproduction check.

The article now has `[▦ DATA]`. Creation date `2026-08-19` is preserved;
`updated_at: 2026-08-28` records substantive caption and attribution corrections.
Results use headings, figures and concise captions, with probe reuse and the
single-seed limitation explicit. There was no Discussion section to remove.
The locking thresholds and mathematical symbols are defined. The Lowet
attribution was corrected after checking the
[original two-network study](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004072):
it varied E-to-E and E-to-I jointly; the pathway separation here is an adaptation,
not evidence that each pathway independently locked in the published study.
No scientific recipe or measurement was changed.

Isolated browser HTML and all six paged images were inspected against the fresh
presentation: six figures load, three equations render, axes retain their
intended ranges, and reported values match retained results. Fixed a duplicate
References heading and a caption semicolon consumed by Typst; the existing
synthetic stage test now compiles HTML/PDF and guards those render failures.
The preview uses a read-only validated catalogue copied into a temporary tree,
not the shared publication or materialization directories.

After the rerun and article fixes, **48 exp085/exp086 interface tests** and
**218 Pingstore/writing tests** passed. Ruff and package type checks passed.
The new HTML/PDF regression reuses the existing synthetic presentation fixture;
no additional production execution was needed for rendering fixes.

No training, historical import, archive mutation, materialization or publication
was performed. No shared source edits or ownership conflicts were encountered.
The article is **not marked Reviewed**. Commit/push and any publication remain
subject to separate authorization. Temporary execution, validation and rendering
evidence is in
`/var/folders/d7/m7d98tgd3hx0kxnm64vwrk_40000gn/T/pinglab-exp085-rerun-4fp93gf9/`.

## Writing Guide 17 conformance pass — 2026-08-28

Revised the title and abstract to lead with the supported pathway comparison.
The abstract is 93 words, uses interpolated observations and contains no citation;
the Lowet citation remains beside the literature comparison. Defined the model
and abbreviations, completed the reference's publication details, and added
normalization, population averaging and illustrative-probe selections to captions.
Methods now has an orientation and five action-led scientific operations, about
400 words excluding its displayed equation. Its phase-concentration equation
defines every symbol and is numbered in HTML and paged output. A local Typst
show rule preserves the equation number omitted by the current HTML exporter;
no shared renderer was changed. Existing results, figures and scientific caveats
were preserved, as were the creation date and today's existing update date.

The dedicated synthetic stage test now checks abstract length/citation exclusion,
Results without narrative paragraphs, the flat Methods list, HTML equation
numbering, and both populated and unavailable HTML/PDF views. All **42 exp085
tests** and **210 writing tests** passed after the exp054 owner corrected an
unrelated concurrent Typst error. Inspected all seven populated pages and the
unavailable page, plus browser figures, equation numbering and navigation.
Validated discovery still exposes `exp085-r003-present`; no other article declares
exp085 data. No scientific stages were rerun during this writing pass.

**Gap at that approval gate:** the immutable network schematic displayed
implementation labels such as `drive_A_300_Hz`, `time × batch × 128` and `coba_lif`.
These should become scientific labels through an exp085 presentation-only change
and a separately authorized new present run against the existing analysis.
No completed figure was edited, and no new presentation was silently generated.
The article remains unreviewed; no commit or publication was performed.

The live starting text, scoped diffs, validated catalogue and isolated renders
are retained under
`/var/folders/d7/m7d98tgd3hx0kxnm64vwrk_40000gn/T/pinglab-exp085-writing-bs_3b3vx/`.

## Authorized schematic relabeling — 2026-08-28

With approval for presentation only, completed **`exp085-r004-present`** from
the explicit existing `exp085-r002-analyse` input, retaining its unchanged
`exp085-r001-compute` ancestry. The existing validated allocator supplied the
identity. No compute or analysis stage ran. This closes the schematic-label
conformance gap above: labels and SVG tooltips now identify the drives, spike
channels and excitatory/inhibitory populations scientifically. The article
defines LIF as leaky integrate-and-fire. No shared renderer was modified.

Relabeling changes only SVG text and tooltip content; tests compare every
element, attribute and nontext value with the original generated drawing.
Unexpected labels fail before atomic completion. All five PNG figures and
`protocol.json` are byte-identical to `exp085-r003-present`; scientific fields
in `numbers.json` are identical, with only execution bookkeeping differing.
Revalidated all original manifests/payloads against their saved references;
the three original runs remain unchanged.

The new presentation retains **14 files / 1,469,879 bytes**, including provenance,
with payload digest
`sha256:b16e236dc05a58fb3d6f950e335c4c7e68feac27bd13df4cd9a79a03e19c43f9`.
This is additional presentation storage, not archive savings. **44 exp085 tests**
and **210 writing tests** passed, along with Ruff and package type checking.
Discovery validated the new run. Browser/paged inspection confirmed legible
schematic labels, all six figures loading, working navigation and numbered math.
`[▦ DATA]` and authored dates remain unchanged.

No completed runs, archive, publication selection or materialization directory
were changed. Nothing was staged or committed; Reviewed still requires approval.
Audit evidence is under
`/var/folders/d7/m7d98tgd3hx0kxnm64vwrk_40000gn/T/pinglab-exp085-relabel-4glr1apw/`.

## Future-run data retention — 2026-08-28

Each job explicitly selects the graph channels consumed by analysis. PRC voltage/conductance traces are kept only for the I-pulse doublet example. Pathway jobs retain E populations, plus target I spikes and the required conductances for the event-aligned mechanism. The shared prefix retains branching state but no trajectory file. Unused graph voltage/conductance channels and duplicate named spike recordings are not collected or written; exact inputs and parameters remain execution evidence.

These changes affect future execution only. Existing immutable runs and R2
archives are unchanged. Required arrays keep their original numerical values;
selected NPZ outputs use lossless compression. No production rerun or new
publication was performed for this cleanup.
