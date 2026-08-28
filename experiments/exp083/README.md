# exp083 — default PING response to input drive

## Contract migration

This experiment belongs to `demo`, not `gamma-gated-sparsity`. Its migration
follows Runner Guide 3.0.0 and Storage Guide 3.0.0. There is no trained upstream
bank, checkpoint selection, or training stage. The original combined runner is
retired and refuses execution; package imports preserve the constants and
measurement helpers previously used by exp084.

```sh
uv run python -m experiments.exp083.compute
uv run python -m experiments.exp083.analyse --source <exp083-compute-run-id>
uv run python -m experiments.exp083.present --source <exp083-analysis-run-id>
```

The corresponding `experiments/exp083/<stage>.py` script commands also work.
Only compute creates inputs and runs the graph. Analysis and presentation require
explicit completed v3 sources; there is no latest-run fallback, implicit upstream
execution, plot-only shortcut, materialization or publication. The contract-only
phase ran no production compute; the separately authorized local execution is
recorded below.

All stages use the shared allocator and atomic completion helpers. `--run-id`
accepts an unused source-neutral reservation allocated by those helpers before
dispatch; arbitrary IDs, completed runs and interrupted executions are rejected.
Failure leaves hidden temporary evidence for inspection, never a visible completed
run. Retry using a fresh reservation, not automatic recovery. No shared collection
or simulator changes are needed for this independent demo experiment.

## Retained evidence and stage boundaries

- **Compute:** retain one compiled `network.bundle` with its graph and manifest,
  plus all eight compressed condition files. Each holds the full input, E and I
  binary recordings: 10,000 timesteps, five trials, and respectively 128, 80 and
  20 channels. No trial, neuron or time sample is removed. Graph visualization
  moves to presentation; compute does not produce figures or numerical analyses.
- **Analyse:** pin compute, validate its recipe, grid, compiled graph digest and
  recording keys/shapes/dtypes, then calculate all 40 trial measurements and eight
  condition summaries. Retain full mean spectra for all eight rates and exact
  display-trial raster coordinates for the three original representative rates.
  Copy only the small graph description/manifest needed to draw the network.
  Large raw recordings remain in the compute source.
- **Present:** pin analysis and validate its complete compute ancestry. Draw the
  saved summaries, spectra, coordinates and graph without simulation, graph
  compilation, random input generation, frequency estimation or aggregation.
  The flat export preserves `network.svg`, `response.png`,
  `representative_rasters.png`, `spectra.png`, `protocol.json` and `numbers.json`,
  plus the shared presentation metadata projection.

Every source is validated using the shared v3 layout and checksum helpers, with
both its payload digest and authoritative manifest hash pinned. Ancestry is
checked before work and rechecked before completion, including the compute
ancestor during presentation. Wrong experiments/stages, unexpected input roles,
changed manifests, modified payloads and invalid scientific grids are rejected.
Historical imports will require a separately audited source-specific plan; this
contract does not authorize relabelling unidentified historical evidence.

## Preserved science

The sweep retains input rates 0, 25, 50, 75, 100, 125, 150 and 200 Hz/channel;
network seed 83; trial seeds 8300–8304; 1,000 ms duration; 0.1 ms timestep; and
200 ms transient exclusion. Each rate reuses the same trial RNG seeds and one
compiled default PING graph, without parameter tuning. The original float32
uniform-threshold input generator is unchanged.

Frequency measurement retains the named 5–80 Hz dominant-rhythm policy,
prominence threshold 3, band-edge rejection, and half-frequency correction at
30% relative power. Population rates retain means and sample SD (`ddof=1`);
frequency summaries use resolved trials only. Rhythmicity retains the original
autocorrelation lobe–trough contrast, median and IQR, including zero for silence.
E/I timing retains the strongest correlation within ±20 ms using 1 ms bins;
negative lag means I follows E under the preserved correlation ordering.

Representative rates remain 25, 75 and 150 Hz/channel, display trial 0. All
original plot labels, limits and half-IQR error bars are preserved. The spectra
plot still displays 20–100 Hz although the estimator searches 5–80 Hz; this can
hide lower-frequency fundamentals and must be considered during the later
science/writing review, not silently changed during contract migration.

## Historical-source audit — 2026-08-28

The cached Gold-2 inventory contains **zero exp083 files / zero bytes**, as do
its base and composite source inventories. Live R2 metadata was read without
modification and matched the cached files byte for byte:

| Metadata | Bytes | SHA-256 |
| --- | ---: | --- |
| `campaigns/gold-2/run.json` | 1,103 | `d4d067148589ae8469a37ee765c77282c9ab081eff22dfda0a24d596cbba913c` |
| `campaigns/gold-2/inventory.json` | 2,101,788 | `c7b9455968e34ac3be2df46a57ab4fc0ffcd94dc88799bf11b36c9b673a88f68` |

The live inventory likewise contains zero exp083 entries. A recursive read-only
listing of the documented standalone prefix `r2:pinglab/archive/exp083` returned
zero objects. These checks do not establish absence from every possible remote
location. No local completed exp083 run was found, and validated discovery found
no exp083 presentation. At that audit checkpoint the article was `[≡ TXT]`.

No data was imported, repacked, deleted, selected for publication or materialized.
Shared upstream bank requirements are **zero files / zero bytes**. An import
selection and retained-byte estimate cannot be supplied until the historical
recordings and producer provenance are located. No HPC producer identity should
be inferred: the original runner describes a bounded local experiment. The
existing article's findings are not verified by code or synthetic tests.

## Contract verification before local execution

Tests use synthetic recordings in isolated temporary stores, never production
simulation or the operational Pingstore. They cover recipe preservation, exact
trial/input retention, estimator replay, stage isolation, source-neutral
reservations, atomic completion, failure retention, schema/layout/checksum
rejection, manifest and ancestor mutation, invalid recording/display payloads,
CLI source requirements, flat presentation output and exp084's legacy interface.

An AST comparison against the live pre-migration source confirmed the input
generator and all four measurement/aggregation functions are unchanged. The
original dedicated tests and article are unchanged. Ruff and the exp083 type
check pass. The focused run including exp084 compatibility and shared staged
layout tests passed **85 tests**; the article's unavailable-data test also passed.
The final expanded regression passed **181 tests**, including the additional
failure/hidden-input probes, shared frequency estimators, writing inputs/status
and staged-layout checks. Final read-only discovery validated 33 presentation
runs across the shared store, with no exp083 presentation; other tasks added
runs during this work. No operational exp083 run was created.

Historical result comparison, selective import, actual-data browser HTML/paged
review, scientific claim revisions and author review remain pending. No Reviewed
status is assigned. No commit or push has been performed. Concurrent exp054,
exp080, exp082, exp084 and other experiment/shared-file edits are outside this
task's ownership and were not modified or staged.

## Authorized new local execution — 2026-08-28

After the historical-source audit, the author authorized a fresh execution.
These are **new local simulations and derived outputs, not historical imports**:

| Run | Independent operation | Command wall time | Complete run bytes |
| --- | --- | ---: | ---: |
| `exp083-r001-compute` | Full eight-condition, five-trial sweep | 17.96 s | 1,932,358 |
| `exp083-r002-analyse` | Measurements from the explicit compute input | 1.39 s | 1,172,286 |
| `exp083-r003-present` | Figures from the explicit analysis input | 2.27 s | 1,196,777 |

The three commands total **21.62 seconds** on the local Apple M5, excluding
subsequent verification and report rendering. Compute export is 1,044,457 bytes;
complete runs total 4,301,421 bytes including provenance. The full input, E and I
binary arrays are retained losslessly, without subsampling. Each downstream run
pins its source payload digest and authoritative manifest hash. All three runs
passed v3 layout, payload and ancestry validation; discovery includes the present
run. No exp083, simulator or measurement source changed, and all pre-existing
completed-run manifests were unchanged when checked. Later verification observed
a concurrent edit to the shared `tools/pingstore/presentation_inputs.py` helper,
outside these stages' execution path; it was left untouched. No materialization,
publication, commit or push occurred. The post-run focused regression passed
**94 tests**.

The circuit was silent at 0 and 25 Hz/channel. All five trials resolved a rhythm
at every tested drive from 50 through 200 Hz/channel; condition median frequency
rose from **14.9295 to 27.9617 Hz**, remaining below 30 Hz. E population mean rates
rose from 9.8094 to 24.8969 Hz across those active conditions. The retained
rhythmicity contrast medians were 1.0; median E/I peak lag was −1 ms. These are
new measurements, not a numerical comparison against missing historical data.

The article's badge is now `[▦ DATA]`, with no changes to its prose or authored
dates. An isolated selected-input preview compiled successfully to HTML and
three paged PNGs; all four figures and the rhythmicity equation were visually
checked in the paged output. HTML contains four images and four captions, but
interactive browser verification was blocked by the browser's local-file URL
policy and was not bypassed. Audit files are under
`/var/folders/d7/m7d98tgd3hx0kxnm64vwrk_40000gn/T/pinglab-exp083-run-4btpg99r/`.

Science/writing review remains pending: the sampled sweep bounds activation
between 25 and 50 Hz/channel but does not establish a mathematical discontinuity.
The preserved 20–100 Hz spectra display omits the 75 Hz/channel condition's
18.8794 Hz fundamental, so its caption needs review alongside the displayed range.
Neither issue was silently changed, and no Reviewed status was assigned.
