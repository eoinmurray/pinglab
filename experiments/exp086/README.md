# exp086 — intermittent phase attraction

## Scope and status

Contract migration follows Experiment Runner Guide 4.3.0 and Storage Guide 4.3.0.
This is a **demo** experiment, not a gamma-gated-sparsity collection member.
The migration does not add it to that collection. The author separately
authorized fresh local execution and omission of two unavailable theory
schematics; the completed stages are recorded below. Historical import,
materialization and publication remain unauthorized.

The original flat runner is retired. No historical exp086 run was found locally
or in the cached Gold-2 inventory (0 files, 0 bytes, also absent from its retained
base/composite inventories). Live R2 metadata has not been checked. The original article
claimed a completed demonstration, but its historical producer, recordings,
commands, hashes and figures have not been recovered. This does not establish
that the experiment was never executed.

There is no trained upstream bank. The scientific code imports exp085's public
network, input and measurement helpers. No exp085 execution is launched.
Exp085 is being migrated concurrently; this task owns none of its changes and
does not assume they will be committed with exp086. The public import names are
preserved. A scoped import-path setup also supports the pre-migration exp085
module, whose old simulator path does not work in a normal Python invocation.

## Independent commands

```sh
uv run python -m experiments.exp086.compute
uv run python -m experiments.exp086.analyse --source <exp086-compute-id>
uv run python -m experiments.exp086.present --source <exp086-analysis-id>
```

The stage files also support direct Python invocation. Each command completes
one immutable v4 run and prints its source-neutral ID. `--run-id` accepts only
an unused identity reserved through the existing stage allocator. There are no
science overrides, automatic input selection, upstream dispatch or publication.
Scientific execution remains local as in the original recipe; no HPC adapter
or training workflow is introduced.

| Stage | Retained outputs |
| --- | --- |
| Compute | Full fixed input spike trains; 500 ms uncoupled prefix spikes and portable runtime state; all nine exact graph bundles and suffix spike recordings |
| Analyse | Original per-condition summaries and selected intermediate; all arrays required by the three measured figures |
| Present | Network SVG from the retained graph; three measured PNGs; protocol and numbers JSON |

Compute exports have no figures or phase estimators. Analysis reads the explicit
compute run and does not simulate. Presentation reads saved arrays and selection,
does not remeasure, and copies no raw recordings. The compute ancestor supplies
the topology; analysis pins it, and presentation validates that ancestry.

All input roots, payload checksums, authoritative manifest hashes and transitive
pins are validated before consumption and rechecked before completion. Hidden
temporary runs remain incomplete on failure. Failed work is not silently resumed;
use a fresh reservation unless recovery is separately authorized. The shared
stage helper retains the actual command, code patch and execution provenance.

### Approved schematic omission

The original runner silently copied `coupling_regimes.svg` and
`intermittent_attraction.svg` from an old publication directory if present.
Those originals are unavailable. The initial contract pass required a validated
presentation input containing both originals. The author subsequently approved
omitting them and retaining the measured figures. The current presentation
contract (`exp086.presentation/v2`, within a v3 run) therefore needs only the
analysis input and exports four figures. It does not invent replacement diagrams
or claim that the original schematics were recovered.

## Preserved science

- Two networks, each 80 excitatory and 20 inhibitory cells, use exp085's exact
  topology, neuron parameters and fan-in rules. Drives are 300 and 260 Hz with
  fixed input seeds 8501/8502 and network seed 85; the timestep is 0.1 ms.
- The total realization lasts 5,000 ms. A single uncoupled 500 ms prefix supplies
  the shared dynamic branch point. Every branch gets a detached copy and the
  same remaining 4,500 ms of inputs. Both reciprocal E-to-E and E-to-I strengths
  take 0.08, 0.07, ..., 0.00 microSiemens with the original 2 ms delay.
- Phase measurements exclude the first 300 ms **after** the coupling decision.
  Population rates use 1 ms Gaussian smoothing. Volley detection uses 15 ms
  minimum spacing and 10% of the maximum rate as prominence. Phase interpolates
  between adjacent volleys; frequency is the reciprocal inter-volley interval.
- Relative velocity is `2*pi*(f_A-f_B)` in rad/s. Twenty-four phase bins define
  density and mean velocity; display velocity uses 8 ms Gaussian smoothing.
  Frequency summaries retain all detected suffix volleys, including those
  before the phase-analysis cutoff, exactly as the original implementation.
- The reported slip count remains `floor(abs(net_cycles) + 1e-9)`: it measures
  whole **net** phase windings, not every potentially reversing slip event.
- Intermediate selection excludes both endpoints, requires at least two net
  slips, and maximizes concentration × peak-to-mean density × nonnegative
  slowing fraction × exp(-alignment error). Ties retain the first condition
  in the descending coupling sweep. If none qualifies, analysis fails without
  invalidating the completed compute evidence.
- There is one trajectory per condition, no cross-seed aggregation or uncertainty
  estimate. No training, model selection or learned checkpoint is involved; the
  retained runtime state is the common branch point, not a trained model.

The numerical and plotting functions are moved without changing their bodies.
Raw binary spikes are losslessly stored as uint8 after validation. The original
trace writer omitted population rates, volley times and smoothed velocity;
analysis now retains these alongside the original phase arrays so presentation
can run independently. No trials, cells or time samples are subsampled.

## Writing and evidence boundaries

The article retains its creation date, 2026-08-19; substantive revisions set
`updated_at` to 2026-08-28. Read-only discovery and complete ancestry validation
now support `[▦ DATA]`. The article reports the fresh runs, not historical
reproduction. It has no Discussion section and has not been marked Reviewed.
Results use named subsections and concise captions under Writing Guide 17.0.0.
The exact net-winding definition and single-realization limitation are explicit.

At the end of the initial contract-only pass, no scientific execution, import or
presentation existed. Synthetic test fixtures were not evidence for article
findings. Subsequent approved execution and rendering are documented below.

## Verification

Dedicated tests cover independent stages, shared detached prefix state and exact
input reuse, saved analysis arrays, selection ties and absence of candidates,
source-neutral reservation, atomic visibility,
interruption refusal, corrupt payload/manifest/ancestry, v2 rejection and direct
CLI entrypoints. Stage fixtures invoke no production simulator; a separate
20-step test checks the real recording contract. The existing six
scientific tests remain unchanged.

The initial contract focused suite passed **193 tests**: exp086's **31** scientific/contract
checks, exp085's existing scientific checks, Pingstore tests and writing-status
tests. A separate in-memory compatibility run against the original committed
exp085 module passed **25 tests** (six subprocess CLI checks excluded because
they necessarily load the live checkout). The live CLI checks pass separately.
AST comparisons confirmed all 12 moved exp086 function bodies are unchanged;
at that point the original six-test file and article were byte-identical to the
starting files. The scientific test file remains unchanged.
Ruff lint/format and scoped whitespace checks pass.

A broader CLI-allowlist run found six failures outside this task: unrecognized
`--frequency-source` flags in exp033/046/054 and `--shard-index` flags in
exp037/042/082. Exp086's flags passed. No shared allowlist or other experiment
was changed to resolve these failures. Read-only discovery found no exp086
present run; the number of other local runs changed during concurrent work.

## Authorized fresh local execution — 2026-08-28

The author subsequently requested a fresh execution instead of historical import.
Compute and analysis completed independently, using the unchanged full recipe:

| Run | Operation | Files | Total bytes | Export bytes |
| --- | --- | ---: | ---: | ---: |
| `exp086-r002-compute` | New local simulation: prefix and all nine branches | 46 | 2,239,186 | 1,317,313 |
| `exp086-r003-analyse` | New analysis pinned to that compute run | 15 | 13,029,072 | 12,145,668 |

Compute ran from 18:34:52 to 18:38:21 UTC. Both runs retain their execution
commands, code patches and source-neutral identities through the shared v3
stage helpers. These are newly generated observations, not recovered historical
results, and establish no numerical agreement with the missing original run.

The first attempt, `.exp086-r001-compute.tmp`, failed after the uncoupled prefix:
the new spike writer incorrectly rejected auxiliary full-profile recordings.
The simulator returns voltage, conductance and additional named spike traces
alongside the four exposed populations. The fix explicitly retains every sample
of the four exposed spike populations and excludes those unused auxiliary
trajectories; the complete dynamic branch-point state remains separately saved.
The simulation and measurement recipe did not change. A real-simulator 20-step
regression now checks this output contract; the exp086 suite passed **32 tests**.
The failed directory is preserved unchanged, not reused or treated as evidence.

| Equal coupling, microSiemens | Whole net phase windings | Phase concentration |
| --- | ---: | ---: |
| 0.08 | 0 | 0.991668 |
| 0.06, selected intermediate | 3 | 0.784966 |
| 0.00 | 16 | 0.028095 |

The selected condition accumulated 3.981114 net cycles; the unchanged floor rule
reports three whole windings. Its peak phase density was 7.614196 times the mean,
with the slowest phase bin 0.261799 rad from the density peak. This is one fixed
input realization and does not establish reliability across seeds.

Validation checks the v3 payload and manifest pins, all nine recording shapes,
the retained prefix and inputs, and exact numerical replay of every summary
and all **126 saved analysis arrays**. Compute/analysis source files remained
unchanged during the retry. Concurrent edits to the unrelated presentation-input
resolver and its tests were recorded, not included as work owned by this task.

At the end of compute/analysis, no presentation had been executed because the two
original schematic SVGs were missing. The author then approved their omission.

Command logs, pre-execution source hashes, failure inventory, exact replay
verification and read-only discovery output are retained in
`.r2/exp086-execution-m7lp1o83/`. These audit files supplement the immutable runs;
they are not additional scientific runs or publication inputs.

## Authorized presentation and writing review — 2026-08-28

`exp086-r004-present` completed independently from `exp086-r003-analyse`, which
pins `exp086-r002-compute`. It contains **12 files, 1,328,895 total bytes**;
its flat export contains seven files and **661,604 bytes**. Four are figures:
`network.svg`, `uncoupled.png`, `coupling_regimes_measured.png` and
`intermittent_attraction_measured.png`. The other files are numbers, protocol
and bookkeeping. No measurements were recomputed or raw recordings copied.

The three completed stages total **16,597,153 bytes**, including complete
provenance; exports total **14,124,585 bytes**. The preserved failed attempt is
separate (1,775,723 bytes). There is no historical-import saving to claim: no
archive bytes were imported or deleted, and historical numerical agreement
cannot be established. There are no upstream data banks or downstream article
consumers for this experiment; exp085 is a code dependency only.

Presentation identity pins:

- Payload: `sha256:0f798f5f072d9b1a58a22f5b4829d06d2b04423e94bb5ffb7cf01db3858c570f`
- Manifest SHA-256: `28f7778497cdc30cdf2e5169488ecbf49d157d24418e988b9b373d5e0c4374e6`

The article now derives starting frequencies, detuning and concentration from
retained results. The old claim of exact frequency cancellation is replaced by
the measured 0.10 Hz residual under strong coupling. Three whole net windings
at the selected coupling are not three individually counted slip events.
The recipes, selection score and all scientific function bodies are unchanged.

Browser and five-page Typst review found and corrected missing image alt text,
duplicate References, broken section links, mean-frequency notation rendered
as absolute-value bars, and distorted standalone HTML image ratios. Mean
frequency now uses a MathML-supported macron accent; figure sizing is scoped to
this article. Regression coverage renders both formats, checks four figures,
mean accents, link targets, a single References heading and explicit run
selection; missing inputs show a notice and corrupt selected JSON fails.

Final checks passed **32 exp086 tests** and **190 integration tests**
(Pingstore, writing status and the original exp085 scientific tests). Ruff lint,
format and scoped whitespace checks passed. The full repository suite and a
site-wide build were not run; no shared-file fix was required.

The fresh numerical results support the reported qualitative demonstration.
Primary-source checks confirm the general intermittent-synchronization account
in [Lowet 2017](https://elifesciences.org/articles/26642) and the 80-E/20-I,
eight-afferent, conductance-normalized setup in
[Lowet 2015](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004072).
The article's approximate macaque ranges (detuning, interaction amplitude and
phase locking) are preserved but their individual figure-level endpoints have
not been independently re-established in this pass. They remain a writing-review
qualification, not calibrated targets or evidence of quantitative replication.
The present LIF model is not the paper's Hodgkin-Huxley model.

Audit output is in `.r2/exp086-presentation-ov491ill/`: validated projection,
run verification, logs, article HTML and paged PNGs. The standalone review uses
copies of the four article/helper sources and redirects only the scratch
projection import; all figures are read directly from the validated present run.
No shared rendering source, shared configuration or completed run was edited.
No output was materialized or published. Nothing was staged or committed.
