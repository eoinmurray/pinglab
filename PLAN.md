# Gamma-gated-sparsity refractory-period plan

Date: 2026-09-08

Establish **1.2 ms excitatory / 0.6 ms inhibitory refractory periods** as explicit parameters of the gamma-gated-sparsity collection. Preserve existing results that already used these durations, replace the inconsistent timestep conditions, and align the separate theoretical and reference calculations.

**Refractory reconciliation complete — 2026-09-09.** Steps 1–4, 5.1 and 5.3–5.10, the retained-experiment review in step 6, and steps 7–8 are complete; 5.2 was skipped at the author's direction. All twelve replacement training cells completed on HPC, the complete 90-reused/12-new bank is retained locally and on HPC, and the replacement evaluations, theory, presentations and affected writings are reconciled. The final audit found no outstanding refractory repair or scientific rerun. Author inspection of localhost article layouts remains separate, as do exp110's unfinished manuscript scaffolds. **exp048 and exp111 are excluded.** The live collection includes exp110; experiments outside the collection are not part of the rerun scope. Earlier dated progress entries below are retained as history, not current status.

## 1. Evidence and preservation baseline

**Status: complete — 2026-09-08.** Inventory and provenance baseline established; execution-equivalence approval remains in step 4.

The audited legacy simulator uses fixed counters of 12 E / 6 I timesteps. At dt = 0.1 ms these already implement the proposed 1.2/0.6-ms model. At other timesteps the physical durations differ.

The retained bank, `exp022-r001-compute`, contains:

| Training timestep | Models | Existing E/I refractory durations | Planned treatment |
|---:|---:|---:|---|
| 0.05 ms | 3 | 0.6 / 0.3 ms | Replace |
| 0.1 ms | 90 | 1.2 / 0.6 ms | Preserve |
| 0.25 ms | 3 | 3 / 1.5 ms | Replace or replace grid condition |
| 0.5 ms | 3 | 6 / 3 ms | Replace or replace grid condition |
| 1 ms | 3 | 12 / 6 ms | Replace or replace grid condition |

The 90 models are eligible for reuse after forward-execution and provenance validation; this includes the three 0.1-ms timestep-sweep models. Existing fixed-0.1-ms spiking evidence can remain scientifically usable under the adopted model, subject to the equivalence checks below. Adoption does not establish equivalence to a 3/1.5-ms model or claim either pair is a universal physiological constant.

Keep completed runs and their export bytes immutable. Record reused versus newly generated evidence explicitly; do not relabel older 0.25-ms runs as having used 1.2/0.6 ms.

### Step 1 results

- [x] Validate the retained bank's v4 layout and full export checksum.
- [x] Inventory all 102 cells and confirm both checkpoint roles for every cell.
- [x] Check cell identities, seeds, timesteps and original source revisions against retained configurations.
- [x] Record the exact 90-model reuse-candidate / 12-model replacement partition below.
- [x] Preserve the original bank and its provenance without modifying run contents.

The read-only check used `tools.pingstore.contracts.validate_operational_run_directory` and the shared export resolver. It passed for `.pingstore/runs/exp022-r001-compute`: **408 files, 2,017,164,932 bytes**, comprising 102 cell directories, each containing `config.json`, `metrics.json`, `weights.pth` and `weights_final.pth`. These checkpoint roles remain distinct. The verified export identity is:

```text
sha256:9e3c93df9541809d1d019fe5290afbf7dff7d07ec14b07160fabe7ad79c9a0a8
```

For all 102 cells, the directory name agrees with `training_cell_name` and the seed suffix; retained `dt` and `seed` agree with the resolved training arguments. The original source revision agrees with `git_sha`, using `imported_cell_provenance.repository_commit` for inherited cells and `campaign_repository_commit` for retrained cells. Do not mistake the enclosing repair campaign or the local import commit for the training revision of an inherited model.

| Original training lineage | Full source revision | Bank cells | Reuse candidates | Replacements |
|---|---|---:|---:|---:|
| Inherited Gold-2 training | `4ad223d32620dd9f03698b89f28aedfe944d43ac` | 60 | 48 | 12 |
| Firing-rate repair training | `ac6f49884084811e3e05d49e8b45735d514ff245` | 42 | 42 | 0 |
| **Total** | | **102** | **90** | **12** |

The authoritative run record describes this bank's operation as a **local import**, with the historical scientific execution on **Slurm**. It has no operational inputs and is self-contained; the former source identity `exp022-gold-2-repaired-slurm` remains historical ancestry. Keep that distinction in any replacement bank's history.

### Complete cell inventory

Each row represents exactly three retained cells: append `__seed42`, `__seed43` and `__seed44` to the listed prefix. Together these 34 rows identify all 102 cells in the checksum-pinned bank.

| Cell prefix | dt (ms) | Original training revision | Treatment |
|---|---:|---|---|
| `coba__canonical` | 0.1 | `4ad223d3` | Reuse candidate |
| `coba__off` | 0.1 | `4ad223d3` | Reuse candidate |
| `coba__rt10hz` | 0.1 | `ac6f4988` | Reuse candidate |
| `coba__rt1hz` | 0.1 | `ac6f4988` | Reuse candidate |
| `coba__rt25hz` | 0.1 | `ac6f4988` | Reuse candidate |
| `coba__rt2p5hz` | 0.1 | `ac6f4988` | Reuse candidate |
| `coba__rt5hz` | 0.1 | `ac6f4988` | Reuse candidate |
| `frozen_ping` | 0.1 | `4ad223d3` | Reuse candidate |
| `ping__canonical` | 0.1 | `4ad223d3` | Reuse candidate |
| `ping__dt0p05` | 0.05 | `4ad223d3` | Replace |
| `ping__dt0p1` | 0.1 | `4ad223d3` | Reuse candidate |
| `ping__dt0p25` | 0.25 | `4ad223d3` | Replace |
| `ping__dt0p5` | 0.5 | `4ad223d3` | Replace |
| `ping__dt1` | 1 | `4ad223d3` | Replace |
| `ping__low_w_in__win0p05` | 0.1 | `ac6f4988` | Reuse candidate |
| `ping__low_w_in__win0p1` | 0.1 | `ac6f4988` | Reuse candidate |
| `ping__low_w_in__win0p3` | 0.1 | `ac6f4988` | Reuse candidate |
| `ping__low_w_in__win0p9` | 0.1 | `ac6f4988` | Reuse candidate |
| `ping__off` | 0.1 | `4ad223d3` | Reuse candidate |
| `ping__rt10hz` | 0.1 | `ac6f4988` | Reuse candidate |
| `ping__rt1hz` | 0.1 | `ac6f4988` | Reuse candidate |
| `ping__rt25hz` | 0.1 | `ac6f4988` | Reuse candidate |
| `ping__rt2p5hz` | 0.1 | `ac6f4988` | Reuse candidate |
| `ping__rt5hz` | 0.1 | `ac6f4988` | Reuse candidate |
| `ping__tg12` | 0.1 | `4ad223d3` | Reuse candidate |
| `ping__tg18` | 0.1 | `4ad223d3` | Reuse candidate |
| `ping__tg27` | 0.1 | `4ad223d3` | Reuse candidate |
| `ping__tg4p5` | 0.1 | `4ad223d3` | Reuse candidate |
| `ping__tg6` | 0.1 | `4ad223d3` | Reuse candidate |
| `ping__tg9` | 0.1 | `4ad223d3` | Reuse candidate |
| `ping__variable_rate` | 0.1 | `4ad223d3` | Reuse candidate |
| `trainable_ping_init` | 0.1 | `4ad223d3` | Reuse candidate |
| `trainable_small_init` | 0.1 | `4ad223d3` | Reuse candidate |
| `trainable_zero_init` | 0.1 | `4ad223d3` | Reuse candidate |

Family totals are canonical 6, activity frontier 36, initialization 12, timestep 15, low-input-weight 12, GABA-timescale 18 and variable-rate 3. Only the four non-0.1-ms timestep groups enter the replacement set; the chosen replacement grid is recorded in step 2.

**Interpretation and remaining boundary:** none of the 102 retained configurations explicitly declares E/I refractory durations. The durations in the baseline table come from the earlier simulator-source audit and each cell's timestep, not from a saved refractory declaration or newly measured interspike intervals. A source cross-check confirms that the current simulator and the locally available original `4ad223d3` training source pass fixed 12/6 counters through the production path; the prior audit's check of the repair revision is not a new historical-source verification in this step. Matching counters does not itself prove unchanged execution. Step 4 must still establish full-network equivalence before reuse is accepted.

This step changed only this plan. It did not change the simulator, configurations, checkpoints, run records, source pins or article tags, and did not execute training, inference, analysis, presentation or HPC jobs.

## 2. Resolve the timestep grid

**Status: complete — 2026-09-08.** Adopt the recommended exact-refractory grid. This completes the design decision; applying it to execution and analysis code remains in subsequent steps.

- [x] Choose the sweep representation before training replacements.
- [x] Verify integer refractory counters at all five timesteps using exact rational arithmetic.
- [x] Record replacement identities, presentation-duration handling and dependent analysis changes.

**Selected grid:** `0.05, 0.1, 0.2, 0.3, 0.6 ms`, with seeds **42, 43 and 44** at each condition. All five timesteps represent both **1.2-ms E and 0.6-ms I** refractory periods exactly in integer steps.

| Selected timestep | E counter | I counter | Trial steps | Realized trial duration | Training required |
|---:|---:|---:|---:|---:|---|
| 0.05 ms | 24 | 12 | 4,000 | 200 ms | Three replacement models |
| 0.1 ms | 12 | 6 | 2,000 | 200 ms | Three existing reuse candidates |
| 0.2 ms | 6 | 3 | 1,000 | 200 ms | Three new models |
| 0.3 ms | 4 | 2 | 666 | 199.8 ms | Three new models |
| 0.6 ms | 2 | 1 | 333 | 199.8 ms | Three new models |

This keeps a five-condition, three-seed sweep: **12 training runs and three conditional reuses**, preserving the full bank's 90/12 partition. The selected sweep spans **twelvefold**, so replace old twentyfold-range claims and the old coarse-condition axis labels when regenerating results. The former 0.25/0.5/1-ms grid is not selected: it cannot represent both target refractory durations exactly.

### Replacement identities

For each seed in 42, 43 and 44:

| Old cell prefix | New bank cell prefix | Treatment |
|---|---|---|
| `ping__dt0p05` | `ping__dt0p05` | Retrain at 0.05 ms with corrected counters; same cell name in a new run |
| `ping__dt0p1` | `ping__dt0p1` | Reuse only after step 4 equivalence checks |
| `ping__dt0p25` | `ping__dt0p2` | Fresh training at 0.2 ms |
| `ping__dt0p5` | `ping__dt0p3` | Fresh training at 0.3 ms |
| `ping__dt1` | `ping__dt0p6` | Fresh training at 0.6 ms |

These are replacements in the experimental design, not renamed old checkpoints. Preserve the historical bank unchanged. Update exp022's grid and recipe identities, exp044's derived grid and validators, and downstream figures and writings to use the selected values.

### Numerical conventions to implement

1. **Refractory counters:** use a tolerance-checked integer conversion of the physical duration divided by dt. Do not use bare truncation: binary floating-point arithmetic gives `int(1.2 / 0.1) == 11`, although the required E counter is 12; the same issue affects 0.05 and 0.2 ms. Round to the nearest integer, verify that counter × dt reproduces the requested duration within numerical tolerance, and reject an unrepresentable collection condition. Apply the same counters and release convention in production and independent reference implementations.
2. **Trial duration:** keep the nominal presentation request at **200 ms** and preserve production training's whole-step, downward conversion. Handle ratios numerically close to integers correctly. At 0.3/0.6 ms this gives the 199.8-ms trials shown above. Record nominal duration, step count and realized duration; use the realized duration for rates and time axes. Apply this convention consistently to training, evaluation, snapshot validation and any independent reference comparisons. In particular, current `tools/snnsim/config.py` truncates while `experiments/exp044/evidence.py` rounds, producing a one-step mismatch at 0.3 ms that must be corrected before execution.
3. **Analysis bins:** make frequency analysis use physical time consistently. Current `experiments/exp022/analyse.py` groups `round(1 / dt)` steps but treats every bin as 1 ms; the new coarse conditions instead produce 0.9-ms and 1.2-ms bins. Correct the realized bin interval in smoothing and frequency calculations, or use physical-time binning; audit shared rhythmicity helpers for the same assumption. Exp044 rate normalization must likewise use realized trial duration.

The 0.1-ms condition retains its existing 2,000-step presentation, 12/6 refractory counters and 1-ms analysis bins. These design checks do not replace step 4's execution tests. No code, checkpoints or runs were changed for step 2.

## 3. Make the collection model explicit

**Status: complete — 2026-09-08.** Implementation and scoped integration checks completed. This status does not approve retained-model reuse; that requires step 4.

- [x] Define collection-owned refractory constants in `experiments/helpers/operating_point.py`, keeping collection choices separate from generic simulator defaults.
- [x] Thread physical durations through model construction, training, simulation, inference, configuration saving/loading and execution provenance.
- [x] Derive counters from the requested physical durations and the actual runtime timestep; pass those counters into both E and I updates. Remove reliance on stale module-level counters.
- [x] Implement step 2's consistent duration-to-step conversion and record nominal and realized durations throughout the new timestep conditions.
- [x] Preserve the implementation's voltage integration, spike/reset ordering, recurrent delays, input encoding/draws, RNG ordering and backend selection at dt = 0.1 ms; verify complete before/after numerical equivalence separately in step 4.
- [x] Handle retained configurations deliberately: they lack refractory fields. Collection consumers explicitly resolve 1.2/0.6 ms rather than fall back silently to generic defaults. Historical configurations are not reinterpreted globally.
- [x] Make graph-facing physical-duration handling explicit where used by collection checks, without adding noncollection experiment reruns.
- [x] Include the resolved refractory parameters in scientific recipe validation and new execution metadata.

Before editing execution or storage code, follow the current Experiment Runner Guide and Storage Guide. Keep bank-reuse support specific to exp022; do not introduce a general storage merger.

### Step 3 results

**Model and execution.** The collection now passes `--refractory-e-ms 1.2 --refractory-i-ms 0.6 --refractory-policy exact` explicitly, including when loading retained checkpoints. `tools/snnsim/timing.py` performs tolerance-checked conversion at the actual runtime dt; `COBANet` passes the resolved counters into both population updates. Generic simulator defaults remain 3/1.5 ms with nearest-step conversion. Unsupported collection timesteps, including 0.25 ms, fail rather than silently round the requested refractory periods. Training, simulation, inference, snapshots, probes and weight-dump construction all receive the parameters. Saved configurations and execution metrics include physical durations, policy, counters and realized refractory durations; CLI overrides take precedence over loaded configuration fields.

**Selected grid and physical time.** Exp022 now registers `0.05, 0.1, 0.2, 0.3, 0.6 ms`; exp044 derives the same grid. The nominal 200-ms request produces exactly the counters, trial lengths and realized durations recorded in step 2. The simulator and collection validators use matching floor-with-integer-tolerance rules. Rates use actual elapsed duration, including supplied-input probes. Exp022 spectra and shared autocorrelation measurements use the realized bin interval; exp044 raster rates and time limits use the recorded trial length. New measurement recipes identify these conventions.

**Historical evidence and recipe versions.** New spiking recipe versions declare the model explicitly while validators retain their historical versions. Exp022 analysis/presentation can resolve either complete 102-cell registry without renaming old cells. New-grid diagnostics reject an incompatible old bank before launching any simulation. Exp044 requires explicit refractory declarations for new non-0.1-ms training cells; missing fields remain permissible only for retained 0.1-ms reuse candidates, whose scientific reuse still depends on step 4. Exp082 showcases now have a separate v2 recipe, with v1 showcases still readable; its fixed historical import contract requires the frozen original v1 recipe.

Exp033's historical mean-field configuration is now literal and independent of changing defaults; exp054's existing recipes bind that frozen theory. **The theoretical 3/1.5-ms model has not yet been changed or recomputed.** Its adoption and exp054's separate replacement-theory input remain step 5. The legacy graph adapter now resolves its authored refractory counters at the graph's dt, and graph execution rejects conflicting CLI refractory overrides. No active collection experiment requires a new graph-authored model after exp111's removal.

### Step 3 validation

- **883 tests passed** across two integrated suites: 524 collection/helper tests, plus 359 simulator/exp082 tests. The simulator suite covered model updates, configuration, LIF primitives, CLI forwarding, inference, training, physical-time metrics and graph numerical conformance; 92 slow tests were deselected. After the final supplied-input nominal-duration correction, its three focused checks also passed.
- Strong-drive public-network tests enforce the expected E and I interspike intervals at all five selected timesteps with both integrators, even with the obsolete module counters deliberately set to 999. Duration and synthetic-rhythm checks cover the 0.3/0.6-ms conditions.
- A CPU `aot_eager` compiled-step smoke comparison at 0.1 ms matched eager spikes, voltage, conductance, readout and available gradients exactly over 24 steps. This is **not** a CUDA/Inductor check or retained-bank before/after equivalence. Those broader checks and independent Brian2 scheduling remain in step 4.
- Ruff passed for every changed implementation/test file in this step; `git diff --check` passed.
- Read-only `uv run pingstore discover` succeeded, reporting **79 validated present runs**. Retained compute recipes passed the applicable historical readers, including both existing exp082 showcases. The affected articles and dependent exp110 still have qualifying local presentation data, so their availability tags remain `data`; author-assigned `reviewed` tags and guide-version tags were preserved. Full article revisions and current-guide conformance remain step 7.
- **Two existing article checks remain failing**, separately reproduced and excluded from the 524-test run: exp022's `test_every_registered_training_run_has_guide_and_results_sections` expects absent `TR-*` heading patterns, and exp037's `test_reviewed_article_structure_and_scientific_caveats` expects the older 2026-09-02 article date. Neither is a refractory implementation failure; their article text/tests were not changed to suppress them. The collection suite also emitted 17 existing Matplotlib `tight_layout` warnings.

No scientific training, HPC submission, experiment compute/analyse/present rerun, stored-run mutation, source repinning, commit or push was performed for step 3. The original bank remains the baseline. The narrow 90-reused/12-new bank assembly path is still unimplemented and belongs to step 5; **step 4 is next**.

## 4. Verify preservation and enforcement

**Status: complete — 2026-09-08.** All local preservation, reference, timing and provenance checks passed, followed by all five CUDA cases in CSD3 validation job **35088396**. The 90 dt = 0.1-ms models are accepted for reuse with their original provenance; the other twelve training cells remain replacements.

- [x] Compare actual full-network execution before and after the change at dt = 0.1 ms using identical weights, inputs and seeds. Check E/I spikes and readout outputs, and a representative training forward/backward pass.
- [x] Test strong-drive refractory enforcement for both populations across the chosen timesteps, including post-spike voltage release. Sparse firing alone cannot demonstrate correct enforcement.
- [x] Cover eager and compiled execution and configuration save/load round trips. Direct primitive tests alone missed the original wiring defect.
- [x] Verify consistent counter conversion and release scheduling in the independent Brian2 comparisons and relevant graph checks.
- [x] Check the new 0.3/0.6-ms conditions for consistent trial lengths, rate normalization and frequency-bin timing across production, analysis and reference implementations.
- [x] Validate all reused sources through shared v4 layout/digest checks and preserve their explicit input references.

Establish the dt = 0.1-ms equivalence before relying on retained models and measurements as the preserved baseline.

### Step 4 preservation results

The opt-in regression in `tools/snnsim/tests/models/test_refractory_preservation.py` loads the unmodified pre-adoption `models.py` and `config.py` from commit **`255ab3b6e11fe4e92cd5f96c1634a2a58150bb05`** into isolated Python modules. It compares those implementations with the explicit-1.2/0.6-ms implementation using the checksum-pinned retained bank. This isolates the refractory change; it does not claim to recreate the original HPC training environment or replay every historical training epoch.

- **All 90 reuse candidates passed in both checkpoint roles: 180 checkpoint comparisons.** Each uses the full saved 784-input / 1024-E / 256-I / 10-output network and a complete **200-ms, 2,000-step** trial. The inputs are cached MNIST test images at indices 0/1/2 for training seeds 42/43/44, with encoder seed 20260415 and a fixed 25-Hz maximum pixel rate. This is a controlled equivalence input, including for the variable-rate-trained models, rather than a new accuracy estimate.
- Under the production reset-state convention, initialized parameters, every recorded E/I spike, E/I voltage and conductance trace, readout trajectory, final logits, population rates and final Torch RNG state were **bit-for-bit identical** before and after the change. Strict checkpoint loading passed.
- **Four complete two-image training forward/backward comparisons passed:** canonical COBA, canonical PING, PING with the 5-Hz rate penalty, and trainable PING initialization, all at seed 42 using final-epoch checkpoints. The combined objective (cross-entropy plus the applicable rate penalty), logits and every available parameter gradient matched exactly and remained finite. Each model had at least one nonzero parameter gradient. No optimizer updates or checkpoint writes were performed.
- A supplementary pass using seeded randomized initial membrane states also passed all 180 checkpoint comparisons and the same four backward checks. The production reset-state suite passed **6 tests in 147.13 s**; the randomized-state suite passed **6 tests in 146.29 s**.

Both source validation and the post-test `check_unchanged()` passed. The 90 cells can be preserved with respect to the tested refractory implementation change; the twelve other-timestep cells remain replacements. These checks do not establish model invariance across arbitrary timesteps or bitwise equivalence between different hardware backends.

Reproduce the production-state check with:

```sh
PINGLAB_REFRACTORY_BANK_CHECK=1 PINGLAB_NO_COMPILE=1 \
  uv run pytest -q tools/snnsim/tests/models/test_refractory_preservation.py
```

Add `PINGLAB_REFRACTORY_RANDOMIZED_CHECK=1` for the supplementary initial-state case. The tests require the explicitly pinned local bank and existing MNIST cache; they never download data or choose a different bank.

| Implementation file | SHA-256 tested |
|---|---|
| Baseline `models.py` | `110281eb28c4a7b993056d7294d7f8ce9d5ef79351c3648b130c7ed009e0b9aa` |
| Baseline `config.py` | `5233e98870cbb41eb30601cf84d1b352438e8c5992fdef2af355cccec8a47286` |
| Adopted `models.py` | `0fd68e4082eb5fd0cdd2db062c097232f4f6908351215d5be6376a87d64a9101` |
| Adopted `config.py` | `fbc210cebeaaa0caf52bdb4f1538d19c6e33ee091b9aa1747bb10af4bc2d89fd` |
| Adopted `timing.py` | `7fc6e7a8bbf823750f7c4f6c66cb5f0b0d53d286046d02e20c079013dcc84ea4` |

### Step 4 independent and compiled checks

- The integrated local verification suite passed **137 tests**, with five CUDA cases skipped because the local Mac has no CUDA device. This includes CLI/save-load precedence and metadata, both-population refractory enforcement, physical-time measurements, Brian2 and graph comparisons. Ruff and `git diff --check` passed for the step's changes.
- Public-forward strong-drive tests and pulse-release tests cover E and I at every selected dt under both integrators. After a spike, voltage is held at reset through step `N − 1` and first integrates again on step `N`, where `N` is the requested population's integer refractory count. Strong-drive spike intervals equal those counts.
- **Native Brian2 2.9.0:** an uncoupled 4-E/1-I production COBANet with externally held conductances and independent Brian2 neurons agree on all spike steps at all five timesteps, under both moderate and saturating drive. Voltage traces agree within **0.0002 mV**, comparing production float32 with Brian2 float64. Independent rational arithmetic gives 666/333 steps and 199.8 ms for the coarse trials. These are discrete-scheduling checks, not a continuous-time convergence claim.
- The existing collection-labelled recurrent Brian2 GABA-sweep test still used 3/1.5 ms. Its reference parameters were corrected explicitly to **1.2/0.6 ms**, and all six GABA conditions at **dt = 0.1 ms** passed the existing acceptance criteria. Generic reference defaults remain independent.
- Graph/legacy comparisons span all five timesteps with independently calculated physical refractory counters and explicit one-update recurrent delays. Spikes, voltage and conductance trajectories match exactly; logits meet the existing tolerance. Historical graph-authoring defaults were preserved.
- **CPU Inductor:** five actual compiled public-network forward/backward comparisons passed, one per dt, with `fullgraph=True`. Eager execution is explicitly kept eager. Spikes match exactly; continuous traces and readouts use `rtol=atol=1e-5`; all six trainable weight gradients use `rtol=1e-4, atol=1e-5`. There is no fallback to an eager backend masquerading as compilation.
- **CSD3 CUDA:** all five corresponding GPU cases passed in **102.68 s** in interruptible diagnostic job **35088396**, on an **NVIDIA A100-SXM4-80GB**, with **Torch 2.11.0+cu128 / CUDA 12.8 / Python 3.10.20**. JUnit records five tests, zero failures/errors and zero skips. Slurm confirms **COMPLETED, exit 0:0, 6 min 50 s total elapsed**, including environment setup, with batch MaxRSS 4,460,332 KiB. The job requested one GPU and 15 minutes and used a separate frozen source snapshot and isolated environment. Snapshot archive SHA-256: `30854d2c0137c91fcadf353be2acb729debd7a99672e484f1c3f5eb10a0d0314`; validated lockfile SHA-256: `44df18b40d04655e4edf520d4802fef5c52c517477a6075ea194cc5521644dc7`. Spikes match exactly and continuous values/gradients meet the same tolerances as the CPU Inductor tests. The sole warning notes that optional TF32 acceleration is disabled; precision settings were not relaxed to obtain a pass.

Across the integrated local suite, both retained-bank initial-state suites and the GPU suite, **154 tests passed**. The five local CUDA skips were all exercised successfully on CSD3. Only validation tests and this plan changed in step 4; no production simulator defect was found. No scientific reruns, replacement training, bank assembly, source repinning, publication, commit or push was performed. **Step 5 is next.**

### Step 4 retained-source checks

Read-only validation covered **153 visible v4 runs and 258 input references**. The eighteen active collection members account for **106 runs, 186 references and 4,027,098,785 export bytes**. Discovery reports 79 present runs in total, including 61 for active collection members. No source was repinned or selected.

The original exp022 bank remains **102 cells / 408 files / 2,017,164,932 bytes**. All **180 reusable checkpoints** and **24 replacement-cell checkpoints** passed their separate best-validation/final-epoch role checks. The bank's export digest, `run.json` and README bytes remained unchanged. Original training lineage remains 48 reuse candidates plus 12 replacements from `4ad223d3`, and 42 reuse candidates from `ac6f4988`.

| Explicit source | Validated payload digest |
|---|---|
| `exp022-r001-compute` | `sha256:9e3c93df9541809d1d019fe5290afbf7dff7d07ec14b07160fabe7ad79c9a0a8` |
| `exp041-r001-compute` | `sha256:8c851d5f8510c96d90ffd645bd68b88821c6be17bbe09a04b140ac72e29c6466` |
| `exp041-r002-analyse` | `sha256:bbf666b78e993f512db12bbb3ff45ed85857b3ef72772bc584bd8c0f6fe09d99` |
| `exp044-r002-compute` | `sha256:a5be3d1e5c53d3159a33b77fd35261e9cf9a00bef9c1f1b1ec885b4eb9dac3e7` |
| `exp054-r008-compute` | `sha256:20a28f14c6c58bebd3729f4b8de5f15c59e32b8374876108886a0198fabc6a66` |
| `exp054-r009-analyse` | `sha256:75da549448c13d65b2531189a25af908c9845e7cb9ddec6300ddb1a5809f0e50` |
| `exp033-r001-compute` | `sha256:d1b4c971440b52b556d50a8b84548edc2513a86cfb978b2ce2c86202bfd478ea` |

Exp041's retained chain validates all 18 conditions/checkpoints at dt = 0.1 ms. All **136 native raster payloads** in exp054-r008 validate at 1024 E / 256 I, dt = 0.1 ms and GABA = 6 ms, retaining the 100–1000-ms recording interval. Grouping events separately by neuron and trial gives:

| Population | Within-neuron ISI pairs checked | Minimum observed ISI | Below requested refractory |
|---|---:|---:|---:|
| E | 4,635,175 | 13 steps = 1.3 ms | 0 below 12 steps |
| I | 4,798,061 | 7 steps = 0.7 ms | 0 below 6 steps |

Thus **9,433,236 recorded intervals contain no violations**. Observed minima need not equal the refractory limits: a neuron must also reach threshold after voltage integration resumes. These recordings alone do not measure the exact voltage-release step; the independent strong-drive and voltage-release checks establish that convention.

Two pre-existing follow-ups belong in step 7: exp081 has qualifying local presentation data but lacks its separate `data`/`txt` tag; the helper's older `F_GAMMA_HZ = 43.95` reference needs reconciliation with the retained exp041 6-ms median of **59.136573 Hz**, including estimator/source identity before changing it. No article or scientific operating-point value was changed during this verification step.

## 5. Required runs and stage changes

**Status: 5.1, 5.3 and 5.4 complete; 5.2 skipped by the author; 5.5–5.10 not started.** Work through **5.1–5.10 one at a time**, on the author's request. A request for one substep authorizes that substep only. Record its changes, validation, run identities, outcomes and any remaining limitations here before marking it complete; do not automatically start the next substep.

All substeps preserve completed runs, original source pins for unchanged consumers, the non-refractory scientific settings and the separate best-validation/final-epoch checkpoint roles. Exp048, exp111 and noncollection scientific reruns remain excluded. **Next: 5.5, awaiting the author's request.**

### 5.1. Implement exp022's bank reuse workflow

- [x] **Complete 5.1 — 2026-09-08.**
- Add an exp022-specific, validated v4 path for exactly **90 reused cells and 12 replacement/new cells**. The current whole-bank import cannot selectively replace cells; do not create a general storage merger.
- Bind reuse to the source identity/digest recorded in steps 1 and 4. Retain original per-cell provenance, seeds, settings and both checkpoint roles, and reject missing, duplicated, unexpected or incompatible cells.
- Implement the allocation, partial-completion/recovery and finalization boundaries needed by 5.3–5.4. Account for the 34 seed-42 diagnostic snapshots; regenerating these is the default, avoiding a separate diagnostic-reuse mechanism.

**Done when:** implementation and focused validation pass, and the exact 90/12 assembly can be inspected. This substep does not launch benchmarks or production training.

#### Step 5.1 results

Implemented the narrow workflow in `experiments/exp022/reuse_contract.py` and `reuse.py`, exposed through `compute.py`, with operator commands in `experiments/exp022/README.md`. No general storage merger or new Pingstore CLI was introduced.

**Verified source and partition.** Read-only inspection of the actual retained `exp022-r001-compute` passed with the exact digest above. All 102 original cells, original scientific settings, complete epoch histories and separate checkpoint roles validate. The reusable subset contains **90 cells / 360 files / 1,779,904,000 bytes**: **48 inherited cells** from `4ad223d32620dd9f03698b89f28aedfe944d43ac` and **42 firing-rate-repaired cells** from `ac6f49884084811e3e05d49e8b45735d514ff245`. Their original files are copied unchanged; local import is not relabelled as their original training execution.

The twelve replacement workers are fixed as follows; each row includes seeds **42, 43 and 44**:

| New condition | New cell-name pattern | Old condition replaced | E/I counter steps |
|---:|---|---:|---:|
| 0.05 ms | `ping__dt0p05__seed{42,43,44}` | 0.05 ms | 24 / 12 |
| 0.2 ms | `ping__dt0p2__seed{42,43,44}` | 0.25 ms | 6 / 3 |
| 0.3 ms | `ping__dt0p3__seed{42,43,44}` | 0.5 ms | 4 / 2 |
| 0.6 ms | `ping__dt0p6__seed{42,43,44}` | 1 ms | 2 / 1 |

The three 0.1-ms timestep cells remain within the 90 reused cells. The complete inspectable plan, including every reused name, original provenance and file hash, is emitted without writing by:

```sh
uv run python -m experiments.exp022.compute --reuse-plan
```

**Execution boundaries.** `--reuse-reserve` allocates a hidden compute writer before dispatch and immediately records its retained-bank input pin in `run.json`, so pruning protects its ancestry. Only the twelve replacement cells can be dispatched through `--reuse-train-cell`; `--reuse-status` reports completion without exposing the bank downstream. Workers require the original clean source revision, resolved production contract and environment lockfile. Per-cell ownership rejects duplicate or unconfirmed owners, and stale recovery preserves original attempts. Slurm ownership checks include suspended/stopped jobs.

`--reuse-finalize` is a separate operation for step 5.4. It requires all twelve validated training results and execution records, assembles a standalone 102-model bank, and regenerates **all 34 seed-42 final-epoch diagnostics**. New execution metadata moves into `run.json`; scientific configurations, epoch histories and both checkpoint roles remain available to existing readers. Successful output contains **408 cell files plus 34 flat recording files**. Only the final atomic rename makes it consumable. Worker/finalizer exclusion and source checks protect this boundary.

`--recover-finalization` handles interrupted assembly explicitly. Before the assembled export is checkpointed, the twelve trained cells survive and diagnostics are regenerated. After checkpointing, recovery verifies the recorded export checksum and can finish after scratch cleanup, reservation removal or a failed final rename without retraining. Completed runs cannot be reopened. Human-readable history records allocation, source identity, original lineage, recovery and completion.

**Validation:** **188 tests passed, one pre-existing writing-heading test deselected, in 20.31 s**. This includes **64 new tests** (27 retained-source contract, 14 scientific recording, 23 workflow/recovery), the existing exp022 execution/timestep tests and relevant Pingstore layout/pruning tests. The complete fixture assembly uses real checkpoint loading at the production tensor shapes; its diagnostics are controlled fixtures, not scientific reruns. Failure injection covers changed sources, incompatible/duplicate/missing cells, missing or swapped checkpoint roles, incomplete/unproven training, active/stale ownership, snapshot failure, late cleanup/validation/rename failure and changed prepared exports. The original source remains byte-identical. Ruff and scoped diff checks pass.

Read-only discovery succeeded with **68 validated present runs**; exp022 and exp110's declared presentation inputs remain available, so their existing `data` classifications remain correct. No article content, source pin or author review tag changed in this substep.

**Scope held:** no production identity allocated, benchmark, training, diagnostic simulation, HPC submission, real-bank assembly, downstream stage, publication, commit or push. Allocation and recovery were exercised only in temporary test stores. A clean reviewed execution checkout and measured HPC resource plan are still required for the later production work. **Step 5.2 is next and has not started.**

### 5.2. Benchmark the replacement conditions and size the HPC jobs

**Status: skipped at the author's direction — 2026-09-08.** No fresh benchmarks were run. The author authorized proceeding directly to 5.3. Production limits use the retained A100 timings and verified current allocation/QoS limits recorded below; they are conservative estimates, not measured replacement-run timings.

Original benchmarking scope, not executed:
- Run bounded benchmarks at **0.05, 0.2, 0.3 and 0.6 ms**, using the intended network, batch size and training configuration. Benchmark outputs do not replace completed production training.
- Measure steady training time and memory, distinguish startup/compilation from steady execution, and estimate wall time, GPU-hours, concurrency and output size for all twelve replacements.
- Record the exact production cell list, reviewed source/environment identity, resource margins and proposed submission commands. Confirm the HPC allocation can support them.

**Original acceptance criterion (waived):** measured estimates and a concrete submission plan. The twelve full training runs are now authorized under 5.3.

### 5.3. Train the twelve replacement/new timestep cells

- [x] **Complete 5.3 — 2026-09-08.**

**Status: complete — all twelve replacement cells validated, with successful final Slurm outcomes and full training provenance recorded on HPC and mirrored locally.**
- Submit the prepared HPC work: **seeds 42, 43 and 44 at each of 0.05, 0.2, 0.3 and 0.6 ms**, with explicit **1.2/0.6-ms** refractories. Do not retrain any of the ninety preserved cells.
- Allocate the compute identity before submission and record jobs, source revision, environment and resolved parameters. Monitor completion and recover failed cells through the validated workflow.
- Validate each new cell's training result and both checkpoint roles. Keep outputs in the compute working directory until 5.4 completes the bank; incomplete outputs are not downstream scientific inputs.

**Done when:** all twelve production cells pass validation, with their job outcomes and provenance recorded.

#### Step 5.3 preparation

- Frozen execution commit: **`d9622c88bcd05a729bc92e92fcf22cd9a3ff4547`**, based on `255ab3b6e11fe4e92cd5f96c1634a2a58150bb05`. This is an isolated execution checkout; the shared working tree was not committed or pushed. Transfer bundle SHA-256: `28ca7c8211f5dfc5f5688b72df3424cb80bd41389b3dbd425fc39c9329b57b1c`.
- The frozen simulator's `models.py`, `config.py` and `timing.py` match the exact step-4 tested hashes. Lockfile SHA-256 remains `44df18b40d04655e4edf520d4802fef5c52c517477a6075ea194cc5521644dc7`. All twelve commands match their retained counterparts except the intended timestep/refractory settings and output paths: 7,000 samples, 50 epochs, batch 256, 1,024 E / 256 I, 25-Hz input, 6-ms GABA, learning rate 0.0004 and the original initialization/readout choices remain unchanged.
- Added support for initializing an **already allocated fresh v4 reservation** in the execution checkout, preserving its globally allocated ID. Initialization refuses mismatched origin/identity, populated exports, existing records or scratch work. Added a reuse-only Slurm worker which cannot dispatch finalization. **202 tests passed in the frozen checkout (one existing writing-heading test deselected);** this includes fourteen new reservation-transfer cases. Shell syntax and Ruff checks pass.
- Historical A100 times for 0.05 ms were **2 h 20 min–2 h 28 min per cell**, with about **24.8 GB peak GPU memory**. Step-count scaling gives approximately **36–38 min at 0.2 ms**, **24–26 min at 0.3 ms**, and **12–14 min at 0.6 ms**. These are planning estimates, not new benchmarks.
- Intended submission: three fine-timestep tasks with **4-hour limits**, nine coarser tasks with **2-hour limits**; each requests **one A100, 32 CPUs and 250 GB host RAM**, on Ampere with noninterruptible `gpu1` QoS. Twelve-way concurrency is within the verified 64-GPU user limit. Estimated actual use is approximately **11 GPU-hours**; the requested maximum is **30 GPU-hours**. The current allocation has sufficient balance. Queue delays remain unknown; maintenance capacity is not counted as freely available.
- Training outputs will remain in the hidden writer until separately authorized step 5.4. No diagnostics, bank finalization, downstream stages or publication are authorized by 5.3.

#### Step 5.3 submission

The compute identity **`exp022-r007-compute`** was allocated in the canonical local store before dispatch and initialized from the exact fresh reservation in the frozen HPC checkout. Its hidden local record retains the input pin and authoritative remote submission metadata. The HPC input bank is a byte-preserving copy of an existing **validated v4** `exp022-r001-compute` with the approved digest; no historical-schema conversion or source mutation was needed.

Both Slurm scheduler validation calls passed, followed by successful submissions:

| Array job | Tasks | Conditions | Per-task limit |
|---|---:|---|---|
| **35089250** | 0–2 | 0.05 ms, seeds 42/43/44 | 4 hours |
| **35089251** | 0–8 | 0.2, 0.3 and 0.6 ms, each with seeds 42/43/44 | 2 hours |

The remote environment is verified as **Python 3.10.20 / PyTorch 2.11.0+cu128 / CUDA 12.8**, matching step 4, with the unchanged lockfile and readable prepopulated MNIST files. Submission commands, frozen cell selections, scheduler responses, source revision and resource/environment settings are recorded in the hidden writer's `run.json` and dated README. Training attempts add their actual node/GPU/job provenance separately.

The first scheduler check shows **all twelve tasks PENDING**, with no start time assigned. Read-only local discovery also succeeded (70 present runs); exp022 and exp110 still have valid presentation data, so their existing `data` tags remain unchanged.

Completion monitoring checked every fifteen minutes, restricted to step 5.3 and meaningful progress/failure notifications. It is now **paused**, following successful validation and recording of all twelve cells and both checkpoint roles. **The bank remains incomplete and unavailable to downstream stages.**

#### Step 5.3 progress — 2026-09-08, 19:24 UTC

All twelve tasks started at approximately **19:04 UTC (20:04 BST)** on A100-SXM4-80GB GPUs. The production reuse-status validator reports **three complete, nine running, no retry or recovery candidates**. It accepted all three **0.6-ms** cells, including their completed 50-epoch metrics and both checkpoint roles, against the frozen campaign and source pin.

| Completed cell | Array task | Slurm outcome | Slurm elapsed |
|---|---|---|---|
| `ping__dt0p6__seed42` | `35089251_6` | `COMPLETED`, exit `0:0` | 16 min 16 s |
| `ping__dt0p6__seed43` | `35089251_7` | `COMPLETED`, exit `0:0` | 16 min 16 s |
| `ping__dt0p6__seed44` | `35089251_8` | `COMPLETED`, exit `0:0` | 16 min 16 s |

The remaining jobs were progressing normally at the approximately 19:21 UTC log check: **5/50 epochs at 0.05 ms**, **19–20/50 at 0.2 ms**, and **29/50 at 0.3 ms**. Observed throughput suggests roughly **two hours remaining overall**, governed by the three fine-timestep cells; this is a live estimate, not a completion guarantee. No training errors were found. Outputs remain in the hidden writer; step 5.3 remains incomplete until all twelve cells validate and their final job outcomes and provenance are recorded. Step 5.4 has not started.

#### Step 5.3 progress — 2026-09-08, 19:43 UTC

The production reuse-status validator now accepts **all nine cells at 0.2, 0.3 and 0.6 ms**, with completed training metrics and both checkpoint roles. The frozen campaign/source checks passed, with **three running cells and no retry or recovery candidates**. The three fine-timestep cells each had **12/50 epochs complete**, with epoch 13 advancing at the 19:41 UTC log check. Their observed remaining-time estimate is approximately **1 hour 40–45 minutes**. No execution errors were found.

New scheduler outcomes since the preceding check:

| Cell | Array task | Slurm outcome at 19:42 UTC | Slurm elapsed |
|---|---|---|---|
| `ping__dt0p2__seed42` | `35089251_0` | `COMPLETED`, exit `0:0` | 36 min 40 s |
| `ping__dt0p2__seed43` | `35089251_1` | Still `RUNNING`; worker outputs already validated | 38 min 16 s at check |
| `ping__dt0p2__seed44` | `35089251_2` | Still `RUNNING`; worker outputs already validated | 38 min 16 s at check |
| `ping__dt0p3__seed42` | `35089251_3` | `COMPLETED`, exit `0:0` | 26 min 40 s |
| `ping__dt0p3__seed43` | `35089251_4` | `COMPLETED`, exit `0:0` | 26 min 41 s |
| `ping__dt0p3__seed44` | `35089251_5` | `COMPLETED`, exit `0:0` | 26 min 43 s |

At that check, the two scheduler exits were still pending. All outputs remain hidden and the bank remains incomplete; step 5.4 has not started.

#### Step 5.3 progress — 2026-09-08, 19:56 UTC

Slurm now confirms **all nine coarse-timestep jobs `COMPLETED`, exit `0:0`**. The two previously pending exits, `35089251_1` and `35089251_2` (0.2-ms seeds 43 and 44), both completed after **39 min 11 s**, at **19:43:34 UTC**. All nine cells already passed the production output validator, including both checkpoint roles.

Only the three 0.05-ms tasks remain running. At 19:56 UTC, seeds 42/43 had **17/50 epochs complete** and were finishing epoch 18; seed 44 had **18/50 complete** and was training epoch 19. Observed epoch times were **156–161 seconds**, suggesting approximately **1 hour 25–35 minutes remaining**, including completion checks. No execution errors were found. Monitoring continues within step 5.3; the bank is still hidden and step 5.4 remains unstarted.

#### Step 5.3 completion — 2026-09-08, 21:26 UTC

**All twelve cells completed their 50 epochs, passed the frozen production validator, and have final Slurm outcomes `COMPLETED`, exit `0:0` and derived exit `0:0`.** No retries or recoveries were required. The last job finished at **21:23:13 UTC (22:23:13 BST)**.

| Timestep | Array tasks | Elapsed times, seeds 42 / 43 / 44 |
|---|---|---|
| 0.05 ms | `35089250_0`–`35089250_2` | 2 h 17 min 55 s / 2 h 18 min 50 s / 2 h 17 min 55 s |
| 0.2 ms | `35089251_0`–`35089251_2` | 36 min 40 s / 39 min 11 s / 39 min 11 s |
| 0.3 ms | `35089251_3`–`35089251_5` | 26 min 40 s / 26 min 41 s / 26 min 43 s |
| 0.6 ms | `35089251_6`–`35089251_8` | 16 min 16 s / 16 min 16 s / 16 min 16 s |

Elapsed allocation time sums to **10.98 GPU-hours**, with one A100 per task. The longest task took **2 h 18 min 50 s**, within its four-hour limit; every coarse task finished within its two-hour limit.

The completion check accepted the exact twelve-cell inventory, **24 checkpoint-role records** and **60 required-file identities** (`config.json`, `metrics.json`, `metrics.jsonl`, `weights.pth`, `weights_final.pth` per cell). It matched original training attempts, frozen commands, source/environment identities, actual Slurm job IDs and array indexes to successful final accounting rows. Both checkpoint roles were resolved and checked against their file hashes. The retained source-bank digest passed validation before and after the operation; none of the ninety preserved cells was retrained.

Under the exclusive bank lock, a metadata-only operation recorded these results in **`bank_reuse.training_completion`** inside the hidden HPC writer's authoritative `run.json`, with dated README history. The exact remote record was verified during transfer and mirrored into the hidden local reservation, with a local README receipt. The frozen campaign, reservation, scientific parameters and original source pin remain unchanged.

**Step 5.3 is complete; the compute bank is not yet complete.** Training payloads remain on HPC in the hidden `exp022-r007-compute` writer, and both local and remote exports remain empty. No diagnostics, bank assembly/finalization, downstream stages, consumer repinning, publication or push occurred. The completion monitor is paused. **Step 5.4 is the next separately requested substep.**

### 5.4. Assemble and validate the complete exp022 bank

- [x] **Complete 5.4 — 2026-09-09.**
- Combine the **90 preserved cells with the 12 validated new cells** into a new standalone **102-model** compute run using 5.1's workflow.
- Generate the **34 seed-42 diagnostic snapshots**, recording the current explicit model separately from original training provenance. Diagnostic regeneration does not imply retraining preserved models.
- Validate the complete cell inventory, both checkpoint roles, reused/new origins, source digests, diagnostic completeness and v4 layout. Complete the run atomically with its dated README history.

**Done when:** the new complete bank has a recorded run ID and verified payload digest. The original bank remains unchanged; consumers using only unchanged cells retain their existing pins.

#### Step 5.4 preparation — 2026-09-09

The author authorized this substep. Prepared a separate finalization job for the existing **`exp022-r007-compute`** reservation and the same frozen execution commit **`d9622c88bcd05a729bc92e92fcf22cd9a3ff4547`**. The job calls the tested `--reuse-finalize` workflow to copy the ninety preserved cells, incorporate the twelve validated replacements, generate all thirty-four fixed digit-0/sample-0 seed-42 recordings using final-epoch checkpoints, and validate/complete the bank atomically.

Prepared resources are **one A100, 32 CPUs, 250 GB host RAM and a two-hour limit**, using the existing environment and MNIST cache. This is a conservative resource limit, not a measured runtime estimate. The private submission driver revalidates the frozen source and completed training, records the exact wrapper and submission metadata, initially submits the job held, then releases it after recording its job ID and dropping the bank lock. It refuses duplicate submission or unreviewed recovery of a prior finalization attempt. Shell and embedded Python syntax checks passed. No frozen repository code was changed.

SSH authentication initially blocked execution; the author restored the login before submission. No work was submitted while access was unavailable.

#### Step 5.4 submission — 2026-09-09, 06:06 UTC

The live HPC checkout still matches the exact frozen commit and is clean. Source-bank and twelve-cell training validation passed again. Slurm scheduler validation succeeded, and **job `35110575`** was submitted held, recorded in the hidden writer's `run.json` and README, then released after the bank lock was dropped. The first scheduler check reports **PENDING**, with no start time assigned.

The job requests one A100, 32 CPUs and 250 GB RAM with a two-hour limit. It operates only on the existing `exp022-r007-compute` reservation and uses the unchanged frozen finalizer. Step 5.4 remains incomplete until all 102 models and 34 recordings validate and the complete run has a verified payload digest. Step 5.5 remains outside the authorized scope.

At the **06:11 UTC** progress check, job `35110575` is **RUNNING** and has written **9 of 34 diagnostic recordings**. Runtime provenance confirms A100-SXM4-80GB, Python 3.10.20, PyTorch 2.11.0+cu128 and CUDA 12.8. The finalization attempt started at 06:06:30 UTC. No execution errors are reported; the existing optional TF32 performance warning does not change precision settings.

The existing fifteen-minute completion monitor was redirected to this authorized substep. It completed independent bank validation and local replication, and is now paused. The private validation and replication helpers passed syntax checks. No analysis or presentation was included.

#### Step 5.4 HPC completion — 2026-09-09

Job **`35110575` completed successfully**, with both exit and derived exit **`0:0`**, after **12 min 21 s** (06:06:07–06:18:28 UTC). The finalizer completed the bank on its first attempt, without recovery or retraining.

Independent validation accepted the complete **102-model** inventory, **204 checkpoint roles**, **34 seed-42 diagnostic recordings**, exact diagnostic refractory arguments and source/checkpoint lineage. All ninety reused cells retain their original four files byte-for-byte; the twelve new models' checkpoints match the accepted training-completion records. The completed v4 run contains **442 export files, 3,578,615,807 bytes**.

- Run ID: **`exp022-r007-compute`**.
- Payload digest: **`sha256:6bfeda8ce5e32bb35748f335338ee6af29bedf76ac606babc9823a466da640c0`**.
- Recorded compute completion: **2026-09-09T06:18:23+00:00**.

The complete HPC bank is retained. Local replication and final acceptance are complete, as recorded below; no downstream stage has started.

#### Step 5.4 final acceptance — 2026-09-09

**Step 5.4 is complete.** Downloaded all **442 files / 3,578,615,807 bytes** into the preallocated local reservation while holding the bank/store locks. Every file identity and the complete payload digest matched the validated HPC bank before atomic visibility. Preserved the HPC execution provenance and appended the local reservation/transfer history to README.

The completed local run is **`.pingstore/runs/exp022-r007-compute/`**, with exactly `run.json`, `README.md` and `export/`. A second, independent acceptance pass on the local copy verified **102 model identities, all 204 checkpoint roles, the 90/12 provenance partition, unchanged reused-file hashes, and all 34 fixed digit-0/sample-0 recordings with explicit 1.2/0.6-ms refractory arguments**. Its digest is identical to the HPC digest above.

Read-only discovery also succeeded after local completion: **71 validated present runs**. Exp022's retained presentation inputs and exp110's existing presentation data remain available, so their `data` tags remain correct. No article tags, source selections or writings were changed.

The completion monitor is **paused**. No recovery, retraining, analysis, presentation, consumer repinning, publication or push was performed. **Step 5.5 is next and has not started.**

### 5.5. Regenerate exp022 analysis and presentation

- [x] **Complete 5.5.**
- Run analyse, then present, explicitly against the new complete bank and diagnostic evidence.
- Check the corrected timestep labels, the twelvefold sweep range, physical-time frequency calculations and the 90-reused/12-new bank description.
- Validate the new outputs and input lineage, inspect the scientific figures, and record both stage identities.

**Done when:** exp022 has validated replacement analysis and presentation. Publication and wholesale consumer repinning are not part of this substep.

#### Step 5.5 final acceptance — 2026-09-09

**Step 5.5 is complete.** Ran analysis locally against the complete `exp022-r007-compute` bank, then rendered the saved analysis. The accepted replacement pair is:

| Stage | Run ID | Completed (UTC) | Export |
| --- | --- | --- | --- |
| Analyse | **`exp022-r010-analyse`** | 06:49:33 | 37 files: results, curves, scientific cell configurations and 34 measured probes |
| Present | **`exp022-r013-present`** | 06:51:47 | 43 files: numbers, seven family learning-curve figures, 34 probe figures and one four-panel comparison |

- Analysis payload: **`sha256:e35366410e4b7711a6eca02fb054155cb4950415b9e2f469cd6fb73626c81b2f`**.
- Presentation payload: **`sha256:f3cbc15b2baef15470fd685569351a5486f7c59b660d8c63d95662729c4198c9`**.
- Exact lineage: presentation → analysis → `exp022-r007-compute`, whose payload remains **`sha256:6bfeda8ce5e32bb35748f335338ee6af29bedf76ac606babc9823a466da640c0`**. Each immediate reference contains both the explicit ID and digest.

The numerical exports explicitly describe **90 retained models reused unchanged and 12 newly trained models**, including the cell partition and original retained-bank reference. They contain all **102 fifty-epoch learning curves**, **204 checkpoint-role records**, and **34 newly measured fixed-probe recordings**, with no missing rasters or carried historical images. The presentation draws only saved measurements; neither stage trained models or simulated activity.

**Corrections required by execution and figure review:**

- Fixed presentation readers to resolve canonical v4 raster paths. Analysis now writes flat `<cell>--rasters.npz` files and declares their paths explicitly; rendering lineage points to the actual saved files.
- Retained step 3's physical-time spectral estimator: actual bin durations, 3-ms Gaussian smoothing and Hann-windowed FFT. Population-rate plots now normalize the last fractional bin by its actual width.
- Decoded the saved float32 timestep through its shortest decimal representation, avoiding an artificial tiny terminal bin caused by direct widening to float64. Coarse recordings retain their actual **199.8-ms** duration.
- Added explicit millisecond units to the corrected timestep legend and exported the **12-fold** sweep range.
- Replaced misleading “asynchronous”/“I silent” annotations for unresolved or low-count spectra with “no resolved spectral peak”/“too few I spikes”. Actual silence remains distinguished. Peak annotations identify a spectral peak without claiming that every short probe establishes sustained gamma oscillation.
- Preserved the shared stage helper's dated README execution and source history when appending stage-specific descriptions.

**Corrected timestep results** (three seeds per condition; mean best validation accuracy and mean final-epoch validation rates from retained training history):

| Timestep (ms) | Best validation accuracy (%) | E rate (Hz) | I rate (Hz) | Seed-42 probe duration (ms) | Seed-42 spectral peak (Hz) |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 88.92 | 14.05 | 85.64 | 200.0 | 60.00 |
| 0.10 | 89.89 | 16.32 | 106.10 | 200.0 | 55.00 |
| 0.20 | 89.94 | 16.57 | 114.04 | 200.0 | 70.00 |
| 0.30 | 89.76 | 16.35 | 118.92 | 199.8 | 60.06 |
| 0.60 | 88.92 | 16.67 | 150.69 | 199.8 | 45.18 |

The timestep-condition means span **1.02 percentage points** in best validation accuracy. These are training-validation summaries, not the separate exp044 evaluation. Spectral peaks come from one approximately 200-ms digit-0/sample-0 probe per condition; they are not a population-level frequency estimate. The corrected coarse conditions must not be interpreted as the same models as the old 0.25/0.5/1-ms conditions.

**Validation:** 74 relevant exp022 tests passed, with the existing writing-heading test deliberately excluded; all 12 physical-time/reuse-partition checks also passed after the float32 regression case was added. Independent acceptance validated complete v4 layouts and payload digests, exact input references, every saved learning curve and checkpoint-role inventory, all 34 probes' durations and rate integrals against raw spike counts, presentation-number preservation, and all 42 rendering-lineage entries. Inspected all figure families and probe figures in contact sheets, with full-size checks of the timestep, long-legend and low-count annotation cases. Corrected annotation wrapping was checked before the final presentation run.

Both accepted stages ran from the shared working tree at Git base `255ab3b6e11fe4e92cd5f96c1634a2a58150bb05`; their run records honestly report `dirty: true` and `code_dirty: true`, rather than claiming the frozen HPC training commit. Lockfile SHA-256 remains `44df18b40d04655e4edf520d4802fef5c52c517477a6075ea194cc5521644dc7`. Accepted source file SHA-256 values:

- `experiments/exp022/analyse.py`: `55001631aa50d20ab5f5a0e123e66abb964124e27fd327b195c853c959f61aad`.
- `experiments/exp022/present.py`: `c14e2c0d860d8d7f572cf014633e73d43d9f696eda55409db684b5d990bf8859`.
- `experiments/exp022/recipe.py`: `2f69390165b9722d9963e4db52e2f5466fae17c01d8c3b77cd26965242374cca`.

Completed intermediate runs remain immutable: `exp022-r008-analyse`/`exp022-r009-present` preceded the timestep-precision correction, and `exp022-r011-present`/`exp022-r012-present` preceded final annotation corrections. **Use the accepted pair above for this substep**, not those intermediate renderings. No source selection was changed.

Read-only discovery passed with **75 valid present runs**, including the accepted replacement. Exp022 and the dependent synthesis retain usable local presentation data, so their existing `data` tags remain correct. No article revision or Writing Guide version advancement was performed: exp022's historical hardcoded timestep prose/table still requires reconciliation during the planned writing/adoption pass. No materialization, publication, consumer repinning, commit, push or exp044 evaluation occurred. **Step 5.6 is next.**

### 5.6. Evaluate the corrected exp044 timestep sweep

- [x] **Complete 5.6.**
- Use the new bank to run **all 15 evaluations and five seed-42 snapshots** across the selected five-timestep grid. This is the default execution plan; the numerical minimum would be twelve evaluations and four snapshots, but that requires an additional partial-reuse path.
- Run analyse, then present, with explicit stage inputs. Verify 666/333 steps and 199.8-ms realized trials at 0.3/0.6 ms, rate normalization and corrected axes.
- Inspect the figures and quantify changes relative to the retained old sweep without treating old coarse conditions as the same model.

**Done when:** the replacement compute, analysis and presentation runs validate, with IDs/digests and the main numerical changes recorded.

#### Step 5.6 final acceptance — 2026-09-09

**Step 5.6 is complete.** Executed the production recipe locally against **`exp022-r007-compute`**, with `PINGLAB_SMOKE=0`. All fifteen final-epoch checkpoints were evaluated on 1,000 official MNIST test images each, and all five seed-42/sample-index-0 probes were regenerated. No training was launched. The existing corrected runners passed preflight and worked without further simulation-code changes.

| Stage | Completed run | UTC execution window | Export files |
| --- | --- | --- | --- |
| Compute | **`exp044-r007-compute`** | 09:53:54–09:56:19 | 21: fifteen evaluations, five recordings and evidence |
| Analyse | **`exp044-r008-analyse`** | 09:56:48–09:56:51 | 2: results and measured raster samples |
| Present | **`exp044-r009-present`** | 09:57:03–09:57:05 | 7: six figure files and numbers |

**Measured compute runtime: 2 min 25 s**, below the prior conservative 5–10-minute estimate. No HPC allocation or queue was involved.

- Compute payload: **`sha256:a2669866f1b24132e6a19424ffae809c74de0986f8c54e3abcc44eda9a651156`**.
- Analysis payload: **`sha256:72ea334caab7252c8fd009e23a62acc846244109993e424f15e4e0d0b22f85d5`**.
- Presentation payload: **`sha256:5d1433f4338ab6661b9b07a01720f360d8eb072ed85c87c3a9d5556fa223213d`**.
- The compute pins `exp022-r007-compute` at **`sha256:6bfeda8ce5e32bb35748f335338ee6af29bedf76ac606babc9823a466da640c0`**. Analysis pins both compute and bank; presentation pins analysis. All references and source boundaries validated unchanged.

**Final-epoch official-test results** (mean ± SEM over seeds 42–44):

| Timestep (ms) | Accuracy (%) | E rate (Hz) | I rate (Hz) |
| --- | --- | --- | --- |
| 0.05 | 88.80 ± 0.20 | 14.27 ± 0.16 | 87.29 ± 1.31 |
| 0.10 | 89.63 ± 0.18 | 16.53 ± 0.11 | 107.85 ± 0.77 |
| 0.20 | 89.37 ± 0.23 | 16.86 ± 0.04 | 116.28 ± 0.20 |
| 0.30 | 88.37 ± 0.07 | 16.70 ± 0.01 | 121.70 ± 0.11 |
| 0.60 | 88.87 ± 0.07 | 17.02 ± 0.05 | 153.90 ± 0.43 |

Mean test accuracy spans **1.27 percentage points**, and mean E rates span **14.27–17.02 Hz**. From 0.1 to 0.6 ms, E firing rises **2.94%** and I firing rises **42.70%**, despite constant physical refractory periods. E-rate means are not strictly monotonic: 0.3 ms is slightly below 0.2 ms. This remains a comparison of separately trained pipelines, not a fixed-weight convergence test or an estimate of gamma-period invariance.

**Comparison with retained old sweep `exp044-r003-analyse`:**

- The unchanged **0.1-ms** point reproduces all three old evaluation rows exactly, including accuracy and E/I rates; the seed-42 raw E/I probe arrays are also identical. This is the direct preserved-control check.
- At **0.05 ms**, mean accuracy remains 88.80%, while E firing changes by **+0.43 Hz** and I firing by **−10.93 Hz**. This point uses newly trained models with corrected 1.2/0.6-ms refractories; the old physical refractories were 0.6/0.3 ms, so matching timestep does not imply matching model dynamics.
- The old **0.25/0.5/1-ms** conditions were replaced by **0.2/0.3/0.6 ms**. They are not paired equivalent conditions. The old whole-sweep accuracy span was 1.30 percentage points versus 1.27 now, but the grids differ. The old coarse-step decline in I firing is absent from the corrected grid; I firing now increases across its five points.

**Validation:** all **80 relevant tests passed**, with the pre-existing article-render test excluded in accordance with the no-writing-tests rule. Independent acceptance checked complete v4 layouts/digests and lineage, fifteen complete evaluations (**15,000 image evaluations**), final-checkpoint provenance, all five raw probes and selected raster arrays, mean/SEM calculations, and numerical preservation through presentation. All evaluation metadata declare exact **1.2/0.6-ms** refractories, with E/I counter pairs **24/12, 12/6, 6/3, 4/2 and 2/1**. No interspike interval in any recorded neuron violated its configured refractory period. The 0.3/0.6-ms trials have **666/333 steps and 199.8-ms** realized duration; raster rates independently match full-population spike counts divided by that duration. Inference uses the shared realized-duration rate denominator covered by the public timing-boundary tests.

Inspected the timestep, raster-strip and training-curve figures. All corrected labels and physical-time axes are legible and unclipped. The existing Matplotlib `tight_layout` warnings occurred, but the emitted figures passed visual inspection. No plot changes were required. Updated exp044's README to describe the current grid while preserving the historical recipe description.

All three runs record the actual shared-tree source base **`255ab3b6e11fe4e92cd5f96c1634a2a58150bb05`**, with dirty/code-dirty flags and lockfile SHA-256 **`44df18b40d04655e4edf520d4802fef5c52c517477a6075ea194cc5521644dc7`**. They do not claim execution from the frozen HPC training commit.

Read-only discovery passed with **76 valid present runs**, including `exp044-r009-present`. Exp044 and the dependent synthesis retain usable local data, so their `data` tags remain correct. Article text, guide-version tags and selections were not changed: the later writing pass must reconcile exp044's hardcoded twentyfold range, old timestep lists and exact-200-ms prose. No publication, materialization, consumer repinning, commit or push occurred. **Step 5.7 is next and has not started.**

### 5.7. Adopt 1.2/0.6 ms in exp033 and recompute the theory

- [x] **Complete 5.7.**
- Confirm that step 3's frozen historical exp033 **and exp054** theory recipes/validators remain independent of live defaults before changing the mean-field parameters.
- Add the adopted mean-field recipe with **1.2-ms E / 0.6-ms I** refractory parameters, preserving the frozen historical 3/1.5-ms recipe and its readers.
- Recompute fixed points, Jacobians, onset thresholds/frequencies, ramps, cycles and sensitivity results; the refractory terms enter the gain function.
- Run compute → analyse → present, retaining the existing exp041 spiking-frequency measurements. Validate source identities and inspect the changed theoretical predictions and figures.

**Done when:** the adopted theory has validated compute, analysis and presentation runs suitable for explicit reuse by exp054.

#### Step 5.7 in progress — 2026-09-09

Added the adopted exp033 recipe v2 with **1.2/0.6-ms** gain refractories. Historical exp033 v1 and exp054 recipes retain **3/1.5 ms**, with gain evaluation bound to the recorded recipe. Analysis and presentation propagate the source configuration; current collection reuse rejects the historical theory recipe. Other physical parameters, grids, solvers and estimators are unchanged.

The complete local production compute started at **10:03:58 UTC**, using the reservation **`exp033-r010-compute`**. Its reference onset is approximately **0.593905 nA**; sensitivity integrations remain in progress. The hidden writer is not completed evidence. Integration quadrature warnings will be assessed during numerical acceptance. The existing **`exp041-r002-analyse`** frequency input is retained for the subsequent analysis.

The combined exp033, exp054 and collection-refractory regression passed **96 tests** (two writing tests excluded). Live historical exp033 and exp054 readers still validate their retained evidence and frozen configurations. Step 5.7 remains incomplete until all three new stages and scientific acceptance checks finish; step 5.8 has not started.

#### Step 5.7 numerical correction and restart — 2026-09-09

Independent checks found that fixed-point gains on the sensitivity grids agree with a stable formula to within **2.8e-14/ms**, but the oscillatory trajectories reach stronger inputs. At E input **0.9753635691850147 nA**, sigma **3 mV**, the original integrand gives **0.03691324/ms** instead of **0.03464614/ms**, approximately **6.54% too high**. The expression `1 + erf(u)` loses precision for sufficiently negative arguments; its quadrature warnings cannot be accepted as harmless for this run.

Stopped the first attempt, retaining **`.exp033-r010-compute.tmp`** as incomplete and unused. Recipe v2 now explicitly records the mathematically equivalent `erfcx(-u)` integrand for negative arguments, retaining the positive-tail overflow guard. Frozen v1 and exp054 historical execution keep their original arithmetic. A 70-digit independent integration confirms the corrected diagnostic value; all **97 regression tests** pass, including exact historical arithmetic, monotonic gain through the problematic range, recipe isolation and stage contracts.

Restarted the complete production computation as **`exp033-r011-compute`** at **10:12:36 UTC**. No partial outputs from the interrupted attempt are reused. Acceptance will check fixed points, half-step Jacobians and gains sampled across all ramps and cycles, including each trajectory's extreme inputs.

#### Step 5.7 final acceptance — 2026-09-09

**Step 5.7 is complete.** The corrected full production compute finished locally in **2 min 17 s**, without quadrature warnings. Analysis and presentation completed independently against the saved evidence. Accepted runs:

| Stage | Run ID | Completed (UTC) | Export |
| --- | --- | --- | --- |
| Compute | **`exp033-r011-compute`** | 10:14:53 | 2 files; 48,944,322 bytes of indexed numerical evidence |
| Analyse | **`exp033-r012-analyse`** | 10:15:00 | 3 files; 445,601 bytes of measurements and figure coordinates |
| Present | **`exp033-r014-present`** | 10:16:15 | 7 files; six SVGs and numbers, 665,208 bytes |

- Compute payload: **`sha256:efa4b1c3a47d36eedbd14302a0c57f573630cee9e5f0ed7221f1139585a02b55`**.
- Analysis payload: **`sha256:b1aa29e8040d0722d2e79e1cb34acb3d5a75209764b5fcb420be7c76db510b98`**.
- Presentation payload: **`sha256:a387208bf1ee340bdad5400aab5f8ee70b5bed1766a379dc44e10f28bbf8b15b`**.
- Analysis pins this compute and unchanged **`exp041-r002-analyse`**, digest **`sha256:bbf666b78e993f512db12bbb3ff45ed85857b3ef72772bc584bd8c0f6fe09d99`**. Presentation pins the analysis. Full v4 ancestry, layout and checksums validate.

All **7,062 fixed-point/eigenvalue grid records** are present: reference plus six decay sweeps, seven reductions and both grids for four noise scales. All **250 ramp trajectories** and **five cycles** are complete and finite, including dense waveform samples. Independent stable-integral agreement is within **3.82e-17/ms** at the checked fixed points and **9.03e-17/ms** across **33,578 trajectory gain checks**, including extreme inputs. Fixed-point RHS residuals are below **9.67e-13**. All eleven refined reference/decay/noise onsets pass the half-step Jacobian check; the largest resulting real eigenvalue magnitude is **3.67e-9/ms**. The largest coarse/fine noise-onset discrepancy is **2.15e-11 nA**.

Compared with historical **`exp033-r007-analyse`**:

| Reference measurement | Historical 3/1.5 ms | Adopted 1.2/0.6 ms |
| --- | --- | --- |
| Hopf onset | 0.59633710 nA | **0.59390477 nA (−0.408%)** |
| Onset frequency | 27.56644477 Hz | **27.56644477 Hz**, unchanged to reported precision |
| E cycle peak-to-peak amplitude at onset + 0.4 nA | 6.74535 Hz | **6.89754 Hz (+2.256%)** |
| Absolute cross-correlation lag | 5.22723 ms | **5.15463 ms** |
| 3D coarse onset | 0.70 nA | **0.69 nA**, on the same 0.01-nA grid |

All six 2D reductions still have no detected Hopf on the tested grid; the 3D reduction retains it. Noise onsets for sigma **3/4/5/6 mV** are now **0.625697 / 0.593905 / 0.614524 / 0.754064 nA**. All four retain the existing sampled supercriticality criterion. These are finite-grid and trajectory diagnostics, not a first-Lyapunov-coefficient proof or a universal minimum-dimension claim. Effective noise remains uncalibrated. The comparison includes the documented numerical stabilization as well as the refractory change; historical high-input waveform/sensitivity values are not retroactively certified.

The frequency overlay uses the exact medians of the existing **18 exp041 measurements**; no spiking data were recomputed or modified. Its difference from the archived exp033 summary is at most **2.55e-6 Hz**, the already documented historical precision discrepancy. New mean-field onset frequencies remain materially unchanged across all six decay conditions.

Inspected all six figures. The intermediate **`exp033-r013-present`** exposed native-renderer label/tick overlaps and a reduction legend over the traces. The final presentation fixes those and makes waveform/state units explicit; its numerical JSON is byte-identical to the intermediate presentation. Both completed presentations remain immutable; the interrupted `r010` writer remains hidden and unused.

Validation: **97 regression tests passed**, followed by **34 exp033 tests** after the figure edits; writing tests were excluded. Lint and whitespace checks pass. Read-only discovery validates **78 present runs**, including the accepted new presentation. Exp033, exp054 and exp110 still have usable local presentation data, so their `data` tags remain correct. Guide-version tags were not advanced without a full writing pass; exp033 still records v35.4.0 and needs the later guide/conformance update.

Detailed local QA is saved in `/Users/eoin/.codex/execution-snapshots/exp033-step57-qa/validation.json`. No article prose, publication binding, materialization, consumer repinning, commit or push changed. **Step 5.8 is next and has not started:** use `exp033-r011-compute` / `exp033-r012-analyse` as explicit new-theory evidence alongside the preserved exp054 spikes.

### 5.8. Refresh exp054 using separate preserved-spike and new-theory inputs

- [x] **Complete 5.8.**
- Extend exp054's input contracts to accept the preserved **exp054-r008-compute** spiking probes alongside the explicit refreshed theory from 5.7. Keep spike and theory configurations and lineage distinct; preserve historical readers.
- Run analyse → present using all **136 retained spiking probes** and the existing exp041 frequencies. **Do not rerun the HPC coupling grid merely to refresh theory.**
- Validate the combined lineage, inspect the figures and record how the theoretical comparison and conclusions changed.

**Done when:** refreshed exp054 analysis and presentation validate with explicit old-spike/new-theory sources and unchanged spike payloads.

#### Step 5.8 final acceptance — 2026-09-09

**Step 5.8 is complete.** Added `analyse --theory-source <exp033-analysis-id>` and an explicit `exp054.theory-refresh/v1` analysis configuration. Its `spike_source_recipe` preserves the complete original spiking-run configuration, including the unused embedded historical theory declaration; its separate `theory_recipe` records adopted exp033 v2 with **1.2/0.6-ms** refractory parameters and the stabilized gain integral. This is an analysis/presentation contract, not a relabelled compute recipe. Historical exp054 compute and analysis readers remain supported.

The reader validates the selected exp033 analysis, its compute ancestry, matching recipes and the exact common exp041 frequency input. Presentation rechecks that saved theory coordinates and numbers match the pinned theory analysis. The old mean-field payload inside the spiking compute is not consumed in this path. No new spikes, HPC job or numerical theory solve ran.

| Stage | Run ID | Execution (UTC) | Export |
| --- | --- | --- | --- |
| Analyse | **`exp054-r013-analyse`** | 10:21:52–10:21:57 | 3 files, 959,507 bytes |
| Present | **`exp054-r014-present`** | 10:22:36–10:22:39 | 8 files, 787,712 bytes: seven PNG figures and numbers |

- Analysis digest: **`sha256:74ff2bdae10086f5452ae23f8c58d92867ec23edb027654f2848f895c95f8cef`**.
- Presentation digest: **`sha256:ed1a573e60f61234a555f470f4c43befe4da9588bbf785146dc815f7c89f98fb`**.
- Preserved spikes: **`exp054-r008-compute`**, unchanged digest **`sha256:20a28f14c6c58bebd3729f4b8de5f15c59e32b8374876108886a0198fabc6a66`**.
- New theory: **`exp033-r012-analyse`**, digest **`sha256:b1aa29e8040d0722d2e79e1cb34acb3d5a75209764b5fcb420be7c76db510b98`**, retaining ancestry through `exp033-r011-compute`.
- Existing frequencies: **`exp041-r002-analyse`**, digest **`sha256:bbf666b78e993f512db12bbb3ff45ed85857b3ef72772bc584bd8c0f6fe09d99`**.

All **136 probe files** validate. Full-population post-burn rates, coupling-grid contrasts, private/shared null scans, autocorrelogram coordinates and displayed spike coordinates are **exactly unchanged** from `exp054-r009-analyse`. All **seven regenerated PNGs are byte-identical** to `exp054-r012-present` and were visually inspected. These exp054 figures contain empirical results; the combined theory-comparison figure belongs to exp110 and will be refreshed in 5.10.

The retained theory comparison now reports reference onset **0.59390477 nA**, versus **0.59633710 nA** previously (**−0.408%**). Frequency remains **27.56644477 Hz**; the largest frequency change across all six inhibitory decays is below **4.15e-10 Hz**. The sampled hysteresis width remains zero; amplitude-squared regression R² is **0.9994368**, and the existing sampled supercriticality verdict is unchanged. These remain the qualified diagnostics described in 5.7, including its numerical stabilization. The refresh does not resolve the existing quantitative mismatch between mean-field and spiking frequencies or calibrate effective noise.

Validation: **70 regression tests passed**, including eight new separate-source tests. They cover stage isolation, preserved source recipes, old-theory rejection, mismatched frequency pins, corrupted sources and theory substitutions in analysis outputs. One test that also renders article prose was excluded. Lint and whitespace checks pass. Detailed verification is in `/Users/eoin/.codex/execution-snapshots/exp054-step58-qa/validation.json`.

Read-only discovery validates **79 present runs**, including the new exp054 presentation. Exp054 and the dependent exp110 synthesis retain valid local presentation data; their `data` and v36.0.0 tags remain unchanged. No article prose, publication, materialization, collection-wide consumer repinning, commit or push occurred. **Step 5.9 is next and has not started.**

### 5.9. Correct exp023's displayed refractory metadata

- [x] **Complete 5.9.**
- Correct displayed refractory metadata to the actual **1.2/0.6-ms** execution, with explicit provenance explaining the historical declaration mismatch.
- Regenerate affected analysis/presentation metadata as needed while preserving existing simulations and numerical measurements. Never rewrite historical export bytes or imply that reused simulations were newly executed.
- Validate the corrected displays, numerical preservation and lineage; record any new stage identities.

**Done when:** exp023's displayed parameters accurately describe its retained evidence, with no scientific resimulation.

#### Step 5.9 final acceptance — 2026-09-09

**Step 5.9 is complete.** Corrected reporting metadata for the active, independently verified **6-ms-GABA** chain, `exp023-r011-compute` → `exp023-r012-analyse` → `exp023-r013-present`. No new analysis was needed: existing scientific measurements remain authoritative, and the correction creates only a new presentation.

The frozen source recipe declares E/I refractories of **3/1.5 ms**, but its **0.1-ms** execution used **12/6 simulation-step counters**, giving **1.2/0.6 ms**. The recorded Git base `cb00575381d35c01cbca8c1d4d19396a7624cb85` defines those module counters using the 0.25-ms default and its production step calls omit the runtime-derived counters. The execution was code-dirty, so the Git base alone is not claimed to reproduce its entire source. Retained voltage evidence independently corroborates the audit:

| Retained scope | Selected neuron | Complete post-spike reset holds | Hold duration |
| --- | --- | --- | --- |
| COBA E | 465 | 20 | 12 steps = 1.2 ms |
| PING E | 184 | 17 | 12 steps = 1.2 ms |
| PING I | 16 | 32 | 6 steps = 0.6 ms |

Each complete hold remains at reset for the stated steps and departs on the following sample. Initial and end-truncated intervals are excluded. COBA I is silent and supplies no refractory-duration observation. The minimum population ISIs are 40, 21 and 11 steps for COBA E, PING E and PING I respectively; none violate the resolved periods. The fourteen f–I records retain population totals rather than per-neuron trajectories, so their refractory interpretation comes from the shared audited execution path, not an invented per-cell ISI check.

Accepted replacement: **`exp023-r014-present`**, completed **2026-09-09 10:31:54 UTC** as an explicit **`metadata-correction`** operation, with **25 files / 1,362,420 export bytes**.

- New presentation digest: **`sha256:339b60faa112871156979e8047876672029c4a26e24d5365c3d13f8e39fe2fb6`**.
- Pinned prior presentation: **`exp023-r013-present`**.
- Pinned unchanged analysis: **`exp023-r012-analyse`**, digest **`sha256:76224197aca292ba1d64a11d0dd9a2de11a52cba24c60f38be15478f5d40bad6`**.
- Pinned unchanged compute: **`exp023-r011-compute`**, digest **`sha256:c08f1e26a231130156e54d294da5950f5191411775d79e012d21b5ba8e710671`**.

All **24 figure files are byte-identical** to the prior presentation. Raster rates, spectral peaks, f–I curves, estimator settings and all remaining scientific values are exactly preserved. The numbers export changes only the two reported refractory values, a distinct **`exp023.reported-configuration/v1`** schema identifying the reporting correction, and normal presentation execution fields. The original compute recipe is retained separately in the new run's execution configuration, and `run.json` plus README explain the declaration mismatch. No historical export, manifest or README was edited.

The explicit command is:

```sh
uv run python -m experiments.exp023.present --source exp023-r012-analyse \
  --metadata-source exp023-r013-present
```

The correction is restricted to the audited compute identity, payload, recipe and recorded Git base; it does not reinterpret arbitrary historical runs. In particular, it does not rehabilitate `exp023-r008-compute`, whose GABA declaration was wrong. The ordinary presentation path and historical readers remain available.

Validation: **40 tests passed**, including preservation and unaudited-source rejection checks; the article-render test was excluded. Lint and whitespace checks pass. Full layouts, payloads and all three input references validate. Read-only discovery now lists **80 valid present runs**. Detailed acceptance is saved in `/Users/eoin/.codex/execution-snapshots/exp023-step59-qa/validation.json`.

Exp023's Methods already reads the two refractory values directly from the selected presentation's `config.biophysics`, so selecting `exp023-r014-present` supplies **1.2/0.6 ms** without changing prose. No browser rendering or publication selection was changed. Exp023 and exp110 retain usable presentation data and their existing `data` / v36.0.0 tags. No simulation, remeasurement, figure redraw, HPC job, materialization, commit or push ran. **Step 5.10 is next and has not started.**

### 5.10. Refresh exp110's synthesis

- [x] **Complete 5.10 — 2026-09-09.**
- After 5.6, 5.8 and 5.9, regenerate exp110's presentation from the corrected exp044 and exp054 results and reconcile the refreshed exp023 metadata, preserving its unchanged spiking inputs from other experiments.
- Adapt exp110's plotting call to select `exp054.recipe.spike_configuration(source_recipe)` from the new separate-source analysis configuration, while retaining the complete source configuration in provenance. Use `exp054-r013-analyse` for the refreshed theory comparison; the ordinary collection dispatcher still describes the historical combined-compute workflow and must not silently substitute it.
- Update the manuscript's affected comparisons, claims, methods, tables and captions under the current Writing Guide, preserving author-assigned `reviewed` tags.
- Validate the inputs and figures and record the substantive changes. Follow the repository rule that the author checks localhost Typst renderings.

**Done when:** exp110's presentation and manuscript consistently reflect the replacement evidence. Step 7 still covers the collection-wide writing/provenance pass.

#### Step 5.10 results

- **Presentation:** `exp110-r021-present`, payload `sha256:b33dcc71d836b61aec282b3c6b9f8a1c4993149f45c599a82df995e3428bb37d`; local execution **10:39:16–10:39:18 UTC (2 seconds)**, six PNG/PDF exports totalling **641,660 bytes**. Recipe `exp110.presentation/v11` selects the preserved spike configuration for plotting and records the complete separate spike/theory configuration without alteration.
- **Replacement inputs:** `exp054-r013-analyse` (`sha256:74ff2bdae10086f5452ae23f8c58d92867ec23edb027654f2848f895c95f8cef`) and `exp044-r009-present` (`sha256:5d1433f4338ab6661b9b07a01720f360d8eb072ed85c87c3a9d5556fa223213d`). All five prior exp037/exp041/exp046 presentation and analysis references remain exactly unchanged from `exp110-r020-present`.
- **Figures:** Figure 2 uses the refreshed 1.2/0.6-ms theory; its coupling maps and rasters remain pixel-identical. Figure 7C uses the 0.05/0.1/0.2/0.3/0.6-ms evaluations; its exp037 perturbation panels remain pixel-identical. Figure 6's entire PNG remains byte-identical. All three scientific PNGs were visually inspected; no clipping was found.
- **Manuscript:** updated under Writing Guide **36.0.0**, dated 2026-09-09. Corrected Figure 2's onset to **0.594 nA** (frequency remains **27.6 Hz**), Figure 7's twelvefold timestep range, accuracy **88.37–89.63%** (span **1.27 percentage points**) and endpoint E rates **14.27–17.02 Hz**. Captions, design table and Methods now agree on fixed **1.2/0.6-ms** holds and **199.8-ms** presentations at 0.3/0.6 ms. Removed the obsolete pending-reruns note, recorded twelve replacement networks plus three reused 0.1-ms networks, and preserved the fixed-weight-convergence limitation. Added the gain-integrand cancellation correction to the numerical appendix account; the theory differences are not attributed exclusively to refractoriness.
- **Exp023 and unchanged article inputs:** the build used `exp023-r014-present`, whose reported E/I refractory values are 1.2/0.6 ms and whose figures are unchanged. Other direct article inputs are `exp025-r007-present`, `exp038-r008-present`, `exp042-r019-present`, `exp049-r012-present` and `exp082-r026-present`. No new simulations or analyses were needed for this substep.
- **Validation:** **14 targeted tests passed**, Ruff and `git diff --check` passed, discovery validated **81 present runs**, all seven article inputs were available, and the local **38-entry Demolab build passed**. All nine embedded article images matched their selected source files exactly, including the refreshed exp110 bundle and exp023 presentation. Detailed source references, image comparisons and checks are saved in `/Users/eoin/.codex/execution-snapshots/exp110-step510-qa/validation.json`.
- **Scope:** retained the author-approved standalone article, local TOC and existing Methods/appendix scaffolds. `data` and `v36.0.0` remain; no author review tag was changed. This is a targeted scientific update, not completion of the unfinished manuscript scaffolds. Localhost article rendering remains for the author to inspect. No deployment, materialization, stored-run mutation, commit or push occurred. **Step 7 remains separately scoped.**

## 6. Evidence that does not need a scientific rerun

No scientific rerun is required solely for this refractory choice in:

`exp024, exp025, exp037, exp038, exp041, exp042, exp046, exp047, exp049, exp082`.

Their relevant retained spiking execution used dt = 0.1 ms and therefore already used 1.2/0.6 ms. Preserve original source pins and correct descriptions where necessary. Exp023's simulations can also remain. Exp080 and exp081 use passive, nonspiking models and require no refractory change.

Intentional transmitted-spike insertion and inhibitory replay transformations retain their existing semantics; they are not native refractory violations to repair.

## 7. Recipe compatibility, provenance and writings

**Complete — 2026-09-09.**

- [x] Freeze historical scientific recipe versions before changing defaults, then add new versions for the adopted model.
- [x] Fix exp033 validators that compare old evidence against current defaults.
- [x] Freeze exp054's historical mean-field configurations: its old recipe versions currently import live exp033 defaults and would otherwise become unreadable after the change.
- [x] Separate preserved spike configuration from replacement theory configuration in exp054's input contracts and lineage.
- [x] Correct declared 3/1.5-ms metadata with explicit provenance; never silently alter historical export bytes or imply a reused run was newly computed.
- [x] Update collection model descriptions, exp022's bank/timestep inventory, exp023's displayed parameters, and exp033/exp044/exp054/exp110 Methods, results, tables and captions where affected.
- [x] Apply the current Writing Guide, maintain the separate data/text and guide-version tags, and preserve author-assigned `reviewed` tags. Do not add automated tests for writings.
- [x] Build updated presentations/articles and inspect the generated scientific figures. Follow the repository rule that the user checks localhost Typst renderings.

### Step 7 results — 2026-09-09

- **Recipe compatibility:** rechecked the historical exp033 readers and the separate exp054 spike/theory input contract implemented in earlier steps. Closed the remaining default-recipe gap: new exp054 combined compute now uses `exp054.recipe/v6`, embedding exp033 v2 with 1.2/0.6-ms E/I refractory periods and the stable gain integral. All ten exp054 v1–v5 production/smoke definitions exactly match the pre-edit snapshot; historical v5 retains its original 3/1.5-ms theory. The explicit preserved-spike/theory-refresh route remains separate and accepts eligible v4/v5/v6 spike sources. No new production compute was needed to adopt the previously completed replacement evidence.
- **Model and bank documentation:** updated the collection description and exp022/exp033/exp054/exp110 execution records. The current exp022 bank has 102 models: 90 reused and twelve newly trained timestep models, with 34 new diagnostic recordings. The timestep grid is 0.05/0.1/0.2/0.3/0.6 ms; E/I holds are exactly 24/12, 12/6, 6/3, 4/2 and 2/1 steps. The two coarsest conditions last 199.8 ms; the others last 200 ms. Unchanged consumers retain their original bank pins.
- **Reference clock:** retained `F_GAMMA_HZ = 43.95` numerically and clarified its role as a prior-spiking reference defining the approximately 22.8-ms replay clock. It is not the retained exp041 6-ms median (59.136573 Hz), the new exp033 Hopf frequency, or a detected-cycle estimate. Changing it would change downstream scientific protocols and is not justified by this refractory reconciliation.
- **Writings:** revised exp022, exp023, exp033, exp044 and exp054; verified exp110's step-5.10 revision against the same accepted evidence. The articles distinguish new training/theory from reused spikes, document the actual timestep durations and fixed refractory periods, report the new exp044 accuracy/rate range and exp033 onset (0.594 nA, 27.6 Hz), and explain that the theory changed both refractory parameters and gain evaluation. Separately trained timestep conditions are not presented as fixed-weight numerical convergence. The five substantive article edits are dated 2026-09-09; all six affected writings carry `data` and `v36.0.0`. Exp081's previously flagged missing availability tag is corrected to `data`; its cross-link rule was checked and guide tag advanced to `v36.0.0`, preserving its author-assigned `reviewed` tag, scientific prose and dates. Other author review tags are unchanged.
- **Presentation-only cleanup:** created `exp033-r015-present` from unchanged `exp033-r012-analyse`, replacing an ambiguous sensitivity annotation with “onset criterion met”. Its `numbers.json` is byte-identical to r014; seven exports, payload digest `sha256:0d82cbed8f973735ba236cc668308e61f02ac56c58cbb4b02c41b9faee99106f`. Created `exp022-r014-present` from unchanged `exp022-r010-analyse`, removing internal run-ID stamps from seven learning-curve SVGs. All 35 PNGs are byte-identical to r013; numbers differ only in the presentation identity, not scientific content; 43 exports, payload digest `sha256:4aacfbc5dc72727c91ba806974d7e6d7ea33d7c23c22a8821b5d9c06a0e57d41`. Both preserve their exact analysis input references. Historical presentations remain intact. Exp054 r014 and exp110 r021 remain the accepted presentations for their respective refreshed results.
- **Validation:** 102 relevant implementation tests passed, one existing article-rendering test was deselected from the final regression command, and one warning came from deliberately evaluating the historical unstable gain path. Ruff and `git diff --check` passed. Read-only discovery validated 83 present runs. The final Demolab build succeeded for all 38 entries, including the metadata-only exp081 correction. All 50 directly referenced scientific images in the six affected article HTML files match the selected presentation files by SHA-256. New exp022 and exp033 scientific figure exports were visually inspected; article layout inspection on localhost remains with the author, as required by repository policy. No new automated writing tests were created.
- **Evidence record:** `/Users/eoin/.codex/execution-snapshots/refractory-step7-qa/validation.json` records the source map, figure checks, presentation provenance and recipe comparison. The same directory holds pre-edit article snapshots, scoped article diffs, the historical recipe snapshot, discovery output and figure-inspection images.
- **Scope:** no HPC submission, training, new scientific simulation, materialization, deployment, stored-run mutation, commit or push occurred in this step. Exp110's unfinished Methods/appendix scaffolds remain outside this targeted reconciliation; author layout review is not claimed complete.

## 8. Completion criteria

**Complete — 2026-09-09. All seven criteria passed.**

- [x] The collection explicitly requests 1.2/0.6 ms throughout its spiking, mean-field and independent reference implementations.
- [x] The timestep grid and any quantization convention are documented and agree across implementations.
- [x] Full-network checks establish that preserved dt = 0.1-ms execution remains unchanged.
- [x] The new bank contains exactly 90 reused models and 12 replacement/new timestep models, with validated provenance.
- [x] Exp044 and mean-field calculations have replacement evidence; exp054 preserves its spiking probes while using refreshed theory.
- [x] Exp110 and affected articles reflect the replacement evidence and adopted model.
- [x] Historical runs remain intact and readable, unchanged consumers retain valid pins, and exp048, exp111 and noncollection experiments remain outside the rerun scope.

The completed numerical work comprised **12 training runs, timestep evaluations and revised mean-field calculations**, plus the planned diagnostic recordings. No additional refractory compute is required by this final audit.


### Step 8 final audit — 2026-09-09

| Completion criterion | Evidence and outcome |
| --- | --- |
| Explicit collection model | Current spiking recipes declare 1.2/0.6 ms with exact conversion; exp022 training commands are checked by the collection tests. Exp033 v2 and exp054 v6 mean-field gains use the same physical durations. The Brian2 and graph reference checks explicitly use this model. Generic simulator/reference defaults, passive models and historical recipes remain distinct. **Passed.** |
| Timestep and duration agreement | Fresh timing/collection tests verify the five-condition grid, E/I counters 24/12, 12/6, 6/3, 4/2, 2/1 and trial steps 4000/2000/1000/666/333. Unsupported 0.25-ms exact conversion is rejected. The documented 199.8-ms coarse trials agree with saved replacement evaluation provenance. **Passed.** |
| Preserved 0.1-ms execution | Current `models.py`, `config.py` and `timing.py` SHA-256 values exactly match the adopted files tested in step 4. The recorded 180 full-network checkpoint comparisons, four backward comparisons, independent references and compiled CPU/CUDA checks therefore remain applicable; those expensive simulations were not repeated during this closeout. **Passed.** |
| Complete replacement bank | Reran the read-only independent bank validator: 102 model identities, 90 reused/12 new, 204 checkpoint roles, 34 diagnostic recordings and 442 export files. Reused cell bytes, new checkpoint training-completion records, diagnostic lineage and explicit refractory commands all match. Bank payload remains `sha256:6bfeda8ce5e32bb35748f335338ee6af29bedf76ac606babc9823a466da640c0`. **Passed.** |
| Replacement science and preserved spikes | Accepted exp044 and exp033 run identities/digests match their acceptance records. Exp054's current analysis still pins `exp054-r008-compute` separately from `exp033-r012-analyse`; the original spike payload digest is unchanged. Prior raw-spike, timing, theory and numerical-preservation acceptance remains valid. **Passed.** |
| Writings and synthesis | The built source map agrees with the six accepted presentations listed in step 7, including exp110 r021's corrected exp054 analysis and exp044 presentation references. Affected articles retain `data` and `v36.0.0`; step 7's successful 38-entry build and 50 embedded-figure checks are retained as evidence. No article changes were necessary during this audit. **Passed.** |
| Historical integrity, pins and exclusions | Fresh discovery validates 168 completed v4 runs and returns 83 qualifying presentations. All 249 operational input references resolve to matching payload digests. All seven explicitly recorded step-4 baseline digests and 28 run references found in the later QA records match current runs. Exp054's ten historical recipe/profile definitions remain unchanged. The active registry has eighteen collection members and excludes exp048/exp111; the existing user-authorized removals are not reversed. **Passed.** |

**Fresh validation:** 38 collection/timing tests passed; independent full-bank acceptance and read-only discovery passed. No unresolved failure was found. The final audit report is `/Users/eoin/.codex/execution-snapshots/refractory-step8-qa/validation.json`; its directory also contains the new bank-validation and discovery outputs. The report records exact run references, implementation hashes and article hashes so these conclusions remain tied to the inspected state.

**Closeout:** corrected the stale opening status paragraph. No experiment code, article prose, model data, source selections or completed run contents changed in step 8. No HPC work, scientific execution, publication, commit or push was performed. Author localhost layout inspection remains pending and is not implied by completion of this refractory plan. Exp110's unrelated unfinished manuscript scaffolds remain separately scoped.
