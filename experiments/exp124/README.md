# SHD

Collection: `datasets`. Current full-data training and matched inference, 15 October 2026.

The experiment declares both networks directly in `recipe.py`. New training uses
SNNLab 0.14.0 `snnlab.sim.execution.train`; the historical 20-epoch results below
were generated with SNNLab 0.13.0 and the earlier explicit PyTorch training loop.
The report contains only the final two training curves and a two-layer sample
raster for each model. Its explicit presentation pin is `exp124-r151-present`.
The retained local chain is `exp124-r149-compute` → `exp124-r150-analyse` →
`exp124-r151-present`. The compute contains both trained models and the matched
inference recordings; analysis measures both and presentation consumes that one
analysis input.

```sh
.venv/bin/python experiments/exp124/analyse.py --source exp124-r149-compute
.venv/bin/python experiments/exp124/present.py --source exp124-r150-analyse --final-report
```

The execution history below records earlier identities and availability at the
time. Superseded local runs were pruned after the explicitly authorized
consolidation; the current retention state is described at the end.

## Native training migration

The native trainer now owns initialization, stratified minibatches with merged
small tails, Adam parameter groups, updates, constraints, online training metrics,
full validation, epoch callbacks, parameter observations and final checkpoints.
The Lang declarations contain the actual initializers rather than zero placeholders
followed by manual mutation. Adam, the centered readout initializer, identity gain,
rate-normalized scoring, equal-class objective and noisy SHD event encoder use
public registered extensions where the standard library lacks that exact policy.
Checksum-verified acquisition and the utterance split remain experiment-owned.

Each condition exports raw `native_metrics.json`, native `manifest.json` and
`tensors.npz` checkpoint files, and a `timing.json` from SNNLab's synchronized
training-timing API. Its disjoint phases cover setup, minibatch training (including
encoding and optimization), training-set evaluation, validation, checkpointing and
reporting. Online training has zero separate training-set evaluation time.
`duration_seconds` is the native total less setup; lazy CUDA compilation may occur
inside the first training/validation phase. Data acquisition, caller-side initial
weight auditing, Modal startup and transfer are outside the native train-call timer.

SNNLab 0.14 rejects split population-rate observations with online training
metrics. This experiment keeps online metrics and native parameter observations;
activity comes from separate inference rasters. New checkpoint inference uses
`infer` and typed `RecordingSpec`/`SignalRecording` without altering the trained
graph. A CPU check matched direct replay scores and every E/I spike exactly for
both models. Legacy NPZ-only sources retain their explicit replay path.
Native initialization draw order
and indexed corruption/batch seeds differ from the old loop. The physical recipe
is preserved, but the new trajectory is not a bitwise continuation of the retained
20-epoch runs. COBA and PING are still initialized and presented identically to
one another. Native checkpoints include optimizer, RNG and data-order state;
resumption currently requires the original exact dataset path/protocol, so the
archived checkpoint is not advertised as automatically portable after Pingstore's
atomic temporary-directory rename or a Modal download.

Small checks on 14 October 2026:

1. `exp124-r132-compute` → `exp124-r133-analyse`: 20 training / 20 validation,
   two epochs, both models; finite final weights, all learned matrices changed,
   matching paired initial weights, native checkpoint/export equality and valid
   ancestry. Explicit encoder comparison matched the old binning/corruption for
   the same input seed. Native full-data batching gives 28×256 + 172 examples.
2. Native interruption after one update and checkpoint resumption reproduced
   uninterrupted final weights and epoch metrics exactly for both networks.
   This used the same stable dataset path, two epochs, one batch per epoch.
3. `exp124-r134-compute` → `exp124-r135-analyse`: 200 training / 100 validation,
   20 epochs, batch 64, four local CPU threads. COBA train/validation accuracy
   25.5%/17%, cross-entropy 2.7851/2.8319; PING 60.5%/26%, 1.7351/2.2345.
   Native train-call times were 54.735/82.097 s: training 38.513/60.995 s,
   validation 15.595/20.476 s, setup 0.453/0.245 s and checkpoints 0.033/0.034 s.
   These reduced-data checks are not full-data performance claims.

4. `exp124-r137-compute`: final CPU adapter check, one epoch / 20 training /
   20 validation, both models; corrected compatibility epoch timing is positive.
5. H100 checks `exp124-r138-compute` (COBA) and `exp124-r139-compute` (PING)
   → paired `exp124-r140-analyse`: 200 training / 100 validation, three epochs,
   batch 256. Native checkpoints matched final NPZ weights exactly; all three
   learned matrices changed, every fixed parameter stayed fixed, final weights
   were finite, and paired data/initialization checks passed on SNNLab 0.14.0.
   These are three-update GPU plumbing checks, not learning-performance evidence;
   the 20-epoch CPU pilot above establishes reduced-data learning. Native call
   times were 26.128/40.783 s, with training 12.925/23.250 s and validation
   10.738/15.041 s including initial validation and lazy kernel compilation.
   Warm epochs measured about 2.5+1.6 s COBA / 4.1+2.4 s PING training+validation.

Modal's `ImageGetOrCreate` lookup repeatedly lost its connection before workers
or reservations existed. Subsequent checks reused the exact successfully built
image `im-t6qOAS12AWPIYJmweacQI1` through `PINGLAB_MODAL_IMAGE_ID`, while uploading
fresh sources. The helper checks installed SNNLab against the submitted lock
before execution. A PING pre-worker upload was rejected because its helper file
changed during upload; sources were then frozen and the retry passed. These
pre-worker failures produced no training runs. Do not edit mounted experiment or
tools files during a new Modal source upload.

The failed pre-training checks `.exp124-r130-compute.tmp`,
`.exp124-r131-compute.tmp` and `.exp124-r136-compute.tmp` retain
configuration/API failure evidence. The H100 r136 check failed in initial
validation because the native objective validator distinguished `cuda` from
`cuda:0`; execution now resolves the concrete device index before training. They are
not downstream inputs. The existing report pin remains on validated 20-epoch
results until replacement full-data training and matched inference complete.

## Retained 20-epoch report evidence

| Run | Purpose |
| --- | --- |
| exp124-r105-compute | COBA full-data training and final learned weights |
| exp124-r104-compute | PING full-data training and final learned weights |
| exp124-r120-analyse | Paired training measurements and compatibility validation |
| exp124-r127-compute-phase-sample-raster | Fresh matched inference with final weights, no optimizer updates |
| exp124-r128-analyse | Sample predictions, per-layer spike counts and mean firing rates |
| exp124-r129-present | Both training curves and both two-layer raster figures |

The presentation references the training analysis and sample-raster analysis
and recording explicitly. Both branches retain the same two trained compute
inputs. No historical smoke, calibration or pilot run is needed to render the
current report.

## Training recipe

COBA has two feedforward layers of 128 excitatory conductance-based LIF neurons,
without inhibitory populations. PING has two layers of 128 E / 32 I neurons,
with fixed reciprocal feedback. Both use 700 cochlear spike inputs and 20 signed,
non-spiking leaky-integrator readouts. Only input→E1, E1→E2 and E2→readout
weights are learned, with identical initialization across models.

The default `calibrated` profile uses 1 ms steps and one-second presentations.
E/I membrane time constants are 20 ms, AMPA/GABA decays 10 ms, and refractory
periods zero. Rest/reset is −65 mV and threshold −50 mV. PING summed incoming
E→I and I→E strengths are 0.05 μS each. Input/interlayer Gaussian parent means
are 0.09 μS with SD 0.009 μS, source-fan-in and retained-fraction normalization,
and 95%/90% initial zeros without permanent masks. Readout initialization is
centered. Scores use mean readout voltage, fixed gain 20 and E2-rate normalization
with a 20 Hz reference and 1 Hz denominator floor.

Adam uses betas 0.9/0.999, zero weight decay and input/interlayer/readout learning
rates 0.000005/0.00002/0.01. Class-stratified ordering retains each utterance once
per epoch without oversampling; the objective averages cross-entropy within
represented classes and then across classes. Reported cross-entropy is the
unweighted example average. Batch size 256 produces 29 updates per epoch,
including 172 examples in the final batch. Tails smaller than 20 are merged
when applicable. Twenty epochs give 580 updates per model.

The built-in surrogate uses slope 100/15 per mV and gradient scale 0.01.
Voltage-gradient damping is 1 for both models. No clipping or firing-rate
regularization is applied. CUDA uses compiled neuron kernels, fast projection
arithmetic, structural extension validation and no discarded forward snapshots;
CPU inference is eager. Training corruption inserts events at 1.6 Hz per channel
and deletes occupied events with probability 0.0008. Evaluation corruption is
fixed across epochs. Every presentation resets neural state.

All 8,156 official training utterances are retained: seed-42 per-class 10%
holdout gives 7,340 training / 816 validation examples with natural class counts.
The split is utterance-disjoint, not speaker-disjoint. The official test set is
untouched, and these are development-validation results.

## Measured full-data results

| Model | Online train accuracy / loss | Final validation accuracy / loss | Condition time |
| --- | --- | --- | --- |
| COBA | 64.9% / 1.1037 | 65.2% / 1.1388 | 1546.40 s (25.77 min) |
| PING | 76.5% / 0.6955 | 76.1% / 0.7518 | 2163.20 s (36.05 min) |

The workers ran concurrently on separate H100 GPUs with four CPU threads and
16 GiB RAM each. The compute-stage interval spanned 2,200 s (36.67 min), not
the 3,709.60 s sum of condition times. Condition times exclude data acquisition,
network construction, initialization, launcher startup and transfer; stage span
also includes network setup and result retention. Both apps and launchers ended.
No resumable training checkpoints were written.

Training metrics aggregate pre-update minibatch scores, without a separate
training-set evaluation. Validation uses final epoch weights over the complete
holdout. Epoch zero has validation only. All three learned matrices changed and
final weights were finite; per-batch gradient diagnostics were not measured.
Paired analysis checked scientific settings, exact spike arrays and initial
learned matrices. Previous development measurements are historical, not additional
replicates of the current recipe.

## Matched sample inference

The sample-raster phase replays final weights on the first validation utterance
in source order: official-training index 458, class 0. It was selected before
examining predictions or activity. A fresh seed-46 corruption draw is shared
exactly by both models. CPU inference records every E/I neuron's binary spikes
at all 1,000 timesteps, with no optimizer updates. Both models predicted class 0.

The COBA figure stacks layers 1 and 2 with 128 excitatory neurons each. The PING
figure stacks its two layers, each with black E ticks below red I ticks. Inhibitory
neurons have identical spike trains within a layer because their parameters and
uniform inputs are identical. These are single-example rasters, not population
averages or a general gamma-frequency estimate.

## Explicit stage commands

These commands operate on the retained training inputs; each creates a new run
and prints its identity. Substitute actual returned IDs in subsequent commands.

```sh
.venv/bin/python experiments/exp124/analyse.py --source exp124-r105-compute --source exp124-r104-compute
.venv/bin/python experiments/exp124/compute.py --phase sample-raster --source exp124-r105-compute --source exp124-r104-compute --device cpu --cpu-threads 4
.venv/bin/python experiments/exp124/rasters.py --source <sample-raster-compute-id>
.venv/bin/python experiments/exp124/present.py --source <training-analysis-id> --final-report --raster-analysis <sample-raster-analysis-id>
```

Full-data Modal training uses `experiments/helpers/modal_compute.py` with
`--experiment exp124 --variant calibrated --full-dataset --epochs 20
--batch-size 256 --gpu H100 --cpu 4`, selecting `--models coba` and `--models ping`
in separate concurrent invocations. Include PyYAML in the Modal CLI environment:
`uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run --detach`.
Do not run training merely to regenerate figures.

## Simplification and pruning

At the author's request, the report was reduced to the final training curves and
fresh matched rasters under Writing Guide 36.0.0. The new presentation retained
all final training numbers and both training PNGs exactly, while dropping its
historical presentation dependency. Superseded local runs were pruned only after
source/lineage validation, figure inspection, a successful report build and a
hash-bound dry-run/confirmation. The six runs listed above remain necessary
ancestry. Other experiments were outside the prune scope. The two inactive
historical failure reservations were removed under the exclusive operation lock
after checking writer locks and absence of checkpoints. No remote backup objects
were deleted. No automated tests, commit, push or publication were performed.

The earlier diagnostics document is a historical development account; its run
IDs are no longer retained locally. Optional gamma-probe code remains available
for a newly requested replay of the final trained weights.

## Native 50-epoch full-data execution

The validated native checks authorized the following concurrent submissions on
14 October 2026. Both use seed 42, calibrated physical settings, 7,340 training /
816 validation utterances, batch 256, 50 epochs (1,450 updates), one H100 / four
CPU cores / 16 GiB each, SNNLab 0.14.0 and a 24-hour function timeout.

| Model | Compute run | Modal app | Launcher session | Log |
| --- | --- | --- | --- | --- |
| COBA | exp124-r142-compute | ap-FGWK2fQUBDoDnoFnTNO7ni | 32412 | /private/tmp/exp124-native-full50-h100-coba.log |
| PING | exp124-r141-compute | ap-TLrJIwBcB9cEOc2uYD360E | 67485 | /private/tmp/exp124-native-full50-h100-ping.log |

Both were dispatched concurrently using the validated built image
`im-t6qOAS12AWPIYJmweacQI1`; current sources and the dependency lock were uploaded
and the worker verifies its installed SNNLab version against that lock. No full
50-epoch metrics are available at submission. The existing r129 presentation and
20-epoch evidence stay intact until completed native sources, analysis, fresh
matched native checkpoint inference and final presentation validate. The report
will retain only its two training curves and two two-layer inference rasters,
with native timing phases reported in prose/Methods rather than extra figures.


### Terminal status recorded 15 October 2026

COBA `exp124-r142-compute` completed 50 epochs / 1,450 updates and downloaded
successfully; its launcher exited zero and Modal logged app completion. Shared
v4 source validation and native checkpoint loading passed. Final weights match
the native checkpoint exactly, all three learned matrices changed, fixed
parameters stayed unchanged, and all final weights are finite. Final online
train accuracy/cross-entropy were 83.9646%/0.528120; validation 82.5980%/0.591874.
Native train-call time was 3,512.481828 s: setup 9.066445 s, training
3,160.174714 s, validation 342.373319 s, separate training-set evaluation zero,
checkpoint 0.109301 s and reporting 0.758050 s. Condition time (total less setup)
was 3,503.415384 s / 58.3903 min.

PING `exp124-r141-compute` received a cancellation signal at
2026-10-14 23:01:51 UTC and logged successful cancellation at 23:01:52 UTC,
followed by `Runner terminated`. The launcher exited one with
`RemoteError: Function call was cancelled by user or a failure.` The available
log does not identify the initiator or underlying cause. The last completed
epoch was 32, with logged online train accuracy/cross-entropy 84.5%/0.4821 and
validation 81.1%/0.5772. These rounded log observations are not a completed
validated compute source. No replacement run was submitted.

The local PING reservation contains only its reservation record and README;
no checkpoint or scientific exports were downloaded. The result-volume listing
for dispatch `61042a8c579142fabb3eebf1a20f092d` returned no such directory. The
native checkpoint was configured for final completion, so this workflow has no
retained PING epoch-32 weights to resume or replay. The hidden r141 reservation
is retained as cancellation evidence alongside the pre-training failure
reservations r130/r131/r136; none is a downstream scientific input.

The progress heartbeat was disabled after handling this terminal interruption.
The paired native analysis, new matched inference and replacement presentation
were not executed because PING has no completed source. The report remains
pinned to `exp124-r129-present`, preserving the previous validated paired
20-epoch curves and rasters. A new paid PING training run requires explicit
authorization under the monitoring instruction; no restart, pruning, automated
tests, commit, push or publication was performed in this failure-handling step.

### Authorized PING replacement — 15 October 2026

Anomancer explicitly authorized restarting PING from scratch after the terminal
cancellation above. New run `exp124-r143-compute` was dispatched on Modal app
`ap-9SjHIeuBldOYKXfHfWyC5H`, launcher session `14951`, with log
`/private/tmp/exp124-native-full50-h100-ping-restart.log`. It uses the same
full-data calibrated native recipe, seed 42, batch 256 and 50 epochs on one H100 /
four CPU cores / 16 GiB, with the validated dependency image and lock guard.
The authored scientific configuration was checked against completed COBA r142
before submission. The cancelled r141 reservation remains failure evidence;
it is not a training source, and the replacement does not resume its weights.

COBA r142 is reused; no additional COBA training was submitted. Consequently,
the final retained native pair will be separate executions rather than two
concurrent completed runs: the original concurrent PING attempt was cancelled.
Report timing and Methods must reflect this when the replacement completes.
Monitoring has been reactivated. On completion the explicit paired inputs are
COBA r142 and PING r143, followed by fresh matched native inference and the
four-figure presentation. The old r129 report pin remains until this workflow
validates. At dispatch no replacement epoch metrics are available.

### Replacement interruption — 15 October 2026

Replacement PING r143 also terminated before completion. Local launcher output
and independently retrieved Modal app logs both record a cancellation signal at
2026-10-15 07:52:42 UTC and successful cancellation at 07:52:43 UTC. Launcher
14951 exited one with the same `Function call was cancelled by user or a failure`
error. Neither log identifies the cancellation initiator or underlying cause.
There is no preceding numerical exception or stated out-of-memory/timeout error
in the retrieved logs. The submitted helper requests a 24-hour function timeout;
that configured limit does not explain this interruption.

The last completed epoch was 33/50: logged online train accuracy/cross-entropy
85.4%/0.4686 and validation 79.8%/0.5947. These are rounded log observations,
not a validated completed result. The local r143 reservation contains only its
reservation record and README. Listing result-volume dispatch
`4b96b8409eba4e05b2ed1e58de446333` returned no such directory. Final-only
checkpoint persistence and success-only result transfer left no retained
checkpoint/scientific export for this attempt. The r143 hidden reservation and
local logs are retained cancellation evidence, alongside r141 and the earlier
pre-training failures r130/r131/r136.

The progress monitor was disabled after recording this second interruption.
No further run was submitted. Completed COBA r142 and the old r129 report pin
remain intact. Paired native analysis, matched inference and replacement
presentation cannot use the incomplete r143 source. A further paid retry should
first address durable intermediate checkpoints and investigate cancellation;
no restart or execution-code change was made during this monitoring step.

### Durable Modal dispatch — 15 October 2026

The cancellation investigation found that both interruptions followed Mac sleep
by approximately two minutes. Local power logs show clamshell sleep at
14 October 23:59:47 BST followed by cancellation at 15 October 00:01:51 BST
(124 s), and idle sleep at 15 October 08:50:50 BST followed by cancellation at
08:52:42 BST (112 s). The old launcher used synchronous `.remote()`; Modal's
caller-disconnection cancellation policy applies even when the app is detached.
This is the supported explanation for the two cancellations, rather than the
configured 24-hour training timeout.

The shared launcher now reserves once, calls `.spawn()`, persists the call ID and
returns. Invoke it with `modal run --detach`. Check or collect the explicit new
run with `experiments/helpers/modal_jobs.py --run-id <id> --collect`; this does
not build/upload code or submit another job. A periodic monitor can run that
command every five minutes and retrieve app logs separately. A durable volume
completion marker allows collection independently of the local dispatcher or
call-result expiry. See [Modal helper instructions](../helpers/MODAL.md) for the
complete commands and recovery semantics.

Nine operational tests passed, including interrupted-download recovery and
rejection of incorrect identity/digests. A 30-second CPU-only Modal probe proved
that a fresh process could retrieve a result after its submitting process had
exited while the call was still pending. No SHD training or GPU was used for
that live probe, and no replacement training run was submitted by this change.
Historical r141/r143 cancellations and their reservations remain unchanged;
COBA r142 and the old r129 report pin remain intact. Intermediate training
checkpoint persistence is still a separate limitation of exp124's final-only
checkpoint policy.

### Durable full-data PING restart — 15 October 2026

Anomancer explicitly authorized a fresh 50-epoch PING run using the durable
launcher. `exp124-r144-compute` was submitted at 09:01:59 UTC to H100/4 CPU
with the calibrated native SNNLab 0.14.0 profile, 7,340 training and 816
validation utterances, batch 256, and seed 42. Its scientific configuration
matches completed COBA r142 apart from the model selection.

Modal app `ap-1Zeuijk5s9FPpHQuzOi4Tm`, durable call
`fc-01M4ZCR02KXYHC2AK3E8KX8SKP`, and dispatch
`204ae16d1e6443469244032b6bfa9df0` are recorded in the reservation's
`.scratch/modal-dispatch.json`. The detached launcher exited successfully;
a fresh process subsequently confirmed that the call remained pending.
Launcher exit and its final app-log message do not indicate training completion.

At dispatch, the five-minute monitor was configured to check and collect r144 through
`modal_jobs.py --run-id exp124-r144-compute --collect`, retrieving remote app
logs separately. No local launcher needs to remain connected during training.
On completion, the intended retained pair is COBA r142 and PING r144, followed
by paired analysis, fresh checkpoint inference for both-layer rasters, and the
four-figure report presentation. These final executions are separate, not
concurrent. The report remained pinned to r129 until the new workflow validated.
No r144 epoch measurements were available at dispatch. Failed r141/r143
reservations remain cancellation evidence; the final-only checkpoint limitation
is unchanged.

### Durable native full-data results — 15 October 2026

PING r144 completed all 50 epochs and 1,450 updates, and the independent
collector downloaded and validated its output after the submitting process had
exited. The native checkpoint exactly matches every final NPZ parameter; all
three learned matrices changed, fixed parameters stayed unchanged, final weights
are finite, and epoch history covers 0–50. Paired analysis also validated exact
stimulus arrays, dataset/configuration compatibility and initial learned matrices
against completed COBA r142.

| Condition | Compute | Final online train accuracy / CE | Final validation accuracy / CE | Condition time |
| --- | --- | --- | --- | --- |
| COBA | `exp124-r142-compute` | 83.9646% / 0.528120 | 82.5980% / 0.591874 | 3,503.415384 s (58.3903 min) |
| PING | `exp124-r144-compute` | 88.9782% / 0.342976 | 84.4363% / 0.472460 | 6,048.826863 s (100.8138 min) |

SNNLab's synchronized native timing recorded the following disjoint phases:

| Phase | COBA seconds | PING seconds |
| --- | ---: | ---: |
| Setup | 9.066445 | 8.236962 |
| Minibatch training, including encoding/forward/backward/optimization | 3,160.174714 | 5,497.042891 |
| Validation, including epoch zero | 342.373319 | 549.759185 |
| Separate train evaluation | 0 | 0 |
| Checkpoint persistence | 0.109301 | 0.129249 |
| Reporting | 0.758050 | 1.895539 |
| Native total | 3,512.481828 | 6,057.063826 |

Condition time is native total minus setup, excluding caller preparation,
acquisition and result transfer. Summed condition time is 9,552.242247 s
(159.2040 min), not concurrent elapsed time. COBA executed from 14 October
21:55:22–22:54:23 UTC; PING executed from 15 October 09:02:58–10:44:25 UTC.
The 12 h 49 min 3 s span between the first worker start and final completion
includes the overnight gap and must not be described as training duration.
Cancelled attempts are outside these condition timers.

The explicit final chain is paired training analysis `exp124-r145-analyse`,
fresh matched checkpoint inference `exp124-r146-compute-phase-sample-raster`,
raster analysis `exp124-r147-analyse`, and four-figure presentation
`exp124-r148-present`. Native inference recorded both layers for each model on
the first validation utterance (official-training index 458, class 0), with the
same fresh seed-46 corruption draw and no optimizer. Both predicted class 0.
PING inhibitory neurons were synchronous within each layer; the single example
is illustrative and is not a gamma-frequency measurement.

The first CLI raster invocation failed before reserving a run because importing
the command entry point a second time registered the encoder twice. Passing its
already-loaded batch encoder into raster inference fixed this call path; no
training or scientific parameter change was needed. All four new figures were
inspected, and the report now pins r148 with native Methods and bound timing
values under Writing Guide 36.0.0. The previous 20-epoch evidence remains intact.
Hidden r130/r131/r136 pre-training failures and r141/r143 cancellation
reservations are deliberately retained as failure evidence, not analysis inputs.

Final validation: the static site build passed (43 pages); rendered exp124
values, native timing fields and all four r148 figure links were inspected.
Read-only Pingstore discovery validates and exposes r148 alongside the retained
r129 presentation. The Modal app is stopped with zero tasks, repeated collection
returns `collected`, and the r144 temporary reservation is gone; no result
transfer remains pending. The progress monitor was disabled after completion.
No automated tests, commit, push or publication were performed for this result
workflow.

### Self-contained three-run consolidation — 15 October 2026

Anomancer explicitly requested merging the completed COBA and PING compute
results, rerunning analysis and presentation, and retaining only those three
exp124 runs locally. This is an authorized evidence replacement, not new
training and not ordinary phase assembly with retained input dependencies.

`consolidate.py --source exp124-r142-compute --source exp124-r144-compute
--raster-source exp124-r146-compute-phase-sample-raster --replace-sources`
created `exp124-r149-compute`. It contains the shared SHD spike arrays, both
complete native model/checkpoint bundles, and matched raw inference rasters.
The consolidation checked configurations, dataset/version identity, exact
initial learned matrices, 1,450-update checkpoints, complete epoch histories,
finite final weights and learned/fixed parameter changes. All 20 copied
scientific files were verified byte-for-byte against their sources.

The merged run has no live upstream run dependencies. Its execution metadata
preserves the complete original COBA, PING and inference run records, source
payload digests and per-file source/checksum mappings as historical provenance.
The original separate H100 executions and their native timings remain distinct
from the local consolidation operation. Original native checkpoint protocol
paths are preserved; consolidation does not promise portable training resume.

Fresh `exp124-r150-analyse` consumes only r149 and measures both learning and
the included inference. Fresh `exp124-r151-present` consumes only r150 and
produces the same two training curves and two both-layer rasters. Training
results and raster measurements matched the old analyses exactly; presentation
numbers matched exactly and all four PNGs were byte-identical. The report now
pins r151; its scientific claims and metrics are unchanged. Writing Guide
36.0.0 tags and the existing substantive date were preserved.

The static site build passed with all four r151 figure links and unchanged
rendered metrics/timings. A reviewed exp124-only prune plan
`sha256:6ee53221c58ed8d6c4e2a1a03949b6a0f469c96536c464e1af4543583c6d7d40`
removed 20 superseded completed runs (0.67 GiB). Five inactive failed
reservations r130/r131/r136/r141/r143 were removed under the shared exclusive
operation lock after dead-writer, checkpoint/transfer and job-status checks.
Exactly r149 compute, r150 analyse and r151 present remain for exp124; other
experiments were outside the pruning scope. Historical availability statements
earlier in this README describe their state before this authorized pruning.

After pruning, native CPU inference from both consolidated checkpoints exactly
reproduced the retained class scores and every E/I spike. A second site build
passed (43 pages), and rendered Dataset provenance lists only r149/r150/r151.
Read-only discovery exposes the single r151 presentation. No exp124 temporary
runs remain. No training, commit, push or publication was performed by this
consolidation.
