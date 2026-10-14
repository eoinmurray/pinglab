# Current-based MNIST learning and inference timing

Executed on Modal on 14 October 2026. Article: Uncategorized (`demo`),
`experiments/exp126/report/report.mdx`; Writing Guide 36.0.0.

The fixed recipe trains a snnTorch `Synaptic` network (784 → 256 → 10) on
exactly 6,000 MNIST training images, with 1,000 separate validation images and
all 10,000 official test images. The training fraction is 10% of the official
60,000-image training split; validation does not reduce those 6,000 examples.
Class-balanced selection uses 600 training and 100 validation images per digit.
Both hidden and output neurons have decaying additive synaptic currents and
leaky membrane states. This is current-based: no conductance times
voltage-dependent driving force appears in the equations. Fixed defaults:
30 epochs, 50 timesteps, Adam, surrogate gradients, batch size 128, and
validation-loss checkpoint selection across trained epochs only.

## Execution commands

Run each stage independently from the repository root. Compute downloads MNIST
only when invoked. Neither analysis nor presentation runs training or inference.

```sh
# Local reference: same scientific recipe, four CPU threads.
uv run python experiments/exp126/compute.py --device cpu --cpu-threads 4

# Modal authentication, if not already configured:
uvx --from 'modal>=1.1,<2' modal setup

# Paid Modal GPU training; reserves and returns a validated v4 compute run.
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run experiments/helpers/modal_compute.py --experiment exp126 --gpu A10 --cpu 4

# Optional alternative GPU, without changing the scientific protocol:
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run experiments/helpers/modal_compute.py --experiment exp126 --gpu L4 --cpu 4

# Use the actual identity printed by the selected compute invocation.
uv run python experiments/exp126/analyse.py --source <compute-run-id>
uv run python experiments/exp126/present.py --source <analyse-run-id>
```

Do not launch both GPU commands by default; they are alternative devices.
There is no smoke profile for this experiment. The shared wrapper's `--smoke`
flag does not reduce its fixed recipe.

## Outputs and timing scope

Compute exports selected data and official indices, initial/selected/final
model parameters, per-epoch train/validation metrics, and official-test
predictions. Raw timings, exact configuration, dataset fingerprint, software
versions, device identity and execution provenance live in `run.json`.
Analyse checks partitions, checkpoint selection, finite metrics and parameters,
reconstructs test accuracy and confusion counts, and summarizes timing.
Present creates `training_curves.svg/.pdf`, `timing.svg/.pdf`, and `numbers.json`.
The latter includes a projection of benchmark metadata for article display.

Optimization timing excludes evaluation and checkpoint writing. The full
training-protocol timer includes epoch-zero evaluation, all 30 training epochs,
full training/validation evaluations, selected-checkpoint serialization, and
progress printing. Final-parameter serialization and selected-checkpoint loading
are outside this timer. Setup (including download, data export and upload to
the device) is timed separately. CUDA boundaries synchronize outstanding work.

Inference is warmed using validation images, then measured five times over the
complete test set with the same encoding seed each time. Its wall time includes
encoding, forward simulation, spike-count decoding, cross-entropy/accuracy and
CPU prediction transfer, but excludes dataset loading, host-to-device image
transfer, model loading and warmup. Reported ms/image is amortized batch time,
not single-image latency. Repeated passes characterize timing variability,
not accuracy uncertainty or independent learning replicates.

## Would Modal GPU be faster?

**Plausible, not established.** CUDA can accelerate the dense matrix operations
and keeps the input tensors and random encoding on the GPU, but 50 sequential
snnTorch steps launch many relatively small kernels. With only 6,000 training
images and 256 hidden neurons, launch overhead can compete with useful work.
A10 or L4 is a reasonable first benchmark; a larger GPU is not an established
improvement for this workload. Compare identical local and Modal protocols
before claiming a speedup. CPU/CUDA stochastic streams and numerical
trajectories need not match bitwise.

The training/inference timers exclude image building, provisioning, dataset
download and result transfer. These can dominate the end-to-end duration of a
short cloud job, so a faster GPU compute timer need not mean an earlier local
result. See [Modal GPU documentation](https://modal.com/docs/guide/gpu) and
[snnTorch Synaptic documentation](https://snntorch.readthedocs.io/en/latest/snn.neurons_synaptic.html).

## Recorded execution — 14 October 2026

Compute: `exp126-r001-compute`; analyse: `exp126-r002-analyse`;
present: `exp126-r004-present`. All completed and passed v4 validation.
The initial article pinned that presentation and declared `data`, with Writing Guide 36.0.0.
Modal app: [completed run](https://modal.com/apps/me-93863/main/ap-hHD56jQ5T1iT4QZSER0f8T).
Remote copy: `pinglab-compute-results/93b0576bfc8547f08995fb134f977793/.pingstore/runs/exp126-r001-compute`.

The NVIDIA A10 run selected epoch 30 by validation loss. Test accuracy was
90.35% (9,035/10,000), with mean cross-entropy 0.820822. Final training accuracy
was 92.82% and validation accuracy 90.1%. Optimization took 91.097 s; the full
training protocol took 122.397 s. Five warmed full-test inference passes had a
median duration of 1.446356 s (range 1.430474–1.460339 s), corresponding to
6,913.93 images/s and 0.144636 ms/image amortized batch time. All five passes
produced identical predictions at their fixed encoding seed.

This is one training replicate and one GPU run. No matched local execution was
performed, so CPU-to-GPU speedup remains unmeasured. Cloud startup, provisioning
and result download are outside the reported compute timing. The Modal app
completed, the results were downloaded and validated, and no task-owned hidden
run directories remain. No tests, publication, commit or push were performed.

## Equation-matched snnlab comparison — executed

The original baseline and its completed runs remain unchanged. The matched protocol is a separate condition family within exp126;
its executed v2 used snnlab 0.9.0 and the current v3 uses the 0.10.1 default structural validation policy;
its timings must not be paired with the original fused-linear baseline.

The reference uses snnTorch `Synaptic`; the replica uses snnlab `GraphExecutor`
with registered custom neuron and surrogate definitions. Both implement additive
synaptic-current decay (0.8), membrane decay (0.9), strict threshold comparison,
detached delayed subtractive reset, and snnTorch's fast-sigmoid derivative
`1 / (1 + 25 * abs(u))**2`. This is an executor/extension comparison, not a
benchmark of snnlab's stock CUBA_LIF reset or stock surrogate. The custom neuron
owns the synaptic filter; graph projections have exactly zero numerical decay
and supply instantaneous signed weighted drive. Biases are separate trainable
constant-input projections. Both conditions use the same unfused matrix-plus-bias
operations, initial tensors, fixed MNIST partitions, 30-epoch recipe and Adam.
No library code was changed to make one condition faster.

Input encoding uses a deterministic CPU uniform stream, copied to the GPU for
dense time-major Bernoulli thresholding. This avoids GPU-dependent random streams. Each batch
uses its own deterministic seed, independent of model execution. Workers hash
all encoded training and learning-curve evaluation batches; analysis requires
matching aggregate hashes, initial tensors, dataset checksums, software versions,
GPU model and requested CPU threads. Encoding and checksum auditing are timed
separately from optimization. Full protocol time includes both, evaluations and
checkpoint writing. Equal-model devices can have different clocks and shared
host conditions; repeats do not eliminate this source of hardware variation.

### Launch concurrently

```sh
# Both implementations: calls are spawned before waiting for either result.
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run experiments/exp126/modal_compare.py::compare --implementation both --seeds 42 --gpu A10 --cpu 4

# Optional three paired seeds: six independent phase runs, at most two active GPUs.
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run experiments/exp126/modal_compare.py::compare --implementation both --seeds 42,43,44 --gpu A10 --cpu 4

# Each implementation can also be launched independently, including concurrently
# from separate terminals. Use identical seeds and GPU settings for both.
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run experiments/exp126/modal_compare.py::compare --implementation snntorch --seeds 42 --gpu A10 --cpu 4
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run experiments/exp126/modal_compare.py::compare --implementation snnlab --seeds 42 --gpu A10 --cpu 4

# Assemble every selected completed phase explicitly before analysis.
uv run python experiments/exp126/assemble.py --source <snntorch-phase-id> --source <snnlab-phase-id>
uv run python experiments/exp126/analyse.py --source <assembled-compute-id>
uv run python experiments/exp126/present.py --source <comparison-analyse-id>
```

Choose one launch pattern; commands are alternatives, not a sequence to run
in full. Each worker requests one GPU and four CPUs. Modal may queue work if
account capacity is insufficient; the launcher permits concurrency but cannot
guarantee simultaneous provisioning. It uses no within-container input
concurrency, so implementations never compete for the same GPU. Each worker receives a fresh phase reservation before dispatch and completes
independently. The phase IDs have the form `exp126-rNNN-compute-phase-snntorch`
and `exp126-rNNN-compute-phase-snnlab`; authoritative scope and phase are also
recorded in `run.json`. All stages and phase names share one allocation counter. Completed
siblings are fetched even if another condition fails. There are no automatic retries or downstream stage launches. Assembly is an
explicit local command after the two phase downloads; analysis accepts one
assembled full compute identity and rejects direct phase inputs. The printed FunctionCall and Volume paths
identify remote work/results if the local client disconnects. Failed or interrupted
reservations remain hidden for reviewed cleanup or transfer recovery; rerunning
allocates new identities and must not substitute for checking remote status.

### Comparison outputs

1. `equation_agreement.svg/.pdf`: overlay hidden synaptic current and membrane,
   output spike marks, and gradient agreement for the first paired seed.
2. `learning_comparison.svg/.pdf`: train/validation loss and accuracy, validation
   accuracy versus cumulative optimization time, and paired test accuracies.
3. `timing_comparison.svg/.pdf`: individual encoding/forward/forward-backward/
   optimization-step measurements and five full-test inference passes per run.
4. `comparison_table.csv` and `numbers.json`: selected epochs, test accuracy,
   total optimization/protocol time, inference throughput, peak allocated CUDA
   memory, initial-state/spike/gradient discrepancies, and speedup ratios.

The initial parity probe uses eight validation examples and all 50 steps. State
and gradient comparisons require finite values, configured absolute/relative
error bounds, and zero spike disagreements. Input mismatches abort analysis.
Parity failures remain visible in the analysis, but speedup ratios are omitted.
This initial probe is not a claim of bitwise equality throughout training;
learning curves can reveal later divergence. Curve lines represent individual
training replicates, not pooled uncertainty bands.

Fixed-batch benchmarks use 128 validation examples, three warmup repetitions and
five timed repetitions per operation. Model weights reset before each repeat.
Forward-plus-backward includes cross-entropy and backward but excludes optimizer
updates; the full step includes zeroing, loss, backward, clipping and the first
Adam update, including lazy moment allocation. Model/optimizer construction is
outside timing. Forward-only excludes loss and runs under inference mode. The
microbenchmarks use preencoded events; encoding is its own operation. Full-test
inference includes encoding, loss, decoding and CPU prediction transfer, but
excludes input auditing, loading, checkpoint loading and warmup. All boundaries
synchronize CUDA. Peak memory means PyTorch allocated memory during the complete
training/test protocol, not total reserved or system-wide GPU memory.

The matched comparison has completed, and the agreement diagnostics passed.
Its validated presentation contains the three comparison figures and table.
The revised article pins the combined report presentation documented below,
which preserves the original baseline and adds the completed comparison.

## Phase assembly contract

`assemble.py` selects exactly one completed attempt for each implementation and
seed. It rejects full-condition inputs, unknown phases, missing/duplicate phase
coverage, unexpected exports, incomplete training/evaluation/benchmark evidence,
incompatible recipes, different initial tensors or data, mismatched encoded
input checksums, software/resources, and source provenance. The scientific code
fingerprint must match both phases and the current recipe implementation;
there is no implicit compatibility exception or selection of the latest attempt.

Assembly copies the recipe's eight scientific role files per condition into one
scientific unit named `snntorch-seed-42` or `snnlab-seed-42` (and corresponding
units for additional paired seeds). It creates a fresh `exp126-rNNN-compute`
with `scope: full`, `execution.operation: assemble`, explicit phase input pins,
a deterministic output mapping, its own export checksum and dated history.
It runs no training, inference or missing phase work. Source phase directories
remain unchanged and must accompany the full run for ancestry and timing provenance.

Comparison analysis reads scientific bytes from the assembled units and validates
them against the declared phase inputs; original phase records remain authoritative
for the actual device execution timings. Analysis references the assembled full
run, and presentation references the analysis. This produces two phase runs,
one assembled full compute run, one analysis run and one presentation run for
one paired seed. With three paired seeds, six phase runs assemble into one full
comparison run. The original full snnTorch baseline remains a separate supported
protocol with separately preserved evidence.

Shared Pingstore helpers now enforce phase identity/scope agreement, reserve
unique experiment-wide counters under an allocation lock, and require consumers
to explicitly opt into a named phase. Older v4 compute records without scope
remain unchanged. Phase and assembly regression coverage has been written but
not executed, following the repository's instruction not to run tests before
commit and push. The completed Modal executions are recorded below.

## Remote assembly option

Add `--assemble-on-modal` to the `both` launcher to explicitly request CPU-only
assembly on Modal after both GPU phases complete. It reserves a fresh full-run
identity locally, validates and reconstructs the phase sources from their
completed Volume copies, invokes the same `assemble.py` implementation, retains
the complete assembled result remotely, then downloads the phases and full run.
Analysis and presentation remain separate local commands. Remote assembly never
re-runs training or inference. The function wrappers use serialized dispatch;
use the documented isolated Python 3.12 client to match the image's Python version
without changing the project environment.

## Coallocated matched hardware

The independent-worker launcher can receive A10 and A10G variants from Modal;
raw CUDA random streams can then differ despite equal seeds. The v2 protocol
uses CPU random draws and audits the actual encoded tensors. For a strict
hardware comparison, use two GPUs in one allocation, with one concurrent worker
process per GPU and eight shared CPUs (four PyTorch threads per process):

```sh
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run experiments/exp126/modal_compare.py::compare_coallocated --seed 42 --gpu A10 --cpu-per-phase 4
```

This verifies equal GPU names before scientific work, downloads MNIST once to
avoid cache extraction races, starts both phase processes before waiting, and
explicitly assembles their completed Volume results on Modal before downloading
the full run and phase ancestry. CPU-host sharing can affect measured timings.
Encoding includes CPU generation and host-to-device transfer; the neural-kernel
microbenchmarks exclude encoding. Cloud-provider pinning was rejected by Modal
and is not part of this launcher. The first v1 pair remains separate evidence
and is rejected by the v2 assembler because its scientific source/configuration
and encoded inputs differ.

## Completed paired L4 execution — 14 October 2026

The two GPU processes ran concurrently in one Modal allocation with two NVIDIA
L4 GPUs and eight shared CPUs (four PyTorch threads per process). The unallocated
two-A10 request was stopped before switching to L4. Both phases completed the
full 30-epoch recipe, then Modal assembled a new full compute run without
additional training or inference. The full result and its immutable phase
ancestry were downloaded, validated, analysed and presented locally.

1. snnTorch phase: `exp126-r016-compute-phase-snntorch`.
2. snnlab phase: `exp126-r017-compute-phase-snnlab`.
3. Remote assembled full compute: `exp126-r018-compute`.
4. Local analysis: `exp126-r019-analyse`.
5. Local presentation: `exp126-r020-present`.

Modal app: [completed paired run](https://modal.com/apps/me-93863/main/ap-a7U57vV8dKtYtGdYljv6UT).
Remote results remain in `pinglab-compute-results`: phase dispatches
`2e0c5679a1f4438ab17b058fd3e85a9c`, `dc8de631b48b4bef87339705fb2745b5`;
assembly dispatch `7ae7765ffc7c4f9b91a314fe2b371951`.

Initial currents, membranes, spikes, scores and loss agreed exactly; maximum
absolute parameter-gradient difference was 3.814697e-6, within the predeclared
tolerance. Dataset, initial parameters, encoded input checksums, hardware model
and software matched. Subsequent training trajectories diverged.

| Measurement | snnTorch | snnlab |
| --- | ---: | ---: |
| Selected epoch | 24 | 19 |
| Test accuracy | 88.04% | 88.74% |
| Optimization time | 158.402 s | 277.486 s |
| Full training protocol | 355.148 s | 534.645 s |
| Full-test inference median | 4.766826 s | 7.552804 s |
| Test throughput | 2097.83 images/s | 1324.01 images/s |
| Fixed-batch forward median | 58.218 ms | 123.847 ms |
| Fixed-batch forward/backward median | 111.783 ms | 192.950 ms |
| Fixed-batch complete step median | 114.276 ms | 194.989 ms |

snnTorch was 1.752x faster for optimization and 1.584x faster for full-test
inference in this single paired execution. These are executor/extension results
on two L4 GPUs, with CPU encoding and shared-host contention. They do not
establish performance of snnlab's stock neuron, independent-seed accuracy
uncertainty, CPU-to-GPU speedup, or end-to-end cloud turnaround. The CSV stores
snnTorch time divided by snnlab time, so ratios below one favour snnTorch.

The three SVG/PDF figure pairs were rendered and visually inspected. No tests,
publication, commit or push were performed. Both completed phases from the
rejected v1 comparison (`exp126-r009-compute-phase-snntorch` and
`exp126-r010-compute-phase-snnlab`) remain immutable separate evidence; their
hardware and encoded inputs differ and they were not used in these results.

Task-owned failed/queued reservations r005–r008, r011–r015 remain hidden.
All corresponding Modal apps are stopped with zero tasks, and the completed
r020 identity preserves the allocation high-watermark. Cleanup was attempted
under the exclusive shared operation lock but deferred because an unrelated
exp124 local compute writer holds the lock. No visible runs were changed.

## Revised report — 14 October 2026

The article now presents the original baseline and completed paired L4
comparison, including all five plots, both benchmark tables, agreement
diagnostics, selected-model accuracy, timing boundaries and limitations.
It remains in Uncategorized (`demo`), tagged `data` and `v36.0.0`.

A new report presentation, `exp126-r021-present`, explicitly consumes
`exp126-r019-analyse` and original baseline presentation `exp126-r004-present`.
It regenerates the three comparison figures and reuses the baseline figures
unchanged, with baseline numbers included as a display projection. The article
pins this single presentation. It does not rerun compute or analysis or modify
any completed source run. Reproduce this presentation with:

```sh
uv run python experiments/exp126/present.py --source exp126-r019-analyse --baseline exp126-r004-present
```

Validation: read-only Pingstore discovery and presentation validation passed;
baseline numbers and SVG/PDF bytes matched their source; a disposable Astro
static build succeeded and included all five figure URLs and reported values.
No tests, publication, server startup, commit or push were performed.

## Performance diagnosis — 14 October 2026

A separate diagnostic Modal L4 run used the initial parameters and 128
validation images from the paired comparison, with the same 50-step encoded
events, four CPU threads and disabled TF32. Both implementations were measured
on the same GPU, with randomized condition/operation order, three warmups and
seven unprofiled timing repetitions. Operator counts and Python profiles were
collected separately. No model was trained; no production snnlab file was edited.

Compute: `exp126-r023-compute`; local analysis: `exp126-r024-analyse`;
presentation: `exp126-r025-present`. The article remains explicitly pinned to
`exp126-r021-present`; this diagnostic presentation is separate evidence.
Modal app: [completed profiler](https://modal.com/apps/me-93863/main/ap-BIvoz18Q9lbzTwMo6Iljuo).

| Condition (ablations cumulative after the original) | Forward median (ms) | Forward/loss/backward median (ms) |
| --- | ---: | ---: |
| snnTorch reference | 63.547 | 116.795 |
| Original snnlab | 126.189 | 196.641 |
| Remove finite-value checks | 76.007 | 131.116 |
| Also remove binary-spike checks | 68.770 | 119.428 |
| Also bypass zero-decay projection arithmetic | 61.865 | 103.676 |
| Also replace registered surrogate with cached snnTorch fast-sigmoid callable | 52.306 | 95.087 |

The dominant cause is value validation inside every simulation timestep.
`checked_state` checks each of three states for finiteness, then `checked_tensor`
checks finiteness again during its shape/device/dtype contract. Across two
populations and 50 steps, each forward pass contains 712 finite checks and
100 binary checks, producing 812 scalar extractions and CUDA stream
synchronizations. Backward adds 100 custom-surrogate derivative checks, for
912 such synchronizations. Removing finite checks leaves 100; removing both
value-check categories leaves zero internal scalar/stream synchronization calls.
The explicit synchronization at timing boundaries remains.

Finite checks alone removed 80.1% of the measured forward gap and 82.1% of the
forward/backward gap. With binary checks removed, these became 91.7% and 96.7%.
This establishes one dominant overhead category repeated many times, plus smaller
costs. Zero-decay projections still compute `previous * 0 + drive` and retain an
unnecessary autograd path; bypassing that arithmetic saved another 6.90 ms
forward and 15.75 ms forward/backward in the cumulative ablation.

The final surrogate intervention replaces per-call registry dispatch and a
dynamically defined autograd class with a cached snnTorch callable. It also
changes the floating-point form of the same mathematical derivative; therefore
it does not isolate Python caching cost alone. Every diagnostic variant produced
identical initial class scores and gradients within the predefined tolerances
on this batch. The ablations are not safe production settings or proof across
all batches/training trajectories. No full-training or full-test speedup was
measured for the modified paths, and cumulative effects need not add linearly.

The first optimization target should be restructuring runtime value checks to
avoid per-state, per-step CPU/GPU barriers while preserving error detection;
next specialize exactly instantaneous projections, then cache surrogate dispatch.
[NVIDIA's synchronization guide](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/sync-free-code.html)
explains why Python branches on GPU tensor values introduce these barriers.

Reproduce with `modal run experiments/exp126/modal_profile.py` using the isolated
Python 3.12 client documented above; then explicit local analyse/present commands
with the returned identities. All profiling apps have stopped. The failed r022
reservation remains hidden because another local writer holds the shared
operation lock. The earlier failed reservations remain for the same reason.
Diagnostics and figure were validated; targeted Ruff checks passed. No tests,
publication, commit or push were performed.

## Recheck and upstream issue — 14 October 2026

A separately launched L4 diagnostic repeated the experiment with 15 timing
repetitions, three warmups and cached AST transforms. Compute was
`exp126-r027-compute`; analysis was `exp126-r028-analyse`. Median forward times
were 63.066 ms for snnTorch, 125.896 ms for original snnlab, and 67.485 ms after
removing finite and binary checks. Forward/backward values were 114.467,
197.461 and 118.302 ms, respectively. The corresponding gap reductions were
93.0% and 95.4%. Original synchronization counts were again 812 forward and
912 forward/backward. Initial scores matched exactly and gradient maximum
absolute differences remained at most 9.536743e-7 for the non-cached variants.

Both profiled library files matched the snnlab 0.9.0 release commit
`4e545e9c5b368c8f138d35078562f9559523e443` byte for byte. The duplicate state
finite scans and per-step binary branches were checked directly in those files.
The surrogate ablation is explicitly a combined implementation change rather
than an isolated caching measurement.

[Upstream issue 13](https://github.com/eoinmurray/snnlab/issues/13) records both diagnostic measurements,
source permalinks, a minimal barrier reproduction, scope limits and acceptance
criteria. The posted body was read back and verified against the prepared text.
All recheck workers have stopped. The failed r022/r026 profiling reservations remain hidden because an unrelated active writer holds the shared operation lock.

## snnlab 0.10.1 speed-test rerun — 14 October 2026

The project dependency minimum and lockfile were upgraded to snnlab 0.10.1.
The matched model now relies on the library default and verifies it against
the recorded structural policy. The explicit strict executor remains a separate
initial fixed-batch timing reference, with score/gradient agreement checks.
The matched scientific protocol is now `exp126.matched/v3`; network equations,
partitions, initial parameters, encoding, Adam, 30 epochs and selection are
unchanged. Legacy ablation profiling is guarded against newer incompatible
validation APIs; retained v2 comparison analysis remains readable.

A two-L4 allocation was unavailable, so its queued request was stopped and two
independent L4 workers were submitted before waiting for either. Each used four
CPU threads and the same recorded software. Both phases started a second apart
and overlapped. Assembly verified matched GPU names, recipes, sources, initial
arrays and actual encoded input checksums. The full run was assembled on Modal,
then downloaded with both phase ancestors for local analysis and presentation.

1. snnTorch phase: `exp126-r033-compute-phase-snntorch`.
2. snnlab phase: `exp126-r034-compute-phase-snnlab`.
3. Remote full compute assembly: `exp126-r035-compute`.
4. Local analysis: `exp126-r036-analyse`.
5. Local presentation: `exp126-r037-present`, including the original baseline
   from the explicitly named `exp126-r004-present`.

Modal app: [completed upgraded comparison](https://modal.com/apps/me-93863/main/ap-gb8eMAURbf9Nzdci9RkMa2).
Phase dispatches: `ee6f9bc1fb3b445fbe317ec4778b9542` and
`dc61f18c94944b8db043200b81ae771d`; full dispatch:
`8aa546f63323458bb54875215389b8b5`.

| Measurement | snnTorch 0.9.4 | snnlab 0.10.1 default | Earlier snnlab 0.9.0 |
| --- | ---: | ---: | ---: |
| Optimization | 158.197 s | 205.165 s | 277.486 s |
| Full training protocol | 362.198 s | 418.014 s | 534.645 s |
| Full-test inference median | 4.793001 s | 5.461361 s | 7.552804 s |
| Forward, batch 128 | 59.129 ms | 62.582 ms | 123.847 ms |
| Forward/backward, batch 128 | 109.857 ms | 114.299 ms | 192.950 ms |
| First Adam step, batch 128 | 114.754 ms | 119.526 ms | 194.989 ms |
| Test accuracy | 88.04% | 88.74% | 88.74% |

The new default's fixed-batch kernels are close to snnTorch, but the full
optimization workload still took 29.7% longer and test inference 13.9% longer.
Before/after snnlab durations fell by 26.1% for optimization and 27.7% for test
inference. The earlier run shared one eight-CPU host; this rerun used independent
four-CPU hosts. These are single-pair observations with unmeasured clock/load
variation, not an isolated cross-version causal estimate or seed uncertainty.

Within snnlab 0.10.1 on its same worker, strict versus default forward medians
were 129.696 versus 62.582 ms, forward/backward 207.380 versus 114.299 ms, and
full-step 216.777 versus 119.526 ms. Each used three warmups and five timed
repetitions; default timing preceded the separate strict reference. Four
policy/operator profiles were collected separately, before the training timer.
Strict recorded 512/612 CUDA stream synchronizations for forward/forward-backward;
the default recorded 12/12, all from initialization rather than per-step value
checks. Strict/default scores and gradients agreed bitwise on the 128-image
initialization batch. snnTorch/snnlab state and spike probes matched exactly and
gradients passed the predeclared tolerance.

Both implementations reproduced their previous learning histories, selected and
final parameter arrays and complete test predictions exactly. Phase scientific
payloads and the assembled scientific payload therefore have the same SHA-256
digests as their corresponding earlier results, while new run identities and
metadata record the new software, policy and timings. No completed source run
was modified.

Presentation adds `validation_policy.svg/.pdf` and `validation_policy_table.csv`
to the three comparison figure pairs, summary CSV and numbers. These show five
timing repeats and medians, plus synchronization counts from individual profiled
passes. All retained outputs were validated; the timing and policy figures were
rendered and visually checked; targeted Ruff and diff checks passed. No pytest,
publication, server startup, commit or push was performed. The existing article
remains pinned to its earlier complete report presentation; the rerun's plots
and numbers are available in r037 for a later article update.

All task-owned Modal workers and canceled requests are stopped. Task-owned
empty failed/queued reservations r005–r008, r011–r015, r022, r026 and r029–r032
were removed under the exclusive shared operation lock after worker checks;
the completed r037 identity preserves the allocation high-watermark. All visible
runs, including the rejected v1 pair, remain unchanged. No exp126 temporary
directories remain.

## Remaining-cost diagnosis, 2026-10-14

Explicit input r035; Modal L4 with four CPUs; snnlab 0.10.1; one shared GPU
for randomized, interleaved library and ablation measurements. Completed compute
r038 and analysis r039 feed presentation r041. Presentation r040 is retained
but its figure labels used an order inconsistent with sorted JSON keys; its
numbers/table are valid, and r041 corrects the figure ordering. Use r041.

Both implementations used the same initial or snnTorch final parameter arrays,
128 fixed validation examples and shared encoded events. Three warmups and eleven
unprofiled repetitions per condition measured forward, forward/backward and a
warmed Adam step with clipping. Parameters were restored outside each timer;
Adam moments evolved separately per variant. Operator and Python profiles were
separate from timing repetitions. All variants had bitwise identical scores;
parameter gradients passed the established tolerance at both checkpoints.

| Trained-weight condition | Forward ms | Forward/backward ms | Warmed step ms |
| --- | ---: | ---: | ---: |
| snnTorch | 77.47 | 141.38 | 142.25 |
| snnlab default | 74.72 | 139.81 | 142.19 |
| Skip zero-decay projection recurrence | 67.66 | 120.70 | 121.14 |
| Omit unused continuation/parameter snapshots | 74.16 | 140.89 | 141.13 |
| Both diagnostic changes | 67.26 | 121.45 | 122.11 |

The earlier 29.7% optimization gap does not reproduce in this controlled probe;
it cannot be assigned wholly to library implementation. Different host load,
CPU scheduling or clocks remain hypotheses, not experimentally established
causes. This is not a full training or test-inference rerun.

A substantial remaining avoidable cost is zero-decay projection bookkeeping:
about 21 ms per trained warmed step (14.8%). Operator counts fall from 1201 to
805 multiplications and 501 to 301 additions, including forward and backward.
Unused snapshot work is much smaller (about 1 ms); differences at that scale
are not reliably additive. The default still has twelve initialization value
synchronizations beyond the common optimizer checks, not the old hundreds of
per-timestep synchronizations. No synchronization ablation was measured here.
Profiles show only about 15 ms of CUDA kernel execution versus about 142 ms
wall time: host dispatch/launch overhead is substantial, but profiled times
are not a causal decomposition of unprofiled wall time.

Diagnostic AST patches are scoped and restored; no snnlab source was edited.
Zero-decay skipping is checked only for finite inputs and can change nonfinite
propagation; snapshot removal discards continuation semantics. These are
optimization evidence, not production fixes. Stage payloads validated, Ruff
passed, and r041 was visually checked. Read-only discovery confirmed exp126
presentation availability; existing `data` and `v36.0.0` tags remain appropriate.
The article remains pinned to r021. Modal stopped successfully and no task-owned
temporary directories remain.
