# SHD

Collection: `datasets`. Implemented smoke protocol and training diagnostics, 14 October 2026.

This experiment trains COBA and PING hidden layers on Spiking Heidelberg Digits
using `snnlab.lang` graph declarations and `snnlab.sim.execution.GraphExecutor`.
An ordinary PyTorch loop trains the input, interlayer and readout matrices;
the new paper profile uses Adam and earlier profiles use AdamW.

## GPU-oriented paper profile

The default `--variant paper` has COBA layers of 128 conductance-based LIF
neurons each, with no inhibitory populations or recurrent projections. PING
has 128 E + 32 I neurons in each layer, with its fixed reciprocal E/I loops.
Both use 700 inputs, two learned feedforward projections and a signed,
20-class non-spiking readout. The network is declared directly in `recipe.py`.

| Setting | Prepared value |
| --- | --- |
| Simulation timestep | 1 ms |
| Training batch size | 256 |
| Training metrics | Online minibatch scores before each update; no epoch-zero training evaluation |
| Evaluation | Validation once per epoch; metric tensors transferred at the split boundary |
| Per-batch gradient diagnostics | Disabled |
| Membrane time constants | 20 ms for E and I |
| AMPA/GABA decay time constants | 10 ms |
| Readout membrane / synapse time constants | 20 / 10 ms |
| Refractory periods | Zero |
| Optimizer | Adam, learning rate 0.001 for every learned matrix, betas 0.9/0.999 |
| Surrogate | Paper fast-sigmoid derivative, beta 100 in voltage normalized by the 15 mV threshold gap |
| Voltage-gradient damping / gradient clipping | 1 / disabled |
| Class scores | Maximum readout voltage over time; no gain or rate normalization |
| Input corruption | 1.6 Hz insertion per channel, 0.0008 deletion probability per occupied bin |
| Firing-rate regularization | Omitted as requested |

Spike noise is applied during training and evaluation, as in the
[paper's Methods](https://arxiv.org/pdf/1910.07407v1#page=15). Training corruption
changes with each presentation. Evaluation reuses fixed seeded corruption across
epochs; both conditions receive matched sample ordering, initialization and noise.
The original source spikes remain unchanged in the scientific export.

`--full-dataset` partitions all 8,156 official training utterances, with a seeded
10% holdout within each class: 7,340 train and 816 validation examples. Classes
retain their natural unequal counts; shuffled minibatches consume every training
example once per epoch, including the final incomplete batch. The official test
set is untouched. `--validation-fraction` explicitly changes the holdout.
Without `--full-dataset`, the small 200/100-example, five-epoch defaults remain.

This is an adaptation, not a reproduction: conductance dynamics, nonnegative
hidden weights, conductance initialization and fixed PING loops are retained.
The paper's current-based voltage and signed hidden weights are not substituted
into these networks. Batch size 256 and 1 ms steps now match the published GPU
timing benchmarks; they do not reproduce the original 2019 convergence curve,
which used 0.5 ms steps. A one-second presentation and stratified 10% validation
holdout are explicit implementation choices. The readout initialization is uncentered,
and paper mode does not enable the earlier pilot's score normalization.

The following command was dispatched as `exp124-r066-compute` on eight Modal
CPU cores with 16 GiB RAM, batch size 16 and 0.5 ms steps, with full training
split reevaluation each epoch; COBA then PING sequentially. At the author’s request,
the app was stopped during COBA’s first training epoch. Epoch-zero evaluation
took 867.8 seconds: training loss 4.3847 / accuracy 5.0%, validation loss 4.3835 /
accuracy 5.0%. No training epoch completed; PING never started. There is no
completed compute result or resumable checkpoint, and no curves were added to
the report. The monitor was deleted. The stopped app is
[ap-3ehv6aPQNOkEc6N47exgS8](https://modal.com/apps/me-93863/main/ap-3ehv6aPQNOkEc6N47exgS8).

```sh
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run --detach experiments/helpers/modal_compute.py --experiment exp124 --gpu none --cpu 8 --variant paper --full-dataset --validation-fraction 0.1 --epochs 20
```

For a future completed run, explicitly run analyse and present against its
actual returned IDs. The cancelled run must not be used as a source:

```sh
.venv/bin/python experiments/exp124/analyse.py --source <completed-compute-id>
.venv/bin/python experiments/exp124/present.py --source <returned-analysis-id> --full-training --reference-present exp124-r065-present
```

These stages accept uneven full-data class counts. The full-training presentation
preserves all previous figure bytes and numbers from the explicit reference,
adds `full_coba_training_curves` and `full_ping_training_curves`, and stores the
new measurements under `full_training`. The report pin changes only after the
new presentation and preservation checks pass.
No completed evidence, report figures or presentation pins have been changed.
Graph compilation, simulator construction and cached-label split inspection
were checked before training. After the SNNLab upgrade, a 32-step, two-example
synthetic CPU forward/backward check produced identical scores, population counts
and gradients for all three learned matrices between strict/full-snapshot and
fast/output-only execution, for both models. All gradients were finite. That check used no optimizer steps. The full-data run was subsequently
submitted and cancelled; trained SHD accuracy, full-presentation training
stability and complete training time were not established. The empty cancelled reservation was removed under the exclusive operation lock
after the newer GPU benchmark allocations preserved the allocation high-watermark. There is no
active worker or pending result transfer. Future COBA and PING runs should use
separate concurrent Modal CPU workers, as requested by the author.

## Concurrent GPU throughput checks

Two independent H100 workers completed concurrent bounded checks:
`exp124-r068-compute` is COBA and `exp124-r067-compute` is PING. Each trains
520 examples and validates 100 for one epoch with a 900-second worker cap.
This supplies two full 256-example optimizer updates and one eight-example
remainder. Per-update CUDA-event durations include input preparation, forward,
backward and optimizer work; host metrics are read only at epoch boundaries.
Both ran on NVIDIA H100 80GB HBM3, SNNLab 0.12.1. COBA completed in 11.33 s
and PING in 24.66 s, excluding dataset acquisition, graph construction, startup
and result transfer. Their analyses are `exp124-r069-analyse` and
`exp124-r070-analyse`. Every learned matrix changed and final weights were finite.
Per-batch gradient diagnostics were intentionally not measured.

| Model | First full update (s) | Second full update (s) | Peak GPU memory (GiB) |
| --- | ---: | ---: | ---: |
| COBA | 3.863 | 2.851 | 3.76 |
| PING | 8.451 | 5.936 | 4.51 |

Using the second full update and validation timings gives provisional 20-epoch
full-data estimates of 30–40 minutes for COBA and 60–80 minutes for PING,
excluding startup and transfer. Concurrent elapsed time is roughly the slower
condition's duration. Only one warmed full update was measured per model, so
these are planning estimates, not measured full-data training times. The
small subsets reached 5.0% COBA / 4.0% PING validation accuracy after three
updates; this says little about eventual recognition performance.

These are throughput checks, not full-data accuracy measurements. No replacement
20-epoch full-data run has been submitted.

Training curves now use online pre-update minibatch loss and accuracy; they are
not a final-weight reevaluation of the training set. Epoch-zero training values
are absent. Validation still uses fixed-noise held-out utterances at epoch zero
and after each epoch. Gradient norms are not measured in this profile; their
analysis flags are null rather than claimed to have passed. Finite loss is
checked at the epoch boundary, and final learned weights are checked before
completion. The same lack of resumable checkpoints remains.

## SNNLab 0.12.1 execution settings

Pinglab requires SNNLab >=0.12.1; the lockfile and local environment use 0.12.1.
Modal builds from that updated lockfile. Every exp124 training and evaluation
forward explicitly disables parameter, final-state and runtime-state snapshots,
while retaining the scores and spike-count outputs. Final learned weights still
come directly from the model's parameter map, so scientific weight exports are
unchanged. Each presentation starts with fresh neuronal state.

The recipe records `projection_mode="fast"`, `extension_validation="structural"`,
`compile_mode=None` and the three snapshot-retention flags. Fast arithmetic is
explicit because it is part of numerical provenance. The lowered readout
projection has zero numerical decay and uses the release's direct-drive
shortcut; the hidden AMPA/GABA projections retain their nonzero recurrences.
The snapshot controls also remove copies, but no SHD runtime improvement has
yet been measured.

The paper's custom surrogate keeps neuron kernels eager under the new compiler
API; enabling compilation would not compile those kernels. Built-in dataset
prefetch does not cover this experiment's custom SHD event batching. Neither
feature is enabled, and the agreed scientific training recipe is unchanged.
See the [release notes](https://github.com/eoinmurray/snnlab/blob/v0.12.1/CHANGELOG.md)
for the controls and their compatibility limits.

## Earlier two-layer recipes

The network is defined directly in `recipe.py`, with no shared PING builder.
These earlier recipes used 700 inputs → E1 → E2 → 20 non-spiking readouts. Each hidden
layer contains 128 E and 32 I neurons. PING enables a separate fixed reciprocal
E/I loop in each layer; COBA sets both loops to zero. Only E1 spikes feed E2.
Input→E1, E1→E2 and E2→readout weights are learned. The interlayer AMPA
projection is nonnegative. The earlier selected `tuned` recipe starts with 90% zeros;
surviving lower-clamped Gaussian weights have parent mean/SD 1.8/0.18 μS divided
by the source fan-in (128) and retained fraction (0.1). Zeros are not a permanent
mask. The original dense initialization remains available as the `baseline`
diagnostic variant, with the same two-layer architecture.

No architecture-selection argument is provided. Analysis reads recorded learned
parameter identities, including earlier evidence, and checks each matrix separately.
Two-layer histories include E1/I1/E2/I2 rates; the original E/I rate fields
refer to the final hidden layer. Earlier article figures retain their measured
one-layer provenance.

The first two-layer 200/100-example, five-epoch local smoke completed as
`exp124-r017-compute` → `exp124-r018-analyse`. Both conditions had finite
gradients and nonzero gradients/weight changes in all three learned matrices,
including E1→E2. Both ended at 5% validation accuracy; this verifies execution
and gradient flow, not improved recognition. Both excitatory layers fired;
both inhibitory populations fired in PING and remained silent in COBA.

```sh
uv run python experiments/exp124/compute.py --variant tuned --device cpu --train-examples 1000 --validation-examples 200 --epochs 20
```

## Previously selected learning recipe (`--variant tuned`)

The iterative evidence and rejected settings are recorded in
[Training diagnosis](diagnostics.md). The selected profile uses hidden learning
rate 0.0004, readout learning rate 0.01, voltage-gradient damping 100, and batches
of 20 containing one example of every class. Readout weights are centered across
source neurons at initialization. Mean leaky-integrator voltage is multiplied
by a fixed gain of 20 and normalized by each example's mean E2 firing rate,
using a 20 Hz reference and 1 Hz floor. The normalization is an experiment-local
operation declared through SNNLab's public extension API; the network stays in
`recipe.py` and uses the same simulator.

Those recipes used the 200/100-example, five-epoch smoke defaults. `--variant baseline` retains
the earlier training settings for diagnosis. `--batch-size`, `--learning-rate`,
`--readout-learning-rate`, `--readout-gain`, `--damping`, `--input-strength` and
`--interlayer-strength` provide explicit overrides. Boolean options also accept
`--no-balanced-batches`, `--no-center-readout` and `--no-normalize-readout`.
Balanced batches require a batch size divisible by 20.
`--models coba` or `--models ping` restricts a focused diagnostic. Without that
option, both models run sequentially, COBA first. A single-model analysis does
not claim to have checked paired initialization.

## Original two-layer smoke protocol

1. Authenticate the official training archive against the publisher's MD5 list
   and record SHA-256 identities. With seed 42, select 15 utterances per class:
   10 for training (200 total) and five for validation (100 total). No official
   test data is downloaded or evaluated.
2. Use 700 inputs and two layers of 128 E/32 I conductance-based LIF neurons, one-second
   presentations and a 1 ms simulation step. Bin events with binary OR and
   record collisions and events discarded beyond the presentation window.
3. Train input, interlayer and signed non-spiking readout weights for five epochs in batches
   of 16. Use AdamW at 0.0004, zero weight decay, clipping at 1 and voltage-gradient
   damping at 1000 in both conditions. Hidden conductance weights are nonnegative.
   Fixed E→I/I→E weights in each layer are zero for COBA and fan-in-normalized summed strengths
   of 1/2 microSiemens for PING. Initial input weights have 95% zeros, without a
   permanent mask; their parent mean/SD are 0.9/0.09 before normalization.
4. Match learned-weight initialization and shuffled sample order between models.
   Reset neuronal state for every presentation. Evaluate the complete training
   and validation splits at epoch zero and after each epoch. Record loss,
   accuracy, E/I rates, gradient norms and weight changes. Retain final-epoch
   weights, not a test-selected checkpoint.

E/I refractory durations are 2/1 ms, aligned to the coarse grid. The signed
readout is a non-spiking leaky integrator with a 20 ms time constant, scored by
mean voltage over the presentation. This is a training smoke, not a
fine-resolution validation of gamma dynamics or an official-test benchmark.
Historical SHD runs are not execution inputs.

## Independent stages

```sh
uv run python experiments/exp124/compute.py --device cpu
uv run python experiments/exp124/analyse.py --source <compute-run-id>
uv run python experiments/exp124/present.py --source <analysis-run-id>
```

For the larger local learning check:

```sh
uv run python experiments/exp124/compute.py --variant tuned --device cpu --train-examples 1000 --validation-examples 200 --epochs 20
uv run python experiments/exp124/analyse.py --source <new-compute-run-id>
uv run python experiments/exp124/present.py --source <new-analysis-run-id>
```

Example counts must be positive multiples of 20 for balanced classes. This run
selects 60 examples per class with the same seed: first 50 for training, last
10 for validation. It starts from fresh weights; it does not resume the smoke.
Each run records the complete two-layer recipe; earlier completed evidence is
retained unchanged. Without count/epoch overrides, compute retains the original
200/100-example, five-epoch defaults, with `--variant tuned` for the previously selected learning recipe.

The earlier one-layer 1,000/200-example, twenty-epoch local run completed as
`exp124-r014-compute` → `exp124-r015-analyse` → `exp124-r016-present`.
COBA took 736.2 seconds and finished at 12.8% training / 12.5% validation
accuracy; PING took 676.0 seconds and finished at 10.1% / 11.0%. Both reduced
loss, changed hidden/readout weights and passed finite-gradient checks. These
results fell short of the hoped-for 20% validation accuracy. The report pins
the new presentation and retains the original smoke figures through its
explicit reference input.

Compute defaults to CPU with four threads and caches the authenticated source
archive under `/tmp/shd` (`--cache` overrides it). `--cpu-threads` overrides the
default thread count; Modal's `--cpu` allocation sets it through the environment.
It exports sparse selected
input events, indices, labels and speakers, plus per-condition graphs,
initial/final weights and learning histories. Timings and device metadata live
in `run.json`. Exported graphs contain initializer placeholders: replay must
explicitly bind the retained initial or final matrices.

Analysis validates partitions, matched initialization and complete epochs, then
measures learning and weight changes. Presentation renders flat figures and
report numbers from explicit analysis inputs. Authored prose lives in
`report/report.mdx`.

For the same smoke on Modal A10G:

```sh
uv run --with 'modal>=1.1,<2' modal run experiments/helpers/modal_compute.py --experiment exp124 --smoke
```

This starts paid compute and returns validated evidence locally. Analyse that
returned compute run explicitly, then pass local and Modal analysis IDs with
repeated `--source` arguments to presentation. The comparison rejects different
recipes or selected examples; CPU/GPU numerical identity is not assumed.
No stage launches another stage or publishes.

For the requested two-layer 1,000/200-example, twenty-epoch Modal CPU run:

```sh
uv run --with 'modal>=1.1,<2' modal run --detach experiments/helpers/modal_compute.py --experiment exp124 --gpu none --cpu 8 --train-examples 1000 --validation-examples 200 --epochs 20
```

The Modal attempt `exp124-r019-compute` used eight CPU cores/threads and
16 GiB RAM. At the author's request it was stopped after two completed COBA
epochs, without a completed compute result or resumable checkpoint. Training
was restarted locally as `exp124-r020-compute`, using four CPU threads and the
same two-layer 1,000/200-example, twenty-epoch recipe for both models.
The cancelled Modal reservation was removed after the worker stopped. The local
run was also cancelled at the author's request before completing both models;
it is not a completed scientific result and cannot be resumed.

For CPU-only Modal execution, add `--gpu none --cpu 4` or `--gpu none --cpu 8`.
These settings change execution resources, not the scientific smoke recipe.

## Earlier one-layer CPU benchmark — 14 October 2026

All rows use the same seed, selected examples, scientific configuration and
five-epoch protocol. Condition time includes epoch-zero evaluation and full
split evaluation after each epoch; setup, download and transfer are excluded.

| Execution | Compute run | COBA (s) | PING (s) |
| --- | --- | ---: | ---: |
| Local CPU, 4 threads | exp124-r006-compute | 37.9 | 37.8 |
| Modal CPU, 4 cores/threads | exp124-r011-compute | 326.4 | 317.1 |
| Modal CPU, 8 cores/threads | exp124-r012-compute | 234.4 | 230.0 |
| Modal A10G | exp124-r005-compute | 444.0 | 443.0 |

Eight Modal cores reduced combined condition time by about 28% relative to
four, but local execution was still about six times faster. These are single
runs, not repeated hardware benchmarks. Both Modal CPU runs passed v4 payload
validation and finite-gradient/hidden-weight-change checks. Final validation
accuracy remained 9% for COBA and 8% for PING. Intermediate predictions differed
slightly; even seeded initial weights differed from local by at most about
1.4e-7 across CPU architectures. Numerical trajectories are not bitwise identical.
Modal exposed x86_64 architecture but no CPU model name.

Follow the [Experiment Rules](../../md/notes/experiment-rules.md) and
[Pingstore Rules](../../md/notes/pingstore-rules.md).
Dataset: [Spiking Heidelberg Digits](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/).

## Completed two-layer readiness checks

The previously selected `tuned` recipe passed tiny fitting: COBA reached 90% training
accuracy after 300 updates and PING 100% after 100. In the 200/100-example,
twenty-epoch pilots, seed 42 ended at 18% COBA / 44% PING validation accuracy;
seed 43 ended at 14% / 22%. Both pilots had finite, nonzero gradients and changed
weights in all three learned matrices. This supports another larger learning
check, with weak COBA performance and substantial seed variation still unresolved.
The 30%-in-ten-epochs target is unverified.

The report pins `exp124-r065-present`, which retains the validated one-layer
`exp124-r016-present` as an explicit input and adds the two-layer diagnostic
curves. Both 200/100 pilots took 8–9 minutes combined on local CPU. The proposed
1,000/200-example, twenty-epoch run has not been submitted; allow approximately
35–45 minutes locally for both sequential models, as an estimate.

To recreate the diagnostic presentation, pass the explicit analyse IDs listed
in its `run.json` with repeated `--source`, plus `--diagnostics`,
`--selected-source exp124-r047-analyse`, and
`--reference-present exp124-r016-present`. The selected-source highlights the
chosen recipe; matching configurations from other seeds remain separate curves.
See `diagnostics.md` for all completed trials and retained failure reservations.
