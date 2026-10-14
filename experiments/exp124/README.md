# SHD

Collection: `datasets`. Implemented smoke protocol and training diagnostics, 14 October 2026.

This experiment trains COBA and PING hidden layers on Spiking Heidelberg Digits
using `snnlab.lang` graph declarations and `snnlab.sim.execution.GraphExecutor`.
An ordinary PyTorch AdamW loop trains the input, interlayer and readout matrices.

## Current network: two hidden layers

The network is defined directly in `recipe.py`, with no shared PING builder.
The only architecture is 700 inputs → E1 → E2 → 20 non-spiking readouts. Each hidden
layer contains 128 E and 32 I neurons. PING enables a separate fixed reciprocal
E/I loop in each layer; COBA sets both loops to zero. Only E1 spikes feed E2.
Input→E1, E1→E2 and E2→readout weights are learned. The interlayer AMPA
projection is nonnegative. The default `tuned` recipe starts with 90% zeros;
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
uv run python experiments/exp124/compute.py --device cpu --train-examples 1000 --validation-examples 200 --epochs 20
```

## Default learning recipe

The iterative evidence and rejected settings are recorded in
[Training diagnosis](diagnostics.md). The selected profile uses hidden learning
rate 0.0004, readout learning rate 0.01, voltage-gradient damping 100, and batches
of 20 containing one example of every class. Readout weights are centered across
source neurons at initialization. Mean leaky-integrator voltage is multiplied
by a fixed gain of 20 and normalized by each example's mean E2 firing rate,
using a 20 Hz reference and 1 Hz floor. The normalization is an experiment-local
operation declared through SNNLab's public extension API; the network stays in
`recipe.py` and uses the same simulator.

Counts and five-epoch smoke defaults are unchanged. `--variant baseline` retains
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
uv run python experiments/exp124/compute.py --device cpu --train-examples 1000 --validation-examples 200 --epochs 20
uv run python experiments/exp124/analyse.py --source <new-compute-run-id>
uv run python experiments/exp124/present.py --source <new-analysis-run-id>
```

Example counts must be positive multiples of 20 for balanced classes. This run
selects 60 examples per class with the same seed: first 50 for training, last
10 for validation. It starts from fresh weights; it does not resume the smoke.
Each run records the complete two-layer recipe; earlier completed evidence is
retained unchanged. Without count/epoch overrides, compute retains the original
200/100-example, five-epoch defaults, now with the selected learning recipe.

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

The selected default recipe passed tiny fitting: COBA reached 90% training
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
