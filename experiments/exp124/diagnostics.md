# Historical two-layer SHD training diagnosis

Historical development notes: the author requested pruning of these superseded local runs on 14 October 2026. Their IDs below are non-operational history. The current report uses only the final training and matched-raster lineage listed in README.

Historical local scientific checks, 14 October 2026. Both models used two 128 E/32 I
hidden layers, with network declarations in `recipe.py`. These are reduced
training experiments, not automated tests or official-test benchmarks.

The current default is the GPU `calibrated` profile: COBA has two 128-neuron
layers without inhibitory populations; PING has two 128 E/32 I layers. Its
operating-point calibration and measured two-seed readiness checks are recorded
in [README](README.md#operating-point-correction-and-gpu-readiness).
The measurements below establish readiness only for the earlier `tuned`
profile. Completed outputs and the article's presentation pin retain their
original provenance.

## What the checks establish

1. The original settings barely fit even 20 examples. Nonzero gradients and
   changed weights were insufficient evidence that the training recipe worked.
2. Sparse interlayer initialization alone did not fix fitting. The combined
   profile uses 90% initial interlayer zeros, compensated surviving weights,
   voltage-gradient damping 100 instead of 1000, and score gain 20.
3. With hidden learning rate 0.0004 and readout learning rate 0.01, both models
   reached at least 90% training accuracy on 20 examples in 100 optimizer
   updates, across seeds 42 and 43. Their tiny validation results were erratic;
   this establishes capacity to fit, not generalization.
4. Balanced 20-example batches contain one example of every class. They avoid
   uneven class exposure in each update without oversampling any utterance.
   Centering each output's initial weights across its input neurons reduces
   initial class offsets. Neither intervention alone establishes strong COBA
   generalization on the reduced pilot.
5. Per-example rate normalization multiplies the scores by 20 Hz divided by
   mean E2 firing rate, with a 1 Hz denominator floor. This fixed operation has
   no learned parameters and remains differentiable. It preserves the argmax
   for fixed weights while changing the scale of the training loss. It improved
   PING's final pilot validation accuracy from 31% to 44%; COBA rose from 15%
   to 18%. These are single-seed comparisons on a reused tuning split.

The same initial learned weights and batch order are used between COBA and
PING in each run. Scientific configuration, graphs, dataset identities,
initial/final weights, gradients, predictions, firing rates and timings are
retained in each completed compute run. The current diagnostic baseline uses
an additional initialization RNG draw compared with the earlier smoke, so
ablation comparisons use the matched diagnostic runs rather than claiming
identical initial weights to the older smoke.

## Completed observations

Accuracies below are **final training / final validation**, not the best epoch.
Time includes the configured models, epoch-zero evaluation and complete split
assessment after every epoch; acquisition and setup are excluded. All rows use
batch size 20. A 20-example epoch is one optimizer update; a 200-example epoch
is ten. The official SHD test partition remains untouched.

| Compute evidence | Settings | Train/validation examples | Epochs | COBA | PING | Configured models: minutes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| exp124-r021-compute | baseline, seed 42 | 20/20 | 30 | 5.0% / 5.0% | 10.0% / 15.0% | 1.7 |
| exp124-r022-compute | sum, seed 42 | 20/20 | 30 | 10.0% / 0.0% | 45.0% / 0.0% | 1.6 |
| exp124-r024-compute | gain, seed 42 | 20/20 | 30 | 20.0% / 10.0% | 20.0% / 5.0% | 1.6 |
| exp124-r025-compute | diverse, seed 42 | 20/20 | 30 | 5.0% / 5.0% | 10.0% / 5.0% | 1.5 |
| exp124-r027-compute | combined, seed 42 | 20/20 | 100 | 85.0% / 5.0% | 35.0% / 10.0% | 4.7 |
| exp124-r028-compute | combined + faster readout, seed 42 | 20/20 | 100 | 95.0% / 15.0% | 90.0% / 30.0% | 4.4 |
| exp124-r029-compute | combined + faster readout, seed 43 | 20/20 | 100 | 100.0% / 5.0% | 95.0% / 0.0% | 4.8 |
| exp124-r030-compute | faster readout, shuffled, seed 42 | 200/100 | 10 | 7.0% / 6.0% | 31.0% / 23.0% | 4.2 |
| exp124-r038-compute | faster readout, balanced, seed 42 | 200/100 | 10 | 21.5% / 11.0% | 29.5% / 25.0% | 4.0 |
| exp124-r040-compute | faster readout, balanced, gain 5, seed 42 | 200/100 | 20 | 22.5% / 16.0% | 30.0% / 25.0% | 9.3 |
| exp124-r042-compute | faster readout, balanced, centered, seed 42 | 200/100 | 20 | 30.0% / 15.0% | 38.5% / 31.0% | 9.0 |
| exp124-r044-compute | faster readout, balanced, centered, normalized, seed 42 | 200/100 | 20 | 24.0% / 18.0% | 57.0% / 44.0% | 8.7 |

| exp124-r046-compute | temporal maximum, normalized, seed 42 | 200/100 | 20 | 29.0% / 8.0% | 36.5% / 13.0% | 8.9 |
| exp124-r049-compute | normalized mean, input strength 0.45, seed 42 | 200/100 | 20 | 29.0% / 15.0% | 59.0% / 31.0% | 9.5 |
| exp124-r051-compute | normalized mean, hidden LR 0.002, seed 42 | 200/100 | 20 | 25.0% / 20.0% | 68.5% / 25.0% | 10.1 |

| exp124-r053-compute | normalized mean, hidden LR 0.001, seed 42 | 200/100 | 20 | 24.0% / 21.0% | 61.5% / 25.0% | 9.5 |
| exp124-r055-compute | selected defaults, 100 updates, seed 43 | 20/20 | 100 | 65.0% / 0.0% | 100.0% / 15.0% | 5.1 |
| exp124-r057-compute | selected defaults, COBA only, 300 updates, seed 43 | 20/20 | 300 | 90.0% / 15.0% | — | 6.8 |

| exp124-r059-compute | selected defaults, confirmation seed 43 | 200/100 | 20 | 25.5% / 14.0% | 60.5% / 22.0% | 8.1 |

All pilot rows use the combined profile unless a named override is given.
Successful tiny fitting is a positive control; pilot validation is the next
readiness check. Selecting settings using this validation split makes its
accuracy optimistic for subsequent independent evaluation.

## Rejected or incomplete attempts

1. Removing voltage-gradient damping entirely produced a nonfinite PING input
   gradient. That failed execution is not a completed result.
2. Native `reduce_max` is unsupported by the current executor. The failed
   attempt did not train. Temporal maximum is instead implemented as a
   registered experiment-local operation through SNNLab's public extension API.
   The completed maximum-readout pilot ended at 8% COBA / 13% PING validation
   accuracy and was rejected.
3. Cancelled 1,000-example runs have no resumable training checkpoint and are
   excluded from completed comparisons.

The previously selected preset was `tuned`: hidden learning rate 0.0004, readout
learning rate 0.01, balanced batches, centered initialization, and per-example
rate-normalized mean voltage. Its numeric settings match the completed
`exp124-r044-compute` pilot; only the preset name differs. During exploration,
`exp124-r053-compute` used a provisional `tuned` name with hidden rate 0.001.
Its frozen configuration remains authoritative; reproduce that particular
trial with an explicit `--learning-rate 0.001` override.

The selected recipe took 300 updates to reach 90% COBA training accuracy on
20 examples (seed 43); PING reached 100% after 100 updates. Normalization
improved pilot performance but slowed this COBA fitting control. These are
final-step measurements, not best-epoch substitutions.

## Readiness and next run

The previously selected `tuned` profile passed tiny fitting (COBA 90% at 300 updates; PING 100%
at 100) and both reduced pilots showed held-out accuracy above 5% chance.
Pilot final validation was COBA 18% / PING 44% for seed 42 and COBA 14% / PING
22% for seed 43. All three learned matrices had finite nonzero gradients and
changed weights. COBA remains weak and seed variation is substantial. This
makes a larger learning check useful; it does not establish 30% in ten epochs.
No larger run has been submitted.

```sh
.venv/bin/python experiments/exp124/compute.py --variant tuned --device cpu --train-examples 1000 --validation-examples 200 --epochs 20
```

This creates a fresh run with the earlier selected settings and trains COBA then PING
sequentially. It has 50 optimizer updates per epoch, versus ten in the pilot.
Allow roughly 35–45 minutes combined on the local four-thread CPU, estimated
from measured two-layer timings; this is not a newly measured larger-run time.
Use explicit returned IDs for the separate analyse and present commands.

The article pins `exp124-r065-present`, with explicit analyses and validated
`exp124-r016-present` as the retained one-layer input. The original figures and
numbers were checked unchanged. New curves were visually inspected, v4 payload
and ancestry validation passed, discovery confirmed the `data` tag, and the
site build rendered the new metrics successfully. No automated tests were run.
The cancelled r020 temporary reservation was removed under exclusive operation
and writer locks after preserving the newer allocation. Failed r023 (unsupported
native maximum) and r026 (nonfinite PING gradient without damping) temporary
runs remain as failure evidence and are not operational inputs.
