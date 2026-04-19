---
layout: ../layouts/MarkdownLayout.astro
title: "Training"
---

# Training

All models are trained with the same procedure: cross-entropy loss, Adam optimiser, and surrogate-gradient backpropagation through the full unrolled spike train. See the [appendix](#appendix-backpropagation-through-time) for the BPTT recursion that underwrites this.

## Surrogate gradients

The spike function $S = \mathbf{1}[U \geq \theta]$ has zero gradient almost everywhere, so backward passes use a surrogate. Pinglab uses fast-sigmoid (matching [snntorch.surrogate.fast_sigmoid](https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html); motivation and alternatives discussed in [Tutorial 5 § The Surrogate Gradient Approach](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html)) with pseudo-derivative

$$
\frac{\partial \tilde S}{\partial U}\bigg|_{U} = \frac{k}{(1 + k\,|U - \theta|)^2}
$$

with slope $k = 1$ on both the biophysical path (mV-scale membrane) and the snnTorch-clone path (dimensionless membrane). The snnTorch-library path uses the library default $k = 25$ — the asymmetry is deliberate and is part of what the parity test exposes. This keeps the forward dynamics exact while giving BPTT a usable gradient through the otherwise-discontinuous spike function.

## Input encoding

Pixels become Poisson spike trains. For a pixel with normalised intensity $x \in [0, 1]$, input neuron $i$ fires a Bernoulli spike at each step with probability

$$
p_i(t) = x_i \cdot r_{\max} \cdot \Delta t / 1000, \qquad r_{\max} = 25 \text{ Hz}
$$

so the per-neuron firing rate is $x_i \cdot r_{\max}$ Hz, independent of $\Delta t$. The resulting $(T, B, N_{\text{in}})$ binary tensor is the model's only input; stimulus duration is $T_{\text{ms}} = 1000$ ms for image tasks. Because encoding is stochastic, the network sees a different spike realisation of the same image every epoch — a form of data augmentation intrinsic to the rate code. At evaluation time, a seeded torch.Generator is threaded through the encoder so train-time eval and standalone infer on the same weights produce identical spike trains.

## Loss and optimisation

The network output (spike-count readout, unified across all models) is treated as logits and passed to cross-entropy loss. The optimiser is Adam with an optional ReduceLROnPlateau scheduler, or the --adaptive-lr schedule which divides the learning rate by 10 after epoch 20. Gradients are clipped to unit norm before each step.

## Gradient dampening for COBA/PING

The biophysical models face a gradient scale mismatch: voltage updates involve multiplication by $\Delta t / C_m$, which can be very small, while the surrogate gradient through the spike function is $O(1)$. Naive backpropagation through the membrane ODE produces gradients that are either vanishingly small (through voltage) or discontinuously large (through spikes).

To stabilise training, the backward pass through each voltage update is scaled by $1/\alpha$ where $\alpha$ is a dampening factor (CM_BACK_SCALE, typically 80 for the unitless models and 1000 for COBA/PING). In the forward pass nothing changes — the gradient is simply attenuated:

$$
\frac{\partial \mathcal{L}}{\partial V_i^t} \leftarrow \frac{1}{\alpha} \frac{\partial \mathcal{L}}{\partial V_i^t}
$$

This keeps the biophysical forward dynamics exact while bringing gradient magnitudes into a range compatible with Adam's learning rate.

*Future direction: principled costate control.* Our constant scaling factor is a blunt instrument. Burghi et al. (2026, "costate") introduce a controller $K_\theta(t, x)$ directly into the adjoint equations; by choosing $K_\theta$ to make the costate dynamics *exponentially contracting*, they bound the gradient matrices uniformly while preserving the global optimum. Replacing our constant $\alpha$ with a state-dependent $K_\theta$ is a natural next step.

## Dale's law in training

When Dale's law is enforced, weight matrices are clamped to $W \geq 0$ in the forward pass. Gradients still flow through the unclamped parameters, so Adam can push a weight below zero — but it will be clipped back to zero on the next forward pass. The optimiser explores the full real line; the network only ever sees non-negative weights.

## Early stopping and checkpointing

The best model state (by test accuracy) is saved at each epoch. Training terminates early if accuracy has not improved for a configurable number of epochs (default patience: 15). The saved weights.pth is the *best-epoch* state, not the final-epoch state.

## Known hyperparameter pitfalls

- **smnist encoding is row-by-row.** encode_smnist presents each MNIST row as a 28-pixel input vector for 10 ms, so N_IN = 28 and T_ms = 280. --w-in-sparsity 0.9 leaves $\sim$3 inbound connections per hidden neuron — the network goes silent. Use sparsity $\leq 0.5$ for smnist.
- **Biophysical models need --cm-back-scale 1000.** Without it, voltage gradients blow up in epoch 1. Only affects lif_step (coba, ping), not snn_lif_step (cuba-family).
- **Refractory rounding uses round, not ceil.** int(round(ref_ms/dt)) with min 1 step. Ceiling caused a step-jump at $\Delta t = 2$ ms that biased the dt-sweep against fine resolutions.
- **Warning tracker fires under joint conditions only.** ⚠ dead / ⚠ saturated require paired activity extremes **and** no-progress windows.

## Input modes and tasks

The models are exercised against three classes of input and a small family of tasks. The driving distinction is between stimuli designed to probe network dynamics (synthetic drive, synthetic spike trains) and stimuli drawn from a real classification task (image datasets).

### Input modes

*Synthetic-conductance.* Direct conductance injection into layer-1 E neurons, used for baseline oscillation studies where we want to study PING dynamics in isolation from any encoding stage. Drive is generated Börgers-style as a step function with per-neuron heterogeneity $X_i$ plus an Ornstein–Uhlenbeck noise process:

$$
g_i^{\text{ext}}(t) = T_E(t)(1 + \sigma_e X_i) + \eta_i(t)
$$

where $T_E$ switches between async and PING-regime values during a stimulus window, $X_i \sim \mathcal{N}(0, 1)$ is per-neuron heterogeneity, and $\eta$ is a discrete Ornstein–Uhlenbeck process with $\tau_\eta = 3$ ms. Drive is calibrated at $\Delta t_{\text{cal}} = 0.1$ ms and rescaled at runtime so that the steady-state AMPA conductance is invariant across $\Delta t$.

*Synthetic-spikes.* Poisson spike trains over $N_{\text{in}}$ input neurons with a rate that steps between a baseline and a stimulus level during a fixed window. Used when we want a spiking input with no spatial structure.

*Dataset.* Real images encoded as per-pixel Poisson spike trains. Supported datasets are scikit-digits ($8 \times 8$, 10 classes), MNIST ($28 \times 28 = 784$), and sequential MNIST.

### Sequential MNIST

sMNIST presents each $28 \times 28$ image row-by-row: at each moment only the 28 pixels of the current row are presented, for $t_{\text{row}} = 10$ ms, giving a $T = 280$ ms sample with $N_{\text{in}} = 28$. This is the standard ML benchmark for temporal credit assignment in SNNs. Because the 28-neuron bottleneck is too narrow to classify from, sMNIST runs require $\geq 2$ hidden layers and recurrence on at least one of them.

### dt stability sweep

Each model is trained at a reference $\Delta t_{\text{train}}$ (typically 0.1 ms or 1.0 ms) and then evaluated on a range of inference timesteps $\Delta t_{\text{infer}} \in [0.05, 2.0]$ ms without retraining. Poisson input encoding is itself $\Delta t$-dependent, so naively re-encoding at each inference $\Delta t$ introduces noise unrelated to the network's internal dynamics. To control for this, a FrozenEncoder generates a reference spike train at the finest $\Delta t_{\text{ref}}$ of the sweep and OR-pool downsamples it by blocks of size $k = \Delta t_{\text{target}}/\Delta t_{\text{ref}}$:

$$
s^{\text{target}}_{b, i} = \max_{t \in [kb, k(b+1))} s^{\text{ref}}_{t, i}
$$

This guarantees that the spike pattern seen by the network at $\Delta t_{\text{target}}$ is **provably** the OR-pool of the pattern at $\Delta t_{\text{ref}}$ — any accuracy difference across $\Delta t$ reflects the network, not the encoder.

## Calibration tasks

### Shared across all models (at each $\Delta t$)

```
--dataset mnist  --dt {0.1|1.0}  --t-ms 200  --epochs 40
--adaptive-lr  --observe video  --input-rate 50
```

### Differences between models

| Flag | snntorch-clone | cuba | coba | ping |
| ---- | ------------------ | ---- | ---- | ---- |
| --model | snntorch-clone | cuba | ping | ping |
| --kaiming-init | ✓ | ✓ | — | — |
| --lr | 0.01 | 0.01 | 0.0001 | 0.0001 |
| --no-dales-law | ✓ | ✓ | — | — |
| --ei-strength | 0 | 0 | 0 | 0.5 |
| --cm-back-scale | — | — | 1000 | 1000 |
| --w-in | (kaiming) | (kaiming) | 0.3 | 1.2 |
| --w-in-sparsity | (kaiming) | (kaiming) | 0.95 | 0.95 |

### Fairness controls

- **E-I weights are frozen.** For PING, $W^{EE}, W^{EI}, W^{IE}$ have requires_grad = False. Only $W_{\text{in}}$ and $W_{\text{out}}$ train across all models — otherwise the $\Delta t$-stability comparison isn't apples-to-apples.
- **Uniform randomize_init.** All models share the same hidden-state initialisation so epoch-0 dynamics are comparable.
- **Shared linear decoder.** No spiking output layer; all models read out continuous logits from accumulated hidden spikes. Removes decoder style as a confound.

## Appendix: backpropagation through time

Backpropagation Through Time (BPTT) is how gradients are computed in any network whose output at time $t$ depends on state from earlier timesteps. The idea is to *unroll* the recurrent computation into a deep feedforward graph and then run standard backpropagation through it.

### The unrolled computation graph

Consider a recurrent system with hidden state $h^t$ that evolves according to:

$$
h^t = f(h^{t-1}, x^t; \theta), \quad y^t = g(h^t; \theta)
$$

where $x^t$ is the input, $y^t$ is the output, and $\theta$ are the parameters (shared across time). Iterating for $T$ steps produces a chain $h^0 \to h^1 \to \cdots \to h^T$. For gradient computation, we treat this chain as a deep feedforward network of depth $T$ where layer $t$'s weights are tied to layer $t-1$'s.

### Forward and backward passes

The gradient of the loss with respect to a parameter $\theta_k$ is a sum over all timesteps at which $\theta$ influenced the computation:

$$
\frac{\partial \mathcal{L}}{\partial \theta_k} = \sum_{t=1}^T \frac{\partial \mathcal{L}}{\partial h^t} \cdot \frac{\partial h^t}{\partial \theta_k}
$$

Each $\partial \mathcal{L}/\partial h^t$ is itself recursive:

$$
\frac{\partial \mathcal{L}}{\partial h^t} = \frac{\partial \mathcal{L}}{\partial h^{t+1}} \cdot \frac{\partial h^{t+1}}{\partial h^t} + \frac{\partial \mathcal{L}}{\partial y^t} \cdot \frac{\partial y^t}{\partial h^t}
$$

The backward pass marches from $t = T$ down to $t = 1$, accumulating gradient contributions at each step.

### Exploding and vanishing gradients

The BPTT recursion contains a product of Jacobians $\partial h^{t+1}/\partial h^t$ across every timestep:

$$
\frac{\partial h^T}{\partial h^1} = \prod_{t=1}^{T-1} \frac{\partial h^{t+1}}{\partial h^t}
$$

If the Jacobian norms are consistently greater than 1, this product explodes exponentially in $T$; if consistently less than 1, it vanishes. Standard mitigations include gradient clipping, architectural choices that stabilise the Jacobian, and in the SNN setting, surrogate gradients that smooth the otherwise-discontinuous spike function.

### BPTT in spiking neural networks

SNNs are a natural fit for BPTT: each timestep of the simulation is one step of the recursion, and the "hidden state" includes membrane potentials, synaptic conductances, and refractory counters. Simulating a 200 ms input at $\Delta t = 0.1$ ms gives $T = 2000$ unrolled steps — a very deep computation graph.

The spike function $s = \mathbb{1}[V \geq V_{\text{th}}]$ is not differentiable, so the recursion would normally fail at every spike event. Surrogate gradients replace $\partial s / \partial V$ with a smooth approximation during the backward pass only, leaving the forward dynamics exact.

Because SNNs have biophysical state variables with physical units ($V$ in mV, $g$ in μS), the Jacobians can be wildly scaled — voltage updates involve tiny factors like $\Delta t / C_m$, while surrogate gradients through spikes are $O(1)$. This is the origin of the gradient scale mismatch addressed by the dampening trick above.
