---
layout: ../../layouts/MarkdownLayout.astro
title: "Wednesday 2026-04-15 12:00 — Δt-stability paper draft: the exponential synapse is necessary and sufficient"
date: 2026-04-15
---

# The exponential synapse is necessary and sufficient for $\Delta t$-stability in leaky integrate-and-fire networks

*Wednesday 2026-04-15 12:00 CEST · archived paper draft*

This entry is the Δt-stability paper draft as of 2026-04-15, preserved as a single journal entry after the experiment-page layer was retired. The writeup has its own internal section numbering (§1–§7) rather than the usual journal H2 skeleton, because it is the paper, not a dated finding.

**Abstract.** Surrogate-gradient SNNs trained at one simulation step $\Delta t$ often fail when deployed at another. Prior work attributes this to a CUBA-vs-COBA dichotomy. We decompose that claim on a five-model ladder spanning the gap, plus two intermediate ablations, evaluated on MNIST across 14 inference $\Delta t$ values in $[0.05, 2.0]$ ms. The exponential synapse alone closes $65$ of the $69$ percentage points of the gap; hard reset, refractory periods, and conductance-based membrane updates contribute only baseline-accuracy gains. The practical implication: a $\Delta t$-independent low-pass synaptic filter is sufficient for $\Delta t$-invariance, with no other biophysical machinery required.

## 1. Introduction

Surrogate-gradient SNNs [neftci-2019] are simulated as discrete-time recurrent networks with fixed step $\Delta t$. Because $\Delta t$ is conventionally tied to encoding window or hardware rather than to the dynamics, a network trained at one $\Delta t$ can lose tens of percentage points when deployed at another.

The published framing is a model-class issue: current-based (CUBA) models are $\Delta t$-sensitive; conductance-based (COBA) models are robust. But CUBA and COBA differ along several axes — synaptic dynamics, reset policy, refractory periods, voltage-update form. We ask which axis carries the claim by building a feature-incremental ladder and measuring each step.

## 2. Model ladder

| Model | Update rule |
| ----- | ----------- |
| snntorch | $U_{t+1} = \beta U_t + W s + b$ |
| cuba | $U_{t+1} = \beta U_t + \tfrac{1-\beta}{\Delta t} W s + (1-\beta) b$ |
| cuba-exp | $g \leftarrow e^{-\Delta t/\tau_{\text{AMPA}}} g + W s$; $U_{t+1} = \beta U_t + (1-\beta) g + (1-\beta) b$ |
| coba | $g \leftarrow g(1 - \Delta t/\tau) + W s$; $C\,dV/dt = -g_L(V-V_L) - g(V-V_e)$ |
| ping | coba + frozen E→I→E recurrent weights |

All models share input encoding (Poisson 50 Hz, 200 ms), linear readout, and trainable surface (input + output weights only; recurrent weights in *ping* are frozen). A sixth model, *snntorch-library*, appears only in §4.5 as a parity reference against the *snntorch* row.

## 3. Methods

**Task.** MNIST classification. 1000 train / 500 test samples. Each model trained 40 epochs at $\Delta t \in \{0.1, 1.0\}$ ms (Adam, BPTT, fast-sigmoid surrogate). Best-accuracy checkpoint retained.

**Δt-sweep.** Each checkpoint evaluated at 14 inference $\Delta t$ values in $[0.05, 2.0]$ ms. Inputs are *frozen*: each image is encoded once at the finest sweep $\Delta t$, then OR-pooled to the target — eliminating Poisson resampling as a confound.

**Hyperparameters.** $N_{\text{in}} = 784$, $N_{\text{hid}} = 1024$, $N_{\text{out}} = 10$. CUBA-family: Kaiming uniform init, lr $10^{-2}$, subtract-threshold reset. Biophysical: half-normal init, lr $10^{-4}$, hard reset, 2 ms refractory. $\tau_m = 10$ ms, $\tau_{\text{AMPA}} = \tau_{\text{GABA}} = 5$ ms.

## 4. Results

### 4.1 Calibration

![Training curves for all 7 models at both training dts](/figures/journal/2026-04-15-1200-mnist-dt-stability-draft/training_curves.1000.40.png)

All five headline models reach $\geq 85\%$ best test accuracy within 40 epochs at both training $\Delta t$ values; no model shows divergent or stuck behaviour. Lowest is *cuba* at $\Delta t = 0.1$ ms (85.0 %).

![Calibration accuracy bar chart](/figures/journal/2026-04-15-1200-mnist-dt-stability-draft/calibration_accuracy.1000.40.png)

### 4.2 Dynamics

Hidden-population statistics sit in biological range at both training $\Delta t$: excitatory rates $5$–$40$ Hz, ISI CV in $[0.37, 1.97]$, active fraction $11$–$83$ %. Only *ping* exhibits a dominant population-rate spectral peak, at $f_0 \approx 25$–$37$ Hz — the expected gamma regime. No pathological signatures (saturation, silence, NaN loss) anywhere.

### 4.3 Δt-sweep

![Δt-sweep: 5 model curves × 2 training dts](/figures/journal/2026-04-15-1200-mnist-dt-stability-draft/dt_sweep_combined.1000.40.png)

Four behaviours stand out:

1. **snntorch** collapses asymmetrically. Fine-trained generalises to coarse $\Delta t$, but coarse-trained loses 70 pp at $\Delta t = 0.05$ ms. The unscaled bias $b/(1-\beta) \propto b\,\tau/\Delta t$ balloons at fine $\Delta t$.
2. **cuba** collapses at coarse $\Delta t$ when fine-trained: many presynaptic spikes pile into one step, the membrane overshoots, hard reset discards the excess.
3. **cuba-exp, coba** are flat across the full range. The exponential synapse smooths the input spike stream; per-step drive variance is bounded by $\mathrm{Var}[g]$, which is itself $\Delta t$-invariant.
4. **ping** has a hard ceiling at $\Delta t \approx 1.5$ ms ($\approx \tau_{\text{GABA}}$); above it the E→I→E loop cannot complete a cycle within a step and gamma dies.

The passing models share one thing: an exponential synapse.

### 4.4 Feature attribution

![Ablation attribution along the cuba → coba ladder](/figures/journal/2026-04-15-1200-mnist-dt-stability-draft/ablation_attribution.1000.40.png)

At the worst-case cell (train $\Delta t = 0.1$ ms, inference $\Delta t = 2.0$ ms), the 69 pp gap decomposes as:

| Step | $\Delta$ acc |
| ---- | -----------: |
| cuba → cuba-exp (exp synapse) | $+65$ pp |
| cuba-exp → coba (conductance V) | $+4$ pp |
| coba → ping (E–I gamma) | $0$ pp |

94 % of the gap closes from the exponential synapse alone.

### 4.5 Library parity

![Parity: snntorch vs library snn.Leaky](/figures/journal/2026-04-17-1100-snntorch-parity-and-calibration/parity_and_calibration.png)

To rule out a strawman, we ran a sixth model using snnTorch's library *snn.Leaky* and *fast_sigmoid* primitives directly. At train $\Delta t = 1$ ms the two are indistinguishable (85 % vs 86 %). Off-diagonal, the library version collapses *more* (19 % vs 44 % at $\Delta t_{\text{infer}} = 0.1$ ms) — consistent with the tighter surrogate slope (25 vs 1) overfitting to the train-$\Delta t$ spike statistics. Our numbers therefore *understate* the fragility a library-default user would see. Figure and numbers come from the [Friday 2026-04-17 11:00 parity entry](/journal/2026-04-17-1100-snntorch-parity-and-calibration/).

## 5. Discussion

### 5.1 Mechanism

The argument is about per-step *variance*, not mean. Under cuba's Euler discretisation with Poisson inputs at rate $R$,

$$\mathrm{Var}[D_{\text{step}}] \approx R\,\Delta t / \tau_m^2,$$

so the per-step drive standard deviation scales as $\sqrt{\Delta t}$ — at coarse $\Delta t$ it exceeds threshold in a single step and hard reset discards the overshoot. Under cuba-exp, the synaptic conductance is a low-passed $W s$ with steady-state variance

$$\mathrm{Var}[g_{\text{ss}}] = W^2 R \tau_{\text{AMPA}} / 2,$$

which is $\Delta t$-invariant: the per-step Bernoulli variance ($\propto \Delta t$) cancels the decay normalisation ($\propto \Delta t$). The exp synapse decouples the integration step seen by the membrane from the per-step spike-count variance.

### 5.2 Implications

The published "biophysical models are $\Delta t$-robust" claim is empirically correct but mechanistically imprecise: the robustness comes from a single feature — the exponential synapse — that is shared by most biophysical models but does not require any other biophysical machinery. For practitioners: adding an exponential synapse to a CUBA LIF layer is sufficient. The code change is two state variables per neuron and one extra exponential. The full biophysical stack adds $3$–$5$ pp of baseline accuracy on MNIST, but zero extra $\Delta t$-stability.

### 5.3 Limitations

Single-seed runs throughout: the qualitative shape of every $\Delta t$-sweep is consistent with theoretical prediction, but the small per-feature deltas ($\leq 5$ pp) need multi-seed replication to quantify precisely. MNIST with Poisson rate coding is an easy substrate; tasks with explicit temporal structure may stress the ladder differently. Width is fixed at 1024.

## 6. Future work

- **Multi-seed replication** of the full ladder (5 seeds × 7 models × 2 train-$\Delta t$ ≈ 3–4 H100-hours) for error bars on the sub-5-pp deltas.
- **Sensitivity to $\tau_{\text{AMPA}}$.** Sweep $\tau_{\text{AMPA}} \in \{2, 5, 10, 20\}$ ms; predict (a) stability preserved, (b) feasible inference range $\Delta t \lesssim 0.4\,\tau_{\text{AMPA}}$.
- **Library-form + exp synapse.** Add an exp synapse on top of the snnTorch-library update rule; strong form of our claim is that it too becomes $\Delta t$-stable.
- **Full-MNIST replication.** Disambiguates pure architectural fragility vs sample-efficiency artefact for cuba's coarse-$\Delta t$ collapse.
- **Harder tasks.** Sequential MNIST, N-MNIST, DVS-Gesture, Spiking Heidelberg Digits — settings where fine spike timing matters.
- **Depth and width scaling.** Does the requirement extend to every synaptic path in a multi-layer net? Does the stability ordering survive at $N_{\text{hid}} \in \{256, 1024, 4096\}$?
- **PING ceiling formalisation.** Predict the ceiling from $\tau_{\text{GABA}}$, refractory, and loop delays; confirm by varying those constants.

## 7. Conclusion

$\Delta t$-stability in surrogate-gradient SNNs is not a property of CUBA vs COBA — it is a property of whether the input stream reaches the membrane through a low-pass filter with a $\Delta t$-independent time constant. The exponential synapse is that filter: necessary (removing it from coba breaks stability) and sufficient (adding it to cuba restores it).
