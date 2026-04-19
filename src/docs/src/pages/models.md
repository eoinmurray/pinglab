---
layout: ../layouts/MarkdownLayout.astro
title: "Models"
---

# Models

The shared model ladder used across experiments. Each step adds one axis of biophysical realism or time-discretisation rigour. For the project-level "why," see [Introduction](/introduction/).

## The model ladder

The ladder: six models, ordered by increasing biophysical realism and time-discretisation rigour. All take MNIST Poisson spikes as input and output digit-class logits. The first two (snnTorch-library, snnTorch) share the same update rule — snnTorch-library is the reference, snnTorch is pinglab's in-repo version — and serve as a calibration pair.

| Model | Description |
| ----- | ----------- |
| snnTorch-library | Thin wrapper around snnTorch's *snn.Leaky* + fast-sigmoid surrogate. Parity reference for the pinglab path. |
| snnTorch | snnTorch-library form, pinglab implementation: $\beta$ a dimensionless hyperparameter. No dt semantics. Not dt-stable. |
| CUBA | Exact Euler discretisation of $\tau\,dV/dt = -V + I$. |
| CUBA-exp | CUBA + exponential synapse. |
| COBA | Full biophysical: exp synapse + hard reset + refractory + conductance-$V$. |
| PING | COBA + E→I→E loop producing gamma oscillation. Dt-stable below $\tau_{\text{GABA}}$ ceiling. |

### snnTorch-library

A thin wrapper around snnTorch's own *snn.Leaky* module with the library's fast-sigmoid surrogate ([Eshraghian et al. 2023](https://arxiv.org/abs/2109.12894)). The update rule is $U_{t+1} = \beta U_t + W s - S \theta$ with reset by subtraction; $\beta$ is a dimensionless hyperparameter and there are no $\Delta t$ semantics.

Its purpose is calibration, not experimentation. At matched config, snnTorch-library and pinglab's snnTorch should train to within a small tolerance — any residual gap localises to the LIF-step implementation or surrogate-gradient details, not to the architecture. See [notebook 003](/notebook/nb003/) for the side-by-side training comparison and Δt-stability sweep.

### snnTorch

The reference model — matches the snnTorch library's snn.Leaky update exactly. $\beta$ is a dimensionless hyperparameter in $[0, 1)$, treated on par with weights. **No $\Delta t$ exists in the model** — simulation runs for t in range(num_steps) and time is abstract, step-indexed. A spike from neuron $i$ in step $t$ adds exactly $W_i$ to the membrane; bias is added every step. Reset on spike: subtract threshold (preserves overshoot).

This is the deep-learning framing of an SNN: a recurrent network with binary activations and a leaky scalar state, parameterised by $\beta$. It has no $\Delta t$-invariance because there is no $\Delta t$ in its spec.

**Compared to snnTorch-library:** identical update rule, different implementation — any numeric gap between the two localises to the LIF step or surrogate-gradient code, not to the model spec.

### CUBA

Proper continuous-time discretisation of the physicist's LIF: $\tau\,dV/dt = -V + I(t)$, where $I(t) = \sum_k W \delta(t - t_k) + b$ models presynaptic spikes as Dirac deltas plus a constant bias current. Two key consequences:

- **Per-spike kick** $= (1-\beta)/\Delta t \cdot W \approx W/\tau$ — neither $\Delta t$ nor $\beta$ appears. dt-invariant in magnitude.
- **Per-ms bias contribution** $= b$ at any $\Delta t$ — dt-invariant.

Same parameters, learning rate, and training cost as snnTorch — only the forward rule differs. Empirically $\Delta t$-invariant **in expectation** across $\Delta t \in [0.05, 2.0]$ ms, but breaks under discrete-spike variance at extreme $\Delta t$ when trained at fine $\Delta t$.

**Compared to snnTorch:** same LIF neuron class, but the forward rule derives from proper Euler integration with explicit $\Delta t$ semantics — this isolates whether dt-sensitivity is a property of the model or of the discretisation.

### CUBA-exp

CUBA augmented with an **exponential synapse**: incoming presynaptic spikes deposit charge into a synaptic conductance $g$ that decays with time constant $\tau_{\text{AMPA}} \approx 5$ ms, rather than depositing directly into $V$. The membrane then sees a smoothly-varying $g$ instead of sharp per-step impulses:

$$
g(t + \Delta t) = e^{-\Delta t/\tau_{\text{AMPA}}} g(t) + W s(t), \quad U(t+\Delta t) = \beta U(t) + (1 - \beta) g(t) + (1-\beta) b
$$

The exp synapse resolves CUBA's residual variance issue: at large $\Delta t$, many input spikes pile up into a single step, but $g$ smooths them over $\tau_{\text{AMPA}}$ before they reach $V$. Per-step $V$-drive variance stays low at every $\Delta t$.

**Compared to CUBA:** same discretisation philosophy, plus an exponential synapse that low-passes presynaptic drive before it reaches the membrane — this isolates whether the synaptic low-pass alone carries dt-stability.

### COBA

Conductance-based LIF with exponential synapses — the simplest biophysical model. Incoming spikes drive a synaptic conductance $g$ that decays with time constant $\tau_{\text{AMPA}} \approx 5$ ms, not an instantaneous current jump. The membrane voltage follows a continuous-time ODE driven by $g \cdot (V - V_e)$, Euler-discretised with explicit $\Delta t$. Refractory period ($\tau_{\text{ref}} \approx 2$ ms) enforced by a step counter. Dale's law on: weights non-negative (half-normal init, clamped in forward). Lower learning rate ($10^{-4}$ vs $10^{-2}$ for CUBA) because BPTT through conductance gradients is steeper; cm_back_scale = 1000 dampens voltage gradients to keep training stable.

In our code, COBA is PINGNet with ei_strength = 0 — same dynamics, no inhibitory population. Rate-integrating by construction, so naturally dt-invariant.

**Compared to CUBA-exp:** adds hard reset, refractory period, and a conductance-based membrane driven by $g(V - V_e)$ — this isolates whether the remaining biophysical features contribute beyond what the exp synapse alone already delivers.

### PING

Full E-I network with frozen recurrent weights, producing gamma oscillations. Excitatory and inhibitory populations (E:I ratio 4:1) share COBA-style dynamics; recurrent $W^{EE}, W^{EI}, W^{IE}$ connect them into the standard PING loop. The recurrent matrices are **frozen at init** — only $W_{\text{in}}$ and $W_{\text{out}}$ train, matching the trainable surface area of the other three models so the ladder is apples-to-apples.

With default parameters and input rate 50 Hz, the network locks at $f_0 \approx 40$ Hz with $\text{CV} \approx 0.4$ and $\sim 80\%$ activity — the intended gamma regime, not saturation. PING has a hard $\Delta t$ ceiling at $\Delta t \geq 1.5$ ms: once $\Delta t$ approaches $\tau_{\text{GABA}}$, the E→I→E loop cannot complete within one step, inhibition lags, and the network saturates.

**Compared to COBA:** adds an inhibitory population (E:I = 4:1) and the frozen E→I→E recurrent loop that produces gamma oscillations — this isolates what the gamma rhythm contributes on top of a purely feedforward conductance-based model.

### The Ladder

Each step up adds one axis of realism or rigour:

- **snnTorch-library → snnTorch**: same update rule, different implementation. Isolates *"does the pinglab LIF step and surrogate-gradient code match the library's numerics?"*
- **snnTorch → CUBA**: same neuron class, cleaner discretisation. Isolates *"does proper Euler fix CUBA's dt-sensitivity?"*
- **CUBA → CUBA-exp**: same discretisation philosophy, add exponential synapse. Isolates *"is the synaptic low-pass what's carrying stability?"*
- **CUBA-exp → COBA**: add hard reset, refractory, conductance-based membrane. Isolates *"do the remaining biophysical features add anything?"*
- **COBA → PING**: add inhibitory population and E→I→E coupling. Isolates *"what does the gamma oscillation contribute?"*

## Discretisation: snnTorch vs CUBA

The snnTorch vs CUBA split isolates what part of "CUBA is $\Delta t$-sensitive" is a property of the **model** and what part is a property of the **discretisation**.

**snnTorch's canonical form.** snn.Leaky runs $U_{t+1} = \beta U_t + W X - S \theta$ with $\beta$ a dimensionless hyperparameter and no $\Delta t$ anywhere in the spec.

**pinglab's hybrid (snnTorch).** We layer continuous-time semantics by setting $\beta = e^{-\Delta t/\tau_{\text{snn}}}$ via patch_dt, but weights and biases stay unscaled. A bias $b$ is added once per step, so over fixed real time it fires $T_{\text{ms}}/\Delta t$ times — 10× more often at fine $\Delta t$. That residual $\Delta t$-dependence is the asymmetric bias-balloon failure.

**Proper Euler (CUBA).** Integrating $\tau\,dV/dt = -V + I(t)$ exactly over one step yields per-spike kick $(1-\beta)/\Delta t \cdot W \approx W/\tau$ and per-ms bias contribution $b$ — both $\Delta t$-invariant **in expectation**.

**What CUBA still doesn't fix.** Variance, not mean. At coarse $\Delta t$ many coincident input spikes land in one step, hard reset discards the overshoot, and accuracy collapses. COBA dodges this because its exponential synapse integrates input through $\tau_{\text{AMPA}} \approx 5$ ms before it reaches the membrane.
