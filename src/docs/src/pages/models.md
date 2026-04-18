---
layout: ../layouts/MarkdownLayout.astro
title: "Models"
---

# Models

The shared five-model ladder used across experiments. Each step adds one axis of biophysical realism or time-discretisation rigour. For the project-level "why," see [Motivation](/motivation/).

## The model ladder

Five models, ordered by increasing biophysical realism and time-discretisation rigour. All take MNIST Poisson spikes as input and output digit-class logits.

| Model | Update rule | Description | E-I |
| ----- | ----------- | ----------- | --- |
| snntorch | $U_{t+1} = \beta U_t + W s + b$ | snnTorch-library form: $\beta$ a dimensionless hyperparameter. No dt semantics. Not dt-stable. | no |
| cuba | $U_{t+1} = \beta U_t + \frac{1-\beta}{\Delta t} W s + (1-\beta) b$ | Exact Euler discretisation of $\tau\,dV/dt = -V + I$. Dt-stable in mean; breaks under discrete-spike variance. | no |
| cuba-exp | $g \leftarrow e^{-\Delta t/\tau_{\text{AMPA}}} g + W s$; $U_{t+1} = \beta U_t + (1-\beta) g + (1-\beta) b$ | cuba + exponential synapse. **The feature that delivers dt-stability.** | no |
| coba | $g \leftarrow g(1-\Delta t/\tau) + W s$; $C\,dV/dt = -g_L(V-V_L) - g(V-V_e)$ | Full biophysical: exp synapse + hard reset + refractory + conductance-$V$. | no |
| ping | coba with frozen recurrent $W^{EE}, W^{EI}, W^{IE}$ | coba + E→I→E loop producing gamma oscillation. Dt-stable below $\tau_{\text{GABA}}$ ceiling. | yes |

### snntorch

The reference model — matches the snnTorch library's snn.Leaky update exactly. $\beta$ is a dimensionless hyperparameter in $[0, 1)$, treated on par with weights. **No $\Delta t$ exists in the model** — simulation runs for t in range(num_steps) and time is abstract, step-indexed. A spike from neuron $i$ in step $t$ adds exactly $W_i$ to the membrane; bias is added every step. Reset on spike: subtract threshold (preserves overshoot).

This is the deep-learning framing of an SNN: a recurrent network with binary activations and a leaky scalar state, parameterised by $\beta$. It has no $\Delta t$-invariance because there is no $\Delta t$ in its spec.

### cuba

Proper continuous-time discretisation of the physicist's LIF: $\tau\,dV/dt = -V + I(t)$, where $I(t) = \sum_k W \delta(t - t_k) + b$ models presynaptic spikes as Dirac deltas plus a constant bias current. Two key consequences:

- **Per-spike kick** $= (1-\beta)/\Delta t \cdot W \approx W/\tau$ — neither $\Delta t$ nor $\beta$ appears. dt-invariant in magnitude.
- **Per-ms bias contribution** $= b$ at any $\Delta t$ — dt-invariant.

Same parameters, learning rate, and training cost as snntorch — only the forward rule differs. Empirically $\Delta t$-invariant **in expectation** across $\Delta t \in [0.05, 2.0]$ ms, but breaks under discrete-spike variance at extreme $\Delta t$ when trained at fine $\Delta t$.

### cuba-exp

cuba augmented with an **exponential synapse**: incoming presynaptic spikes deposit charge into a synaptic conductance $g$ that decays with time constant $\tau_{\text{AMPA}} \approx 5$ ms, rather than depositing directly into $V$. The membrane then sees a smoothly-varying $g$ instead of sharp per-step impulses:

$$
g(t + \Delta t) = e^{-\Delta t/\tau_{\text{AMPA}}} g(t) + W s(t), \quad U(t+\Delta t) = \beta U(t) + (1 - \beta) g(t) + (1-\beta) b
$$

The exp synapse resolves cuba's residual variance issue: at large $\Delta t$, many input spikes pile up into a single step, but $g$ smooths them over $\tau_{\text{AMPA}}$ before they reach $V$. Per-step $V$-drive variance stays low at every $\Delta t$.

**This is the feature that delivers $\Delta t$-stability.** Empirical ablation shows cuba-exp recovers 65 percentage points of the 69pp $\Delta t$-stability gap between cuba and coba. Every further biophysical feature contributes baseline accuracy but not $\Delta t$-stability.

### coba

Conductance-based LIF with exponential synapses — the simplest biophysical model. Incoming spikes drive a synaptic conductance $g$ that decays with time constant $\tau_{\text{AMPA}} \approx 5$ ms, not an instantaneous current jump. The membrane voltage follows a continuous-time ODE driven by $g \cdot (V - V_e)$, Euler-discretised with explicit $\Delta t$. Refractory period ($\tau_{\text{ref}} \approx 2$ ms) enforced by a step counter. Dale's law on: weights non-negative (half-normal init, clamped in forward). Lower learning rate ($10^{-4}$ vs $10^{-2}$ for CUBA) because BPTT through conductance gradients is steeper; cm_back_scale = 1000 dampens voltage gradients to keep training stable.

In our code, coba is PINGNet with ei_strength = 0 — same dynamics, no inhibitory population. Rate-integrating by construction, so naturally dt-invariant.

### ping

Full E-I network with frozen recurrent weights, producing gamma oscillations. Excitatory and inhibitory populations (E:I ratio 4:1) share coba-style dynamics; recurrent $W^{EE}, W^{EI}, W^{IE}$ connect them into the standard PING loop. The recurrent matrices are **frozen at init** — only $W_{\text{in}}$ and $W_{\text{out}}$ train, matching the trainable surface area of the other three models so the ladder is apples-to-apples.

With default parameters and input rate 50 Hz, the network locks at $f_0 \approx 40$ Hz with $\text{CV} \approx 0.4$ and $\sim 80\%$ activity — the intended gamma regime, not saturation. PING has a hard $\Delta t$ ceiling at $\Delta t \geq 1.5$ ms: once $\Delta t$ approaches $\tau_{\text{GABA}}$, the E→I→E loop cannot complete within one step, inhibition lags, and the network saturates.

### Why this ladder

Each step up adds one axis of realism or rigour:

- **snntorch → cuba**: same neuron class, cleaner discretisation. Isolates *"does proper Euler fix CUBA's dt-sensitivity?"*
- **cuba → cuba-exp**: same discretisation philosophy, add exponential synapse. Isolates *"is the synaptic low-pass what's carrying stability?"*
- **cuba-exp → coba**: add hard reset, refractory, conductance-based membrane. Isolates *"do the remaining biophysical features add anything?"*
- **coba → ping**: add inhibitory population and E→I→E coupling. Isolates *"what does the gamma oscillation contribute?"*

## Discretisation: snntorch vs cuba

The snntorch vs cuba split isolates what part of "CUBA is $\Delta t$-sensitive" is a property of the **model** and what part is a property of the **discretisation**.

**snnTorch's canonical form.** snn.Leaky runs $U_{t+1} = \beta U_t + W X - S \theta$ with $\beta$ a dimensionless hyperparameter and no $\Delta t$ anywhere in the spec.

**pinglab's hybrid (snntorch).** We layer continuous-time semantics by setting $\beta = e^{-\Delta t/\tau_{\text{snn}}}$ via patch_dt, but weights and biases stay unscaled. A bias $b$ is added once per step, so over fixed real time it fires $T_{\text{ms}}/\Delta t$ times — 10× more often at fine $\Delta t$. That residual $\Delta t$-dependence is the asymmetric bias-balloon failure.

**Proper Euler (cuba).** Integrating $\tau\,dV/dt = -V + I(t)$ exactly over one step yields per-spike kick $(1-\beta)/\Delta t \cdot W \approx W/\tau$ and per-ms bias contribution $b$ — both $\Delta t$-invariant **in expectation**.

**What cuba still doesn't fix.** Variance, not mean. At coarse $\Delta t$ many coincident input spikes land in one step, hard reset discards the overshoot, and accuracy collapses. coba dodges this because its exponential synapse integrates input through $\tau_{\text{AMPA}} \approx 5$ ms before it reaches the membrane.

## Architecture conventions

- **Depth via --n-hidden.** Single int = 1 layer (back-compat); multiple = stacked layers. E.g. --n-hidden 128 256 builds Input → H1 → H2 → Output.
- **Recurrence is inferred.** Presence of --w-rec MEAN STD enables hidden-to-hidden recurrence on all hidden layers by default. Restrict with --rec-layers 2. There is no --recurrent flag.
- **E-I is inferred.** Presence of --ei-strength > 0 enables E-I split on all hidden layers by default. Restrict with --ei-layers 2.
- **Linear decoder produces signed logits.** Output head is logit = spike_counts @ W_out + b with signed Kaiming $W_{\text{out}}$ — negative logits are expected.
