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
| COBA | Full biophysical: exp synapse + hard reset + refractory + conductance-$V$. Not a separate CLI model — run as *--model ping --ei-strength 0*. |
| PING | COBA + E→I→E loop producing gamma oscillation. Dt-stable below $\tau_{\text{GABA}}$ ceiling. |

### snnTorch-library

A thin wrapper around snnTorch's own *snn.Leaky* module with the library's fast-sigmoid surrogate ([Eshraghian et al. 2023](https://arxiv.org/abs/2109.12894)). Per-step update for one hidden layer:

$$
U_{t+1} = \beta\, U_t + W\, s_t + b \;-\; \theta\, S_t, \qquad S_t = \mathbf{1}[U_t \geq \theta]
$$

with reset-by-subtraction on spike. The canonical snnTorch spec treats $\beta \in [0, 1)$ as a dimensionless hyperparameter and has no $\Delta t$ in the update. Pinglab's usage injects continuous-time semantics by plugging $\beta = e^{-\Delta t/\tau_{\text{snn}}}$ with $\tau_{\text{snn}} = 10$ ms every time $\Delta t$ changes, so both paths see the same $\tau_{\text{mem}}$ when compared.

Its purpose is calibration, not experimentation. At matched config, snnTorch-library and pinglab's snnTorch should train to within a small tolerance — any residual gap localises to the LIF-step implementation or surrogate-gradient details, not to the architecture. See [notebook 003](/notebook/nb003/) for the side-by-side training comparison and Δt-stability sweep.

### snnTorch

Pinglab's in-repo reimplementation of the same forward rule:

$$
U_{t+1} = \beta\, U_t + W\, s_t + b, \qquad S_t = \sigma_{\text{surr}}(U_t - \theta)
$$

$$
U_{t+1} \leftarrow \begin{cases} 0 & \text{if reset}=\text{zero and } S_t = 1 \\ U_{t+1} - \theta\, S_t & \text{if reset}=\text{subtract} \end{cases}
$$

In its canonical spec $\beta$ is dimensionless and there is no $\Delta t$ — simulation runs for $t = 0, \dots, T_{\text{steps}}-1$ and time is step-indexed. A spike from neuron $i$ in step $t$ adds exactly $W_i$ to the membrane; bias is added every step. Threshold $\theta = 1$.

This is the deep-learning framing of an SNN: a recurrent network with binary activations and a leaky scalar state, parameterised by $\beta$. It has no $\Delta t$-invariance because there is no $\Delta t$ in its spec — the dt-dependence comes from pinglab's decision to plug in $\beta = e^{-\Delta t/\tau_{\text{snn}}}$ while leaving $W$ and $b$ unscaled (see [Discretisation](#discretisation-snntorch-vs-cuba) below).

**Compared to snnTorch-library:** identical update rule, different implementation — any numeric gap between the two localises to the LIF step or surrogate-gradient code, not to the model spec.

### CUBA

Proper continuous-time discretisation of the physicist's LIF:

$$
\tau\, \frac{dV}{dt} = -V + I(t), \qquad I(t) = \sum_k W\, \delta(t - t_k) + b
$$

with presynaptic spikes as Dirac deltas plus a constant bias current. Integrating exactly over one step of length $\Delta t$ with $\beta = e^{-\Delta t / \tau}$ gives

$$
U_{t+1} = \beta\, U_t + \frac{1 - \beta}{\Delta t}\, W\, s_t + (1 - \beta)\, b
$$

followed by the same spike / reset rule as snnTorch. Two key consequences:

- **Per-spike kick** $= (1-\beta)/\Delta t \cdot W \approx W/\tau$ as $\Delta t \to 0$ — neither $\Delta t$ nor $\beta$ appears in the limit. Dt-invariant in magnitude.
- **Per-ms bias contribution** $= (1-\beta)/\Delta t \cdot b \approx b/\tau$ per ms — dt-invariant, whereas the snnTorch path injects $b$ once per step and grows bias drive by $1/\Delta t$.

Same parameters, learning rate, and training cost as snnTorch — only the forward rule differs. Empirically $\Delta t$-invariant **in expectation** across $\Delta t \in [0.05, 2.0]$ ms, but breaks under discrete-spike variance at extreme $\Delta t$ when trained at fine $\Delta t$.

**Compared to snnTorch:** same LIF neuron class, but the forward rule derives from proper Euler integration with explicit $\Delta t$ semantics — this isolates whether dt-sensitivity is a property of the model or of the discretisation.

### CUBA-exp

CUBA augmented with an **exponential synapse**: incoming presynaptic spikes deposit charge into a synaptic conductance $g$ that decays with time constant $\tau_{\text{AMPA}} = 2$ ms, rather than depositing directly into $V$. The membrane then sees a smoothly-varying $g$ instead of sharp per-step impulses. Per-step update, with $\beta = e^{-\Delta t/\tau_{\text{mem}}}$ and $\alpha = e^{-\Delta t/\tau_{\text{AMPA}}}$:

$$
g_{t+1} = \alpha \bigl(g_t + W\, s_t\bigr)
$$

$$
U_{t+1} = \beta\, U_t + (1 - \beta)\, g_{t+1} + (1 - \beta)\, b
$$

followed by the same spike / reset rule. The synapse is updated "kick then decay": spikes add to $g$, then the whole conductance decays one step. The bias path is unchanged from CUBA (per-ms contribution $\approx b/\tau$).

The exp synapse resolves CUBA's residual variance issue: at large $\Delta t$, many input spikes pile up into a single step, but $g$ smooths them over $\tau_{\text{AMPA}}$ before they reach $V$. Per-step $V$-drive variance stays low at every $\Delta t$.

**Compared to CUBA:** same discretisation philosophy, plus an exponential synapse that low-passes presynaptic drive before it reaches the membrane — this isolates whether the synaptic low-pass alone carries dt-stability.

### COBA

Conductance-based LIF with exponential synapses — the simplest biophysical model. The membrane follows

$$
C_m\, \frac{dV}{dt} = -g_L\,(V - E_L) - g_e\,(V - E_e) - g_i\,(V - E_i)
$$

Euler-discretised with explicit $\Delta t$:

$$
V_{t+1} = V_t + \frac{\Delta t}{C_m}\Bigl[-g_L(V_t - E_L) + g_e(E_e - V_t) + g_i(E_i - V_t)\Bigr]
$$

Each synaptic conductance evolves as an exponential synapse: $g_{t+1} = e^{-\Delta t/\tau_{\text{syn}}}(g_t + W\,s_t)$, with $\tau_{\text{AMPA}} = 2$ ms for excitation and $\tau_{\text{GABA}} = 9$ ms for inhibition. In feedforward-only COBA ($ei\text{-}strength = 0$), the inhibitory term drops: $g_i \equiv 0$. On crossing threshold the voltage is hard-reset, $V \leftarrow V_{\text{reset}}$, and the neuron is refractory for $\tau_{\text{ref}}^{E} = 3$ ms (E) or $\tau_{\text{ref}}^{I} = 1.5$ ms (I). Dale's law on: weights non-negative (half-normal init, clamped in forward).

**Biophysical constants:**

| Parameter | E population | I population |
| --------- | ------------ | ------------ |
| $\tau_m$ (ms) | 20 | 5 |
| $C_m$ (nF) | 1.0 | 0.5 |
| $g_L$ (µS) | 0.05 | 0.1 |
| $\tau_{\text{ref}}$ (ms) | 3 | 1.5 |
| $E_L$ (mV) | −65 | −65 |
| $V_{\text{th}}$ (mV) | −50 | −50 |
| $V_{\text{reset}}$ (mV) | −65 | −65 |
| $E_e$ (mV, reversal) | 0 | 0 |
| $E_i$ (mV, reversal) | −80 | −80 |

Training needs a lower learning rate ($10^{-4}$ vs $10^{-2}$ for CUBA) because BPTT through conductance gradients is steeper; $c_{m,\text{back}} = 80$ dampens voltage gradients via a forward-identity, backward-scale trick to keep training stable.

COBA is not a separate CLI model — run it as PINGNet with the inhibitory population switched off: *--model ping --ei-strength 0*. Same dynamics, no E→I→E loop. Rate-integrating by construction, so naturally dt-invariant.

**Compared to CUBA-exp:** adds hard reset, refractory period, and a conductance-based membrane driven by $g(V - V_e)$ — this isolates whether the remaining biophysical features contribute beyond what the exp synapse alone already delivers.

### PING

Full E-I network with frozen recurrent weights, producing gamma oscillations. Excitatory and inhibitory populations (E:I ratio 4:1) share COBA-style membrane dynamics; three frozen recurrent matrices $W^{EE}, W^{EI}, W^{IE}$ connect them into the standard PING loop. Per-step synaptic updates (kick-then-decay with $\alpha = e^{-\Delta t/\tau_{\text{AMPA}}}$, $\gamma = e^{-\Delta t/\tau_{\text{GABA}}}$):

$$
g^{E \to E}_{t+1} = \alpha\bigl(g^{E \to E}_t + W^{EE}\, s^{E}_t\bigr) + W_{\text{in}}\, s^{\text{inp}}_t
$$

$$
g^{E \to I}_{t+1} = \alpha\bigl(g^{E \to I}_t + W^{EI}\, s^{E}_t\bigr)
$$

$$
g^{I \to E}_{t+1} = \gamma\bigl(g^{I \to E}_t + W^{IE}\, s^{I}_t\bigr)
$$

Feedforward input is added post-decay (i.e. as an instantaneous conductance kick that the next membrane step integrates once). Recurrent components go through kick-then-decay as usual. The E population integrates $g^{E \to E}$ excitation + $g^{I \to E}$ inhibition via COBA; the I population integrates $g^{E \to I}$ only. Both use the COBA membrane equation above with their respective $(C_m, g_L, \tau_{\text{ref}})$. Pinglab defaults use $W^{EE} = 0$ (Börgers: PING needs no E→E), $W^{EI} \approx 1$ µS (just suprathreshold), $W^{IE} \approx 3$ µS (2–3× $W^{EI}$, Viriyopase et al.). The recurrent matrices are **frozen at init** — only $W_{\text{in}}$ and $W_{\text{out}}$ train, matching the trainable surface area of the other three models so the ladder is apples-to-apples.

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

**What CUBA still doesn't fix.** Variance, not mean. At coarse $\Delta t$ many coincident input spikes land in one step, hard reset discards the overshoot, and accuracy collapses. COBA dodges this because its exponential synapse integrates input through $\tau_{\text{AMPA}} = 2$ ms before it reaches the membrane.

## Common primitives

Shared machinery under every model in the ladder.

### Poisson input encoding

Pixels become Poisson spike trains. For a pixel with normalised intensity $x \in [0, 1]$, input neuron $i$ fires a Bernoulli spike at each step with probability

$$
p_i(t) = x_i \cdot r_{\max} \cdot \Delta t / 1000, \qquad r_{\max} = 25 \text{ Hz}
$$

so the per-neuron firing rate is $x_i \cdot r_{\max}$ Hz, independent of $\Delta t$. The resulting $(T_{\text{steps}}, N_{\text{in}})$ binary tensor is the model's only input. Stimulus duration is $T_{\text{ms}} = 1000$ ms.

### Surrogate gradient

The spike function $S = \mathbf{1}[U \geq \theta]$ has zero gradient almost everywhere, so backward passes use a surrogate. Pinglab uses fast-sigmoid with pseudo-derivative

$$
\frac{\partial \tilde S}{\partial U}\bigg|_{U} = \frac{k}{(1 + k\,|U - \theta|)^2}
$$

with slope $k = 1$ on both the biophysical path (mV-scale membrane) and the snnTorch-clone path (dimensionless membrane). The snnTorch-library path uses the library default $k = 25$ — the asymmetry is deliberate and is part of what the parity test exposes.

### Linear readout

Every model in the ladder uses the same output head: accumulate last-hidden-layer spikes over the trial and project through a trained linear layer.

$$
\hat y_t = \Bigl(\sum_{s \leq t} s^{\text{hid}}_s\Bigr) W_{\text{out}} + b_{\text{out}}
$$

No output-layer spiking dynamics, no softmax — logits are read out at each step and the final $\hat y_{T}$ is used for cross-entropy loss. Shared readout means any accuracy gap across the ladder cannot localise to the output.

### Weight init

Feedforward weights are sampled from a fan-in-normalised half-normal (Dale's law) or normal (signed):

$$
W \sim \mathcal{N}(\mu, \sigma^2), \qquad W \leftarrow W / N_{\text{pre}}
$$

with optional sparsity $s \in [0, 1)$: a fraction $s$ of entries are zeroed, and surviving entries are rescaled by $1/(1-s)$ so that total expected synaptic input per post-neuron is preserved. Under Dale's law ($\mu \geq 0$, clamp $< 0$ to $0$), signs are fixed at init and the forward pass re-clamps on every step.
