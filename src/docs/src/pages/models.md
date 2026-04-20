---
layout: ../layouts/MarkdownLayout.astro
title: "Models"
---

# Models

The shared model ladder used across experiments: five models, ordered by increasing biophysical realism and time-discretisation rigour. All take MNIST Poisson spikes as input and output digit-class logits. The first two (snnTorch-library, standard-snn) share the same update rule — snnTorch-library is the reference, standard-snn is pinglab's in-repo version — and serve as a calibration pair. For the project-level "why," see [Introduction](/introduction/).

| Model | Description |
| ----- | ----------- |
| snnTorch-library | Thin wrapper around snnTorch's *snn.Leaky* + fast-sigmoid surrogate. Parity reference for the pinglab path. |
| standard-snn | snnTorch-library form, pinglab implementation: $\beta$ a dimensionless hyperparameter. No dt semantics. Not dt-stable. |
| CUBA | Exact (exponential-Euler + zero-order hold) discretisation of $\tau\,dV/dt = -V + I$. Dt-stable across $\Delta t \in [0.05, 2.0]$ ms. |
| COBA | Full biophysical: exp synapse + hard reset + refractory + conductance-$V$. Not a separate CLI model — run as *--model ping --ei-strength 0*. |
| PING | COBA + E→I→E loop producing gamma oscillation. Dt-stable below $\tau_{\text{GABA}}$ ceiling. |

Each step up the ladder adds one axis of realism or rigour:

- **snnTorch-library → standard-snn**: same update rule, different implementation. Isolates *"does the pinglab LIF step and surrogate-gradient code match the library's numerics?"*
- **standard-snn → CUBA**: same neuron class, cleaner discretisation. Isolates the dt-sensitivity question — *the exp-Euler + ZOH form fixes it* ([nb003](/notebook/nb003/)).
- **CUBA → COBA**: add hard reset, refractory, conductance-based membrane, and exponential synapses. Isolates *"do the biophysical features add anything beyond the CUBA baseline?"*
- **COBA → PING**: add inhibitory population and E→I→E coupling. Isolates *"what does the gamma oscillation contribute?"*

## The model ladder

### snnTorch-library

Implementation: [SNNTorchLibraryNet](https://github.com/eoinmurray/pinglab/blob/main/src/pinglab/models.py) in *src/pinglab/models.py*.

A thin wrapper around snnTorch's own [*snn.Leaky*](https://snntorch.readthedocs.io/en/latest/snn.neurons_leaky.html) module with the library's [fast-sigmoid surrogate](https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html) ([Eshraghian et al. 2023](https://arxiv.org/abs/2109.12894); see also [snnTorch Tutorial 3](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_3.html) for the *snn.Leaky* walkthrough and [Tutorial 5](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html) for the feedforward SNN reference pattern). Per-step update for one hidden layer:

$$
U_{t+1} = \beta\, U_t + W\, s_t + b \;-\; \theta\, S_t, \qquad S_t = \mathbf{1}[U_t \geq \theta] \tag{1}
$$

with reset-by-subtraction on spike — snnTorch's *reset_mechanism="subtract"* option, one of *subtract*, *zero*, or *none* ([*snn.Leaky* API](https://snntorch.readthedocs.io/en/latest/snn.neurons_leaky.html)). The canonical snnTorch spec treats $\beta \in [0, 1)$ as a dimensionless hyperparameter and has no $\Delta t$ in the update (see [Tutorial 3 § The Decay Rate: beta](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_3.html#the-decay-rate-beta), where β is introduced as a step-indexed scalar).

snnTorch does, however, recommend a way to reintroduce a timestep: [Tutorial 3 § 1.4](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_3.html#1-4-lapicque-s-lif-neuron-model) writes *beta = torch.exp(-delta_t / tau)* at a fixed $(\Delta t, \tau)$ (e.g. $\Delta t = 1$ ms, $\tau = 5$ ms, giving $\beta = 0.819$). Pinglab uses exactly this pattern with $\tau_{\text{snn}} = 10$ ms, recomputing β whenever Δt changes so snnTorch-library and standard-snn share the same $\tau_{\text{mem}}$. Note that this is a **partial discretisation**: β tracks Δt, but $W$ and $b$ — which already absorbed the Tutorial 3 § 1.2 $(1-\beta)$ prefactor — do not. A full Euler step would restore $(1-\beta)/\Delta t$ on the spike drive and $(1-\beta)$ on the bias; that's the CUBA rung. Consequently a $\Delta t$-sweep probes a regime the canonical spec never promised to support, and [notebook 003](/notebook/nb003/) documents what happens when users take it there anyway.

Its purpose is calibration, not experimentation. At matched config, snnTorch-library and pinglab's standard-snn should train to within a small tolerance — any residual gap localises to the LIF-step implementation or surrogate-gradient details, not to the architecture. See [notebook 003](/notebook/nb003/) for the side-by-side training comparison and Δt-stability sweep.

### standard-snn

Implementation: [CUBANet](https://github.com/eoinmurray/pinglab/blob/main/src/pinglab/models.py) in *src/pinglab/models.py*. Selected by the default constructor args — *discretisation="snntorch"* and *exponential_synapse=False* — which is what distinguishes this path from its CUBA siblings on the same class.

Pinglab's in-repo reimplementation of the same forward rule as [*snn.Leaky*](https://snntorch.readthedocs.io/en/latest/snn.neurons_leaky.html):

$$
U_{t+1} = \beta\, U_t + W\, s_t + b, \qquad S_t = \sigma_{\text{surr}}(U_t - \theta) \tag{2}
$$

$$
U_{t+1} \leftarrow \begin{cases} 0 & \text{if reset}=\text{zero and } S_t = 1 \\ U_{t+1} - \theta\, S_t & \text{if reset}=\text{subtract} \end{cases} \tag{3}
$$

The canonical spec is $\Delta t$-free by deliberate simplification, not by oversight. [Tutorial 3 § 1.1–1.2](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_3.html#the-decay-rate-beta) *derives* the Euler form $U(t+\Delta t) = (1-\Delta t/\tau)\, U(t) + (\Delta t/\tau)\, I_{\text{in}} R$, defines $\beta = e^{-\Delta t/\tau}$, then steps away from "physically viable assumptions" to set $\Delta t = 1$, $R = 1$, and subsume the resulting $(1-\beta)$ prefactor into a learnable weight $W$. What remains is $U[t+1] = \beta\, U[t] + W\, X[t]$: time is step-indexed, $\beta$ is a dimensionless hyperparameter, and each input spike adds exactly $W_i$ to the membrane. Bias is added every step; threshold $\theta = 1$.

This is the deep-learning framing of an SNN: a recurrent network with binary activations and a leaky scalar state, parameterised by $\beta$ — the BPTT-through-surrogate setup detailed in [snnTorch Tutorial 5](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html). It has no $\Delta t$-invariance because $\Delta t$ was collapsed out of the spec upstream. Pinglab's dt-sweep reintroduces it only on one side of the simplification: $\beta = e^{-\Delta t/\tau_{\text{snn}}}$ is recomputed per $\Delta t$, but $W$ and $b$ — which have already absorbed the old $(1-\beta)$ — are left unscaled. A bias $b$ is added once per step, so over fixed real time it contributes $T_{\text{ms}}/\Delta t$ times — 10× more often at fine $\Delta t$. That asymmetric reintroduction of $\Delta t$ is the bias-balloon failure.

**Compared to snnTorch-library:** identical update rule, different implementation — any numeric gap between the two localises to the LIF step or surrogate-gradient code, not to the model spec.

**Compared to CUBA (next rung):** CUBA literally undoes Tutorial 3 § 1.2's $\Delta t = 1$ substitution — integrating $\tau\, dV/dt = -V + I$ over a step of arbitrary length $\Delta t$ puts the $(1-\beta)/\Delta t$ and $(1-\beta)$ prefactors back onto the spike and bias terms. That restores the $\Delta t$-invariance that the canonical simplification sacrificed.

### CUBA

Implementation: [CUBANet](https://github.com/eoinmurray/pinglab/blob/main/src/pinglab/models.py) in *src/pinglab/models.py*. Selected by *discretisation="continuous"* (standard-snn leaves it at "snntorch"), with *exponential_synapse=False*.

Proper continuous-time discretisation of the physicist's LIF:

$$
\tau\, \frac{dV}{dt} = -V + I(t), \qquad I(t) = \sum_k W\, \delta(t - t_k) + b \tag{4}
$$

with presynaptic spikes as Dirac deltas plus a constant bias current. Derivation to the discrete update via **exact / exponential-Euler integration**: solve the homogeneous part of (4) in closed form over one step — giving the exponential decay factor $\beta = e^{-\Delta t/\tau}$ that characterises this scheme, and distinguishing it from forward Euler, which would approximate the decay as $(1 - \Delta t/\tau)$. The inhomogeneous (input-driven) part is integrated under a **zero-order hold** on the input — i.e. the input is held piecewise-constant across each step — which is where the $(1-\beta)$ prefactors on the drive terms originate:

$$
V(t + \Delta t) = e^{-\Delta t/\tau}\, V(t) + \frac{1}{\tau} \int_0^{\Delta t} e^{-(\Delta t - u)/\tau}\, I(t + u)\, du \tag{5}
$$

Split $I$ into spike and bias parts. The bias $b$ is already constant. The spike part is where the zero-order hold does work: rather than integrate literal Dirac deltas — which would make each kick depend on *where in the step* the spike lands, an unpleasant within-step dependence — the within-step spike count $s_t$ is treated as a rectangular current pulse of height $W\, s_t / \Delta t$ spread uniformly across the step. That is the explicit ZOH approximation, and it is the honest source of the $1/\Delta t$ factor in the spike drive:

$$
V_{t+1} = \beta\, V_t + \frac{1}{\tau} \int_0^{\Delta t} e^{-(\Delta t - u)/\tau} \left[\frac{W\, s_t}{\Delta t} + b\right] du, \qquad \beta = e^{-\Delta t/\tau} \tag{6}
$$

The integrand's $u$-dependence is purely in the exponential, so

$$
\frac{1}{\tau} \int_0^{\Delta t} e^{-(\Delta t - u)/\tau}\, du = 1 - e^{-\Delta t/\tau} = 1 - \beta \tag{7}
$$

Pulling that factor out of both terms yields the update:

$$
U_{t+1} = \beta\, U_t + \frac{1 - \beta}{\Delta t}\, W\, s_t + (1 - \beta)\, b \tag{8}
$$

followed by the same spike / reset rule as standard-snn. Two key consequences:

- **Per-spike kick** $= (1-\beta)/\Delta t \cdot W \approx W/\tau$ as $\Delta t \to 0$ — neither $\Delta t$ nor $\beta$ appears in the limit. Dt-invariant in magnitude.
- **Per-ms bias contribution** $= (1-\beta)/\Delta t \cdot b \approx b/\tau$ per ms — dt-invariant, whereas the standard-snn path injects $b$ once per step and grows bias drive by $1/\Delta t$.

Same parameters, learning rate, and training cost as standard-snn — only the forward rule differs. Empirically $\Delta t$-stable across the full $\Delta t \in [0.05, 2.0]$ ms sweep: [notebook 003](/notebook/nb003/) trains CUBA at $\Delta t = 0.1$ and $\Delta t = 1.0$ and finds test accuracy stays within $\sim$1% of the training-$\Delta t$ reference across every evaluation $\Delta t$, whereas standard-snn and snntorch-library collapse by 20–60 percentage points outside a narrow band around their training $\Delta t$. Proper discretisation is what carries CUBA's stability, independent of any synapse model.

**Weight rescaling across $\Delta t$.** The ZOH prefactor $(1-\beta)$ depends on $\Delta t$, so a weight trained at $\Delta t_{\text{ref}}$ must be rescaled to stay equivalent at $\Delta t'$. [Parthasarathy, Burghi & O'Leary](/papers/#temporal-discretisation) (their Eq 27) give the rescaling in closed form:

$$
\frac{W(\Delta t')}{W(\Delta t_{\text{ref}})} = \frac{1 - e^{-\Delta t' / \tau_m}}{1 - e^{-\Delta t_{\text{ref}} / \tau_m}} \tag{9}
$$

This is exactly the one-shot balance step CUBA applies at init: weights are drawn bit-identical to standard-snn, then scaled by (9) so that at training-$\Delta t$ both models produce the same initial firing rate. Once CUBA is trained, evaluating at a different $\Delta t$ with the *same* frozen weights works because the $(1-\beta)/\Delta t$ and $(1-\beta)$ prefactors in the update absorb the rescaling in the forward pass — there's no need to re-rescale *W* post-hoc.

**Compared to standard-snn:** same LIF neuron class, but the forward rule derives from exp-Euler integration with a zero-order hold on the input, keeping explicit $\Delta t$ semantics — this isolates whether dt-sensitivity is a property of the model or of the discretisation.

### COBA

Implementation: [PINGNet](https://github.com/eoinmurray/pinglab/blob/main/src/pinglab/models.py) in *src/pinglab/models.py*. Selected via *ei_strength=0*, which zeroes the inhibitory population's weights — leaving a feedforward-only COBA. PING keeps *ei_strength* nonzero to turn on the E→I→E loop.

Conductance-based LIF with exponential synapses — the simplest biophysical model. The membrane follows

$$
C_m\, \frac{dV}{dt} = -g_L\,(V - E_L) - g_e\,(V - E_e) - g_i\,(V - E_i) \tag{10}
$$

Derivation to the discrete update via **exponential Euler under a zero-order hold** on the conductances. Collect coefficients on $V$ on the right-hand side of (9):

$$
C_m\, \frac{dV}{dt} = -(g_L + g_e + g_i)\, V + (g_L E_L + g_e E_e + g_i E_i) \tag{11}
$$

Define $g_{\text{tot}} = g_L + g_e + g_i$, $\tau_{\text{eff}} = C_m / g_{\text{tot}}$, and $V_\infty = (g_L E_L + g_e E_e + g_i E_i) / g_{\text{tot}}$. Dividing (12) by $g_{\text{tot}}$ puts it in canonical first-order form:

$$
\tau_{\text{eff}}\, \frac{dV}{dt} = -(V - V_\infty) \tag{12}
$$

Under a zero-order hold on $g_e, g_i$ across the step, $\tau_{\text{eff}}$ and $V_\infty$ are constant, so (13) is a first-order linear ODE with the familiar closed form:

$$
V_{t+1} = V_\infty + (V_t - V_\infty)\,\exp(-\Delta t / \tau_{\text{eff}}) \tag{13}
$$

This matches CUBA's treatment of its homogeneous part, so CUBA → COBA on the ladder isolates the biophysical additions rather than an integrator swap. A forward-Euler variant is still available via *--coba-integrator fwd* for parity studies.

Each synaptic conductance evolves as an exponential synapse: $g_{t+1} = e^{-\Delta t/\tau_{\text{syn}}}(g_t + W\,s_t)$, with $\tau_{\text{AMPA}} = 2$ ms for excitation and $\tau_{\text{GABA}} = 9$ ms for inhibition. In feedforward-only COBA ($ei\text{-}strength = 0$), the inhibitory term drops: $g_i \equiv 0$. On crossing threshold the voltage is hard-reset, $V \leftarrow V_{\text{reset}}$, and the neuron is refractory for $\tau_{\text{ref}}^{E} = 3$ ms (E) or $\tau_{\text{ref}}^{I} = 1.5$ ms (I). Dale's law on: weights non-negative (half-normal init, clamped in forward).

Biophysical constants ($\tau_m$, $C_m$, $g_L$, $\tau_{\text{ref}}$, $E_L$, $V_{\text{th}}$, $V_{\text{reset}}$, $E_e$, $E_i$) are tabulated on the [Parameters & Units](/parameters-and-units/#coba--ping-biophysical-constants) page.

Training needs a lower learning rate ($10^{-4}$ vs $10^{-2}$ for CUBA) because BPTT through conductance gradients is steeper; voltage gradients are dampened via a forward-identity, backward-scale trick — see [Training § Gradient dampening for COBA/PING](/training/#gradient-dampening-for-cobaping).

COBA is not a separate CLI model — run it as PINGNet with the inhibitory population switched off: *--model ping --ei-strength 0*. Same dynamics, no E→I→E loop. Rate-integrating by construction, so naturally dt-invariant.

**Compared to CUBA:** adds exponential synapses ($\tau_{\text{AMPA}}$, $\tau_{\text{GABA}}$), hard reset, refractory period, and a conductance-based membrane driven by $g(V - V_e)$ — this isolates whether the biophysical features contribute beyond the CUBA baseline.

### PING

Implementation: [PINGNet](https://github.com/eoinmurray/pinglab/blob/main/src/pinglab/models.py) in *src/pinglab/models.py*. Uses the default nonzero *ei_strength* (with *w_ei* and *w_ie* supplied from CLI config) to activate the E→I→E recurrent loop; COBA zeros these to collapse to the feedforward case.

Full E-I network with frozen recurrent weights, producing gamma oscillations. Excitatory and inhibitory populations (E:I ratio 4:1) share COBA-style membrane dynamics; three frozen recurrent matrices $W^{EE}, W^{EI}, W^{IE}$ connect them into the standard PING loop. Per-step synaptic updates (kick-then-decay with $\alpha = e^{-\Delta t/\tau_{\text{AMPA}}}$, $\gamma = e^{-\Delta t/\tau_{\text{GABA}}}$):

$$
g^{E \to E}_{t+1} = \alpha\bigl(g^{E \to E}_t + W^{EE}\, s^{E}_t\bigr) + W_{\text{in}}\, s^{\text{inp}}_t \tag{14}
$$

$$
g^{E \to I}_{t+1} = \alpha\bigl(g^{E \to I}_t + W^{EI}\, s^{E}_t\bigr) \tag{15}
$$

$$
g^{I \to E}_{t+1} = \gamma\bigl(g^{I \to E}_t + W^{IE}\, s^{I}_t\bigr) \tag{16}
$$

Feedforward input is added post-decay (i.e. as an instantaneous conductance kick that the next membrane step integrates once). Recurrent components go through kick-then-decay as usual. The E population integrates $g^{E \to E}$ excitation + $g^{I \to E}$ inhibition via COBA; the I population integrates $g^{E \to I}$ only. Both use the COBA membrane equation above with their respective $(C_m, g_L, \tau_{\text{ref}})$. Pinglab defaults use $W^{EE} = 0$ (Börgers: PING needs no E→E), $W^{EI} \approx 1$ µS (just suprathreshold), $W^{IE} \approx 3$ µS (2–3× $W^{EI}$, Viriyopase et al.). The recurrent matrices are **frozen at init** — only $W_{\text{in}}$ and $W_{\text{out}}$ train, matching the trainable surface area of the other three models so the ladder is apples-to-apples.

With default parameters and input rate 50 Hz, the network locks at $f_0 \approx 40$ Hz with $\text{CV} \approx 0.4$ and $\sim 80\%$ activity — the intended gamma regime, not saturation. PING has a hard $\Delta t$ ceiling tied to $\tau_{\text{GABA}}$; see [Parameters & units](/parameters-and-units/) for the derivation.

**Compared to COBA:** adds an inhibitory population (E:I = 4:1) and the frozen E→I→E recurrent loop that produces gamma oscillations — this isolates what the gamma rhythm contributes on top of a purely feedforward conductance-based model.

## Model training

| Flag | standard-snn | cuba | coba | ping |
| ---- | ------------------ | ---- | ---- | ---- |
| --model | standard-snn | cuba | ping | ping |
| --dataset | mnist | mnist | mnist | mnist |
| --dt | {0.1\|1.0} | {0.1\|1.0} | {0.1\|1.0} | {0.1\|1.0} |
| --t-ms | 200 | 200 | 200 | 200 |
| --epochs | 40 | 40 | 40 | 40 |
| --adaptive-lr | ✓ | ✓ | ✓ | ✓ |
| --observe | video | video | video | video |
| --input-rate | 50 | 50 | 50 | 50 |
| --kaiming-init | ✓ | ✓ | — | — |
| --lr | 0.01 | 0.01 | 0.0001 | 0.0001 |
| --no-dales-law | ✓ | ✓ | — | — |
| --ei-strength | 0 | 0 | 0 | 0.5 |
| --cm-back-scale | — | — | 1000 | 1000 |
| --w-in | (kaiming) | (kaiming) | 0.3 | 1.2 |
| --w-in-sparsity | (kaiming) | (kaiming) | 0.95 | 0.95 |

