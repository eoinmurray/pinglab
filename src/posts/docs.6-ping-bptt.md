---
title: docs.6-ping-bptt
description: The BPTT algorithm derived for our PING conductance-based LIF network.
---

# BPTT for conductance-based PING networks

This document derives backpropagation through time (BPTT) for our
conductance-based LIF network with PING inhibitory feedback, synaptic
delays, and surrogate spike gradients. Every equation maps to the
implementation in `simulate_network.py`.

## Notation

| Symbol | Shape | Description |
| --- | --- | --- |
| $B$ | scalar | Batch size |
| $N_E$ | scalar | Number of excitatory neurons |
| $N_I$ | scalar | Number of inhibitory neurons |
| $N = N_E + N_I$ | scalar | Total neuron count |
| $T$ | scalar | Number of simulation timesteps |
| $V_t$ | $[B, N]$ | Membrane voltage at step $t$ |
| $g_{e,t}$ | $[B, N]$ | Excitatory conductance at step $t$ |
| $g_{i,t}$ | $[B, N]$ | Inhibitory conductance at step $t$ |
| $s_t$ | $[B, N]$ | Spike indicator at step $t$ (0 or 1) |
| $I_t^\text{ext}$ | $[B, N]$ | External input current at step $t$ |
| $W_{ee}$ | $[N_E, N_E]$ | E→E weight matrix |
| $W_{ei}$ | $[N_I, N_E]$ | E→I weight matrix |
| $W_{ie}$ | $[N_E, N_I]$ | I→E weight matrix |
| $W_{ii}$ | $[N_I, N_I]$ | I→I weight matrix |
| $d$ | scalar | Synaptic delay (in timesteps) |

Subscripts $E$ and $I$ denote excitatory and inhibitory subpopulations.
We write $s_{E,t} = s_t[:, :N_E]$ and $s_{I,t} = s_t[:, N_E:]$.

## 1. Forward pass

The simulation proceeds as a sequence of stages per timestep.

### Stage A: Apply delayed spikes → conductances

At each timestep $t$, spikes emitted $d$ steps ago are read from the
circular delay buffer and converted to conductance increments via the
weight matrices:

$$
\Delta g_{e,t} = \begin{bmatrix} s_{E,t-d_{ee}} \, W_{ee}^T \\ s_{E,t-d_{ei}} \, W_{ei}^T \end{bmatrix}, \quad
\Delta g_{i,t} = \begin{bmatrix} s_{I,t-d_{ie}} \, W_{ie}^T \\ s_{I,t-d_{ii}} \, W_{ii}^T \end{bmatrix}
$$

where $s_{E,t-d} \in [B, N_E]$ and the matmul produces the conductance
contribution to each postsynaptic neuron. The top block targets E
neurons, the bottom block targets I neurons.

Conductances are first decayed, then incremented:

$$
\tilde{g}_{e,t} = \alpha_e \, g_{e,t-1}, \quad g_{e,t} = \tilde{g}_{e,t} + \Delta g_{e,t}
$$

$$
\tilde{g}_{i,t} = \alpha_i \, g_{i,t-1}, \quad g_{i,t} = \tilde{g}_{i,t} + \Delta g_{i,t}
$$

where $\alpha_e = e^{-dt/\tau_\text{ampa}}$ and
$\alpha_i = e^{-dt/\tau_\text{gaba}}$ are the exponential decay factors.
In the code, decay is out-of-place (`g_e = g_e * decay_e`) to preserve
the autograd graph.

### Stage B: LIF voltage update

The conductance-based LIF equation (Euler discretization):

$$
V_t' = V_{t-1} + \frac{dt}{C_m}\bigl[g_L(E_L - V_{t-1}) + g_{e,t}(E_e - V_{t-1}) + g_{i,t}(E_i - V_{t-1}) + I_t^\text{ext}\bigr]
$$

Expanding into a form that separates the voltage-dependent terms:

$$
V_t' = V_{t-1}\!\left(1 - \frac{dt}{C_m}(g_L + g_{e,t} + g_{i,t})\right) + \frac{dt}{C_m}\!\left(g_L E_L + g_{e,t} E_e + g_{i,t} E_i + I_t^\text{ext}\right)
$$

This is a linear-affine map:

$$
V_t' = a_t \, V_{t-1} + b_t
$$

where:

$$
a_t = 1 - \frac{dt}{C_m}(g_L + g_{e,t} + g_{i,t}), \quad
b_t = \frac{dt}{C_m}(g_L E_L + g_{e,t} E_e + g_{i,t} E_i + I_t^\text{ext})
$$

An optional voltage floor clamp is applied: $V_t' \leftarrow \max(V_t', V_\text{floor})$.

### Stage C: Spike generation (surrogate gradient)

The spike decision uses a surrogate gradient function:

$$
s_t = H_\sigma(V_t' - V_{th}) \cdot \mathbb{1}[\text{can\_spike}_t]
$$

**Forward pass:** $H_\sigma(u) = \mathbb{1}[u \ge 0]$ (hard threshold).

**Backward pass:** The non-differentiable Heaviside is replaced by the
fast-sigmoid surrogate:

$$
H_\sigma'(u) = \frac{1}{(1 + |u|)^2}
$$

After spiking, voltage is reset (out-of-place via `torch.where` for
autograd compatibility):

$$
V_t = \begin{cases} V_\text{reset} & \text{if } s_t = 1 \text{ or not can\_spike} \\ V_t' & \text{otherwise} \end{cases}
$$

### Stage D: Write spikes to delay buffer

The spike vector $s_t$ is written to the circular delay buffer at
position $(t + d) \bmod L$ where $L$ is the buffer length:

$$
\text{buf}_{ee}[(t + d_{ee}) \bmod L] \leftarrow s_{E,t}, \quad
\text{buf}_{ei}[(t + d_{ei}) \bmod L] \leftarrow s_{E,t}
$$
$$
\text{buf}_{ie}[(t + d_{ie}) \bmod L] \leftarrow s_{I,t}, \quad
\text{buf}_{ii}[(t + d_{ii}) \bmod L] \leftarrow s_{I,t}
$$

**Currently, $s_t$ is detached before writing** — see Section 4.

## 2. Readout and loss

After $T$ timesteps, logits are computed from the output subpopulation
$E_\text{out} \subset E$ (neurons $[\text{out\_start}, \text{out\_stop})$).

### Spike-count readout

$$
\text{logits}_c = \sum_{t=1}^{T} s_c(t), \quad c \in E_\text{out}
$$

### Voltage readout

$$
\text{logits}_c = \frac{1}{T} \sum_{t=1}^{T} V_c(t)
$$

### Hybrid readout

$$
\text{logits}_c = \sum_{t=1}^{T} s_c(t) + \alpha \cdot \frac{1}{T} \sum_{t=1}^{T} V_c(t)
$$

The loss is cross-entropy:

$$
\mathcal{L} = -\sum_c y_c \log \text{softmax}(\text{logits})_c
$$

## 3. BPTT: the backward pass

### 3.1 Loss → logits

For hybrid readout, denote $\bar{\ell}_c = \partial \mathcal{L} / \partial \text{logits}_c = \text{softmax}(\text{logits})_c - y_c$. Then:

$$
\frac{\partial \mathcal{L}}{\partial s_c(t)} = \bar{\ell}_c, \quad
\frac{\partial \mathcal{L}}{\partial V_c(t)} = \bar{\ell}_c \cdot \frac{\alpha}{T}
$$

### 3.2 Logits → pre-reset voltage $V_t'$

The spike gradient passes through the surrogate:

$$
\frac{\partial s_t}{\partial V_t'} = H_\sigma'(V_t' - V_{th}) = \frac{1}{(1 + |V_t' - V_{th}|)^2}
$$

Through the `torch.where` reset:

$$
\frac{\partial V_t}{\partial V_t'} = \begin{cases} 0 & \text{if neuron spiked or refractory} \\ 1 & \text{otherwise} \end{cases}
$$

Combined gradient arriving at $V_t'$ for an output neuron $c$:

$$
\delta_{V_t'}^{(c)} = \bar{\ell}_c \cdot H_\sigma'(V_t' - V_{th}) + \frac{\alpha}{T} \cdot \bar{\ell}_c \cdot \frac{\partial V_t}{\partial V_t'}
$$

### 3.3 Pre-reset voltage → state variables

From the LIF update (Stage B), the Jacobians are:

$$
\frac{\partial V_t'}{\partial V_{t-1}} = 1 - \frac{dt}{C_m}(g_L + g_{e,t} + g_{i,t}) = a_t
$$

$$
\frac{\partial V_t'}{\partial g_{e,t}} = \frac{dt}{C_m}(E_e - V_{t-1})
$$

$$
\frac{\partial V_t'}{\partial g_{i,t}} = \frac{dt}{C_m}(E_i - V_{t-1})
$$

With our biophysical parameters ($dt=1$, $C_m=1$, $E_e=0$, $E_i=-80$,
$V \approx -65$):

| Jacobian entry | Value | Character |
| --- | --- | --- |
| $\partial V'/\partial V$ | $\approx 0.8$ | Contractive |
| $\partial V'/\partial g_e$ | $\approx +65$ | **Explosive** (excitatory driving force) |
| $\partial V'/\partial g_i$ | $\approx -15$ | Moderate (inhibitory driving force) |

### 3.4 Conductance → weight matrices

From the matmul $g_e^{(\text{post})} = s_\text{pre} \cdot W^T$ in Stage A:

$$
\frac{\partial \mathcal{L}}{\partial W_{ee}} = \sum_t \left(\frac{\partial \mathcal{L}}{\partial g_e^{(E)}(t)}\right)^T s_{E,t-d_{ee}}
$$

$$
\frac{\partial \mathcal{L}}{\partial W_{ei}} = \sum_t \left(\frac{\partial \mathcal{L}}{\partial g_e^{(I)}(t)}\right)^T s_{E,t-d_{ei}}
$$

$$
\frac{\partial \mathcal{L}}{\partial W_{ie}} = \sum_t \left(\frac{\partial \mathcal{L}}{\partial g_i^{(E)}(t)}\right)^T s_{I,t-d_{ie}}
$$

$$
\frac{\partial \mathcal{L}}{\partial W_{ii}} = \sum_t \left(\frac{\partial \mathcal{L}}{\partial g_i^{(I)}(t)}\right)^T s_{I,t-d_{ii}}
$$

Each term is an outer product $[B, N_\text{post}]^T \times [B, N_\text{pre}]$
summed over the batch — this is the gradient PyTorch autograd computes
for the matmul in `apply_delayed_events`.

### 3.5 Conductance temporal chain

From the decay equation:

$$
\frac{\partial g_{e,t}}{\partial g_{e,t-1}} = \alpha_e = e^{-dt/\tau_\text{ampa}}
$$

With $\tau_\text{ampa} = 2$ ms, $dt = 1$ ms: $\alpha_e = e^{-0.5} \approx 0.607$.

This chain alone is contractive. The instability comes from the coupling
through the LIF equation.

### 3.6 The full state-transition Jacobian

Combining Sections 3.3–3.5, the per-timestep state transition maps
$z_t = (V_t, g_{e,t}, g_{i,t})$ to $z_{t+1}$. The Jacobian is:

$$
J_t = \frac{\partial z_{t+1}}{\partial z_t} = \begin{pmatrix}
a_t & \frac{dt}{C_m}(E_e - V_{t-1}) & \frac{dt}{C_m}(E_i - V_{t-1}) \\[6pt]
\frac{\partial g_{e,t+d}}{\partial V_t} & \alpha_e & 0 \\[6pt]
\frac{\partial g_{i,t+d}}{\partial V_t} & 0 & \alpha_i
\end{pmatrix}
$$

The off-diagonal entries $\partial g / \partial V$ arise from the
**spike feedback path**: a change in $V_t$ may cause or suppress a spike,
which after delay $d$ modifies conductances. Expanding:

$$
\frac{\partial g_{e,t+d}}{\partial V_t} = W_{ee}^T \cdot H_\sigma'(V_t - V_{th})
$$

$$
\frac{\partial g_{i,t+d}}{\partial V_t} = W_{ie}^T \cdot H_\sigma'(V_t - V_{th})
$$

(for an E neuron; I neurons contribute via $W_{ei}$, $W_{ii}$).

### 3.7 Spectral analysis

The spectral radius of $J_t$ determines gradient growth. The dominant
eigenvalue comes from the $V \to g_e \to V$ feedback loop:

$$
\rho(J_t) \gtrsim \sqrt{\left|\frac{dt(E_e - V)}{C_m}\right| \cdot \|W_{ee}\| \cdot |H_\sigma'|}
$$

For the product of Jacobians over $T$ steps:

$$
\prod_{\tau=t}^{T} J_\tau \sim \rho(J)^{T-t}
$$

With $(E_e - V) \approx 65$, $\|W_{ee}\| \sim 0.1$, $H_\sigma' \sim 0.5$:

$$
\rho(J) \sim \sqrt{65 \times 0.1 \times 0.5} \approx 1.8
$$

After 200 timesteps: $1.8^{200} \approx 10^{51}$. Even the more
conservative per-step gain through a single spike-mediated path:

$$
|G_\text{step}| = \left|\frac{dt(E_e-V)}{C_m}\right| \cdot |w| \cdot |H_\sigma'| \approx 65 \times 0.1 \times 0.5 = 3.25
$$

gives $3.25^{50} \approx 10^{25}$ over 50 steps — well beyond float32.

## 4. Spike buffer detachment

### What we do

In `emit_and_schedule_spikes`, spikes are detached before buffering:

```python
spiked_bool = spiked.detach().bool()
```

This sets the off-diagonal blocks of $J_t$ to zero:

$$
J_t^{(\text{detached})} = \begin{pmatrix}
a_t & \frac{dt}{C_m}(E_e - V) & \frac{dt}{C_m}(E_i - V) \\[6pt]
0 & \alpha_e & 0 \\[6pt]
0 & 0 & \alpha_i
\end{pmatrix}
$$

This is upper-triangular with eigenvalues $\{a_t, \alpha_e, \alpha_i\}$,
all $< 1$. The gradient is now stable.

### What gradient survives

The only path from $\mathcal{L}$ to a weight matrix is through the
**read side** of the delay buffer:

$$
\mathcal{L} \xrightarrow{\partial} \text{logits} \xrightarrow{\partial} V_t' / s_t \xrightarrow{\partial} g_{e,t} \xrightarrow{\partial} W_{ee}
$$

Gradient reaches $W_{ee}$ via the matmul in `apply_delayed_events`. But
the presynaptic spike vector $s_{E,t-d}$ is detached, so gradient
*stops at $W_{ee}$* and does not flow further upstream.

Result: only the last E→E block (E\_hid→E\_out, 2,560 params) receives
gradient.

### The tradeoff

| | Full BPTT | Detached BPTT |
| --- | --- | --- |
| Temporal credit assignment | Full | None |
| Gradient stability | $\rho(J) > 1$ → explosion | $\rho(J) < 1$ → stable |
| Trainable parameter reach | All weight matrices | Last E→E block only |
| PING loop trainable | Yes | No |

## 5. The PING gradient circuit

The E↔I feedback loop creates a recurrent gradient path:

$$
V_E(t) \xrightarrow{H_\sigma'} s_E(t)
\xrightarrow[\text{delay } d_{ei}]{W_{ei}} g_e^{(I)}(t\!+\!d_{ei})
\xrightarrow{\frac{dt(E_e - V_I)}{C_m}} V_I(t\!+\!d_{ei})
$$

$$
\xrightarrow{H_\sigma'} s_I(t\!+\!d_{ei})
\xrightarrow[\text{delay } d_{ie}]{W_{ie}} g_i^{(E)}(t\!+\!d_{ei}\!+\!d_{ie})
\xrightarrow{\frac{dt(E_i - V_E)}{C_m}} V_E(t\!+\!d_{ei}\!+\!d_{ie})
$$

One full PING cycle takes $d_{ei} + d_{ie}$ timesteps (2 ms with 1 ms
delays). The gradient amplification per cycle:

$$
G_\text{cycle} = \left|\frac{dt(E_e - V_I)}{C_m}\right| \cdot |H_\sigma'|_{(I)} \cdot \left|\frac{dt(E_i - V_E)}{C_m}\right| \cdot |H_\sigma'|_{(E)} \cdot \|W_{ei}\| \cdot \|W_{ie}\|
$$

With $E_e - V_I \approx 65$, $E_i - V_E \approx -15$,
$H_\sigma' \sim 0.5$, $\|W_{ei}\| \sim 0.3$, $\|W_{ie}\| \sim 0.15$:

$$
G_\text{cycle} \approx 65 \times 0.5 \times 15 \times 0.5 \times 0.3 \times 0.15 \approx 10.9
$$

In a 200 ms simulation with $\sim$25 Hz PING rhythm ($\sim$5 full
cycles):

$$
G_\text{total} \approx 10.9^5 \approx 1.6 \times 10^5
$$

Combined with the E→E feedforward amplification, this exceeds float32
precision.

## 6. Truncated BPTT

Truncated BPTT (TBPTT) periodically detaches the state every $K$ steps:

$$
\text{every } K \text{ steps: } V_t \leftarrow \text{detach}(V_t), \quad
g_{e,t} \leftarrow \text{detach}(g_{e,t}), \quad
g_{i,t} \leftarrow \text{detach}(g_{i,t})
$$

This bounds the Jacobian product to $K$ terms. For stability:

$$
\rho(J)^K < 3.4 \times 10^{38} \implies K < \frac{38 \ln 10}{\ln \rho(J)}
$$

With $\rho(J) \approx 1.8$: $K < 149$. With $\rho(J) \approx 6.5$
(worst case): $K < 47$.

**However**, TBPTT on $(V, g_e, g_i)$ does not cut the spike buffer
chain. Spikes written at step $t$ are read at step $t+d$ — this temporal
dependency goes through the buffer, not the state variables. To fully
truncate, one must also periodically detach the buffer contents. Our
per-step spike detachment is the extreme case: $K = 1$ for the spike
buffer.

## 7. Weight gradient equations (summary)

Regardless of how the adjoint $\lambda$ is computed (full BPTT, TBPTT,
detached, or an alternative rule), the weight gradients take the same
form:

$$
\nabla_{W_{ee}} \mathcal{L} = \sum_{t=1}^{T} \lambda_{g_e}^{(E)}(t)^T \, s_{E,t-d_{ee}}
$$

$$
\nabla_{W_{ei}} \mathcal{L} = \sum_{t=1}^{T} \lambda_{g_e}^{(I)}(t)^T \, s_{E,t-d_{ei}}
$$

$$
\nabla_{W_{ie}} \mathcal{L} = \sum_{t=1}^{T} \lambda_{g_i}^{(E)}(t)^T \, s_{I,t-d_{ie}}
$$

$$
\nabla_{W_{ii}} \mathcal{L} = \sum_{t=1}^{T} \lambda_{g_i}^{(I)}(t)^T \, s_{I,t-d_{ii}}
$$

where $\lambda_{g_e}^{(E)}(t) \in [B, N_E]$ is the gradient of
$\mathcal{L}$ with respect to the excitatory conductance of E neurons
at timestep $t$.

With detached spikes, $s$ carries no gradient and these outer products
are the *final* weight gradient. Without detachment, backpropagating
through $s_{t-d}$ adds recursive terms — precisely the terms that
explode.

## 8. Population-level Jacobian

Our network has four populations. Writing the state as
$(V_E, V_I, g_e^{(E)}, g_e^{(I)}, g_i^{(E)}, g_i^{(I)})$, the full
Jacobian decomposes into blocks:

$$
J = \begin{pmatrix}
a_E & 0 & \frac{dt}{C_E}(E_e\!-\!V_E) & 0 & \frac{dt}{C_E}(E_i\!-\!V_E) & 0 \\[4pt]
0 & a_I & 0 & \frac{dt}{C_I}(E_e\!-\!V_I) & 0 & \frac{dt}{C_I}(E_i\!-\!V_I) \\[4pt]
W_{ee}^T H_\sigma' & 0 & \alpha_e & 0 & 0 & 0 \\[4pt]
W_{ei}^T H_\sigma' & 0 & 0 & \alpha_e & 0 & 0 \\[4pt]
0 & W_{ie}^T H_\sigma' & 0 & 0 & \alpha_i & 0 \\[4pt]
0 & W_{ii}^T H_\sigma' & 0 & 0 & 0 & \alpha_i
\end{pmatrix}
$$

The E→E recurrent loop corresponds to the $(1,3)$–$(3,1)$ block cycle.
The PING loop corresponds to the
$(1,3)$–$(4,1)$–$(2,4)$–$(5,2)$–$(1,5)$ path (E spikes → I
conductances → I voltage → I spikes → E inhibitory conductances → E
voltage).

With detachment, rows 3–6 lose the $W^T H_\sigma'$ entries, making $J$
upper-triangular with spectral radius $\max(|a_E|, |a_I|, \alpha_e, \alpha_i) < 1$.

## 9. Continuous-time adjoint

In the continuous-time limit, BPTT corresponds to solving the adjoint
ODE backward. Define:

$$
\lambda_V(t) = \frac{d\mathcal{L}}{dV(t)}, \quad
\lambda_{g_e}(t) = \frac{d\mathcal{L}}{dg_e(t)}, \quad
\lambda_{g_i}(t) = \frac{d\mathcal{L}}{dg_i(t)}
$$

The adjoint dynamics (time reversed) are:

$$
-\frac{d\lambda_V}{dt} = \lambda_V \cdot \frac{\partial f_V}{\partial V} + \lambda_{g_e} \cdot \frac{\partial f_{g_e}}{\partial V} + \lambda_{g_i} \cdot \frac{\partial f_{g_i}}{\partial V}
$$

where $f_V = dV/dt$, $f_{g_e} = dg_e/dt$, etc. The $\partial f_g / \partial V$
terms involve the spike mechanism:

$$
\frac{\partial f_{g_e}(t')}{\partial V(t)} = \sum_k w_{kj} \, H_\sigma'(V(t_k) - V_{th}) \, \delta(t' - t_k - d)
$$

The delta function makes this a sum of impulses at spike times, each
scaled by the weight and surrogate gradient. The adjoint equation
inherits the same multiplicative driving-force instability as the
discrete case.

The weight gradient in continuous time:

$$
\frac{d\mathcal{L}}{dW_{ee}} = \int_0^T \lambda_{g_e}^{(E)}(t) \cdot s_E(t - d_{ee})^T \, dt
$$

which discretizes to the sum in Section 7.

## Summary

| Quantity | Expression | Value |
| --- | --- | --- |
| V→V Jacobian | $a_t = 1 - \frac{dt}{C_m}(g_L + g_e + g_i)$ | $\approx 0.8$ (stable) |
| V→$g_e$ Jacobian | $\frac{dt}{C_m}(E_e - V)$ | $\approx 65$ (explosive) |
| V→$g_i$ Jacobian | $\frac{dt}{C_m}(E_i - V)$ | $\approx -15$ |
| $g_e$ decay | $\alpha_e = e^{-dt/\tau_\text{ampa}}$ | $\approx 0.607$ (stable) |
| $g_i$ decay | $\alpha_i = e^{-dt/\tau_\text{gaba}}$ | $\approx 0.857$ (stable) |
| Per-step spike path gain | $(E_e - V) \cdot w \cdot H_\sigma'$ | $\sim 3$–$20$ |
| Per PING cycle gain | Full E↔I loop | $\sim 11$ |
| Detached Jacobian $\rho$ | $\max(a_t, \alpha_e, \alpha_i)$ | $< 1$ (stable) |

The conductance-based LIF model creates a fundamental tension: the
$(E_e - V)$ driving force that enables shunting inhibition and PING
oscillations is the same term that makes BPTT unstable. See
[docs.5](docs.5-ping-exploding-gradients) for possible solutions.
