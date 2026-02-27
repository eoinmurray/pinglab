---
title: docs.5-input-spike-wiring
description: Spec and test plan for spike-driven external input wiring
---

# Spike-driven input wiring (E-only)

This doc is the implementation spec for the next simulator input mode.

Scope is intentionally narrow:

- E-only network
- no inhibition
- no oscillatory network mechanism
- no direct waveform-to-voltage injection

The question is only: how does a signal physically enter the network?

## Principle

External input is a virtual presynaptic population `X`, not a voltage waveform.

For each excitatory neuron `E_i`, create an independent input source `X_i`.

`X_i` emits spikes. Those spikes are delivered through synapses into `E_i` exactly like ordinary presynaptic events.

## Input generator

For each `X_i`, sample an inhomogeneous Poisson spike train with rate

$$
\lambda(t) = \lambda_0 [1 + m\sin(2\pi f t)]
$$

with nonnegative clamp:

$$
\lambda(t) \leftarrow \max(0, \lambda(t))
$$

Discrete-time probability per step:

$$
p_t = \text{clip}(\lambda(t)\,dt, 0, 1)
$$

Implementation rules:

- generate per-neuron trains independently
- do not share a global input train
- keep `dt` small (target about `1 ms`)

## Synaptic delivery into E

Each external spike contributes a synaptic event current to its target `E_i`.

Use standard current-based synapse dynamics:

Here `x_i[t]` is the external spike indicator for neuron `i` at step `t`:
`x_i[t] = 1` if `X_i` spikes in that bin, otherwise `x_i[t] = 0`.

$$
s_i[t+1] = s_i[t] e^{-dt/\tau_{in}} + x_i[t]
$$

$$
I^{ext}_i[t] = w_{in} s_i[t]
$$

and membrane update includes leak plus this synaptic drive:

$$
V_i[t+1] = V_i[t] + \text{leak terms} + I^{ext}_i[t]
$$

Default regime:

- `\tau_in` in `2-5 ms`
- `w_in` small enough that one spike is a nudge, not a guaranteed threshold crossing
- zero input delay for first pass

## What to measure

We are decoding the input envelope from population output, not from single-neuron traces.

Processing path:

1. collect all E spikes
2. bin at `2-5 ms`
3. convert to population rate
4. low-pass filter below `10 Hz`

That final smoothed population-rate trace is the decoded signal estimate.

## Low-pass math (decode stage)

Let `r_b[t]` be the binned population-rate sequence (Hz) sampled every `\Delta t` seconds.

Use a first-order causal low-pass filter with cutoff `f_c`:

$$
\tau_c = \frac{1}{2\pi f_c}, \qquad f_c = 10\ \text{Hz}
$$

Continuous-time form:

$$
\tau_c \frac{dy(t)}{dt} + y(t) = r_b(t)
$$

Discrete-time implementation (forward-Euler style):

$$
\alpha = \frac{\Delta t}{\tau_c + \Delta t}
$$

$$
y[t] = y[t-1] + \alpha \left(r_b[t] - y[t-1]\right)
$$

Equivalent weighted-average form:

$$
y[t] = (1-\alpha) y[t-1] + \alpha r_b[t]
$$

where `y[t]` is the decoded envelope estimate before any final normalization.

### Frequency response and lag

For sinusoidal component frequency `f`, first-order LPF magnitude and phase are:

$$
|H(f)| = \frac{1}{\sqrt{1 + (f/f_c)^2}}
$$

$$
\phi(f) = -\arctan(f/f_c)
$$

So the decoded signal is attenuated and delayed (causal lag). At `f=f_c`, gain is `1/\sqrt{2}` and phase lag is `-45^\circ`.

Approximate time delay at frequency `f`:

$$
\Delta t_{\text{lag}}(f) \approx \frac{|\phi(f)|}{2\pi f}
$$

Interpretation for simulator:

- if envelope frequency is much lower than `10 Hz`, recovery is close in shape and lag is small
- as envelope frequency approaches `10 Hz`, lag and attenuation become significant
- correlation should be interpreted with this causal phase shift in mind

## Simulator test protocol

Run this test in simulator once spike-input mode exists.

### Test A: Independent vs shared input

- condition 1: independent `X_i` per neuron
- condition 2: one shared train copied to all neurons

Expected:

- independent case: realistic noisy population decoding
- shared case: unrealistically strong synchrony and inflated coherence

### Test B: Synapse strength sweep

Sweep `w_in` low to high while keeping all else fixed.

Expected:

- low `w_in`: weak recoverable envelope
- moderate `w_in`: best decoding
- high `w_in`: deterministic-looking locking and loss of stochastic realism

### Test C: Population size

Use `N_E` in `{50, 100, 200}`.

Expected:

- larger `N_E` improves envelope estimate SNR
- envelope is noisier but still present at lower `N_E`

## Pass criteria

Minimum pass for this stage:

1. input is implemented as spikes-through-synapses, not direct current waveform injection
2. independent-input condition recovers envelope in low-pass population rate
3. shared-input artifact is visible and clearly worse/interpretable
4. single spikes do not deterministically force firing

## Why this is the anchor

When we later add `E->I`, `I->E`, and PING dynamics, input wiring should remain unchanged.

That gives a clean causal claim:

- if behavior changes later, it comes from network dynamics, not from moving the input goalposts.
