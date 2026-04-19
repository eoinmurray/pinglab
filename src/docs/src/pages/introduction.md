---
layout: ../layouts/MarkdownLayout.astro
title: "Introduction"
---

# Introduction

A spiking neural network, or SNN, is a neural network whose units talk to each other in binary events called *spikes*, not continuous activations. Each unit has a leaky membrane voltage. Incoming spikes push the voltage up. When it crosses a threshold the unit fires its own spike, then resets.

Computation unfolds over time. The signal between layers is sparse, timed, and discrete — closer to biology than a standard deep network, and a natural fit for event-driven neuromorphic hardware.

Pinglab is a mod for spiking neural networks: take a standard feedforward LIF classifier and swap in a PING (Pyramidal-Interneuron Gamma) layer — the cortical circuit where excitatory pyramidal neurons and fast inhibitory interneurons form a feedback loop that produces rhythmic activity in the 30–80 Hz band.

The rest of the pipeline stays intact: Poisson-encoded inputs, surrogate-gradient BPTT, cross-entropy readout. The only thing that changes is the hidden layer.

Framed that way, the project sits at the intersection of three fields.

From a neuroscience angle, gamma oscillations are ubiquitous in cortex but their computational role remains debated. If a PING-modded SNN can be trained to classify inputs while maintaining its gamma rhythm, that is evidence the PING mechanism is not merely an epiphenomenon of cortical wiring but a circuit that can actively support computation under load.

On the machine-learning side, standard SNN architectures are simple feedforward LIF networks with no E-I structure, and adding biophysical constraints — conductance-based synapses, Dale's law, inhibitory feedback — is usually seen as unnecessary complication. This project is the opposite bet: treat PING as a drop-in replacement for the hidden layer and measure what the added structure costs and what it buys. Yan et al. (2025, "Rhythm-SNN") is complementary — they modulate neurons with external oscillations; we let gamma emerge from the E-I structure itself.

For neuromorphic hardware, the angle is timing. Chips typically rely on a global clock to synchronise neuron updates, and the clock is a design cost: routing, power, and a single point of failure. A PING-modded SNN brings its own clock — the E-I loop sets its own tempo, so the timing signal is intrinsic to the network rather than an external scaffold.
