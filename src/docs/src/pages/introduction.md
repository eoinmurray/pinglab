---
layout: ../layouts/MarkdownLayout.astro
title: "Introduction"
---

# Introduction

The PING (Pyramidal-Interneuron Gamma) network is a biophysical model of cortical gamma oscillations: excitatory pyramidal neurons and fast inhibitory interneurons form a feedback loop that produces rhythmic activity in the 30–80 Hz band. This project trains PING networks for classification using surrogate gradients — an investigation that sits at the intersection of three fields.

## Neuroscience

Gamma oscillations are ubiquitous in cortex, but their computational role remains debated. If a PING network can be trained to classify inputs while maintaining its gamma rhythm, this provides evidence that the PING mechanism is not merely an epiphenomenon — it can support computation.

## Machine learning

Standard SNN architectures for ML are simple feedforward LIF networks with no E-I structure. Adding biophysical constraints (conductance-based synapses, Dale's law, inhibitory feedback) is usually seen as an unnecessary complication. Recent work has begun to challenge this: Yan et al. (2025, "Rhythm-SNN") show that modulating spiking neurons with heterogeneous oscillatory signals substantially improves robustness and energy efficiency on temporal tasks. Our approach is complementary — rather than imposing oscillations as an external modulator, we let gamma emerge from the network's own E-I structure.

## Neuromorphic computing

Neuromorphic hardware typically relies on a global clock to synchronise neuron updates. A PING network's gamma oscillation is a self-organised clock — the E-I loop sets its own tempo without external timing.
