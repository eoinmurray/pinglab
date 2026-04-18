---
layout: ../layouts/MarkdownLayout.astro
title: "Introduction"
---

# Introduction

The PING (Pyramidal-Interneuron Gamma) network is a biophysical model of cortical gamma oscillations: excitatory pyramidal neurons and fast inhibitory interneurons form a feedback loop that produces rhythmic activity in the 30–80 Hz band.

This project trains PING networks for classification using surrogate gradients — an investigation that sits at the intersection of three fields.

From a neuroscience angle, gamma oscillations are ubiquitous in cortex but their computational role remains debated. If a PING network can be trained to classify inputs while maintaining its gamma rhythm, that is evidence the PING mechanism is not merely an epiphenomenon but can support computation. On the machine-learning side, standard SNN architectures for classification are simple feedforward LIF networks with no E-I structure, and adding biophysical constraints — conductance-based synapses, Dale's law, inhibitory feedback — is usually seen as unnecessary complication. Recent work has begun to challenge this: Yan et al. (2025, "Rhythm-SNN") show that modulating spiking neurons with heterogeneous oscillatory signals substantially improves robustness and energy efficiency on temporal tasks, and our approach is complementary — rather than imposing oscillations as an external modulator, we let gamma emerge from the network's own E-I structure. The appeal for neuromorphic computing is similar: hardware typically relies on a global clock to synchronise neuron updates, but a PING network's gamma oscillation is a self-organised clock — the E-I loop sets its own tempo without external timing.
