---
title: docs.6-rhythm-snn-lit-review
description: Literature Review
---
**Slide 1**
Welcome

**Slide 2**
Paper overview

**Slide 3**
Context

**Slide 4**
These are the main claims, we will go through them as we walk through the slides.

**Slide 5**
Essentially what they are doing here is applying an oscillating mask which pauses neuron updates during the OFF phase. So the membrane potential stays constant and no spikes are allowed. The modulating signal is a variable square wave which we shall see in the next slide.

**Slide 6**
So groups of neurons get different modularity signals. You can see how they are distributed across layers in plot (b). Each modulation has a duty cycle and period, you see the shapes in upper (c). In RHS (c) you can see that the updates change depending on modulation signal. Gradients skip OFF states, and create a “highway”.

**Slide 7**
They deem this system “Rhythm-SNN”, and operate various versions of it on various tasks.

**S-MNIST** - sequential MNIST.
**P-MNIST** - permuted version of S-MNIST.
**SHD** - classifying audio recordings.
**ECG** - electrocardiograms.
**GSC** - identifying text in utterances.
**VoxCeleb1** - identifying text in utterances.
**PTB** - next word prediction.
**DVS** - Recognising gestures from video clips.
**Intent N-DNS** - 500h of speech in various languages and noise environs.

