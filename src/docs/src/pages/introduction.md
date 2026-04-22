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

## How the repo works

Pinglab is organised around one CLI and one lab notebook. Every result on this site traces back to a notebook **entry** whose **runner** drives the same CLI the experiments were developed in.

**Oscilloscope** — the training / inference / inspection CLI at *src/pinglab/oscilloscope.py*. *train* and *infer* subcommands cover every run; notebook runners shell out to it.

**Entries** — dated writeups at *src/docs/src/pages/notebook/&lt;slug&gt;.mdx*, listed newest-first on the home page. There is no separate paper layer — a paper-shaped writeup is still an entry. Slugs are *nb* + zero-padded global number (*nb001*, *nb002*, …); the date lives in the frontmatter and the long-form byline, not the slug. Most entries follow Introduction / Method / Findings / Implications / Next steps.

**Runners** — each entry that cites generated figures gets a runner at *src/pinglab/notebook/&lt;slug&gt;.py*. One command regenerates every figure the entry references and writes a *numbers.json* alongside them under *src/docs/public/figures/notebook/&lt;slug&gt;/*. The runner is the promotion gate between *src/artifacts/* (raw, gitignored) and *src/docs/public/figures/* (published). Provenance is the runner plus the commit: the runner's *SLUG*, *MAX_SAMPLES*, *EPOCHS*, *TIER* constants pin the run config; git history pins the model code.

Beyond those three: *src/pinglab/* is the code (pure Python under [uv](https://docs.astral.sh/uv/)), *src/papers/* is the bibliography, and *CLAUDE.md* + [Style guide](/styleguide/) + project-scoped memory hold the collaboration rules for how the human and AI work on this together.

## Running

Install [uv](https://docs.astral.sh/uv/) and [bun](https://bun.sh/), then clone the repo:

```sh
git clone https://github.com/eoinmurray/pinglab.git
cd pinglab
```

Train or inspect a model:

```sh
uv run python src/pinglab/oscilloscope.py --help
uv run python src/pinglab/oscilloscope.py train --model standard-snn \
  --dataset mnist --max-samples 1000 --epochs 3
```

Reproduce a notebook entry — invoke its runner (argument-free; regenerates every figure and number the entry cites):

```sh
uv run python src/pinglab/notebook/<slug>.py
```

Run the docs site (served at *localhost:3000*):

```sh
cd src/docs && bun install && bun dev
```

Run the tests (live in *src/pinglab/tests/unit/*; *slow* and *regression* markers gate the slower subset):

```sh
uv run pytest
```

## Glossary

Project-specific terms. Definitions here are load-bearing — if something elsewhere contradicts a definition, this page wins.

- **Entry** — the published writeup for a notebook investigation, at *src/docs/src/pages/notebook/&lt;slug&gt;.mdx*. The permanent, dated document that readers see.
- **Runner** — the Python script that produces every figure and number an entry cites, at *src/pinglab/notebook/&lt;slug&gt;.py*. Shells out to the oscilloscope and writes to *src/docs/public/figures/notebook/&lt;slug&gt;/*. Entry and runner share a slug and are always paired 1:1.
- **Ladder** — the feature-incremental set of models stepping from a vanilla SNN to full PING: *standard-snn → cuba → coba → ping*. Each rung adds one biophysical feature.
- **CUBA / COBA** — current-based vs conductance-based synapses. The axis the Δt-stability experiment decomposes.
- **PING** — Pyramidal-Interneuron Gamma. The E→I→E feedback loop that produces gamma oscillations (30–80 Hz).
- **Δt-stability** — the diagnostic of training at one integration timestep and evaluating at another. The shared lens across pinglab experiments.
- **Calibration** — tuning hyperparameters (weight scales, thresholds, input drive) so models on the ladder are comparable before an experiment runs.
- **Trainable surface** — what optimisation actually updates. Across the ladder it is input + output weights; recurrent weights in *ping* are frozen at init, not trained.
- **Promotion gate** — a manual step that moves content between layers: run output → frozen figure, notebook entry → paper section, ad-hoc preference → persistent memory. Code does not cross these gates.
- **Frozen figure** — a published PNG under *src/docs/public/figures/…* copied from *src/artifacts/…* with a sidecar JSON carrying the git SHA at freeze time and the run config.
