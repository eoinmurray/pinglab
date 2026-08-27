# Experiment writing

This directory contains the Typst sources for Pinglab's published notebook
entries. This is the canonical home for conventions governing `expXXX.typ`
files; expand this guide as those conventions settle.

## Location and naming

- Store each entry directly as `writings/expXXX.typ`, where `XXX` is the
  experiment's zero-padded three-digit identifier.
- Use the same identifier for related experiment code and artifact paths when
  they exist.

## Titles

For the experiment's overall title (`meta.title`), use a short, plain-English
phrase naming its main finding or controlled comparison. Prefer a specific
relationship ("Firing Rate Tracks Gamma Frequency") over vague topics or
promotional claims. State a finding only when supported by results; otherwise
name what is being tested. Aim for 5–10 words, retaining technical terms needed
for precision.

## Abstracts

Write a short abstract as a standalone summary of roughly 60–100 words.

- Begin with one plain-English sentence stating what the experiment learned or
  tests.
- Then state the controlled comparison or intervention, the principal
  measurement and result, and the narrow conclusion supported by the evidence.
- Define necessary technical terms or notation at first use.
- Use past tense for completed experiments and future tense for planned
  experiments. Never present an expected outcome as an observed result.
- Omit general background, implementation detail, citations, and implications
  that the experiment does not establish.
- Take reported values from retained experiment evidence; do not invent or
  hand-copy them when they can be interpolated from an artifact.

Example for a completed experiment (replace the bracketed placeholders with
artifact-backed values):

> Slower inhibition makes the network's rhythm slower. We tested this by
> varying $\tau_{\mathrm{GABA}}$, the time taken for inhibitory synaptic current
> to decay, while holding the network architecture, input, and analysis fixed.
> Across three seeds, gamma frequency decreased from **[A]** to **[B] Hz**,
> while firing rate and classification accuracy remained within **[C]** of
> baseline. Within the tested range, inhibitory timescale therefore controls
> rhythm frequency without materially changing task performance.

## Dependancies

Apply these rules to every experiment entry:

1. Name the section `Dependancies` exactly and place it immediately after the
   abstract.
2. Name every upstream experiment and the exact immutable Pingstore run used
   from it.
3. If there are no upstream experiment dependencies, say so explicitly.
4. It should be a very short section.

Example:

```typst
== Dependancies

This experiment depends on #link("/exp022/")[exp022]. It uses the immutable
Pingstore run `exp022-tr06-r2`, specifically the `TR-06` checkpoints and
configurations for seeds 42–44 under `files/state/`.
```

## Results

Apply these rules to every experiment entry:

1. Name the section `Results` exactly and place it before `Methods`.
2. Select three to five key plots that tell the experiment's scientific story.
   Do not include every diagnostic merely because it exists.
3. Give each plot a specific, descriptive name that states its principal
   comparison or finding.
4. Introduce each plot in continuous plain-English prose. Explain what question
   it addresses, what quantities it shows, and what pattern was expected before
   interpreting the experimental data.
5. When it helps the reader understand the prediction, include an optional SVG
   theory diagram before the data plot. Its caption must identify it as an
   expectation or mechanism, not experimental evidence.
6. Include a figure generated from the retained experimental data artifacts.
   Its caption must identify the measurement, conditions, aggregation, and
   uncertainty display needed to read it correctly.
7. After the data figure, explain in plain English what is visible and compare
   it directly with the expectation. State agreement, partial agreement,
   disagreement, or unresolved evidence plainly rather than forcing a positive
   result.
8. Write the explanation as flowing prose, not as a checklist of labelled
   sentences. Keep expected outcomes distinct from observed results, and take
   all reported values from retained evidence.
9. Results sections should be numbered but not "Plot 1, Plot 2" etc just 1. Title etc

Example for a completed experiment (replace the bracketed placeholders with
artifact-backed observations and values):

```typst
== Results

The experiment produced three key results. Together they test whether inhibitory
duration changes the network rhythm without simply changing its overall activity
or damaging task performance.

=== Plot 1 — Slower inhibition produces a slower rhythm

We first asked whether longer-lasting inhibition delays the next excitatory
volley. The plot compares inhibitory decay time $tau_"GABA"$ with the measured
gamma frequency. We expected frequency to fall as inhibition became slower,
because excitatory cells should remain suppressed for longer between volleys.

#figure(
  image("/.artifacts/expXXX/frequency-theory.svg", width: 70%),
  caption: [Expected mechanism: longer inhibition increases the interval
  between excitatory volleys. This schematic represents theory, not data.],
)

#figure(
  image("/.artifacts/expXXX/frequency-vs-decay.svg", width: 100%),
  caption: [Gamma frequency across inhibitory decay times. Points show means
  across seeds; error bars show ±1 standard deviation.],
)

The experimental data show [plain-English description of the observed trend,
including the important values]. This [matches / partly matches / contradicts]
the expected decrease because [plain-English comparison with the prediction].

=== Plot 2 — Firing rate remains stable

We next checked whether the frequency change was merely caused by the network
becoming more or less active. Excitatory and inhibitory firing rates are shown
across the same decay times. If inhibitory duration controls timing rather than
overall activity, we expected both rates to remain broadly stable.

#figure(
  image("/.artifacts/expXXX/rate-vs-decay.svg", width: 100%),
  caption: [Excitatory and inhibitory firing rates across conditions. Points
  show means across seeds; error bars show ±1 standard deviation.],
)

The measured rates [plain-English description]. Compared with the expectation,
this [supports / weakens] the timing interpretation because [reason].

=== Plot 3 — Task performance is preserved

Finally, we tested whether changing the rhythm affected useful computation. The
plot shows classification accuracy across inhibitory decay times. We expected
accuracy to remain near baseline wherever coherent rhythmic activity was
preserved.

#figure(
  image("/.artifacts/expXXX/accuracy-vs-decay.svg", width: 100%),
  caption: [Test accuracy across inhibitory decay times. Points show means
  across seeds; error bars show ±1 standard deviation.],
)

Accuracy [plain-English description of the observed result]. This
[matches / partly matches / contradicts] our expectation because [reason].
Taken together, the three plots show [narrow artifact-backed conclusion].
```

## Methods

Apply these rules to every experiment entry:

1. Name the section `Methods` exactly.
2. Place `Methods` after `Results`.
3. Organize it into no more than ten top-level numbered steps. Avoid nested
   procedural lists when a single step can state the action clearly.
4. Include only the few key numbered equations needed to represent what was
   executed or measured.
5. Put detailed mathematical derivations of key terms in appendices and refer
   to them from `Methods`.
6. Define every key scientific term, symbol, and quantity at first use. After
   each equation, define all of its terms and give units where applicable.
7. Bring the reader through the experiment in causal order: starting state,
   intervention, execution, measurement, aggregation, and evidence boundary.
   Use a short opening orientation and explain why each step leads to the next.

`Methods` records what was actually executed. Headline findings and their
interpretation belong in `Results`; mathematical working that is not necessary
to understand the procedure belongs in an appendix.

Example:

```typst
== Results

Increasing inhibitory decay reduced gamma frequency while preserving rhythmic
activity across the tested range. [Artifact-backed results and figures.]

== Methods

The experiment asks whether the duration of inhibition controls the network's
rhythm. We answer this by changing inhibition alone, simulating the resulting
activity, and measuring its dominant frequency.

+ *Start from matched networks.* We used three independently trained PING
  networks. A PING network contains excitatory cells that recruit inhibitory
  cells, which then suppress the excitatory population.

+ *Change the inhibitory timescale.* We varied
  $tau_"GABA" in {4.5, 6, 9, 12, 18, 27}$ ms while holding architecture,
  input, training, and analysis fixed. Here $tau_"GABA"$ is the time taken for
  inhibitory synaptic conductance to decay.

+ *Generate the input.* Pixel $i$ emitted a spike at timestep $t$ according to

  $ S_i(t) tilde "Bernoulli"(x_i r_"max" Delta t / 1000). quad "(1)" $

  Here $S_i(t)$ is the input spike, $x_i$ is normalized pixel intensity,
  $r_"max"$ is the maximum encoding rate in hertz, and $Delta t$ is the
  simulation timestep in milliseconds.

+ *Measure the rhythm.* We binned excitatory spikes into the population count
  $n_E(t)$ and defined gamma frequency as

  $ f_gamma = arg max_(f in [30, 80]) P_E(f). quad "(2)" $

  Here $P_E(f)$ is the Welch power spectrum of $n_E(t)$ at frequency $f$, and
  $f_gamma$ is its largest gamma-band peak.

+ *Aggregate independent networks.* For condition $c$,

  $ macron(f)_(gamma,c) = 1/S sum_(s=1)^S f_(gamma,c,s). quad "(3)" $

  Here $S=3$ is the number of trained seeds and $s$ indexes one independently
  trained network.

+ *State the evidence boundary.* Quantitative results use all seeds and
  registered conditions. Single-trial rasters are illustrative only.

== Appendix A: Derivation of the gamma-frequency estimator

Starting from $n_E(t)$, derive the discrete Fourier transform, Welch spectrum,
window normalization, frequency resolution, and parabolic peak interpolation
used in Equation 2.
```
