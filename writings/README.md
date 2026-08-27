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

## Inputs and outputs

Apply these rules to every experiment entry:

1. Name the section `Inputs and outputs` exactly and place it immediately after
   the abstract, before `Design Scope`.
2. Use two short paragraphs labelled `Inputs` and `Outputs`. Name every upstream
   experiment used and identify the reusable outputs this experiment provides,
   such as checkpoints, configurations, datasets, or analysis results. Explain
   briefly what each input is used for and what each output is suitable for.
3. For retained inputs and outputs, give the exact immutable Pingstore run IDs
   and paths within those runs. Keep `run.json` authoritative for provenance;
   this section is a reader's guide.
4. State explicitly when there are no upstream experiment inputs or no reusable
   outputs.
5. Label planned outputs as planned, distinguish them from outputs actually
   produced, and never invent a run reference.
6. Keep the section very short. Do not inventory every plot or repeat the
   conclusions.

Illustrative example (replace the bracketed run IDs and example paths with
retained evidence):

```typst
== Inputs and outputs

*Inputs:* Uses checkpoints and configurations for seeds 42–44 from
#link("/exp022/")[exp022] as starting states for inhibition sweeps, retained in
immutable Pingstore run `[upstream-run-id]` under `export/state/`.

*Outputs:* Provides measured frequency responses for subsequent comparisons,
retained in immutable Pingstore run `[this-run-id]` at
`export/frequency-response.csv`.
```

## Design Scope

Apply these rules to every experiment entry:

1. Name the section `Design Scope` and place it after `Inputs and outputs`.
2. Describe the experiment's parameter space in plain English: what system is
   studied, what changes, the tested values or ranges, and what remains fixed.
   Explain what each varied parameter means physically or operationally.
3. State the relevant starting conditions, comparison groups, and sampling
   scale, including independent seeds or repetitions where applicable.
4. Explain why these comparisons address the experiment's question and identify
   the main boundaries: what the experiment does not vary or test.
5. Keep this section short. Prefer ordinary prose over notation or configuration
   dumps; leave execution details and equations to `Methods`. Distinguish planned
   settings from those actually tested.

## Prior art

Apply these rules to every experiment entry:

1. Name the section `Prior art` and place it after `Design Scope`, before
   `Results`.
2. Give a short, plain-English account of relevant work by others: what they
   studied, how they approached it, and what they established.
3. Explain how this experiment relates to that work—such as reproducing,
   adapting, extending, or testing a limitation—without implying novelty merely
   because the implementation differs.
4. Cite the sources beside the claims they support. Prefer original research for
   specific findings; use reviews for broader context.
5. Include only background needed to understand this experiment. If no directly
   relevant work has been identified, state that limitation rather than claiming
   none exists.

## Results

Apply these rules to every experiment entry:

1. Name the section `Results` exactly and place it before `Methods`.
2. Select only the key plots needed to tell the experiment's scientific story
   and support its conclusions; there is no fixed plot count. A single compound
   figure may suffice. Do not include every diagnostic merely because it exists.
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

## References

Apply these rules to every experiment entry:

1. Place a numbered `References` section at the bottom of the entry, after any
   appendices.
2. Use `#cite(...)` for inline citations and `#reference-list(...)` for the
   reference list.
3. List sources in order of first citation, with authors, title, publication
   venue, year, and a DOI or stable URL where available.
4. Reuse the same number for repeated citations. Keep citation numbers
   synchronized with list positions.
5. Include only cited sources and verify that each supports its associated
   claim. Keep literature references distinct from upstream experiment and run
   provenance.
