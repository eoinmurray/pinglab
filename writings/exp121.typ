#import "templates/article-layout.typ": journal-article
#import "templates/abstract.typ": journal-abstract
#import "templates/methods.typ": journal-methods, method-card
#import "templates/result-card.typ": journal-result-card, with-result-sections
#import "templates/dataset.typ": data-file, input-assets, inputs-ready, pending-report
#import "/.demolab/lib.typ": data-image
#let data-file = data-file.with(article: "exp121")
#let meta = (
  tags: ("data", "v36.0.0"),
  title: "Decision Timing Across Input and Internal Parameters",
  created_at: "2026-10-01T00:00:00Z",
  updated_at: "2026-10-01",
  description: "Seven matched input and internal parameter sweeps measure decision timing and network rhythm in one trained network.",
  collection: "neuromodulation",
)
#let inputs = ("exp121",)
#let preview-figures = (
  (path: "exp121/figure-1.svg", label: "Input rate"),
  (path: "exp121/figure-2.svg", label: "GABA decay"),
  (path: "exp121/figure-3.svg", label: "E/I capacitance"),
  (path: "exp121/figure-4.svg", label: "E/I leak conductance"),
  (path: "exp121/figure-5.svg", label: "AMPA decay"),
  (path: "exp121/figure-6.svg", label: "I→E inhibitory strength"),
  (path: "exp121/figure-7.svg", label: "E spike threshold"),
)
#let render-report(data-file) = [
  #journal-abstract(body: [
    We tested how quickly a trained network reached a correct decision that
    stayed correct. We varied seven control parameters: *input spike rate,
    GABA decay time, membrane capacitance, leak conductance, AMPA decay time,
    inhibitory strength onto excitatory neurons, and excitatory spike threshold*.
    Higher input rates clearly sped up decisions. Internal changes altered the
    rhythm but gave much smaller or inconsistent timing gains. We used one
    trained network without retraining; only the inhibitory-strength sweep
    changed learned connection strengths.
  ])
  == Results
  #with-result-sections[
    #journal-result-card(
      title: "Faster input, earlier decisions",
      visual: [#figure(
        data-image(data-file("exp121/figure-1.svg"), width: 100%,
          alt: "Four bar histograms of decision times at four input rates, with network frequency and burst interval CV plotted against input rate in two companion panels."),
        caption: [*Input rate and decision timing.* (A–D) Bar histograms in 20 ms bins with shared axes,
          ordered by increasing input rate. Each successful trial
          contributes once; wrong or tied final decisions are excluded.
          Annotations report successes out of 100. (E–F) Median and interquartile
          range across trials with at least three detected inhibitory volleys:
          96, 99, 100 and 100 at 2.5, 5, 10 and 20 Hz input. Colours identify
          the same conditions across panels. All measurements use 400 ms;
          stable correctness is retrospective through that horizon.],
        kind: image, supplement: [Figure],
      ) <fig:input-rate>],
    )
    #journal-result-card(
      title: "GABA decay and decision timing",
      visual: [#figure(
        data-image(data-file("exp121/figure-2.svg"), width: 100%,
          alt: "Four bar histograms of stable-correct decision times, network frequency and interval CV across GABA decay conditions."),
        caption: [*GABA decay sweep.* GABA decay times: 3, 4.8, 5.4 and 6 ms. AMPA remained 2 ms.
          Input remained 5 Hz. (A–D) Bar histograms in 20 ms bins with shared axes, ordered by increasing control value;
          final errors/ties have no valid time. Annotations report successes out of 100.
          (E–F) Median and interquartile range across trials with at least three
          detected I volleys (99 per condition). Colours identify matched conditions across panels.
          The same 100 images and 400 ms horizon were used; the baseline was
          reused from the input-rate study.],
        kind: image, supplement: [Figure],
      ) <fig:gaba>],
    )
    #journal-result-card(
      title: "Capacitance and decision timing",
      visual: [#figure(
        data-image(data-file("exp121/figure-3.svg"), width: 100%,
          alt: "Four bar histograms of stable-correct decision times, network frequency and interval CV across E/I capacitance conditions."),
        caption: [*E/I capacitance sweep.* Multipliers 0.5, 0.8, 0.9 and 1 applied jointly to baseline E/I capacitances of 1/0.5 nF. Actual E/I pairs: 0.5/0.25, 0.8/0.4, 0.9/0.45 and 1/0.5 nF.
          Input remained 5 Hz. (A–D) Bar histograms in 20 ms bins with shared axes, ordered by increasing control value;
          final errors/ties have no valid time. Annotations report successes out of 100.
          (E–F) Median and interquartile range across trials with at least three
          detected I volleys (100, 100, 100 and 99 in increasing multiplier order). Colours identify matched conditions across panels.
          The same 100 images and 400 ms horizon were used; the baseline was
          reused from the input-rate study.],
        kind: image, supplement: [Figure],
      ) <fig:capacitance>],
    )
    #journal-result-card(
      title: "Leak and decision timing",
      visual: [#figure(
        data-image(data-file("exp121/figure-4.svg"), width: 100%,
          alt: "Four bar histograms of stable-correct decision times, network frequency and interval CV across E/I leak conductance conditions."),
        caption: [*E/I leak conductance sweep.* Multipliers 0.5, 0.8, 0.9 and 1 applied jointly to baseline E/I leaks of 0.05/0.10 µS. Actual E/I pairs: 0.025/0.05, 0.04/0.08, 0.045/0.09 and 0.05/0.10 µS.
          Input remained 5 Hz. (A–D) Bar histograms in 20 ms bins with shared axes, ordered by increasing control value;
          final errors/ties have no valid time. Annotations report successes out of 100.
          (E–F) Median and interquartile range across trials with at least three
          detected I volleys (100, 100, 99 and 99 in increasing multiplier order). Colours identify matched conditions across panels.
          The same 100 images and 400 ms horizon were used; the baseline was
          reused from the input-rate study.],
        kind: image, supplement: [Figure],
      ) <fig:leak>],
    )
    #journal-result-card(
      title: "AMPA decay and decision timing",
      visual: [#figure(
        data-image(data-file("exp121/figure-5.svg"), width: 100%,
          alt: "Four bar histograms of stable-correct decision times, with network frequency and interval CV across AMPA decay conditions."),
        caption: [*AMPA decay sweep.* Times: 1, 1.6, 1.8 and 2 ms; GABA remained 6 ms.
          (A–D) Decision-time histograms in 20 ms bins, with shared axes and
          conditions in increasing order. Final errors/ties are excluded;
          annotations report successes out of 100. (E–F) Median and interquartile
          range for frequency and interval CV, using 60, 98, 99 and 99 valid
          trials respectively. Shorter AMPA decay slowed the measured rhythm.
          Colours match conditions. All runs used the same images, 5 Hz inputs
          and 400 ms horizon; the baseline was reused.],
        kind: image, supplement: [Figure],
      ) <fig:ampa>],
    )
    #journal-result-card(
      title: "Inhibitory strength and timing",
      visual: [#figure(
        data-image(data-file("exp121/figure-6.svg"), width: 100%,
          alt: "Three bar histograms of stable-correct decision times, with network frequency and interval CV across I to E inhibitory strength conditions."),
        caption: [*I→E inhibitory strength sweep.* Multipliers: 0.8, 1 and 1.2.
          (A–C) Decision-time histograms in 20 ms bins, with shared axes and
          conditions in increasing order. Final errors/ties are excluded;
          annotations report successes out of 100. (D–E) Median and interquartile
          range for frequency and interval CV, using 99 valid trials per condition.
          At 0.8×, the median paired decision was 0.3 ms earlier among joint
          successes, but final accuracy fell from 86 to 84 out of 100.
          Colours match conditions. All runs used the same images, 5 Hz inputs
          and 400 ms horizon; the baseline was reused.],
        kind: image, supplement: [Figure],
      ) <fig:inhibition>],
    )
    #journal-result-card(
      title: "Spike threshold and decision timing",
      visual: [#figure(
        data-image(data-file("exp121/figure-7.svg"), width: 100%,
          alt: "Three bar histograms of stable-correct decision times, with network frequency and interval CV across excitatory spike thresholds."),
        caption: [*Excitatory spike threshold sweep.* Thresholds: −52, −50 and −48 mV;
          the inhibitory threshold remained −50 mV. (A–C) Decision-time
          histograms in 20 ms bins, with shared axes and conditions in increasing
          order. Final errors/ties are excluded; annotations report successes
          out of 100. (D–E) Median and interquartile range for frequency and
          interval CV, using 100, 99 and 99 valid trials respectively. At −52 mV,
          the median paired decision was 0.8 ms earlier among joint successes,
          with 87/100 correct versus 86/100 at baseline. Colours match conditions.
          All runs used the same images, 5 Hz inputs and 400 ms horizon;
          the baseline was reused.],
        kind: image, supplement: [Figure],
      ) <fig:threshold>],
    )
  ]
  #journal-methods(body: (
    method-card([Images], [
      We used the same 100 MNIST test images throughout: 10 of each digit.
    ]),
    method-card([Trained network], [
      We reused one trained network with 1,024 excitatory neurons, 256
      inhibitory neurons and 10 outputs. No retraining was performed. Only the
      inhibitory-strength sweep changed its learned weights.
    ]),
    method-card([Input spike rate], [
      We tested maximum pixel rates of 2.5, 5, 10 and 20 Hz. Brighter pixels
      produced more spikes. Shared random draws made the lower-rate spikes a
      subset of the higher-rate spikes.
    ]),
    method-card([GABA decay time], [
      We tested 6, 5.4, 4.8 and 3 ms on both inhibitory pathways. Shorter
      decay made inhibition fade sooner and reduced its total effect per
      spike. AMPA decay stayed at 2 ms.
    ]),
    method-card([Membrane capacitance], [
      We multiplied excitatory and inhibitory capacitances together by 1, 0.9,
      0.8 and 0.5. Their baseline values were 1 and 0.5 nF.
    ]),
    method-card([Leak conductance], [
      We applied those same multipliers to excitatory and inhibitory leak
      conductances. Their baseline values were 0.05 and 0.10 µS.
    ]),
    method-card([AMPA decay time], [
      We tested 2, 1.8, 1.6 and 1 ms on input-to-excitatory and both recurrent
      excitatory pathways. Peak weights stayed fixed, so shorter decay reduced
      excitation per spike. GABA stayed at 6 ms; the readout was unchanged.
    ]),
    method-card([Inhibitory strength], [
      We multiplied inhibitory-to-excitatory weights by 0.8, 1 and 1.2.
      All other weights and decay times stayed at baseline.
    ]),
    method-card([Excitatory spike threshold], [
      We tested −52, −50 and −48 mV in excitatory neurons only.
      The inhibitory threshold stayed at −50 mV.
    ]),
    method-card([Matched runs], [
      Each internal parameter was varied separately, with identical 5 Hz input
      spikes. Every presentation lasted 400 ms, using 0.1 ms steps and freshly
      reset network states.
    ]),
    method-card([Decision time], [
      We counted spikes from each output. A decision became stably correct
      when the correct digit took the lead and never lost or tied it before
      the run ended. Runs ending wrong or tied had no stable-correct time.
    ]),
    method-card([Network frequency], [
      We detected bursts in the inhibitory population and divided one second
      by the average time between bursts. At least three bursts were required.
    ]),
    method-card([Burst variability], [
      We divided the standard deviation of burst intervals by their mean.
      Lower values mean more regular timing.
    ]),
    method-card([Plots], [
      Histograms count decision times in 20 ms bins. Frequency and variability
      panels show medians and the middle half of valid trial values.
    ]),
  ))
]
#let report-body = if inputs-ready(data-file, inputs) { render-report(data-file) } else {
  pending-report(data-file, inputs,
    [How do input and internal parameters affect stable-correct decisions?], preview-figures)
}
#let meta = meta + (assets: input-assets("exp121", inputs))
#let body = journal-article("exp121", inputs, report-body)
