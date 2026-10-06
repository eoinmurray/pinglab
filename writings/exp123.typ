#import "templates/article-layout.typ": journal-article
#import "templates/result-card.typ": result-card, with-result-sections, result-figure-ref
#import "templates/abstract.typ": journal-abstract
#import "templates/methods.typ": journal-methods, method-card
#import "templates/dataset.typ": data-file, inputs-ready, pending-report, input-assets
#import "/.demolab/lib.typ": data-image, data-json
#let data-file = data-file.with(article: "exp123")
#let meta = (
  title: "Output Evidence Against Time and Gamma Cycles",
  created_at: "2026-10-06",
  updated_at: "2026-10-06",
  description: "Paired inhibitory-decay probes of output evidence, matched-coordinate accuracy and accuracy–deadline–spike-cost frontiers across input encoding rates.",
  collection: "neuromodulation",
  tags: ("data", "v36.0.0"),
)
#let inputs = ("exp123",)
#let preview-figures = (
  (path: "exp123/confidence_time.png", label: "Physical-time trajectories"),
  (path: "exp123/confidence_cycles.png", label: "Cycle-position trajectories"),
  (path: "exp123/sustained_decisions.png", label: "Sustained decisions"),
  (path: "exp123/accuracy_comparison.png", label: "Matched-coordinate accuracy"),
  (path: "exp123/deadline_tradeoffs.png", label: "Accuracy and spike cost"),
  (path: "exp123/pareto_frontiers.png", label: "Pareto candidates and frequency"),
)
#let render-report() = {
  let r = data-json(data-file("exp123/numbers.json"))
  [
  #journal-abstract(body: [
    We traced cumulative output-spike softmax evidence under inhibitory-decay
    modulation in a frozen PING network. Longer decay delayed high scores in
    physical time. Cycle coordinates reduced timing separation for some
    examples, while others retained differences in progression per cycle.
    These visual comparisons do not show a universal collapse. Scores often
    saturated and could be confidently wrong, limiting their interpretation
    as calibrated confidence.
    We also compared matched-coordinate accuracy and accuracy–deadline–spike-cost
    frontiers across encoding rates, using network spikes as an activity-cost proxy.
  ])
  == Results
  #with-result-sections[
    #result-card[
      === Evidence across physical time
      #figure(
        data-image(data-file("exp123/confidence_time.png"), width: 100%,
          alt: "Ten panels show mean true-class cumulative-spike softmax scores versus elapsed time under three inhibitory-decay settings."),
        caption: [A–J: mean true-class softmax score from cumulative output spikes for
          the first ten MNIST test images, with 100 paired encodings each.
          Black/red/cyan: 3/6/12 ms inhibitory decay. Full 200 ms, no burn-in,
          one frozen network. Means have no uncertainty bands; softmax scores
          are uncalibrated and can saturate despite incorrect predictions.],
      ) <fig:exp123-time>
    ]
    #result-card[
      === Evidence across gamma cycles
      #figure(
        data-image(data-file("exp123/confidence_cycles.png"), width: 100%,
          alt: "The same ten images and modulation curves are plotted against inhibitory-volley-defined cycle position."),
        caption: [A–J: the same scores and colours as
          #result-figure-ref(<fig:exp123-time>), expressed in cycle position.
          The first I volley is zero. Curves average all 100 paired draws
          over their shared coverage, without extrapolation. Initial latency
          is excluded; clustering and score saturation can mimic alignment.],
      ) <fig:exp123-cycles>
    ]
    #result-card[
      === Sustained correct decisions
      #figure(
        data-image(data-file("exp123/sustained_decisions.png"), width: 100%,
          alt: "Twenty small plots pair decision time and cycle position for each of ten images; black medians and red interquartile ranges show trends across decay settings."),
        caption: [A: sustained correct-decision time. B: corresponding cycle position.
          Columns identify images 00–09. Black: median; red: interquartile
          range among successful draws, not a confidence interval. Failed
          trials have no point; cycle coverage can exclude additional trials.
          Image 08 has one late success and a larger time scale. Decisions
          are defined retrospectively through 200 ms.],
      ) <fig:exp123-decisions>
    ]
    #result-card[
      === Accuracy at matched coordinates
      #figure(
        data-image(data-file("exp123/accuracy_comparison.png"), width: 100%,
          alt: "Three accuracy curves compare inhibitory decay at matched elapsed time, cycle position and cumulative input count; three corresponding coverage plots show available paired trials."),
        caption: [A–C: accuracy against time, cycle position and cumulative input
          spikes. D–F: paired-trial coverage. Sample: 500 images, 50 per digit,
          three paired encodings; decay 3/6/12 ms in black/red/cyan. Shading:
          image-bootstrap 95% intervals. Dashed lines mark 80% coverage;
          accuracy is omitted below it. Later coordinates select subsets.
          Equal spike counts do not guarantee equal information.],
      ) <fig:exp123-accuracy>
    ]
    #result-card[
      === Accuracy and spike cost
      #figure(
        data-image(data-file("exp123/deadline_tradeoffs.png"), width: 100%,
          alt: "Nine panels show accuracy versus deadline, network spikes versus deadline and accuracy versus network spikes at three input encoding rates."),
        caption: [A/D/G: accuracy versus deadline. B/E/H: mean network spikes versus
          deadline. C/F/I: accuracy versus spike cost. Rows: 12.5/25/50 Hz
          encoding; colours: 3/6/12 ms decay. Points mark five deadlines,
          connected within each setting. Sample: 500 images, three paired
          encodings. Shading in A/D/G: image-bootstrap 95% intervals.
          Cost includes E, I and output spikes, excluding input.
          Spike cost is not physical energy; encoding rate also changes drive.],
      ) <fig:exp123-tradeoffs>
    ]
    #result-card[
      === Pareto candidates and frequency
      #figure(
        data-image(data-file("exp123/pareto_frontiers.png"), width: 100%,
          alt: "Three connected bubble plots show empirical Pareto candidates at different encoding rates, with bubble area representing spike cost; three spectral-frequency plots verify decay modulation."),
        caption: [A–C: Pareto comparisons at 12.5/25/50 Hz encoding. Position shows
          deadline and accuracy; bubble area shows mean network spikes on a
          common scale. Colours identify decay; traces connect deadlines.
          Outlined bubbles are empirically nondominated; faint bubbles are
          dominated. D–F: peaks of mean E-population PSDs at the same rates.
          Peaks can reflect harmonics or heterogeneous rhythms. Sample and cost definition:
          #result-figure-ref(<fig:exp123-tradeoffs>). Frontier membership is
          sample-dependent.],
      ) <fig:exp123-pareto>
    ]
  ]
  #journal-methods(body: (
    method-card([Network], [
      We reused frozen final-epoch MNIST weights trained at 6 ms decay:
      1,024 E and 256 I neurons, fixed leak and capacitance, graph execution.
      Training used mean-voltage evidence; this experiment scored spikes
      (#result-figure-ref(<fig:exp123-time>, panel: "A–J");
      #result-figure-ref(<fig:exp123-tradeoffs>, panel: "A–I")).
    ]),
    method-card([Paired inputs], [
      We presented the first ten test images with 100 paired encodings each
      at 25 Hz; decay 3/6/12 ms, 200 ms duration, 0.1 ms steps, no burn-in
      (#result-figure-ref(<fig:exp123-time>, panel: "A–J");
      #result-figure-ref(<fig:exp123-cycles>, panel: "A–J")).
    ]),
    method-card([Output scores], [
      We applied softmax at temperature one to cumulative output counts,
      then averaged true-class scores across draws. Zero counts give 0.1;
      predictions and maximum scores identify errors and saturation
      (#result-figure-ref(<fig:exp123-time>, panel: "A–J")).
    ]),
    method-card([Cycle coordinate], [
      We smoothed I counts with a 1 ms Gaussian; peaks required ≥2 ms
      spacing and prominence ≥max(1 count, 20% of maximum). Accepted
      sequences required a complete interior cycle and mean I participation ≥50%
      (#result-figure-ref(<fig:exp123-cycles>, panel: "A–J");
      #result-figure-ref(<fig:exp123-accuracy>, panel: "B/E")).
    ]),
    method-card([Compare trajectories], [
      We set the first I volley to zero and excluded initial latency.
      Scores were interpolated between volleys on a shared 0.05-cycle grid,
      without extrapolation; all 100 draws remained in each image's mean
      (#result-figure-ref(<fig:exp123-cycles>, panel: "A–J")).
    ]),
    method-card([Sustained decisions], [
      We found the first unique correct count winner persisting through
      200 ms; ties interrupted continuity. Failed trials were omitted.
      Medians and quartiles used successful draws; cycle summaries additionally
      required decisions within accepted volley coverage, without extrapolation
      (#result-figure-ref(<fig:exp123-decisions>, panel: "A/B")).
    ]),
    method-card([Accuracy sample], [
      We sampled 50 test images per digit without replacement, selection
      seed 123, with three paired encodings per image and unchanged network
      and trial settings (#result-figure-ref(<fig:exp123-accuracy>, panel: "A–F")).
    ]),
    method-card([Matched predictions], [
      We measured cumulative-count argmax accuracy at 1 ms, 0.25-cycle and
      25-input-spike increments; lowest-index ties, including errors.
      Cycle queries used the last completed update; input budgets used the
      first crossing update, allowing simultaneous-spike overshoot
      (#result-figure-ref(<fig:exp123-accuracy>, panel: "A–C")).
    ]),
    method-card([Coordinate coverage], [
      We required each image/draw coordinate to exist in all decay settings,
      without extrapolation. Accuracy was hidden below 80% paired-trial
      coverage; axes ended one step beyond the last coordinate retaining 50%
      (#result-figure-ref(<fig:exp123-accuracy>, panel: "D–F")).
    ]),
    method-card([Image uncertainty], [
      We averaged eligible draws within images, then images equally.
      Percentile 95% intervals used 1,000 paired image resamples stratified
      by digit, seed 12345; they exclude training variation
      (#result-figure-ref(<fig:exp123-accuracy>, panel: "A–C");
      #result-figure-ref(<fig:exp123-tradeoffs>, panel: "A/D/G")).
    ]),
    method-card([Encoding-rate probe], [
      We added 12.5 and 50 Hz to the reused 25 Hz sample. Shared uniforms
      produced nested input spikes, verified for every image/draw.
      Rate changes sampling and drive, rather than isolating SNR
      (#result-figure-ref(<fig:exp123-tradeoffs>, panel: "A–I")).
    ]),
    method-card([Deadline cost], [
      We scored all trials at 20/40/80/120/200 ms, including errors, and
      counted E + I + output spikes through each deadline, excluding input
      (#result-figure-ref(<fig:exp123-tradeoffs>, panel: "B/E/H");
      #result-figure-ref(<fig:exp123-pareto>, panel: "A–C")).
    ]),
    method-card([Pareto candidates], [
      We compared 15 decay/deadline candidates per encoding rate using
      unrounded correct and spike totals. Dominance required accuracy no
      lower, deadline/cost no higher, and one strict improvement.
      Membership frequencies used 1,000 paired image-bootstrap resamples
      (#result-figure-ref(<fig:exp123-pareto>, panel: "A–C")).
    ]),
    method-card([Spectral frequency], [
      We measured full-200-ms E-count Welch PSDs at native sampling:
      Hann window, centred, no detrending. We selected the peak of the mean of 1,500
      spectra within 5–150 Hz by spectrum-neighbour parabolic
      interpolation, without harmonic rejection
      (#result-figure-ref(<fig:exp123-pareto>, panel: "D–F")).
    ]),
  ))
  == Appendix: fixed forward parameters
  #table(
    columns: (2fr, 1fr, 1fr),
    table.header([Parameter], [E], [I]),
    [Population size], [1,024], [256],
    [Capacitance (nF)], [1.0], [0.5],
    [Leak conductance (µS)], [0.05], [0.10],
    [Rest and reset (mV)], [−65], [−65],
    [Threshold (mV)], [−50], [−50],
    [Refractory hold (ms)], [1.2], [0.6],
  )
  AMPA decay is 2 ms; reciprocal projections have a one-step delay. The
  input has 784 channels and the output ten spiking units, with a 2 ms
  leaky input filter and 2 ms output leak, threshold one and subtractive
  reset. We directly reused final-epoch weights after fifty training epochs,
  from training seed 42; no weights were selected or fitted here. Every
  presentation resets hidden neurons to rest, conductances to zero and
  output state to zero. Encoding generators use the image index times one
  thousand plus the draw identifier, making distinct streams across images.
  ]
}
#let report-body = if inputs-ready(data-file, inputs) { render-report() } else {
  pending-report(data-file, inputs,
    [Do cumulative output-spike softmax trajectories align more closely in time or in inhibitory-volley cycle position?],
    preview-figures, json-inputs: ())
}
#let meta = meta + (assets: input-assets("exp123", inputs))
#let body = journal-article("exp123", inputs, report-body)
