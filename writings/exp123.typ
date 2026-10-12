#import "templates/article-layout.typ": journal-article
#import "templates/result-card.typ": result-card, result-figure-ref, with-result-sections
#import "templates/methods.typ": journal-methods, method-card
#import "templates/dataset.typ": data-file, input-assets, inputs-ready, pending-report
#import "/.demolab/lib.typ": data-image, data-json
#let data-file = data-file.with(article: "exp123")
#let meta = (
  title: "Output Evidence Against Time and Gamma Cycles",
  created_at: "2026-10-06",
  updated_at: "2026-10-09",
  description: "Paired inhibitory-decay probes of output evidence, matched-coordinate accuracy and accuracy–deadline–spike-cost frontiers across input encoding rates.",
  collection: "neuromodulation",
  tags: ("data", "v36.0.0"),
)
#let inputs = ("exp123",)
#let preview-figures = (
  (path: "exp123/confidence_time.png", label: "Physical-time trajectories"),
  (path: "exp123/confidence_cycles.png", label: "Completed-volley trajectories"),
  (path: "exp123/sustained_decisions.png", label: "Sustained decisions"),
  (path: "exp123/accuracy_comparison.png", label: "Matched-coordinate accuracy"),
  (path: "exp123/deadline_tradeoffs.png", label: "Accuracy and spike cost"),
  (path: "exp123/pareto_frontiers.png", label: "Pareto candidates and frequency"),
)

#let render-report() = {
  let r = data-json(data-file("exp123/numbers.json"))
  [

    === Background

    We imported the trained PING network from
    #link("/exp022/")[exp022] — #link("/exp022/")[_Training Runs_]
    and kept its weights fixed. The #link(<sec:exp123-parameters>)[parameter table]
    lists the network and simulation settings.

    === Evidence across time

    #figure(
      data-image(
        data-file("exp123/confidence_time.png"),
        width: 100%,
        alt: "Nine panels in a three-by-three grid show mean true-class cumulative-spike softmax scores versus elapsed time under three inhibitory-decay settings.",
      ),
      caption: [A–I each show one of the first nine MNIST test images.
        We converted each image into 100 random spike trains, with brighter
        pixels having higher average spike rates (25 spikes per second for
        a fully bright pixel).
        Each train was presented for 200 ms to the same trained network with
        fixed weights, using three inhibitory decay times: 3 ms (black),
        6 ms (red) and 12 ms (cyan). The x-axis shows elapsed time. The y-axis
        shows the score assigned to the correct digit, calculated by applying
        softmax to accumulated output spike counts. Each curve averages the
        100 trials; no uncertainty bands are shown. This score is not a
        calibrated probability of being correct.],
    ) <fig:exp123-time>

    We encoded each of the first nine MNIST test images into 100 random
    spike trains and reused each train at $tau_"GABA" = 3, 6, 12$ ms.
    Each presentation lasted 200 ms, with 0.1 ms simulation steps. For each
    trial, the logit for digit $c$ was its accumulated output spike count:

    $ z_c(t) = sum_(k=1)^(floor(t / Delta t_"sim")) s_c[k]. $

    Here, $t$ is elapsed time, $Delta t_"sim"$ is the timestep, $k$ indexes
    simulation steps, and $s_c[k]$ is one when output neuron $c$ spikes,
    otherwise zero. We converted these logits into scores using softmax:

    $ p_c(t) = exp(z_c(t)) / (sum_(j=0)^9 exp(z_j(t))). $

    Here, $p_c(t)$ is the score for digit $c$, and $j$ ranges over all ten
    digits. #result-figure-ref(<fig:exp123-time>, panel: "A–I") shows the
    correct digit's score averaged *after softmax* across 100 trials,
    separately for each decay setting. These scores are uncalibrated.

    === Evidence per completed volley
    #figure(
      data-image(
        data-file("exp123/confidence_cycles.png"),
        width: 100%,
        alt: "The same nine test images as Figure 1 show mean cumulative softmax scores sampled at inhibitory troughs against completed volley count.",
      ),
      caption: [A–I: the same nine MNIST test images, using the score definition
        and colours of #result-figure-ref(<fig:exp123-time>). Each point samples the
        cumulative score at the inhibitory trough after the numbered volley,
        then averages across 100 trials. The trough is the minimum of smoothed
        I activity between successive accepted peaks; the final peak is excluded
        because it has no following boundary.
        Volley zero shows the initial score of 0.1 at time zero, not a detected volley.
        Only counts available in every trial and setting are shown; lines
        connect integer counts without representing elapsed time.
        Clustering and score saturation can mimic alignment.],
    ) <fig:exp123-cycles>

    We detected inhibitory peaks after 1 ms Gaussian smoothing, requiring
    ≥2 ms spacing and prominence ≥max(1 spike, 20% of the maximum).
    Accepted sequences required a complete interior cycle and mean
    inhibitory participation ≥50%.

    We defined each volley's completion as the minimum of smoothed
    inhibitory activity between its peak and the next. At that boundary,
    we sampled the correct digit's softmax score from all output spikes
    accumulated since onset.

    #result-figure-ref(<fig:exp123-cycles>, panel: "A–I") averages these
    samples across 100 trials for each decay setting, using completed-volley
    counts available in every trial and setting. The final unbounded volley
    is excluded; count zero shows the initial score of 0.1.

    === Sustained correct decisions
    #figure(
      data-image(
        data-file("exp123/sustained_decisions.png"),
        width: 100%,
        alt: "The same nine images as Figures 1 and 2 show median stable correct decisions versus GABA decay: time in row A and completed volley count in row B, with points connected across decay settings.",
      ),
      caption: [A: time (ms). B: completed volley count, using the inhibitory
        trough boundaries of #result-figure-ref(<fig:exp123-cycles>).
        B uses the first completion at which the decision is already stable.
        Both rows show GABA decay on y for images 00–08. Each point is the
        median among trials whose correct digit becomes the unique winner
        and stays so through 200 ms. Lines connect the three decay settings.
        Failed trials are excluded; B also
        requires completion-boundary coverage within Figure 2's shared count
        range. Trial counts are integers; medians can be fractional.],
    ) <fig:exp123-decisions>

    For the same nine images and 100 trials per decay setting, we found the
    earliest update at which the correct digit had the largest cumulative
    output spike count and remained the unique winner through 200 ms.
    Ties interrupt stability; trials without a stable correct decision were excluded.

    We used the first completed-volley trough boundary from step 3 at or
    after each stable decision. This is the first sampled point in Figure 2
    at which the decision is already stable, numbered from one.
    Counts required accepted volley sequences, a decision no later than the
    final completion boundary, and Figure 2's shared count range.
    #result-figure-ref(<fig:exp123-decisions>, panel: "A") plots median decision
    time against inhibitory decay; #result-figure-ref(<fig:exp123-decisions>, panel: "B")
    plots median completed-volley count. The two rows can include different
    trial subsets. Stability was assessed retrospectively, not by a stopping rule.

    === Accuracy at matched coordinates
    #figure(
      data-image(
        data-file("exp123/accuracy_comparison.png"),
        width: 100%,
        alt: "Three accuracy curves compare inhibitory decay at matched elapsed time, cycle position and cumulative input count; three corresponding coverage plots show available paired trials.",
      ),
      caption: [A–C: accuracy against time, cycle position and cumulative input
        spikes. D–F: paired-trial coverage. Sample: 500 images, 50 per digit,
        three paired encodings; decay 3/6/12 ms in black/red/cyan. Shading:
        image-bootstrap 95% intervals. Dashed lines mark 80% coverage;
        accuracy is omitted below it. Later coordinates select subsets.
        Equal spike counts do not guarantee equal information.],
    ) <fig:exp123-accuracy>

    === Accuracy and spike cost
    #figure(
      data-image(
        data-file("exp123/deadline_tradeoffs.png"),
        width: 100%,
        alt: "Nine panels show accuracy versus deadline, network spikes versus deadline and accuracy versus network spikes at three input encoding rates.",
      ),
      caption: [A/D/G: accuracy versus deadline. B/E/H: mean network spikes versus
        deadline. C/F/I: accuracy versus spike cost. Rows: 12.5/25/50 Hz
        encoding; colours: 3/6/12 ms decay. Points mark five deadlines,
        connected within each setting. Sample: 500 images, three paired
        encodings. Shading in A/D/G: image-bootstrap 95% intervals.
        Cost includes E, I and output spikes, excluding input.
        Spike cost is not physical energy; encoding rate also changes drive.],
    ) <fig:exp123-tradeoffs>

    === Pareto candidates and frequency
    #figure(
      data-image(
        data-file("exp123/pareto_frontiers.png"),
        width: 100%,
        alt: "Three connected bubble plots show empirical Pareto candidates at different encoding rates, with bubble area representing spike cost; three spectral-frequency plots verify decay modulation.",
      ),
      caption: [A–C: Pareto comparisons at 12.5/25/50 Hz encoding. Position shows
        deadline and accuracy; bubble area shows mean network spikes on a
        common scale. Colours identify decay; traces connect deadlines.
        Outlined bubbles are empirically nondominated; faint bubbles are
        dominated. D–F: peaks of mean E-population PSDs at the same rates.
        Peaks can reflect harmonics or heterogeneous rhythms. Sample and cost definition:
        #result-figure-ref(<fig:exp123-tradeoffs>). Frontier membership is
        sample-dependent.],
    ) <fig:exp123-pareto>

    == Parameters <sec:exp123-parameters>
    #table(
      columns: (2fr, 3fr),
      table.header([Parameter], [Value]),
      [Population sizes], [Input: 784; E: 1,024; I: 256; output: 10],
      [Capacitance], [E: 1.0 nF; I: 0.5 nF],
      [Leak conductance], [E: 0.05 µS; I: 0.10 µS],
      [Rest and reset voltage], [E and I: −65 mV],
      [Spike threshold], [E and I: −50 mV],
      [Refractory hold], [E: 1.2 ms; I: 0.6 ms],
      [AMPA decay], [2 ms],
      [Inhibitory decay sweep], [3, 6 and 12 ms],
      [Reciprocal projection delay], [One simulation step (0.1 ms)],
      [Input filter and output leak], [2 ms each],
      [Output threshold and reset], [Threshold 1; subtractive reset],
      [Simulation step and duration], [0.1 ms; 200 ms; no burn-in],
      [Input encoding rates], [12.5, 25 and 50 Hz],
      [Frozen weights],
      [Final epoch after 50 training epochs; training seed 42;
        directly reused without selection or fitting here],

      [Initial state per presentation],
      [Hidden neurons at rest;
        conductances and output state zero],

      [Encoding generator seed], [1,000 × image index + draw identifier],
    )

  ]
}
#let report-body = if inputs-ready(data-file, inputs) {
  render-report()
} else {
  pending-report(
    data-file,
    inputs,
    [How does inhibitory decay affect output evidence, stable decisions and accuracy–spike-cost trade-offs?],
    preview-figures,
    json-inputs: (),
  )
}
#let meta = meta + (assets: input-assets("exp123", inputs))
#let body = journal-article("exp123", inputs, report-body)
