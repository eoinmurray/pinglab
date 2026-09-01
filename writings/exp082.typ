#import "contents.typ": with-contents, with-numbered-equations, with-result-sections
#import "/.demolab/lib.typ": data-image, data-json
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets
#import "run-view.typ": run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp082")

#let meta = (
  status: "[▦ DATA | v31.2.0]",
  title: "Spike-Count Classification in a Continuous Stream",
  created_at: "2026-08-10T00:00:00Z",
  updated_at: "2026-09-01",
  description: "A multi-seed study of spike-count classification across input rates and presentation durations.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp082",)
#let preview-figures = (
  (path: "exp082/hero_stream.png", label: "capability showcase"),
  (path: "exp082/duration_rate_summary.png", label: "duration rate summary"),
  (path: "exp082/single_trial.png", label: "single trial"),
  (path: "exp082/single_trial_transition.png", label: "single-trial transition"),
  (path: "exp082/alternative_stream.png", label: "nominal-regime counterexample"),
  (path: "exp082/variable_stream.png", label: "variable stream"),
)

#let result-card-style = context {
  if target() == "html" {
    html.elem("style",
      ".pinglab-result-card { margin: 1.25rem 0; padding: 1.2rem 1.35rem 1.3rem; border: 1px solid var(--rule-strong); border-radius: 3px; background: var(--paper); } "
      + ".pinglab-result-card > h4:first-child { margin-top: 0; } "
      + ".pinglab-result-card > :last-child { margin-bottom: 0; } "
      + ".pinglab-result-card-notes { margin-top: 1rem; padding-top: .75rem; border-top: 1px solid var(--rule); font-size: var(--fs-small); line-height: 1.5; color: var(--muted); } "
      + ".pinglab-result-card-notes > p:first-child { margin: 0 0 .25rem; } "
      + ".pinglab-result-card-notes ul { margin: 0; padding-left: 1.2rem; } "
      + ".pinglab-result-card-notes li { margin: .2rem 0; } "
      + "@media (max-width: 520px) { .pinglab-result-card { margin: 1rem 0; padding: .95rem 1rem 1.05rem; } }",
    )
  }
}

#let result-card(body, notes: none) = context {
  let notes-body = if notes == none { none } else if target() == "html" {
    html.elem("aside", attrs: (class: "pinglab-result-card-notes", "aria-label": "Notes"), [
      *Notes.*
      #notes
    ])
  } else { [
    *Notes.*
    #notes
  ] }
  let card-body = [#body #notes-body]
  if target() == "html" {
    html.elem("article", attrs: (class: "pinglab-result-card"), card-body)
  } else { card-body }
}

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
  #let r = data-json(data-file("exp082/numbers.json"))
  #let pct(x) = str(calc.round(100 * x, digits: 1)) + "%"
  #let mean(xs) = xs.sum() / xs.len()
  #let minimum(xs) = calc.min(..xs)
  #let maximum(xs) = calc.max(..xs)
  #let at-duration(duration) = r.grid_per_seed.filter(row => row.duration_ms == duration)
  #let at-rate(rate) = r.duration_200ms_psychometric.filter(row => row.rate_hz == rate)
  #let report-image(path, alt, ratio: 0.77) = context {
    if target() == "html" {
      data-image(data-file(path), width: 100%, alt: alt)
    } else {
      layout(size => {
        let width = size.width
        box(width: width, height: width * ratio, data-image(
          data-file(path),
          width: width,
          height: width * ratio,
          fit: "contain",
          alt: alt,
        ))
      })
    }
  }
  #let rate-mean(rate) = mean(at-rate(rate).map(row => row.accuracy))
  #let dense-means = (10.0, 15.0, 25.0).map(rate-mean)

  == Abstract

  This experiment asked whether networks trained on separate MNIST digits could
  classify digits presented continuously. Recurrent state was preserved between
  digits, while the output decision was reset at each boundary.

  Classification improved with longer, stronger inputs, whereas weak inputs
  often produced no output spikes.

  == Results

  #with-result-sections[

  #result-card-style

  #result-card(notes: [
    - Before inference, we fixed candidate order and selected the first stream
      classified 5/5 correctly. The first candidate qualified, so no failed
      candidate was skipped to obtain this image.
  ])[
    === Five-digit capability stream across 5–25 Hz

    One variable-rate-trained PING network classified a five-digit
    MNIST stream using output-LIF spike counts. Hidden neuronal state continued
    between presentations, while output state and counts reset at each known
    boundary.

    If the classifier tolerated both carried state and changing
    input rate, at least some complete streams should remain correctly classified.

    The first candidate in the predeclared deterministic sequence was
    classified correctly throughout, despite the changing input rate.

    #figure(
      report-image(
        "exp082/hero_stream.png",
        "Five correctly classified digits presented for 200 ms at input rates from 5 to 25 Hz, with excitatory and inhibitory rasters and spike-count evidence.",
      ),
      caption: [Illustrative seed-42 capability showcase across 5, 7.5, 10, 15
        and 25 Hz, with every digit presented for 200 ms. Red traces show the
        true classes.],
    )

  ]

  #result-card[
    === Accuracy across presentation duration and input rate

    Three independently trained PING networks classified continuous
    MNIST streams while presentation duration and maximum-pixel input rate varied.
    Hidden state persisted between digits, but each output decision reset at the
    boundary.

    Longer and stronger presentations should provide more output
    evidence, reducing silent decisions and improving classification.

    Classification improved with longer and stronger inputs. Brief,
    weak presentations frequently produced no output spikes, while performance
    remained strong across the denser input conditions.

    #figure(
      report-image(
        "exp082/duration_rate_summary.png",
        "Seed-mean accuracy across 25 to 200 ms and 0.5 to 25 Hz, with the 200 ms psychometric alongside.",
        ratio: 0.49,
      ),
      caption: [Means across three training replicates,
        #r.config.digits_per_seed_cell test presentations per replicate and
        condition; curve bars are ± sample SD/√3, not population confidence
        intervals.],
    )

  ]

  #result-card[
    === Spike-count decision during one 200 ms trial

    One correctly classified digit from the continuous stream was
    examined to show how excitatory and inhibitory spiking became an output
    spike-count decision.

    A correct decision should end with the true class holding the
    largest cumulative output spike count.

    Class-specific output spikes accumulated until class 4 held the
    largest count at the presentation boundary.

    #figure(
      report-image(
        "exp082/single_trial.png",
        "Digit 4 with rasters of 200 excitatory and 64 inhibitory neurons, and ten softmax count-share trajectories.",
      ),
      caption: [Seed-42 digit #r.single_trial.labels.first() shown here, the first correct
        presentation in the 200 ms, 5 Hz matched stream. Red marks the true and
        winning class. Rasters display the first 200 E and 64 I neurons, not the full
        populations. Softmax count shares explain the readout; they are not calibrated
        probabilities.],
    )

  ]

  #result-card[
    === Output-count transition between 91.5 and 94.5 ms

    A short interval from the same digit-4 presentation was enlarged
    to explain an abrupt change in the displayed class shares.

    From the readout definition, counts should remain flat between
    output spikes and step upward only when the corresponding class neuron spikes.

    The displayed counts formed non-decreasing integer staircases.
    Softmax normalization then changed every class share at each count increment.

    #figure(
      report-image(
        "exp082/single_trial_transition.png",
        "Output spikes, cumulative class counts and softmax count shares from 91.5 to 94.5 ms in the same digit-4 presentation.",
        ratio: 0.70,
      ),
      caption: [Post-hoc enlargement of 91.5–94.5 ms in the same displayed trial.
        Rows show output spikes, cumulative class counts and softmax count shares;
        red marks the true and winning class 4.],
    )

  ]

  #result-card[
    === Three-of-five counterexample under showcase conditions

    The same network, durations, rates and deterministic candidate
    order used for the successful showcase were searched for a stream containing
    exactly three correct decisions.

    If the showcase represented capability rather than perfect
    reliability, comparable streams should also contain classification errors.

    The first stream scoring exactly 3/5 was candidate
    #(r.showcase_selection.selected.alternative + 1), after
    #r.showcase_selection.candidates.len() candidates were evaluated. Digits 9
    and 2 were misclassified as 4 and 9.

    #figure(
      report-image(
        "exp082/alternative_stream.png",
        "A three-of-five counterexample under the same 200 ms and 5 to 25 Hz conditions as the capability showcase.",
      ),
      caption: [Nominal-regime counterexample under the same seed-42 network,
        durations, rates and deterministic candidate order as the showcase. Badges
        show true→predicted labels.],
    )

  ]

  #result-card[
    === Mixed-duration stream with silent and non-silent failures

    A five-digit stream varied both presentation duration and input
    rate to expose the classifier to easy and difficult conditions within one
    continuous trajectory.

    Brief or weak presentations should be most vulnerable because
    they provide less time or input for output evidence to accumulate.

    #r.variable_stream.correct.sum() of
    #r.variable_stream.labels.len() decisions were correct. The 200 ms, 0.5 Hz
    presentation was silent, while the 100 ms, 2 Hz presentation failed despite
    producing output spikes.

    #figure(
      report-image(
        "exp082/variable_stream.png",
        "Five digits with changing rates and durations: three correct predictions, a silent 0.5 Hz failure and a non-silent 2 Hz failure.",
      ),
      caption: [Seed-42 illustration shown here. Counts reset at boundaries while
        hidden state continues. Badges show true→predicted labels; thumbnail
        opacity increases with rate. Display sampling matches the preceding raster.],
    )

  ]

  #context if target() != "html" { pagebreak(weak: true) }
  #block(sticky: true)[
  ]

    == Methods

  ]
  === Compute

  + *Classifiers.* We reused three frozen PING networks, seeds 42–44, from
    #link("/exp022/")[the variable-rate training bank]. We did not retrain them.

  + *Architecture.* Each network contained 1,024 excitatory neurons, 256
    inhibitory neurons and ten output-LIF units. A learned projection carried
    excitatory spikes to the output units.

  + *Training provenance.* Training used 6,300 optimization and 700 validation
    MNIST images for 50 epochs, sampling eleven maximum-pixel rates from 0.5 to
    25 Hz.

  + *Checkpoint selection.* Minimum validation cross-entropy, averaged over
    three encoder draws, selected each checkpoint; validation accuracy broke
    ties. Neither final-epoch weights nor the official test partition selected
    the models.

  + *Test data.* Evaluation sampled the official 10,000-image MNIST test
    partition. Each duration–rate–seed condition contained 40 independent
    five-digit streams, giving 200 digit decisions.

  + *Batching.* Five streams ran per batch with separate neuronal states. The
    corrected execution concatenated streams along the batch axis without
    changing the experimental recipe.

  + *Input encoding.* Pixels generated independent Bernoulli spikes at 0.1 ms
    resolution, scaled by the condition's maximum-pixel input rate.

  + *State handling.* Hidden neuronal state persisted between digits within a
    stream. Output-LIF state and spike counts reset at every known boundary.

  + *Condition grid.* Presentations lasted 25, 50, 100 or 200 ms at each of the
    eleven training rates. Presentation and readout windows were always equal.

  + *Count class spikes.* During each presentation, the score for class $c$ was
    its total output-spike count:
    $ z_c = sum_(k in "presentation") s_c[k]. $ <eq-count>
    Here $k$ is a simulation timestep and $s_c[k]$ is 1 when output unit $c$
    spikes and 0 otherwise.

  + *Predict the digit.* Prediction selected the largest entry in the ten-class
    score vector $z$: $hat(y) = arg max(z)$. Ties selected the lowest class
    index, including class 0 when every output count was zero.

  + *Illustrative streams.* Seed-42 figures reused one five-digit matched stream
    and one fixed changing-duration/rate stream. The nominal-regime figures used
    200 ms digits at 5, 7.5, 10, 15 and 25 Hz.

  + *Candidate selection.* Before inference, we fixed ascending candidate order
    and derived deterministic digit and encoding seeds from each index. We selected
    the first 5/5 stream as the capability showcase and the first 3/5 stream as
    its counterexample; the search stopped after
    #r.showcase_selection.candidates.len() candidates.

  === Analyse

  #set enum(start: 14)

  + *Explanatory close-up.* The single-digit figure used the first correct
    matched presentation. Its 91.5–94.5 ms enlargement was selected post hoc
    around the displayed transition.

  + *Display count shares.* Figures transformed the score vector with
    $ q = "softmax"(z). $ <eq-share>
    Here $q$ is the displayed vector of class shares. It is not a calibrated
    posterior probability.

  + *Measure accuracy.* We calculated accuracy separately for each
    duration–rate–network condition as the proportion of correct decisions.
    Each condition contained 200 decisions.

  + *Retained measurements.* We recorded the original per-seed aggregates of
    accuracy, class spike totals, output spikes per presentation, silence and
    E/I rates. Individual grid decisions were not archived; E/I rates used all
    1,024/256 neurons across the complete presentation.

  + *Summaries and limits.* Seed means and SEM summarize three training
    replicates, not 600 independent network replicates. No additional
    stream-bank repeats, adjusted comparisons, population-level intervals,
    causal gamma manipulation, fixed-rate-training control or separation of
    viewing from integration time support broader claims.

  === Present

  #set enum(start: 19)

  + *Render figures.* The presentation stage rendered the article figures from
    saved analysis and its explicitly linked compute recordings. It did not
    rerun inference, reselect an illustration or recompute condition statistics.

  #set enum(start: 1)
  #run-view("exp082", inputs)

]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file,
    inputs,
    [Can spike-count outputs classify a continuous stream? Compare held-out examples across changing input rate and digit duration while hidden state continues.],
    preview-figures,
    json-inputs: ("exp082",),
  )
}

#let meta = meta + (assets: input-assets("exp082", inputs))
#let body = with-datasets("exp082", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-numbered-equations(body)
#let body = with-contents(body)
