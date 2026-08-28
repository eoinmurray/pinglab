#import "contents.typ": with-contents
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp082")

#let meta = (
  status: "[▦ DATA]",
  title: "Spike-Count Classification in a Continuous Stream",
  date: "2026-08-10",
  updated_at: "2026-08-28",
  description: "A multi-seed study of spike-count classification across input rates and presentation durations.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp082",)
#let preview-figures = (
  (path: "exp082/shared_design_schematic.svg", label: "shared design schematic"),
  (path: "exp082/single_trial.png", label: "single trial"),
  (path: "exp082/variable_stream.png", label: "variable stream"),
  (path: "exp082/duration_rate_summary.png", label: "duration rate summary"),
  (path: "exp082/psychometric_200ms.svg", label: "psychometric 200ms"),
)

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
      box(width: width, height: width * ratio,
        data-image(data-file(path), width: width, height: width * ratio, fit: "contain", alt: alt))
    })
  }
}
#let rate-mean(rate) = mean(at-rate(rate).map(row => row.accuracy))
#let dense-means = (10.0, 15.0, 25.0).map(rate-mean)

  == Abstract

  Longer matched presentation and decision windows improved continuous-stream
  digit classification across three frozen spiking networks trained with
  variable input rates. We reanalysed retained MNIST evaluations crossing four
  durations with eleven rates, comprising #r.scientific_preflight.evaluation_grid.n_presentations
  decisions. Mean accuracy across rates and networks rose from
  #pct(mean(at-duration(25.0).map(row => row.accuracy))) at 25 ms to
  #pct(mean(at-duration(200.0).map(row => row.accuracy))) at 200 ms. At 200 ms,
  mean accuracy remained #pct(minimum(dense-means))–#pct(maximum(dense-means))
  across 10–25 Hz, while weak inputs frequently produced silent outputs.
  These results support tolerance within the tested rates, but do not separate
  viewing time from integration time or establish a causal benefit of rhythmic activity.

  #run-view("exp082", inputs)

  == Results

  === Continuous hidden state, separate decisions

  #figure(
    report-image("exp082/shared_design_schematic.svg",
      "Protocol diagram: digit inputs drive continuous hidden activity, while output state and class counts reset at each digit boundary.", ratio: 0.27),
    caption: [Prospective mechanism, not measured evidence. Hidden state continues
      within each stream; output-LIF state and counts reset at digit boundaries.
      The design tests broad rate tolerance against a narrow preferred rate, with
      output silence as a diagnostic. Presentation and decision duration change together.],
  )

  === A successful spike-count decision

  #figure(
    report-image("exp082/single_trial.png",
      "Digit 4 with rasters of 200 excitatory and 64 inhibitory cells, and ten softmax count-share trajectories."),
    caption: [Retained seed-42 digit #r.single_trial.labels.first(), the first correct
      presentation in the 200 ms, 5 Hz matched stream. Red marks the true and
      winning class. Rasters display the first 200 E and 64 I cells, not the full
      populations. Softmax count shares explain the readout; they are not calibrated
      probabilities. Selecting a success cannot estimate accuracy or show that
      each rhythmic burst improves the decision.],
  )

  === A changing stream exposes sparse-input failures

  #figure(
    report-image("exp082/variable_stream.png",
      "Five digits with changing rates and durations: three correct predictions, a silent 0.5 Hz failure and a non-silent 2 Hz failure."),
    caption: [Retained seed-42 illustration: #r.variable_stream.correct.sum()
      of #r.variable_stream.labels.len() decisions were correct. The 200 ms,
      0.5 Hz window emitted no output spikes; the 100 ms, 2 Hz window also failed.
      The 25 Hz/50 ms, 10 Hz/25 ms and 5 Hz/200 ms presentations succeeded.
      Counts reset at boundaries while hidden state continues. Badges show
      true→predicted labels; thumbnail opacity increases with rate.
      Display sampling matches the preceding raster. Five outcomes do not estimate reliability.],
  )

  === Duration and input rate constrain accuracy

  #figure(
    report-image("exp082/duration_rate_summary.png",
      "Seed-mean accuracy across 25 to 200 ms and 0.5 to 25 Hz, with the 200 ms psychometric alongside.", ratio: 0.49),
    caption: [Means across three trained seeds, #r.config.digits_per_seed_cell
      test presentations per seed and condition; curve bars are ± sample SD/√3,
      not population confidence intervals. Averaged across rates and seeds,
      accuracy was #pct(mean(at-duration(25.0).map(row => row.accuracy))),
      #pct(mean(at-duration(50.0).map(row => row.accuracy))),
      #pct(mean(at-duration(100.0).map(row => row.accuracy))) and
      #pct(mean(at-duration(200.0).map(row => row.accuracy))) at 25, 50, 100 and
      200 ms. Mean silence fell from
      #pct(mean(at-duration(25.0).map(row => row.silent_fraction))) to
      #pct(mean(at-duration(200.0).map(row => row.silent_fraction))).
      Longer viewing and integration time remain confounded.],
  )

  #figure(
    report-image("exp082/psychometric_200ms.svg",
      "At 200 ms, accuracy rises from the sparse edge and remains high at 10, 15 and 25 Hz.", ratio: 0.58),
    caption: [The same 200 ms slice, mean ± sample SD/√3 across seeds 42–44.
      Accuracy rose from #pct(rate-mean(0.5)) at 0.5 Hz to
      #pct(rate-mean(3.0)) at 3 Hz and #pct(rate-mean(7.5)) at 7.5 Hz.
      The 10, 15 and 25 Hz means span #pct(minimum(dense-means))–#pct(maximum(dense-means)).
      The dense end did not collapse. At 15 Hz, seed accuracies span
      #pct(minimum(at-rate(15.0).map(row => row.accuracy)))–#pct(maximum(at-rate(15.0).map(row => row.accuracy))).
      At 0.5 Hz, the range is #pct(minimum(at-rate(0.5).map(row => row.accuracy)))–#pct(maximum(at-rate(0.5).map(row => row.accuracy))),
      with #pct(mean(at-rate(0.5).map(row => row.silent_fraction))) silent windows.
      No fixed-rate-training control or independent stream-bank repeat establishes
      why this transfer works. Comparison with
      #link("/exp048/")[the mean-voltage streaming study] changes both training
      distribution and readout, so it cannot establish decoder superiority.],
  )

  #context if target() != "html" { pagebreak(weak: true) }
  #block(sticky: true)[
    == Methods

    We reused trained networks and retained inference measurements to evaluate
    matched-window spike-count classification, without retraining or new simulation.
  ]
  #set math.equation(numbering: "(1)")
  #show math.equation.where(block: true): equation => context {
    if target() == "html" {
      html.elem("div", attrs: (class: "exp082-equation", style: "display:flex;align-items:center;gap:1em"), {
        html.elem("div", attrs: (style: "flex:1;min-width:0"), equation)
        html.elem("span", numbering("(1)", ..counter(math.equation).at(equation.location())))
      })
    } else { equation }
  }

  + *Select frozen classifiers.* Three PING networks (seeds 42–44) came from
    #link("/exp022/")[the variable-rate training bank], with 1,024 excitatory,
    256 inhibitory and ten output-LIF units. Training used 6,300 optimization
    and 700 validation images from MNIST for 50 epochs, sampling eleven
    maximum-pixel rates between 0.5 and 25 Hz.
    Minimum validation cross-entropy averaged over three encoder draws selected
    each checkpoint, with validation accuracy breaking ties; neither final-epoch
    weights nor the official test partition determined selection.

  + *Present continuous test streams.* Each duration–rate–seed condition used
    40 independent five-digit streams sampled from the official 10,000-image
    test partition, giving 200 decisions; five streams ran per batch with
    separate neuronal states. Pixels generated independent Bernoulli spikes at
    0.1 ms resolution. Within a stream, hidden state persisted, but output-LIF
    state and counts reset at each boundary. The grid crossed 25, 50, 100 and
    200 ms with all eleven training rates; presentation and readout windows
    were always equal. The corrected retained execution concatenated streams
    along the batch axis, without changing the recipe.

  + *Read output spike counts.* The learned projection from excitatory spikes
    drove the output-LIF units. For a presentation covering timesteps $a$ through
    $b-1$, the count for class $c$ at timestep $u$ was
    $ z_c (u) = sum_(t=a)^u s_c^"out" (t), quad a <= u < b. $ <eq-count>
    Here $t$ indexes timesteps and $s_c^"out" (t)$ is the binary output spike.
    Prediction was $arg max_c z_c (b-1)$; ties selected the lowest class index,
    including class 0 for silent windows. The displayed count share was
    $ p_c (u) = frac(exp(z_c (u)), sum_(k=0)^9 exp(z_k (u))). $ <eq-share>
    Here $k$ indexes the ten digit classes; $p_c$ is a softmax transformation,
    not a calibrated posterior. Seed-42 illustrations reused a five-digit
    matched stream and a fixed changing-duration/rate stream; the single-digit
    explanation selected the first correct matched presentation.

  + *Measure condition-level performance.* Accuracy for duration $d$ (ms),
    maximum-pixel input rate $r$ (Hz) and trained seed $s$ was
    $ A_(d,r,s) = 1/N sum_(i=1)^N bb(1)[hat(y)_(i,d,r,s) = y_i]. $ <eq-accuracy>
    Here $N=200$, $i$ indexes digit presentations, $hat(y)$ and $y$ are predicted
    and true labels, and $bb(1)$ is an indicator. We retained original
    per-seed aggregates of accuracy, class spike totals, output spikes per
    presentation, silence and E/I rates; individual grid decisions were not
    archived. E/I rates use all 1,024/256 cells and the complete presentation
    duration. Seed means and sample SD/√3 summarize three networks, not 600
    independent network replicates; no additional stream-bank repeats, adjusted
    comparisons or population-level intervals support broader generalization.
    No causal gamma manipulation, fixed-rate-training control or separation of
    viewing from integration time was performed.
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [Can spike-count outputs classify a continuous stream? Compare held-out examples across changing input rate and digit duration while hidden state continues.],
    preview-figures, json-inputs: ("exp082",),
  )
}

#let meta = meta + (assets: input-assets("exp082", inputs))
#let body = with-datasets("exp082", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-contents(body)
