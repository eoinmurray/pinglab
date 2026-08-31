#import "contents.typ": with-contents, with-result-sections
#import "/.demolab/lib.typ": data-image, data-json
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets
#import "run-view.typ": run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp082")

#let meta = (
  status: "[▦ DATA | v28.0.0]",
  title: "Spike-Count Classification in a Continuous Stream",
  date: "2026-08-10",
  updated_at: "2026-08-31",
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
  often produced no output spikes. The results demonstrate continuous-stream
  classification, but do not separate viewing time from decision time or
  establish a causal role for gamma.

  == Results

  #with-result-sections[

  === Five-digit capability stream across 5–25 Hz

  The predeclared selection produced a 5/5 stream on its first candidate. This
  selected success demonstrates capability, not expected accuracy; the aggregate
  duration–rate results provide the population estimate.

  #figure(
    report-image(
      "exp082/hero_stream.png",
      "Five correctly classified digits presented for 200 ms at input rates from 5 to 25 Hz, with excitatory and inhibitory rasters and spike-count evidence.",
    ),
    caption: [Illustrative seed-42 capability showcase across 5, 7.5, 10, 15
      and 25 Hz, with every digit presented for 200 ms. Before inference, we fixed
      candidate order and selected the first stream classified 5/5 correctly.
      The first candidate qualified immediately, so no
      failed candidate was skipped to obtain this image. Red traces show the true
      classes.],
  )

  === Accuracy across presentation duration and input rate

  Averaged across rates and seeds, accuracy was
  #pct(mean(at-duration(25.0).map(row => row.accuracy))),
  #pct(mean(at-duration(50.0).map(row => row.accuracy))),
  #pct(mean(at-duration(100.0).map(row => row.accuracy))) and
  #pct(mean(at-duration(200.0).map(row => row.accuracy))) at 25, 50, 100 and
  200 ms. Mean silence fell from
  #pct(mean(at-duration(25.0).map(row => row.silent_fraction))) to
  #pct(mean(at-duration(200.0).map(row => row.silent_fraction))). Longer viewing
  and integration time remain confounded.

  #figure(
    report-image(
      "exp082/duration_rate_summary.png",
      "Seed-mean accuracy across 25 to 200 ms and 0.5 to 25 Hz, with the 200 ms psychometric alongside.",
      ratio: 0.49,
    ),
    caption: [Means across three trained seeds, #r.config.digits_per_seed_cell
      test presentations per seed and condition; curve bars are ± sample SD/√3,
      not population confidence intervals.],
  )

  In the 200 ms slice shown at right, accuracy rose from #pct(rate-mean(0.5))
  at 0.5 Hz to #pct(rate-mean(3.0)) at 3 Hz and #pct(rate-mean(7.5)) at
  7.5 Hz. The 10, 15 and 25 Hz means spanned
  #pct(minimum(dense-means))–#pct(maximum(dense-means)), so the dense end did
  not collapse. At 15 Hz, seed accuracies spanned
  #pct(minimum(at-rate(15.0).map(row => row.accuracy)))–#pct(maximum(at-rate(15.0).map(row => row.accuracy))).
  At 0.5 Hz, they spanned
  #pct(minimum(at-rate(0.5).map(row => row.accuracy)))–#pct(maximum(at-rate(0.5).map(row => row.accuracy))),
  with #pct(mean(at-rate(0.5).map(row => row.silent_fraction))) silent windows.
  No fixed-rate-training control or independent stream-bank repeat establishes
  why this transfer works. Comparison with #link("/exp048/")[the mean-voltage
  streaming study] changes both training distribution and readout, so it cannot
  establish decoder superiority.

  === Spike-count decision during one 200 ms trial

  Selecting a successful presentation cannot estimate accuracy or show that
  each rhythmic burst improves the decision.

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

  === Output-count transition between 91.5 and 94.5 ms

  #figure(
    report-image(
      "exp082/single_trial_transition.png",
      "Output spikes, cumulative class counts and softmax count shares from 91.5 to 94.5 ms in the same digit-4 presentation.",
      ratio: 0.70,
    ),
    caption: [Post-hoc enlargement of 91.5–94.5 ms in the same displayed trial.
      Each output spike increments one class count; the softmax transformation can
      therefore produce an abrupt change in every displayed count share. Red marks
      the true and winning class 4.],
  )

  From the readout definition, the second row should consist of non-decreasing
  integer staircases: each output spike increments exactly one cumulative class
  count $z_c[k]$ by one, while every count remains flat between spikes. The third
  row should also be piecewise constant, but its softmax shares
  $p_"class"(c,k)$ may jump either upward or downward. A spike multiplies the
  corresponding class's unnormalised weight by $e$, the base of natural
  logarithms, while normalization changes every displayed share.
  The enlargement therefore explains the apparent jump, but does not estimate
  performance or establish a causal role for rhythmic bursts.

  === Three-of-five counterexample under showcase conditions

  Under the nominal showcase conditions, the first stream scoring exactly 3/5
  was the #(r.showcase_selection.selected.alternative + 1)th candidate, after
  #r.showcase_selection.candidates.len() candidates were evaluated. Digits 9 and
  2 were misclassified as 4 and 9. Together with the sparse-input example below,
  this prevents the selected 5/5 showcase from implying perfect reliability.

  #figure(
    report-image(
      "exp082/alternative_stream.png",
      "A three-of-five counterexample under the same 200 ms and 5 to 25 Hz conditions as the capability showcase.",
    ),
    caption: [Nominal-regime counterexample under the same seed-42 network,
      durations, rates and deterministic candidate order as the showcase. Badges
      show true→predicted labels.],
  )

  === Mixed-duration stream with silent and non-silent failures

  In the variable stream, #r.variable_stream.correct.sum() of
  #r.variable_stream.labels.len() decisions were correct. The 200 ms, 0.5 Hz
  window emitted no output spikes; the 100 ms, 2 Hz window also failed. The
  25 Hz/50 ms, 10 Hz/25 ms and 5 Hz/200 ms presentations succeeded. Five
  outcomes do not estimate reliability.

  #figure(
    report-image(
      "exp082/variable_stream.png",
      "Five digits with changing rates and durations: three correct predictions, a silent 0.5 Hz failure and a non-silent 2 Hz failure.",
    ),
    caption: [Seed-42 illustration shown here. Counts reset at boundaries while
      hidden state continues. Badges show
      true→predicted labels; thumbnail opacity increases with rate.
      Display sampling matches the preceding raster.],
  )

  #context if target() != "html" { pagebreak(weak: true) }
  #block(sticky: true)[
  ]

    == Methods

    We reused trained networks and recorded grid measurements to evaluate
    matched-window spike-count classification. We additionally simulated a bounded,
    predeclared sequence of illustrative streams; we did not retrain any network.
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
    were always equal. The corrected execution concatenated streams
    along the batch axis, without changing the recipe.

  + *Read output spike counts.* The learned projection from excitatory spikes
    drove the output-LIF units. For a presentation covering timesteps $k_a$ through
    $k_b-1$, the count-based class score at timestep $k$ was
    $ z_c[k] = sum_(j=k_a)^k s_c^"out"[j], quad k_a <= k < k_b. $ <eq-count>
    Here $j$ indexes timesteps and $s_c^"out"[j]$ is the dimensionless binary output spike.
    Prediction was $arg max_c z_c[k_b-1]$; ties selected the lowest class index,
    including class 0 for silent windows. The displayed count share was
    $ p_"class"(c,k) = frac(exp(z_c[k]), sum_(j=0)^9 exp(z_j[k])). $ <eq-share>
    Here $j$ indexes the ten digit classes; $p_"class"$ is a softmax transformation,
    not a calibrated posterior. Seed-42 illustrations reused a five-digit
    matched stream and a fixed changing-duration/rate stream. For the additional
    nominal-regime illustrations, every digit lasted 200 ms and rates were 5,
    7.5, 10, 15 and 25 Hz. We fixed an ascending candidate index, with deterministic
    digit and encoding seeds derived from it, before inference; we selected the
    first 5/5 stream as the capability showcase and the first 3/5 stream as its
    counterexample. The search stopped after
    #r.showcase_selection.candidates.len() candidates. The single-digit
    explanation selected the first correct matched presentation, and its
    91.5–94.5 ms enlargement was selected post hoc around the displayed transition.

  + *Measure condition-level performance.* Accuracy for presentation duration $T_"present"$ (ms),
    maximum-pixel input rate $r_"input,max"$ (Hz) and trained seed $xi$ was
    $
      "Acc"(T_"present",r_"input,max",xi) = 1/N_"eval" sum_(i=1)^(N_"eval") bb(1)[hat(y)_(i,T_"present",r_"input,max",xi) = y_i].
    $ <eq-accuracy>
    Here $N_"eval"=200$, $i$ indexes digit presentations, $hat(y)$ and $y$ are predicted
    and true labels, and $bb(1)$ is an indicator. We recorded the original
    per-seed aggregates of accuracy, class spike totals, output spikes per
    presentation, silence and E/I rates; individual grid decisions were not
    archived. E/I rates used all 1,024/256 neurons and the complete presentation
    duration. Seed means and SEM summarize three training replicates, not 600
    independent network replicates; no additional stream-bank repeats, adjusted
    comparisons or population-level intervals support broader generalization.
    No causal gamma manipulation, fixed-rate-training control or separation of
    viewing from integration time was performed.
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
#let body = with-contents(body)
