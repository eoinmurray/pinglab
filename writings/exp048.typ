#import "templates/article-layout.typ": journal-article
#import "templates/result-card.typ": result-figure-ref, result-card, with-result-sections
#import "templates/references.typ": journal-references
#import "/.demolab/lib.typ": cite, data-image, data-json
#import "templates/dataset.typ": data-file, inputs-ready, pending-report, run-view, input-assets
#import "templates/abstract.typ": journal-abstract
#import "templates/methods.typ": journal-methods
#let data-file = data-file.with(article: "exp048")

#let meta = (
  tags: ("data", "v35.4.0"),
  title: "[DEPRECATED] Accuracy Across Duration and Input Rate",
  created_at: "2026-06-08T00:00:00Z",
  updated_at: "2026-08-31T00:00:00Z",
  description: "Streaming psychometric curves identify the presentation durations and encoding rates that support classification in a frozen trained PING network.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp048",)
#let preview-figures = (
  (path: "exp048/varying_headline_stream.png", label: "varying headline stream"),
  (path: "exp048/acc_grid_tau_rate.png", label: "acc grid tau rate"),
)

// Clip only footer bookkeeping in the article; source images remain unchanged.
#let report-image(path, trim-bottom: 0%, alt: none) = context {
  if target() == "html" {
    html.elem("div", attrs: (style: "clip-path:inset(0 0 " + str(trim-bottom / 1%) + "% 0)"), data-image(
      path,
      width: 100%,
      alt: alt,
    ))
  } else {
    layout(size => {
      let graphic = data-image(path, width: size.width, alt: alt)
      let dimensions = measure(graphic)
      pad(bottom: dimensions.height * trim-bottom, block(
        width: size.width,
        height: dimensions.height * (100% - trim-bottom),
        clip: true,
        graphic,
      ))
    })
  }
}

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
  #let r = data-json(data-file("exp048/numbers.json"))
  #let cfg = r.config
  #let rate-at(rate) = r.encoding_rate_psychometric.curve.filter(x => x.input_rate_hz == rate).at(0)
  #let p05 = rate-at(0.5)
  #let p2 = rate-at(2.0)
  #let p5 = rate-at(5.0)
  #let p10 = rate-at(10.0)
  #let varying-correct = r.varying_headline.seg_correct.fold(0, (total, x) => total + x)
  #let varying-conditions = (
    r
      .varying_headline
      .segments
      .map(
        x => str(x.at(0)) + " ms at " + str(x.at(1)) + " Hz",
      )
      .join("; ")
  )
  #let varying-predictions = (
    range(r.varying_headline.labels.len())
      .map(
        i => str(r.varying_headline.labels.at(i)) + "→" + str(r.varying_headline.seg_preds.at(i)),
      )
      .join(", ")
  )
  #let grid-cells = cfg.tau_grid_ms.len() * cfg.rate_grid_hz.len()
  #let segments-per-cell = cfg.n_grid_streams * cfg.n_per_stream * cfg.train_seeds.len()

  #let body = [
    #journal-abstract(body: [
    We asked whether a PING classifier trained on separate digits could operate on
    a continuous MNIST stream without retraining. We preserved hidden state across
    digit boundaries while varying duration and input rate, with a
    segment-matched decoder.

    Classification became informative when the stream supplied sufficient input
    over time, with losses under brief or weak presentations. This deprecated
    study motivated later variable-rate training and spike-count readout; it
    did not test blind boundary detection.
    ])

    == Results

    #with-result-sections[

      #result-card[
      === Variable-duration digit stream

      The label-to-prediction pairs were #varying-predictions, giving
      #varying-correct of #r.varying_headline.labels.len() correct segments. The
      #r.varying_headline.segments.at(0).at(0) ms, #p10.input_rate_hz Hz error
      occurred at a condition with #calc.round(100 * p10.accuracy, digits: 1)% mean
      accuracy, so this single error does not establish an encoding-rate failure
      floor (#result-figure-ref(<fig:exp048-result-1>)).

  #figure(
        report-image(
          data-file("exp048/varying_headline_stream.png"),
          trim-bottom: 3%,
          alt: "A digit stream where each segment has its own duration and input rate, with errors marked in red.",
        ),
        caption: [Historical illustrative classification when presentation duration and encoding rate vary
          between segments. The segment conditions are #varying-conditions.
          *(A)* Input thumbnails, whose opacity increases with encoding rate;
          *(B)* E-neuron spikes; *(C)* I-neuron spikes; and *(D)* class
          probability against time, with the true class emphasized in red.
          The raster panels show sampled neurons in rank order; their endpoint labels
          denote population sizes, not the number of displayed neurons.],
      ) <fig:exp048-result-1>

      ]

      #result-card[
      === Duration-rate accuracy

      Accuracy remained at the empty-input floor through #p05.input_rate_hz Hz,
      became informative by #p2.input_rate_hz Hz, and reached
      #calc.round(100 * p5.accuracy, digits: 1)% at #p5.input_rate_hz Hz (#result-figure-ref(<fig:exp048-result-2>)).

  #figure(
        report-image(
          data-file("exp048/acc_grid_tau_rate.png"),
          trim-bottom: 6.7%,
          alt: "A duration-by-input-rate accuracy heatmap beside a fixed-duration encoding-rate psychometric curve.",
        ),
        caption: [Temporal and encoding-rate limits of the frozen PING classifier.
          *(A)* Per-segment accuracy (%) is shown for presentation duration (ms,
          horizontal) and Poisson encoding rate (Hz per channel, vertical), using
          #segments-per-cell segments per cell across #cfg.train_seeds.len() seeds.
          *(B)* Probability of a correct
          classification (%) is plotted against encoding rate (Hz) with presentation
          and readout fixed at #r.encoding_rate_psychometric.presentation_ms ms.
          Points and the shaded band summarize seeds (mean ±1 standard error). The
          inset enlarges the linear
          0–#p10.input_rate_hz
          Hz interval without changing the axis scale. The
          dashed line marks #(cfg.n_classes)-class chance and the dotted line the
          #r.encoding_rate_psychometric.trained_rate_hz Hz training rate. The
          empty-input floor is the accuracy obtained when almost no input spikes arrive.],
      ) <fig:exp048-result-2>


      ]
    ]

    #journal-methods(
      orientation: [
    Reused seed-level measurements and illustrative figures described continuous
    digit classification with frozen PING weights. The documented protocol below
    specifies the evaluation and decoder; missing raw recordings prevent prediction
    replay and independent confirmation of historical model identities.
      ],
      compute: [
    + *Select classifiers and digits.* MNIST handwritten digits #cite(1) were
      sampled from the official test partition, separately from training and
      validation. The protocol selected the best validation epoch from each of
      #cfg.train_seeds.len() independently trained networks (seeds
      #cfg.train_seeds.map(str).join(", ")), each with #cfg.n_e excitatory (E) and
      #cfg.n_i inhibitory (I) neurons, trained on #cfg.trained_t_ms ms presentations
      at #cfg.input_rate_hz Hz. Illustrative streams used seed #cfg.seed and
      distinct digit classes; population estimates used all seeds.

    + *Encode and concatenate.* Independent Bernoulli draws at each
      $Delta t_"sim" = #cfg.dt$ ms timestep approximated Poisson input across #cfg.n_in pixel
      channels; firing probability was pixel intensity times encoding rate times
      $Delta t_"sim"/1000$. Encoding rates referred to full-intensity pixels, with lower
      intensities scaling them proportionally. Digits were concatenated with no
      gaps; rate and duration changed at known boundaries while hidden state
      continued without resetting within the stream.

    + *Vary evidence.* Crossed presentation durations and encoding rates defined
      #grid-cells conditions, each evaluated using #cfg.n_grid_streams streams of
      #cfg.n_per_stream digits per seed. Lower-rate evaluations used
      #r.encoding_rate_psychometric.new_streams_per_seed streams of
      #r.encoding_rate_psychometric.digits_per_stream digits per seed at fixed
      #r.encoding_rate_psychometric.presentation_ms ms; higher-rate points reused
      that duration's grid measurements. #link(<sec-conditions-and-decoder-identities>)[Conditions and decoder identities] gives the complete grids and
      the separate constant-rate versus rate-compensated duration comparison.
      ],
      analyse: [
    #set enum(start: 4)

    + *Integrate output evidence.* E-neuron spikes drove a non-spiking leaky
      integrator with a one-timestep delay and zero initial state:

      $ u_"out"[k] = beta_"out" u_"out"[k-1] + (1 - beta_"out") / (Delta t_"sim") bold(s)^E[k-1] W_"out". $

      Here $k$ indexes timesteps from zero, u#sub[out] is the dimensionless class-state vector,
      s#super[E] the dimensionless E-spike vector, W#sub[out] the trained output weights, and
      β#sub[out] the leak factor. The decoder used the trained output time constant
      τ#sub[out], defaulting to 2 ms when unspecified; its historical value was
      not independently confirmed.
      ],
      present: [
    #set enum(start: 5)

    + *Read and score segments.* A trailing mean used the current presentation
      duration T#sub[present], with w timesteps and w#sub[k] available during startup:

      $ z[k] = 1 / w_k sum_(j=k-w_k+1)^(k) u_"out"[j], quad w_k = min(w, k+1). $

      Here $j$ indexes the window and $z[k]$ is the class-score vector. At each
      known segment endpoint the predicted class $hat(y)$ was

      $ hat(y)[k] = arg max_c p_"class"(c,k). $

      Here $c$ indexes classes and $p_"class"$ is softmax-normalized evidence, not calibrated
      confidence; quantitative sweeps selected scores directly. Readout duration
      matched presentation duration rather than varying independently, unlike the
      whole-trial average used in training. Correct predictions were counted per
      seed; captions report across-seed means and sample standard errors.
      ],
    )
    #run-view("exp048", inputs)

    == Appendix: Conditions and decoder identities <sec-conditions-and-decoder-identities>

    The grid crossed durations
    #cfg.tau_grid_ms.map(x => str(x) + " ms").join(", ") with encoding rates
    #cfg.rate_grid_hz.map(x => str(x) + " Hz").join(", ") per channel.
    Each duration–rate condition contained #segments-per-cell classified presentations across
    #cfg.train_seeds.len() seeds. The low-rate extension used
    #r.encoding_rate_psychometric.new_rates_hz.map(x => str(x) + " Hz").join(", ")
    with presentation and readout fixed at
    #r.encoding_rate_psychometric.presentation_ms ms.

    The separate duration comparison used
    #cfg.tau_sweep_ms.map(x => str(x) + " ms").join(", "), with
    #cfg.n_streams streams of #cfg.n_per_stream digits per seed. Constant encoding
    remained at #cfg.input_rate_hz Hz; the compensated rate multiplied this by
    #cfg.trained_t_ms ms divided by the current duration. The fixed illustrative
    stream contained #cfg.n_digits_headline digits at #cfg.tau_headline_ms ms and
    #cfg.input_rate_hz Hz. The variable conditions appear in Figure 1.

    For a constant-duration stream, the boundary time t#sub[k] of segment k was

    $ t_j = j T_"present". $

    Here $j$ indexes segments from zero and $t_j$ is physical boundary time; varying-duration boundaries instead summed
    preceding durations. The output probabilities were

    $ p_"class"(c,k) = "softmax"(z[k])_c. $

    Softmax converted the evidence vector into non-negative values summing to one.
    The fixed illustrative stream selected its softmax maximum; the varying stream
    and sweeps selected the maximum logit. Matched readout and presentation obeyed

    $ T_"readout" = T_"present". $

    Here T#sub[readout] and T#sub[present] are the respective durations in ms.
    The window length was

    $ w = T_"present" / (Delta t_"sim"). $

    At intermediate times the trailing window could include preceding-segment
    activity; at the scored endpoint it covered the current segment. This
    duration-matched decoder used known boundaries, not inferred boundary detection.
    The output leak was

    $ beta_"out" = exp(-Delta t_"sim" \/ tau_"out"). $

    The exponential factor retained part of the previous state at each timestep;
    τ#sub[out] set the evidence-decay timescale. Equation (1) applied from timestep
    one onward, with u#sub[out] [0] = 0. The window average divided by its available
    length during startup, as stated in Equation (2).

    #journal-references((
      (
        text: [Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner.
          “Gradient-based learning applied to document recognition.”
          _Proceedings of the IEEE_ 86(11), 2278–2324 (1998).],
        doi: "10.1109/5.726791",
      ),
    ))

  ]
  #body
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file,
    inputs,
    [How do presentation duration and input rate affect streaming classification? Compare continuous-input performance across durations and encoding rates.],
    preview-figures,
    json-inputs: ("exp048",),
  )
}

#let meta = meta + (assets: input-assets("exp048", inputs))
#let body = journal-article("exp048", inputs, report-body, dataset-placed: inputs-ready(data-file, inputs))
