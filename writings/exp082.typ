#import "templates/article-layout.typ": journal-article
#import "templates/result-card.typ": journal-result-card, result-figure-ref, with-result-sections
#import "/.demolab/lib.typ": data-image, data-json
#import "templates/dataset.typ": data-file, input-assets, inputs-ready, pending-report, run-view
#import "templates/abstract.typ": journal-abstract
#import "templates/methods.typ": journal-methods, method-card
#let data-file = data-file.with(article: "exp082")

#let meta = (
  tags: ("data", "reviewed", "v36.0.0"),
  title: "Spike-Count Classification in a Continuous Stream",
  created_at: "2026-08-10T00:00:00Z",
  updated_at: "2026-09-10",
  description: "A multi-seed study of spike-count classification across input rates and presentation durations.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp082",)
#let preview-figures = (
  (path: "exp082/continuous_stream_compound.png", label: "continuous-stream capability and operating range"),
  (path: "exp082/single_trial.png", label: "single trial"),
  (path: "exp082/single_trial_transition.png", label: "single-trial transition"),
  (path: "exp082/variable_stream.png", label: "variable stream"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
  #let r = data-json(data-file("exp082/numbers.json"))
  #let pct(x) = str(calc.round(100 * x, digits: 1)) + "%"
  #let image-policy = r.config.at("image_stream_policy", default: none)
  #assert(image-policy in (none, "shared-across-training-seeds-durations-rates/v1"),
    message: "unsupported exp082 image-stream policy")
  #let shared-image-bank = image-policy != none
  #let mean(xs) = xs.sum() / xs.len()
  #let accuracy(duration, rate) = pct(mean(
    r
      .grid_per_seed
      .filter(
        row => row.duration_ms == duration and row.rate_hz == rate,
      )
      .map(row => row.accuracy),
  ))
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
  #journal-abstract(body: [
    Input calibration in
    #link("/exp080/")[exp080] — #link("/exp080/")[_Decoder Accuracy Improves with Input Rate_]
    motivated the rate range used to train a variable rate PING classifier in
    #link("/exp022/")[exp022] — #link("/exp022/")[_Training Runs._]
    We tested whether these networks could classify continuous digit streams
    while retaining hidden state. #if shared-image-bank [With image sequences
    matched across conditions, longer, stronger inputs improved accuracy.] else [
    Longer, stronger inputs improved accuracy.] One network classified five
    successive digits correctly despite changing durations and rates. Weak inputs
    exposed silent and incorrect responses. Decisions required supplied digit
    boundaries, and the selected successful example does not estimate reliability.
  ])

  == Results

  #with-result-sections[
    #journal-result-card(
      title: "Continuous classification and accuracy",
      observation: [
        One frozen PING network correctly classified five successive digits with
        varying presentation durations and input rates
        (#result-figure-ref(<fig:exp082-result-1>, panel: "A–D")). Hidden neuronal state continued
        between digits, while output state and counts reset at supplied boundaries.
        This example demonstrates capability, not reliability across arbitrary streams.
        We reused the first stream with five correct decisions from a predefined
        candidate sequence; the first candidate qualified.

        At 25 Hz, increasing presentation duration from 25 to 200 ms raised mean
        accuracy from #accuracy(25, 25) to #accuracy(200, 25)
        (#result-figure-ref(<fig:exp082-result-1>, panel: "E–F")). At 200 ms, increasing input
        rate from 0.5 to 25 Hz raised accuracy from #accuracy(200, 0.5) to
        #accuracy(200, 25). #if shared-image-bank [These comparisons used the same
        ordered images; variation from spike encoding and the small, fixed image
        sample remains.] else [Image samples were only partly paired across
        conditions, so these differences also include image-sampling variation.]
      ],
      visual: [#figure(
        report-image(
          "exp082/continuous_stream_compound.png",
          "Five correctly classified digits with varying durations and input rates, alongside mean accuracy across presentation duration and input rate and the 200-ms rate–accuracy curve.",
          ratio: 0.667,
        ),
        caption: [Seed-42 network presented with digits
          #r.hero_stream.labels.map(str).join(", ", last: " and "). Each segment is
          labelled with its duration and maximum-pixel input rate. *(A)* Input
          thumbnails with true→predicted labels.
          *(B)* Spikes from the first 200 excitatory neurons; *(C)* spikes from
          the first 64 inhibitory neurons; *(D)*
          softmax-normalized output-count shares. Red traces identify the true classes.
          Panels A–D reuse the selected illustrative recording, independently
          of the quantitative evaluation in E–F.
          *(E)* Accuracy across presentation duration and maximum-pixel
          input rate; *(F)* the 200-ms rate–accuracy curve. Values are means across
          three independently trained networks, each evaluated on
          #r.config.digits_per_seed_cell digit presentations per condition.
          #if shared-image-bank [The same 40 ordered five-image streams were
          used for every network and condition, with separate encoding draws.]
          E has no uncertainty intervals; F shows SEM across networks.],
      ) <fig:exp082-result-1>],
    )

    #journal-result-card(
      title: "Spike counts identify the digit",
      observation: [
        By the presentation’s end, the true class, digit
        #r.single_trial.labels.first(), had accumulated the largest output-spike
        count (#result-figure-ref(<fig:exp082-result-3>)). Intermediate leaders
        did not determine the decision.
      ],
      visual: [#figure(
        report-image(
          "exp082/single_trial.png",
          "A correctly classified digit with excitatory and inhibitory rasters and ten softmax count-share trajectories.",
        ),
        caption: [First correct digit in the seed-42 network’s matched 200-ms,
          5-Hz stream. *(A)* Spikes from the first 200 excitatory neurons; *(B)*
          spikes from the first 64 inhibitory neurons; *(C)* softmax-normalized
          output-count shares. Red identifies the true class. These shares are
          not calibrated probabilities. The predicted digit is the class with
          the largest share at the presentation’s end.],
      ) <fig:exp082-result-3>],
    )

    #journal-result-card(
      title: "Spikes change the displayed shares",
      observation: [
        Output counts stepped upward at spikes and remained constant between them
        (#result-figure-ref(<fig:exp082-result-4>)). Softmax normalization changed
        the displayed class shares at these increments; abrupt share changes need
        not reflect abrupt changes in the underlying network state.
      ],
      visual: [#figure(
        report-image(
          "exp082/single_trial_transition.png",
          "Output spikes, cumulative class counts and softmax count shares from 91.5 to 94.5 ms in the same digit presentation.",
          ratio: 0.70,
        ),
        caption: [Post-hoc enlargement of 91.5–94.5 ms from the digit-4
          presentation. *(A)* Output spikes; *(B)* cumulative class counts; *(C)*
          softmax-normalized count shares. Red identifies the true class.],
      ) <fig:exp082-result-4>],
    )

    #journal-result-card(
      title: "Weak inputs expose failure modes",
      observation: [
        A stream with varying durations and input rates produced
        #r.variable_stream.correct.sum() correct decisions
        (#result-figure-ref(<fig:exp082-result-5>)). The 200-ms, 0.5-Hz presentation
        produced no output spikes, whereas the 100-ms, 2-Hz presentation was
        misclassified despite output activity.
      ],
      visual: [#figure(
        report-image(
          "exp082/variable_stream.png",
          "Five digits with changing rates and durations: three correct predictions, a silent 0.5 Hz failure and a non-silent 2 Hz failure.",
        ),
        caption: [Seed-42 network under the duration–rate conditions labelled
          above each segment. *(A)* Input thumbnails; *(B)* excitatory spikes;
          *(C)* inhibitory spikes; *(D)* softmax-normalized output-count shares.
          Badges show true→predicted labels. Thumbnail opacity indicates relative
          input rate; population sampling matches Figure 1.],
      ) <fig:exp082-result-5>],
    )
  ]

  #journal-methods(body: (
    method-card([Input range and classifiers], [
      The filtered-input calibration in
      #link("/exp080/")[exp080] — #link("/exp080/")[_Decoder Accuracy Improves with Input Rate_]
      motivated the input-rate range. We reused three frozen networks, trained
      with seeds 42–44, from
      #link("/exp022/")[exp022] — #link("/exp022/")[_Training Runs._]
      Each contained 1,024 excitatory neurons, 256 inhibitory neurons and ten
      output leaky integrate-and-fire neurons, with learned input-to-excitatory
      and excitatory-to-output projections and fixed recurrent weights.
      Evaluation held all weights fixed; the revised sampling protocol did not
      require retraining.
    ]),
    method-card([Simulation and refractory periods], [
      We used a 0.1-ms integration timestep. Hidden excitatory and inhibitory
      neurons remained at reset for 1.2 and 0.6 ms after a spike, respectively,
      corresponding to 12 and six timesteps. Output neurons had no refractory
      period. #if shared-image-bank [The repeated evaluation specified these
      physical durations explicitly, preserving the refractory holds used by
      the earlier fixed-timestep measurements.]
    ]),
    method-card([Training and checkpoint selection], [
      Training used 6,300 optimization images and 700 validation images for 50
      epochs. Maximum-pixel input rates were sampled from 0.5, 0.75, 1, 1.5, 2,
      3, 5, 7.5, 10, 15 and 25 Hz. After each epoch, we evaluated the network on
      three independently spike-encoded versions of the validation images.
      Mean validation cross-entropy was
      $ L_"CE" = -1/(D N) sum_(d=1)^D sum_(i=1)^N log p_(i,y_i)^((d)). $ <eq-validation-ce>
      Here $N=700$ is the number of validation images, $D=3$ is the number of
      encoding draws, $y_i$ is the true class of image $i$, and $p_(i,y_i)^((d))$
      is its true-class softmax share for draw $d$, defined below. We selected
      the epoch with the lowest $L_"CE"$. If epochs tied, we chose the one with
      higher validation accuracy, then the earlier epoch.
    ]),
    method-card([Evaluation streams], [
      #if shared-image-bank [
        We repeated the quantitative evaluation using one prespecified bank of
        40 five-image streams from the official 10,000-image MNIST test partition.
        Image sampling used seed 82000. Within each stream, we sampled five
        images uniformly without replacement and without class stratification;
        labels could repeat and images could recur across streams. Every network,
        duration and input rate used the same ordered image indices, so both
        image identity and preceding-image order were paired across conditions.
      ] else [
        We sampled images from the official 10,000-image MNIST test partition
        separately by condition. The retained seed formula accidentally gave
        ten duration-rate pairs identical image streams while other conditions
        remained unpaired; condition comparisons therefore also contain image-sampling
        variation.
      ]
      We tested all eleven training rates at 25, 50, 100 and 200 ms
      (#result-figure-ref(<fig:exp082-result-1>, panel: "E–F")). Each
      duration–rate–network condition contained 40 five-digit streams, giving
      200 decisions. The 132 network–duration–rate conditions therefore yielded
      26,400 decisions, not 26,400 independently sampled images. Duration and rate
      were constant within each quantitative stream. Batches contained five
      streams with separate neuronal states.
    ]),
    ..if shared-image-bank { (method-card([Earlier evaluation and pairing], [
      The earlier evaluation sampled images separately by condition. Its seed
      formula mapped 44 duration–rate combinations onto only 34 image-sampling
      seeds per network: ten pairs shared identical streams while the other
      conditions remained unpaired. The current quantitative results replace
      that mixed design with a common image bank. Image selection and encoding
      seeds both changed; differences from the earlier estimates cannot be
      attributed to image pairing alone. We reused the previously selected
      capability recording independently of this repeated grid evaluation.
    ]),) } else { () },
    method-card([Input encoding], [
      Pixels generated independent Bernoulli spikes at 0.1-ms resolution. Spike
      probability was proportional to pixel intensity and the condition’s
      maximum-pixel input rate. #if shared-image-bank [Image selection and spike
      encoding used separate random generators. Each network–duration–rate–stream
      combination had a distinct encoding seed, giving 5,280 seeds across the
      quantitative grid. Each generator advanced across its stream's five
      presentations. Pairing therefore matched images and their order, not
      realized input spike trains.] Digits followed without gaps; segment labels give
      their durations and input rates (#result-figure-ref(<fig:exp082-result-1>, panel: "A")
      and #result-figure-ref(<fig:exp082-result-5>, panel: "A")).
    ]),
    method-card([State and decision boundaries], [
      Each stream began with freshly initialized hidden and output states.
      Hidden neuronal and synaptic state then persisted between digits within
      that stream.
      Output-neuron state and spike counts reset at every supplied digit
      boundary. The readout accumulated evidence over the full presentation
      (#result-figure-ref(<fig:exp082-result-1>, panel: "B–D") and
      #result-figure-ref(<fig:exp082-result-5>, panel: "B–D")).
    ]),
    method-card([Spike-count classification], [
      The cumulative score for class $c$ at timestep $k$ was
      $ z_c[k] = sum_(j=k_0)^k s_c[j], $ <eq-cumulative-count>
      where $k_0$ is the first timestep of the presentation and $s_c[j]$ equals
      1 when output neuron $c$ spikes at timestep $j$, and 0 otherwise. The
      displayed class share was
      $ p_c[k] = exp(z_c[k]) / (sum_(a=0)^9 exp(z_a[k])), $ <eq-count-share>
      where $a$ indexes the ten digit classes. These softmax shares are not
      calibrated probabilities.

      At the final timestep $k_"end"$, we predicted
      $
        hat(y) = arg max_(c in {0, dots, 9}) z_c[k_"end"]
        = arg max_(c in {0, dots, 9}) p_c[k_"end"],
      $ <eq-count-decision>
      where $hat(y)$ is the predicted digit: the class with the greatest final
      cumulative count, equivalently the largest final displayed share
      (#result-figure-ref(<fig:exp082-result-3>, panel: "C")). Ties selected
      the lowest class index, including class 0 when all outputs were silent.
    ]),
    method-card([Performance summaries], [
      We calculated accuracy for each duration–rate–network condition and
      summarized network-level measurements as means and SEM across three training
      replicates (#result-figure-ref(<fig:exp082-result-1>, panel: "E–F")).
      Accuracy included silent presentations and all other decisions. SEM was
      sample SD divided by the square root of three; it summarizes observed
      variation across networks and their encoding draws, not uncertainty across
      independent image banks. The 200-ms curve reuses the corresponding heatmap
      evaluations. #if shared-image-bank [One shared image bank and one encoding
      draw per network, condition and stream do not separately estimate
      image-sampling, encoding and training variability.]
    ]),
    method-card([Capability example], [
      We reused a seed-42 recording selected with the predefined sequence of
      duration–rate pairs:
      #r.showcase_selection.configuration.conditions.map(pair => "(" + str(pair.at(0)) + " ms, " + str(pair.at(1)) + " Hz)").join(", ", last: " and ").
      We fixed candidate order and digit-sampling and encoding seeds before
      inference. #result-figure-ref(<fig:exp082-result-1>) uses the first candidate
      achieving five correct decisions. This search used five distinct digit
      classes per candidate, unlike the quantitative image sampling, which
      allowed repeated labels. The first candidate, with image-selection seed
      820000 and encoding seed 830000, qualified. It was not drawn from the
      quantitative image bank.
    ]),
    method-card([Failure example], [
      We reused a separately specified seed-42 stream with pairs
      (200 ms, 0.5 Hz), (50 ms, 25 Hz), (100 ms, 2 Hz), (25 ms, 10 Hz) and
      (200 ms, 5 Hz). This example was not selected through the capability search
      (#result-figure-ref(<fig:exp082-result-5>)).
    ]),
    method-card([Readout close-up], [
      We selected the first correct digit from a separate 200-ms, 5-Hz stream
      (#result-figure-ref(<fig:exp082-result-3>)) and enlarged its 91.5–94.5-ms
      interval post hoc (#result-figure-ref(<fig:exp082-result-4>)).
      Equations @eq-cumulative-count and @eq-count-share define the displayed
      counts and shares.
    ]),
    method-card([Scope of inference], [
      We evaluated classification with known boundaries and continuing hidden
      state, without testing autonomous segmentation, a hidden-state-reset
      control, or gamma activity’s causal contribution to recognition.
      Presentation and readout windows changed together, so the duration
      comparison does not separate extra sensory exposure from extra integration
      time. The selected success and separately specified failure stream are
      illustrations, not estimates of the frequency of such outcomes.
    ]),
  ))

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
#let body = journal-article("exp082", inputs, report-body, dataset-placed: inputs-ready(data-file, inputs))
