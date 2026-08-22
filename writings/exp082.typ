#let meta = (
  title: "Variable-rate streaming with a spike-count readout",
  date: "2026-08-10",
  description: "Successor to exp048 using the variable-rate PING training bank and a readout matched to each presentation window.",
  collection: "gamma-gated-sparsity",
  status: "draft",
)

#let r = json("/artifacts/data/exp082/numbers.json")
#let provisional = (
  r.profile != "production"
  or r.config.digits_per_seed_cell < 200
  or r.config.psychometric_rates_hz.len() < r.config.training_rates_hz.len()
)

#let body = [
  == Abstract

  This experiment tests whether PING networks trained across input rates can classify continuous MNIST streams with the same spiking readout used during training. Excitatory spikes drive ten output LIF neurons, and each class logit is the corresponding neuron's total spike count within the current digit's presentation window. A variable-rate, variable-duration stream shows the online network response, while a 200 ms rate psychometric and a duration-by-rate map measure performance across the trained input distribution.

  == Methods

  === 1. Training source

  The experiment loads `ping__variable_rate__seed42`, `ping__variable_rate__seed43`, and `ping__variable_rate__seed44` from exp022. Each PING network is trained by sampling one maximum-pixel Poisson rate independently per image presentation from 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, and 25 Hz. The recurrent weights and classifier are frozen throughout this experiment.

  *Note.* The exp022 cells are registered and `tools/snn` implements the required output-LIF `spike-count` training, restore, and output-raster path. Presentation boundaries reset only the output LIF and its counter; the hidden PING state remains continuous. This entry still fails loudly if the variable-rate checkpoint bank is absent or uses another readout.

  Streaming inference uses each seed's checkpoint selected by validation accuracy, because this is a deployment evaluation of the selected classifier rather than an analysis of final-epoch drift.

  === 2. Spike-count readout

  At each timestep, the 1024-element excitatory spike vector $bold(s)^E(t)$ drives a trained 1024-by-10 projection $W_"out"$ into ten output LIF neurons. For a presentation beginning at timestep $a$ and ending before timestep $b$, class evidence is

  $ z_c = sum_(t=a)^(b-1) s_c^"out"(t), quad c in {0, dots, 9}. quad "(1)" $

  Here $s_c^"out"(t)$ is the spike emitted by output LIF neuron $c$ at timestep $t$, and $bold(z)$ contains the ten dimensionless output spike-count logits. The predicted digit is the index of the largest count. Dividing counts by the common presentation duration gives firing rates for reporting and leaves this argmax unchanged, but rates are not passed to cross-entropy as logits. The output neurons therefore have membrane state, leak, threshold, spikes, and reset; unlike `mem-mean`, their membrane voltages are not the logits.

  === 3. Variable-rate and variable-duration inference

  Every digit is read from only the spikes generated during its own presentation:

  $ T_"readout" = T_"presentation". quad "(2)" $

  Here $T_"readout"$ is the spike-summation window and $T_"presentation"$ is the duration of the corresponding digit. The five-digit headline stream changes both presentation duration and maximum-pixel Poisson rate at digit boundaries. A factorial evaluation then crosses presentation durations 25, 50, 100, and 200 ms with the eleven training rates. Every cell is evaluated on 200 classified digits independently for seeds 42, 43, and 44. Independent five-digit streams are evaluated in simulator batches; each batch member has separate neuronal state, while hidden state remains continuous only between the five digits belonging to that stream. Statistical grid cells retain compact per-presentation spike counts and population totals. Full rasters are retained for the illustrative stream and an internal matched-condition validation stream.

  === 4. Psychometric protocol

  The rate psychometric fixes presentation and readout duration at 200 ms, varies the maximum-pixel rate across the eleven training values, and reports held-out classification accuracy across the three trained seeds. This isolates rate robustness from shortened evidence windows.

  == Results

  #if provisional [
    #block(inset: 10pt, fill: rgb("f3f0e8"), radius: 3pt)[
      *Provisional validation output.* These figures verify the complete inference, measurement, and rendering path. Each seed-level grid cell contains only #r.config.digits_per_seed_cell classified digits, and the checkpoints come from reduced training runs, so the numerical values are not estimates of publication performance.
    ]
  ]

  === 1. Streaming classification and temporal evidence

  The variable stream changes both input rate and presentation duration at digit boundaries. Each decision still uses exactly the spikes emitted during that digit, so evidence never leaks across labels.

  #figure(
    image("/artifacts/data/exp082/variable_stream.png", width: 100%, alt: "Excitatory and inhibitory rasters with online spike-count class evidence as input rate and digit duration vary."),
    caption: [Successor to exp048 Figure 1, using the variable-rate-trained PING network and its native output-LIF spike-count readout. Presentation duration and maximum-pixel input rate vary between segments. Thumbnail opacity increases with encoding rate; badges show true label to prediction, with errors in red. The middle panels plot E- and I-cell spike rasters against time (ms). The lower panel plots the softmax of cumulative output spike counts within each segment, with the true-class trace emphasized in red. Counts reset at each boundary while hidden PING state remains continuous.],
  )

  The factorial summary separates the two manipulations. The left panel crosses presentation duration with maximum-pixel input rate. The right panel holds presentation and readout at 200 ms, giving the rate psychometric without a simultaneous change in integration time.

  #figure(
    image("/artifacts/data/exp082/duration_rate_summary.png", width: 100%, alt: "Accuracy map over presentation duration and input rate beside the 200 millisecond input-rate psychometric curve."),
    caption: [Accuracy across matched presentation/readout durations and input rates, with the 200 ms rate psychometric shown separately.],
  )

  #figure(
    image("/artifacts/data/exp082/psychometric_200ms.svg", width: 85%, alt: "Classification accuracy versus maximum-pixel input rate at 200 milliseconds."),
    caption: [Input-rate psychometric at a fixed 200 ms presentation and spike-count window.],
  )

  === 2. Comparison with exp048

  Exp048 used a fixed-25-Hz training distribution and reconstructed a sliding `mem-mean` output. This experiment instead trains across the evaluated rates and uses the trained output-LIF spike counts directly. A numerical difference between the experiments therefore combines a change in training distribution with a change in readout; it is not an isolated comparison of either intervention.
]
