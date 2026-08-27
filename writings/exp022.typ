#import "/.demolab/lib.typ": data-json, data-image, cite, reference-list
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#let data-file = data-file.with(article: "exp022")

#let meta = (
  status: "Results available",
  title: "Training Runs",
  date: "2026-08-11",
  description: "Seven controlled training families, their retained checkpoint bank, validation learning curves, and raster diagnostics.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp022",)

// Do not evaluate report data or claims when no presentation input is selected.
#let render-report(data-file) = [
#let r = data-json(data-file("exp022/numbers.json"))
#let family-rows(family, model: none) = r.cells.filter(
  cell => cell.family == family and (model == none or cell.model == model),
)
#let mean-field(family, field, model: none) = {
  let values = family-rows(family, model: model).map(cell => cell.at(field))
  calc.round(values.sum() / values.len(), digits: 1)
}
#let range-field(family, field) = {
  let values = family-rows(family).map(cell => cell.at(field))
  [#calc.round(calc.min(..values), digits: 1)–#calc.round(calc.max(..values), digits: 1)]
}
#let family-coverage(family) = {
  let status = r.family_status.at(family)
  [#status.trained of #status.cells registered cells have retained training results.]
}
#let study-names = (
  exp024: "the convergence audit", exp025: "the accuracy–rate frontier",
  exp037: "the spike-loss and noise comparison", exp038: "the inhibitory-loop intervention",
  exp041: "the frequency–rate comparison", exp042: "the gamma-disruption study",
  exp044: "the timestep audit", exp046: "the spikes-per-cycle study",
  exp048: "the duration and input-rate comparison", exp049: "the recurrent-training study",
  exp082: "the continuous-stream classification study",
)
#let run-links(items) = items.map(item => link(item + ".html")[#study-names.at(item)]).join([, ])
#let figure-lineage(path) = {
  let records = r.at("presentation_lineage", default: ())
    .filter(item => item.file == path.split("/").last())
  if records.len() == 0 {
    [The source of this figure is not specified.]
  } else if records.first().operation.starts-with("carry") {
    [Previously recorded diagnostic, reused without a new simulation.]
  } else {
    [Based on retained measurements.]
  }
}
#let result-figure(path, alt, caption) = figure(
  data-image(data-file(path), width: 100%, alt: alt),
  caption: [#caption #figure-lineage(path)],
)
#let divider() = context {
  if target() == "html" {
    html.elem("hr", attrs: (style: "margin: 2.75rem 0;"))
  } else {
    v(1.1em)
    line(length: 100%, stroke: 0.6pt + luma(72%))
    v(1.1em)
  }
}

#let body = [
  == Contents

  + #link("#abstract")[Abstract]
  + #link("#results")[Results: TR-01 through TR-07]
  + #link("#methods")[Methods]
  + #link("#appendix-a-training-run-specification-sheets")[Appendix A: shared and TR-specific specification sheets]
  + #link("#references")[References]

  == Abstract

  #let abstract-condition-count = r.cells.map(cell => (cell.family, cell.model, cell.tag)).dedup().len()
  #let abstract-seed-count = r.cells.map(cell => cell.seed).dedup().len()

  We assembled a reusable bank of #r.n_cells spiking networks for MNIST handwritten-digit classification, covering #abstract-condition-count conditions with #abstract-seed-count random seeds each. Training lasted #r.standard.epochs epochs per network. Conditions compared feedforward controls with excitatory–inhibitory recurrent networks and varied activity penalties, inhibitory decay, numerical timestep, recurrent initialization and trainability, and input drive. In the baseline comparison, mean validation accuracy at selected checkpoints was #mean-field("canonical", "acc", model: "coba")% for feedforward networks and #mean-field("canonical", "acc", model: "ping")% for recurrent networks. Their final-epoch excitatory firing rates were #mean-field("canonical", "rate_e", model: "coba") and #mean-field("canonical", "rate_e", model: "ping") Hz, respectively. Retained models and learning histories support subsequent experiments; these training-recipe comparisons do not isolate a causal benefit of gamma timing.

  == Results

  === 1. TR-01 — Canonical full-data reference

  #result-figure(
    "exp022/curves__canonical.svg",
    "Validation-accuracy learning curves for the canonical COBA and PING cells.",
    [Individual learning histories for both architectures and all three seeds in the full-data family; no across-seed averaging or uncertainty bands.],
  )
  #result-figure(
    "exp022/rasters__ping__canonical__seed42.png",
    "Seed-42 excitatory/inhibitory raster and population-rate diagnostic for canonical PING.",
    [Canonical PING, seed 42: one digit-0 diagnostic from the final-epoch checkpoint, with E/I population rates and the accompanying inhibitory spectrum; not a population estimate.],
  )

  === 2. TR-02 — Activity-ceiling sweep

  #result-figure(
    "exp022/curves__theta_u.svg",
    "Validation learning curves across the activity-ceiling conditions for COBA and PING.",
    [Individual histories for two architectures, six ceiling settings, and three seeds on the reduced training pool; line colours distinguish settings and line styles distinguish architectures. No uncertainty bands are shown.],
  )
  #result-figure(
    "exp022/rasters__ping__off__seed42.png",
    "Seed-42 raster for the unconstrained PING endpoint of the activity-ceiling sweep.",
    [PING with the activity penalty off, seed 42, final-epoch digit-0 probe. This is the reference endpoint, not a raster of the strictest ceiling.],
  )

  === 3. TR-03 — Inhibitory-timescale sweep

  #result-figure(
    "exp022/curves__tau_gaba.svg",
    "Validation learning histories across six inhibitory-decay settings.",
    [Individual PING histories for six GABA decay constants and three seeds per setting; the training pool and other recipe settings are fixed. No confidence bands are shown.],
  )
  #result-figure(
    "exp022/rasters__ping__tg6__seed42.png",
    "Seed-42 PING raster at the six-millisecond inhibitory-decay reference.",
    [Reference inhibitory decay of 6 ms, seed 42, final-epoch digit-0 probe; the plotted spectrum describes this example only.],
  )

  === 4. TR-04 — Integration-timestep sweep

  #result-figure(
    "exp022/curves__dt.svg",
    "Validation learning curves across five integration timesteps.",
    [Individual histories for five timesteps and three seeds per setting. Training and evaluation use each cell's own timestep at a fixed presentation duration; no across-seed bands are shown.],
  )
  #result-figure(
    "exp022/rasters__ping__dt0p1__seed42.png",
    "Seed-42 PING raster at the reference 0.1-millisecond timestep.",
    [Reference timestep of 0.1 ms, seed 42, final-epoch digit-0 probe. One reference raster cannot establish timestep convergence.],
  )

  === 5. TR-05 — Recurrent-initialization sweep

  #result-figure(
    "exp022/curves__init.svg",
    "Validation learning histories across frozen and trainable recurrent-loop conditions.",
    [Individual histories for four recurrent conditions and three seeds each. The frozen PING control and three trainable initializations share the feedforward recipe; no confidence bands are shown.],
  )
  #result-figure(
    "exp022/rasters__frozen_ping__seed42.png",
    "Seed-42 raster for the frozen recurrent PING control.",
    [Frozen recurrent PING control, seed 42, final-epoch digit-0 probe; this example does not describe the trainable conditions.],
  )

  === 6. TR-06 — Variable-rate streaming bank

  #result-figure(
    "exp022/curves__variable_rate.svg",
    "Validation learning histories for all three variable-rate spike-count cells.",
    [Three individual seed histories for the variable-rate spike-count recipe. Each validation presentation draws from the specified rate set; these curves do not separate accuracy by input rate and have no uncertainty bands.],
  )
  #result-figure(
    "exp022/rasters__ping__variable_rate__seed42.png",
    "Seed-42 excitatory/inhibitory raster for the variable-rate bank at a five-hertz input rate.",
    [Variable-rate PING, seed 42, final-epoch digit-0 probe at 5 Hz maximum-pixel input rate. This retained E/I diagnostic does not include output-neuron spikes or continuous-stream resets.],
  )

  === 7. TR-07 — Low-input recruitment sweep

  #result-figure(
    "exp022/curves__low_w_in.svg",
    "Validation learning curves for four initial input-coupling settings and three seeds each.",
    [Twelve individual histories under the fixed 1 Hz soft ceiling. Colours distinguish the four input-initialization means; no across-seed means or confidence bands are shown.],
  )
  #for (tag, coupling) in (("0p05", "0.05"), ("0p1", "0.1"), ("0p3", "0.3"), ("0p9", "0.9")) {
    result-figure(
      "exp022/rasters__ping__low_w_in__win" + tag + "__seed42.png",
      "Seed-42 E/I diagnostic for initial input-coupling parent mean " + coupling + ".",
      [Initial input-coupling parent mean #coupling, seed 42, final-epoch digit-0 probe at the same 1 Hz activity-ceiling target. This is one example per setting, not an across-seed statistic.],
    )
  }

  == Methods

  We trained spiking classifiers under controlled conditions, then analysed retained learning histories and reused diagnostic simulations.

  + *Prepare the data.* Stratified MNIST splits provided 54,000 training and 6,000 validation images for the baseline, and 6,300 and 700 for sweeps. The split seed was 42; the official test set was excluded from training and model selection.

  + *Build the networks.* Each network contained 1,024 excitatory neurons and ten spiking leaky integrate-and-fire outputs. Recurrent networks added feedback through 256 inhibitory neurons; feedforward controls disabled it. Input and output projections were trained, while recurrent projections were fixed except in designated conditions. Weights retained their excitatory or inhibitory sign.

  + *Vary the conditions.* Seven families covered the baseline, activity ceilings, inhibitory decay, timestep, recurrent initialization and trainability, and input drive. Each condition used initialization seeds 42–44; Appendix A lists the grids and shared settings.

  + *Present the images.* Pixels generated Poisson spikes over #r.standard.t_ms ms, normally with a maximum-pixel rate of 25 Hz and a 0.1 ms timestep. Variable-rate training sampled uniformly from eleven rates between 0.5 and 25 Hz; timestep conditions ranged from 0.05 to 1 ms.

  + *Calculate class scores.* Most networks used mean pre-reset output voltage. Variable-rate training used spike counts and smaller initial readout weights.

    #block(breakable: false)[
    $ z_"voltage" = 1 / N sum_(t=1)^N v(t), quad
      z_"count" = sum_(t=1)^N s(t). quad "(1)" $

    Each score applies to one output neuron and presentation. Here $t$ indexes the $N$ timesteps, $v(t)$ is pre-reset voltage, and $s(t)$ is a binary spike; voltage and both scores are dimensionless.
    ]

  + *Train the networks.* Training used #r.standard.epochs epochs of AdamW, learning rate 0.0004, batches of #r.standard.batch_size, zero weight decay, and gradient clipping at 1. Surrogate gradients approximated spike derivatives#cite(1); voltage-gradient damping was 1,000 for recurrent networks and 1 for controls. Activity-constrained conditions minimized

    $ L = L_"CE" + lambda lr(⟨max(0, r - r_"max")^2⟩)_"batch". quad "(2)" $

    $L$ is total loss and $L_"CE"$ is mean cross-entropy. Each presentation's mean excitatory firing rate $r$ and ceiling $r_"max"$ are in hertz. Brackets average the individual penalties across the batch; $lambda = 0.041$ with rates in hertz, or zero when disabled.

  + *Select models.* Validation averaged three fixed Poisson encoding draws. Selection minimized mean cross-entropy, breaking ties by higher accuracy and then earlier epoch. Selected and final-epoch models were retained.

  + *Measure activity and retain models.* Accuracy uses selected models, whereas firing rates average final-epoch validation measurements across images and encoding draws. Retained models and histories support subsequent experiments. Learning curves show individual validation histories; baseline summaries average three seeds. Reused seed-42 digit-zero rasters are individual probes, not across-seed estimates; no new diagnostic simulations were performed.

  == Appendix A: Training-run specification sheets

  The following shared sheet and all seven TR sheets specify the controlled recipes and their intended uses. The observed learning curves and diagnostic plots are in Results.

  === A.1. Shared production specification

  All runs map Poisson-encoded pixels through 1,024 excitatory neurons to ten spiking output LIF neurons. COBA and PING use the same input-weight and readout-weight initialization distributions. COBA disables recurrent E/I coupling; PING adds a $1024 arrow 256 arrow 1024$ E/I feedback loop and uses stronger backward-pass gradient damping to stabilize training through that recurrent path.

  These are specified production settings, not newly measured results. Unless a TR-specific sheet says otherwise, every cell uses this contract. In the tables, $cal(N)(mu, sigma^2)$ denotes a normal distribution with mean $mu$ and standard deviation $sigma$; $max(0, x)$ replaces a sampled weight $x$ by zero when negative. LIF means leaky integrate-and-fire.

  #table(
    columns: (1.25fr, 1.3fr, 2.2fr),
    align: (left, left, left),
    table.header([*Parameter*], [*Default*], [*Meaning*]),
    [Dataset], [MNIST], [784 normalized pixels encoded as independent Poisson channels],
    [Presentation duration], [200 ms], [One static digit per training presentation],
    [Integration timestep], [0.1 ms], [2,000 recurrent updates per presentation],
    [Epochs], [50], [Training horizon for every production cell],
    [Seeds], [42, 43, 44], [Three independently initialized cells per configuration],
    [Minibatch], [256], [Presentations per optimization step],
    [Optimizer learning rate], [$4 times 10^(-4)$], [Shared learning rate],
    [Input population], [784], [One channel per image pixel],
    [Excitatory population], [1,024], [Learned stimulus representation],
    [Inhibitory population], [256], [PING feedback population; silent when E/I coupling is disabled],
    [$tau_"AMPA"$], [2 ms], [Fixed excitatory synaptic decay],
    [$tau_"GABA"$], [6 ms], [Default inhibitory decay at the gamma operating point],
    [Input initial-zero fraction], [0.95], [Fraction set to zero at initialization; every entry remains trainable and may regrow],
    [Input summed-coupling parent mean], [0.9], [Parent-Gaussian mean before lower clamping and fan-in normalization; shared by COBA and PING],
    [Readout-weight initialization], [$max(0, cal(N)(1.1206, 0.8350^2))$], [A directly stored lower-clamped Gaussian. Its zero-valued entries remain trainable; COBA and PING use the same initializer],
    [E/I loop strength], [COBA: 0; PING: 1], [The forward architectural difference under comparison],
    [Gradient damping], [COBA: 1; PING: 1,000], [Backward-pass stabilization for the recurrent PING loop; it does not change forward dynamics],
    [Surrogate slope], [1], [Spike-gradient surrogate parameter],
    [Default readout], [mean-voltage], [A *spiking* $1024 arrow 10$ output-LIF layer. At each timestep its pre-reset voltages enter the temporal mean, then emitted spikes subtract the threshold before the next update. These mean voltages—not spike counts—are the logits],
    [Stored projection shapes], [$784 times 1024$; $1024 times 256$; $256 times 1024$; $1024 times 10$], [Input→E, E→I, I→E, and E→class, in source-to-destination orientation],
  )

  === A.2. TR-01 — Canonical full-data reference

  The full-data COBA and PING cells are used for headline accuracy. They use the official MNIST training partition with no spike-budget penalty, so the comparison is not affected by the smaller dataset or regularization used in the sweeps. Ten percent of that partition is reserved for checkpoint selection; the official test partition remains untouched during training. This experiment reports their validation learning histories; independent official-test evaluation belongs to the downstream experiments.

  #table(
    columns: (1.2fr, 1.4fr, 2fr),
    table.header([*Key parameter*], [*Value*], [*Why it differs*]),
    [Architectures], [COBA and PING], [Provides a feedforward control and recurrent E/I model],
    [Training pool], [60,000 official training samples], [54,000 optimizer-training and 6,000 validation samples],
    [Input rate], [25 Hz maximum-pixel rate], [Fixed-rate collection baseline],
    [Spike budget], [Off], [Measures unconstrained capacity],
    [Cells], [2 architectures × 3 seeds = 6], [Across-seed comparison for both models],
    [Readout shape], [$1024 arrow 10$ spiking LIF outputs], [mean-voltage: mean membrane voltage supplies the logits],
  )

  #divider()

  === A.3. TR-02 — Activity-ceiling sweep

  This run measures the trade-off between accuracy and firing rate as the activity ceiling is tightened. For each presentation, the loss computes the population-mean hidden-E firing rate, applies a one-sided quadratic penalty above the target, then averages the penalties across the minibatch. Quiet presentations therefore cannot offset active presentations. Expressing the target in hertz and normalising over samples, neurons, duration, and hidden layers makes the intervention comparable across those dimensions. A smaller training pool keeps the multi-seed sweep manageable; only the activity target changes across conditions. Comparisons with the full-data cells also change training-pool size and cannot isolate the penalty alone. The resulting checkpoints are used by #run-links(("exp024", "exp025", "exp037", "exp038")).

  #table(
    columns: (1.2fr, 1.5fr, 2fr),
    table.header([*Key parameter*], [*Value*], [*Why it differs*]),
    [Architectures], [COBA and PING], [Direct architecture comparison],
    [Training pool], [7,000 official training samples], [6,300 optimizer-training and 700 validation samples],
    [Hidden-E rate target $r_"max"$], [off, 25, 10, 5, 2.5, 1 Hz], [Spans unconstrained through severe sparsity],
    [Penalty strength], [$0.041$ when enabled], [Calibrated for the sample-wise, population-normalized Hz objective],
    [Cells], [2 × 6 settings × 3 seeds = 36], [Error bars at every frontier point],
    [Readout shape], [$1024 arrow 10$ spiking LIF outputs], [mean-voltage: mean membrane voltage supplies the logits],
  )

  #divider()

  === A.4. TR-03 — Inhibitory-timescale sweep

  This run changes $tau_"GABA"$ to test how the inhibitory timescale affects gamma frequency, firing rate, and accuracy. All other scientific settings are held fixed, with 6 ms as the standard condition. The resulting checkpoints are used by #run-links(("exp041", "exp042", "exp046")).

  #table(
    columns: (1.2fr, 1.5fr, 2fr),
    table.header([*Key parameter*], [*Value*], [*Why it differs*]),
    [Architecture], [PING], [The manipulated quantity belongs to the recurrent inhibitory loop],
    [Training pool], [7,000 samples], [Sweep-scale default],
    [$tau_"GABA"$], [4.5, 6, 9, 12, 18, 27 ms], [Moves the inhibitory rhythm across a broad timescale range],
    [Cells], [6 settings × 3 seeds = 18], [Across-seed estimate at each decay],
    [Readout shape], [$1024 arrow 10$ spiking LIF outputs], [mean-voltage: mean membrane voltage supplies the logits],
  )

  #divider()

  === A.5. TR-04 — Integration-timestep sweep

  This run changes the integration timestep while keeping each presentation at 200 ms. It tests whether the observed dynamics depend on numerical resolution and measures the extra compute required by finer timesteps. The finest timestep requires the longest unrolled training trajectories. The resulting checkpoints are used by #run-links(("exp044",)).

  #table(
    columns: (1.2fr, 1.5fr, 2fr),
    table.header([*Key parameter*], [*Value*], [*Why it differs*]),
    [Architecture], [PING], [Tests the recurrent reference model],
    [Training pool], [7,000 samples], [Sweep-scale default],
    [$Delta t$], [0.05, 0.1, 0.25, 0.5, 1 ms], [Changes numerical resolution while holding 200 ms physical time fixed],
    [Steps/presentation], [4,000; 2,000; 800; 400; 200], [Compute and activation memory scale inversely with $Delta t$],
    [Cells], [5 settings × 3 seeds = 15], [Across-seed stability check],
    [Readout shape], [$1024 arrow 10$ spiking LIF outputs], [mean-voltage: mean membrane voltage supplies the logits],
  )

  #divider()

  === A.6. TR-05 — Recurrent-initialization sweep

  This run tests whether training preserves the PING loop or learns a useful loop from weaker initial conditions. Recurrent initialization and trainability change together, while the feedforward network and classifier remain fixed to the PING recipe. The resulting checkpoints are used by #run-links(("exp049",)).

  #table(
    columns: (1.2fr, 1.65fr, 2fr),
    table.header([*Key parameter*], [*Value*], [*Why it differs*]),
    [Architecture], [PING], [Manipulates the recurrent loop directly],
    [Training pool], [7,000 samples], [Sweep-scale default],
    [Loop conditions], [frozen PING; trainable PING init; trainable zero init; trainable 0.1 init], [Separates built-in dynamics from recurrence learned during task training],
    [Trainable projections], [$W_"EI"$ and $W_"IE"$ only in trainable conditions], [The frozen condition is the mechanistic control],
    [Cells], [4 conditions × 3 seeds = 12], [Across-seed comparison],
    [Readout shape], [$1024 arrow 10$ spiking LIF outputs], [mean-voltage: mean membrane voltage supplies the logits],
  )

  #divider()

  === A.7. TR-06 — Variable-rate streaming bank

  This condition trains PING across the input rates used by the continuous-stream classification study. One rate is sampled uniformly for each presentation, and the ten output LIF neurons produce class logits from their total spike counts rather than their mean membrane voltages. This keeps the cross-entropy logits dimensionless and gives one additional output spike the same logit increment. The readout weights use a small Gaussian initialization, $cal(N)(0.05, 0.04^2)$, lower-clamped at zero and governed by the shared non-negative constraint.

  The recorded design rationale for this smaller initializer was to avoid saturation at the mean-voltage scale and silence at the generic fan-in-normalized scale. The present bank retains the chosen $cal(N)(0.05, 0.04^2)$ setting; its learning curves do not independently establish those earlier calibration comparisons. This is a specified spike-count recipe, not a claim that the initializer is uniquely optimal.

  Output firing rates may still be reported as activity measurements. During streaming inference, the output neurons reset at digit boundaries while the hidden PING state continues. The resulting checkpoints are used by #run-links(("exp082",)).

  #table(
    columns: (1.2fr, 1.65fr, 2fr),
    table.header([*Key parameter*], [*Value*], [*Why it differs*]),
    [Architecture], [PING], [Target model for streaming inference],
    [Training pool], [7,000 samples], [Uses the shared sweep-scale training set],
    [Input-rate set], [0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 25 Hz], [Denser sampling within the interval selected by the input-rate calibration study],
    [Sampling rule], [Uniform categorical, independently per presentation], [Makes rate variation part of the training distribution],
    [Readout], [spike-count], [Hidden E spikes drive ten spiking LIF class neurons; each logit is that class neuron's total spikes over the presentation],
    [Readout initialization], [$cal(N)(0.05, 0.04^2)$, constrained non-negative], [Keeps the spiking outputs near threshold at high training rates without the saturation caused by the default mean-voltage scale],
    [Readout shape], [$1024 arrow 10$ spiking LIF outputs], [Ten class neurons emit and reset throughout the presentation],
    [Cells], [1 recipe × 3 seeds = 3], [Models for continuous-stream classification],
  )

  #divider()

  === A.8. TR-07 — Low-input recruitment sweep

  TR-02 asks how changing the activity ceiling affects networks initialized at the
  standard input coupling. TR-07 asks the complementary question: with the strictest
  TR-02 ceiling fixed at 1 Hz from the first epoch, can PING recruit its inhibitory
  loop when the feedforward projection starts weak? It varies the parent mean used
  to initialize expected summed input coupling, before lower clamping and fan-in
  normalization; it does not directly set the mean of the stored synaptic matrix.
  The training pool, optimizer, recurrent loop, and mean-voltage readout remain at the
  reduced-sweep defaults. This bank retains all twelve models, and #run-links(("exp025",))
  aggregates the three seeds at each setting to test recruitment and path dependence.

  #table(
    columns: (1.2fr, 1.65fr, 2fr),
    table.header([*Key parameter*], [*Value*], [*Why it differs*]),
    [Architecture], [PING], [Tests recruitment of the recurrent inhibitory loop],
    [Training pool], [7,000 samples], [6,300 optimizer-training and 700 validation samples],
    [Epochs], [50], [Matches the reduced production standard],
    [Input summed-coupling parent mean], [0.05, 0.1, 0.3, 0.9], [Varies initial feedforward drive from weak coupling to the shared 0.9 standard before clamping and fan-in normalization],
    [Hidden-E rate target], [1 Hz], [Applies the strictest TR-02 activity ceiling from epoch 0],
    [Parameters held fixed], [PING loop, optimizer, dataset split, and mean-voltage readout], [Isolates initial feedforward recruitment from the activity-ceiling sweep],
    [Cells], [4 settings × 3 seeds = 12], [Across-seed estimate for every condition],
    [Readout shape], [$1024 arrow 10$ spiking LIF outputs], [mean-voltage: mean membrane voltage supplies the logits],
  )

  #reference-list((
    (text: [E. O. Neftci, H. Mostafa, and F. Zenke. “Surrogate Gradient Learning in Spiking Neural Networks: Bringing the Power of Gradient-Based Optimization to Spiking Neural Networks.” _IEEE Signal Processing Magazine_ 36(6), 51–63 (2019).], doi: "10.1109/MSP.2019.2931595"),
  ))
]
#body
]

#let body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(data-file, inputs, [], ())
}
