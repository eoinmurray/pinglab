#import "/.demolab/lib.typ": data-json, data-image, cite, reference-list
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#let data-file = data-file.with(article: "exp022")

#let meta = (
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
#let run-links(items) = items.map(item => link(item + ".html")[#item]).join([, ])
#let figure-lineage(path) = {
  let records = r.at("presentation_lineage", default: ())
    .filter(item => item.file == path.split("/").last())
  if records.len() == 0 {
    [Per-figure lineage is not supplied here; consult the selected run's `run.json`.]
  } else if records.first().operation == "carry-historical" {
    [Copied unchanged from #raw(records.first().source_run); this presentation did not rerun the probe.]
  } else {
    [Rendered from retained analysis in #raw(records.first().source_run).]
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
  + #link("#design-scope")[Design Scope]
  + #link("#results")[Results: TR-01 through TR-07]
  + #link("#methods")[Methods]
  + #link("#appendix-a-training-run-specification-sheets")[Appendix A: shared and TR-specific specification sheets]
  + #link("#references")[References]

  == Abstract

  Feedforward and recurrent recipes produced different balances between classification accuracy and spiking activity. We compared a conductance-based feedforward control with pyramidal–interneuron network gamma (PING), an excitatory–inhibitory feedback model, across seven training families. The retained bank contains #r.n_cells cells with controlled changes to activity ceilings, inhibitory decay, numerical timestep, recurrent initialization, input rate, and initial input coupling. Validation learning histories and representative rasters document reusable checkpoints and expose differences between the trained recipes. They do not by themselves establish a causal benefit of gamma timing or performance on the official test partition.

  == Design Scope

  The experiment compared a feedforward conductance-based control (COBA) with an excitatory–inhibitory PING network on the same digit task. All recipes used 784 input channels, 1,024 excitatory cells, and ten spiking output cells; PING enabled a 256-cell inhibitory feedback population. Three initialization seeds were specified per condition. TR-01 used a 60,000-image pool; TR-02 through TR-07 used 7,000, each with a separate validation subset. Detailed settings and downstream uses are preserved in Appendix A.

  TR-02 varied the firing-rate ceiling from off to 1 Hz; TR-03 varied inhibitory decay from 4.5 to 27 ms; TR-04 varied timestep from 0.05 to 1 ms at fixed 200 ms duration. TR-05 varied both recurrent initialization and trainability, TR-06 sampled input rates from 0.5 to 25 Hz with spike-count decoding, and TR-07 varied initial input coupling from 0.05 to 0.9 under a fixed 1 Hz ceiling. These are separate controlled families, not a full factorial design. The architecture comparison also used different backward-pass gradient damping, and TR-06 changed readout and initialization together; neither comparison isolates a single mechanism. Streaming inference and perturbation tests belong to downstream experiments.

  == Results

  Each subsection retains a learning-curve figure and its representative diagnostic raster; TR-07 shows all four specified coupling examples. Curves show individual cells, not means with confidence bands. The numerical summaries below come from the selected `numbers.json`: accuracy is the value at the validation-selected checkpoint, while population rates come from the final retained epoch. They are not paired measurements at one checkpoint.

  *Metric and provenance boundary:* The curve artwork retains the historical axis label “test accuracy,” but the source configurations identify these epoch measurements as validation evaluations over three fixed encoder draws. Read them as validation accuracy, not untouched official-test performance. Captions distinguish curves rendered from saved analysis from historical rasters copied into this presentation. A single raster is an illustration of one probe, not an across-seed estimate or evidence of newly executed simulation.

  === 1. TR-01 — Canonical full-data reference

  The full-data comparison asked how the two unrestricted recipes learned before imposing an activity ceiling. If recurrent inhibition reduced excitatory activity, PING should use fewer spikes, but it was not assumed to preserve the control's accuracy. #family-coverage("canonical")

  #result-figure(
    "exp022/curves__canonical.svg",
    "Validation-accuracy learning curves for the canonical COBA and PING cells; the retained axis label says test accuracy.",
    [Individual learning histories for both architectures and all three seeds in the full-data family; no across-seed averaging or uncertainty bands.],
  )
  #result-figure(
    "exp022/rasters__ping__canonical__seed42.png",
    "Seed-42 excitatory/inhibitory raster and population-rate diagnostic for canonical PING.",
    [Canonical PING, seed 42: one digit-0 diagnostic from the final-epoch checkpoint, with E/I population rates and the accompanying inhibitory spectrum; not a population estimate.],
  )

  Mean checkpoint-selection accuracy was #mean-field("canonical", "acc", model: "coba")% for COBA and #mean-field("canonical", "acc", model: "ping")% for PING. Final-epoch excitatory rates averaged #mean-field("canonical", "rate_e", model: "coba") Hz and #mean-field("canonical", "rate_e", model: "ping") Hz respectively. The lower activity is consistent with inhibitory suppression, but the lower accuracy and differing gradient damping prevent a claim of cost-free or uniquely rhythm-driven improvement.

  === 2. TR-02 — Activity-ceiling sweep

  This family asked how a soft activity penalty traded classification performance against firing rate. Tighter ceilings were expected to reduce activity, with accuracy potentially deteriorating when the penalty constrained useful spikes. #family-coverage("activity_frontier")

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

  Retained checkpoint-selection accuracies span #range-field("activity_frontier", "acc")%, and final-epoch excitatory rates span #range-field("activity_frontier", "rate_e") Hz. This spread is consistent with a performance–activity trade-off; the overview curves alone do not identify a matched-accuracy frontier. The ceiling is a soft loss term, not a guarantee that measured rate stays below its target. #link("/exp025/")[Exp025] makes the condition-matched comparison.

  === 3. TR-03 — Inhibitory-timescale sweep

  Changing inhibitory decay asked whether a wider timing interval altered learning and the trained network's activity. Slower recovery from inhibition was expected to change recruitment and rhythm, but an accuracy curve alone cannot measure gamma frequency. #family-coverage("tau_gaba")

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

  Checkpoint-selection accuracy spans #range-field("tau_gaba", "acc")%, and final-epoch excitatory rate spans #range-field("tau_gaba", "rate_e") Hz. The family therefore changes more than a visually labelled oscillation period. Condition-level frequency and timing interpretations require the dedicated measurements in #run-links(("exp041", "exp042", "exp046")), not extrapolation from the single reference raster.

  === 4. TR-04 — Integration-timestep sweep

  This family varied numerical resolution while holding each input presentation's physical duration fixed. A robust trained classifier should avoid large accuracy changes across the tested timesteps; that expectation does not imply convergence of every dynamical observable. #family-coverage("dt")

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

  Checkpoint-selection accuracy lies within #range-field("dt", "acc")%, while final-epoch excitatory rates span #range-field("dt", "rate_e") Hz. The relatively narrow accuracy interval supports stability of this task outcome within the tested family. The rate variation leaves a stronger claim of dynamical invariance unresolved; #link("/exp044/")[exp044] provides the focused comparison.

  === 5. TR-05 — Recurrent-initialization sweep

  Releasing recurrent weights asked whether learning maintained the built-in PING regime or found a different solution. Because both initial values and trainability differ across conditions, any outcome must be attributed to the complete condition rather than to initialization alone. #family-coverage("init")

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

  Checkpoint-selection accuracies span #range-field("init", "acc")%, but final-epoch excitatory rates span #range-field("init", "rate_e") Hz. Similar classification performance therefore does not establish similar activity or preservation of the original loop. #link("/exp049/")[Exp049] examines the learned weights and trajectories needed to distinguish those possibilities.

  === 6. TR-06 — Variable-rate streaming bank

  Training with randomly selected input rates asked whether one spike-count classifier could provide a reusable starting point across the streaming rate range. The expected outcome was a usable checkpoint bank, not proof of successful continuous-stream decoding. #family-coverage("variable_rate")

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

  Checkpoint-selection accuracy spans #range-field("variable_rate", "acc")%, with final-epoch excitatory rate spanning #range-field("variable_rate", "rate_e") Hz. The retained curves and raster replace the former pending-training placeholders, but their aggregate accuracy does not establish uniform performance at low and high rates. Per-rate decoding and continuing hidden-state tests belong to #link("/exp082/")[exp082].

  === 7. TR-07 — Low-input recruitment sweep

  The final family asked whether weak initial input coupling changed recruitment under the same strict activity ceiling. If the trained outcome retained a strong dependence on initialization, the four conditions could end at substantially different rates or accuracies. #family-coverage("low_w_in")

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

  Across the selected cells, checkpoint-selection accuracy spans #range-field("low_w_in", "acc")% and final-epoch excitatory rate spans #range-field("low_w_in", "rate_e") Hz. The narrow endpoint ranges do not suggest a large persistent difference in these two outcomes across the tested starts. They do not establish identical learning paths or loop recruitment; #link("/exp025/")[exp025] supplies the targeted analysis. The measured rates also illustrate that the 1 Hz penalty target is not a hard constraint.

  == Methods

  We compared feedforward and recurrent spiking classifiers by training seven families of conditions, evaluating classification accuracy, and measuring population activity. The procedure below describes the input encoding, readouts, and training objective. Appendix A gives the shared and TR-specific settings.

  + *Select the data and conditions.* Training used the official MNIST handwritten-digit training partition, reserving ten percent of each selected pool for validation. Each condition used three initialization seeds, 42–44. The seven families varied the architecture or training settings listed in Appendix A. The official test partition was not used for model selection.

  + *Encode and simulate each digit.* Pixels drove independent Poisson input channels for the specified presentation duration. The feedforward control or recurrent excitatory/inhibitory (E/I) network was advanced at the condition's numerical timestep. TR-06 sampled one input rate per presentation uniformly from its specified rate set; other families used the fixed baseline rate.

    $ S_(b i n)^"in" tilde "Bernoulli"(x_(b i) r_b Delta t). quad "(1)" $

    Here $b$ indexes a digit presentation, $i$ an input pixel, and $n$ a numerical timestep. $S_(b i n)^"in"$ is the binary input spike, $x_(b i) in [0, 1]$ is normalized pixel intensity, $r_b$ is the maximum-pixel input rate in hertz for that presentation, and $Delta t$ is the timestep in seconds. Independent draws at each pixel and timestep give the discrete-time Poisson approximation, with spike probability $x_(b i) r_b Delta t$.

  + *Construct class scores.* Input-to-E and E-to-output projections were trained; recurrent projections were trainable only in the designated TR-05 conditions. The default readout averaged pre-reset output membrane voltages over the presentation, whereas TR-06 used output spike counts. These quantities supplied the logits, the class scores before softmax normalization. Both readouts used ten spiking leaky integrate-and-fire (LIF) output neurons. Appendix A specifies the architecture, weight constraints, and readout initializations.

    $ z_(b k)^"mem" = 1 / N_t sum_(n=1)^(N_t) u_(b k n)^"pre", quad
      z_(b k)^"count" = sum_(n=1)^(N_t) S_(b k n)^"out". quad "(2)" $

    Here $k$ indexes one of the ten output classes, $N_t$ is the number of timesteps in a presentation, $u_(b k n)^"pre"$ is the output neuron's membrane state before spike reset, and $S_(b k n)^"out"$ is its binary spike. The output membrane state uses a dimensionless scale with threshold one. Thus both the mean-membrane score $z_(b k)^"mem"$ and spike-count score $z_(b k)^"count"$ are dimensionless; the selected readout supplies $z_(b k)$ to the classification objective.

  + *Optimize classification with an activity penalty.* Training used surrogate derivatives to approximate gradients through discontinuous spike events, with the condition's gradient damping.#cite(1) Classification used the mean cross-entropy of the class scores:

    $ p_(b k) = exp(z_(b k)) / (sum_(ell=1)^K exp(z_(b ell))), quad
      L_"CE" = -1 / B sum_(b=1)^B ln(p_(b,y_b)). quad "(3)" $

    Here $B$ is minibatch size, $K = 10$ is the number of classes, $ell$ indexes the classes in the normalization, $p_(b k)$ is the predicted probability of class $k$, and $y_b$ is the correct class for presentation $b$. $ln$ is the natural logarithm, so the dimensionless loss $L_"CE"$ is measured in nats per presentation. For the single hidden E population, the objective combined this loss with a penalty on each presentation's mean firing rate:

    $ nu_b = 1 / (N_E T) sum_(j=1)^(N_E) C_(b j), quad
      L = L_"CE" + lambda / B sum_(b=1)^B max(0, nu_b - nu_"max")^2. quad "(4)" $

    Here $b$ indexes one of $B$ presentations in a minibatch, $j$ indexes one of $N_E$ hidden excitatory neurons, $C_(b j)$ is its spike count, and $T$ is presentation duration in seconds. Thus $nu_b$ and the ceiling $nu_"max"$ are in hertz. $L_"CE"$ is mean classification cross-entropy, $L$ is the training objective, and $lambda$ scales the squared rate excess (numerically 0.041 when enabled, with rate expressed in hertz). Disabling the penalty sets its contribution to zero. The ceiling is a soft penalty target, so exceeding it remains possible.

  == Appendix A: Training-run specification sheets

  The following shared sheet and all seven TR sheets specify the controlled recipes and their intended uses. The observed learning curves and diagnostic plots are in Results. Operational resource notes are retained as part of the specification, not as evidence that a new job was submitted.

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
    [Default readout], [`mem-mean`], [A *spiking* $1024 arrow 10$ output-LIF layer. At each timestep its pre-reset voltages enter the temporal mean, then emitted spikes subtract the threshold before the next update. These mean voltages—not spike counts—are the logits],
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
    [Readout shape], [$1024 arrow 10$ spiking LIF outputs], [`mem-mean`: mean membrane voltage supplies the logits],
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
    [Readout shape], [$1024 arrow 10$ spiking LIF outputs], [`mem-mean`: mean membrane voltage supplies the logits],
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
    [Readout shape], [$1024 arrow 10$ spiking LIF outputs], [`mem-mean`: mean membrane voltage supplies the logits],
  )

  #divider()

  === A.5. TR-04 — Integration-timestep sweep

  This run changes the integration timestep while keeping each presentation at 200 ms. It tests whether the observed dynamics depend on numerical resolution and measures the extra compute required by finer timesteps. The 0.05 ms cells need the large-memory Cambridge request. The resulting checkpoints are used by #run-links(("exp044",)).

  #table(
    columns: (1.2fr, 1.5fr, 2fr),
    table.header([*Key parameter*], [*Value*], [*Why it differs*]),
    [Architecture], [PING], [Tests the recurrent reference model],
    [Training pool], [7,000 samples], [Sweep-scale default],
    [$Delta t$], [0.05, 0.1, 0.25, 0.5, 1 ms], [Changes numerical resolution while holding 200 ms physical time fixed],
    [Steps/presentation], [4,000; 2,000; 800; 400; 200], [Compute and activation memory scale inversely with $Delta t$],
    [Cells], [5 settings × 3 seeds = 15], [Across-seed stability check],
    [Readout shape], [$1024 arrow 10$ spiking LIF outputs], [`mem-mean`: mean membrane voltage supplies the logits],
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
    [Readout shape], [$1024 arrow 10$ spiking LIF outputs], [`mem-mean`: mean membrane voltage supplies the logits],
  )

  #divider()

  === A.7. TR-06 — Variable-rate streaming bank

  This run trains PING across the input rates used by exp082. One rate is sampled uniformly for each presentation, and the ten output LIF neurons produce class logits from their total spike counts rather than their mean membrane voltages. This keeps the cross-entropy logits dimensionless and gives one additional output spike the same logit increment. The readout weights use a small Gaussian initialization, $cal(N)(0.05, 0.04^2)$, lower-clamped at zero and governed by the shared non-negative constraint.

  The recorded design rationale for this smaller initializer was to avoid saturation at the `mem-mean` scale and silence at the generic fan-in-normalized scale. The present bank retains the chosen $cal(N)(0.05, 0.04^2)$ setting; its learning curves do not independently establish those earlier calibration comparisons. This is a specified spike-count recipe, not a claim that the initializer is uniquely optimal.

  Output firing rates may still be reported as activity measurements. During streaming inference, the output neurons reset at digit boundaries while the hidden PING state continues. The resulting checkpoints are used by #run-links(("exp082",)).

  #table(
    columns: (1.2fr, 1.65fr, 2fr),
    table.header([*Key parameter*], [*Value*], [*Why it differs*]),
    [Architecture], [PING], [Target model for streaming inference],
    [Training pool], [7,000 samples], [Uses the shared sweep-scale training set],
    [Input-rate set], [0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 25 Hz], [Denser sampling within the interval selected by exp080],
    [Sampling rule], [Uniform categorical, independently per presentation], [Makes rate variation part of the training distribution],
    [Readout], [`spike-count`], [Hidden E spikes drive ten spiking LIF class neurons; each logit is that class neuron's total spikes over the presentation],
    [Readout initialization], [$cal(N)(0.05, 0.04^2)$, constrained non-negative], [Keeps the spiking outputs near threshold at high training rates without the saturation caused by the default `mem-mean` scale],
    [Readout shape], [$1024 arrow 10$ spiking LIF outputs], [Ten class neurons emit and reset throughout the presentation],
    [Cells], [1 recipe × 3 seeds = 3], [Checkpoint bank expected by exp082],
  )

  #divider()

  === A.8. TR-07 — Low-input recruitment sweep

  TR-02 asks how changing the activity ceiling affects networks initialized at the
  standard input coupling. TR-07 asks the complementary question: with the strictest
  TR-02 ceiling fixed at 1 Hz from the first epoch, can PING recruit its inhibitory
  loop when the feedforward projection starts weak? It varies the parent mean used
  to initialize expected summed input coupling, before lower clamping and fan-in
  normalization; it does not directly set the mean of the stored synaptic matrix.
  The training pool, optimizer, recurrent loop, and `mem-mean` readout remain at the
  reduced-sweep defaults. Exp022 owns all twelve checkpoints, and #run-links(("exp025",))
  aggregates the three seeds at each setting to test recruitment and path dependence.

  #table(
    columns: (1.2fr, 1.65fr, 2fr),
    table.header([*Key parameter*], [*Value*], [*Why it differs*]),
    [Architecture], [PING], [Tests recruitment of the recurrent inhibitory loop],
    [Training pool], [7,000 samples], [6,300 optimizer-training and 700 validation samples],
    [Epochs], [50], [Matches the reduced production standard],
    [Input summed-coupling parent mean], [0.05, 0.1, 0.3, 0.9], [Varies initial feedforward drive from weak coupling to the shared 0.9 standard before clamping and fan-in normalization],
    [Hidden-E rate target], [1 Hz], [Applies the strictest TR-02 activity ceiling from epoch 0],
    [Parameters held fixed], [PING loop, optimizer, dataset split, and `mem-mean` readout], [Isolates initial feedforward recruitment from the activity-ceiling sweep],
    [Cells], [4 settings × 3 seeds = 12], [Across-seed estimate for every condition],
    [Readout shape], [$1024 arrow 10$ spiking LIF outputs], [`mem-mean`: mean membrane voltage supplies the logits],
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
