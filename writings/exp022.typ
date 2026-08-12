#let meta = (
  title: "Training-run guide",
  date: "2026-08-11",
  description: "A guide to the shared training runs, their parameters, outputs, and downstream consumers.",
  collection: "gamma-gated-sparsity",
  status: "draft",
)

#let r = json("/artifacts/data/exp022/numbers.json")

#let run-links(items) = {
  if items.len() == 0 { [None yet.] } else {
    for (index, item) in items.enumerate() {
      if index > 0 { [, ] }
      link(item + ".html")[#item]
    }
  }
}

#let result-figure(path, alt, caption) = figure(
  image(path, width: 100%, alt: alt),
  caption: caption,
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

#let major-divider() = context {
  if target() == "html" {
    html.elem("hr", attrs: (style: "margin: 3.75rem 0; border-top-width: 2px;"))
  } else {
    v(1.8em)
    line(length: 100%, stroke: 1pt + luma(55%))
    v(1.8em)
  }
}

#let callout(body) = quote(block: true, body)

#let body = [
  // Keep this manual TOC local: demolab suppresses entry headings from the
  // collection-wide outline.
  == Contents

  + #link("#abstract")[Abstract]
  + #link("#summary")[Summary]
  + #link("#training-run-guide")[Training-run guide]
    + #link("#tr-01-canonical-full-data-reference")[TR-01 — Canonical full-data reference]
    + #link("#tr-02-spike-budget-sweep")[TR-02 — Spike-budget sweep]
    + #link("#tr-03-inhibitory-timescale-sweep")[TR-03 — Inhibitory-timescale sweep]
    + #link("#tr-04-integration-timestep-sweep")[TR-04 — Integration-timestep sweep]
    + #link("#tr-05-recurrent-initialization-sweep")[TR-05 — Recurrent-initialization sweep]
    + #link("#tr-06-variable-rate-streaming-bank")[TR-06 — Variable-rate streaming bank]
  + #link("#results-by-training-run")[Results by training run]
    + #link("#tr-01-results-canonical-full-data-reference")[TR-01 — Canonical full-data reference]
    + #link("#tr-02-results-spike-budget-sweep")[TR-02 — Spike-budget sweep]
    + #link("#tr-03-results-inhibitory-timescale-sweep")[TR-03 — Inhibitory-timescale sweep]
    + #link("#tr-04-results-integration-timestep-sweep")[TR-04 — Integration-timestep sweep]
    + #link("#tr-05-results-recurrent-initialization-sweep")[TR-05 — Recurrent-initialization sweep]
    + #link("#tr-06-results-variable-rate-streaming-bank")[TR-06 — Variable-rate streaming bank]

  #major-divider()

  == Abstract

  Exp022 defines the collection's shared training runs and checkpoint bank. It specifies the motivation, parameterization, output-layer shape, and downstream consumers for six training-run types. Five run types comprise #r.n_cells trained cells spanning reference models and controlled sweeps over spike budget, inhibitory timescale, integration timestep, and recurrent initialization. TR-06 adds three PING cells trained across variable input rates with ten spiking output LIF neurons; each class logit is the corresponding neuron's total spike count over the presentation. Together, these runs provide the checkpoints used by the collection's training-dependent experiments.

  #major-divider()

  == Summary

  All runs map Poisson-encoded pixels through 1,024 excitatory neurons to ten spiking output LIF neurons. COBA and PING use the same input-weight and readout-weight initialization distributions. COBA disables recurrent E/I coupling; PING adds a $1024 arrow 256 arrow 1024$ E/I feedback loop and uses stronger backward-pass gradient damping to stabilize training through that recurrent path.

  Unless a run-specific table says otherwise, every cell uses the following contract.

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
    [Input sparsity], [0.95], [Sparsity of the input projection],
    [Input-weight mean], [0.9], [Shared by COBA and PING so input initialization is not an architecture-specific factor],
    [Readout-weight initialization], [$max(0, cal(N)(1.1206, 0.8350^2))$], [The stored $1024 times 10$ weights are sampled directly from this distribution; COBA and PING use the same initializer],
    [E/I loop strength], [COBA: 0; PING: 1], [The forward architectural difference under comparison],
    [Gradient damping], [COBA: 1; PING: 1,000], [Backward-pass stabilization for the recurrent PING loop; it does not change forward dynamics],
    [Surrogate slope], [1], [Spike-gradient surrogate parameter],
    [Default readout], [`mem-mean`], [A *spiking* $1024 arrow 10$ output-LIF layer. At each timestep its pre-reset voltages enter the temporal mean, then emitted spikes subtract the threshold before the next update. These mean voltages—not spike counts—are the logits],
    [Stored projection shapes], [$784 times 1024$; $1024 times 256$; $256 times 1024$; $1024 times 10$], [Input→E, E→I, I→E, and E→class, in source-to-destination orientation],
  )

  #major-divider()

  == Training-run guide

  === TR-01 — Canonical full-data reference

  The full-data COBA and PING cells are used for headline accuracy. They use the official MNIST training partition with no spike-budget penalty, so the comparison is not affected by the smaller dataset or regularization used in the sweeps. Ten percent of that partition is reserved for checkpoint selection; the official test partition remains untouched during training. This experiment uses these cells directly.

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

  === TR-02 — Spike-budget sweep

  This run measures the trade-off between accuracy and firing rate as the spike budget is tightened. It uses a smaller training pool to keep the multi-seed sweep manageable; only the spike cap and its penalty change across conditions. Its absolute rates should therefore not be compared directly with the full-data cells. The resulting checkpoints are used by #run-links(("exp024", "exp025", "exp037", "exp038")).

  #table(
    columns: (1.2fr, 1.5fr, 2fr),
    table.header([*Key parameter*], [*Value*], [*Why it differs*]),
    [Architectures], [COBA and PING], [Direct architecture comparison],
    [Training pool], [7,000 official training samples], [6,300 optimizer-training and 700 validation samples],
    [Spike budget $theta_u$], [off, 5, 2, 1, 0.5, 0.2 spikes/neuron/trial], [Spans unconstrained through severe sparsity],
    [Penalty strength], [$10^(-3)$ when enabled], [Applies the upper-rate regularizer],
    [Cells], [2 × 6 settings × 3 seeds = 36], [Error bars at every frontier point],
    [Readout shape], [$1024 arrow 10$ spiking LIF outputs], [`mem-mean`: mean membrane voltage supplies the logits],
  )

  #divider()

  === TR-03 — Inhibitory-timescale sweep

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

  === TR-04 — Integration-timestep sweep

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

  === TR-05 — Recurrent-initialization sweep

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

  === TR-06 — Variable-rate streaming bank

  This run trains PING across the input rates used by exp082. One rate is sampled uniformly for each presentation, and the ten output LIF neurons produce class logits from their total spike counts rather than their mean membrane voltages. This keeps the cross-entropy logits dimensionless and gives one additional output spike the same logit increment. Output firing rates may still be reported as activity measurements. During streaming inference, the output neurons reset at digit boundaries while the hidden PING state continues. The resulting checkpoints are used by #run-links(("exp082",)).

  #table(
    columns: (1.2fr, 1.65fr, 2fr),
    table.header([*Key parameter*], [*Value*], [*Why it differs*]),
    [Architecture], [PING], [Target model for streaming inference],
    [Training pool], [7,000 samples], [Uses the shared sweep-scale training set],
    [Input-rate set], [0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 25 Hz], [Denser sampling within the interval selected by exp080],
    [Sampling rule], [Uniform categorical, independently per presentation], [Makes rate variation part of the training distribution],
    [Readout], [`spike-count`], [Hidden E spikes drive ten spiking LIF class neurons; each logit is that class neuron's total spikes over the presentation],
    [Readout shape], [$1024 arrow 10$ spiking LIF outputs], [Ten class neurons emit and reset throughout the presentation],
    [Cells], [1 recipe × 3 seeds = 3], [Checkpoint bank expected by exp082],
  )

  #major-divider()

  == Results by training run

  Each completed run reports one training-curve figure and one representative seed-42 raster. The figures summarize all configurations and seeds; the raster is a diagnostic example, not an across-seed statistic.

  === TR-01 results — Canonical full-data reference

  *Run status:* complete · 6/6 cells represented

  #result-figure(
    "/artifacts/data/exp022/curves__canonical.svg",
    "Test-accuracy learning curves for the canonical COBA and PING cells.",
    [Training curves for the six full-data reference cells.],
  )

  #result-figure(
    "/artifacts/data/exp022/rasters/ping__canonical__seed42.png",
    "Seed-42 raster and population-rate diagnostic for canonical PING.",
    [Sample raster: canonical PING, seed 42, held-out digit-0 probe.],
  )

  #divider()

  === TR-02 results — Spike-budget sweep

  *Run status:* complete · 36/36 cells represented

  #result-figure(
    "/artifacts/data/exp022/curves__theta_u.svg",
    "Test-accuracy learning curves across the spike-budget sweep.",
    [Training curves across both architectures, six spike budgets, and three seeds.],
  )

  #result-figure(
    "/artifacts/data/exp022/rasters/ping__off__seed42.png",
    "Seed-42 raster for the no-budget PING endpoint in the spike-budget sweep.",
    [Sample raster: PING with spike budget off, seed 42.],
  )

  #divider()

  === TR-03 results — Inhibitory-timescale sweep

  *Run status:* complete · 18/18 cells represented

  #result-figure(
    "/artifacts/data/exp022/curves__tau_gaba.svg",
    "Test-accuracy learning curves across the inhibitory-timescale sweep.",
    [Training curves across six $tau_"GABA"$ values and three seeds.],
  )

  #result-figure(
    "/artifacts/data/exp022/rasters/ping__tg6__seed42.png",
    "Seed-42 raster for PING at tau GABA 6 milliseconds.",
    [Sample raster: PING at $tau_"GABA"=6$ ms, seed 42.],
  )

  #divider()

  === TR-04 results — Integration-timestep sweep

  *Run status:* complete · 15/15 cells represented

  #result-figure(
    "/artifacts/data/exp022/curves__dt.svg",
    "Test-accuracy learning curves across the integration-timestep sweep.",
    [Training curves across five integration timesteps and three seeds.],
  )

  #result-figure(
    "/artifacts/data/exp022/rasters/ping__dt0p1__seed42.png",
    "Seed-42 raster for PING at the standard 0.1 millisecond timestep.",
    [Sample raster: PING at $Delta t=0.1$ ms, seed 42.],
  )

  #divider()

  === TR-05 results — Recurrent-initialization sweep

  *Run status:* complete · 12/12 cells represented

  #result-figure(
    "/artifacts/data/exp022/curves__init.svg",
    "Test-accuracy learning curves across recurrent initialization conditions.",
    [Training curves across four recurrent-loop conditions and three seeds.],
  )

  #result-figure(
    "/artifacts/data/exp022/rasters/frozen_ping__seed42.png",
    "Seed-42 raster for the frozen recurrent PING control.",
    [Sample raster: frozen PING recurrent loop, seed 42.],
  )

  #divider()

  === TR-06 results — Variable-rate streaming bank

  *Run status:* pending · 0/3 cells trained

  *TODO — training curves.* Add the three variable-rate learning curves after the Cambridge jobs complete.

  *TODO — sample raster.* Add a seed-42 E/I/output raster at a declared held-out rate, plus low- and high-rate diagnostics if one raster hides a rate-dependent failure.
]
