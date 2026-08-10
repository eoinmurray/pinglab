#let meta = (
  title: "Shared training catalogue",
  date: "2026-06-28",
  description: "The registered training types, architectures, compute requirements, diagnostics, and reusable checkpoints for the gamma-gated-sparsity collection.",
  collection: "gamma-gated-sparsity",
  status: "draft",
)

#let r = json("/artifacts/data/exp022/numbers.json")

#let body = [
  == Abstract

  This entry is the gamma-gated-sparsity collection's training catalogue. It defines each type of training, records the resulting learning curves and activity diagnostics, and exposes one shared checkpoint bank to downstream experiments. The completed bank contains #r.n_cells independently trained cells across the canonical, spike-budget, inhibitory-timescale, timestep, and recurrent-initialization studies. A sixth training type is planned for streaming inference: PING trained with a variable Poisson input rate and a summed-spiking readout. That extension is specified here but is not counted among the completed cells.

  == Methods

  === 1. Shared architecture and training contract

  #enum(
    [*Encode the stimulus.* Each MNIST image supplies 784 independent Poisson input channels. Completed training types use a fixed maximum-pixel rate of 25 Hz. The planned streaming bank instead samples one rate independently for each image presentation.],
    [*Simulate the network.* A trial lasts #r.standard.t_ms ms. The standard timestep is #r.standard.dt_ms ms, giving 2000 recurrent updates per presentation. PING contains 1024 excitatory and 256 inhibitory neurons; COBA retains the excitatory pathway but disables recurrent E/I coupling.],
    [*Form the readout.* Completed cells use `mem-mean`: a 1024-to-10 projection drives ten output LIF membranes and the class logits are their time-averaged voltages. The planned variable-rate cells use `rate`: the 1024 excitatory spike trains are summed through time and projected once through a 1024-to-10 linear readout.],
    [*Train and retain.* Every registered configuration trains seeds 42, 43, and 44 for #r.standard.epochs epochs. The canonical reference sees #r.standard.max_samples_canonical pooled MNIST samples; sweep cells see #r.standard.max_samples_sweeps. Each cell retains its configuration, weights, epoch metrics, and a representative held-out raster.],
  )

  #table(
    columns: 3,
    align: (left, left, left),
    table.header([*Component*], [*Shape*], [*Role*]),
    [Poisson input], [$T times B times 784$], [one event stream per image pixel],
    [input projection $W_"in"$], [$784 times 1024$], [pixels to excitatory population],
    [excitatory population], [$B times 1024$], [trained stimulus representation],
    [inhibitory population], [$B times 256$], [PING feedback population],
    [E→I projection $W_"EI"$], [$1024 times 256$], [excitatory recruitment of inhibition],
    [I→E projection $W_"IE"$], [$256 times 1024$], [inhibitory feedback],
    [class readout $W_"out"$], [$1024 times 10$], [excitatory activity to digit logits],
  )

  Here $T$ is the number of timesteps and $B$ is the minibatch size. Matrix shapes use the stored source-to-destination orientation.

  === 2. Registered training types

  The registry is organised by scientific training type. A type may contain several parameter variants, and every variant contains three independent seed cells.

  #table(
    columns: 7,
    align: (left, left, left, left, right, right, left),
    table.header([*Training type*], [*Models*], [*Varied parameter*], [*MNIST*], [*Cells*], [*Trained*], [*Primary use*]),
    [full-data reference], [COBA, PING], [none], [all], [6], [6], [canonical accuracy],
    [spike-budget training], [COBA, PING], [$theta_u in {"off", 5, 2, 1, 0.5, 0.2}$], [10%], [36], [36], [accuracy–sparsity frontier],
    [inhibitory-timescale training], [PING], [$tau_"GABA" in {4.5, 6, 9, 12, 18, 27}$ ms], [10%], [18], [18], [rhythm timescale],
    [timestep training], [PING], [$Delta t in {0.05, 0.1, 0.25, 0.5, 1}$ ms], [10%], [15], [15], [numerical stability],
    [recurrent-initialization training], [PING], [loop initialization and trainability], [10%], [12], [12], [built-in versus learned recurrence],
    [variable-rate streaming training], [PING], [input-rate distribution and readout], [10% initially], [3], [0], [exp082 streaming inference],
  )

  *Note, planned variable-rate bank.* Exp080 selects the interval 0.5–25 Hz for later PING training. The proposed cells sample uniformly from the discrete set 0.5, 1, 2, 5, 10, and 25 Hz, independently per presentation, and use the summed-spiking `rate` readout. This training changes both the input distribution and readout relative to the existing `ping__off__seed*` cells. It is therefore a new training type, not another point in the spike-budget sweep.

  *Note, engine support.* `tools/snn` accepts the categorical rate set through `--input-rates`, samples it reproducibly per presentation, serializes both the values and sampling rule, and restores them through `--load-config`. The three cells are registered for the Cambridge job array. This entry still does not report their results because the full training jobs have not run.

  === 3. Cambridge HPC compute plan

  Surrogate-gradient backpropagation stores state across every timestep. A standard presentation therefore traverses 2000 sequential recurrent steps, and the backward pass is both memory-heavy and bandwidth-sensitive. The completed registry represents approximately 185 A100-GPU-hours and 49 million sample-forwards. The $Delta t = 0.05$ ms cells are the exceptional memory case, previously requiring approximately 31 GB.

  The Cambridge run should use one independently resumable job-array element per cell. Each element writes only its own cell directory, validates the emitted configuration before treating an existing checkpoint as complete, and records the scheduler job identifier with the run provenance. Canonical full-data cells should not share an allocation with sweep cells because their runtime is substantially longer. The final array layout, GPU type, memory request, concurrency cap, and wall-time request remain notes until the available Cambridge partition is confirmed.

  == Results

  Each completed training type is reported in the same order: definition, learning curve, representative seed-42 raster, and outcome. The complete raster gallery remains in the appendix.

  === 1. Full-data reference

  coba and ping with no spike budget, all 70k images, seeds 42/43/44 (6 cells), the full-data baseline. Unconstrained, the two architectures are essentially tied: coba reaches ≈ 95.5 % and ping ≈ 94.0 %, with near-zero across-seed spread. This is the reference point from which the spike-budget sweep pulls them apart.

  #figure(
    image(
      "/artifacts/data/exp022/curves__canonical.svg",
      width: 100%,
      alt: "Test-accuracy learning curves over epochs, canonical reference.",
    ),
    caption: [Test accuracy over epochs, canonical full-MNIST cells (coba dashed, ping solid; three seeds each). The two loops reach parity when no spike budget is imposed.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/ping__canonical__seed42.png", width: 100%, alt: "Seed-42 spike raster and population-rate diagnostic for the canonical PING training."),
    caption: [Representative activity after full-data PING training. The raster shows excitatory and inhibitory spikes for the common held-out digit-0 probe; the lower panel shows the population-rate diagnostic.],
  )

  === 2. Spike-budget training

  coba and ping across spike budgets ∈ off, 5, 2, 1, 0.5, 0.2, three seeds each (36 cells), so the accuracy–rate frontier carries error bars at every point. The spike budget is a per-neuron cap on firing (spikes/trial); lower is tighter. This is the headline result: as the budget tightens, ping degrades gracefully (91.6 → 86.5 %) while coba collapses (90.7 → 60.1 %), a gap that widens monotonically from ≈ 0 to ≈ 26 points. ping's γ-rhythm gates sparsity; coba's bare feedforward code cannot pay the budget without shedding accuracy.

  #figure(
    image(
      "/artifacts/data/exp022/curves__theta_u.svg",
      width: 100%,
      alt: "Test-accuracy learning curves over epochs, spike-budget sweep.",
    ),
    caption: [Test accuracy over epochs across the spike-budget sweep. Tighter budgets plateau lower (the spike-economy trade-off), and coba falls far faster than ping.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/ping__off__seed42.png", width: 100%, alt: "Seed-42 spike raster and population-rate diagnostic for PING trained without a spike budget on the ten-percent MNIST subset."),
    caption: [Representative activity for the spike-budget training type at the no-budget endpoint. The raster uses seed 42 and the common held-out digit-0 probe. The appendix shows every budget for both architectures.],
  )

  *Scope.* This frontier is a 10%-MNIST result: the whole sweep trains on the subset (§2 of Methods), and we have not re-run it at full data. The comparison is nonetheless internally clean: every cell here shares the same data fraction, so the coba-vs-ping gap is a budget effect, not a data-fraction artifact. And the direction of the density difference (§7) makes 10% the *conservative* regime for this claim: at full MNIST the no-budget baseline fires ≈ 2× harder, so a tight per-neuron cap would have further to pull coba down, widening the gap rather than closing it. We therefore read the ≈ 26-point separation as a floor, and do not claim the absolute numbers transfer unchanged to full MNIST.

  === 3. Inhibitory-timescale training

  ping across τ_GABA ∈ 4.5, 6, 9, 12, 18, 27 ms, three seeds each (18 cells). Accuracy is largely insensitive to inhibitory decay (≈ 88–92 % across the ladder), but the *rhythm* is not: measured from the trained networks, the γ-frequency falls monotonically with τ_GABA (≈ 50 Hz at 4.5 ms → ≈ 19 Hz at 27 ms), sitting at ≈ 45 Hz at the canonical τ_GABA = 6 ms, matching the operating point (Appendix).

  #figure(
    image(
      "/artifacts/data/exp022/curves__tau_gaba.svg",
      width: 100%,
      alt: "Test-accuracy learning curves over epochs, τ_GABA ladder.",
    ),
    caption: [Test accuracy over epochs across the τ_GABA ladder; cells converge to similar accuracy regardless of inhibitory decay.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/ping__tg6__seed42.png", width: 100%, alt: "Seed-42 spike raster and population-rate diagnostic for PING at the standard inhibitory time constant."),
    caption: [Representative activity for inhibitory-timescale training at the standard $tau_"GABA"$. The raster uses seed 42 and the common held-out digit-0 probe; the appendix shows the complete ladder.],
  )

  === 4. Timestep training

  ping across Δt ∈ 0.05, 0.1, 0.25, 0.5, 1.0 ms (physical T fixed), three seeds each (15 cells), the documented timestep exception. Accuracy is flat across the sweep (≈ 90.4–91.4 %): the integrator is robust to timestep from 0.1 to 1.0 ms, and the 0.05 ms cells (which need ≈ 31 GB and so ran on a 5090) agree.

  #figure(
    image(
      "/artifacts/data/exp022/curves__dt.svg",
      width: 100%,
      alt: "Test-accuracy learning curves over epochs, Δt sweep.",
    ),
    caption: [Test accuracy over epochs across the integration-timestep sweep; accuracy is insensitive to Δt over the tested range.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/ping__dt0p1__seed42.png", width: 100%, alt: "Seed-42 spike raster and population-rate diagnostic for PING at the standard integration timestep."),
    caption: [Representative activity for timestep training at the standard $Delta t$. The raster uses seed 42 and the common held-out digit-0 probe; the appendix shows all timestep variants.],
  )

  === 5. Recurrent-initialization training

  ping with four recurrent-loop inits (frozen PING, trainable from PING / zero / small seed), three seeds each (12 cells). All reach ≈ 89–91 %, but only the frozen-PING control keeps the true E/I regime (E ≈ 10 Hz, I ≈ 62 Hz): the trainable-loop cells drift toward a feedforward code (high E, low or zero I): the zero-init cells never engage inhibition at all (I ≈ 0 Hz). Comparable accuracy, but the rhythm is not learned when it is not built in.

  #figure(
    image(
      "/artifacts/data/exp022/curves__init.svg",
      width: 100%,
      alt: "Test-accuracy learning curves over epochs, init variants.",
    ),
    caption: [Test accuracy over epochs across the recurrent-loop inits; trainable-loop cells learn noisier curves than the frozen control.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/frozen_ping__seed42.png", width: 100%, alt: "Seed-42 spike raster and population-rate diagnostic for the frozen recurrent PING control."),
    caption: [Representative activity for recurrent-initialization training. This seed-42 control keeps the recurrent PING loop frozen; the appendix shows all four initialization and trainability conditions.],
  )

  === 6. Variable-rate streaming training

  *Note, awaiting training.* No learning curve or raster exists yet. After the registered Cambridge jobs complete, this section will show the three mixed-rate learning curves, fixed-rate held-out accuracy across 0.5–25 Hz, and matched rasters at low, intermediate, and high rates. It will compare the new `rate`-readout cells with the existing fixed-25-Hz `mem-mean` cells. The checkpoints are the training source for #link("/exp082/")[exp082], which supersedes exp048 for variable-rate streaming inference.

  === 7. Training data and spike density

  The appendix rasters split cleanly along one axis that is easy to miss: the *canonical* cells see all 70k MNIST images, but every sweep (including the no-budget spike-budget = off cell, which is otherwise identical to canonical) trains on 10%. That difference deserves its own read, because it changes how *busy* the trained network is before any sweep parameter enters.

  We isolate it by sending the *same* fixed digit-0 image through the no-budget coba and ping networks at each fraction (seed 42). Same architecture, same operating point ($tau_"AMPA" = 2$ ms, $tau_"GABA" = 6$ ms), same spike budget (none): the only difference is 10× the training images, so any gap in the rasters is the data's doing alone.

  #figure(
    image(
      "/artifacts/data/exp022/comparison__data_fraction.png",
      width: 100%,
      alt: "Two-by-two grid of spike rasters: COBA and PING trained on 100% versus 10% of MNIST, same digit-0 input.",
    ),
    caption: [The same digit-0 image through the no-budget coba (top) and ping (bottom) networks, trained on all of MNIST (left) versus 10% (right); E cells black below the divider, I cells red above, per-panel mean rates annotated. More training data yields a visibly denser code in both architectures.],
  )

  More data buys a denser code. In both loops the full-MNIST network fires roughly twice as hard:

  - *coba* excitatory rate ≈ 420 Hz on all of MNIST against ≈ 212 Hz on 10%;
  - *ping* inhibitory rate ≈ 140 Hz against ≈ 63 Hz (excitatory ≈ 19 vs ≈ 11 Hz).

  where the *mean rate* is a population's total spikes divided by its neuron count and the 200 ms window. The canonical rasters are visibly busier: the extra spikes are the network recruiting capacity to separate the fuller set of class variations, which the 10% subset never forces it to. ping keeps its γ rhythm at both fractions, and the sparser 10% network even reads as a *cleaner* gamma (Appendix A.2); coba stays asynchronous throughout (I silent, no loop). The density shift is a change of degree, not regime.

  The practical consequence is a caveat on the appendix as a whole: *absolute* firing rates are not comparable across the canonical-vs-sweep boundary, because the 10% cells sit at a systematically lower density for reasons that have nothing to do with the swept parameter. This is exactly why the canonical reference (§1) carries the headline rates and the sweeps are read as *trends within a family* (the spike-budget frontier, the τ_GABA-to-γ scaling) rather than as absolute numbers set against the full-data baseline.

  === 8. Why more data drives a denser code

  The figure measures the density gap; it does not prove its cause. But only two things change between a row's two cells (the number of training images and, downstream of that, the number of weight updates), and both push firing *up*. Neither is specific to the E/I loop, which is why coba and ping move together rather than one architecture reacting and the other not.

  - *More to separate.* The 70k-image set carries far more within-class variation than its 10% subset. To keep the ten digit classes separable at the readout (a linear map from per-neuron spike counts to class scores), the network must partition a higher-dimensional representation, and it pays for that resolution in spikes: more neurons recruited, each firing more. The smaller subset is an easier separation problem that a sparser code already fits, so the network never has to spend the extra spikes.
  - *More weight updates.* Epochs are fixed at 50 for both, but an epoch over 70k images is ≈ 10× the mini-batches of an epoch over 10%, so the canonical cells take ≈ 10× as many gradient steps. These are the *no-budget* cells, so nothing opposes the drift: each update is free to grow the input and recurrent weights that set the drive $g (V - E)$, and more updates compound into a higher operating point.

  where the terms are:

  - $g$ — the synaptic conductance a presynaptic spike opens (larger weights → larger $g$);
  - $V$ — the postsynaptic membrane voltage;
  - $E$ — the synapse's reversal potential, so $g (V - E)$ is the current a spike injects.

  The spike budget is precisely the counter-pressure to both levers: the sweep's tightening $theta_u$ (the per-neuron cap on spikes/trial) caps the rate growth described here, which is why the budgeted cells sit far below their no-budget siblings. The canonical–off pair simply removes that cap, leaving data fraction as the only lever. A clean way to separate the two mechanisms would be to train a 10% cell for ≈ 10× the epochs, matching the update count at the smaller set: if the rate closes most of the gap the growth is update-driven, and if it does not the residual is the genuine cost of the harder separation. We have not run that control; the two levers are offered as the mechanism the rasters are consistent with, not as a decomposition we have measured.

  == Appendix — per-config spike rasters (digit 0)

  The same fixed MNIST image (digit 0, sample 0) sent through each trained network (seed 42 as the per-config representative), so every raster is directly comparable. E cells sit below the divider, I cells above; the lower panel is the 1 ms population rate. These are the visual counterpart to the firing-rate and rhythm numbers above: sparse γ-rhythmic PING versus dense asynchronous COBA, and how each regime deforms under the sweeps.

  === A.1 Canonical reference

  #figure(
    image(
      "/artifacts/data/exp022/rasters/coba__canonical__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, coba canonical, seed 42.",
    ),
    caption: [COBA, canonical (no spike budget, all MNIST).],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/ping__canonical__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, ping canonical, seed 42.",
    ),
    caption: [PING, canonical (no spike budget, all MNIST).],
  )

  === A.2 Spike-budget sweep

  #figure(
    image(
      "/artifacts/data/exp022/rasters/coba__off__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, coba off, seed 42.",
    ),
    caption: [COBA, spike budget = off.],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/coba__tu5__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, coba tu5, seed 42.",
    ),
    caption: [COBA, spike budget = 5.],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/coba__tu2__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, coba tu2, seed 42.",
    ),
    caption: [COBA, spike budget = 2.],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/coba__tu1__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, coba tu1, seed 42.",
    ),
    caption: [COBA, spike budget = 1.],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/coba__tu0p5__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, coba tu0p5, seed 42.",
    ),
    caption: [COBA, spike budget = 0.5.],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/coba__tu0p2__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, coba tu0p2, seed 42.",
    ),
    caption: [COBA, spike budget = 0.2.],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/ping__off__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, ping off, seed 42.",
    ),
    caption: [PING, spike budget = off.],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/ping__tu5__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, ping tu5, seed 42.",
    ),
    caption: [PING, spike budget = 5.],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/ping__tu2__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, ping tu2, seed 42.",
    ),
    caption: [PING, spike budget = 2.],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/ping__tu1__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, ping tu1, seed 42.",
    ),
    caption: [PING, spike budget = 1.],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/ping__tu0p5__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, ping tu0p5, seed 42.",
    ),
    caption: [PING, spike budget = 0.5.],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/ping__tu0p2__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, ping tu0p2, seed 42.",
    ),
    caption: [PING, spike budget = 0.2.],
  )

  === A.3 τ_GABA ladder

  #figure(
    image(
      "/artifacts/data/exp022/rasters/ping__tg4p5__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, ping tg4p5, seed 42.",
    ),
    caption: [PING, τ_GABA = 4.5 ms.],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/ping__tg6__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, ping tg6, seed 42.",
    ),
    caption: [PING, τ_GABA = 6 ms.],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/ping__tg9__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, ping tg9, seed 42.",
    ),
    caption: [PING, τ_GABA = 9 ms.],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/ping__tg12__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, ping tg12, seed 42.",
    ),
    caption: [PING, τ_GABA = 12 ms.],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/ping__tg18__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, ping tg18, seed 42.",
    ),
    caption: [PING, τ_GABA = 18 ms.],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/ping__tg27__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, ping tg27, seed 42.",
    ),
    caption: [PING, τ_GABA = 27 ms.],
  )

  === A.4 Δt sweep

  #figure(
    image(
      "/artifacts/data/exp022/rasters/ping__dt0p05__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, ping dt0p05, seed 42.",
    ),
    caption: [PING, Δt = 0.05 ms.],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/ping__dt0p1__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, ping dt0p1, seed 42.",
    ),
    caption: [PING, Δt = 0.1 ms.],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/ping__dt0p25__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, ping dt0p25, seed 42.",
    ),
    caption: [PING, Δt = 0.25 ms.],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/ping__dt0p5__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, ping dt0p5, seed 42.",
    ),
    caption: [PING, Δt = 0.5 ms.],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/ping__dt1__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, ping dt1, seed 42.",
    ),
    caption: [PING, Δt = 1 ms.],
  )

  === A.5 Init variants

  #figure(
    image(
      "/artifacts/data/exp022/rasters/frozen_ping__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, frozen_ping, seed 42.",
    ),
    caption: [PING, frozen loop (control).],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/trainable_ping_init__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, trainable_ping_init, seed 42.",
    ),
    caption: [PING, trainable loop, PING init.],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/trainable_zero_init__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, trainable_zero_init, seed 42.",
    ),
    caption: [PING, trainable loop, zero init.],
  )

  #figure(
    image(
      "/artifacts/data/exp022/rasters/trainable_small_init__seed42.png",
      width: 100%,
      alt: "Spike raster of E and I populations with the I-population power spectrum, trainable_small_init, seed 42.",
    ),
    caption: [PING, trainable loop, small init.],
  )

]
