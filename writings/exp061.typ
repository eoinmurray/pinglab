#import "/.demolab/lib.typ": cite, reference-list

#let meta = (
  title: "Does finer Δt stabilise the integration?",
  date: "2026-07-11",
  description: "The first stability experiment in the Spiking Heidelberg Digits (SHD) program: sweeping the integration timestep Δt ∈ {1.0, 0.5, 0.25} ms shows finer integration does not stabilise the free signed-recurrent network. The NaN-epoch rate does not fall and the pre-clip gradient explosion gets orders of magnitude worse, so the kill criterion fires: the divergence is not coarse-Δt stiffness.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
)

// Every number below is read from this run's numbers.json, never hand-typed.
#let r = json("/artifacts/data/exp061/numbers.json")
// Cross-import: exp060's data, for the "NaN at epoch N with tiny weights" fact
// (that divergence is exp060's, not this run's — its first NaN epoch lives there).
#let r060 = json("/artifacts/data/exp060/numbers.json")
#let cells = r.cells.sorted(key: c => -c.dt)   // coarse → fine
#let coarse = cells.at(0)                        // Δt = 1.0
#let mid = cells.at(1)                           // Δt = 0.5
#let finest = cells.last()                       // Δt = 0.25
// Order of magnitude of a value (⌊log10⌋), for the runaway gradient norms.
#let oom(x) = calc.round(calc.log(x, base: 10))
// Render a Δt in ms preserving a trailing ".0" on whole values (Typst prints the
// float 1.0 as "1"), so the swept steps read as 1.0 / 0.5 / 0.25 like the recipe.
#let dtf(x) = if x == calc.round(x) [#str(x).0] else [#x]
// Config inputs held fixed across the sweep (exp060's Rung A recipe), from data.
#let cfg = r.config
// BPTT unroll depth T_steps = T / Δt for a given cell's Δt.
#let tsteps(dt) = calc.round(cfg.t_ms / dt)

#let results-caption = [
  *What we see refutes it.* Not one of the three metrics improves as Δt shrinks.
  The NaN-epoch rate does not fall toward zero: #coarse.nan_epochs of
  #coarse.epochs epochs diverge at Δt = #dtf(coarse.dt), #mid.nan_epochs at #dtf(mid.dt), and
  #finest.nan_epochs at #dtf(finest.dt) (flat-to-worse, not a descent). The peak pre-clip
  gradient norm moves the wrong way and violently, climbing from ≈ 10#super[#oom(coarse.max_grad_norm)]
  at Δt = #dtf(coarse.dt) to ≈ 10#super[#oom(finest.max_grad_norm)] at Δt = #dtf(finest.dt), roughly
  #(oom(finest.max_grad_norm) - oom(coarse.max_grad_norm)) orders of magnitude
  the wrong way. Only the excitatory-to-excitatory recurrent-weight norm
  $||W_"ee"||$ stays in one range
  (#calc.round(finest.max_wee_norm, digits: 1)–#calc.round(mid.max_wee_norm, digits: 1)),
  so this is not a runaway-weight story.
]

#let reading-body = [
  The coarse-Δt run reproduces #link("/exp060/")[exp060]'s divergence
  (#coarse.nan_epochs of #coarse.epochs epochs return NaN, #coarse.nan_forward_batches
  NaN forward passes), so the mechanism is genuinely exercised; this is not a
  null run. But finer Δt does not remove it: at Δt = #dtf(finest.dt) the NaN-epoch rate is
  #finest.nan_epochs of #finest.epochs, no lower than at Δt = #dtf(coarse.dt), and the peak
  pre-clip gradient norm is *larger* at every halving, reaching
  ≈ 10#super[#oom(finest.max_grad_norm)] at the finest step.

  *The kill criterion fires.* The plan pre-registered: _if NaN persists at
  Δt = #dtf(finest.dt), coarse integration is not the cause, so drop Δt from the recipe and
  look upstream._ It persists. So the divergence is not exponential-Euler stiffness
  at a coarse timestep. The mechanism is visible in the sweep itself: quartering Δt
  quarters the step but *quadruples* the BPTT unroll (#tsteps(coarse.dt) → #tsteps(finest.dt) steps), and the
  longer gradient chain through the signed E→I→E recurrence explodes far worse
  than any stiffness a finer step relieves. Δt is not the stabiliser; it leaves
  the recipe.

  *Where this points.* With neither Δt (here) nor weight size (exp060: a NaN at
  epoch #r060.first_nan_epoch with tiny weights) explaining the divergence, the plan's reserved
  fallback, a forward-pass state clamp bounding voltage and conductance, becomes
  the live lead. The reduced-scale CPU cross-check agreed directionally: no NaN at
  #cfg.local_subset samples, but the same non-monotonic gradient blow-up peaking at Δt = #dtf(finest.dt).

  #emph[Caveat.] Single seed, so these are point estimates, not error bars. The
  effect is large (orders of magnitude, not margins), so a seed sweep is unlikely
  to reverse the direction; the plan's three-seed default is the confirming run.
]

#let body = [
  == Abstract

  #link("/exp060/")[exp060] showed the free signed-recurrent ceiling trains on
  the Spiking Heidelberg Digits (SHD) spoken-digit classification benchmark#cite(1)
  but diverges intermittently: scattered epochs return NaN (not-a-number), the
  pre-clip gradient norm spikes into the millions, and the excitatory-to-excitatory
  (E→E) recurrent weight grows without bound. Here E and I denote the excitatory
  and inhibitory populations, and "signed" means the recurrent weights are free to
  take either sign (no Dale's law). A NaN appeared at *epoch #r060.first_nan_epoch, when the weights
  were still small*, so the divergence is not weight-growth runaway. The
  #link("/ar063/")[program plan] parks that observation as its first hypothesis:
  the NaN is exponential-Euler *integration stiffness* at the coarse Δt = #dtf(coarse.dt) ms
  (the numerical error a large integration timestep introduces into a stiff
  differential equation), so halving then quartering Δt should drop the NaN-epoch
  rate toward zero.

  This experiment sweeps Δt ∈ {#dtf(coarse.dt), #dtf(mid.dt), #dtf(finest.dt)} ms on a single seed (#r.seed),
  holding exp060's Rung A recipe otherwise fixed, and measures the plan's three
  registered stability metrics against Δt. The hypothesis is refuted: not one of
  the three improves as Δt shrinks. The NaN-epoch rate does not fall, the peak
  pre-clip gradient norm climbs orders of magnitude in the wrong direction, and the
  kill criterion fires, so coarse-Δt stiffness is not the cause and Δt leaves the
  recipe.

  == Methods

  This holds exp060's Rung A recipe (its free signed-recurrent configuration) fixed
  and varies only the integration timestep, Δt ∈ {#dtf(coarse.dt), #dtf(mid.dt), #dtf(finest.dt)} ms, on a single
  seed (#r.seed): a fast first read rather than an error-barred result. It measures
  the plan's three registered stability metrics against Δt:

  - the *NaN-epoch rate*, the fraction of epochs with a non-finite loss or a
    NaN forward pass;
  - the *max pre-clip gradient norm*, the peak global gradient magnitude before
    clipping, the quantity that spiked to millions in exp060;
  - the *max recurrent-weight norm*, the peak $||W_"ee"||$, the E→E block that grew
    unbounded.

  #table(
    columns: 2,
    [Parameter], [Value],
    [Dataset], [SHD, #(r.max_samples)-sample subset],
    [Model], [COBANet (conductance-based spiking network), #cfg.n_hidden hidden units, all four recurrent blocks trainable, signed (no Dale's law)],
    [Swept variable], [$Delta t in {#dtf(coarse.dt), #dtf(mid.dt), #dtf(finest.dt)}$ ms at trial duration $T = #cfg.t_ms$ ms, so the number of integration steps $T_"steps" in {#tsteps(coarse.dt), #tsteps(mid.dt), #tsteps(finest.dt)}$],
    [Seeds], [#cfg.n_seeds (seed #r.seed), a quick first read],
    [Training], [#r.epochs epochs, learning rate #cfg.lr, weight decay #cfg.weight_decay, batch size #cfg.batch_size, surrogate-gradient backpropagation through time (BPTT)],
    [Regulariser], [upper firing-rate bound (threshold $theta_u = #cfg.fr_reg_upper_theta$, strength $s_u = #cfg.fr_reg_upper_strength$), load-bearing, from exp060],
    [Compute], [#r.compute],
  )

  Δt is the *only* thing that varies across the three runs; everything else is
  exp060's recipe verbatim, so any change in the stability metrics is
  attributable to Δt alone. Δt = #dtf(coarse.dt) reproduces exp060's exact failure setting,
  so the finer-Δt arms are read against a grounded baseline rather than an
  abstract one.

  #quote(block: true)[
    *Kill criterion.* If NaN persists at Δt = #dtf(finest.dt) (the finest step), coarse
    integration is not the cause, so drop Δt from the recipe and look upstream
    (a forward-pass state clamp).
  ]

  == Results

  #figure(
    image("/artifacts/data/exp061/stability_vs_dt.svg", width: 100%,
      alt: "Three panels against Δt (coarse to fine): NaN-epoch rate, max pre-clip gradient norm on a log axis, and max E-to-E recurrent-weight norm."),
    caption: [
      The three stability metrics against Δt in ms (coarse → fine, left → right on
      each panel): the NaN-epoch rate (%, left), the peak pre-clip gradient norm
      (log scale, centre), and the peak E→E recurrent-weight norm $||W_"ee"||$
      (right). *What we expect if the plan's hypothesis holds:* all three fall as
      Δt shrinks, the NaN-epoch rate toward zero and the gradient and weight norms
      toward bounded values. #results-caption
    ],
  )

  #figure(
    image("/artifacts/data/exp061/grad_trace.svg", width: 100%,
      alt: "Max pre-clip gradient norm per epoch, one line per Δt, on a log y-axis."),
    caption: [
      The max pre-clip gradient norm per epoch (x-axis epoch, y-axis log scale),
      one line per Δt. This shows *when* the gradient spikes fire, not just their
      peak: a stiff integration diverges on individual forward passes rather than
      drifting up smoothly.
    ],
  )

  #reading-body

  #reference-list((
    (text: [B. Cramer, Y. Stradmann, J. Schemmel & F. Zenke (2020). The Heidelberg spiking data sets for the systematic evaluation of spiking neural networks. _IEEE Transactions on Neural Networks and Learning Systems_ 33(7):2744–2757.], doi: "10.1109/TNNLS.2020.3044364"),
  ))
]
