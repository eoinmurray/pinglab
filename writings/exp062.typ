#let meta = (
  title: "Is Dale's law the implicit stabiliser?",
  date: "2026-07-11",
  description: "The SHD program's second stability experiment: with the free signed-recurrent net's divergence shown (exp061) to be a recurrent gradient explosion rather than Δt stiffness, test whether Dale's law (the non-negativity constraint that bounds the recurrent weights) is what keeps a matched net trainable.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
)

// Every number below is read from this run's numbers.json, never hand-typed.
#let r = json("/artifacts/data/exp062/numbers.json")
#let free = r.cells.filter(c => c.dales_law == false).at(0)
#let dales = r.cells.filter(c => c.dales_law == true).at(0)
#let oom(x) = calc.round(calc.log(x, base: 10))

#let results-caption = [
  *What we see confirms it.* The free net (red) carries #free.nan_epochs of
  #free.epochs epochs NaN and a peak ‖W_ee‖ of
  #calc.round(free.max_wee_norm, digits: 1); Dale's law (black) drops the
  NaN-epoch rate to *zero* and holds ‖W_ee‖ at
  #calc.round(dales.max_wee_norm, digits: 1), while best accuracy barely moves
  (#calc.round(free.best_acc_pct, digits: 1)% → #calc.round(dales.best_acc_pct, digits: 1)%).
  The constraint buys stability for a few points of accuracy.
]

#let reading-body = [
  The free net diverges exactly as before: #free.nan_epochs of #free.epochs
  epochs go NaN, with a peak pre-clip gradient norm of
  ≈ 10#super[#oom(free.max_grad_norm)], reproducing
  #link("/exp061/")[exp061]'s coarse-Δt cell. Flip the single flag to
  Dale's law and, at otherwise identical settings, the divergence is *gone*:
  #dales.nan_epochs NaN epochs, the peak gradient norm bounded at
  ≈ #calc.round(dales.max_grad_norm) (some
  #(oom(free.max_grad_norm) - oom(dales.max_grad_norm)) orders of magnitude
  tamer), and ‖W_ee‖ held at #calc.round(dales.max_wee_norm, digits: 1) against
  the free net's #calc.round(free.max_wee_norm, digits: 1).

  *The hypothesis holds.* Dale's law is the implicit stabiliser. Its
  non-negativity projection clamps the recurrent weights each step, bounding the
  E→I→E loop gain below the runaway threshold the free signed net crosses. The
  divergence never starts, so there is no gradient explosion to clip. This is the
  first *stable recipe* the program's goal asked for: a conductance-based E/I net
  that trains to completion on SHD with no NaN epochs and bounded dynamics,
  decisively above chance (#calc.round(dales.best_acc_pct, digits: 1)% against the
  #calc.round(r.chance_pct)% chance floor).

  *The cost, and where this points.* Stability is not free: the constrained net
  gives up a few points of peak accuracy
  (#calc.round(dales.best_acc_pct, digits: 1)% vs
  #calc.round(free.best_acc_pct, digits: 1)%), the price of denying the signed
  recurrence its full expressivity. So two threads open. For the _minimal stable
  recipe_, Dale's law now answers it; the queue's remaining probe (exp063, weight
  decay) and the reserved state clamp become alternative stabilisers that may not
  be needed. For _recovering the free net's accuracy without its divergence_,
  those same tools become the live question: can weight decay or a forward-pass
  state clamp tame the signed net and keep its extra accuracy?

  #emph[Caveat.] Single seed, a point contrast, not error-barred. The effect is
  categorical (#free.nan_epochs vs #dales.nan_epochs NaN epochs), so a seed sweep
  is unlikely to reverse it; the confirming 3-seed run is cheap.
]

#let body = [
  == Abstract

  *The claim: Dale's law is the implicit stabiliser that keeps a recurrent
  excitatory/inhibitory (E/I) spiking network trainable on the Spiking Heidelberg
  Digits (SHD) spoken-digit classification benchmark.* Dale's law is the
  biological constraint that a neuron is either excitatory or inhibitory, so its
  outgoing synaptic weights all keep a fixed sign. Enforced during training as a
  non-negativity projection on the recurrent weights, it bounds those weights at
  every step. The question is whether that bound is what separates a net that
  trains to completion from one whose loss overflows to NaN (not-a-number, the
  signature of numerical divergence).

  Two otherwise-identical cells, free (signed recurrence) versus Dale's law, run
  at the coarse timestep where the free net is known to overflow. The constraint
  removes the divergence outright: the free net loses #free.nan_epochs of
  #free.epochs epochs to NaN, Dale's law #dales.nan_epochs, and best test accuracy
  barely moves. Dale's law is the implicit stabiliser.

  == Methods

  #link("/exp061/")[exp061] killed the first hypothesis: the free
  signed-recurrent net's NaN divergence is not exponential-Euler stiffness at a
  coarse integration timestep Δt. Finer Δt made the pre-clip gradient explosion
  _worse_, because quartering Δt quadruples the backpropagation-through-time
  (BPTT) unroll and the recurrent gradient chain blows up. So the divergence is a
  property of the *free recurrence* itself, not the timestep. The
  #link("/ar063/")[plan]'s next candidate is the one structural constraint the
  free net drops: *Dale's law.* Its projection clamps the recurrent weights
  non-negative (and fixes their sign), which bounds the gain of the E→I→E
  (excitatory→inhibitory→excitatory) recurrent loop, plausibly below the runaway
  threshold the free net crosses.

  The test, step by step:

  + *Two cells, one difference.* Hold every parameter fixed except the constraint:
    *free* (signed weights, the exp060/exp061 setting that diverges) versus
    *constrained* (Dale's law).
  + *The coarse-Δt regime.* Run both at the timestep Δt = #(r.dt_ms) ms where the
    free net is known to overflow to NaN (#link("/exp061/")[exp061]; here
    #free.nan_epochs of #r.epochs epochs), at a single seed and full exp060 scale.
  + *The registered metrics.* Measure the plan's free-versus-constrained contrast:
    the NaN-epoch rate (the fraction of the run's training epochs whose loss is
    non-finite), the peak norm of the recurrent excitatory-to-excitatory weight
    matrix ‖W_ee‖, and the best test accuracy.

  #table(
    columns: 2,
    [Parameter], [Value],
    [Dataset], [Spiking Heidelberg Digits (SHD), #(r.max_samples)-sample subset],
    [Model], [COBANet (conductance-based E/I spiking network), #r.config.n_hidden hidden units, all four recurrent weight blocks trainable],
    [Contrast], [free (_--no-dales-law_, signed) vs constrained (_--dales-law_, projected non-negative)],
    [Integration], [$Delta t = #(r.dt_ms)$ ms, trial duration $T = #(r.config.t_ms)$ ms (the setting where the free net diverges)],
    [Seeds], [#r.config.seeds (seed #r.seed), a quick single-seed contrast],
    [Training], [#r.epochs epochs, learning rate #(r.config.lr), weight-decay #(r.config.weight_decay), batch size #r.config.batch_size, surrogate-gradient backpropagation through time],
    [Regulariser], [upper firing-rate bound (threshold $theta_u = #(r.config.fr_reg_upper_theta)$, strength $s_u = #(r.config.fr_reg_upper_strength)$)],
    [Compute], [#r.compute],
  )

  Dale's law is the *only* thing that differs between the two cells, so any gap in
  the stability metrics is attributable to the constraint alone. Both cells run as
  a RunPod fan-out, one pod per cell.

  #quote(block: true)[
    *Kill criterion.* If the Dale's-law net also overflows to NaN at matched
    settings, the constraint is not what stabilises: the difference lies
    elsewhere.
  ]

  == Results

  #figure(
    image("/artifacts/data/exp062/free_vs_dales.svg", width: 100%,
      alt: "Three bar-pair panels (NaN-epoch rate, max recurrent-weight norm, and best test accuracy), free versus Dale's law."),
    caption: [
      Free (red) versus Dale's law (black) across the three registered stability
      metrics, one bar pair per panel. Left to right: (a) NaN-epoch rate (percent
      of training epochs with non-finite loss); (b) peak norm of the recurrent
      excitatory-to-excitatory weight matrix, ‖W_ee‖ (dimensionless); (c) best
      test accuracy (percent), with the dashed line marking the
      #calc.round(r.chance_pct)% chance level for the #r.config.n_classes SHD classes.
      #results-caption
    ],
  )

  #figure(
    image("/artifacts/data/exp062/loss_traces.svg", width: 100%,
      alt: "Test cross-entropy loss per epoch for the free and Dale's-law nets; NaN epochs appear as gaps in the curve."),
    caption: [
      Per-epoch test cross-entropy loss, free (red) versus Dale's law (black).
      The horizontal axis is the training epoch; the vertical axis is test
      cross-entropy loss. NaN epochs (those whose loss is non-finite) appear as
      gaps in the curve: the free net leaves holes where its loss overflows,
      while the constrained net runs continuously to completion.
    ],
  )

  #reading-body
]
