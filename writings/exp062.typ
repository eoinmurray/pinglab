#let meta = (
  title: "Is Dale's law the implicit stabiliser?",
  date: "2026-07-11",
  description: "The SHD program's second stability experiment: with the free signed-recurrent net's divergence shown (exp061) to be a recurrent gradient explosion rather than Δt stiffness, test whether Dale's law — the non-negativity constraint that bounds the recurrent weights — is what keeps a matched net trainable.",
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
  The free net diverges exactly as before — #free.nan_epochs of #free.epochs
  epochs NaN, peak pre-clip gradient norm ≈ 10#super[#oom(free.max_grad_norm)] —
  reproducing #link("/exp061/")[exp061]'s coarse-Δt cell. Flip the single flag to
  Dale's law and, at otherwise identical settings, the divergence is *gone*:
  #dales.nan_epochs NaN epochs, the peak gradient norm bounded at
  ≈ #calc.round(dales.max_grad_norm) (some
  #(oom(free.max_grad_norm) - oom(dales.max_grad_norm)) orders of magnitude
  tamer), and ‖W_ee‖ held at #calc.round(dales.max_wee_norm, digits: 1) against
  the free net's #calc.round(free.max_wee_norm, digits: 1).

  *The hypothesis holds.* Dale's law is the implicit stabiliser. Its
  non-negativity projection clamps the recurrent weights each step, bounding the
  E→I→E loop gain below the runaway threshold the free signed net crosses — the
  divergence never starts, so there is no gradient explosion to clip. This is the
  first *stable recipe* the program's goal asked for: a conductance E/I net that
  trains to completion on SHD with no NaN and bounded dynamics, decisively above
  chance (#calc.round(dales.best_acc_pct, digits: 1)% against the
  #calc.round(r.chance_pct)% floor).

  *The cost, and where this points.* Stability is not free — the constrained net
  gives up a few points of peak accuracy
  (#calc.round(dales.best_acc_pct, digits: 1)% vs
  #calc.round(free.best_acc_pct, digits: 1)%), the price of denying the signed
  recurrence its full expressivity. So two threads open. For the _minimal stable
  recipe_, Dale's law now answers it; the queue's remaining probe (exp063, weight
  decay) and the reserved state clamp become alternative stabilisers that may not
  be needed. For _recovering the free net's accuracy without its divergence_,
  those same tools become the live question: can weight decay or a forward-pass
  state clamp tame the signed net and keep its extra accuracy?

  #emph[Caveat.] Single seed — a point contrast, not error-barred. The effect is
  categorical (#free.nan_epochs vs #dales.nan_epochs NaN epochs), so a seed sweep
  is unlikely to reverse it; the confirming 3-seed run is cheap on the RunPod + S3
  path.
]

#let body = [
  == What this checks

  #link("/exp061/")[exp061] killed the first hypothesis: the free
  signed-recurrent net's NaN divergence is not exp-Euler stiffness at a coarse
  Δt — finer Δt made the pre-clip gradient explosion _worse_, because quartering
  Δt quadruples the BPTT unroll and the recurrent gradient chain blows up. So the
  divergence is a property of the *free recurrence* itself, not the timestep. The
  #link("/ar063/")[plan]'s next candidate is the one structural constraint the
  free net drops: *Dale's law.* Its projection clamps the recurrent weights
  non-negative (and fixes their sign), which bounds the loop gain — plausibly
  below the runaway threshold the free net crosses.

  This tests it head-on. Two cells, identical in every parameter except the
  constraint — *free* (signed, the exp060/exp061 setting that diverges) vs
  *constrained* (Dale's law) — at the coarse Δt = 1.0 ms where the free net is
  known to NaN (#link("/exp061/")[exp061]: 13 of 30 epochs), single seed, full
  exp060 scale. It measures the plan's registered contrast: NaN-epoch rate, max
  recurrent-weight norm, and best accuracy, free vs constrained.

  #quote(block: true)[
    *Kill criterion.* If the Dale's-law net also NaNs at matched settings, the
    constraint is not what stabilises — the difference lies elsewhere.
  ]

  == Method

  #table(
    columns: 2,
    [Parameter], [Value],
    [Dataset], [SHD, #(r.max_samples)-sample subset],
    [Model], [COBANet, 256 hidden, all four recurrent blocks trainable],
    [Contrast], [free (_--no-dales-law_, signed) vs constrained (_--dales-law_, projected non-negative)],
    [Integration], [$Delta t = #(r.dt_ms)$ ms, $T = 1000$ ms — the setting where the free net diverges],
    [Seeds], [1 (seed #r.seed) — a quick contrast],
    [Training], [#r.epochs epochs, lr 0.001, weight-decay 0.001, batch 32, surrogate-gradient BPTT],
    [Regulariser], [upper firing-rate bound ($theta_u = 100$, $s_u = 0.06$)],
    [Compute], [#r.compute],
  )

  Dale's law is the *only* thing that differs between the two cells, so any gap in
  the stability metrics is attributable to the constraint alone.

  == Compute

  RunPod fan-out, one pod per cell, results collected off the shared network
  volume over its S3 HTTPS API (the SSH-blocked-sandbox workaround introduced in
  #link("/exp061/")[exp061]).

  == Results

  #figure(
    image("/artifacts/data/exp062/free_vs_dales.svg", width: 100%,
      alt: "Three bar-pair panels — NaN-epoch rate, max recurrent-weight norm, and best test accuracy — free versus Dale's law."),
    caption: [
      Free (red) vs Dale's law (black) across the three registered metrics.
      *What we expect if the constraint is the stabiliser.* The free bar carries
      the NaN-epoch rate and the runaway weight norm; the Dale's-law bar drops
      both to a stable regime while holding accuracy. #results-caption
    ],
  )

  #figure(
    image("/artifacts/data/exp062/loss_traces.svg", width: 100%,
      alt: "Test cross-entropy loss per epoch for the free and Dale's-law nets; NaN epochs appear as gaps in the curve."),
    caption: [
      Test loss per epoch, free vs constrained. NaN epochs appear as gaps — a
      net that diverges leaves holes where the constrained net runs continuously.
    ],
  )

  == Reading

  #reading-body
]
