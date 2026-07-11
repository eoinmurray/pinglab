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

// Results-dependent prose — finalised against the run's actual outcome, with
// every number still read from numbers.json.
#let results-caption = [_What we see_ is written in the Reading section below.]
#let reading-body = [_(finalised once the run's numbers are in.)_]

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
