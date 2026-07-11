#let meta = (
  title: "Does finer Δt stabilise the integration?",
  date: "2026-07-11",
  description: "The SHD program's first stability experiment: sweeping the integration timestep Δt ∈ {1.0, 0.5, 0.25} ms shows finer integration does NOT stabilise the free signed-recurrent net — the NaN-epoch rate does not fall and the pre-clip gradient explosion gets orders of magnitude worse. The kill criterion fires: the divergence is not coarse-Δt stiffness.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
)

// Every number below is read from this run's numbers.json, never hand-typed.
#let r = json("/artifacts/data/exp061/numbers.json")
#let cells = r.cells.sorted(key: c => -c.dt)   // coarse → fine
#let coarse = cells.at(0)                        // Δt = 1.0
#let mid = cells.at(1)                           // Δt = 0.5
#let finest = cells.last()                       // Δt = 0.25
// Order of magnitude of a value (⌊log10⌋), for the runaway gradient norms.
#let oom(x) = calc.round(calc.log(x, base: 10))

#let results-caption = [
  *What we see refutes it.* Not one of the three metrics improves as Δt shrinks.
  The NaN-epoch rate does not fall toward zero — #coarse.nan_epochs of
  #coarse.epochs epochs diverge at Δt = 1.0, #mid.nan_epochs at 0.5, and
  #finest.nan_epochs at 0.25: flat-to-worse, not a descent. The peak pre-clip
  gradient norm moves the wrong way and violently, climbing from ≈ 10#super[#oom(coarse.max_grad_norm)]
  at Δt = 1.0 to ≈ 10#super[#oom(finest.max_grad_norm)] at Δt = 0.25 — roughly
  #(oom(finest.max_grad_norm) - oom(coarse.max_grad_norm)) orders of magnitude
  the wrong way. Only ‖W_ee‖ stays in one range
  (#calc.round(finest.max_wee_norm, digits: 1)–#calc.round(mid.max_wee_norm, digits: 1)),
  so this is not a runaway-weight story.
]

#let reading-body = [
  The coarse-Δt run reproduces #link("/exp060/")[exp060]'s divergence
  (#coarse.nan_epochs of #coarse.epochs epochs NaN, #coarse.nan_forward_batches
  NaN forward passes), so the mechanism is genuinely exercised — this is not a
  null run. But finer Δt does not remove it: at Δt = 0.25 the NaN-epoch rate is
  #finest.nan_epochs of #finest.epochs, no lower than at Δt = 1.0, and the peak
  pre-clip gradient norm is *larger* at every halving, reaching
  ≈ 10#super[#oom(finest.max_grad_norm)] at the finest step.

  *The kill criterion fires.* The plan pre-registered: _if NaN persists at
  Δt = 0.25, coarse integration is not the cause — drop Δt from the recipe and
  look upstream._ It persists. So the divergence is not exp-Euler stiffness at a
  coarse timestep. The mechanism is visible in the sweep itself: quartering Δt
  quarters the step but *quadruples* the BPTT unroll (1000 → 4000 steps), and the
  longer gradient chain through the signed E→I→E recurrence explodes far worse
  than any stiffness a finer step relieves. Δt is not the stabiliser; it leaves
  the recipe.

  *Where this points.* With neither Δt (here) nor weight size (exp060: a NaN at
  epoch 2 with tiny weights) explaining the divergence, the plan's reserved
  fallback — a forward-pass state clamp bounding voltage and conductance — becomes
  the live lead. The reduced-scale CPU cross-check agreed directionally: no NaN at
  128 samples, but the same non-monotonic gradient blow-up peaking at Δt = 0.25.

  #emph[Caveat.] Single seed — point estimates, not error bars. The effect is
  large (orders of magnitude, not margins), so a seed sweep is unlikely to reverse
  the direction; the plan's 3-seed default is the confirming run, and the RunPod
  path (results collected over the S3 volume API) makes it cheap to launch.
]

#let body = [
  == What this checks

  #link("/exp060/")[exp060] showed the free signed-recurrent ceiling trains on
  SHD but diverges intermittently: scattered epochs return NaN, the pre-clip
  gradient norm spikes into the millions, and the E→E recurrent weight grows
  without bound. Crucially a NaN appeared at *epoch 2, when the weights were
  still small* — so the divergence is not weight-growth runaway. The
  #link("/ar063/")[program plan] parks that observation as its first hypothesis:
  the NaN is exp-Euler *integration stiffness* at the coarse Δt = 1.0 ms, and
  halving then quartering Δt should drop the NaN-epoch rate toward zero.

  This is that experiment. It holds exp060's Rung A recipe fixed and varies only
  the integration timestep, Δt ∈ {1.0, 0.5, 0.25} ms, on a single seed
  (#r.seed) — a fast first read, not an error-barred result — and measures the
  plan's three registered stability metrics against Δt:

  - the *NaN-epoch rate* — the fraction of epochs with a non-finite loss or a
    NaN forward pass;
  - the *max pre-clip gradient norm* — the peak global gradient magnitude before
    clipping, the quantity that spiked to millions in exp060;
  - the *max recurrent-weight norm* — the peak $||W_"ee"||$, the block that grew
    unbounded.

  #quote(block: true)[
    *Kill criterion.* If NaN persists at Δt = 0.25 (the finest step), coarse
    integration is not the cause — drop Δt from the recipe and look upstream
    (a forward-pass state clamp).
  ]

  == Method

  #table(
    columns: 2,
    [Parameter], [Value],
    [Dataset], [SHD, #(r.max_samples)-sample subset],
    [Model], [COBANet, 256 hidden, all four recurrent blocks trainable, signed (no Dale's law)],
    [Swept variable], [$Delta t in {1.0, 0.5, 0.25}$ ms · $T = 1000$ ms, so $T_"steps" in {1000, 2000, 4000}$],
    [Seeds], [1 (seed #r.seed) — a quick first read],
    [Training], [#r.epochs epochs, lr 0.001, weight-decay 0.001, batch 32, surrogate-gradient BPTT],
    [Regulariser], [upper firing-rate bound ($theta_u = 100$, $s_u = 0.06$) — load-bearing, from exp060],
    [Compute], [#r.compute],
  )

  Δt is the *only* thing that varies across the three runs; everything else is
  exp060's recipe verbatim, so any change in the stability metrics is
  attributable to Δt alone. Δt = 1.0 reproduces exp060's exact failure setting,
  so the finer-Δt arms are read against a grounded baseline rather than an
  abstract one.

  == Compute

  This sweep ran on a #link("/ar065/")[night-shift] RunPod fan-out — one GPU pod
  per Δt, at the full exp060-matching scale — with results collected over HTTPS.

  That collection path is new, and worth a note because it nearly blocked the
  program. The lab's cloud sandbox permits only outbound HTTPS, and the fan-out's
  original collect step rsyncs artifacts off the shared network volume *over SSH*
  — which the sandbox blocks outright. Pods still fire, train, and write to the
  volume, but their output could not be pulled back. The fix routes around SSH
  entirely: a RunPod network volume is S3-compatible over HTTPS, so the collector
  now reads the trained cells straight off the volume with the S3 API — no
  collector pod, no SSH. A reduced-scale local-CPU run (128 samples) reproduced
  the same directional result as an independent cross-check, but the numbers on
  this page are the full-scale GPU run.

  == Results

  #figure(
    image("/artifacts/data/exp061/stability_vs_dt.svg", width: 100%,
      alt: "Three panels against Δt (coarse to fine): NaN-epoch rate, max pre-clip gradient norm on a log axis, and max E-to-E recurrent-weight norm."),
    caption: [
      The three stability metrics against Δt (coarse → fine, left → right on each
      panel). *What we expect if the plan's hypothesis holds.* All three fall as
      Δt shrinks — the NaN-epoch rate toward zero, the gradient spikes and the
      weight norm toward bounded values. #results-caption
    ],
  )

  #figure(
    image("/artifacts/data/exp061/grad_trace.svg", width: 100%,
      alt: "Max pre-clip gradient norm per epoch, one line per Δt, on a log y-axis."),
    caption: [
      The max pre-clip gradient norm per epoch, one line per Δt (log axis). This
      shows *when* the gradient spikes fire, not just their peak: a stiff
      integration diverges on individual forward passes rather than drifting up
      smoothly.
    ],
  )

  == Reading

  #reading-body
]
