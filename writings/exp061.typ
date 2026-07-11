#let meta = (
  title: "Does finer Δt stabilise the integration?",
  date: "2026-07-11",
  description: "The SHD program's first stability experiment: sweep the integration timestep Δt ∈ {1.0, 0.5, 0.25} ms on the free signed-recurrent net and measure whether finer integration drops the NaN-epoch rate and the pre-clip gradient spikes exp060 showed.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
)

// Every number below is read from this run's numbers.json, never hand-typed.
#let r = json("/artifacts/data/exp061/numbers.json")
#let cells = r.cells.sorted(key: c => -c.dt)   // coarse → fine
#let finest = cells.last()

// Results-dependent prose — finalised against the run's actual outcome (below),
// with every number still read from numbers.json so nothing is hand-typed.
#let results-caption = [_What we see_ is written in the Reading section below.]
#let reading-body = [_(finalised once the run's numbers are in.)_]

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

  The sweep is wired for a #link("/ar065/")[night-shift] RunPod fan-out — one pod
  per Δt — and that path runs the full exp060-matching scale on a GPU. This run,
  however, used *local CPU* (#r.compute) at a reduced first-read scale: the lab's
  cloud sandbox permits only outbound HTTPS, and the fan-out's result-collection
  step rsyncs artifacts off the shared volume *over SSH*, which the sandbox
  blocks. Pods can be fired and will train and write to the volume, but their
  output cannot be pulled back from inside the sandbox — so results here come
  from the CPU path, which writes locally and needs no SSH. Fixing collection to
  ride the permitted HTTPS channel is a follow-up; it does not change the science
  this entry measures.

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
