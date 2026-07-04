#let meta = (
  title: "Rate floor stable across a 20× Δt sweep",
  date: "2026-06-02",
  description: "A Δt audit: the exp025 headline E rate stays in a 9.5–14 Hz band across a 20× integration-timestep sweep while accuracy holds and the gamma period is invariant.",
  collection: "gamma-gated-sparsity",
  status: "revising",
)

#let run = json("/artifacts/data/exp044/numbers.json")

#let body = [
  The trained networks this entry uses are produced once in the shared training
  hub, #link("/exp022/")[exp022 (Training)], and reused here rather than retrained.

  == Abstract

  The Δt audit asks whether the #link("/exp025/")[exp025] headline E rate is a
  physical (Hz) property of the trained network or an artefact of the integration
  timestep. The rate stays in a 9.5–14 Hz band across a 20× Δt sweep, accuracy
  holds at 88.5–90.7%, and the gamma cycle period in physical ms is invariant — but
  the rate dependence on Δt is non-monotonic, with Δt = 0.5 ms producing a _lower_
  mean rate than Δt = 0.25 ms.

  == Method

  #table(
    columns: 2,
    [Parameter], [Value],
    [Integration timestep $Delta t$], [0.1 ms],
    [Trial duration $T$], [200 ms],
    [MNIST samples (80/20 stratified split of 2000)],
    [1600 train / 400 test (≈ 2.9% of the 70k-sample MNIST corpus)],
    [Epochs], [100],
  )

  Train one PING from scratch per $Delta t in {0.05, 0.1, 0.25, 0.5, 1.0}$ ms ×
  seed $in {42, 43, 44}$ = 15 cells. Total physical time $T = 200$ ms held constant
  (step count varies 4000 → 200). Batch size 64 throughout — smaller than
  #link("/exp025/")[exp025]'s 256, but matched across the sweep so per-step compute
  and memory stay comparable, and the Δt = 0.05 cells (4000 timesteps × $N_E$ ×
  $N_I$) fit in a single A100. All other PING recipe parameters held to
  #link("/exp025/")[exp025].

  After training, run inference on the test set; report mean E rate (Hz), accuracy,
  and a single-trial raster from seed 42 per Δt for visual cycle-period inspection.

  == Results

  #figure(
    image("/artifacts/data/exp044/dt_sweep.svg", width: 100%),
    caption: [
      Hidden E rate (black) and test accuracy (red) as Δt varies 20× on a log
      scale. Error bars from three seeds.
    ],
  )

  #figure(
    image("/artifacts/data/exp044/raster_strip.pdf", width: 100%),
    caption: [
      Single-trial rasters at each Δt, x-axis in _physical_ ms (not steps). All five
      panels show E and I bursts at the same gamma cadence (≈ 30 ms cycle). The
      cycle physics is Δt-invariant.
    ],
  )

  #figure(
    image("/artifacts/data/exp044/training_curves.svg", width: 100%),
    caption: [
      Top: test accuracy converges by epoch ≈ 10–15 across all Δt. Bottom: test E
      rate has _not_ converged in 30 epochs — every curve is still rising. The
      Δt-dependent rate scaling is a snapshot of medium-tier training at epoch 30,
      not a fixed-point ceiling.
    ],
  )
]
