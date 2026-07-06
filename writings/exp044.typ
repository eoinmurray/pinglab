#let meta = (
  title: "Rate floor stable across a 20× Δt sweep",
  date: "2026-06-02",
  description: "A Δt audit: the exp025 headline E rate stays in a 9–14 Hz band across a 20× integration-timestep sweep while accuracy holds and the gamma period is invariant.",
  collection: "gamma-gated-sparsity",
  status: "final",
)


#let body = [
  The trained networks this entry uses are produced once in the shared training
  hub, #link("/exp022/")[exp022 (Training)], and reused here rather than retrained.

  == Abstract

  The Δt audit asks whether the #link("/exp025/")[exp025] headline E rate is a
  physical (Hz) property of the trained network or an artefact of the integration
  timestep. The rate stays within a 9–14 Hz band across a 20× Δt sweep, accuracy
  holds at 90.4–91.4%, and the gamma cycle period in physical ms is invariant. The
  rate is not fully Δt-independent, though: it rises monotonically with the
  timestep, from ≈ 9.2 Hz at Δt = 0.05 ms to ≈ 13.4 Hz at Δt = 1 ms, so a coarser
  step inflates the rate while leaving accuracy and cycle period intact.

  == Method

  #table(
    columns: 2,
    [Parameter], [Value],
    [Integration timestep $Delta t$], [0.05–1.0 ms (swept)],
    [Trial duration $T$], [200 ms],
    [MNIST samples (80/20 stratified split of 500)],
    [400 train / 100 test (≈ 0.7% of the 70k-sample MNIST corpus)],
    [Epochs], [50],
  )

  The #link("/exp022/")[exp022] hub trains one PING per $Delta t in {0.05, 0.1, 0.25, 0.5, 1.0}$ ms ×
  seed $in {42, 43, 44}$ = 15 cells; this entry loads them and evaluates. Total
  physical time $T = 200$ ms is held constant (step count varies 4000 → 200). Batch
  size 64 throughout, smaller than #link("/exp025/")[exp025]'s 256 but matched
  across the sweep so per-step compute and memory stay comparable, and the Δt = 0.05
  cells (4000 timesteps × $N_E$ × $N_I$) fit in a single A100. All other PING recipe
  parameters held to #link("/exp025/")[exp025].

  Run inference on the test set; report mean E rate (Hz), accuracy, and a
  single-trial raster from seed 42 per Δt for visual cycle-period inspection.

  == Results

  #figure(
    image("/artifacts/data/exp044/dt_sweep.svg", width: 100%,
      alt: "Hidden E rate and test accuracy against integration timestep; accuracy is flat while the E rate rises gently with coarser Δt."),
    caption: [
      Hidden E rate (black) and test accuracy (red) as Δt varies 20×; markers are
      per-Δt means over three seeds. Accuracy holds near 91% while the E rate rises
      monotonically from ≈ 9.2 Hz at Δt = 0.05 ms to ≈ 13.4 Hz at Δt = 1 ms.
    ],
  )

  #figure(
    image("/artifacts/data/exp044/raster_strip.png", width: 100%,
      alt: "Single-trial E and I spike rasters at five integration timesteps, plotted against physical time; the burst cadence lines up across all five."),
    caption: [
      Single-trial rasters at each Δt, x-axis in _physical_ ms (not steps). All five
      panels show E (black) and I (red) bursts locked to the same gamma cadence. The
      cycle physics is Δt-invariant.
    ],
  )

  #figure(
    image("/artifacts/data/exp044/training_curves.svg", width: 100%,
      alt: "Per-cell test accuracy and E rate versus epoch, coloured by Δt; accuracy converges early while the E rate is still climbing at epoch 50."),
    caption: [
      Top: test accuracy converges by epoch ≈ 10–15 across all Δt. Bottom: test E
      rate has _not_ converged at epoch 50; every curve is still rising, ordered by
      Δt. The Δt-dependent rate is a snapshot at epoch 50, not a fixed-point ceiling.
    ],
  )
]
