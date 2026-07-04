#let meta = (
  title: "Switching the loop on at inference cuts E rate ≈10×",
  date: "2026-05-30",
  description: "Loads a trained COBA network and switches the I-loop on at inference by sweeping ei_strength from 0 to 1; the same weights fire ≈10× slower.",
  collection: "gamma-gated-sparsity",
  status: "revising",
)

#let run = json("/artifacts/data/exp038/numbers.json")

#let body = [
  The trained networks this entry uses are produced once in the shared training
  hub, #link("/exp022/")[exp022 (Training)], and reused here rather than retrained.

  == Abstract

  Loads a trained COBA network and switches the I-loop on at inference by sweeping
  _ei_strength_ from 0 to 1. The same feedforward weights that fire at ≈ 52 Hz
  without the loop fire at ≈ 5 Hz with the loop engaged, and accuracy stays within
  ≈ 12 pp of the COBA baseline. PING gating is a post-hoc sparsity knob — the
  architecture, not the training, supplies the gamma dynamics.

  == Method

  #table(
    columns: 2,
    [Parameter], [Value],
    [Integration timestep $Delta t$], [0.1 ms],
    [Trial duration $T$], [200 ms],
    [MNIST samples (80/20 stratified split of 2000)],
    [1600 train / 400 test (≈ 2.9% of the 70k-sample MNIST corpus)],
    [Epochs], [10],
  )

  The PING and COBA baseline definitions and the training recipe are in
  #link("/exp025/")[exp025]; this entry runs the COBA → PING I-loop transfer probe
  at eval time on the trained baselines. (The input-rate sweep / f–I curve material
  that previously lived here has moved to #link("/exp023/")[exp023], the natural home
  for "architectural response to drive".)

  *Inference-time probe.* The trained COBA baseline (seed 42, $theta_u =$ off) is
  loaded and _ei_strength_ — the I-loop gain — is overridden at eval time across 11
  values from 0 to 1. $W_"in"$ and $W_"out"$ load from the COBA checkpoint;
  $W^(E I)$ and $W^(I E)$ are freshly initialised (the COBA checkpoint stores these
  at zero, so skipping the load leaves a functional I-loop). No retraining.

  == Results

  #figure(
    image("/artifacts/data/exp038/loop_transfer_compound.png", width: 100%),
    caption: [
      Switching the recurrent I-loop on _at inference_ on a trained COBA network —
      no retraining, $W^(E I) \/ W^(I E)$ freshly wired (the COBA checkpoint stores
      them at zero). *Top* — the same feedforward weights fire densely and
      asynchronously at _ei = 0_ (COBA) and in gamma bands at _ei = 1_ (PING); the
      gamma dynamics come from the inhibitory architecture, not from training.
      *Bottom left* — E rate falls ≈ 10× (≈ 60 → 5 Hz) as the loop engages while I
      rises to ≈ 28 Hz; the suppression is continuous in loop strength. *Bottom
      right* — accuracy _degrades_ without retraining, from the ≈ 87% COBA baseline
      to ≈ 63% at full strength (a ≈ 24 pp cost). So the architecture supplies the
      rate-gating for free, but using it well needs training _with_ the loop
      (#link("/exp025/")[exp025]): the sparsity is architectural, the accuracy is
      learned.
    ],
  )

  #figure(
    image("/artifacts/data/exp038/ei_rasters.png", width: 100%),
    caption: [
      The full transition: trained COBA replayed at six inference-time _ei_strength_
      values (same trial, same feedforward weights, a fresh I-loop each row). At
      _ei = 0_ the asynchronous-dense COBA pattern persists; by _ei ≈ 0.4_ the same
      weights produce gamma cycles, sharpening toward _ei = 1_. The rhythm appears
      continuously as the loop is wired in — no retraining at any point.
    ],
  )
]
