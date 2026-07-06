#let meta = (
  title: "Training — the collection's shared cells",
  date: "2026-06-28",
  description: "The gamma-gated-sparsity collection's single training hub: every cell is trained once to a shared root and the analysis notebooks load those weights instead of retraining.",
  collection: "gamma-gated-sparsity",
  status: "revising",
)

#let body = [
  == Abstract

  The collection's single training hub: every cell is trained once to a shared root, and the analysis notebooks load those weights instead of retraining (train-once / reuse-many). Status *draft* — nothing is trained to the gamma standard yet.

  == Methods

  === 1. The compute cost

  Each trial is surrogate-gradient backprop-through-time: 2000 timesteps ($T = 200$ ms at $Delta t = 0.1$ ms) over a 1280-neuron conductance network. What makes it costly:

  - *Fine timestep.* COBA synapses inject $g (V - E)$, stiffening the membrane so $Delta t = 0.1$ ms — 10× finer than a current-based model, and non-optional: the gamma rhythm lives there.
  - *Sequential.* The 2000 steps are a dependency chain no hardware parallelises, so the job is *bandwidth-bound* — a GPU beats a CPU only ≈ 10×, not the 50–100× of large matmuls.
  - *Memory-heavy.* The backward pass stores every step's activations — ≈ 12 GB at batch 256.
  - *Scale.* 87 cells × 50 epochs ≈ *185 A100-GPU-hours* (≈ 49M sample-forwards) — ≈ 159 days on CPU, so every cell trains on a GPU.

  === 2. Gold star trainings

  87 cells across five families, each fixing its time constants, timestep, and MNIST fraction (re-split 80/20). The standard is $tau_"AMPA" = 2$ ms, $tau_"GABA" = 6$ ms (loop in gamma, ≈ 44 Hz); the sweeps each vary one axis — spike budget, τ_GABA, timestep (#link("/exp044/")[exp044]), or init.

  #table(
    columns: 7,
    align: (left, right, right, right, left, right, right),
    table.header([*Family*], [$tau_"GABA"$ (ms)], [$Delta t$ (ms)], [Epochs], [MNIST], [Cells], [At spec]),
    [canonical (θ_u off)], [6], [0.1], [50], [all (70k)], [6], [0 / 6],
    [θ_u sweep], [6], [0.1], [50], [10%], [36], [0 / 36],
    [τ_GABA ladder], [4.5 – 27], [0.1], [50], [10%], [18], [0 / 18],
    [Δt sweep], [6], [0.05 – 1.0], [50], [10%], [15], [0 / 15],
    [init variants], [6], [0.1], [50], [10%], [12], [0 / 12],
    [*total*], [], [], [], [], [*87*], [*0 / 87*],
  )

  The *canonical* family (coba, ping at θ_u = off) is the full-MNIST reference the sweeps are read against. _At spec_ means trained to the gamma standard — *none yet*: the cells on disk from the original notebooks are at the old standard (2.9–5% MNIST, $tau_"GABA" = 9$ ms), so the run trains all 87 fresh.

  Choices behind the table:

  - *50 epochs, halved from 100.* Accuracy plateaus by ≈ 15–20 epochs (#link("/exp024/")[exp024]); 50 keeps a ≈ 30-epoch tail for the post-convergence _dynamics_ the collection studies (rate drift, confidence inflation) while nearly halving the run.
  - *3 seeds throughout.* Every cell — canonical and every sweep — trains three seeds (42, 43, 44), so each point on each frontier carries a cross-seed band and the headline effects (PING's rate attractor) read as robust, not one-run luck. The θ_u interior used to be single-seed; it no longer is.
  - *Full MNIST vs 10%.* The canonical reference carries the headline numbers, so it sees all 70k; the sweeps need only the trend across their parameter, so 10% suffices and cuts their cost tenfold.

  === 3. Compute options

  Bandwidth is the main driver but not the whole story — the RTX 4090 below has half the A100's bandwidth yet measured _faster_. Costs and wall-clocks cover the full 50-epoch registry; measured where the card was to hand, projected for the 5090:

  #table(
    columns: 5,
    align: (left, left, right, right, left),
    table.header([*Option*], [GPU (bandwidth)], [Samples/s], [Full-run cost], [Wall-clock]),
    [Modal], [A100 (2.0 TB/s)], [74], [≈ \$615], [≈ half a day (fans out)],
    [RunPod / Vast.ai], [RTX 4090 fleet (1.0 TB/s)], [100 (meas.)], [≈ \$47 – 95], [≈ 10 hr on ≈ 15 pods (≈ 5.7 d serial)],
    [Cambridge Wilkes3], [A100], [74], [≈ £102], [offline through 2026],
    [benjy (CUED, shared)], [A6000 (768 GB/s)], [26.5], [£0], [≈ 22 days],
    [Owned workstation], [RTX 5090 (1.8 TB/s)], [≈ 150 (proj.)], [≈ £4,700 once], [≈ 3.8 days],
  )

  The 4090 (≈ 100 samples/s, measured) beats both its bandwidth projection (≈ 37) and the A100, so the 5090 is scaled from it (≈ 1.5×). Wilkes3 is cheapest-and-fast but offline through 2026; benjy is one shared GPU forcing an older PyTorch.

  ==== The compute option we selected

  The run fans out across RunPod *RTX 4090s*, split by stakes: the 6 heavy canonical cells (≈ 9.7 hr each) on *secure on-demand* — non-preemptible, so a mid-run death is rare — and the 81 light sweep cells (≈ 1 hr each) on cheaper *community* pods, where a lost cell costs an hour, not a day. Together ≈ \$67, against Modal's ≈ \$615, and unlike the 5090 the 4090s actually provision. A pre-baked cu128 Docker image lets pods boot ready to train; a launcher buckets the 87 cells so the six canonical cells land on separate pods (pinning wall-clock to the ≈ 9.7 hr floor of one canonical cell), with a done-marker check keeping the run idempotent and resumable. Artifacts sync back to the shared root; the figures below build locally. At ≈ 15 pods (6 secure + 9 community) it finishes in ≈ 10 hours — more do not help, since one canonical cell cannot be split. The owned *5090* is the long-term answer for routine iteration.

  === 4. Saved per cell

  Each training writes one directory under the shared root (_src/artifacts/notebooks/training/<cell>/_), with the same files every time — the contract the analysis notebooks read:

  #table(
    columns: 2,
    align: (left, left),
    table.header([*File*], [*Holds*]),
    [weights.pth · weights\_final.pth], [trained model state (all parameter tensors) — best-accuracy epoch *and* final epoch, which differ once the rate attractor drifts past the accuracy plateau (exp024).],
    [config.json], [full run config — model, lr, epochs, dt, t\_ms, tau\_gaba, θ_u, w\_in, ei\_strength, ei\_ratio, readout\_w\_out\_scale, max\_samples, n\_params, and provenance (git sha, run id, env hash).],
    [metrics.json], [summary + per-epoch history (below).],
    [metrics.jsonl], [the per-epoch records, one JSON per line (live streaming log).],
    [test\_predictions.json], [predicted vs true label per test image.],
    [output.log · run.sh], [training stdout; the exact re-runnable train command.],
  )

  Top-level fields in metrics.json: *mode, schema_version, model* (run mode, artifact-schema version, model name); *run_finished_at*; *config* (key fields incl. the sweep axes; full set in config.json); *init, end* (dynamics before/after — rate_e, rate_i, cv, act, contrast, f0_hz); *epochs* (per-epoch records, below); *best_acc, best_epoch*; *total_elapsed_s*; *perf* (device, torch, peak memory, epoch times, samples/sec).

  Each per-epoch record — one entry in the _epochs_ list, one line in metrics.jsonl:

  #table(
    columns: 2,
    align: (left, left),
    table.header([*Field*], [*Holds*]),
    [ep, acc], [epoch number; test accuracy (%).],
    [loss, test\_loss], [mean train / test cross-entropy loss.],
    [rate\_e, rate\_i, test\_rate\_e, test\_rate\_i], [mean E / I firing rate, train and test (Hz).],
    [cv, act, contrast, f0\_hz], [ISI coefficient of variation; active hidden fraction; E-population rhythmicity (lobe−trough contrast in $[0,1)$) and gamma frequency (Hz, from the autocorrelogram lobe lag; null when no rhythm resolves).],
    [lobe\_lag\_ms, trough\_lag\_ms, iei\_mode\_lag\_ms, lobe\_to\_trough], [raw rhythmicity scalars behind contrast/f0 — autocorrelogram lobe and trough lags, inter-event-interval mode lag, unbounded lobe/trough ratio.],
    [lr, grad\_norm], [learning rate; clipped global gradient norm (epoch mean).],
    [grad\_norms, grad\_ratios, weight\_norms], [per-weight-type gradient norm, update/weight ratio, and Frobenius weight norm — keyed by parameter name: W\_ff.0 (input), W\_ff.1 (readout), plus W\_ei.1, W\_ie.1 only in the init family's trainable-loop cells (gnorm\_\_<param> in the jsonl).],
    [test\_margin, test\_confidence, test\_logit\_scale], [logit-discrimination diagnostics (#link("/exp024/")[exp024]) — true−runner-up logit margin, softmax confidence in the answer, mean $|"logit"|$.],
    [skipped\_steps, new\_best, samples], [steps skipped on non-finite gradients; new-best flag; samples seen.],
    [elapsed\_s, train\_compute\_s, eval\_s, observe\_s], [cumulative wall time + per-epoch timing breakdown.],
  )

  == Results

  === 1. Canonical reference (θ_u off, all MNIST)

  coba and ping at θ_u = off, all 70k images, seeds 42/43/44 (6 cells) — the full-data baseline. Unconstrained, the two architectures are essentially tied: coba reaches ≈ 95.5 % and ping ≈ 94.0 %, with near-zero across-seed spread. This is the reference point from which the spike-budget sweep pulls them apart.

  #figure(
    image("/artifacts/data/exp022/curves__canonical.png", width: 100%),
    caption: [Test accuracy over epochs, canonical full-MNIST cells (coba dashed, ping solid; three seeds each). The two loops reach parity when no spike budget is imposed.],
  )

  === 2. θ_u spike-budget sweep

  coba and ping across θ_u ∈ off, 5, 2, 1, 0.5, 0.2, three seeds each (36 cells) — so the accuracy–rate frontier carries error bars at every point. θ_u is the spike budget in spikes/trial. This is the headline result: as the budget tightens, ping degrades gracefully (91.6 → 86.5 %) while coba collapses (90.7 → 60.1 %), a gap that widens monotonically from ≈ 0 to ≈ 26 points. ping's γ-rhythm gates sparsity; coba's bare feedforward code cannot pay the budget without shedding accuracy.

  #figure(
    image("/artifacts/data/exp022/curves__theta_u.png", width: 100%),
    caption: [Test accuracy over epochs across the θ_u sweep. Tighter budgets (smaller θ_u) plateau lower — the spike-economy trade-off — and coba falls far faster than ping.],
  )

  === 3. τ_GABA ladder

  ping across τ_GABA ∈ 4.5, 6, 9, 12, 18, 27 ms, three seeds each (18 cells). Accuracy is largely insensitive to inhibitory decay (≈ 88–92 % across the ladder), but the *rhythm* is not: measured from the trained networks, the γ-frequency falls monotonically with τ_GABA (≈ 50 Hz at 4.5 ms → ≈ 19 Hz at 27 ms), sitting at ≈ 45 Hz at the canonical τ_GABA = 6 ms — matching the operating point (Appendix).

  #figure(
    image("/artifacts/data/exp022/curves__tau_gaba.png", width: 100%),
    caption: [Test accuracy over epochs across the τ_GABA ladder; cells converge to similar accuracy regardless of inhibitory decay.],
  )

  === 4. Δt sweep

  ping across Δt ∈ 0.05, 0.1, 0.25, 0.5, 1.0 ms (physical T fixed), three seeds each (15 cells) — the documented timestep exception. Accuracy is flat across the sweep (≈ 90.4–91.4 %): the integrator is robust to timestep from 0.1 to 1.0 ms, and the 0.05 ms cells (which need ≈ 31 GB and so ran on a 5090) agree.

  #figure(
    image("/artifacts/data/exp022/curves__dt.png", width: 100%),
    caption: [Test accuracy over epochs across the integration-timestep sweep; accuracy is insensitive to Δt over the tested range.],
  )

  === 5. Init variants

  ping with four recurrent-loop inits — frozen PING, trainable from PING / zero / small seed — three seeds each (12 cells). All reach ≈ 89–91 %, but only the frozen-PING control keeps the true E/I regime (E ≈ 10 Hz, I ≈ 62 Hz): the trainable-loop cells drift toward a feedforward code (high E, low or zero I) — the zero-init cells never engage inhibition at all (I ≈ 0 Hz). Comparable accuracy, but the rhythm is not learned when it is not built in.

  #figure(
    image("/artifacts/data/exp022/curves__init.png", width: 100%),
    caption: [Test accuracy over epochs across the recurrent-loop inits; trainable-loop cells learn noisier curves than the frozen control.],
  )

  == Appendix — per-config spike rasters (digit 0)

  The same fixed MNIST image (digit 0, sample 0) sent through each trained network (seed 42 as the per-config representative), so every raster is directly comparable. E cells sit below the divider, I cells above; the lower panel is the 1 ms population rate. These are the visual counterpart to the firing-rate and rhythm numbers above — sparse γ-rhythmic PING versus dense asynchronous COBA, and how each regime deforms under the sweeps.

  === A.1 Canonical reference

  #figure(
    image("/artifacts/data/exp022/rasters/coba__canonical__seed42.png", width: 100%),
    caption: [COBA, canonical = off · all MNIST.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/ping__canonical__seed42.png", width: 100%),
    caption: [PING, canonical = off · all MNIST.],
  )

  === A.2 θ_u spike-budget sweep

  #figure(
    image("/artifacts/data/exp022/rasters/coba__off__seed42.png", width: 100%),
    caption: [COBA, theta u = off.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/coba__tu5__seed42.png", width: 100%),
    caption: [COBA, theta u = 5.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/coba__tu2__seed42.png", width: 100%),
    caption: [COBA, theta u = 2.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/coba__tu1__seed42.png", width: 100%),
    caption: [COBA, theta u = 1.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/coba__tu0p5__seed42.png", width: 100%),
    caption: [COBA, theta u = 0.5.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/coba__tu0p2__seed42.png", width: 100%),
    caption: [COBA, theta u = 0.2.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/ping__off__seed42.png", width: 100%),
    caption: [PING, theta u = off.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/ping__tu5__seed42.png", width: 100%),
    caption: [PING, theta u = 5.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/ping__tu2__seed42.png", width: 100%),
    caption: [PING, theta u = 2.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/ping__tu1__seed42.png", width: 100%),
    caption: [PING, theta u = 1.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/ping__tu0p5__seed42.png", width: 100%),
    caption: [PING, theta u = 0.5.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/ping__tu0p2__seed42.png", width: 100%),
    caption: [PING, theta u = 0.2.],
  )

  === A.3 τ_GABA ladder

  #figure(
    image("/artifacts/data/exp022/rasters/ping__tg4p5__seed42.png", width: 100%),
    caption: [PING, tau gaba = τ=4.5.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/ping__tg6__seed42.png", width: 100%),
    caption: [PING, tau gaba = τ=6.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/ping__tg9__seed42.png", width: 100%),
    caption: [PING, tau gaba = τ=9.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/ping__tg12__seed42.png", width: 100%),
    caption: [PING, tau gaba = τ=12.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/ping__tg18__seed42.png", width: 100%),
    caption: [PING, tau gaba = τ=18.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/ping__tg27__seed42.png", width: 100%),
    caption: [PING, tau gaba = τ=27.],
  )

  === A.4 Δt sweep

  #figure(
    image("/artifacts/data/exp022/rasters/ping__dt0p05__seed42.png", width: 100%),
    caption: [PING, dt = dt=0.05.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/ping__dt0p1__seed42.png", width: 100%),
    caption: [PING, dt = dt=0.1.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/ping__dt0p25__seed42.png", width: 100%),
    caption: [PING, dt = dt=0.25.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/ping__dt0p5__seed42.png", width: 100%),
    caption: [PING, dt = dt=0.5.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/ping__dt1__seed42.png", width: 100%),
    caption: [PING, dt = dt=1.],
  )

  === A.5 Init variants

  #figure(
    image("/artifacts/data/exp022/rasters/frozen_ping__seed42.png", width: 100%),
    caption: [PING, init = frozen_ping.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/trainable_ping_init__seed42.png", width: 100%),
    caption: [PING, init = trainable_ping_init.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/trainable_zero_init__seed42.png", width: 100%),
    caption: [PING, init = trainable_zero_init.],
  )

  #figure(
    image("/artifacts/data/exp022/rasters/trainable_small_init__seed42.png", width: 100%),
    caption: [PING, init = trainable_small_init.],
  )

]
