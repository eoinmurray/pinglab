#let meta = (
  title: "Log",
  date: "2026-07-11",
  description: "The lab notebook for the SHD program: decisions, failures, aborts, and anomalies, one section per session, newest first. Digest at the top for the morning read.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
)

#let body = [
  The program's notebook — the _why_ and _what-happened_ that the run-stamps do
  not capture. Append-only, timestamped, newest first. Relaxed and chronological
  by design;
  still a published page, so run numbers cite the record, never hand-typed. The
  goal and the queue live in the #link("/ar063/")[plan]; the operating policy in
  the #link("/ar065/")[night-shift].

  == Digest

  _Latest, for the 30-second morning read. Updated 2026-07-11 20:35 BST._

  *We have a stable recipe.* #link("/exp062/")[exp062] confirmed *Dale's law is
  the implicit stabiliser*: at the exact settings where the free signed net NaNs
  13/30 epochs, the Dale's-law net trains *NaN-free* (0/30) with the gradient
  bounded (3·10⁴ → ≈ 190) and W_ee bounded (9.2 → 5.4), costing a few points of
  accuracy (56.8% → 53.2%). The program's core goal — a conductance E/I net that
  trains stably on SHD — now has a working answer.

  Getting there: #link("/exp061/")[exp061] *killed the Δt hypothesis* (finer Δt
  made the gradient explosion orders of magnitude worse, not better — it is a
  recurrent-unroll gradient explosion, not coarse-Δt stiffness).

  *Reframe for the remaining queue.* Stability is solved by the constraint, so
  the open question shifts: can weight decay (exp063) or the forward-pass state
  clamp stabilise the _free_ signed net and keep its extra accuracy, rather than
  paying Dale's law's accuracy tax?

  *Infra (resolved).* SSH-blocked sandbox → the pod collector's rsync-over-SSH
  fails; fixed by reading the RunPod volume over its *S3 HTTPS API*
  (`collect_via_s3`). RunPod path fully usable.

  == Sessions

  === 2026-07-11 20:35 BST — exp062: Dale's law is the stabiliser (supported)

  Ran the second stability probe; the program's goal is met.

  - *exp062 — Dale's law free vs constrained (done, supported).* Two cells,
    identical but for the constraint, at Δt = 1.0 where the free net diverges.
    #link("/exp062/")[Result]: the free net NaNs 13/30 epochs (peak gradient
    ≈ 3·10⁴, W_ee 9.2); the Dale's-law net trains *NaN-free* (0/30, gradient
    ≈ 190, W_ee 5.4) at 53.2% vs the free net's 56.8%. The non-negativity
    projection bounds the E→I→E loop gain below the runaway threshold, so the
    divergence never starts. First stable recipe achieved.

  - *Direction.* The remaining queue changes meaning. exp063 (weight decay) and
    the state clamp are no longer needed _for stability_ — Dale's law provides
    it — but they are the way to ask whether the free signed net's extra accuracy
    can be kept without its divergence. That is the more interesting question now.

  === 2026-07-11 20:10 BST — exp061 killed the Δt hypothesis; RunPod collect fixed

  Ran the first stability experiment and solved a compute blocker on the way.

  - *exp061 — Δt sweep (done, kill).* Swept Δt ∈ {1.0, 0.5, 0.25} ms on the free
    signed-recurrent net, single seed, full exp060 scale, on RunPod.
    #link("/exp061/")[The result] refutes the plan's hypothesis: NaN is
    reproduced at the coarse Δt (so the mechanism is real) but *not removed* by
    finer Δt — it persists at Δt = 0.25, and the peak pre-clip gradient norm
    explodes monotonically worse (3·10⁴ → 5·10¹² → 4·10¹⁷) because quartering Δt
    quadruples the BPTT unroll (1000 → 4000 steps). Kill criterion fires; the
    divergence is a gradient explosion over the recurrent unroll, not exp-Euler
    stiffness. Δt leaves the recipe.

  - *Instrumentation.* Added two per-epoch metrics to the trainer the sweep
    needed but that were previously invisible: `grad_norm_max` (peak pre-clip
    global norm; only the mean was recorded) and `nan_forward_batches` (NaN
    forward passes were silently skipped). Additive only.

  - *Compute blocker, fixed.* The RunPod fan-out's collector rsyncs off the
    shared volume over SSH; the cloud sandbox blocks outbound :22, so pods
    trained fine but results were stranded on the volume. Routed around it: a
    RunPod network volume is S3-compatible over HTTPS, so `collect_via_s3` reads
    the trained cells straight off the volume with the S3 API. The earlier
    "failed" plumbing pods turned out to have trained correctly all along — only
    collection was broken.

  === 2026-07-11 17:36 BST — program opened

  Stood the program up end to end in one sitting.

  - *Built the substrate.* SHD training now works through the CLI: an event-based
    loader (official train/test split, lazy binning to the model's dt), all four
    recurrent blocks made trainable (the W_ii CLI path was a latent dead flag,
    now fixed), and an AdamW weight-decay knob. The
    #link("/exp059/")[data-look entry] confirmed the dataset is clean and
    class-separable.

  - *Ran the first training smoke* (#link("/exp060/")[exp060], Rung A: free
    signed-recurrent ceiling, dt 1.0, 1000-sample subset). It trains — loss
    3.18 → 0.56, best test accuracy 61% against a 5% floor — so the pipeline is
    sound. But the free dynamics are unstable: scattered epochs return NaN
    train/test loss, pre-clip gradient norms spike to millions, and the E→E
    recurrent weight grows without bound.

  - *Chased the instability and was wrong once.* First guess was unbounded weight
    growth; added weight decay. It tamed the gradient explosion (7.5M → ~900) and
    lifted final accuracy (50% → 58%), but did _not_ bound W_ee at 1e-3 and made
    the NaN _more_ frequent — and a NaN showed up at epoch 2 with tiny weights.
    So the root cause is not weight size; it is an intermittent forward-pass
    divergence of the signed conductance dynamics, most likely exp-Euler
    stiffness at coarse dt. Parked for exp061.

  - *Reframed the program.* What began as debugging is the actual goal: a stable
    training recipe, with each ingredient attributed. Queued exp061 (dt sweep),
    exp062 (Dale's law as the implicit stabiliser), exp063 (weight-decay sweep),
    with a forward-state clamp held in reserve.
]
