#let meta = (
  title: "PING tolerates 80% dropped spikes but collapses on added noise",
  date: "2026-05-30",
  description: "Perturbs the hidden spike stream of trained PING and COBA at inference. PING forgives most dropped spikes but breaks under a small rate of added Poisson noise; COBA is flat to both. The gamma cycle made visible.",
  collection: "gamma-gated-sparsity",
  status: "final",
)

// Provenance (HOUSESTYLE H9): every run number below is read from the run's
// numbers.json, never hand-typed, so a re-run updates the prose automatically.
#let run = json("/artifacts/data/exp037/numbers.json")
#let cfg = run.config
#let mean(a) = a.sum() / a.len()

// Trained-baseline hidden E rates + accuracy (θ_u off, averaged over seeds).
#let base = run.baseline_results
#let coba_base_rate = calc.round(mean(base.filter(r => r.model == "coba" and r.theta_u == none).map(r => r.rate_e)))
#let ping_base_rate = calc.round(mean(base.filter(r => r.model == "ping" and r.theta_u == none).map(r => r.rate_e)))
#let coba_base_acc = calc.round(mean(base.filter(r => r.model == "coba" and r.theta_u == none).map(r => r.final_acc)))
#let ping_base_acc = calc.round(mean(base.filter(r => r.model == "ping" and r.theta_u == none).map(r => r.final_acc)))
#let rate_ratio = calc.round(coba_base_rate / ping_base_rate)

// Perturbation sweep points (drop = fraction of emitted spikes; add = Hz/neuron).
#let pert = run.perturbation
#let at(model, mode, level) = pert.filter(r => r.model == model and r.mode == mode and calc.abs(r.level - level) < 0.001).first().acc
#let ping_drop80 = calc.round(at("ping", "drop", 0.8))
#let ping_drop_full = calc.round(at("ping", "drop", 1.0))
#let coba_add_max = calc.round(at("coba", "add", 40.0))
#let add_max = calc.round(calc.max(..pert.filter(r => r.mode == "add").map(r => r.level)))
// PING add "knee": the lowest added rate at which test accuracy first drops below 80%.
#let ping_add_knee = calc.round(pert.filter(r => r.model == "ping" and r.mode == "add" and r.acc < 80).sorted(key: r => r.level).first().level)
// The add panel reads as a percentage of each model's own baseline E rate.
#let ping_base_exact = mean(base.filter(r => r.model == "ping" and r.theta_u == none).map(r => r.rate_e))
#let coba_base_exact = mean(base.filter(r => r.model == "coba" and r.theta_u == none).map(r => r.rate_e))
#let ping_add_knee_pct = calc.round(ping_add_knee / ping_base_exact * 100)
#let coba_add_max_pct = calc.round(add_max / coba_base_exact * 100)

#let body = [
  The trained networks this entry uses are produced once in the shared training
  hub, #link("/exp022/")[exp022 (Training)], and reused here rather than retrained.

  == Abstract

  Perturbs the hidden spike stream of trained PING and COBA networks at inference
  (drop spikes, add spikes) to ask whether the PING rate floor is dynamical or
  informational. PING tolerates dropping ≈ 80% of its emitted spikes (accuracy
  #ping_drop80% at that level, from #ping_base_acc% unperturbed) yet collapses once
  added Poisson noise passes ≈ #ping_add_knee_pct% of its own baseline rate. COBA is
  roughly flat to both, holding #coba_add_max% across the whole add sweep (which
  reaches only ≈ #coba_add_max_pct% of its far higher baseline). The asymmetry, drops
  forgiven while adds break the gating, is the gamma cycle made visible: a structural
  feature of the architecture, not a readout-side trade-off.

  == Method

  #table(
    columns: 2,
    [Parameter], [Value],
    [Integration timestep $Delta t$], [#cfg.dt ms],
    [Trial duration $T$], [#cfg.t_ms ms],
    [Held-out MNIST test samples], [#cfg.max_samples],
    [Baseline training epochs (in #link("/exp022/")[exp022])], [#cfg.epochs],
  )

  The PING and COBA baseline definitions and training recipe are in
  #link("/exp025/")[exp025], where the rate-floor mechanism is also worked out. This
  entry tests whether the floor is *dynamical* (locked by the cycle period) or
  *informational* (locked by the readout's spike-count requirement) by perturbing
  the hidden spike stream of the trained networks at inference.

  *The perturbation.* A per-step callback on the COBANet's _\_hidden_perturb_fn_
  slot fires every timestep of every trial, with no warm-up, schedule, or exclusion.
  Trials are $T = #cfg.t_ms$ ms at $Delta t = #cfg.dt$ ms → 2000 fires per trial,
  applied across the held-out test set against the same trained network. At each
  timestep $t$:

  + _update conductances_ from the previous step's spikes:
    $g_e^((t)) <- g_e^((t-1)) dot d_"AMPA" + bold(tilde(s))^(E,(t-1)) W_(e e) +
    "input"^((t)) W_"in"$, with the analogous update for $g_i^((t))$ from
    $bold(tilde(s))^(I,(t-1)) W_(i e)$ and the E→I conductance from
    $bold(tilde(s))^(E,(t-1)) W_(e i)$;
  + _LIF step_ integrates $V^((t))$ and emits raw spike vectors
    $bold(s)^(E,(t)) in {0,1}^(B times N_E)$,
    $bold(s)^(I,(t)) in {0,1}^(B times N_I)$;
  + _perturbation callback_ rewrites the raw vectors →
    $bold(tilde(s))^(E,(t)), bold(tilde(s))^(I,(t))$;
  + _record_ the perturbed vectors into the spike buffer;
  + _readout accumulator_ adds $bold(tilde(s))^(E,(t)) W_"out"$ to the mem-mean
    integrator.

  Step 1 of timestep $t+1$ consumes the perturbed spikes through
  $W_(e e), W_(e i), W_(i e)$: a dropped E spike fails to drive I next step and
  contributes nothing to the readout; an injected I spike adds inhibition next step
  and counts in the rate metric. E and I get the same mode and level with
  independent draws.

  *Drop mode.* For each (batch, neuron, timestep) slot $i$, draw
  $u_i tilde "Uniform"(0,1)$ i.i.d. and keep each emitted spike with probability
  $1 - p_"drop"$:

  $ tilde(s)_i = s_i dot bb(1)[u_i >= p_"drop"]. $

  Drop thins the count fed to both the readout and the next-step conductance update
  while leaving the E→I→E loop intact. Sweep $p_"drop" in {0.0, 0.1, ..., 1.0}$, 11
  levels.

  *Add mode.* For each slot draw $u_i tilde "Uniform"(0,1)$ i.i.d. and flip silent
  slots on at Poisson statistics, phase-independent:

  $ tilde(s)_i = min(s_i + bb(1)[u_i < r_"add" dot Delta t \/ 1000], 1). $

  Sweep $r_"add" in {0, 2, ..., #add_max}$ Hz per neuron, 21 levels, applied equally
  to E and I. Because the two architectures sit at very different baselines
  (COBA ≈ #coba_base_rate Hz per E cell versus PING ≈ #ping_base_rate Hz, a factor of
  ≈ #rate_ratio), a fixed added rate is a much larger relative insult to PING than to
  COBA, so the right panel of Figure 1 expresses the added rate as a percentage of
  each model's own baseline E rate (the architecture-fair view). The per-step RNG is
  seeded separately from the input encoder, so the Poisson input stream matches the
  unperturbed baseline. Total: 2 models × (11 drop + 21 add) = 64 forward passes.

  == Results

  #figure(
    image(
      "/artifacts/data/exp037/perturbation_curves.svg",
      width: 100%,
      alt: "Two panels of test accuracy versus perturbation level for COBA (red) and PING (black), both x-axes in percent. Left: accuracy versus percent of emitted spikes dropped. Right: accuracy versus added Poisson noise as a percent of each model's own baseline rate. PING falls steeply on the right and reaches chance; COBA's sweep spans only a small percent of its high baseline and stays flat.",
    ),
    caption: [
      *Left (drop):* Bernoulli mask, percent of emitted spikes dropped. *Right (add):*
      Poisson injection as a percent of each model's own baseline E rate, so the two
      architectures are insult-matched. The asymmetry is direct: PING (black) holds
      accuracy to ≈ #ping_drop80% at 80% drop but collapses once added noise passes
      ≈ #ping_add_knee_pct% of its baseline, while COBA (red) stays at #coba_add_max%
      across its whole sweep, which reaches only ≈ #coba_add_max_pct% of its far higher
      baseline. The dashed line marks chance.
    ],
  )

  #figure(
    image(
      "/artifacts/data/exp037/perturb_rasters__drop__ping.png",
      width: 100%,
      alt: "Three stacked single-trial rasters of trained PING at drop levels 0, 50, and 100 percent; E spikes in black above I spikes in red. The gamma banding is preserved at 0 and 50 percent and silent at 100 percent.",
    ),
    caption: [
      Trained PING replayed on the same MNIST digit 0 trial across three drop levels
      (0, 50, 100%); E (black) above I (red). The gamma cadence persists as spikes are
      thinned and only vanishes at total drop: dropping spikes thins the train without
      injecting phase-incoherent activity.
    ],
  )

  #figure(
    image(
      "/artifacts/data/exp037/perturb_rasters__add__ping.png",
      width: 100%,
      alt: "Three stacked single-trial rasters of trained PING at added Poisson rates of 0, 20, and 40 Hz; E spikes in black above I spikes in red. The gamma banding dissolves into asynchronous firing as the added rate rises.",
    ),
    caption: [
      Trained PING replayed across three added-noise levels (0, 20, 40 Hz per neuron).
      The gamma banding dissolves into asynchronous firing as the added rate rises,
      matching the accuracy cliff in the previous figure.
    ],
  )

  #figure(
    image(
      "/artifacts/data/exp037/perturb_rasters__drop__coba.png",
      width: 100%,
      alt: "Three stacked single-trial rasters of trained COBA at drop levels 0, 50, and 100 percent; dense asynchronous E firing that thins uniformly with drop and is silent at 100 percent.",
    ),
    caption: [
      Trained COBA replayed across the same drop sweep. With no cycle to preserve,
      dropping spikes just thins a uniform asynchronous mean.
    ],
  )

  #figure(
    image(
      "/artifacts/data/exp037/perturb_rasters__add__coba.png",
      width: 100%,
      alt: "Three stacked single-trial rasters of trained COBA at added Poisson rates of 0, 20, and 40 Hz; dense asynchronous firing whose mean rate rises but whose structure is unchanged.",
    ),
    caption: [
      Trained COBA replayed across the same add sweep. Added spikes blend into COBA's
      asynchronous mean: no temporal structure to corrupt, just a higher mean rate.
    ],
  )
]
