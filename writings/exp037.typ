#let meta = (
  title: "Perturbations: PING tolerates 80% drops but only 15% adds",
  date: "2026-05-30",
  description: "Perturbs the hidden spike stream of trained PING and COBA at inference; PING forgives ≈80% dropped spikes but breaks on ≈15% added — the gamma cycle made visible.",
  collection: "gamma-gated-sparsity",
  status: "revising",
)

#let run = json("/artifacts/data/exp037/numbers.json")

#let body = [
  The trained networks this entry uses are produced once in the shared training
  hub, #link("/exp022/")[exp022 (Training)], and reused here rather than retrained.

  == Abstract

  Perturbs the hidden spike stream of trained PING and COBA networks at inference
  (drop spikes, add spikes) to ask whether the PING rate floor is dynamical or
  informational. PING tolerates ≈ 80% drop but only ≈ 15% add before accuracy
  collapses; COBA is roughly flat to both perturbations. The asymmetry — drops
  forgiven, adds break the gating — is the gamma cycle made visible: a structural
  feature of the architecture, not a readout-side trade-off.

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

  The PING and COBA baseline definitions and training recipe are in
  #link("/exp025/")[exp025], where the rate-floor mechanism is also worked out. This
  entry tests whether the floor is *dynamical* (locked by the cycle period) or
  *informational* (locked by the readout's spike-count requirement) by perturbing
  the hidden spike stream of the trained networks at inference.

  *The perturbation.* A per-step callback on the COBANet's _\_hidden_perturb_fn_
  slot fires every timestep of every trial — no warm-up, no schedule, no exclusion.
  Trials are $T = 200$ ms at $Delta t = 0.1$ ms → 2000 fires per trial; the test
  set (≈ 400 trials) gives ≈ 800 000 fires per (model, mode, level) sweep point
  against the same trained network. At each timestep $t$:

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

  Sweep $r_"add" in {0, 2, ..., 40}$ Hz, 21 levels. On the right panel of Figure 1
  the add level is expressed *per population* — $r_"add"^E = "pct" dot macron(r_E)$,
  $r_"add"^I = "pct" dot macron(r_I)$, each architecture scaled by its own
  baseline — because the baselines differ by an order of magnitude
  (COBA $macron(r_E) = 65$ Hz vs PING $macron(r_E) = 6$ Hz), so a fixed
  "10 Hz of noise" would be 15% of COBA's rate but 170% of PING's. The per-step RNG
  is seeded separately from the input encoder, so the Poisson input stream matches
  the unperturbed baseline. Total: 2 models × (11 drop + 21 add) = 64 forward passes.

  == Results

  #figure(
    image("/artifacts/data/exp037/perturbation_curves.svg", width: 100%),
    caption: [
      *Left (drop)*: Bernoulli mask, % of emitted spikes dropped. *Right (add)*:
      Poisson injection per population (each scaled by its own baseline, so the
      comparison is architecture-fair). The asymmetry is direct: PING tolerates drop
      up to ≈ 80% but collapses on add between 15% and 50% relative noise, while
      COBA stays at 75% even with 100% extra Poisson.
    ],
  )

  #figure(
    image("/artifacts/data/exp037/perturb_rasters__drop__ping.png", width: 100%),
    caption: [
      Trained PING replayed on the same MNIST digit 0 trial across the six drop
      levels of Figure 1's left panel (0, 30, 60, 80, 90, 100%); E (black) above I
      (red). The gamma cadence persists across all sub-total levels — dropped spikes
      thin the train without injecting phase-incoherent activity.
    ],
  )

  #figure(
    image("/artifacts/data/exp037/perturb_rasters__add__ping.png", width: 100%),
    caption: [
      Trained PING replayed across the six pct-of-baseline add levels of Figure 1's
      right panel (0, 5, 15, 25, 40, 80%). The cycle visibly dissolves across
      25–40%, matching Figure 1's accuracy cliff.
    ],
  )

  #figure(
    image("/artifacts/data/exp037/perturb_rasters__drop__coba.png", width: 100%),
    caption: [
      Trained COBA replayed across the same drop sweep as Figure 2. With no cycle to
      preserve, dropping spikes just thins a uniform asynchronous mean.
    ],
  )

  #figure(
    image("/artifacts/data/exp037/perturb_rasters__add__coba.png", width: 100%),
    caption: [
      Trained COBA replayed across the same pct-of-baseline add sweep as Figure 3.
      Added spikes blend into COBA's asynchronous mean — no temporal structure to
      corrupt, just a higher mean rate.
    ],
  )
]
