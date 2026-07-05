#let meta = (
  title: "PING locks E rate ≈10× below COBA",
  date: "2026-05-30",
  description: "A head-to-head of COBA and PING on MNIST under a matched recipe: PING locks the hidden E rate ≈10× lower at comparable accuracy — the loop buys per-spike economy.",
  collection: "gamma-gated-sparsity",
  status: "revising",
)

#let run = json("/artifacts/data/exp025/numbers.json")

#let body = [
  The trained networks this entry uses are produced once in the shared training
  hub, #link("/exp022/")[exp022 (Training)], and reused here rather than retrained.

  == Abstract

  Head-to-head comparison of COBA (recurrent inhibitory loop disabled) against PING
  (loop active) on MNIST under matched architecture and training recipe. PING locks
  the hidden E rate to ≈ 10 Hz while COBA runs at ≈ 96 Hz; accuracy is within a few
  points across the two, so the loop buys roughly an order-of-magnitude per-spike
  economy. The rate-vs-accuracy frontier traced by sweeping the $theta_u$ rate
  regulariser shows the floor is structural, not a trade-off the optimiser can
  navigate away.

  == Method

  *Training recipe (canonical / medium tier):*

  #table(
    columns: 2,
    [Parameter], [Value],
    [Integration timestep $Delta t$], [0.1 ms],
    [Trial duration $T$], [200 ms],
    [MNIST samples (80/20 stratified split of 2000)],
    [1600 train / 400 test (≈ 2.9% of the 70k-sample MNIST corpus)],
    [Epochs], [100],
  )

  Two configurations of the same COBANet architecture, differing only in whether the
  E→I→E inhibitory loop is active. *COBA* (_--ei-strength 0_) disables the loop;
  excitatory cells drive each other but receive no structured inhibition. *PING*
  (_--ei-strength 1_) enables the loop, producing pyramidal-interneuron gamma (PING)
  oscillations at a cadence set by $tau_"AMPA"$ and $tau_"GABA"$.

  *Architecture.* $N_E = 1024$ excitatory cells, $N_I = 256$ inhibitory cells,
  single hidden layer. Input: 784 channels (MNIST pixels), Poisson-encoded at 25 Hz
  peak rate. Readout: mem-mean (time-averaged E spike vector projected through a
  trained linear layer $W_"out"$; see #link("/ar006/")[ar006] for the full readout
  specification). Dale's law enforced.

  *What is trainable.* Only the input weights $W_"in"$ ($784 times 1024$, 95%
  sparse) and the readout $W_"out"$ ($1024 times 10$). The recurrent weights
  $W_(e e)$ (fixed at zero — Börgers-style PING needs no E→E coupling), $W_(e i)$,
  and $W_(i e)$ are initialised once and held _requires_grad = False_. The synaptic
  time constants $tau_"AMPA"$, $tau_"GABA"$ are module-level constants. 813k
  trainable parameters out of 2.4M total.

  *Training.* Adam, lr = $4 times 10^(-4)$, batch size 256, gradient norm clipped to
  1.0. Cross-entropy loss on 10-class MNIST (definition in #link("/ar006/")[ar006]).
  Gradient stabiliser: _--v-grad-dampen 1000_ (uniform scaling of per-step voltage
  gradients). Three seeds (42, 43, 44) for baselines, one seed (42) for sweep cells.

  *Recipe difference.* The only parameter that differs between COBA and PING besides
  _--ei-strength_ is the $W_"in"$ initialisation: COBA uses mean 0.3 (std 0.03), PING
  uses mean 1.2 (std 0.12). PING needs stronger input drive to reliably recruit the
  I-loop at init.

  *Spike-budget regulariser.* To probe the rate axis, the training loss adds a soft
  upper bound on per-trial spike count. For each E cell $n$ with mean per-trial spike
  count $macron(z)_n$:

  $ L_"rate" = lambda sum_n "ReLU"(macron(z)_n - theta_u)^2, $

  with $theta_u$ the per-cell budget (spikes/trial) and $lambda = 10^(-3)$; only
  cells over budget contribute, cells under it are free, and the total loss is
  cross-entropy plus $L_"rate"$. We sweep six budgets — off, 5, 2, 1, 0.5, 0.2
  spikes/trial (at $T = 200$ ms: no penalty, 25, 10, 5, 2.5, 1 Hz) — giving twelve
  $("model", theta_u)$ cells.

  *Rate-floor decomposition.* The affine law $r_E = p dot f_gamma$
  (#link("/exp041/")[exp041], #link("/exp046/")[exp046]) factors the E rate into
  per-cycle participation $p$ and gamma frequency $f_gamma$. Both are measured at
  every cell — $f_gamma$ from the Welch PSD peak of the E-population trace, $p$ via
  I-burst peak detection and per-(cell, cycle) spike counting (style of
  #link("/exp046/")[exp046]).

  *Basin and landscape probes.* To test basin attractivity, four PING networks are
  trained with $W_"in"$ initialised at 0.05, 0.1, 0.3, and 1.2 ($theta_u = 0.2$ from
  epoch 0; Figure 3). To map the loss landscape around the operating point, each
  network trained at the heaviest penalty ($theta_u = 0.2$) has its $W_"in"$ scaled
  by a common scalar $s in [0.05, 3]$ at inference with all other weights frozen,
  metrics averaged over the test set at 24 values of $s$ (Figures 4–5).

  == Results

  #figure(
    image("/artifacts/data/exp025/results_compound.png", width: 100%),
    caption: [
      The headline comparison in one 2×2 frame (rasters, learning, and the
      accuracy–rate frontier together). *Top* — trained-baseline single-trial
      rasters on the same MNIST digit 0 input: COBA (_--ei-strength 0_) fires densely
      and asynchronously at ≈ 97 Hz (I silent, loop off); PING (_--ei-strength 1_)
      fires in gamma bands at ≈ 28 ms cadence (≈ 10 Hz per E cell) with synchronous I
      bursts (red) above E (black). Same architecture, parameter count, and recipe —
      only PING's recurrent E↔I matrices are non-zero. *Bottom left* — test accuracy
      per epoch (mean over three seeds): both reach ≈ 88%, so both learn the task.
      *Bottom right* — the accuracy–rate frontier across the spike-budget penalty
      $theta_u$: PING sits up-and-left of COBA — the same accuracy at a fraction of
      the hidden-E rate — with the $theta_u$-off operating points starred and
      labelled (PING 88% at ≈ 10 Hz, COBA 90% at ≈ 97 Hz). COBA red, PING black; E
      black / I red in the rasters. _The headline of this entry: gamma gating buys an
      order-of-magnitude per-spike economy at matched accuracy._
    ],
  )

  #figure(
    image("/artifacts/data/exp025/theta_p_fgamma.svg", width: 100%),
    caption: [
      Six $("PING", theta_u)$ cells, 256 test trials each. *$p$ stays in 0.19–0.24*
      across the entire sweep — the architecture protects the participation gate —
      while *$f_gamma$ slides from ≈ 37 Hz to ≈ 15 Hz* as the penalty tightens. The
      amber dashed $p dot f_gamma$ curve overlays the measured E rate within 4%; the
      rate change is entirely in $f_gamma$. PING accuracy holds at 83–87% the whole
      way; COBA's collapses from 83% to 61%.
    ],
  )

  #figure(
    image("/artifacts/data/exp025/low_w_in_sweep.svg", width: 100%),
    caption: [
      Per-epoch training traces from four PING networks (seed 42, $theta_u = 0.2$
      from epoch 0), one per column. Top: test accuracy. Bottom: test-set E (black)
      and I (red) firing rates. At $W_"in" = 0.05$ and $0.1$ the I population is
      silent for the first one to two epochs; once $W_"in"$ crosses the recruitment
      threshold the loop engages and the network locks into PING. Final accuracies:
      84.0% / 84.5% / 85.0% / 83.75%; final I rates: 11.1 / 15.5 / 6.9 / 16.7 Hz.
    ],
  )

  #figure(
    image("/artifacts/data/exp025/w_in_scale_sweep.svg", width: 100%),
    caption: [
      Inference-time $W_"in"$ scale sweep on the two networks trained under the
      heaviest penalty ($theta_u = 0.2$); every $W_"in"$ weight multiplied by a
      common scalar $s$, all other weights frozen, 24 values of $s in [0.05, 3]$. Top
      row: CE loss, spike-budget penalty $L_"rate"$, total objective CE + $L_"rate"$.
      Bottom row: test accuracy (chance dotted), E rate, I rate. PING black, COBA red.
      Vertical dashed line at $s = 1$ marks the trained operating point; dotted line
      marks ≈ $f^*$, PING's recruitment cliff. Loss panels clipped at 4 — COBA's
      penalty reaches ≈ 32 at $s = 3$ ($"rate"^2$ scaling).
    ],
  )

  #figure(
    image("/artifacts/data/exp025/w_in_scale_sweep_vs_rate.svg", width: 100%),
    caption: [
      The same 24-point sweep as Figure 4, re-projected with hidden E rate on the
      x-axis. Filled stars mark each cell's trained operating point ($s = 1$): PING
      at $E ≈ 3.8$ Hz, COBA at $E ≈ 1.0$ Hz. PING's accuracy reaches its plateau by
      $E ≈ 3$ Hz; COBA climbs slowly and only reaches ≈ 70% even at $E ≈ 28$ Hz.
    ],
  )
]
