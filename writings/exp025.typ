#let meta = (
  title: "PING locks E rate ≈10× below COBA",
  date: "2026-05-30",
  description: "A head-to-head of COBA and PING on MNIST under a matched recipe: PING locks the hidden E rate about an order of magnitude lower at comparable accuracy, so the loop buys per-spike economy.",
  collection: "gamma-gated-sparsity",
  status: "final",
)

// Provenance (HOUSESTYLE H9/H19): every run number below is interpolated from the
// run's numbers.json, never hand-typed, so a re-run updates the prose automatically.
#let run = json("/artifacts/data/exp025/numbers.json")
#let mean(a) = a.sum() / a.len()
#let rate-target(r) = if r.keys().contains("rate_target_hz") {
  r.rate_target_hz
} else if r.theta_u == none {
  none
} else {
  r.theta_u * 5
}
#let pfg-rows = if run.keys().contains("rate_target_p_fgamma") {
  run.rate_target_p_fgamma
} else {
  run.theta_p_fgamma
}

// Frontier points are averaged over the three independent TR-02 seeds.
#let res = run.results
#let coba_off_rate = calc.round(mean(res.filter(r => r.model == "coba" and rate-target(r) == none).map(r => r.rate_e)))
#let ping_off_rate = calc.round(mean(res.filter(r => r.model == "ping" and rate-target(r) == none).map(r => r.rate_e)))
#let coba_off_acc = calc.round(mean(res.filter(r => r.model == "coba" and rate-target(r) == none).map(r => r.final_acc)))
#let ping_off_acc = calc.round(mean(res.filter(r => r.model == "ping" and rate-target(r) == none).map(r => r.final_acc)))

// Rate-floor decomposition (Figure 2): the five rate target sweep cells per model.
#let ping_pfg = pfg-rows.filter(r => r.model == "ping" and rate-target(r) != none)
#let coba_pfg = (
  pfg-rows.filter(r => r.model == "coba" and rate-target(r) != none).sorted(key: rate-target)
)
#let p_lo = calc.round(calc.min(..ping_pfg.map(r => r.p)), digits: 2)
#let p_hi = calc.round(calc.max(..ping_pfg.map(r => r.p)), digits: 2)
#let fg_hi = calc.round(calc.max(..ping_pfg.map(r => r.f_gamma)))
#let fg_lo = calc.round(calc.min(..ping_pfg.map(r => r.f_gamma)))
#let ping_acc_lo = calc.round(calc.min(..ping_pfg.map(r => r.acc)))
#let ping_acc_hi = calc.round(calc.max(..ping_pfg.map(r => r.acc)))
#let coba_acc_loose = calc.round(coba_pfg.last().acc)
#let coba_acc_tight = calc.round(coba_pfg.first().acc)
#let pfg_err_max = calc.round(calc.max(..ping_pfg.map(r => calc.abs((r.p * r.f_gamma - r.e_rate) / r.e_rate) * 100)))
#let ping_off_pfg = pfg-rows.filter(r => r.model == "ping" and rate-target(r) == none).first()
#let ping_cadence = calc.round(1000 / ping_off_pfg.f_gamma)

// Low-W_in recruitment sweep (Figure 3), columns ordered 0.05 / 0.1 / 0.3 / 0.9.
#let low_accs = run.low_w_in_sweep.map(r => calc.round(r.final_acc, digits: 1))
#let low_is = run.low_w_in_sweep.map(r => calc.round(r.rate_i, digits: 1))

// Inference-time W_in scale sweep (Figures 4-5), trained point at s = 1.
#let ping_ws = run.w_in_scale_sweep.filter(r => r.cell == "ping@rt1hz" or r.cell == "ping@tu0.2")
#let coba_ws = run.w_in_scale_sweep.filter(r => r.cell == "coba@rt1hz" or r.cell == "coba@tu0.2")
#let coba_pen_s3 = calc.round(coba_ws.filter(r => r.scale == 3.0).first().penalty)
#let coba_acc_s3 = calc.round(coba_ws.filter(r => r.scale == 3.0).first().acc)
#let coba_e_s3 = calc.round(coba_ws.filter(r => r.scale == 3.0).first().rate_e)
#let ping_star_e = calc.round(ping_ws.filter(r => r.scale == 1.0).first().rate_e, digits: 1)
#let coba_star_e = calc.round(coba_ws.filter(r => r.scale == 1.0).first().rate_e, digits: 1)
#let ping_plateau = calc.round(calc.max(..ping_ws.map(r => r.acc)))

#let body = [
  The trained networks this entry uses are produced once in the shared training
  hub, #link("/exp022/")[exp022 (Training)], and reused here rather than retrained.

  == Abstract

  Head-to-head comparison of COBA (recurrent inhibitory loop disabled) against PING
  (loop active) on MNIST under matched architecture and training recipe. PING locks
  the hidden E rate to ≈ #ping_off_rate Hz while COBA runs at ≈ #coba_off_rate Hz;
  accuracy is within a few points across the two, so the loop buys better than an
  order-of-magnitude per-spike economy. The rate-vs-accuracy frontier traced by sweeping the hidden-E rate ceiling
  regulariser shows the floor is structural, not a trade-off the optimiser can
  navigate away.

  == Method

  *Training recipe (canonical / medium tier):*

  #table(
    columns: 2,
    [Parameter], [Value],
    [Integration timestep $Delta t$], [0.1 ms],
    [Trial duration $T$], [200 ms],
    [MNIST training pool (7,000 official-training samples)], [6,300 optimizer-training / 700 validation; downstream inference uses a fixed 1,000-image subset of the official test partition],
    [Epochs], [50],
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
  $W_(e e)$ (fixed at zero, since Börgers-style PING needs no E→E coupling), $W_(e i)$,
  and $W_(i e)$ are initialised once and held _requires_grad = False_. The synaptic
  time constants $tau_"AMPA"$, $tau_"GABA"$ are module-level constants. 813k
  trainable parameters out of 2.4M total.

  *Training.* Adam, lr = $4 times 10^(-4)$, batch size 256, gradient norm clipped to
  1.0. Cross-entropy loss on 10-class MNIST (definition in #link("/ar006/")[ar006]).
  The voltage-gradient damping is 1 for COBA and 1000 for PING, where recurrent
  backpropagation requires stabilisation. Every frontier condition is trained with
  three independent seeds (42, 43, 44); plotted points are across-seed means with
  standard errors.

  *Recipe difference.* COBA and PING share $W_"in" tilde cal(N)(0.9, 0.09)$ at 95%
  sparsity and the same directly specified readout initializer. COBA disables the
  recurrent loop; PING enables it and uses the loop-specific gradient damping.

  *Activity regulariser.* To probe the rate axis, the training loss adds a soft
  upper bound on each presentation's population-mean hidden-E firing rate:

  $ r_b = 1 / (N_E T) sum_(n in E) z_(b n), quad
    L_"rate" = lambda / B sum_b "ReLU"(r_b - r_"max")^2, $

  with $r_b$ the population-mean hidden-E firing rate for presentation $b$ and
  $r_"max"$ its ceiling in hertz. Presentations at or below the ceiling contribute
  zero. We sweep off, 25, 10, 5, 2.5, and 1 Hz, giving twelve
  $("model", r_"max")$ conditions and 36 trained cells across seeds.

  *Rate-floor decomposition.* The affine law $r_E = p dot f_gamma$
  (#link("/exp041/")[exp041], #link("/exp046/")[exp046]) factors the E rate into
  per-cycle participation $p$ and gamma frequency $f_gamma$. Both are measured at
  one explicitly representative final-epoch checkpoint (seed 42) per condition: $f_gamma$ from
  the Welch PSD peak of the E-population trace, $p$ via
  I-burst peak detection and per-(cell, cycle) spike counting (style of
  #link("/exp046/")[exp046]).

  *Basin and landscape probes.* To test basin attractivity, four input-weight
  conditions are each trained with seeds 42, 43, and 44 (12 PING networks total),
  trained with $W_"in"$ initialised at 0.05, 0.1, 0.3, and 0.9 ($r_"max" = 1$ Hz from
  epoch 0; Figure 3). To map the loss landscape around the operating point, each
  network trained at the tightest ceiling ($r_"max" = 1$ Hz) has its $W_"in"$ scaled
  by a common scalar $s in [0.05, 3]$ at inference with all other weights frozen,
  metrics averaged over the same fixed 1,000-image held-out subset at 24 values
  of $s$ (Figures 4–5).

  == Results

  #figure(
    image(
      "/artifacts/data/exp025/results_compound.png",
      width: 100%,
      alt: "Two-by-two panel: COBA and PING single-trial rasters, per-epoch learning curves, and the accuracy–rate frontier across hidden-E rate ceilings.",
    ),
    caption: [
      The headline comparison in one 2×2 frame (rasters, learning, and the
      accuracy–rate frontier together). *Top:* trained-baseline single-trial
      rasters on the same MNIST digit 0 input. COBA (_--ei-strength 0_) fires densely
      and asynchronously at ≈ #coba_off_rate Hz (I silent, loop off); PING
      (_--ei-strength 1_) fires in gamma bands at ≈ #ping_cadence ms cadence
      (≈ #ping_off_rate Hz per E cell) with synchronous I bursts (red) above E
      (black), on the same architecture, parameter count, and recipe, with only
      PING's recurrent E↔I matrices non-zero. *Bottom left:* test accuracy per epoch
      (mean over three seeds); both reach ≈ #ping_off_acc%, so both learn the task.
      *Bottom right:* the accuracy–rate frontier across hidden-E rate ceilings
      (mean ± SEM across three seeds at every point). PING sits
      up-and-left of COBA, the same accuracy at a fraction of
      the hidden-E rate, with the unpenalised operating points starred and labelled
      (PING #ping_off_acc% at ≈ #ping_off_rate Hz, COBA #coba_off_acc% at
      ≈ #coba_off_rate Hz). COBA red, PING black; E black and I red in the rasters.
      _The headline of this entry: gamma gating buys an order-of-magnitude per-spike
      economy at matched accuracy._
    ],
  )

  #figure(
    image(
      "/artifacts/data/exp025/theta_p_fgamma.svg",
      width: 100%,
      alt: "PING participation fraction p and gamma frequency f_gamma across the activity-ceiling sweep, with the p·f_gamma product overlaid on the measured E rate.",
    ),
    caption: [
      Five representative seed-42 PING rate-target checkpoints, 256 test
      trials each. *$p$ stays in
      #p_lo–#p_hi* across the entire sweep (the architecture protects the
      participation gate), while *$f_gamma$ slides from ≈ #fg_hi Hz to ≈ #fg_lo Hz*
      as the penalty tightens. The grey dashed $p dot f_gamma$ curve overlays the
      measured E rate within #pfg_err_max%; the rate change is entirely in $f_gamma$.
      PING accuracy holds at #ping_acc_lo–#ping_acc_hi% the whole way; COBA's
      collapses from #coba_acc_loose% to #coba_acc_tight%.
    ],
  )

  #figure(
    image(
      "/artifacts/data/exp025/low_w_in_sweep.svg",
      width: 100%,
      alt: "Across-seed mean per-epoch test accuracy and E/I firing rates for four PING input-weight initializations, one column per W_in value.",
    ),
    caption: [
      Across-seed mean per-epoch training traces for four initial $W_"in"$ values
      (seeds 42–44, $r_"max" = 1$ Hz from epoch 0), one condition per column;
      shaded bands are SEM. Top: test accuracy. Bottom: test-set E (black) and I
      (red) firing rates. The comparison tests whether lower initial input drive
      delays or prevents recruitment of the inhibitory loop. Final mean accuracies:
      #low_accs.at(0)% / #low_accs.at(1)% / #low_accs.at(2)% / #low_accs.at(3)%;
      final I rates: #low_is.at(0) / #low_is.at(1) / #low_is.at(2) / #low_is.at(3) Hz.
    ],
  )

  #figure(
    image(
      "/artifacts/data/exp025/w_in_scale_sweep.svg",
      width: 100%,
      alt: "Inference-time W_in scale sweep: CE loss, activity penalty, total objective, test accuracy, and E/I rates versus scalar s for PING and COBA.",
    ),
    caption: [
      Inference-time $W_"in"$ scale sweep on the two networks trained under the
      tightest ceiling ($r_"max" = 1$ Hz); every $W_"in"$ weight multiplied by a
      common scalar $s$, all other weights frozen, 24 values of $s in [0.05, 3]$. Top
      row: CE loss, activity penalty $L_"rate"$, total objective CE + $L_"rate"$.
      Bottom row: test accuracy (chance dotted), E rate, I rate. PING black, COBA red.
      Vertical dashed line at $s = 1$ marks the trained operating point; dotted line
      marks ≈ $f^*$, PING's recruitment cliff. Loss panels clipped at 4; COBA's
      penalty reaches ≈ #coba_pen_s3 at $s = 3$ ($"rate"^2$ scaling).
    ],
  )

  #figure(
    image(
      "/artifacts/data/exp025/w_in_scale_sweep_vs_rate.svg",
      width: 100%,
      alt: "The W_in scale sweep re-projected with hidden E rate on the x-axis, trained operating points starred for PING and COBA.",
    ),
    caption: [
      The same 24-point sweep as Figure 4, re-projected with hidden E rate on the
      x-axis. Filled stars mark each cell's trained operating point ($s = 1$): PING
      at $E$ ≈ #ping_star_e Hz, COBA at $E$ ≈ #coba_star_e Hz. PING's accuracy
      reaches its plateau (≈ #ping_plateau%) just past its trained operating point;
      COBA climbs slowly and only reaches ≈ #coba_acc_s3% even at $E$ ≈ #coba_e_s3 Hz.
    ],
  )
]
