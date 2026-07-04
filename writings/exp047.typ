#import "/demolab-engine/build/lib.typ": numbers-table, provenance-footer

#let meta = (
  title: "The PING rate is set by the per-synapse weight, not the pool size",
  date: "2026-06-08",
  description: "Sweeping per-synapse inhibitory weight and pool size independently, the rate-vs-weight curves for every pool size collapse onto one.",
  collection: "gamma-gated-sparsity",
  status: "revising",
)

#let run = json("/artifacts/data/exp047/numbers.json")

#let body = [
  == Abstract

  The PING rate is a function of the per-synapse inhibitory weight $W^(I E)$ — measured against a fixed release level $g_i^*$ — and *not* of the pool size $N_I$. Sweeping the two independently, the rate-vs-$W^(I E)$ curves for every $N_I$ collapse onto one.

  == Methods

  *1. The mechanism, in words.* In a PING cycle the E volley recruits I, the I spikes shunt each E cell toward the inhibitory reversal $E_i$, and E stays quiet until that shunt decays enough for the drive to climb $V$ back to threshold. The cycle's clock — and hence the rate — is that recovery: how far one inhibitory event drives the conductance past the level that silences E, and how long it takes to decay back. This is a _per-event_ quantity. Adding more I cells delivers the same shunt through more synapses; it does not change how deep a single event pins a cell or how long the pin lasts. So the rate should track the per-synapse strength — not $N_I$. The rest of this section makes that precise.

  *2. The invariant release level.* Reduce to one representative E cell — the COBANet membrane of #link("/ar003/")[ar003], with the feedforward excitation written as a tonic current $I_"ext"$ as in #link("/exp033/")[exp033]. This single-cell picture is a heuristic to _motivate_ the prediction below, not a proof of it; the full network (§5, Figure 1) is what tests it. When the inhibitory conductance varies slowly the voltage tracks its instantaneous steady state $V_oo$ (ar003, eqs 11–13; valid in the high-conductance state of a volley, where the open conductances pull the effective membrane time constant below $tau_"GABA"$). Setting $V_oo = V_"th"$ gives the conductance at which the cell is _released_ to fire again,

  $ g_i^* = (I_"ext" - I_"th") / (V_"th" - E_i), quad I_"th" eq.triple g_L (V_"th" - E_L). quad (1) $

  $g_i^*$ is built from intrinsic constants and the drive alone — *it contains neither $N_I$ nor the per-synapse weight $W^(I E)$.* It is the level the cycle is pinned to.

  *3. The cycle clock.* Each inhibitory spike adds $W^(I E)$ to $g_i$, which then decays with $tau_"GABA"$ (ar003's exponential synapse, eq 8). After a volley peaks at $g_i^"pk"$ — the sum of the fresh kicks — the conductance decays back through $g_i^*$ after

  $ T_"sup" = tau_"GABA" ln (g_i^"pk") / (g_i^*), quad (2) $

  so the gamma period is $T approx T_"build" + T_"sup"$ and $r_E prop 1 \/ T$. A self-organized loop recruits I only until $g_i$ crosses $g_i^*$ and E shuts off, so the volley overshoots by about one quantum, $g_i^"pk" approx g_i^* + W^(I E)$, and

  $ T_"sup" approx tau_"GABA" ln (1 + (W^(I E)) / (g_i^*)). quad (3) $

  *4. The prediction: collapse onto $W^(I E)$.* Equation (3) has no $N_I$ in it. The suppression — hence the rate — is a function of the per-event jump $W^(I E)$ relative to the invariant $g_i^*$ alone, increasing (rate falling) with $W^(I E)$ and saturating logarithmically. The test is therefore to vary $W^(I E)$ and $N_I$ _independently_ and plot the rate against $W^(I E)$: if (3) holds, the curves for different $N_I$ must lie on top of one another.

  *5. Simulation.* Untrained PING at $N_E = 1024$, $Delta t = 0.1$ ms, $T = 200$ ms. Per-synapse $W^(I E)$ swept over {0.5, 1, 2, 4, 8, 16} μS (spread held at 10% of the mean) and pool size $N_I$ over {16, 64, 256} — the two varied on an independent grid. Fixed $W^(E I) tilde cal(N)(1.0, 0.1)$ μS and uniform 25 Hz Poisson input; inference only. The biophysical default is $W^(I E) = 2$ μS.

  == Results

  The curves collapse (Figure 1). At each per-synapse weight the three pool sizes agree to within a few tenths of a Hz — at the default $W^(I E) = 2$ μS, $r_E = 11.4$, $11.8$, $11.7$ Hz for $N_I = 16, 64, 256$ — and the rate falls smoothly with $W^(I E)$, from $approx 14.1$ Hz at $0.5$ μS to $approx 9.6$ Hz at $16$ μS — the decelerating, saturating dependence eq (3) predicts (the precise functional form is not pinned by six points, and turning the cycle period into a per-cell rate would need a participation factor (3) does not supply). The I rates (right panel) collapse the same way. Pool size does not move the rate; the per-synapse weight does.

  The lesson for sweeping or training $N_I$: the rate is governed by the per-synapse $W^(I E)$ measured against $g_i^*$. Hold $W^(I E)$ fixed and the rate is invariant to pool size; change $N_I$ alone, at fixed per-synapse weight, and nothing moves.

  #figure(
    image("/artifacts/data/exp047/rate_vs_w_ie.svg", width: 100%),
    caption: [E (left) and I (right) per-cell rate versus the per-synapse weight $W^(I E)$, one line per pool size $N_I$. The lines collapse: at any fixed $W^(I E)$ the rate is the same for $N_I = 16$, $64$, or $256$, while it falls steadily with $W^(I E)$ — the logarithmic, saturating dependence of equation (3). The rate is set by the per-synapse kick, not the pool size. Dotted line marks the biophysical default $W^(I E) = 2$ μS.],
  )
]
