#let meta = (
  title: "Gradient Stabilisation",
  date: "2026-06-12",
  description: "Why conductance-based spiking networks need --v-grad-dampen to train, derived from the discrete equations: the recurrent loop makes the backpropagated gradient diverge geometrically, and per-step voltage-gradient damping bounds it.",
  collection: "documentation",
)

#let body = [
  Conductance-based spiking networks (COBA, PING) need one extra ingredient to train at all: a single flag, `--v-grad-dampen`. This article is the deep dive on why it is needed and what it does, derived from the discrete equations the code runs. For the surrounding recipe — BPTT, the spike surrogate, the loss, optimiser, readout, and regulariser — see the #link("/ar006/")[Training] article; this one assumes that background.

  *In plain English.* The E and I cells form a loop that makes the network oscillate at gamma (≈25 Hz). Training sends the gradient backward around that same loop, and each time a neuron crosses its spike threshold the surrogate hands the gradient a large multiplier. The loop is traversed _once per gamma cycle_ — only about five times in a 200 ms trial — but each pass multiplies by a factor well above one (the surrogate slope enters squared), so a handful of passes is enough to overflow to NaN. The forward simulation stays bounded because the spike reset clamps each cell every cycle; the gradient sidesteps that reset, so forward stability buys nothing backward. The fix, `--v-grad-dampen`, divides the voltage gradient by a constant $gamma$ at every step, shrinking the loop's multiplier below one — and thanks to a straight-through trick it changes only the gradient, never the simulation. The rest of this article proves each of those claims from the discrete equations.

  Naive BPTT through a 2000-step COBA/PING trial reaches NaN within a few batches. This article establishes the failure and its remedy from the discrete equations the code runs — no step omitted. The plan is: write the one-step map, differentiate it exactly (Lemma 1), propagate the gradient backward (the recursion), show the recurrent loop forces that gradient to diverge geometrically (Proposition 1), and show that scaling the per-step voltage gradient by $1\/gamma$ makes the loop a contraction while leaving the forward trajectory untouched (Proposition 2).

  == Setup: the discrete update

  Index the timesteps $t = 0, dots, T$ at $Delta t = 0.1$ ms. Each cell holds a state $(V^t, g_e^t, g_i^t)$. With reversal potentials $E_L, E_e, E_i$, capacitance $C_m$, leak $g_L$, synaptic decays $beta_e = e^(-Delta t\/tau_e)$ and $beta_i = e^(-Delta t\/tau_i)$, threshold $theta.alt$ and reset $V_r$, the step is exactly (`lif_step_expeuler`, `exp_synapse`):

  $ s^t = Theta(V^t - theta.alt) chi^t, quad ("S") $

  $ g_e^(t+1) = beta_e (g_e^t + W_e s_"pre"^t), quad g_i^(t+1) = beta_i (g_i^t + W_i s_"pre"^t), quad ("C") $

  $ tilde(V)^(t+1) = V_oo^(t+1) + (V^t - V_oo^(t+1)) alpha^(t+1), quad
  V^(t+1) = cases(V_r & "if " s^t=1 " or refractory", tilde(V)^(t+1) & "otherwise,") quad ("V") $

  where $chi^t in {0,1}$ is the not-refractory gate, and the membrane decay and rest point are built from the _post_-update conductances $g^(t+1)$:

  $ g_"tot" = g_L + g_e^(t+1) + g_i^(t+1), quad
  alpha^(t+1) = e^(-Delta t g_"tot"\/C_m), quad
  V_oo^(t+1) = (g_L E_L + g_e^(t+1) E_e + g_i^(t+1) E_i)/(g_"tot"). quad ("P") $

  The timing matters and is taken from the code: the spike $s^t$ in (S) is read from the _current_ voltage $V^t$, drives the conductance one step later in (C), and that conductance sets the voltage in (V). So one synaptic edge — presynaptic voltage to postsynaptic voltage — spans one timestep.

  == Lemma 1 — the one-step Jacobian

  Differentiate (S), (C), (V) entry by entry. The spike #link("/ar006/")[surrogate] gives the spike derivative

  $ (partial s^t)/(partial V^t) = sigma'(V^t - theta.alt) chi^t, quad sigma'(u) = k/((1+k |u|)^2), quad sigma'(0) = k. quad (1) $

  From (C), the conductance partials are a self-decay and a _presynaptic-voltage_ coupling obtained by chaining (1):

  $ (partial g_e^(t+1))/(partial g_e^t) = beta_e, quad
  (partial g_e^(t+1))/(partial V_"pre"^t) = beta_e W_e sigma'(V_"pre"^t - theta.alt) chi_"pre"^t. quad (2) $

  From (V), with $g$ held, the membrane self-term is the decay; and differentiating (P) gives the conductance-to-voltage term ($partial g_"tot" \/ partial g_e = 1$, $partial V_oo \/ partial g_e = (E_e - V_oo)\/g_"tot"$, $partial alpha \/ partial g_e = -(Delta t \/ C_m) alpha$):

  $ (partial tilde(V)^(t+1))/(partial V^t) = alpha^(t+1), quad
  (partial tilde(V)^(t+1))/(partial g_e^(t+1))
  = underbrace((1-alpha) (E_e - V_oo)/(g_"tot"), "rest shifts") - underbrace((Delta t)/(C_m) alpha (V^t - V_oo), "decay shifts")
  eq.triple kappa_e approx (Delta t)/(C_m)(E_e - V), quad (3) $

  the last step being the leading order in $Delta t$ (using $1-alpha approx Delta t g_"tot"\/C_m$). Define $kappa_i$ identically with $E_i$. Collecting (1)–(3), the one-step Jacobian on $(V, g_e, g_i)$, with the cross-cell coupling in the lower-left block, is

  $ J^t = (partial(V,g_e,g_i)^(t+1))/(partial(V,g_e,g_i)^t) =
  mat(
    rho alpha, kappa_e, kappa_i;
    beta_e W_e sigma'(V_"pre"-theta.alt) chi_"pre", beta_e, 0;
    beta_i W_i sigma'(V_"pre"-theta.alt) chi_"pre", 0, beta_i
  ). quad ("J") $

  The single subtlety, and it is decisive below: the reset in (V) is a `torch.where`, so on any cell that spikes or is refractory the output is the constant $V_r$ and the membrane self-term is gated to zero. We write this as the factor $rho in {0,1}$ on the $partial V^(t+1)\/partial V^t$ entry: $rho = 0$ at a spike (gradient through the cell's own membrane is cut), $rho = 1$ otherwise. Crucially $rho$ multiplies only the membrane self-term — it does *not* touch the spike output $s^t$ in (1)–(2), which is evaluated at $V^t$ _before_ the reset.

  == Backpropagation: the gradient recursion

  Let $lambda^t eq.triple partial cal(L)\/partial(V,g_e,g_i)^t$. Reverse-mode autodiff is exactly the linear recursion

  $ lambda^t = (J^t)^top lambda^(t+1), quad lambda^T = nabla_(x^T) cal(L)
  quad ==> quad
  lambda^t = (product_(s=t)^(T-1) J^s)^top lambda^T. quad (4) $

  So the gradient that reaches step $t$ is governed by the product of one-step Jacobians, and $norm(lambda^t) <= (product_s norm(J^s)) norm(lambda^T)$. Whether this is benign or catastrophic is decided by the voltage component of (J) chained through the recurrent wiring.

  == Proposition 1 — the backpropagated gradient diverges (the problem)

  _Claim._ In a network with a recurrent E→I→E loop, the voltage gradient grows geometrically in the number of gamma _cycles_ traversed — not in the number of timesteps. A 200 ms trial holds only $N approx 5$ cycles against $T = 2000$ steps; the danger is that each cycle multiplies by a large factor, not that there are many cycles.

  _Proof._ Compose (2) and (3): the gradient carried from a postsynaptic voltage at $t{+}1$ to a presynaptic voltage at $t$ across one synapse is the product of the conductance-coupling and the membrane term,

  $ a eq.triple (partial V_"post"^(t+1))/(partial V_"pre"^t)
  = underbrace(kappa, approx (Delta t)/(C_m)(E_"syn"-V)) dot underbrace(beta W, "synapse") dot underbrace(sigma'(V_"pre"-theta.alt), "spike"). quad (5) $

  Traverse the loop once: $V_E -> V_I$ across $W_(e i)$ (edge gain $a_(e i)$), then $V_I -> V_E$ across $W_(i e)$ (edge gain $a_(i e)$). Over one round trip the $E$-voltage gradient maps to itself with the *loop gain*

  $ rho.alt = a_(e i) a_(i e)
  = underbrace(k^2, "two " sigma' " at volley") dot beta_e beta_i W_(e i) W_(i e) kappa_I kappa_E. quad (6) $

  The factor $k^2$ appears *once per gamma cycle*. Each population fires a single synchronous volley per cycle, so the two large kicks $sigma' -> sigma'(0) = k$ (one for $E$, one for $I$) occur together once per loop traversal and nowhere else; between volleys the per-step Jacobians are the mild sub-unit decays $alpha, beta < 1$, which merely carry the gradient along without amplifying it. So (4) collapses, across cycles, to the scalar recursion $lambda_(V,E)^((n)) approx rho.alt lambda_(V,E)^((n+1))$, giving after $N$ cycles

  $ |lambda_(V,E)| gt.eq.slant |rho.alt|^N |lambda_(V,E)^T|. quad (7) $

  *Worked example (default PING init).* Take the released constants: $Delta t = 0.1$ ms, $k = 5$, $C_m^E = 1.0$ nF, $C_m^I = 0.5$ nF, $E_e = 0$, $E_i = -80$ mV, $V approx theta.alt = -50$ mV at the crossing, $tau_"AMPA" = 2$ ms ($beta_e = e^(-0.05) approx 0.95$), $tau_"GABA" = 9$ ms ($beta_i approx 0.99$), and order-µS coupling $W_(e i) approx 1$, $W_(i e) approx 2$ µS. The two edge gains (5) are

  $ a_(e i) = underbrace((0.1)/(0.5) (0 - (-50)), kappa_(-> I) = 10) dot underbrace(0.95 dot 1, beta_e W_(e i)) dot underbrace(5, k) approx 48, $

  $ a_(i e) = underbrace((0.1)/(1.0) (-80 - (-50)), kappa_(-> E) = -3) dot underbrace(0.99 dot 2, beta_i W_(i e)) dot underbrace(5, k) approx -30, $

  so the loop gain is $rho.alt = a_(e i) a_(i e) approx -1.4 times 10^3$. Over the $N approx 5$ cycles of a 200 ms trial — not the $T = 2000$ steps — the voltage gradient is amplified by

  $ |rho.alt|^N approx (1.4 times 10^3)^5 approx 6 times 10^15. $

  A unit gradient seeded at the readout thus returns to $t = 0$ scaled by $tilde 10^16$; summed over the batch and compounded across successive optimiser steps it crosses the fp32 ceiling ($approx 3.4 times 10^38$) and the loss becomes NaN within a few batches. The blow-up is robust: even if threshold spread cuts the effective $sigma'$ tenfold (each edge $div 10$, so $rho.alt div 100$), $|rho.alt| approx 14$ and $|rho.alt|^5 approx 6 times 10^5$ — still divergent. The compounding is per cycle, not per step. ∎

  _Why the forward pass does not blow up the same way._ The forward orbit is bounded because the reset (V) slams each spiking cell to $V_r$ every cycle — that is the $rho = 0$ gate in (J), a strong per-cycle contraction. But $rho.alt$ in (6) is built only from the spike-output edges (5), i.e. from $sigma'$ evaluated _before_ the reset; the gate $rho$ sits on the membrane self-term and never enters the loop product. So the backward loop bypasses precisely the contraction that bounds the forward orbit. Forward stability and backward divergence are not in contradiction: they travel different paths through (J), and the straight-through reset is what separates them.

  == Proposition 2 — per-step voltage-gradient damping bounds it (the solution)

  The flag `--v-grad-dampen` inserts, before the reset, the operation `dv = _scale_grad(dv, 1/γ)` where $d v = (V_oo - V)(1-alpha)$ is the membrane increment in (V). The primitive

  $ "_scale_grad"(x, c) = c x + "detach"(x)(1-c) quad (8) $

  is the identity in the forward pass ($c x + (1{-}c)x = x$) but multiplies the backward gradient through $x$ by $c = 1\/gamma$. Re-differentiating (V) with $d v$ so scaled changes two partials of (J):

  $ (partial V^(t+1))/(partial V^t) = 1 - (1-alpha)/(gamma) in [alpha, 1], quad
  (partial V^(t+1))/(partial g^(t+1)) = kappa/gamma. quad (9) $

  The membrane self-term stays bounded by 1 (for $gamma -> oo$ it tends to 1 — the slow integration pathway is _preserved_, not crushed), while every voltage←conductance edge — and therefore every loop edge (5) — is divided by $gamma$. The loop gain (6) becomes

  $ rho.alt_gamma = a_(e i)/gamma a_(i e)/gamma = rho.alt/(gamma^2), quad (10) $

  so the backward loop is a contraction, $|rho.alt_gamma| <= 1$, as soon as

  $ gamma >= sqrt(|rho.alt|). quad (11) $

  By (8) the forward trajectory $x^t$ is bitwise identical with or without the flag, so this is a pure modification of the gradient, not of the dynamics. ∎

  In code this is the single line `_scale_grad(dv, 1.0 / v_grad_dampen)` in the LIF step. Since $|rho.alt| tilde k^2 c$ for an order-unity loop constant $c$, the threshold (11) scales like $sqrt(|rho.alt|) tilde k sqrt(c)$, consistent with the recipes: $gamma approx 80$ for the unitless standard SNN and $gamma approx 1000$ for COBA/PING (used by #link("/nb025/")[nb025] and downstream), whose larger conductance-scale loop constant demands the larger $gamma$.

  == Corollary — the cost, and choosing γ

  The $1\/gamma$ in (9) lands on _every_ voltage←conductance gradient, not only the recurrent loop. The feedforward input also enters through $g_e$ (the term $W_"in" s_"in"$ in (C)), so the input-weight gradient is suppressed by the same factor: damping trades a slice of the legitimate learning signal for stability. Hence the operating rule — take the smallest $gamma$ satisfying (11), i.e. the smallest value that prevents overflow. Too large a $gamma$ can therefore quietly cap achievable accuracy by starving the input layer of gradient. Distinct from gradient clipping, which rescales the assembled parameter gradient after the fact: damping reshapes the recursion (4) term by term, before any parameter gradient is formed. Tightening (11) per-layer on long-trial tasks is open work.

  *Prediction.* The mechanism implies the flag is load-bearing only when the loop exists: the same network should train with damping fully off when run as COBA ($W_(e i)=0$, loop open), but fail as PING ($W_(e i)>0$) — every optimiser step's gradient going non-finite and being skipped, leaving the network frozen at chance.
]
