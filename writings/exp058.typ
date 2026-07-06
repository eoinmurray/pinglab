#import "/demolab-engine/build/lib.typ": numbers-table, provenance-footer

#let meta = (
  title: "The Canonical V&S state",
  date: "2026-06-24",
  description: "Pushing the conductance-based COBANet into the full four-coupling regime reproduces the Vreeswijk-Sompolinsky balanced, chaotic, asynchronous-irregular state.",
  collection: "miscellaneous",
  status: "draft",
)


#let body = [
  == Abstract

  #link("https://doi.org/10.1126/science.274.5293.1724")[Vreeswijk & Sompolinsky] predict that a strongly coupled, sparse, inhibition-dominated network falls untuned into a _balanced_ state: large E and I currents cancel, and the residual $O(1)$ fluctuations drive irregular (CV ≈ 1), asynchronous, rhythmless firing — _generated_ by the deterministic dynamics, not inherited from the input. We push the conductance-based #link("/ar003/")[COBANet] into the full four-coupling regime and test three predictions: the balanced signatures at fixed fan-in (Figure 1); their survival as $K -> oo$ only under strong coupling, synapse $prop J \/ sqrt(K)$ (Figure 2); and that the irregularity is deterministic chaos, by perturbing clones on noise-free input (Figure 3). All three hold — Poisson ISIs, near-zero correlations, a broadband spectrum, a supra-Poisson CV sustained as $K$ grows, and a positive Lyapunov exponent (λ ≈ +28 s⁻¹ vs ≈ 0 for a decoupled control). The lone caveat (Figure 2) is a finite-$K$ rate drift. (Gamma, the rhythmic alternative this hardware can occupy, is companion entry #link("/exp050/")[exp050].)

  == Methods

  + *Network and the $K$–$J$ mapping.* #link("/ar003/")[COBANet], $N_E = 1024$, $N_I = 256$, $Delta t = 0.25$ ms, $T = 1000$ ms. *Fixed fan-in* (exact-$K$, sparsity $s = 0.99$): every cell draws exactly $K$ inputs. *Four recurrent matrices* $W^(E I) = cal(N)(0.6, 0.18)$, $W^(I E) = cal(N)(3.0, 0.9)$, $W^(I I) = cal(N)(0.4, 0.12)$, $W^(E E) = cal(N)(0.4, 0.12)$ μS ($W^(E E)$ for completeness — the fixed-$K$ balance sets the rates, not it). *Per-cell independent Poisson drive*, uncorrelated (E: 45 Hz × 0.38 μS, I: 8 Hz × 0.25 μS). The simulator is set by sparsity $s$ and weight $w$; the theory's fan-in $K$ and coupling $J$ are derived,

    $ K = (1 - s) N_"pre" quad (1) $

    $ J = w / sqrt(K) quad <==> quad w = J sqrt(K) quad (2) $

    where:

    - $K$ — the *fan-in*: how many presynaptic cells each neuron receives input from (the size of its "crowd");
    - $J$ — the *coupling*: the $O(1)$ synaptic strength held _fixed_ as $K$ grows. The $J \/ sqrt(K)$ scaling is the load-bearing V&S choice — it keeps the mean recurrent input $O(sqrt(K))$ (large, cancels) while the fluctuation stays $O(1)$ (survives);
    - $s$ — connection *sparsity* (the _--ei-sparsity_ flag); $1-s$ is the probability any given pair is wired;
    - $N_"pre"$ — size of the presynaptic pool a cell draws from ($N_E$ or $N_I$);
    - $w$ — mean of the recurrent weight matrix in μS (the value passed to _--w-ei_ etc.).

    Exact-$K$ divides each synapse by its fan-in, so the total a cell receives, $K dot (w \/ K) = w$, is $K$-independent — giving (2). At $s = 0.99$:

    #table(
      columns: 6,
      [matrix], [direction], [$N_"pre"$], [$w$ (μS)], [$K$], [$J = w \/ sqrt(K)$],
      [$W^(E I)$], [E → I], [1024], [0.6], [≈ 10], [≈ 0.19],
      [$W^(E E)$], [E → E], [1024], [0.4], [≈ 10], [≈ 0.13],
      [$W^(I E)$], [I → E], [256], [3.0], [≈ 3], [≈ 1.73],
      [$W^(I I)$], [I → I], [256], [0.4], [≈ 3], [≈ 0.23],
    )

    So $K_E approx 10$, but the 4× smaller I pool gives only $K_I approx 3$. V&S needs $K_E, K_I$ the _same order_ — 4:1 qualifies — but $K_I approx 3$ is marginal: at equal fan-in ($K_I approx 10$) the state stays balanced and asynchronous, the CV merely relaxing from 1.1–1.2 to ≈ 1.0 — so the supra-Poisson burstiness is a lumpy-$K_I$ shot-noise effect, not a balance failure.

  + *Why it is irregular.* Each cell sums ≈ $K$ synapses of strength $prop J \/ sqrt(K)$, so the mean recurrent input is $O(sqrt(K))$ but its fluctuation is $O(1)$. A moderate rate makes the large opposing E and I means *cancel to leading order*, parking the drive near threshold; the residual $O(1)$ fluctuations carry each cell over threshold at random times (Poisson, CV → 1), and since each cell hears its own uncorrelated input, the population is asynchronous. These fluctuations are network-_generated_: they survive $K -> oo$ only under $J \/ sqrt(K)$ scaling (Step 4) and persist under noise-free input (Step 5).

  + *Diagnostics.* From post-burn-in spikes: per-neuron ISI CV, the Welch spectrum of the population-mean E trace, and the mean pairwise cross-correlogram over 100 random E pairs.

  + *The $J \/ sqrt(K)$ coupling test.* Strong coupling means synapse $prop J \/ sqrt(K)$, keeping an $O(1)$ recurrent fluctuation as $K -> oo$; holding $w$ fixed instead gives synapse $prop 1 \/ K$ — the weak (mean-field) limit, fluctuation $prop 1 \/ sqrt(K) -> 0$. At fixed $N_E = 1024$ we sweep $K = 10 -> 160$ under both: *strong* (weights $prop sqrt(K)$, drive folded into the balance — rate $prop K$, $g prop 1 \/ sqrt(K)$) and *weak* (all fixed), reading off whether the irregularity survives. (3 s trials, since the strong rate drifts low at high $K$ and the CV needs enough ISIs.)

  + *Direct Lyapunov exponent (frozen input).* Replace the Poisson drive with a *quenched DC conductance* (a frozen per-cell offset, no per-timestep fluctuation), run two clones on _identical_ input, kick every voltage by $epsilon$ at $t = 0$, and track the *voltage distance* between the copies,

    $ norm(Delta V(t)) = sqrt(sum_(i=1)^(N_E) (V_i^"clean" (t) - V_i^"pert" (t))^2) quad (3) $

    the Euclidean distance between the two copies' $N_E$ E-cell membrane-voltage vectors ($V_i^"clean"$ unperturbed, $V_i^"pert"$ kicked). With noise-free input any divergence is network-generated. We compare the *balanced* four-coupling net against a *decoupled* control: the same COBA E and I cells under the same DC drive, but with all four recurrent matrices set to ≈ 0, so no cell receives synaptic input from any other. Each decoupled cell is then an independent DC-driven integrator — a clock — and a perturbation has nowhere to spread, fixing the $lambda approx 0$ baseline. (It is _not_ a feedforward network: there is no input-layer pathway, just the same architecture with every recurrent wire cut.) Spiking dynamics contract between spikes and expand only at spike-flips, so $epsilon$ must flip spikes ($epsilon = 0.1$ mV; smaller just contracts → spurious $lambda < 0$). The slope of $log norm(Delta V)$ in the post-flip, pre-saturation window is the largest Lyapunov exponent (five seeds). It is an _initial-growth_ estimate ($norm(Delta V)$ saturates at the attractor size), so the sign, not the magnitude, is the result.

  == Results

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [Four diagnostics of the four-coupling state (E black, I red above the divider; ⟨r_E⟩ ≈ 17.5, ⟨r_I⟩ ≈ 25.4 Hz). *What we expect.* Irregular (ISI CV ≈ 1; a constant-current cell sits at 0), asynchronous (near-zero pairwise correlation), rhythmless (broadband, no peak). *What we see.* _Raster:_ scattered, no bands. _PSD:_ broadband, no sustained rhythm — the single-trial spectral max wanders seed to seed (5–90 Hz across six seeds), so it is not a peak; but a _weak_ gamma-band E–I resonance does survive seed-averaging (≈ 2× the floor near 40 Hz — the same E↔I loop that sharpens into PING gamma in #link("/exp050/")[exp050], here heavily damped). _ISI CV:_ median 1.11 (E), 1.20 (I). _Cross-correlogram:_ flat, peak |C(τ)| ≈ 0.01, despite heavy shared input — the active decorrelation of #link("https://doi.org/10.1126/science.1179850")[Renart et al. (2010)]. *Do they align?* Yes on all three — the COBANet enters the balanced state untuned.],
  )

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [Fan-in $K = 10 -> 160$ at fixed $N_E = 1024$ — equivalently sparsity $s = 1 - K \/ N_E$ from 0.99 to 0.84 (top axis) — with recurrent weights scaled $prop sqrt(K)$ (black, the V&S $J \/ sqrt(K)$ rule) versus held fixed (grey); $r_I$ shown in red. ±1 SD over three seeds, 3 s trials. *What we expect.* Strong coupling keeps the $O(1)$ fluctuations, so irregularity _survives_ as $K$ grows; weak coupling loses them ($prop 1 \/ sqrt(K)$), so cells _regularise_. Asynchrony holds in both. *What we see.* _Irregularity:_ strong stays supra-Poisson (CV 1.2–1.3, easing to 1.11 at $K = 160$); weak decays to ≈ 1.0, the Poisson floor. _Asynchrony:_ both ≈ 0.006 throughout. _Rates:_ strong $r_E$ drifts 17.5 → 4.9 Hz (the $O(1 \/ sqrt(K))$ finite-$K$ correction); weak holds ≈ 17 Hz. *Do they align?* Yes — only $J \/ sqrt(K)$ keeps the recurrent irregularity alive as $K$ grows; the gap is the signature. The one mismatch is the strong-coupling rate drift (Next steps); whether the survival is _self-generated_ is settled by Figure 3.],
  )

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [Two clones on _identical_ quenched-DC input, the second kicked by ε = 0.1 mV at $t = 0$; voltage distance $norm(Delta V(t))$, seed-mean (bold) with a ±1 SD band over five seeds, lightly smoothed. *What we expect.* If chaotic, the kick is _amplified_ — $norm(Delta V)$ grows exponentially (λ > 0) — and with noise-free input that can only be network-generated. The decoupled control should not grow (λ ≤ 0). *What we see.* _Balanced (black):_ $norm(Delta V)$ rises steeply (fitted rate λ ≈ +28 s⁻¹) then saturates at the attractor size (≈ 250 mV). _Decoupled (grey):_ neither amplified nor forgotten — a fixed phase offset, λ ≈ 0. *Do they align?* Yes — λ > 0 for the balanced net, ≈ 0 for the control: deterministic chaos, self-generated. Caveat: an _initial-growth_ estimate (saturation caps the window), so the sign, not the magnitude, is the result — a renormalised Benettin scheme would pin exact λ (Next steps).],
  )

  == Next steps

  Figures 1–3 confirm the balanced signatures, their $J \/ sqrt(K)$-only survival, and a positive Lyapunov exponent — the state is self-generated and deterministically chaotic. Two refinements remain.

  + *Hold the rate as $K$ grows.* Strong-coupling $r_E$ drifts 17.5 → 4.9 Hz (the $O(1 \/ sqrt(K))$ finite-$K$ correction). A DC offset to E alone _fails_ — it pushes E off the balanced point and inflates CV to 1.4–1.7. Honest pinning must co-adjust the E _and_ I drives along the balanced manifold (solve the 2D balance for a target $(r_E, r_I)$), giving a clean constant-rate CV comparison.

  + *Pin down the asymptotic $lambda$.* Figure 3 gives the sign; the magnitude is an initial-growth estimate (ε-dependent, saturation-capped). The exact exponent needs a _renormalised_ Benettin scheme — lockstep clones, periodic rescaling of $Delta V$, averaging $log$-growth over a long trajectory — which also yields the full spectrum and the attractor dimension.
]
