#import "contents.typ": with-contents
#import "/.demolab/lib.typ": data-json, data-image, cite, reference-list
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp033")

#let meta = (
  status: "[▦ DATA]",
  title: "Gamma Emerges at a Hopf Bifurcation",
  date: "2026-05-28",
  updated_at: "2026-08-28",
  description: "A four-variable population-rate model links oscillatory onset to synaptic timescales, with explicit limits on its connection to spiking recruitment.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp033",)
#let preview-figures = (
  (path: "exp033/bifurcation_compound.svg", label: "bifurcation compound"),
  (path: "exp033/sigma_sensitivity.svg", label: "sigma sensitivity"),
  (path: "exp033/eigenvalues_complex.svg", label: "eigenvalues complex"),
  (path: "exp033/freq_vs_tau_gaba.svg", label: "freq vs tau gaba"),
  (path: "exp033/hysteresis.svg", label: "hysteresis"),
  (path: "exp033/limit_cycle.svg", label: "limit cycle"),
  (path: "exp033/timeseries.svg", label: "timeseries"),
  (path: "exp033/phase_planes.svg", label: "phase planes"),
  (path: "exp033/reduction_ladder.svg", label: "reduction ladder"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let run = data-json(data-file("exp033/numbers.json"))
#let cfg = run.config
#let hopf = run.results.hopf
#let crit = run.results.criticality
#let lc = run.results.limit_cycle
#let d3 = run.results.reductions.three_d_qss
#let istar = calc.round(hopf.I_ext_star, digits: 2)
#let fstar = calc.round(hopf.freq_star_Hz, digits: 1)
#let fstar0 = calc.round(hopf.freq_star_Hz, digits: 0)
#let omegastar = calc.round(hopf.omega_star, digits: 3)
#let a2mant = calc.round(crit.A2_slope * 10000, digits: 1)
#let a2r2 = calc.round(crit.A2_r2, digits: 3)
#let gapmant = calc.round(crit.hyst_gap * 1000000, digits: 0)
#let hystwidth = calc.round(crit.hyst_width_nA, digits: 0)
#let estar = calc.round(hopf.fp_at_star.at(0) * 1000, digits: 2)
#let irate = calc.round(hopf.fp_at_star.at(1) * 1000, digits: 2)
#let elag = calc.round(lc.e_leads_i_ms, digits: 1)
#let istar3 = calc.round(d3.I_ext_star, digits: 2)
#let fstar3 = calc.round(d3.freq_star_Hz, digits: 0)
#let tg = calc.round(cfg.tau_GABA_ms, digits: 0)
#let fspk = calc.round(run.results.frequency_vs_tau_gaba.spiking_exp041.at("6.0"), digits: 1)
#let sens = run.results.sigma_sensitivity
#let sens-first = sens.rows.first()
#let sens-last = sens.rows.last()
#let sens-i-lo = calc.round(calc.min(..sens.rows.map(r => r.hopf.I_ext_star)), digits: 2)
#let sens-i-hi = calc.round(calc.max(..sens.rows.map(r => r.hopf.I_ext_star)), digits: 2)
#let sens-a-hi = calc.round(sens-first.limit_cycle.e_peak_to_peak * 1000, digits: 1)
#let sens-a-lo = calc.round(sens-last.limit_cycle.e_peak_to_peak * 1000, digits: 1)


#let body = [
  #set math.equation(numbering: "(1)")
  == Abstract

  A four-variable population-rate model lost stability through an oscillatory
  crossing at #istar nA and approximately #fstar Hz. Reusing numerical sweeps
  with fixed cellular and coupling parameters, we compared inhibitory decay,
  effective voltage noise and quasi-steady reductions. Up/down amplitude ramps
  were consistent with a supercritical onset at the sampled resolution. Slower
  inhibition reduced frequency in both the model and separately trained spiking
  networks, without quantitative agreement. Eliminating fast excitation preserved
  a three-variable oscillation; all six tested two-variable reductions lost it.
  \[(!) This supplies a candidate mechanism for recruitment, not a demonstrated
  identification of the spiking transition.\]

  #run-view("exp033", inputs)

  == Results: Up/down sweeps show no resolved hysteresis at oscillatory onset

  === Oscillatory onset

  #figure(
    data-image(data-file("exp033/bifurcation_compound.svg"), width: 100%,
      alt: "Three panels showing eigenvalue crossing, amplitude ramps and frequency versus inhibitory decay."),
    caption: [
      At the reference noise scale of 4 mV, one conjugate pair crosses at
      #istar nA with onset frequency #fstar Hz. Up/down amplitudes nearly coincide;
      inhibitory-decay trends agree qualitatively with separately measured spiking
      rhythms. \[(!) This is a candidate explanation of the
      #link("/exp025/")[input-coupling recruitment transition], whose empirical
      marker is an inhibitory-rate crossing under input-weight scaling, not a
      fitted Hopf current. The model alone identifies neither that transition nor
      a minimum sustainable firing rate.\]
    ],
  ) <fig-overview>

  === Sensitivity to the effective noise scale

  #figure(
    data-image(data-file("exp033/sigma_sensitivity.svg"), width: 100%,
      alt: "Noise-scale sensitivity of onset drive, absolute frequency, equilibrium rates and relative-onset amplitude."),
    caption: [
      Across 3, 4, 5 and 6 mV, the retained tests support a reversible onset;
      the threshold spans #sens-i-lo–#sens-i-hi nA while frequency remains near
      #fstar Hz. E amplitude at onset plus 0.4 nA falls from #sens-a-hi to
      #sens-a-lo Hz. At the 4 mV reference, equilibrium E/I rates are #estar and
      #irate Hz: \[(!) the unstable equilibrium is low-rate, not silent.\] The noise
      parameter is free; this sweep is not a calibration to spiking activity.
    ],
  ) <fig-sigma>

  === Linear stability

  #figure(
    data-image(data-file("exp033/eigenvalues_complex.svg"), width: 100%,
      alt: "Four eigenvalues per drive, with one conjugate pair crossing the imaginary axis."),
    caption: [
      Each drive contributes four eigenvalues. Horizontal position is growth
      rate and vertical position angular frequency, both in inverse milliseconds;
      colour denotes drive. Cyan circles mark the first oscillatory crossing;
      the remaining pair is damped there. The 401-point continuation comes from
      a separate execution of the same mean-field model used in the
      #link("/exp054/")[coupling-map comparison], with matching retained onset
      and amplitude summaries; it is not independent spiking confirmation.
    ],
  ) <fig-eigenvalues>

  === Inhibitory decay and rhythm frequency

  #figure(
    data-image(data-file("exp033/freq_vs_tau_gaba.svg"), width: 100%,
      alt: "Both frequencies decrease with inhibitory decay; the mean-field curve crosses above the spiking curve at 27 ms."),
    caption: [
      Mean-field onset frequency and the median measured frequency of three
      separately trained spiking networks per decay time both decrease. At #tg ms
      they are #fstar and #fspk Hz; \[(!) the mean-field curve is below the spiking curve
      through 18 ms and above it at 27 ms.\] The
      #link("/exp041/")[trained-network frequency study] supplies the measurements,
      not fitted gain parameters. Retraining changes weights as well as inhibitory
      decay, whereas this model holds couplings fixed. \[(!) Omitted spike synchrony is
      one possible explanation of the mismatch, not a tested cause.\] No uncertainty
      interval is shown.
    ],
  ) <fig-frequency>

  === Reversibility at the sampled resolution

  #figure(
    data-image(data-file("exp033/hysteresis.svg"), width: 100%,
      alt: "Upward and downward drive ramps have nearly matching peak-to-peak excitatory amplitudes."),
    caption: [
      The 25-point ramps give a maximum branch gap of
      $#gapmant times 10^(-6)$ $"ms"^(-1)$ and measured hysteresis width #hystwidth nA
      at the 0.1 Hz amplitude threshold. The rising-branch squared-amplitude fit
      has slope $#a2mant times 10^(-4)$ $"ms"^(-2)$/nA and $R^2 = #a2r2$.
      \[(!) These diagnostics support a supercritical interpretation; they do not rule
      out a narrow bistable interval or an unstable cycle. No first Lyapunov
      coefficient was computed #cite(1).\]
    ],
  ) <fig-hysteresis>

  === Above-onset waveform

  #figure(
    data-image(data-file("exp033/limit_cycle.svg"), width: 100%,
      alt: "Excitatory and inhibitory rates over three onset periods, displayed on separate vertical axes."),
    caption: [
      Reused E (black) and I (red) trajectories at onset plus 0.4 nA,
      over a window of three onset periods. Rates are in inverse milliseconds
      on separate axes. The absolute cross-correlation peak lag is #elag ms;
      \[(!) this magnitude is not a signed causal delay or a synaptic round-trip time.\]
      Raw samples are unavailable for remeasurement; the retained waveform and
      scalar are historical observations, not a new simulation.
    ],
  ) <fig-cycle>

  === Timing through the feedback loop

  #figure(
    data-image(data-file("exp033/timeseries.svg"), width: 100%,
      alt: "The four state variables share a time axis and show the lagged excitatory-inhibitory feedback sequence."),
    caption: [
      Reused trajectories at onset plus 0.4 nA, in loop order
      $E -> g_e^I -> I -> g_i^E$. Rates are in inverse milliseconds and
      conductances in µS. AMPA closely tracks E while the other variables show
      larger phase offsets. \[(!) These traces illustrate the feedback timing; the ordering does not measure four independent transmission delays.\]
    ],
  ) <fig-timeseries>

  === Pairwise projections

  #figure(
    data-image(data-file("exp033/phase_planes.svg"), width: 100%,
      alt: "The same four-variable trajectory projected onto each of the six coordinate pairs."),
    caption: [
      Reused projections at onset plus 0.4 nA; rate coordinates are in
      inverse milliseconds and conductances in µS. The E–AMPA projection is narrow;
      other pairs enclose larger areas. \[(!) A periodic orbit is a curve and need not
      be planar. These projections do not demonstrate a centre manifold or prove
      that a particular pair closes the dynamics.\]
    ],
  ) <fig-phase>

  === Quasi-steady reductions

  #figure(
    data-image(data-file("exp033/reduction_ladder.svg"), width: 100%,
      alt: "At common drive, four- and three-variable probes oscillate while the rate-slaved two-variable probe decays."),
    caption: [
      Reused inhibitory-conductance deviations (µS) after small kicks at
      1 nA. The full model and AMPA-slaved three-variable reduction oscillate;
      the rate-slaved two-variable probe decays. Their onset frequencies are
      approximately #fstar0 and #fstar3 Hz, respectively; these are not measured
      frequencies of the displayed 1 nA traces. All six original-variable
      two-dimensional reductions have negative divergence. \[(!) The three-variable
      minimum applies to this QSS family, not to all possible models #cite(1).\]

    ],
  ) <fig-ladder>

  == Methods

  We reused a deterministic population-rate analysis and compared its onset
  frequencies with independent measurements from trained spiking networks.

  + *Define the population model.* E/I rates relaxed toward noisy LIF gains
    #cite(2), with membrane times 20/5 ms and AMPA/GABA times 2/#tg ms.
    Fixed excitatory/inhibitory driving forces were 65/15 mV; lumped conductance
    increments were 1/2 µS. The four state variables followed

    $ tau_E dot(E) &= -E + Phi_E (I_"ext" - 15 g_i^E), \
      tau_I dot(I) &= -I + Phi_I (65 g_e^I), \
      tau_"AMPA" dot(g)_e^I &= -g_e^I + tau_"AMPA" tilde(W)^(E I) E, \
      tau_"GABA" dot(g)_i^E &= -g_i^E + tau_"GABA" tilde(W)^(I E) I. $ <eq-model>

    Here $E,I$ are rates ($"ms"^(-1)$), $g$ conductances (µS), $I_"ext"$ drive (nA),
    $Phi$ steady-state gains, $tilde(W)$ conductance increments, and $tau$ time
    constants; dots denote derivatives in milliseconds. Fixed driving forces
    omit shunting; rate relaxation and the free noise scale define a
    phenomenological closure, not a self-consistent noise theory.

  + *Locate oscillatory instability.* Fixed points were continued over 401 drives
    from 0–4 nA using nonlinear root finding. Centred differences of size
    $10^(-6)$ formed each Jacobian; the first complex-pair crossing was refined
    with Brent's method. Frequency was

    $ f^* = 1000 omega^* / (2 pi), $ <eq-frequency>

    with angular frequency $omega^*$ in rad/ms and $f^*$ in Hz; reduced-model
    crossings retained 0.01 nA grid resolution.

  + *Test onset reversibility.* LSODA integrated 25 drives from onset minus
    0.1 to onset plus 0.55 nA in each direction, carrying endpoint states forward.
    Each step lasted 2,000 ms; E peak-to-peak amplitude used the final 500 ms.
    The classifier required branch gap below $10^(-4)$ $"ms"^(-1)$, positive
    squared-amplitude slope and $R^2 > 0.9$; it was a numerical diagnostic, not
    a normal-form coefficient.

  + *Measure waveforms and reductions.* At onset plus 0.4 nA, 700 ms integrations
    supplied 1,500 samples over three onset periods for amplitude and absolute
    demeaned I–E cross-correlation lag, and 2,000 over four periods for projections.
    A separate 300 ms comparison measured amplitudes after 150 ms at onset plus
    1 nA; the illustrated reduction ladder used 400 ms at a common 1 nA.
    We tested AMPA elimination and all six two-variable QSS reductions.

  + *Vary noise and inhibitory decay.* Noise scales 3–6 mV used 121- and
    241-point drive grids over 0–1.2 nA, with refined crossings and repeated
    amplitude tests. At six inhibitory decays, onset frequencies were compared
    with three-seed medians of reused final-epoch spiking measurements.
    Each network frequency came from the interpolated peak of trial-averaged
    population spectra; these were not medians of individual-trial peaks.

  == Appendix: From spiking membranes to a population-rate closure

  === Summary of COBA model.

  Here $V$ is membrane voltage (mV), $C_m$ capacitance (nF), $g_L$ leak
  conductance (µS), and $E_L$, $E_e$, $E_i$ are leak, excitatory and inhibitory
  reversal potentials (mV). $V_"th"$ and $V_"reset"$ are threshold and reset;
  $s$ denotes a spike indicator in discrete time and a spike train in continuous
  time. $Delta t$ is the timestep, $W$ a conductance increment per spike (µS),
  and superscripts identify the receiving E or I population.

  We start from the COBANet model (#link("/exp100/")[conductance-based spiking model specification]): conductance-based E
  and I membranes, a threshold-reset rule, and three exponential synapses (no E→E; I
  receives no inhibition):
  $ C_m^E dot(V)^E = -g_L^E (V^E - E_L) - g_e^E (V^E - E_e) - g_i^E (V^E - E_i) $ <eq-old-1>
  $ C_m^I dot(V)^I = -g_L^I (V^I - E_L) - g_e^I (V^I - E_e) $ <eq-old-2>
  $ s_(t+1) = bb(1)[V >= V_"th"], quad V <- V_"reset" "if " s_(t+1)=1
    " or refractory" $ <eq-old-3>
  $ g^E_(e,t+1) = e^(-Delta t \/ tau_"AMPA") g^E_(e,t) + W_"in" s^"inp"_t $ <eq-old-4>
  $ g^E_(i,t+1) = e^(-Delta t \/ tau_"GABA") g^E_(i,t) + W_"ie" s^i_t $ <eq-old-5>
  $ g^I_(e,t+1) = e^(-Delta t \/ tau_"AMPA") g^I_(e,t) + W_"ei" s^e_t $ <eq-old-6>

  === Continuous-time form with tonic drive.

  Recast in continuous time. The synapses @eq-old-4 to @eq-old-6 are the exp-Euler form of
  first-order filters $tau dot(g) = -g + tau sum W s$, used here as ODEs. At a
  constant input rate, $g_e^E$ in @eq-old-4 settles to a steady mean, so its excitatory current
  into the E membrane @eq-old-1 is a near-constant depolarising drive; we replace it by a
  tonic current $I_"ext"$, the swept control parameter. The E membrane then carries
  $I_"ext"$ in place of $g_e^E$:
  $ C_m^E dot(V)^E = -g_L^E (V^E - E_L) - g_i^E (V^E - E_i) + I_"ext" $ <eq-old-7>
  $ C_m^I dot(V)^I = -g_L^I (V^I - E_L) - g_e^I (V^I - E_e) $ <eq-old-8>

  Here $g_i^E$ is the inhibition onto E and $g_e^I$ the excitation onto I, each a
  continuous-time exponential filter of the presynaptic spikes:
  $ tau_"AMPA" dot(g)_e^I = -g_e^I + tau_"AMPA" W^(E I) s^E (t) $ <eq-old-9>
  $ tau_"GABA" dot(g)_i^E = -g_i^E + tau_"GABA" W^(I E) s^I (t) $ <eq-old-10>

  with $s^E (t), s^I (t)$ the population spike trains and $W^(E I), W^(I E)$ the
  recurrent weight matrices; @eq-old-9 and @eq-old-10 are the continuous forms of @eq-old-5 and @eq-old-6.

  === Homogeneous coupling and population means.

  The motivating spiking network has $N_E = 1024$ excitatory cells and
  $N_I = 256$ inhibitory cells. These are population sizes, not extra state
  variables in the four-variable closure.

  Now resolve the populations: index E cells by $j in {1, ..., N_E}$ and I cells by
  $k in {1, ..., N_I}$. The recurrent drive in @eq-old-9 and @eq-old-10 is the presynaptic sum
  $W^(E I) s^E = sum_j W^(E I)_(k j) s_j^E$ (and
  $W^(I E) s^I = sum_k W^(I E)_(j k) s_k^I$). Replace each random weight by its
  population mean, $W^(E I)_(k j) -> w^(E I)$ and $W^(I E)_(j k) -> w^(I E)$; the
  sums become
  $ sum_j W^(E I)_(k j) s_j^E --> w^(E I) sum_j s_j^E = w^(E I) N_E E(t) $ <eq-old-11>
  $ sum_k W^(I E)_(j k) s_k^I --> w^(I E) sum_k s_k^I = w^(I E) N_I I(t) $ <eq-old-12>

  introducing the _population-mean firing rates_
  $ E(t) eq.triple 1/N_E sum_(j=1)^(N_E) s_j^E (t), quad
    I(t) eq.triple 1/N_I sum_(k=1)^(N_I) s_k^I (t). $ <eq-old-13>

  A _smooth-rate ansatz_ (short-window averaging) treats $E(t), I(t)$ as continuous,
  dropping weight heterogeneity and finite-size noise, the shot noise
  $"Var"[E(t)] prop E(t) \/ N_E$ for independent spike contributions at a fixed
  averaging window, with an analogous expression for I. Here $"Var"$ denotes
  variance across realizations. The corresponding typical fluctuation scale is
  $O(N^(-1 \/ 2))$, where $N$ is population size.
  \[(!) This scaling assumes independent or sufficiently weakly correlated
  contributions; recurrent synchrony can violate it. The earlier interpretation
  was that residual fluctuations at these finite population sizes smear onset
  and sustain weak noisy gamma below threshold. That is retained as a proposed
  explanation, not an effect isolated by this deterministic calculation.\]

  With no cell index left, every E cell sees the same $g_i^E$ and every I cell the
  same $g_e^I$, collapsing the per-cell conductances to population means. Defining
  lumped couplings
  $ tilde(W)^(E I) eq.triple w^(E I) N_E, quad tilde(W)^(I E) eq.triple w^(I E) N_I $ <eq-old-14>

  the conductance dynamics become
  $ tau_"AMPA" dot(g)_e^I = -g_e^I + tau_"AMPA" tilde(W)^(E I) E $ <eq-old-15>
  $ tau_"GABA" dot(g)_i^E = -g_i^E + tau_"GABA" tilde(W)^(I E) I $ <eq-old-16>

  Two equations, down from $N_E + N_I$. (The fan-in scale $tilde(W)$ folds into
  $W^(E I), W^(I E)$ in A.6.)

  _Running system, end of A.3: conductances are now two population means; the
  membrane is still per-cell but sees those means:_
  $ C_m^E dot(V)_j^E & = -g_L^E (V_j^E - E_L) - g_i^E (V_j^E - E_i) + I_"ext" \
         C_m^I dot(V)_k^I & = -g_L^I (V_k^I - E_L) - g_e^I (V_k^I - E_e) \
    tau_"AMPA" dot(g)_e^I & = -g_e^I + tau_"AMPA" tilde(W)^(E I) E \
    tau_"GABA" dot(g)_i^E & = -g_i^E + tau_"GABA" tilde(W)^(I E) I $

  === Driving-force linearisation.

  The synaptic current is conductance times a _driving force_, $-g (V - E_"rev")$,
  a $g$–$V$ product, hence nonlinear. Freeze $V$ at rest, $V_"rest" = E_L = -65$ mV,
  _in the driving force only_ (leak and threshold keep their full $V$-dependence,
  handled by the f-I curve in A.5). Each driving force becomes a fixed voltage gap:
  $ Delta V_"inh" eq.triple V_"rest" - E_i = -65 - (-80) = 15 "mV" $ <eq-old-17>
  $ Delta V_"exc" eq.triple V_"rest" - E_e = -65 - 0 = -65 "mV" $

  The synaptic currents in @eq-old-7 and @eq-old-8 then lose their $V$-dependence and become
  proportional to conductance alone:
  $ -g_i^E (V_j^E - E_i) approx -g_i^E Delta V_"inh" $ <eq-old-18>
  $ -g_e^I (V_k^I - E_e) approx -g_e^I Delta V_"exc" = +g_e^I dot |E_e - V_"rest"| $ <eq-old-19>

  (inhibition pulls $V$ down, $Delta V_"inh" = +15$ mV; excitation pushes it up,
  $|Delta V_"exc"| = 65$ mV). Removing the $g$–$V$ coupling reduces COBA to a
  current-based (CUBA) form; the cost is shunting: with $V$ fixed we ignore that
  conductance also lowers the effective time constant
  ($tau_"eff" = C_m \/ g_"tot"$, with $g_"tot"$ the total membrane conductance).

  _Running system, end of A.4: the synaptic currents are now linear in conductance
  (no $V$ left in the driving force):_
  $ C_m^E dot(V)_j^E & = -g_L^E (V_j^E - E_L) - g_i^E Delta V_"inh" + I_"ext" \
         C_m^I dot(V)_k^I & = -g_L^I (V_k^I - E_L) + g_e^I |Delta V_"exc"| \
    tau_"AMPA" dot(g)_e^I & = -g_e^I + tau_"AMPA" tilde(W)^(E I) E \
    tau_"GABA" dot(g)_i^E & = -g_i^E + tau_"GABA" tilde(W)^(I E) I $

  === Population rate from an f-I curve.

  Under @eq-old-17 to @eq-old-19 the membrane equations @eq-old-7 and @eq-old-8 read
  $C_m dot(V) = -g_L (V - E_L) + I_"syn"$, LIF with a synaptic current. A LIF cell
  under constant net current $I$ fires at its f-I rate $phi(I)$; replacing each
  cell's spikes by that rate gives
  $ E(t) approx phi_E (I_"eff"^E (t)), quad I(t) approx phi_I (I_"eff"^I (t)) $ <eq-old-20>

  with effective input currents (from @eq-old-7 and @eq-old-8 with @eq-old-17 to @eq-old-19 substituted)
  $ I_"eff"^E (t) = I_"ext" (t) - g_i^E (t) Delta V_"inh" $ <eq-old-21>
  $ I_"eff"^I (t) = g_e^I (t) |Delta V_"exc"| $ <eq-old-22>

  (I receives only excitation; E receives the drive minus inhibitory current.) The
  instantaneous-rate replacement @eq-old-20 assumes slow inputs. We approximate
  the finite population response by relaxation on $tau_E$ and $tau_I$; this is
  \[(!) a closure assumption, not an exact consequence of the single-cell gain\]:
  $ tau_E dot(E) = -E + Phi_E (I_"ext" - g_i^E Delta V_"inh") $ <eq-old-23>
  $ tau_I dot(I) = -I + Phi_I (g_e^I |Delta V_"exc"|) $ <eq-old-24>

  where $Phi_E, Phi_I$ are the smooth steady-state gain functions (the noisy LIF steady-state
  curve defined in #link(<sec-noisy-lif-gain-and-parameter-values>)[Noisy LIF gain and parameter values]). Two more equations down, together
  with @eq-old-15 and @eq-old-16, four equations in $(E, I, g_e^I, g_i^E)$.

  _Running system, end of A.5: a closed 4D rate model in $(E, I, g_e^I, g_i^E)$,
  constants not yet absorbed:_
  $ tau_E dot(E) & = -E + Phi_E (I_"ext" - g_i^E Delta V_"inh") \
             tau_I dot(I) & = -I + Phi_I (g_e^I |Delta V_"exc"|) \
    tau_"AMPA" dot(g)_e^I & = -g_e^I + tau_"AMPA" tilde(W)^(E I) E \
    tau_"GABA" dot(g)_i^E & = -g_i^E + tau_"GABA" tilde(W)^(I E) I $

  === Absorb the driving-force constants. <sec-absorb-the-driving-force-constants>

  The prefactors $Delta V_"inh", |Delta V_"exc"|$ in @eq-old-23 and @eq-old-24 and the fan-in
  scalings in @eq-old-15 and @eq-old-16 are constants carrying no dynamics; fold them into the
  couplings:
  $ W^(E I) eq.triple tilde(W)^(E I) dot |Delta V_"exc"|, quad
    W^(I E) eq.triple tilde(W)^(I E) dot Delta V_"inh" $ <eq-old-25>

  \[(!) Define current-valued coordinates $h_e^I = g_e^I |Delta V_"exc"|$ and
  $h_i^E = g_i^E Delta V_"inh"$ (nA). This invertible rescaling loses no dynamics.
  The figures retain the original conductances $g$ in µS; the equations below
  use $h$ and current-valued couplings $W$ (nA per spike).\]

  === The 4D system

  After A.1–A.6, the mean-field equations are
  $ tau_E dot(E) = -E + Phi_E (I_"ext" - h_i^E), quad
    tau_I dot(I) = -I + Phi_I (h_e^I) $ <eq-old-26>
  $ tau_"AMPA" dot(h)_e^I = -h_e^I + tau_"AMPA" W^(E I) E, quad
    tau_"GABA" dot(h)_i^E = -h_i^E + tau_"GABA" W^(I E) I $ <eq-old-27>

  in state $(E, I, h_e^I, h_i^E)$. The tested quasi-steady reductions are examined in #link(<sec-appendix-which-variables-can-be-eliminated>)[Appendix: Which variables can be eliminated?]; this is not
  a claim that four physical coordinates are the only possible description.

  === 4D Jacobian

  At a fixed point $(E^*, I^*, h_e^(I*), h_i^(E*))$:
  $ J = mat(
      -1 \/ tau_E, 0, 0, -Phi'_E \/ tau_E;
      0, -1 \/ tau_I, Phi'_I \/ tau_I, 0;
      W^(E I), 0, -1 \/ tau_"AMPA", 0;
      0, W^(I E), 0, -1 \/ tau_"GABA"
    ) $ <eq-old-28>

  Here $J$ is the derivative of the vector field with respect to its state;
  $Phi'_E, Phi'_I$ are gain derivatives with respect to input current, evaluated
  at the fixed-point arguments. Each linear mode evolves as $e^(lambda t)$,
  where $lambda$ is an eigenvalue: a negative real part decays and a positive
  real part grows. A simple Hopf requires one conjugate pair to cross with nonzero
  angular frequency while the other modes remain damped; criticality requires
  nonlinear information. Simultaneous crossings need a different analysis.

  At the retained crossing,

  $ I_"ext"^* = #istar "nA", quad omega^* = #omegastar "rad/ms", $
  $ f^* = 1000 omega^* / (2 pi) approx #fstar "Hz". $

  Here $I_"ext"^*$ is onset drive, $omega^*$ angular frequency, and $f^*$
  frequency in cycles per second. \[(!) The factor 1000 converts milliseconds
  to seconds; it was missing from the earlier Hz equation.\]

  The eigenvalue plot can be read as a sequence of linear response tests.
  A point on the real axis is a non-oscillating mode; an off-axis conjugate
  pair oscillates while decaying to the left of the imaginary axis or growing
  to its right. The cyan crossing marks the change from damping to amplification.
  A double-Hopf has two simultaneously imaginary pairs; nearby nonlinear
  interactions can produce invariant tori with two angular phases #cite(3).
  \[(!) This is general context, not an observed outcome here. The earlier
  assertion that a second pair crosses at higher drive is not supported by
  the retained 0–4 nA sweep. Moreover, this model has

  $ "tr" J = -1/tau_E - 1/tau_I - 1/tau_"AMPA" - 1/tau_"GABA" < 0, $

  where $"tr" J$ is the sum of the four eigenvalues. Two simultaneously
  imaginary pairs would require zero trace, so a double-Hopf at an equilibrium
  is excluded for this particular four-filter model with finite positive time
  constants. This does not exclude every possible torus mechanism.\]

  === Noisy LIF gain and parameter values <sec-noisy-lif-gain-and-parameter-values>
  $ phi(mu) &= [tau_"ref" + tau_m sqrt(pi) Q(mu)]^(-1), \
    Q(mu) &= integral_a^b e^(u^2) (1 + "erf" u) dif u, \
    a &= (V_"reset" - mu_V) \/ sigma_V, \
    b &= (V_"th" - mu_V) \/ sigma_V, \
    mu_V &= E_L + mu \/ g_L. $ <eq-old-29>

  Here $mu$ is mean input current (nA), $mu_V$ its equivalent voltage (mV),
  $u$ the dimensionless integration variable, and $"erf"$ the error function.
  $Q$ abbreviates the dimensionless integral; $a$ and $b$ are its scaled reset
  and threshold bounds. This is the same Siegert gain, split for readability.
  $tau_m$ and $tau_"ref"$ are membrane and refractory times (ms); $sigma_V$ is
  the effective voltage-noise scale entering this formula, not a measured
  membrane standard deviation. The rate is in inverse milliseconds. We use
  $E_L = V_"reset" = -65$ mV, $V_"th" = -50$ mV; E/I values are
  $tau_m = (20, 5)$ ms, $g_L = (0.05, 0.10)$ µS and
  $tau_"ref" = (3, 1.5)$ ms, respectively.
  The lumped conductance increments are $tilde(W)^(E I) = 1$ µS and
  $tilde(W)^(I E) = 2$ µS; fixed driving-force magnitudes are 65 and 15 mV.
  Fan-in-normalised mean weights give $tilde(W) = w N$, where $w$ is the mean
  presynaptic weight and $N$ the number of presynaptic cells. These values are
  inherited baseline settings, not fitted final-epoch weights.

  In the spiking model's strength notation,

  $ tilde(W)^(E I) = w^(E I) N_E = s, quad
    tilde(W)^(I E) = w^(I E) N_I = r s. $

  Here $s = 1$ µS is excitatory-to-inhibitory strength and $r = 2$ is the
  dimensionless inhibitory/excitatory strength ratio. Fan-in normalization
  makes the individual mean weights $s/N_E$ and $r s/N_I$. Here $s$ is a
  coupling strength, distinct from the spike indicators in A.1.
  \[(!) This restores the baseline parameter mapping; it does not identify
  these fixed couplings with the final trained weights.\]

  Quadrature uses at most 200 subdivisions and caps the exponent $u^2$ at 700.
  \[(!) The original execution reported subdivision, roundoff and convergence warnings.
  Their quantitative impact remains unresolved; agreement of retained summaries
  does not validate the underlying quadrature.\] The noise scale 3–6 mV was varied
  without calibration to spiking voltage statistics.

  == Appendix: Which variables can be eliminated? <sec-appendix-which-variables-can-be-eliminated>

  The four first-order filters preserve the sequence of excitation, recruitment
  and inhibition. Removing original variables by QSS substitution can destroy
  this feedback timing, but it is not the same operation as reducing the dynamics
  onto a centre manifold #cite(1). The following algebra retains the original
  reduction attempts, using the current-valued $h$ coordinates from #link(<sec-absorb-the-driving-force-constants>)[Absorb the driving-force constants].

  + *Route A: the textbook Wilson-Cowan model (slave the conductances).* The standard
    2D tool is two rates with instantaneous coupling, the 4D model with instantaneous
    synaptic response at unchanged steady-state coupling. Slave each conductance to its filter's steady value @eq-old-15 and @eq-old-16,
    $h_e^I = tau_"AMPA" W^(E I) E$ and $h_i^E = tau_"GABA" W^(I E) I$, and substitute
    into @eq-old-26:

    $ tau_E dot(E) = -E + Phi_E (I_"ext" - tau_"GABA" W^(I E) I), $ <eq-old-30>

    $ tau_I dot(I) = -I + Phi_I (tau_"AMPA" W^(E I) E). $ <eq-old-31>

    Its divergence (the Jacobian trace),

    $ (partial dot(E)) / (partial E) + (partial dot(I)) / (partial I)
        = -1/tau_E - 1/tau_I < 0, $ <eq-old-32>

    is a negative constant, so Bendixson–Dulac forbids a periodic orbit: no Hopf, for
    any drive or coupling on a simply connected region where the field is smooth.
    It has removed the two synaptic response lags; decay constants are filter
    response times, not fixed transmission delays. A controlled QSS approximation
    requires synapses fast relative to the retained dynamics,
    not established here: $tau_"GABA" approx #tg$ ms is not negligible relative
    to the approximately $1000 / #fstar$ ms onset period.

  + *Route B: quasi-steady-state the rates instead.* The dual move: slave the rates,
    $E = Phi_E (I_"ext" - h_i^E)$ and $I = Phi_I (h_e^I)$, into the conductance
    equations @eq-old-27, giving a 2D system in $(h_e^I, h_i^E)$:

    $ tau_"AMPA" dot(h)_e^I = -h_e^I + tau_"AMPA" W^(E I) Phi_E (I_"ext" - h_i^E), $ <eq-old-33>

    $ tau_"GABA" dot(h)_i^E = -h_i^E + tau_"GABA" W^(I E) Phi_I (h_e^I), $ <eq-old-34>

    with the same negative-constant divergence,

    $ (partial dot(h)_e^I) / (partial h_e^I) + (partial dot(h)_i^E) / (partial h_i^E)
        = -1/tau_"AMPA" - 1/tau_"GABA" < 0, $ <eq-old-35>

    so no cycle on a simply connected region with a smooth vector field:
    the displayed rate-slaved probe rings down (@fig-ladder). (These rates are the membrane variables,
    already reduced to an f-I rate.)

  + *Route C: lump into fast and slow timescales.* Slave the two fastest variables,
    the AMPA conductance ($tau_"AMPA" = 2$ ms) and the I rate ($tau_I = 5$ ms),
    keeping the two slowest ${E, h_i^E}$ ($tau_"GABA" = #tg$, $tau_E = 20$ ms):

    $ tau_E dot(E) = -E + Phi_E (I_"ext" - h_i^E), $ <eq-old-36>

    $ tau_"GABA" dot(h)_i^E = -h_i^E + tau_"GABA" W^(I E)
        Phi_I (tau_"AMPA" W^(E I) E). $ <eq-old-37>

    Trace $-1 \/ tau_E - 1 \/ tau_"GABA" < 0$: no cycle. The split is forced anyway:
    the constants interleave, $tau_"AMPA" = 2 < tau_I = 5 < tau_"GABA" = #tg <
    tau_E = 20$ ms, so "fast" and "slow" each mix a conductance with a rate.

  + *All three fail for one structural reason.* The network is a pure ring: the single
    loop $E -> h_e^I -> I -> h_i^E -> E$, no recurrent E→E or I→I and no self-drive,
    so each variable's only diagonal Jacobian term is its own decay and every gain
    $Phi'$ sits off-diagonal. Eliminate _any_ two variables and the 2D trace is
    $-1 \/ tau_a - 1 \/ tau_b < 0$, where $tau_a$ and $tau_b$ are the two
    retained time constants; Bendixson–Dulac then rules out a cycle. Routes A–C
    are three of the $binom(4, 2) = 6$ ways to pick the kept pair; the numerical study sweeps
    all six and none crosses. The negative-divergence argument applies to these
    original-variable QSS reductions. It does not exclude a nonlinear change of
    coordinates or a centre-manifold reduction of the same feedback mechanism.

    Adding recurrent E→E excitation or a cubic self-gain would change the
    diagonal dynamics and could escape this negative-divergence constraint.
    This motivated the earlier comparison with van der Pol/FitzHugh–Nagumo
    self-excitation oscillators.
    \[(!) Such added terms would change the present physical-variable ring
    model, but they are not necessary for every two-dimensional representation
    of PING. A nonlinear centre-manifold reduction can describe the same local
    feedback dynamics without adding a physical E→E connection. The earlier
    universal claim about two-dimensional oscillators was too strong.\]

  + *Three dimensions survive.* Slave only the fastest lag, the AMPA conductance
    $h_e^I = tau_"AMPA" W^(E I) E$ (@fig-phase), leaving a three-lag
    ring:

    $ tau_E dot(E) = -E + Phi_E (I_"ext" - h_i^E), $ <eq-old-38>

    $ tau_I dot(I) = -I + Phi_I (tau_"AMPA" W^(E I) E), $ <eq-old-39>

    $ tau_"GABA" dot(h)_i^E = -h_i^E + tau_"GABA" W^(I E) I, $ <eq-old-40>

    This still Hopfs. Located like the 4D bifurcation (sweep $I_"ext"$, diagonalise the
    $3 times 3$ Jacobian, find the complex-pair crossing), it gives
    $I_"ext"^* = #istar3$ nA and $f^* = #fstar3$ Hz, both above the 4D values, in the retained comparison. The six 2D reductions have no crossing in the sampled grid.
    The displayed probe compares 4D, 3D and the rate-slaved 2D model only
    (@fig-ladder); it is not a time-series panel of all six reductions.

  + *Resolution: a centre manifold is a dynamical reduction, not a coordinate pair.*
    \[(!) Near a generic simple Hopf, the centre manifold is two-dimensional and tangent
    to the critical eigenspace; it need not be a plane. Its restricted vector field
    is a two-dimensional model of the local dynamics. A periodic orbit is a
    one-dimensional curve, and closed pairwise projections alone establish neither
    the manifold nor an autonomous two-variable closure. The three-variable minimum
    found here is restricted to the tested QSS ring family; it is compatible with
    the local two-dimensional centre-manifold description #cite(1).\]


  The original dimensionality question remains useful: if the activity
  settles into a repeating rhythm, why keep four state variables? Amplitude
  and phase describe nearby oscillatory motion, while position
  projections show which coordinates nearly track each other.
  \[(!) A closed loop need not lie in a plane, and a fixed periodic orbit
  itself needs only phase to locate a point. The nearly linear E–AMPA
  projection motivates testing AMPA slaving; it is not proof that the other
  coordinate pairs cannot parameterize a local manifold. The numerical QSS
  tests and the geometric question must be distinguished.\]

  == Appendix: Numerical protocol and interpretation

  LSODA relative/absolute tolerances were $(10^(-7), 10^(-10))$ for ramps,
  $(10^(-9), 10^(-12))$ for waveforms, and $(10^(-8), 10^(-11))$ for comparisons;
  maximum steps were 1, 0.25 and 0.5 ms, respectively. Brent refinement used
  absolute/relative tolerances $(10^(-10), 10^(-12))$. The ramp began from the
  low-rate fixed point with a $10^(-3)$ $"ms"^(-1)$ E-rate kick; the waveform used
  the same kick. The 4D/2D comparison used a $2 times 10^(-3)$ $"ms"^(-1)$ E kick.
  The common-drive ladder kicked E in 4D/3D by that amount and inhibitory
  conductance in 2D by $2 times 10^(-3)$ µS. These probes are not matched
  perturbation-energy comparisons.

  The spiking comparator comprised 18 independently trained networks: three
  seeds at each inhibitory decay of 4.5, 6, 9, 12, 18 and 27 ms. Final-epoch
  measurements used 1,000 fixed MNIST test images and 200 ms trials. Population
  traces were demeaned, full-trial Hann-window power spectra averaged over
  images, and a 5–150 Hz peak interpolated with its neighbouring bins before
  taking the three-network median. Onset eigenfrequencies and finite-drive
  spiking spectral peaks are different measurements. The comparison does not
  identify a causal contribution of gamma timing to classifier accuracy.

  === What the amplitude ramps test

  The measured amplitude is

  $ A = max_(t in cal(W)) E(t) - min_(t in cal(W)) E(t), $

  where $E(t)$ is excitatory population rate and $cal(W)$ is the final
  500 ms observation window of each ramp step. Thus $A$ is peak-to-peak rate,
  in inverse milliseconds, not mean rate or oscillation power. Carrying each
  endpoint into the next drive tests whether the reached attractor depends
  on sweep direction.

  In a supercritical Hopf, a stable small cycle emerges as the equilibrium
  loses stability. In a subcritical Hopf, the nearby cycle is unstable on
  the stable-equilibrium side and can bound its basin of attraction #cite(3).
  \[(!) If a larger stable cycle also exists, a drive ramp can jump to it
  and remain there on reversal, producing a bistable hysteresis window.
  That larger cycle is an additional condition, not guaranteed by the local
  subcritical Hopf alone. Coincident sampled ramps support reversibility at
  their resolution; they do not establish absence of every unstable cycle.\]

  The supercritical normal form predicts the leading amplitude law

  $ Delta I = I_"ext" - I_"ext"^*, quad A approx C sqrt(Delta I), $
  $ A^2 approx C^2 Delta I, quad
    (dif A)/(dif I_"ext") approx C / (2 sqrt(Delta I)). $

  Here $Delta I > 0$ is excess drive (nA) and $C > 0$ converts its square
  root into peak-to-peak rate; its units are $"ms"^(-1)$/$sqrt("nA")$.
  The square-root law explains the formally unbounded onset slope in the
  ideal asymptotic description #cite(3).
  \[(!) The retained finite-range fit, with slope
  $#a2mant times 10^(-4)$ $"ms"^(-2)$/nA and $R^2 = #a2r2$, is consistent
  with this law. It does not measure an infinite derivative or establish
  that most amplitude is acquired within a particular narrow drive band.
  Those earlier claims exceeded the sampled evidence.\]

  === Mechanistic connections and their limits

  The motivating proposal was that the
  #link("/exp025/")[input-coupling recruitment transition] is a supercritical
  Hopf whose timescale comes from the E–I feedback loop.
  \[(!) The present model supplies that candidate mechanism, but the empirical
  recruitment marker uses input-weight scaling and an inhibitory-rate
  crossing. No calibration equates it with this model's current threshold.
  The equilibrium at onset has E/I rates #estar/#irate Hz and is not silent.\]

  The original frequency interpretation was that slower inhibition acts as
  the clock and that synchronous excitatory volleys sharpen the spiking
  rhythm, particularly at short inhibitory decay.
  \[(!) The descending frequency trends support a timescale connection, but
  the spiking networks were separately retrained. Synchrony was not isolated
  as the cause of the mismatch, and the mean-field curve crosses above the
  spiking curve at 27 ms rather than remaining below throughout.\]

  The waveforms illustrate E recruitment of I followed by inhibition of E,
  with near-sinusoidal rates. The earlier account identified the measured
  #elag ms lag with E leading I and with a synaptic round trip.
  \[(!) The retained scalar is the absolute cross-correlation peak lag.
  It loses the sign and cannot by itself establish a causal or round-trip
  delay. AMPA and GABA decay times are filter response times, not fixed
  transmission delays. The loop interpretation remains a mechanism to test,
  not a delay measurement recovered from that scalar.\]

  #reference-list((
    (text: [W. Zhang, V. Kirk, J. Sneyd, and M. Wechselberger.
      “Changes in the criticality of Hopf bifurcations due to certain model
      reduction techniques in systems with multiple timescales.”
      _The Journal of Mathematical Neuroscience_ 1, 9 (2011).],
      doi: "10.1186/2190-8567-1-9"),
    (text: [K. Kreutz-Delgado. “Mean Time-to-Fire for the Noisy LIF Neuron:
      A Detailed Derivation of the Siegert Formula.” _arXiv_ (2015).],
      doi: "10.48550/arXiv.1501.04032"),
    (text: [Y. A. Kuznetsov.
      #link("https://www.ma.ic.ac.uk/~dturaev/kuznetsov.pdf")[_Elements of Applied Bifurcation Theory_],
      second edition. Springer (1998), sections 3.4, 5.2 and 8.6.]),
  ))
]
#body
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [Does gamma emerge through a Hopf bifurcation in the mean-field model? Compare fixed-point stability, oscillation amplitude, and inhibitory-timescale dependence.],
    preview-figures, json-inputs: ("exp033",),
  )
}

#let meta = meta + (assets: input-assets("exp033", inputs))
#let body = with-datasets("exp033", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-contents(body)
