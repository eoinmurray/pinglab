#import "templates/article-layout.typ": journal-article
#import "templates/result-card.typ": result-figure-ref, result-card, with-result-sections
#import "templates/references.typ": journal-references
#import "/.demolab/lib.typ": data-json, data-image, cite
#import "templates/dataset.typ": data-file, inputs-ready, pending-report, run-view, input-assets
#import "templates/abstract.typ": journal-abstract
#import "templates/methods.typ": journal-methods
#let data-file = data-file.with(article: "exp033")

#let meta = (
  tags: ("data", "v36.0.0"),
  title: "Mean Field Analysis",
  created_at: "2026-05-28T00:00:00Z",
  updated_at: "2026-09-18",
  description: "A derived four-variable population-rate model supplies the oscillatory-onset, criticality and inhibitory-timescale evidence used by the manuscript synthesis.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp033",)
#let preview-figures = (
  (path: "exp033/bifurcation_compound.svg", label: "mean-field onset"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let run = data-json(data-file("exp033/numbers.json"))
#let cfg = run.config
#let hopf = run.results.hopf
#let crit = run.results.criticality
#let istar = calc.round(hopf.I_ext_star, digits: 3)
#let fstar = calc.round(hopf.freq_star_Hz, digits: 1)
#let omegastar = calc.round(hopf.omega_star, digits: 3)
#let a2mant = calc.round(crit.A2_slope * 10000, digits: 1)
#let a2r2 = calc.round(crit.A2_r2, digits: 3)
#let tg = calc.round(cfg.tau_GABA_ms, digits: 0)

#let body = [
  #journal-abstract(body: [
  We asked whether a population-rate closure of the recurrent excitatory–inhibitory
  circuit develops oscillatory instability and whether inhibitory decay controls
  its frequency.

  The four-variable model showed a reversible oscillatory onset whose frequency
  decreased with inhibitory decay. It supplies a theoretical comparison for the
  separate spiking-network synthesis, not evidence that those networks undergo
  the same bifurcation.
  ])

  == Results

  #with-result-sections[
  #result-card[
  === Oscillatory onset and timescale

  At the prescribed 4-mV effective noise scale, one complex-conjugate eigenvalue
  pair crossed at #istar nA with onset frequency #fstar Hz. Excitatory-rate
  amplitude increased continuously above onset, the sampled upward and downward
  ramps nearly coincided, and onset frequency decreased as inhibitory decay
  increased (#result-figure-ref(<fig-overview>)). The finite ramps were consistent
  with a supercritical transition under the predefined numerical criteria; no
  first Lyapunov coefficient was calculated. The noise scale was neither measured
  nor fitted to the spiking model.

  #figure(
    data-image(data-file("exp033/bifurcation_compound.svg"), width: 100%,
      alt: "Eigenvalue continuation, upward and downward oscillation-amplitude ramps, and mean-field onset frequency across inhibitory decay constants."),
    caption: [
      *Mean-field oscillatory onset.* (A) Continuous-time Jacobian eigenvalues
      across external drive; the refined crossing is marked. (B) Upward and
      downward peak-to-peak excitatory-rate amplitudes over the final 500 ms of
      each 2-s drive step. (C) Refined onset frequency at inhibitory decay
      constants of 4.5, 6, 9, 12, 18 and 27 ms. These deterministic calculations
      have no statistical uncertainty intervals.
    ],
  ) <fig-overview>
  ]
  ]

  #journal-methods(
    orientation: [
  We derived and evaluated a deterministic four-variable population-rate closure
  independently of the trained spiking classifiers used in the manuscript synthesis.
    ],
    compute: [
  + *Define the population model.* Excitatory and inhibitory population rates
    relaxed toward stationary noisy-LIF gains #cite(1). Exponential AMPA and
    GABA conductances retained the recurrent feedback dynamics:

    $ tau_(r,E) dot(r)_E &= -r_E + Phi_E (I_"ext" - Delta V_"inh" g_I^E), \
      tau_(r,I) dot(r)_I &= -r_I + Phi_I (Delta V_"exc" g_E^I), \
      dot(g)_E^I &= -g_E^I / tau_"AMPA" + G_(E arrow.r I) r_E, \
      dot(g)_I^E &= -g_I^E / tau_"GABA" + G_(I arrow.r E) r_I. $ <eq-model>

    Here $r_E$ and $r_I$ are population rates in inverse milliseconds,
    $g_E^I$ and $g_I^E$ are conductances in µS, and $I_"ext"$ is excitatory
    drive in nA. Rate-relaxation times were 20/5 ms, AMPA/GABA decay times were
    2/#tg ms at reference, fixed excitatory/inhibitory driving-force magnitudes
    were 65/15 mV, and summed recurrent conductances were 1/2 µS. The derivation
    and closure assumptions are given in
    #link(<appendix-model>)[Appendix A — Derivation of the population-rate model].
    ],
    analyse: [
  #set enum(start: 2)

  + *Locate oscillatory instability.* We continued fixed points over 401 drives
    from 0–4 nA. Centred finite differences formed the continuous-time Jacobian;
    the first leading complex-pair crossing was refined with Brent's method. For
    angular frequency $omega_"Hopf"$ in rad/ms, onset frequency was

    $ f_"Hopf" = 1000 omega_"Hopf" / (2 pi). $ <eq-frequency>

    #link(<appendix-continuation>)[Appendix B1 — Linear stability and onset refinement]
    gives the solver, initialization, eigenvalue-selection and refinement details.

  + *Test sampled criticality.* LSODA integrated 25 drives from 0.1 nA below
    to 0.55 nA above onset in ascending and descending order, carrying each
    endpoint into the next step. Each step lasted 2,000 ms; peak-to-peak
    excitatory-rate amplitude used the final 500 ms. Numerical consistency with
    supercriticality required branch gap below $10^(-4)$ $"ms"^(-1)$, positive
    amplitude-squared slope and $R_"fit"^2 > 0.9$; the estimator, regression and
    limitations are specified in
    #link(<appendix-criticality>)[Appendix B2 — Amplitude ramps and classification].

  + *Vary inhibitory decay.* We repeated the fixed-point continuation and
    crossing refinement at $tau_"GABA" = 4.5, 6, 9, 12, 18$ and $27$ ms while
    holding the remaining model parameters fixed. The scope of its relationship
    to the manuscript comparison is stated in
    #link(<appendix-timescale>)[Appendix B3 — Inhibitory-timescale comparison].
    ],
    present: [
  #set enum(start: 5)

  + *Display the retained comparison.* We plotted the reference eigenvalue
    continuation, both amplitude-ramp directions and the inhibitory-timescale
    sweep without statistical uncertainty intervals.
    ],
  )
  #run-view("exp033", inputs)

  == Appendix A — Derivation of the population-rate model <appendix-model>

  === A1 — Conductance-based starting point

  The starting model contained excitatory and inhibitory leaky integrate-and-fire
  membranes. Let $V_m^P$ be membrane voltage, $C_m^P$ capacitance, $g_L^P$ leak
  conductance and $E_L$, $E_e$ and $E_i$ the leak, excitatory and inhibitory
  reversal potentials for population $P in {E,I}$. Excitatory neurons received
  external excitation and recurrent inhibition; inhibitory neurons received
  recurrent excitation:

  $ C_m^E dot(V_m^E) &= -g_L^E (V_m^E - E_L)
      - g_e^E (V_m^E - E_e) - g_i^E (V_m^E - E_i), \
    C_m^I dot(V_m^I) &= -g_L^I (V_m^I - E_L)
      - g_e^I (V_m^I - E_e). $ <eq-membranes>

  A neuron emitted a spike when its candidate voltage crossed $V_"th"$, after
  which voltage reset to $V_"reset"$ and remained refractory for the
  population-specific refractory period. Each conductance followed a first-order
  exponential filter. For example, recurrent excitation onto the inhibitory
  population and inhibition onto the excitatory population obeyed

  $ tau_"AMPA" dot(g)_e^I &= -g_e^I + tau_"AMPA" W^(E I) s^E (t), \
    tau_"GABA" dot(g)_i^E &= -g_i^E + tau_"GABA" W^(I E) s^I (t), $ <eq-synapses>

  where $s^E (t)$ and $s^I (t)$ are presynaptic impulse trains and $W^(E I)$ and
  $W^(I E)$ are recurrent conductance matrices. The original discrete
  exponential-Euler synapses are the timestep form of these continuous filters.

  === A2 — Replace external spikes by tonic drive

  Under a constant external input rate, the external AMPA conductance approaches
  a stationary mean. We replaced its excitatory current by a tonic control current
  $I_"ext"$ and retained the two recurrent conductances:

  $ C_m^E dot(V_m^E) &= -g_L^E (V_m^E - E_L)
      - g_i^E (V_m^E - E_i) + I_"ext", \
    C_m^I dot(V_m^I) &= -g_L^I (V_m^I - E_L)
      - g_e^I (V_m^I - E_e). $ <eq-tonic-drive>

  This substitution defines the swept control parameter. It does not equate a
  particular value of $I_"ext"$ with an empirical input rate or recruitment
  threshold in the separate spiking classifiers.

  === A3 — Average homogeneous recurrent input

  The motivating network had $N_E=1024$ excitatory and $N_I=256$ inhibitory
  neurons. For an inhibitory neuron $k$, recurrent excitation is

  $ (W^(E I) s^E)_k = sum_(j=1)^(N_E) W^(E I)_(k j) s_j^E. $ <eq-presynaptic-sum>

  We replaced individual recurrent weights by their population means and defined
  smooth population rates

  $ r_E (t) &= 1/N_E sum_(j=1)^(N_E) s_j^E (t), \
    r_I (t) &= 1/N_I sum_(k=1)^(N_I) s_k^I (t). $ <eq-population-rates>

  With summed conductances

  $ G_(E arrow.r I) = macron(w)^(E I) N_E, quad
    G_(I arrow.r E) = macron(w)^(I E) N_I, $ <eq-summed-conductance>

  the recurrent filters become

  $ dot(g)_E^I &= -g_E^I / tau_"AMPA" + G_(E arrow.r I) r_E, \
    dot(g)_I^E &= -g_I^E / tau_"GABA" + G_(I arrow.r E) r_I. $ <eq-population-conductances>

  This homogeneous smooth-rate approximation discards weight heterogeneity,
  finite-size fluctuations and correlations produced by recurrent synchrony.
  Consequently, the deterministic closure cannot establish how those effects
  alter or smear an onset in the finite spiking network.

  === A4 — Fix synaptic driving forces

  A conductance current contains the product $-g (V_m - E_"rev")$. We evaluated
  the synaptic driving forces at the resting voltage $E_L=-65$ mV while leaving
  leak, threshold and reset inside the single-neuron gain calculation. With
  $E_e=0$ mV and $E_i=-80$ mV, the magnitudes were

  $ Delta V_"exc" = |E_e-E_L| = 65 "mV", quad
    Delta V_"inh" = |E_L-E_i| = 15 "mV". $ <eq-driving-forces>

  The effective currents entering the excitatory and inhibitory gains were then

  $ I_"eff"^E = I_"ext" - Delta V_"inh" g_I^E, quad
    I_"eff"^I = Delta V_"exc" g_E^I. $ <eq-effective-currents>

  Freezing the driving forces removes the $g V_m$ product and therefore omits
  conductance-dependent shunting and its shortening of the effective membrane
  time constant. This is a modelling approximation, not an exact population
  reduction of the conductance-based network.

  === A5 — Close population rates with noisy-LIF gains

  For population $P$, the stationary noisy-LIF gain was

  $ Phi_P (I) &= [tau_("ref",P) + tau_(m,P) sqrt(pi) Q_P (I)]^(-1), \
    Q_P (I) &= integral_(a_P)^(b_P) e^(u^2) (1 + "erf"(u)) dif u, \
    a_P &= (V_"reset" - mu_(V,P)) / sigma_V, \
    b_P &= (V_"th" - mu_(V,P)) / sigma_V, \
    mu_(V,P) &= E_L + I/g_(L,P). $ <eq-noisy-lif-gain>

  Here $I$ is mean input current in nA, $mu_(V,P)$ its equivalent voltage,
  $sigma_V$ the prescribed effective voltage-noise scale and $u$ a dimensionless
  integration variable. The result is a rate in inverse milliseconds. We used
  $E_L=V_"reset"=-65$ mV, $V_"th"=-50$ mV, E/I membrane times 20/5 ms,
  leak conductances 0.05/0.10 µS, refractory periods 1.2/0.6 ms and
  $sigma_V=4$ mV.

  Replacing the population response instantaneously by $Phi_P$ would remove the
  rate dynamics. Instead, we introduced phenomenological relaxation on the
  membrane timescales:

  $ tau_(r,E) dot(r)_E &= -r_E + Phi_E (I_"eff"^E), \
    tau_(r,I) dot(r)_I &= -r_I + Phi_I (I_"eff"^I). $ <eq-rate-relaxation>

  The relaxation law and the effective noise scale are closure assumptions:
  $sigma_V$ was not estimated from voltage recordings or calculated
  self-consistently from population activity, and the deterministic equations do
  not contain an explicit stochastic drive.

  The recurrent summed conductances were $G_(E arrow.r I)=1$ µS and
  $G_(I arrow.r E)=2$ µS. Under fan-in normalization, an individual pathway's
  mean weight is its summed conductance divided by the number of presynaptic
  neurons. These fixed values are inherited baseline parameters, not fitted
  final-checkpoint weights.

  === A6 — Assemble the four-variable model

  Substituting the effective currents from @eq-effective-currents into the rate
  closure @eq-rate-relaxation and adjoining the population conductance filters
  @eq-population-conductances yields the four equations stated in @eq-model:

  $ tau_(r,E) dot(r)_E &= -r_E + Phi_E (I_"ext" - Delta V_"inh" g_I^E), \
    tau_(r,I) dot(r)_I &= -r_I + Phi_I (Delta V_"exc" g_E^I), \
    dot(g)_E^I &= -g_E^I / tau_"AMPA" + G_(E arrow.r I) r_E, \
    dot(g)_I^E &= -g_I^E / tau_"GABA" + G_(I arrow.r E) r_I. $

  == Appendix B — Numerical protocol and interpretation <appendix-numerics>

  === B1 — Linear stability and onset refinement <appendix-continuation>

  At a fixed point, let $Phi'_E$ and $Phi'_I$ denote gain derivatives with
  respect to their current arguments. In the state ordering
  $(r_E,r_I,g_E^I,g_I^E)$, the continuous-time Jacobian is

  $ J_"flow" = mat(
      -1/tau_(r,E), 0, 0, -Delta V_"inh" Phi'_E/tau_(r,E);
      0, -1/tau_(r,I), Delta V_"exc" Phi'_I/tau_(r,I), 0;
      G_(E arrow.r I), 0, -1/tau_"AMPA", 0;
      0, G_(I arrow.r E), 0, -1/tau_"GABA"
    ). $ <eq-jacobian>

  Each linear mode evolves as $e^(lambda_J t)$. Negative
  $op("Re") lambda_J$ gives decay and positive $op("Re") lambda_J$ gives
  growth. A simple Hopf bifurcation requires one conjugate pair to cross the
  imaginary axis with nonzero angular frequency while the remaining modes stay
  damped #cite(2).

  We solved the two population-rate self-consistency equations at 401 equally
  spaced currents from 0 to 4 nA, recovering conductances from their stationary
  relations. The first solve used rates 0.005 and 0.002 $"ms"^(-1)$; each
  subsequent solve started from the preceding fixed point. Negative trial rates
  contributed zero conductance during root finding.

  We calculated $J_"flow"$ by centred differences with perturbations of
  $10^(-6)$ in the corresponding rate or conductance units. Eigenvalues with
  imaginary-part magnitude above $10^(-6)$ $"ms"^(-1)$ were eligible for the
  leading complex pair. The first change from negative to nonnegative real part
  bracketed onset. Brent's method then recomputed the fixed point and Jacobian
  with absolute current tolerance $10^(-10)$ nA and relative tolerance
  $10^(-12)$.

  Gain quadrature used adaptive integration with at most 200 subdivisions. For
  negative integration variable $u$, we evaluated the integrand as
  $"erfcx"(-u)=exp(u^2)(1+"erf"(u))$ to avoid cancellation; for nonnegative
  $u$, the exponent was capped at 700.

  At the refined crossing,

  $ I_("ext,Hopf") = #istar "nA", quad
    omega_"Hopf" = #omegastar "rad/ms", quad
    f_"Hopf" = #fstar "Hz". $ <eq-recorded-crossing>

  The Jacobian establishes the local linear stability change. It does not by
  itself determine whether the bifurcation is supercritical or subcritical;
  that distinction requires nonlinear information.

  === B2 — Amplitude ramps and classification <appendix-criticality>

  The upward ramp used 25 equally spaced currents from
  $I_"ext"^*-0.1$ to $I_"ext"^*+0.55$ nA. We initialized the lowest-current
  fixed point with a $10^(-3)$ $"ms"^(-1)$ excitatory-rate perturbation and
  carried each integration's final state into the next, including the transition
  to the descending ramp. LSODA used relative tolerance $10^(-7)$, absolute
  tolerance $10^(-10)$ and maximum step 1 ms.

  For each drive, the measured peak-to-peak amplitude was

  $ A_"pp" = max_(t in cal(T)_"obs") r_E (t)
      - min_(t in cal(T)_"obs") r_E (t), $ <eq-amplitude>

  where $cal(T)_"obs"$ is the final 500 ms of the 2-s integration. We defined
  the branch gap as the largest absolute upward/downward amplitude difference
  at matched currents. For all upward-ramp points satisfying
  $I_"ext">I_"ext"^*+10^(-9)$ nA, we fitted

  $ A_"pp"^2 approx m (I_"ext" - I_"ext"^*) + c $ <eq-amplitude-fit>

  by unweighted least squares with freely fitted intercept $c$. The sampled
  onset was labelled consistent with supercriticality when the branch gap was
  below $10^(-4)$ $"ms"^(-1)$, $m>0$, and $R_"fit"^2>0.9$. The retained fit had
  $m=#a2mant times 10^(-4)$ $"ms"^(-2)$/nA and
  $R_"fit"^2=#a2r2$.

  For a supercritical Hopf, the local normal form predicts
  $A_"pp" prop sqrt(I_"ext"-I_"ext"^*)$ #cite(2). The finite-duration,
  finite-grid ramps were consistent with that scaling and showed no resolved
  hysteresis. They do not prove the absence of a narrower bistable interval or
  an unstable cycle, and no first Lyapunov coefficient was computed.

  === B3 — Inhibitory-timescale comparison <appendix-timescale>

  We repeated the same fixed-point continuation and onset refinement at
  inhibitory decay constants of 4.5, 6, 9, 12, 18 and 27 ms. The resulting
  descending onset-frequency trend establishes a relationship internal to this
  mean-field model. The manuscript synthesis separately compares these eigenfrequencies with
  finite-drive spectral peaks from trained spiking classifiers; the shared
  dependence does not establish quantitative calibration or a common
  bifurcation.

  #journal-references((
    (text: [K. Kreutz-Delgado. “Mean Time-to-Fire for the Noisy LIF Neuron:
      A Detailed Derivation of the Siegert Formula.” _arXiv_ (2015).],
      doi: "10.48550/arXiv.1501.04032"),
    (text: [Y. A. Kuznetsov.
      #link("https://www.ma.ic.ac.uk/~dturaev/kuznetsov.pdf")[_Elements of Applied Bifurcation Theory_],
      second edition. Springer (1998), sections 3.4 and 5.2.]),
  ))
]
#body
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [Does the derived four-variable population-rate model develop oscillatory onset, and does inhibitory decay control its frequency?],
    preview-figures, json-inputs: ("exp033",),
  )
}

#let meta = meta + (assets: input-assets("exp033", inputs))
#let body = journal-article("exp033", inputs, report-body, dataset-placed: inputs-ready(data-file, inputs))
