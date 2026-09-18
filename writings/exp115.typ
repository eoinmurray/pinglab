#import "templates/article-layout.typ": journal-article
#import "templates/abstract.typ": journal-abstract
#import "templates/dataset.typ": data-file, input-assets, inputs-ready, pending-report
#import "templates/parameters-table.typ": parameters-table
#import "templates/methods.typ": journal-methods, method-card
#import "templates/result-card.typ": journal-result-card, result-figure-ref, with-result-sections
#import "templates/references.typ": journal-references
#import "/.demolab/lib.typ": cite, data-image, data-json

#let data-file = data-file.with(article: "exp115")
#let equation-range(first, last) = [equations #ref(first, supplement: none)–#ref(last, supplement: none)]

#let meta = (
  tags: ("data", "v36.0.0"),
  title: "Numerical Hopf Investigation of PING",
  created_at: "2026-09-18T00:00:00Z",
  updated_at: "2026-09-18",
  description: "A four-variable mean-field investigation of PING onset, frequency and sampled criticality using equilibrium continuation and time-domain amplitude ramps.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp115",)
#let preview-figures = (
  (path: "exp115/hopf-compound.svg", label: "Hopf onset and closure sensitivity"),
)

#let parameter-section = [
  #counter(figure.where(kind: table)).update(0)
  #parameters-table(
    ([Parameter], [Symbol], [Value], [Role]),
    (
      ([Membrane time constant, E / I], [$tau_(m,E)$ / $tau_(m,I)$], [20 / 5 ms], [Inherited]),
      ([Leak conductance, E / I], [$g_(L,E)$ / $g_(L,I)$], [0.05 / 0.10 µS], [Inherited]),
      ([Membrane capacitance, E / I], [$C_(m,E)$ / $C_(m,I)$], [1 / 0.5 nF], [Inherited]),
      ([Leak / reset potential], [$E_L$ / $V_"reset"$], [−65 / −65 mV], [Inherited]),
      ([Spike threshold], [$V_"th"$], [−50 mV], [Inherited]),
      ([Refractory period, E / I], [$tau_("ref",E)$ / $tau_("ref",I)$], [1.2 / 0.6 ms], [Inherited]),
      ([Excitatory / inhibitory reversal], [$E_e$ / $E_i$], [0 / −80 mV], [Inherited]),
      ([Fixed excitatory / inhibitory driving force], [$Delta V_"exc"$ / $Delta V_"inh"$], [65 / 15 mV], [Derived closure]),
      ([AMPA decay], [$tau_"AMPA"$], [2 ms], [Inherited]),
      ([GABA decay], [$tau_"GABA"$], [Reference 6 ms; sweep 4.5, 6, 9, 12, 18, 27 ms], [Sweep design]),
      ([Summed coupling, E→I / I→E], [$G_(E arrow.r I)$ / $G_(I arrow.r E)$], [1 / 2 µS], [Inherited]),
      ([Population size, E / I], [$N_E$ / $N_I$], [1,024 / 256], [Inherited]),
      ([Effective voltage-noise scale], [$sigma_V$], [Reference 4 mV; sweep 3, 4, 5, 6 mV], [Closure choice]),
      ([Rate-relaxation multiplier], [$kappa$], [Reference 1; sweep 0.5, 0.75, 1, 1.5, 2], [Closure choice]),
      ([Rate relaxation, population $P$], [$tau_(r,P)$], [$kappa tau_(m,P)$], [Derived closure]),
      ([External excitatory drive], [$I_"ext"$], [0–4 nA; 401 points], [Numerical design]),
      ([Initial rate guess, E / I], [$r_E^(0)$ / $r_I^(0)$], [0.005 / 0.002 $"ms"^(-1)$], [Numerical design]),
      ([Hopf refinement], [$I_"ext"^*$], [Brent method; $10^(-10)$ nA absolute and $10^(-12)$ relative tolerance], [Numerical design]),
      ([Criticality ramp], [$I_"ext"-I_"ext"^*$], [−0.10–0.55 nA; 25 points each direction], [Numerical design]),
      ([Ramp duration / observation], [$T$ / $cal(T)_"obs"$], [2,000 ms / final 500 ms], [Numerical design]),
    ),
    columns: (1.25fr, 0.95fr, 1.45fr, 1.05fr),
    table-label: <tab:exp115-parameters>,
    caption: [Parameters used for the four-variable mean-field calculation. Paired values are excitatory / inhibitory. “Inherited” denotes fixed values from the canonical conductance-based PING model; “Derived closure” denotes algebraic consequences of fixing synaptic driving forces at the leak potential or of the stated relaxation relation; “Sweep design” and “Numerical design” denote experimental choices. Only the effective voltage-noise scale $sigma_V$ and the dimensionless multiplier $kappa$, defined by $tau_(r,P)=kappa tau_(m,P)$ for population $P in {E,I}$, are free closure parameters. Their displayed ranges are deterministic sensitivity analyses, not fitted uncertainty intervals. Neither parameter was calibrated from spiking-network activity.],
  )
]

#let methods-section = journal-methods(
  orientation: [
    We calculated equilibrium, Hopf-onset and time-domain criticality evidence for the
    deterministic mean-field closure. The calculations describe this closure;
    they do not establish a bifurcation in the separate spiking network.
  ],
  body: (
    method-card([Define the population model], [
      The circuit follows the reciprocal excitation–inhibition feedback motif
      of pyramidal–interneuronal network gamma (PING).#cite(1)
      The four state variables are the per-neuron excitatory and inhibitory
      rates $r_E,r_I$, in $"ms"^(-1)$, and the excitatory conductance onto I,
      $g_e^I$, and inhibitory conductance onto E, $g_i^E$, in µS.
      In #equation-range(<eq:exp115-model>, <eq:exp115-model-4>), each rate
      relaxes toward its white-noise Siegert gain#cite(2, 3); each conductance decays
      exponentially and receives the source rate multiplied by its summed
      event coupling. Excitatory gain receives tonic current $I_"ext"$, in nA, minus
      inhibitory feedback; inhibitory gain receives excitatory feedback.
      We fixed synaptic driving forces at the leak potential, discarding shunting
      and recurrent fluctuation structure. @tab:exp115-parameters gives the
      inherited neuronal and synaptic parameters; #link(<app:exp115-closure>)[Appendix A]
      states the closure assumptions.
    ]),
    method-card([Specify the closure conditions], [
      We evaluated 120 deterministic conditions crossing six inhibitory decay times
      $tau_"GABA"$, four effective voltage-noise scales $sigma_V$ and five
      rate-relaxation multipliers $kappa$, using the tabulated grids. The
      reference values are 6 ms, 4 mV and 1, respectively. For population
      $P in {E,I}$, rate relaxation time $tau_(r,P)$ is $kappa$ times membrane
      time constant $tau_(m,P)$. Noise scale and relaxation are prescribed
      closure choices, with no fit to spiking activity. Summed event couplings
      were held fixed, so changing inhibitory decay also changed stationary
      feedback strength.
    ]),
    method-card([Continue the equilibrium], [
      We evaluated the exponentially scaled Siegert integral by adaptive
      quadrature and its
      first two current derivatives analytically
      (#link(<app:exp115-gain>)[Appendix B]), using the first-passage construction
      detailed by Kreutz-Delgado.#cite(4) A Powell hybrid solver with an
      analytic Jacobian continued the coupled rate equations over 401 evenly
      spaced drives from 0 to 4 nA, initialized by the tabulated rates and then
      the preceding equilibrium. An independent scalar Brent solve of
      @eq:exp115-scalar-residual checked each solution and supplied a fallback.
      Acceptance required physical rates and scaled residuals below $10^(-10)$.
      Equilibria were shared across $kappa$, which does not change the
      equilibrium equations; flow residuals and Jacobians were recomputed for
      every condition. Failed solves could not form continuation brackets.
    ]),
    method-card([Locate Hopf crossings and onset], [
      We bracketed every sampled sign change in the difference between the
      gain-dependent loop product $K$ and its critical value $K_H$, defined in
      #link(<app:exp115-hopf>)[Appendix C]. Brent refinement used
      absolute tolerance $10^(-10)$ nA and relative tolerance $10^(-12)$,
      recomputing the equilibrium at each trial. Acceptance required a simple
      nonzero imaginary eigenvalue pair, two damped remaining modes and a
      resolved nonzero crossing derivative. We selected the first
      stable-to-unstable crossing as onset only when the lower-drive branch was
      stable. Its drive is $I_"ext"^*$. We calculated its frequency
      $f_"Hopf"$, in Hz, from the quartic identity @eq:exp115-hopf-frequency
      and checked agreement with the critical eigenvalue's positive imaginary
      part, in rad/ms, multiplied by $1000/(2 pi)$.
    ]),
    method-card([Test sampled criticality], [
      At the reference noise scale and relaxation multiplier, we integrated
      25 drives from 0.10 nA below to 0.55 nA above each accepted onset, first
      upward and then downward while carrying each final state into the next
      step. Each step lasts 2,000 ms; peak-to-peak excitatory-rate amplitude is
      measured over the final 500 ms. Numerical consistency with a supercritical
      transition requires an upward/downward branch gap below $10^(-4)$
      $"ms"^(-1)$, positive slope in a fit of squared amplitude against
      $I_"ext"-I_"ext"^*$ above onset, and $R^2>0.9$.
    ]),
    method-card([Check numerical convergence], [
      We applied #link(<app:exp115-numerics>)[Appendix D]'s residual and spectral
      criteria at every candidate crossing. Checks include 801- and 1,601-point
      drive grids with tighter quadrature, equilibrium and root tolerances.
      Acceptance requires matching crossing counts and directions, onset-drive
      changes below $10^(-7)$ nA and frequency changes below $10^(-4)$ Hz.
    ]),
    method-card([Record condition-level sensitivity], [
      We recorded equilibrium rates, conductances and Jacobian eigenvalues along each
      drive branch; every resolved crossing's drive, frequency and crossing
      derivative; the six time-domain ramps; and the corresponding residuals
      and convergence discrepancies. Comparisons retain the closure
      condition and distinguished resolved crossings, intervals without a
      resolved crossing, and failed or unresolved calculations. Variation
      across closure choices is deterministic sensitivity, not sampling
      uncertainty. Finite ramps cannot exclude a narrower bistable interval or
      an unstable cycle, and they do not establish the behaviour of the separate
      spiking network.
    ]),
  ),
)

// Methods and mathematical content remain readable without presentation data.
#let mathematical-appendices = [
  == Appendix A — Conductance-to-rate closure <app:exp115-closure>

  We define a four-variable approximation to the excitatory–inhibitory PING
  loop, retaining the reciprocal feedback motif of PING models.#cite(1)
  The reduction combines a homogeneous mean-input approximation, fixed
  synaptic driving forces, a stationary noisy-neuron gain and phenomenological
  rate relaxation. These are closure assumptions; the resulting equations are
  not an exact reduction of the finite conductance-based spiking network.
  The derivations below define the implemented calculation; numerical
  acceptance and convergence checks are specified in
  #link(<app:exp115-numerics>)[Appendix D].

  === Membrane currents and population averaging

  Let $P in {E,I}$ denote an excitatory or inhibitory population, and let
  $V_m^P$ be a representative neuron's membrane voltage. With external
  excitation replaced by the tonic current $I_"ext"$, the recurrent loop has
  subthreshold equations

  $ C_(m,E) dot(V)_m^E = -g_(L,E)(V_m^E-E_L)
      -g_i^E (V_m^E-E_i)+I_"ext", $ <eq:exp115-membranes>

  $ C_(m,I) dot(V)_m^I = -g_(L,I)(V_m^I-E_L)
      -g_e^I (V_m^I-E_e). $ <eq:exp115-membranes-2>

  Here $C_(m,P)$ is membrane capacitance, $g_(L,P)$ is leak conductance,
  $E_L$ is the leak potential, and $E_e$ and $E_i$ are excitatory and inhibitory
  reversal potentials. The conductances $g_e^I$ and $g_i^E$ denote excitation
  onto I and inhibition onto E. Threshold $V_"th"$, reset $V_"reset"$ and
  absolute refractory period $tau_("ref",P)$ complete the neuron definition.
  The membrane time constant is $tau_(m,P)=C_(m,P)/g_(L,P)$.
  Tonic drive replaces the mean external synaptic current; it supplies no
  calibration between current and external spike rate.

  For a pathway $P arrow.r Q$, write the input to target neuron $k$ as

  $ dot(g)_(P arrow.r Q,k) = -g_(P arrow.r Q,k)/tau_(P arrow.r Q)
      + sum_(j=1)^(N_P) W_(k j)^(P arrow.r Q) s_j^P (t), $ <eq:exp115-event-filter>

  $ s_j^P (t)=sum_n delta(t-t_(j n)^P). $ <eq:exp115-event-filter-2>

  The matrix $W^(P arrow.r Q)$ has target rows and source columns; its entries
  are conductance increments per presynaptic event. $N_P$ is source-population
  size, $t_(j n)^P$ is neuron $j$'s $n$th spike time, and $delta$ is the Dirac
  impulse. The pathway decay is $tau_"AMPA"$ for E→I and $tau_"GABA"$ for I→E.
  Replacing the weighted spike sum by a common smooth per-neuron rate $r_P$
  times the mean incoming weight sum gives

  $ G_(P arrow.r Q)=N_P macron(w)_(P arrow.r Q), $ <eq:exp115-mean-filter>

  $ dot(g)_e^I=-g_e^I/tau_"AMPA"+G_(E arrow.r I)r_E, $ <eq:exp115-mean-filter-2>

  $ dot(g)_i^E=-g_i^E/tau_"GABA"+G_(I arrow.r E)r_I. $ <eq:exp115-mean-filter-3>

  The mean edge increment $macron(w)_(P arrow.r Q)$ includes zero edges;
  $G_(P arrow.r Q)$ is the summed incoming conductance increment per target.
  Fan-in normalization holds $G$ fixed as $N_P$ changes. This approximation
  discards neuron-to-neuron heterogeneity and recurrent spike correlations.
  We use milliseconds, millivolts, microsiemens and nanoamperes, with rates in
  $"ms"^(-1)$; multiplying a rate by 1,000 converts it to hertz.
  The event filter is not normalized to unit area: its stationary conductance
  is $tau_(P arrow.r Q)G_(P arrow.r Q)r_P$. Thus a decay-time sweep also changes
  stationary feedback strength when the event increment is held fixed.

  === Fixed driving forces and dynamic rate closure

  Evaluating the synaptic driving forces at $V_m=E_L$ gives the positive
  magnitudes $Delta V_"exc"=E_e-E_L=65$ mV and
  $Delta V_"inh"=E_L-E_i=15$ mV. The effective gain currents are

  $ I_E=I_"ext"-Delta V_"inh" g_i^E, $ <eq:exp115-currents>

  $ I_I=Delta V_"exc" g_e^I. $ <eq:exp115-currents-2>

  The minus sign carries inhibition. Freezing these forces removes
  conductance-dependent shunting and the associated change in effective
  membrane time constant. We introduce a gain $Phi_P (I)$, defined in
  #link(<app:exp115-gain>)[Appendix B], and relax each rate toward it on
  $tau_(r,P)=kappa tau_(m,P)$:

  $ tau_(r,E) dot(r)_E = -r_E+Phi_E (I_"ext"-Delta V_"inh" g_i^E), $ <eq:exp115-model>

  $ tau_(r,I) dot(r)_I = -r_I+Phi_I (Delta V_"exc" g_e^I), $ <eq:exp115-model-2>

  $ dot(g)_e^I = -g_e^I/tau_"AMPA"+G_(E arrow.r I)r_E, $ <eq:exp115-model-3>

  $ dot(g)_i^E = -g_i^E/tau_"GABA"+G_(I arrow.r E)r_I. $ <eq:exp115-model-4>

  These equations define $dot(x)=F(x,I_"ext")$ in the state ordering
  $x=(r_E,r_I,g_e^I,g_i^E)^T$, where $T$ denotes transpose. The dimensionless
  multiplier $kappa$ and the effective noise scale $sigma_V$ are free closure
  choices, not quantities derived from the conductance dynamics. A stationary
  gain with one relaxation time does not reproduce the full time-dependent
  response of a noisy spiking population. Synaptic filtering remains in the
  mean currents, while fluctuation colour, rate-dependent noise variance,
  finite-population noise and synchrony-dependent corrections are omitted.
  In particular, slow GABA decay does not justify a white-noise approximation
  to the actual synaptic fluctuations. Any predicted bifurcation belongs to
  this deterministic closure.

  == Appendix B — Siegert gain and current derivatives <app:exp115-gain>

  === First-passage derivation and noise convention

  The gain uses the established Ornstein–Uhlenbeck first-passage construction
  for a leaky integrate-and-fire neuron, originating in Siegert's passage-time
  theory and its application to neuronal firing by Ricciardi and
  Sacerdote.#cite(2, 3) The derivation below follows the backward-equation
  construction detailed by Kreutz-Delgado (2015), sections 2.6 and 3,#cite(4)
  adapted to our current and voltage-noise convention.
  For fixed current $I$, our noise convention is

  $ dif V_m = (mu_(V,P)-V_m)/tau_(m,P) dif t
      + sigma_V/sqrt(tau_(m,P)) dif W_t, $ <eq:exp115-ou>

  $ mu_(V,P)=E_L+I/g_(L,P). $ <eq:exp115-ou-2>

  Here $W_t$ is a standard Wiener process whose increments have variance
  $dif t$, and $mu_(V,P)$ is the unreset mean voltage. The scale $sigma_V$ has
  units of mV. The unreset process has stationary voltage variance
  $sigma_V^2/2$, so $sigma_V$ is not its voltage standard deviation. Threshold
  and reset further change the voltage distribution. We prescribe this scale
  rather than estimating it from voltage observations.

  Let $T_P (v)$ be the mean time to first reach $V_"th"$ from voltage $v$,
  excluding refractoriness. Its backward equation and absorbing boundary are

  $ (mu_(V,P)-v)/tau_(m,P) (dif T_P)/(dif v)
      + sigma_V^2/(2 tau_(m,P)) (dif^2 T_P)/(dif v^2)=-1, $ <eq:exp115-backward>

  $ T_P (V_"th")=0. $ <eq:exp115-backward-2>

  Set $z=(v-mu_(V,P))/sigma_V$ and write
  $cal(T)_P (z)=T_P (mu_(V,P)+sigma_V z)$. The natural lower boundary excludes
  the exponentially growing homogeneous solution:

  $ 1/2 cal(T)''_P-z cal(T)'_P=-tau_(m,P), $ <eq:exp115-backward-scaled>

  $ lim_(z arrow -infinity) e^(-z^2)cal(T)'_P (z)=0. $ <eq:exp115-backward-scaled-2>

  Multiplying by $2e^(-z^2)$ and integrating from $-infinity$ to $z$ yields

  $ cal(T)'_P (z)=-2 tau_(m,P)e^(z^2) integral_(-infinity)^z e^(-y^2) dif y
      =-tau_(m,P)sqrt(pi)H(z), $ <eq:exp115-passage-gradient>

  $ H(z)=e^(z^2)(1+op("erf")(z)). $ <eq:exp115-passage-gradient-2>

  Here $y$ is an integration variable and $op("erf")$ is the error function,

  $ op("erf")(z)=2/sqrt(pi) integral_0^z e^(-y^2)dif y. $ <eq:exp115-erf>

  Define the dimensionless reset and threshold bounds

  $ alpha_P=(V_"reset"-mu_(V,P))/sigma_V, $ <eq:exp115-gain-bounds>

  $ beta_P=(V_"th"-mu_(V,P))/sigma_V, $ <eq:exp115-gain-bounds-2>

  $ Q_P (I)=integral_(alpha_P)^(beta_P) H(z)dif z. $ <eq:exp115-gain-bounds-3>

  Integration back from threshold gives

  $ T_P (V_"reset")=tau_(m,P)sqrt(pi)Q_P. $ <eq:exp115-reset-passage>

  Adding the fixed refractory interval
  and taking the reciprocal mean interspike interval gives the Siegert
  gain:#cite(2, 3, 4)

  $ D_P (I)=tau_("ref",P)+tau_(m,P)sqrt(pi)Q_P (I), $ <eq:exp115-siegert>

  $ Phi_P (I)=frac(1, D_P (I)). $ <eq:exp115-siegert-2>

  $D_P$ is a time in ms, and $Phi_P$ is a per-neuron rate in $"ms"^(-1)$.
  This expression is exact for the specified white-noise neuron with fixed
  input; using it inside #equation-range(<eq:exp115-model>, <eq:exp115-model-4>)
  is the additional population closure.

  === First three derivatives

  All gain derivatives below are with respect to current $I$, holding
  $sigma_V$, threshold, reset and neuron parameters fixed. Let
  $eta_P=1/(g_(L,P)sigma_V)$, in $"nA"^(-1)$, so that
  $alpha'_P=beta'_P=-eta_P$. Kernel derivatives instead use the argument $z$:

  $ H'(z)=2z H(z)+2/sqrt(pi), $ <eq:exp115-kernel-derivatives>

  Applying the endpoint rule to $Q_P$ twice gives

  $ D'_P = tau_(m,P)sqrt(pi)eta_P [H(alpha_P)-H(beta_P)], $ <eq:exp115-interval-derivatives>

  $ D''_P = tau_(m,P)sqrt(pi)eta_P^2[H'(beta_P)-H'(alpha_P)], $ <eq:exp115-interval-derivatives-2>

  Differentiating $D_P^(-1)$ then gives the required analytic gain derivatives:

  $ Phi'_P = -frac(D'_P, D_P^2), $ <eq:exp115-gain-derivatives>

  $ Phi''_P = frac(2(D'_P)^2, D_P^3)-frac(D''_P, D_P^2), $ <eq:exp115-gain-derivatives-2>

  Their units are $"ms"^(-1)"nA"^(-k)$ for derivative order $k=1,2$.
  The identities follow by differentiation of @eq:exp115-siegert-2; they are not
  fitted response curves. Since the gain increases with current,
  $Phi'_P>0$ for finite current and positive noise scale, although numerical
  underflow can hide very small slopes.

  For negative $z$, evaluate $H(z)=op("erfcx")(-z)$, where
  $op("erfcx")(z)=e^(z^2)op("erfc")(z)$ and
  $op("erfc")(z)=1-op("erf")(z)$. This avoids cancellation in
  $1+op("erf")(z)$. Strong positive tails require scaled or logarithmic
  quadrature, or increased precision; clipping an exponential changes the gain
  and its derivatives. The recurrence for $H'$ also suffers
  cancellation for sufficiently negative arguments, so their numerical
  evaluation needs the independent checks in
  #link(<app:exp115-numerics>)[Appendix D].

  == Appendix C — Equilibrium, sparse Jacobian and Hopf condition <app:exp115-hopf>

  === Equilibrium and its current derivative

  A superscript $*$ denotes an equilibrium, not necessarily a Hopf point.
  Setting the conductance derivatives to zero gives

  $ g_e^(I*)=tau_"AMPA" G_(E arrow.r I)r_E^*, $ <eq:exp115-equilibrium-g>

  $ g_i^(E*)=tau_"GABA" G_(I arrow.r E)r_I^*. $ <eq:exp115-equilibrium-g-2>

  Define positive current-per-rate couplings, in nA ms,

  $ J_(E arrow.r I)=Delta V_"exc" tau_"AMPA" G_(E arrow.r I), $ <eq:exp115-current-coupling-e>

  $ J_(I arrow.r E)=Delta V_"inh" tau_"GABA" G_(I arrow.r E). $ <eq:exp115-current-coupling-i>

  The remaining equilibrium equations are

  $ r_E^*=Phi_E (I_"ext"-J_(I arrow.r E)r_I^*), $ <eq:exp115-equilibrium-r>

  $ r_I^*=Phi_I (J_(E arrow.r I)r_E^*). $ <eq:exp115-equilibrium-r-2>

  Eliminating $r_I^*$ leaves the scalar residual

  $ R(r)=r-Phi_E (I_"ext"-J_(I arrow.r E)Phi_I (J_(E arrow.r I)r)). $ <eq:exp115-scalar-residual>

  It is strictly increasing, is negative at $r=0$, and is positive at
  $r=1/tau_("ref",E)$. Consequently this closure has a unique physical
  equilibrium at each finite drive; that equilibrium need not be stable.
  No clipping of rates is part of the smooth vector field.

  Write $phi_P^((k))=Phi_P^((k))(I_P^*)$ for gain derivative order $k$ at the
  equilibrium current. At fixed decay times and closure parameters, define

  $ D_*=1+J_(I arrow.r E)J_(E arrow.r I)phi_E^((1))phi_I^((1))>0. $ <eq:exp115-tangent-denominator>

  Implicit differentiation of
  #equation-range(<eq:exp115-equilibrium-r>, <eq:exp115-equilibrium-r-2>) yields

  $ (dif r_E^*)/(dif I_"ext") = phi_E^((1))/D_*, $ <eq:exp115-equilibrium-tangent>

  $ (dif r_I^*)/(dif I_"ext") = frac(J_(E arrow.r I)phi_I^((1))phi_E^((1)), D_*), $ <eq:exp115-equilibrium-tangent-2>

  $ (dif I_E^*)/(dif I_"ext") = 1/D_*, $ <eq:exp115-equilibrium-tangent-3>

  $ (dif I_I^*)/(dif I_"ext") = frac(J_(E arrow.r I)phi_E^((1)), D_*). $ <eq:exp115-equilibrium-tangent-4>

  === Characteristic quartic

  Introduce positive inverse times
  $a=1/tau_(r,E)$, $b=1/tau_(r,I)$, $c=1/tau_"AMPA"$ and
  $d=1/tau_"GABA"$, and the gain-dependent entries
  $u=a Delta V_"inh" phi_E^((1))$ and $v=b Delta V_"exc" phi_I^((1))$.
  Direct differentiation of
  #equation-range(<eq:exp115-model>, <eq:exp115-model-4>) gives

  $ J_"flow"=mat(
    -a,0,0,-u;
    0,-b,v,0;
    G_(E arrow.r I),0,-c,0;
    0,G_(I arrow.r E),0,-d
  ). $ <eq:exp115-jacobian>

  The eigenvalue $lambda_J$ has units $"ms"^(-1)$ and a mode evolves as
  $e^(lambda_J t)$. The determinant has a diagonal product and one feedback
  loop; its sign is positive because the loop contains one inhibitory link:

  $ p(lambda_J)=det(lambda_J bb(I)_4-J_"flow"), $ <eq:exp115-characteristic>

  $ p(lambda_J)=(lambda_J+a)(lambda_J+b)(lambda_J+c)(lambda_J+d)+K, $ <eq:exp115-characteristic-2>

  $ K=u v G_(E arrow.r I)G_(I arrow.r E). $ <eq:exp115-characteristic-3>

  Here $bb(I)_4$ is the four-dimensional identity and $K$ is the positive
  loop product, in $"ms"^(-4)$. Expanding gives

  $ p(lambda_J)=lambda_J^4+A_1 lambda_J^3+A_2 lambda_J^2+A_3 lambda_J+A_4, $ <eq:exp115-quartic>

  $ A_1=a+b+c+d, $ <eq:exp115-quartic-2>

  $ A_2=a b+a c+a d+b c+b d+c d, $ <eq:exp115-quartic-3>

  $ A_3=a b c+a b d+a c d+b c d, $ <eq:exp115-quartic-4>

  $ A_4=a b c d+K. $ <eq:exp115-quartic-5>

  Coefficient $A_j$ has units $"ms"^(-j)$. The first column of the quartic
  Routh array is
  $(1,A_1,Delta_2/A_1,cal(H)/Delta_2,A_4)^T$, where

  $ Delta_2=A_1 A_2-A_3, $ <eq:exp115-hurwitz>

  $ cal(H)=A_1 A_2 A_3-A_3^2-A_1^2 A_4. $ <eq:exp115-hurwitz-2>

  The Routh–Hurwitz criterion requires every entry of that column to be
  positive.#cite(5) Here $A_1,A_2,A_3,A_4$ and $Delta_2$ are positive, so linear
  asymptotic stability is equivalent to $cal(H)>0$.

  At a nonzero imaginary root $lambda_J=i omega_H$, where $i^2=-1$,
  equating real and imaginary parts gives

  $ omega_H^4-A_2 omega_H^2+A_4=0, $ <eq:exp115-imaginary-root>

  $ omega_H (A_3-A_1 omega_H^2)=0. $ <eq:exp115-imaginary-root-2>

  Hence the candidate Hopf condition and angular frequency are

  $ cal(H)=0, $ <eq:exp115-hopf-condition>

  $ omega_H=sqrt(A_3/A_1), $ <eq:exp115-hopf-condition-2>

  $ K_H=(A_1 A_2 A_3-A_3^2)/A_1^2-a b c d. $ <eq:exp115-hopf-condition-3>

  $K_H$ denotes the required loop product. At $K=K_H$, the quartic factors as

  $ p(lambda_J)=(lambda_J^2+omega_H^2)
    (lambda_J^2+A_1 lambda_J+A_4/omega_H^2). $ <eq:exp115-hopf-factor>

  The second factor has strictly damped roots and cannot vanish at
  $i omega_H$, so the imaginary pair is simple and isolated.
  Whether $K(I_"ext")$ reaches $K_H$ inside the specified drive interval is
  determined numerically. Conditional on a crossing, its onset frequency is

  $ f_"Hopf"=(1000/(2 pi))omega_H, $ <eq:exp115-hopf-frequency>

  in Hz when $omega_H$ is in rad/ms. Within this particular four-filter loop,
  frequency at the crossing depends only on the four time constants; gain
  shape controls whether and where the required loop product is reached.

  === Transversality along the equilibrium branch

  At fixed $tau_"GABA"$, $sigma_V$ and $kappa$, only $A_4$ depends on drive.
  Differentiating $p(lambda_J (I_"ext"),I_"ext")=0$ gives

  $ (dif lambda_J)/(dif I_"ext")=-frac(K', p'(lambda_J)), $ <eq:exp115-eigenvalue-tangent>

  where the prime on $K$ denotes
  the total drive derivative along the equilibrium branch and the prime on
  $p$ denotes differentiation in its eigenvalue argument. In particular,

  $ chi=op("Re")((dif lambda_J)/(dif I_"ext"))_"Hopf", $ <eq:exp115-transversality>

  $ chi= (2 A_3 K')/((2 A_3)^2+4 omega_H^2(A_2-2 omega_H^2)^2). $ <eq:exp115-transversality-2>

  Here $I_"ext"^*$ now denotes the candidate Hopf drive; $chi$ has units
  $"ms"^(-1)"nA"^(-1)$. Writing

  $ K_0=a b Delta V_"inh" Delta V_"exc" G_(E arrow.r I)G_(I arrow.r E), $ <eq:exp115-loop-prefactor>

  the required total derivative is

  $ K'=K_0[phi_E^((2))phi_I^((1))(dif I_E^*)/(dif I_"ext")
    +phi_E^((1))phi_I^((2))(dif I_I^*)/(dif I_"ext")]. $ <eq:exp115-loop-tangent>

  Use #equation-range(<eq:exp115-equilibrium-tangent-3>, <eq:exp115-equilibrium-tangent-4>),
  rather than holding the equilibrium fixed.
  Transversality requires $chi != 0$, equivalently $K' != 0$ here.
  Positive $chi$ means loss of stability as drive increases. An imaginary pair
  without transversality is insufficient for an ordinary Hopf bifurcation;
  numerical amplitude ramps provide the nonlinear evidence used here.

  == Appendix D — Numerical tolerances and sampled criticality <app:exp115-numerics>

  We evaluate the gain integral in double precision with exponentially scaled
  adaptive quadrature, targeting absolute error $10^(-12)$ and relative error
  $10^(-10)$. Candidate equilibria require nonnegative rates below their
  refractory ceilings and maximum rate and flow residuals below $10^(-10)$.
  Failed solves cannot form crossing brackets.

  We repeat equilibrium continuation on 401-, 801- and 1,601-point drive grids.
  Each candidate Hopf crossing must contain one simple nonzero imaginary pair,
  two damped remaining modes, a nonzero stable-to-unstable crossing derivative,
  agreement with the quartic identity, and matching crossing count and direction
  on all three grids. Accepted onset drives may differ by less than $10^(-7)$ nA
  and frequencies by less than $10^(-4)$ Hz across the grids.

  For the six inhibitory decay times at $sigma_V=4$ mV and $kappa=1$, define
  25 equally spaced drives from $I_"ext"^*-0.10$ to
  $I_"ext"^*+0.55$ nA. Starting from the lowest-drive equilibrium with a
  $10^(-3)$ $"ms"^(-1)$ excitatory-rate perturbation, integrate each drive for
  2,000 ms and carry its endpoint into the next drive. After the upward sequence,
  reverse the sequence without resetting the state. LSODA uses relative tolerance
  $10^(-7)$, absolute tolerance $10^(-10)$ and maximum step 1 ms.

  For direction $d in {"up","down"}$, measure peak-to-peak excitatory-rate
  amplitude over the final 500 ms,

  $ A_d(I)=max_(t in cal(T)_"obs") r_E(t;I,d)
      -min_(t in cal(T)_"obs") r_E(t;I,d). $ <eq:exp115-amplitude>

  The branch gap is $max_I abs(A_"up"(I)-A_"down"(I))$. Above onset, fit

  $ A_"up"^2 approx m(I_"ext"-I_"ext"^*)+c. $ <eq:exp115-amplitude-fit>

  We label the sampled response consistent with supercriticality only when the
  branch gap is below $10^(-4)$ $"ms"^(-1)$, $m>0$ and $R^2>0.9$. Otherwise it
  is subcritical or inconclusive. This is a finite-duration, finite-grid
  classification: absence of resolved hysteresis does not exclude a narrower
  bistable interval or an unstable periodic orbit. The calculation describes the
  deterministic closure, not the separate spiking network.
]

#let render-report(data-file) = [
  #let numbers = data-json(data-file("exp115/numbers.json"))
  #let onset = numbers.reference.onset
  #let criticality = numbers.reference.criticality
  #let ramps = numbers.conditions.filter(row => row.criticality != none)
  #let gaps = ramps.map(row => row.criticality.branch_gap_per_ms)
  #let r2-values = ramps.map(row => row.criticality.amplitude_squared_r2)
  #let reference-decays = numbers.conditions.filter(row =>
    row.condition.sigma_mV == 4 and row.condition.kappa == 1
  ).sorted(key: row => row.condition.tau_GABA_ms)

  #journal-abstract(body: [
    We tested whether a four-variable mean-field closure of the canonical
    conductance-based PING circuit develops an oscillatory instability and
    whether its finite-amplitude onset is numerically consistent with a
    supercritical transition.

    Equilibrium continuation resolved one Hopf onset in every tested closure
    condition. Matched time-domain ramps were consistent with supercriticality
    at all six inhibitory decay times, while onset frequency decreased with
    inhibitory decay. These results characterize the deterministic closure, not
    the separate spiking network.
  ])

  == Results

  #with-result-sections[
    #journal-result-card(
      title: "Sampled onset is supercritical",
      observation: [
        The reference equilibrium lost stability at
        #calc.round(onset.drive_nA, digits: 6) nA with onset frequency
        #calc.round(onset.frequency_Hz, digits: 2) Hz
        (#result-figure-ref(<fig:exp115-hopf>, panel: "A")). Its upward and
        downward amplitude branches nearly coincided: the maximum branch gap was
        #calc.round(criticality.branch_gap_per_ms, digits: 8) $"ms"^(-1)$ and
        the amplitude-squared fit had $R^2=#calc.round(criticality.amplitude_squared_r2, digits: 4)$
        (#result-figure-ref(<fig:exp115-hopf>, panel: "B")). All
        #numbers.summary.criticality_ramps decay-time ramps met the predefined
        criteria for consistency with supercriticality; their largest branch gap
        was #calc.round(calc.max(..gaps), digits: 8) $"ms"^(-1)$ and their
        minimum $R^2$ was #calc.round(calc.min(..r2-values), digits: 4).
        Across the reference decay series, predicted onset frequency fell from
        #calc.round(reference-decays.first().onset.frequency_Hz, digits: 2) to
        #calc.round(reference-decays.last().onset.frequency_Hz, digits: 2) Hz
        (#result-figure-ref(<fig:exp115-hopf>, panel: "C")). Finite ramps cannot
        exclude a narrower bistable interval or an unstable periodic orbit, and
        they do not establish criticality in the spiking network.
      ],
      visual: [
        #figure(
          data-image(
            data-file("exp115/hopf-compound.svg"),
            width: 100%,
            alt: "Three-panel mean-field calculation showing the leading eigenvalue crossing, matched upward and downward oscillation-amplitude ramps, and onset frequency across inhibitory decay and closure choices.",
          ),
          caption: [
            *Hopf onset, sampled criticality and closure sensitivity.*
            *A:* Largest real part of the equilibrium Jacobian eigenvalues versus
            tonic drive at $tau_"GABA"=6$ ms, $sigma_V=4$ mV and $kappa=1$;
            the red marker identifies the refined onset.
            *B:* Upward and downward peak-to-peak excitatory-rate amplitudes over
            the final 500 ms of each 2-s drive step at the same reference closure;
            the dotted line marks the refined onset.
            *C:* Eigenvalue-derived onset frequency versus inhibitory decay.
            The black curve fixes $sigma_V=4$ mV and $kappa=1$; grey curves show
            the other 19 closure choices. Each curve is deterministic, without
            replicate averaging or statistical uncertainty intervals.
          ],
          kind: image,
          supplement: [Figure],
        ) <fig:exp115-hopf>
      ],
    )
  ]
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(data-file, inputs, [], preview-figures)
}

#let meta = meta + (assets: input-assets("exp115", inputs))
#let body = journal-article("exp115", inputs, [
  #report-body
  #methods-section
  #parameter-section
  #mathematical-appendices
  #journal-references((
    (text: [C. Börgers and N. Kopell. “Synchronization in Networks of
      Excitatory and Inhibitory Neurons with Sparse, Random Connectivity.”
      _Neural Computation_ *15*(3), 509–538 (2003).],
      doi: "10.1162/089976603321192059"),
    (text: [A. J. F. Siegert. “On the First Passage Time Probability Problem.”
      _Physical Review_ *81*, 617–623 (1951).],
      doi: "10.1103/PhysRev.81.617"),
    (text: [L. M. Ricciardi and L. Sacerdote. “The Ornstein–Uhlenbeck Process
      as a Model for Neuronal Activity. I. Mean and Variance of the Firing Time.”
      _Biological Cybernetics_ *35*, 1–9 (1979).],
      doi: "10.1007/BF01845839"),
    (text: [K. Kreutz-Delgado. “Mean Time-to-Fire for the Noisy LIF Neuron —
      A Detailed Derivation of the Siegert Formula.” _arXiv_ preprint
      1501.04032 (2015), sections 2.6 and 3.],
      doi: "10.48550/arXiv.1501.04032"),
    (text: [A. Hurwitz. “Ueber die Bedingungen, unter welchen eine Gleichung
      nur Wurzeln mit negativen reellen Theilen besitzt.” _Mathematische
      Annalen_ *46*, 273–284 (1895).],
      doi: "10.1007/BF01446812"),
  ))
])
