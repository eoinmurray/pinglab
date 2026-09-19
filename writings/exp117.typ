#import "templates/article-layout.typ": journal-article
#import "templates/abstract.typ": journal-abstract
#import "templates/methods.typ": journal-methods, method-card
#import "templates/parameters-table.typ": parameters-table
#import "templates/references.typ": journal-references
#import "templates/result-card.typ": journal-result-card, result-figure-ref, with-result-sections
#import "templates/dataset.typ": data-file, input-assets, inputs-ready, pending-report
#import "/.demolab/lib.typ": cite, data-image, data-json
#let data-file = data-file.with(article: "exp117")

#let meta = (
  tags: ("data", "v36.0.0"),
  title: "Mean-Field Analysis of PING Bifurcations",
  created_at: "2026-09-19T00:00:00Z",
  updated_at: "2026-09-19",
  description: "An independent four-variable mean-field analysis identifies a Hopf bifurcation, sampled dynamics consistent with a supercritical transition, and decreasing onset frequency with slower inhibition.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp117",)
#let preview-figures = (
  (path: "exp117/bifurcation_compound.svg", label: "Hopf bifurcation diagnostics"),
)

#let meta-objectives(body) = context {
  if target() == "html" {
    html.elem("style",
      ".exp117-meta, .exp117-meta * { color:#b42318 !important; }",
    )
    html.elem("section", attrs: (class: "exp117-meta"), body)
  } else {
    [
      #set text(fill: rgb("#b42318"))
      #show heading: set text(fill: rgb("#b42318"))
      #body
    ]
  }
}

#let render-report(data-file) = [
  #let run = data-json(data-file("exp117/numbers.json"))
  #let result = run.result
  #let istar = calc.round(result.I_ext_star_nA, digits: 6)
  #let fstar = calc.round(result.f_Hopf_Hz, digits: 2)
  #let slope = calc.round(result.crossing_slope_per_ms_per_nA, digits: 3)
  #let remaining = calc.round(result.remaining_max_real_per_ms, digits: 3)
  #let criticality = result.criticality
  #let branch-gap = calc.round(1000 * criticality.branch_gap_per_ms, digits: 5)
  #let amplitude-r2 = calc.round(criticality.amplitude_squared_r2, digits: 4)
  #let tau-sweep = result.frequency_vs_tau_GABA
  #let fast-frequency = calc.round(tau-sweep.first().f_Hopf_Hz, digits: 2)
  #let slow-frequency = calc.round(tau-sweep.last().f_Hopf_Hz, digits: 2)

  #journal-abstract(body: [
    We derived an independent four-variable population-rate closure of the
    conductance-based excitatory–inhibitory circuit. Analytical-Jacobian
    continuation identified a Hopf bifurcation at #istar nA with onset frequency
    #fstar Hz. Ascending and descending near-onset integrations produced
    coincident sampled branches and continuous amplitude growth consistent with
    a supercritical transition. This numerical criterion does not exclude a
    narrower bistable interval or unstable cycle. Hopf frequency decreased as
    inhibitory decay lengthened.
  ])

  #meta-objectives([
    == Meta

    *Objectives*

    #enum(
      [✓ Derive a mean-field model of our system. *(Completed.)*],
      [✓ Find out whether it has a Hopf bifurcation and at what frequency it occurs. *(Completed.)*],
      [✓ Test the criticality numerically. *(Completed: consistent with a supercritical transition under the sampled criteria.)*],
      [✓ Plot frequency against $tau_"GABA"$. *(Completed.)*],
    )
  ])

  == Results

  #with-result-sections[
    #journal-result-card(
      title: "A reversible Hopf onset slows",
      observation: [
        At $I_"ext"=#istar$ nA, a complex-conjugate eigenvalue pair crossed the
        imaginary axis at $f_"Hopf"=#fstar$ Hz
        (#result-figure-ref(<fig:exp117-bifurcation>, panel: "A")). The sampled
        amplitude branches differed by at most #branch-gap Hz and had
        $R_"fit"^2=#amplitude-r2$, consistent with a supercritical transition
        (#result-figure-ref(<fig:exp117-bifurcation>, panel: "B")). Across the
        inhibitory-decay sweep, $f_"Hopf"$ decreased from #fast-frequency to
        #slow-frequency Hz (#result-figure-ref(<fig:exp117-bifurcation>, panel:
        "C")). Finite spacing cannot exclude narrower bistability or an
        unstable periodic orbit.
      ],
      visual: [
        #figure(
          data-image(
            data-file("exp117/bifurcation_compound.svg"),
            width: 100%,
            alt: "Eigenvalue crossing, reversible oscillation-amplitude onset, and Hopf frequency across inhibitory decay constants.",
          ),
          caption: [
            *Mean-field Hopf bifurcation.* (A) All four $J_"flow"$ eigenvalues,
            coloured by external drive; open markers identify the refined Hopf
            pair. (B) Peak-to-peak excitatory-rate amplitude over the final 500
            ms of each ascending or descending drive condition, relative to the
            refined Hopf drive. (C) Refined Hopf frequency at
            $tau_"GABA"=4.5, 6, 9, 12, 18$ and $27$ ms, with all other model
            parameters fixed. These deterministic calculations have no
            statistical uncertainty intervals.
          ],
          kind: image,
          supplement: [Figure],
        ) <fig:exp117-bifurcation>
      ],
    )
  ]

  #journal-methods(body: (
    method-card([Derivation of the model], [
      See #link(<exp117-appendix-a>)[Appendix A].
    ]),
    method-card([Hopf bifurcation and onset frequency], [
      At the reference inhibitory decay time of 6 ms, we evaluated the
      equilibrium over 401 equally spaced external drives from 0 to 4 nA.

      *2.1 — Solve for the equilibrium.* We analytically eliminated the two
      conductances and the inhibitory rate from
      @eq:exp117-equilibrium-rates. For each drive, the remaining excitatory
      rate solved the scalar residual

      $ H(r_E)=r_E-Phi_E lr(I_"ext"-Delta V_"inh" tau_"GABA"
        G_(I arrow.r E) Phi_I lr(Delta V_"exc" tau_"AMPA"
        G_(E arrow.r I)r_E))=0. $ <eq:exp117-equilibrium-residual>

      Brent's bracketed method searched $0 <= r_E <= 1/tau_("ref",E)$ with
      absolute tolerance $10^(-13)$ $"ms"^(-1)$ and relative tolerance
      $10^(-12)$. We recovered $r_I$, $g_e^I$ and $g_i^E$ by substitution and
      required an absolute residual below $10^(-11)$ $"ms"^(-1)$.

      *2.2 — Calculate local stability.* At every equilibrium, we evaluated the
      analytical $J_"flow"$ in @eq:exp117-flow-jacobian and numerically
      calculated its four eigenvalues. Eigenvalues with positive imaginary
      part greater than $10^(-8)$ $"ms"^(-1)$ were eligible for the leading
      complex pair. We plotted all four eigenvalues in the complex plane and
      coloured them by external drive; negative and positive real parts denoted
      local decay and growth, respectively
      (#result-figure-ref(<fig:exp117-bifurcation>, panel: "A")).

      *2.3 — Identify and verify the crossing.* The first change of the leading
      complex pair from negative to nonnegative real part supplied a coarse
      drive bracket. Brent's method refined the zero to absolute drive
      tolerance $10^(-12)$ nA and relative tolerance $10^(-12)$, recomputing
      the equilibrium and analytical Jacobian at every trial drive. We
      confirmed nonzero imaginary part, negative real parts for the remaining
      modes, and positive transversality by a centred difference with
      $10^(-5)$ nA half-step. The recorded crossing slope was #slope
      $"ms"^(-1) "nA"^(-1)$ and the largest remaining real part was
      #remaining $"ms"^(-1)$.

      *2.4 — Calculate the onset frequency.* At the refined crossing,
      $omega_"Hopf"$ was the magnitude of the critical pair's imaginary part
      in radians per millisecond. We converted it to hertz using
      $f_"Hopf"=1000 omega_"Hopf"/(2 pi)$
      (#result-figure-ref(<fig:exp117-bifurcation>, panel: "A")).

    ]),
    method-card([Numerical criticality test], [
      We integrated 25 equally spaced drives from 0.1 nA below to 0.55 nA
      above the refined Hopf point, first in ascending and then in descending
      order. The initial state was the lowest-drive equilibrium with an added
      $10^(-3)$ $"ms"^(-1)$ excitatory-rate perturbation; each subsequent
      condition began from the preceding endpoint. LSODA integrated each
      condition for 2,000 ms with relative tolerance $10^(-7)$, absolute
      tolerance $10^(-10)$ and maximum step 1 ms.

      For each drive, peak-to-peak excitatory-rate amplitude $A_"pp"$ was
      measured over the final 500 ms. We classified the sampled dynamics as
      consistent with a supercritical transition when the maximum difference
      between ascending and descending amplitudes was below $10^(-4)$
      $"ms"^(-1)$, and ordinary least-squares regression of $A_"pp"^2$ on
      $I_"ext"-I^*$ above onset had positive slope and
      $R_"fit"^2>0.9$. This finite numerical criterion tests reversible,
      continuous onset; it does not detect an unstable cycle or prove the
      absence of narrower bistability.#cite(2)
    ]),
    method-card([Inhibitory-decay sweep], [
      We repeated the equilibrium continuation and refined complex-pair
      crossing at $tau_"GABA"=4.5, 6, 9, 12, 18$ and $27$ ms while holding all
      other model parameters fixed. At every decay constant, we applied the
      same nonzero-frequency, remaining-mode stability and transversality
      checks used at the 6-ms reference condition, then converted the critical
      angular frequency to $f_"Hopf"$ in hertz.
    ]),
  ))

  #parameters-table(
    ([Parameter], [Symbol], [Value], [Origin]),
    (
      ([Population size, E / I], [$N_E$ / $N_I$], [1,024 / 256], [Imported repository parameter]),
      ([Membrane capacitance, E / I], [$C_(m,E)$ / $C_(m,I)$], [1 / 0.5 nF], [Imported repository parameter]),
      ([Leak conductance, E / I], [$g_(L,E)$ / $g_(L,I)$], [0.05 / 0.10 µS], [Imported repository parameter]),
      ([Membrane time constant, E / I], [$tau_(m,E)$ / $tau_(m,I)$], [20 / 5 ms], [Derived from imported $C_m/g_L$]),
      ([Leak and reset potential], [$E_L$ / $V_"reset"$], [−65 / −65 mV], [Imported repository parameter]),
      ([Spike threshold], [$V_"th"$], [−50 mV], [Imported repository parameter]),
      ([Excitatory / inhibitory reversal], [$E_e$ / $E_i$], [0 / −80 mV], [Imported repository parameter]),
      ([Refractory period, E / I], [$tau_("ref",E)$ / $tau_("ref",I)$], [1.2 / 0.6 ms], [Imported repository parameter]),
      ([AMPA decay], [$tau_"AMPA"$], [2 ms], [Imported repository parameter]),
      ([Reference GABA decay], [$tau_"GABA"$], [6 ms], [Imported repository parameter]),
      ([GABA-decay sweep], [$tau_"GABA"$], [4.5, 6, 9, 12, 18, 27 ms], [Mean-field experimental choice]),
      ([Summed coupling, E→I / I→E], [$G_(E arrow.r I)$ / $G_(I arrow.r E)$], [1 / 2 µS], [Imported repository parameter]),
      ([Fixed driving force, excitatory / inhibitory], [$Delta V_"exc"$ / $Delta V_"inh"$], [65 / 15 mV], [Mean-field choice derived at $E_L$]),
      ([Rate relaxation, E / I], [$tau_(r,E)$ / $tau_(r,I)$], [20 / 5 ms], [Mean-field assumption: set equal to $tau_m$]),
      ([Effective voltage-noise scale], [$sigma_V$], [4 mV], [Mean-field assumption]),
      ([Tonic excitatory drive], [$I_"ext"$], [0–4 nA], [Mean-field control choice]),
      ([Criticality ramp], [$I_"ext"-I^*$], [−0.1–0.55 nA; 25 points], [Mean-field experimental choice]),
    ),
    columns: (1.3fr, 1fr, 1fr, 2.2fr),
    table-label: <tab:exp117-parameters>,
    caption: [Parameters of the four-variable model. “Imported” values
      are literal copies of established conductance-based system parameters;
      exp117 does not read another experiment at runtime. Derived entries follow
      algebraically from imported values. Mean-field assumptions and experimental
      choices define the closure or bifurcation analysis and are not
      measurements of the spiking network.],
  )

  == Appendix A — Derivation of the four-variable model <exp117-appendix-a>

  === A1 — Conductance-based neuron system

  The system contains excitatory and inhibitory leaky integrate-and-fire
  populations. For neuron $j$ in population $P in {E,I}$, membrane voltage
  $V_(m,P,j)$ obeys

  $ C_(m,P) dot(V)_(m,P,j)
      =-g_(L,P)(V_(m,P,j)-E_L)
       -g_(e,P,j)(V_(m,P,j)-E_e)
       -g_(i,P,j)(V_(m,P,j)-E_i). $ <eq:exp117-neuron>

  Here $C_(m,P)$ is capacitance, $g_(L,P)$ is leak conductance, $E_L$ is the
  leak potential, $E_e$ and $E_i$ are excitatory and inhibitory reversal
  potentials, and $g_e$ and $g_i$ are total AMPA and GABA conductances. A
  neuron emits a spike when its candidate voltage crosses $V_"th"$, resets to
  $V_"reset"$, and remains refractory for $tau_("ref",P)$.

  The recurrent architecture contains E→I excitation and I→E inhibition, with
  no E→E or I→I pathway. Excitatory neurons additionally receive an external
  AMPA conductance driven by presynaptic spikes.

  === A2 — Replace spiking input with tonic current

  Let external presynaptic neuron $j$ emit spike train $s_j^X(t)$ and produce
  conductance increment $W_(j k)^(X arrow.r E)$ in excitatory neuron $k$. Its
  external conductance obeys

  $ dot(g)_(X,k)^E
      =-frac(g_(X,k)^E,tau_"AMPA")
       +sum_j W_(j k)^(X arrow.r E)s_j^X(t). $ <eq:exp117-external-filter>

  The corresponding fluctuating current is
  $-g_(X,k)^E(V_(m,E,k)-E_e)$. Under stationary external spiking, we replace
  this term by a constant tonic current $I_"ext"$. The population-specific
  membrane equations used for the closure are therefore

  $ C_(m,E) dot(V)_(m,E)
      &=-g_(L,E)(V_(m,E)-E_L)-g_i^E(V_(m,E)-E_i)+I_"ext", \
    C_(m,I) dot(V)_(m,I)
      &=-g_(L,I)(V_(m,I)-E_L)-g_e^I(V_(m,I)-E_e). $ <eq:exp117-specialized-neurons>

  The tonic-current substitution defines a control parameter; it does not map
  a particular current to an empirical input rate.

  === A3 — Population-average recurrent conductances

  Let the continuous spike train of presynaptic neuron $j$ in population $P$ be

  $ s_j^P(t)=sum_n delta(t-t_(j,n)^P), $ <eq:exp117-spike-train>

  where $t_(j,n)^P$ is spike time $n$ and $delta$ is the Dirac impulse. A
  recurrent conductance onto neuron $k$ in population $Q$ satisfies

  $ dot(g)_(P arrow.r Q,k)
      =-frac(g_(P arrow.r Q,k),tau_(P arrow.r Q))
       +sum_(j=1)^(N_P) W_(j k)^(P arrow.r Q)s_j^P(t). $ <eq:exp117-event-filter>

  Define the smooth per-neuron population rate and mean event weight by

  $ r_P(t)=frac(1,N_P)sum_(j=1)^(N_P) bb(E)[s_j^P(t)], quad
    G_(P arrow.r Q)=N_P macron(w)_(P arrow.r Q). $ <eq:exp117-population-average>

  Here $bb(E)$ denotes the mean-field expectation that replaces each finite
  impulse train by its smooth rate. Replacing individual event weights by their
  population mean then gives

  $ dot(g)_e^I
      &=-frac(g_e^I,tau_"AMPA")+G_(E arrow.r I)r_E, \
    dot(g)_i^E
      &=-frac(g_i^E,tau_"GABA")+G_(I arrow.r E)r_I. $ <eq:exp117-mean-conductances>

  This homogeneous approximation removes weight heterogeneity, finite-size
  fluctuations and spike correlations.

  === A4 — Fixed driving-force closure

  A synaptic conductance contributes current $-g(V_m-E_"rev")$. We evaluate
  its driving force at $E_L=-65$ mV. With $E_e=0$ mV and $E_i=-80$ mV,

  $ Delta V_"exc"=E_e-E_L=65 "mV", quad
    Delta V_"inh"=E_L-E_i=15 "mV". $ <eq:exp117-driving-forces>

  The effective currents seen by the excitatory and inhibitory populations are

  $ I_E=I_"ext"-Delta V_"inh"g_i^E, quad
    I_I=Delta V_"exc"g_e^I. $ <eq:exp117-effective-currents>

  Freezing these driving forces removes the multiplicative $g V_m$ term. It
  therefore omits conductance-dependent shunting and changes in the effective
  membrane time constant; it is a mean-field assumption rather than an exact
  reduction.

  === A5 — Noisy-LIF population gain

  For population $P$, let the stationary noisy-LIF gain be#cite(1)

  $ Phi_P (I)
      &=lr(tau_("ref",P)+tau_(m,P)sqrt(pi)
        integral_(a_P(I))^(b_P(I)) e^(u^2)(1+op("erf")(u)) dif u)^(-1), \
    a_P(I)&=frac(V_"reset"-mu_(V,P)(I),sigma_V), \
    b_P(I)&=frac(V_"th"-mu_(V,P)(I),sigma_V), \
    mu_(V,P)(I)&=E_L+frac(I,g_(L,P)). $ <eq:exp117-gain>

  Here $I$ is mean current in nA, $mu_(V,P)$ is its equivalent mean voltage,
  $sigma_V$ is the chosen effective voltage-noise scale, and $Phi_P$ is a
  per-neuron rate. The fixed $sigma_V$ is not calculated self-consistently from
  network fluctuations.

  Rather than imposing the stationary gain instantaneously, introduce rate
  relaxation with time constant $tau_(r,P)$:

  $ tau_(r,P) dot(r)_P=-r_P+Phi_P (I_P). $ <eq:exp117-rate-relaxation>

  This relaxation law supplies dynamical population rates and is an explicit
  closure assumption.

  === A6 — Four-variable mean-field model

  Choose the state
  $x=(r_E,r_I,g_e^I,g_i^E)$. Substituting the effective currents from
  @eq:exp117-effective-currents into @eq:exp117-rate-relaxation and adjoining
  the two conductance filters @eq:exp117-mean-conductances gives

  $ tau_(r,E) dot(r)_E
      &=-r_E+Phi_E (I_"ext"-Delta V_"inh"g_i^E), \
    tau_(r,I) dot(r)_I
      &=-r_I+Phi_I (Delta V_"exc"g_e^I), \
    dot(g)_e^I
      &=-frac(g_e^I,tau_"AMPA")+G_(E arrow.r I)r_E, \
    dot(g)_i^E
      &=-frac(g_i^E,tau_"GABA")+G_(I arrow.r E)r_I. $ <eq:exp117-four-variable-model>

  These four equations define the mean-field model. They combine
  established single-neuron and exponential-synapse relations with four stated
  closure operations: homogeneous population averaging, tonic external drive,
  fixed synaptic driving forces, and phenomenological rate relaxation with an
  effective noise scale.

  === A7 — Analytical flow Jacobian

  Write the right-hand side of @eq:exp117-four-variable-model as
  $dot(x)=F(x;I_"ext")$. In the state ordering
  $x=(r_E,r_I,g_e^I,g_i^E)$, its continuous-time flow Jacobian is

  $ J_"flow"(x;I_"ext")
      =frac(partial F,partial x)
      =mat(
        -1/tau_(r,E), 0, 0,
          -Delta V_"inh" Phi'_E(I_E)/tau_(r,E);
        0, -1/tau_(r,I),
          Delta V_"exc" Phi'_I(I_I)/tau_(r,I), 0;
        G_(E arrow.r I), 0, -1/tau_"AMPA", 0;
        0, G_(I arrow.r E), 0, -1/tau_"GABA"
      ), $ <eq:exp117-flow-jacobian>

  where $I_E=I_"ext"-Delta V_"inh"g_i^E$ and
  $I_I=Delta V_"exc"g_e^I$. The gain derivatives can also be written
  analytically. Defining

  $ h(u)=e^(u^2)(1+op("erf")(u)), $
  <eq:exp117-gain-integrand>

  differentiation of @eq:exp117-gain by the Leibniz rule gives

  $ Phi'_P(I)
      =frac(tau_(m,P)sqrt(pi),g_(L,P)sigma_V)
       Phi_P(I)^2 lr(h(b_P(I))-h(a_P(I))). $
      <eq:exp117-gain-derivative>

  For an equilibrium $x^*$, local stability is determined by the eigenvalues
  $lambda_J$ of $J_"flow"(x^*;I_"ext")$. A Hopf candidate requires a
  complex-conjugate pair to cross the imaginary axis at nonzero angular
  frequency while the other eigenvalues remain in the left half-plane.

  === A8 — Equilibrium equations

  An equilibrium $x^*=(r_E^*,r_I^*,(g_e^I)^*,(g_i^E)^*)$ is a state that does
  not change with time. Setting all four time derivatives in
  @eq:exp117-four-variable-model to zero gives the four equilibrium conditions

  $ 0
      &=-r_E^*+Phi_E (I_"ext"-Delta V_"inh"(g_i^E)^*), \
    0
      &=-r_I^*+Phi_I (Delta V_"exc"(g_e^I)^*), \
    0
      &=-frac((g_e^I)^*,tau_"AMPA")+G_(E arrow.r I)r_E^*, \
    0
      &=-frac((g_i^E)^*,tau_"GABA")+G_(I arrow.r E)r_I^*. $
      <eq:exp117-equilibrium-four>

  The last two equations can be solved directly for the equilibrium
  conductances,

  $ (g_e^I)^*
      &=tau_"AMPA" G_(E arrow.r I)r_E^*, \
    (g_i^E)^*
      &=tau_"GABA" G_(I arrow.r E)r_I^*. $
      <eq:exp117-equilibrium-conductances>

  Substituting these expressions into the first two conditions leaves two
  coupled self-consistency equations for the equilibrium firing rates,

  $ r_E^*
      &=Phi_E (I_"ext"-Delta V_"inh" tau_"GABA"
        G_(I arrow.r E)r_I^*), \
    r_I^*
      &=Phi_I (Delta V_"exc" tau_"AMPA"
        G_(E arrow.r I)r_E^*). $
      <eq:exp117-equilibrium-rates>

  We reduced these two nonlinear equations to the scalar residual in
  @eq:exp117-equilibrium-residual and solved it for each tested value of
  $I_"ext"$. Objective 2 then evaluated the analytical Jacobian and its
  eigenvalues at each recovered equilibrium. The separate nonlinear ramp test
  then evaluated whether the sampled oscillation onset was reversible and
  continuous. Repeating the equilibrium and eigenvalue calculation across the
  prescribed inhibitory-decay constants supplied the frequency comparison.

  #journal-references((
    (text: [K. Kreutz-Delgado. “Mean Time-to-Fire for the Noisy LIF Neuron:
      A Detailed Derivation of the Siegert Formula.” _arXiv_ (2015).],
      doi: "10.48550/arXiv.1501.04032"),
    (text: [Y. A. Kuznetsov. _Elements of Applied Bifurcation Theory_, third
      edition. Springer (2004), Chapters 3 and 5.],
      doi: "10.1007/978-1-4757-3978-7"),
  ))
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file,
    inputs,
    [Does the derived four-variable model undergo a Hopf bifurcation, is its sampled onset consistent with supercriticality, and how does onset frequency vary with inhibitory decay?],
    preview-figures,
    json-inputs: ("exp117",),
  )
}

#let meta = meta + (assets: input-assets("exp117", inputs))
#let body = journal-article("exp117", inputs, report-body)
