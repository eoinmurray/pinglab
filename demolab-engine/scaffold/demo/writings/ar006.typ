#import "/demolab-engine/build/lib.typ": cite, reference-list

#let meta = (
  title: "The Hodgkin–Huxley membrane, in equations",
  date: "2026-07-03",
  description: "A dense reference sheet for the spiking membrane (current balance, gating kinetics, rate functions, equilibrium potentials, the Jacobian, synaptic drive), almost entirely equations, to exercise the typesetting.",
  collection: "neuron-models",
)

#let gna = $macron(g)_"Na"$
#let gk = $macron(g)_"K"$

#let body = [
  Almost all equations, but not quite: a few lines of prose between the blocks so the
  eye has somewhere to rest. The subject is the four-dimensional system for a space-clamped
  patch of excitable membrane, the model Hodgkin and Huxley fit to the squid giant axon in
  1952#cite(1), with state $bold(x) = (V, m, h, n)^top$: one voltage and three gating variables.
  Everything below is that one system, taken apart one relation at a time.

  == Current balance

  The membrane is a capacitor in parallel with three voltage-gated conductances#cite(1, 10). Charge
  conservation across it gives the equation that ties the whole model together: the rate
  of change of voltage is set by the sum of ionic currents plus whatever is injected:

  $ C_m dot(V) = -underbrace(gna m^3 h (V - E_"Na"), I_"Na") - underbrace(gk n^4 (V - E_"K"), I_"K") - underbrace(g_"L" (V - E_"L"), I_"L") + I_"ext" (t) $

  #text(size: 9pt)[$C_m$, membrane capacitance; $macron(g)_bullet$, peak conductances; $E_bullet$, reversal potentials; $m, h, n in [0,1]$, gating variables; $I_"ext"$, injected current.]

  Each conductance is the peak value scaled by gates raised to a power: the exponents
  ($m^3$, $n^4$) are the number of independent subunits that must all be open at once.
  The gates are what make the patch excitable; the next few blocks say how they move.

  == Gating kinetics

  Each gate relaxes toward a voltage-dependent target, written two equivalent ways,
  as forward/backward rates, or as a target value $x_infinity$ approached with time
  constant $tau_x$:

  $ dot(x) = alpha_x (V) (1 - x) - beta_x (V) x = (x_infinity (V) - x) / tau_x (V), quad
    x_infinity = alpha_x / (alpha_x + beta_x), quad tau_x = 1 / (alpha_x + beta_x), quad x in {m, h, n}. $

  #text(size: 9pt)[$alpha_x (V)$, $beta_x (V)$, voltage-dependent opening/closing rate constants; $x_infinity$, steady-state (in)activation; $tau_x$, its relaxation time constant; $x in {m, h, n}$, the three gating variables.]

  == Rate functions ($V$ in $upright("mV")$, rates in $upright("ms")^(-1)$)

  The rates themselves are the empirical heart of the model: six functions fit to voltage-
  clamp data, no first-principles derivation behind them#cite(1). They are what turns the abstract
  kinetics above into concrete dynamics:

  $
    alpha_m (V) &= (0.1 (V + 40)) / (1 - exp(-(V + 40) \/ 10)), quad & beta_m (V) &= 4 exp(-(V + 65) \/ 18), \
    alpha_h (V) &= 0.07 exp(-(V + 65) \/ 20),                        quad & beta_h (V) &= 1 / (1 + exp(-(V + 35) \/ 10)), \
    alpha_n (V) &= (0.01 (V + 55)) / (1 - exp(-(V + 55) \/ 10)),     quad & beta_n (V) &= 0.125 exp(-(V + 65) \/ 80).
  $

  The $alpha_m$ and $alpha_n$ expressions look singular: numerator and denominator both
  vanish at one voltage, so a naive evaluation returns $0\/0$. The function is smooth there
  anyway: the singularity is removable, and the value is just the limit#cite(9):

  $ alpha_m (V) = cases(
      (0.1 (V + 40)) / (1 - exp(-(V + 40) \/ 10)) & "if " V != -40,
      1 & "if " V = -40,
    ) quad "since" quad lim_(V -> -40) alpha_m (V) = 1. $

  == Steady-state gating as a Boltzmann sigmoid

  Plotted against voltage, each $x_infinity$ is an S-curve, and it is often more convenient
  to fit it directly as a Boltzmann sigmoid with a half-activation voltage $V_(1\/2)$ and a
  slope factor $k$ than to go through the rate functions#cite(7, 8). Its derivative has the tidy
  logistic form that makes the slope at $V_(1\/2)$ exactly $1\/(4k)$:

  $ x_infinity (V) = 1 / (1 + exp(- (V - V_(1\/2)) \/ k)), wide
    (dif x_infinity) / (dif V) = (x_infinity (1 - x_infinity)) / k. $

  #text(size: 9pt)[$V_(1\/2)$, half-activation voltage; $k$, slope factor.]

  == Equilibrium potentials

  The reversal potentials $E_bullet$ that appear in the current balance are not free
  parameters: they are set by the ionic concentrations either side of the membrane. Nernst
  gives the single-ion equilibrium; Goldman–Hodgkin–Katz generalises to the current carried
  by a partially permeant ion and to the resting potential of a membrane permeable to
  several ions at once#cite(2, 3):

  $ E_S = (R T) / (z_S F) ln ([S]_"out") / ([S]_"in"), wide
    I_S = P_S z_S^2 (V F^2) / (R T) dot ([S]_"in" - [S]_"out" e^(-z_S V F \/ R T)) / (1 - e^(-z_S V F \/ R T)), $

  $ V_m = (R T) / F ln (P_"K" ["K"]_"out" + P_"Na" ["Na"]_"out" + P_"Cl" ["Cl"]_"in") / (P_"K" ["K"]_"in" + P_"Na" ["Na"]_"in" + P_"Cl" ["Cl"]_"out"). $

  #text(size: 9pt)[$E_S$, Nernst potential of ion $S$; $V_m$, resting potential of the mixed-permeability membrane; $I_S$, GHK current carried by $S$; $R$, gas constant; $T$, absolute temperature; $F$, Faraday constant; $z_S$, valence of $S$; $P_bullet$, membrane permeabilities; $[S]_"in"$, $[S]_"out"$, intra- and extracellular concentrations.]

  == Linear stability

  With the vector field fully specified, the question of whether the rest state is quiet or
  poised to spike becomes a matter of eigenvalues#cite(4, 5, 6). Write the system as
  $dot(bold(x)) = bold(F)(bold(x))$ and linearise about a fixed point
  $bold(x)^*$ where $bold(F)(bold(x)^*) = bold(0)$. The Jacobian $bold(J) = partial bold(F) \/ partial bold(x)$ is

  $ bold(J) = mat(delim: "[",
      partial_V dot(V), partial_m dot(V), partial_h dot(V), partial_n dot(V);
      partial_V dot(m), -1\/tau_m, 0, 0;
      partial_V dot(h), 0, -1\/tau_h, 0;
      partial_V dot(n), 0, 0, -1\/tau_n;
    ), wide
    partial_V dot(V) = - (g_"L" + gna m^3 h + gk n^4) / C_m, $

  and the rest state is stable iff every eigenvalue has negative real part:
  $det(bold(J) - lambda bold(I)) = 0 ==> "Re"(lambda_i) < 0$ for all $i$.

  #text(size: 9pt)[$bold(F)$, the model's vector field; $bold(x)^*$, a fixed point; $bold(J)$, the Jacobian there; $lambda_i$, its eigenvalues; $partial_a dot(b)$, the partial of $dot(b)$ with respect to $a$.]

  == Synaptic drive and subthreshold response

  So far $I_"ext"$ has been an anonymous forcing term. In a network it is synaptic: presynaptic
  spikes open conductances that pull the membrane toward a synaptic reversal potential#cite(7). A spike
  train $\{ t_k \}$ arriving through a synapse of reversal $E_"syn"$ adds

  $ I_"syn" (t) = g_"syn" (t) (V - E_"syn"), wide
    g_"syn" (t) = macron(g) sum_(k : thin t_k <= t) (t - t_k) / tau_s thin e^(1 - (t - t_k) \/ tau_s), $

  where each presynaptic spike contributes an alpha-function conductance transient of peak
  $macron(g)$ (the synaptic weight) that rises and decays on the synaptic time constant
  $tau_s$. Below threshold, where the membrane is a leaky integrator
  $tau dot(V) = -(V - V_"rest") + R I$ with membrane time constant $tau$ and input resistance
  $R$, the voltage is the convolution

  $ V(t) = V_"rest" + R / tau integral_(-infinity)^t e^(-(t - s) \/ tau) I(s) dif s,
    wide hat(Z)(omega) = R / (1 + upright(i) omega tau), $

  with $hat(Z)(omega)$ the input impedance, a first-order low-pass whose cutoff sits at
  $omega_c = 1 \/ tau$.

  #text(size: 9pt)[$g_"syn"$, synaptic conductance; $E_"syn"$, synaptic reversal potential; ${t_k}$, presynaptic spike times; $macron(g)$, peak conductance (synaptic weight); $tau_s$, synaptic time constant; $tau$, $R$, membrane time constant and input resistance; $hat(Z)(omega)$, input impedance; $omega_c$, low-pass cutoff.]

  #reference-list((
    (text: [Hodgkin AL, Huxley AF (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. _J Physiol_ 117:500–544.], doi: "10.1113/jphysiol.1952.sp004764"),
    (text: [Goldman DE (1943). Potential, impedance, and rectification in membranes. _J Gen Physiol_ 27:37–60.], doi: "10.1085/jgp.27.1.37"),
    (text: [Hodgkin AL, Katz B (1949). The effect of sodium ions on the electrical activity of the giant axon of the squid. _J Physiol_ 108:37–77.], doi: "10.1113/jphysiol.1949.sp004310"),
    (text: [FitzHugh R (1961). Impulses and physiological states in theoretical models of nerve membrane. _Biophys J_ 1:445–466.], doi: "10.1016/S0006-3495(61)86902-6"),
    (text: [Nagumo J, Arimoto S, Yoshizawa S (1962). An active pulse transmission line simulating nerve axon. _Proc IRE_ 50:2061–2070.], doi: "10.1109/JRPROC.1962.288235"),
    (text: [Ermentrout GB, Terman DH (2010). _Mathematical Foundations of Neuroscience._ Springer.], doi: "10.1007/978-0-387-87708-2"),
    (text: [Gerstner W, Kistler WM, Naud R, Paninski L (2014). _Neuronal Dynamics._ Cambridge University Press.], doi: "10.1017/CBO9781107447615"),
    (text: [Izhikevich EM (2007). _Dynamical Systems in Neuroscience._ MIT Press.], doi: "10.7551/mitpress/2526.001.0001"),
    (text: [Sterratt D, Graham B, Gillies A, Willshaw D (2011). _Principles of Computational Modelling in Neuroscience._ Cambridge University Press.], doi: "10.1017/CBO9780511975899"),
    (text: [Cole KS, Curtis HJ (1939). Electric impedance of the squid giant axon during activity. _J Gen Physiol_ 22:649–670.], doi: "10.1085/jgp.22.5.649"),
  ))
]
