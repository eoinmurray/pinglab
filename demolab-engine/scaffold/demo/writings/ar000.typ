#let meta = (
  title: "From a capacitor to the LIF neuron",
  date: "2026-05-30",
  description: "Deriving the leaky integrate-and-fire equation from the RC circuit that a patch of membrane actually is.",
  collection: "neuron-models",
  status: "revising",
)

#let body = [
  A neuron's membrane is a thin lipid bilayer separating two salty solutions. Charge
  piles up on either side, ions leak through embedded channels, and currents are
  injected by synapses or an experimenter's electrode. Electrically, this is just an
  RC circuit. The leaky integrate-and-fire (LIF) neuron is what you get when you write
  Kirchhoff's current law for that circuit and bolt a threshold on top.

  == The membrane is a capacitor

  Two conductors separated by an insulator form a capacitor. The bilayer is the
  insulator; the intracellular and extracellular fluids are the conductors. The
  defining relation for a capacitor is

  $ Q = C_m V, $

  where $V = V_"in" - V_"out"$ is the membrane potential, $Q$ is the charge stored on
  the inner face, and $C_m$ is the membrane capacitance (typically
  $approx 1 mu"F"\/"cm"^2$).

  Differentiating in time gives the capacitive current, the current required to
  change the voltage at rate $(dif V)/(dif t)$:

  $ I_C = (dif Q) / (dif t) = C_m (dif V) / (dif t). $

  == The membrane is also a leaky resistor

  Ion channels punctuate the bilayer. Even at rest some are open, so the membrane has
  a finite conductance $g_L$ (equivalently, a leak resistance $R_m = 1\/g_L$). The
  leak current is driven by how far the voltage sits from its reversal potential
  $V_"rest"$:

  $ I_L = g_L (V - V_"rest") = (V - V_"rest") / R_m. $

  When $V > V_"rest"$, the leak current is positive: charge flows outward and pulls
  $V$ back down. The capacitor and the leak resistor sit in parallel between the
  inside and the outside of the cell.

  == Kirchhoff's current law

  Now inject an external current $I$. Charge is conserved at the membrane node:
  whatever flows in must either charge the capacitor or escape through the leak.

  $ I = I_C + I_L = C_m (dif V) / (dif t) + (V - V_"rest") / R_m. $

  Multiply through by $R_m$ and define the membrane time constant $tau_m = R_m C_m$:

  $ tau_m (dif V) / (dif t) = -(V - V_"rest") + R_m I. $

  This is the subthreshold LIF equation. Nothing has been assumed beyond Kirchhoff's
  law and linear, passive channels. The dynamics are first-order: any voltage
  perturbation decays exponentially toward $V_"rest" + R_m I$ with time constant
  $tau_m$.

  == Adding a spike

  Real neurons don't just relax. When $V$ crosses a threshold, voltage-gated sodium
  channels open, the membrane briefly depolarizes to nearly $+40 "mV"$, and potassium
  channels repolarize it. The full mechanism is the Hodgkin–Huxley model. LIF throws
  all of that away and replaces it with a rule:

  $ V(t) >= V_"th" quad ==> quad "spike at" t, quad V <- V_"reset". $

  Two parameters, no extra differential equations. Often a refractory period is added
  during which $V$ is clamped at $V_"reset"$.

  == What was thrown away

  The derivation makes the model's assumptions explicit:

  - *Linear leak.* Real channels are voltage- and time-dependent; $g_L$ is treated as constant.
  - *Point neuron.* The whole membrane is collapsed to one node: no dendrites, no axonal delay, no spatial structure.
  - *Phenomenological spike.* The action potential is replaced by a reset, so the model says nothing about spike shape, sodium dynamics, or bursting.

  In exchange you get a single linear ODE with a reset, which integrates in a handful
  of lines and scales to networks of thousands of neurons. That is the LIF neuron
  simulated in #link("exp000.html")[exp000]: same equation, same RC circuit, now with
  current injected and a threshold to cross.
]
