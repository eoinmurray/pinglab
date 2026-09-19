#import "templates/article-layout.typ": journal-article
#import "templates/abstract.typ": journal-abstract
#import "templates/dataset.typ": data-file, input-assets, inputs-ready, pending-report
#import "templates/methods.typ": journal-methods, method-card
#import "templates/references.typ": journal-references
#import "templates/result-card.typ": journal-result-card, result-figure-ref, with-result-sections
#import "/.demolab/lib.typ": cite, data-image, data-json

#let data-file = data-file.with(article: "exp116")

#let meta = (
  tags: ("data", "v36.0.0"),
  title: "Minimal PING Onset Test",
  created_at: "2026-09-19T00:00:00Z",
  updated_at: "2026-09-19",
  description: "A minimal test of oscillatory onset, sampled criticality and inhibitory-timescale dependence in a four-variable PING closure.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp116",)
#let preview-figures = (
  (path: "exp116/minimal-hopf-evidence.svg", label: "minimal mean-field onset evidence"),
)

#let render-report(data-file) = [
  #let numbers = data-json(data-file("exp116/numbers.json"))
  #let reference = numbers.reference
  #let primary = numbers.conditions.filter(row => row.purpose == "gaba_sweep")

  #journal-abstract(
    question: [We asked whether a four-variable PING closure develops a continuous oscillatory onset and whether slower inhibition lowers its onset frequency.],
    approach: [We combined equilibrium continuation, one matched amplitude-ramp test, a six-value inhibitory-decay sweep and four endpoint robustness checks.],
    finding: [The closure passed the prespecified onset, sampled-criticality, timescale and robustness criteria.],
    scope: [The calculation supports a qualitative comparison with a separate spiking network, not a shared bifurcation or quantitative frequency match.],
  )

  == Results

  #with-result-sections[
    #journal-result-card(
      title: "Minimal onset evidence",
      observation: [The reference equilibrium lost stability at #calc.round(reference.onset.drive_nA, digits: 6) nA with onset frequency #calc.round(reference.onset.frequency_Hz, digits: 2) Hz (#result-figure-ref(<fig:exp116-evidence>, panel: "A")). The sampled amplitude branches met the predefined criteria for consistency with a supercritical transition (#result-figure-ref(<fig:exp116-evidence>, panel: "B")). Across the six inhibitory decay times, onset frequency fell from #calc.round(primary.first().onset.frequency_Hz, digits: 2) to #calc.round(primary.last().onset.frequency_Hz, digits: 2) Hz, and all four endpoint closure checks retained that direction (#result-figure-ref(<fig:exp116-evidence>, panel: "C")). Finite ramps cannot exclude a narrower bistable interval or unstable periodic orbit.],
      visual: [
        #figure(
          data-image(
            data-file("exp116/minimal-hopf-evidence.svg"),
            width: 100%,
            alt: "Three panels showing a mean-field eigenvalue crossing, upward and downward amplitude ramps, and onset frequency across inhibitory decay with endpoint robustness checks.",
          ),
          caption: [*A:* Largest real part of the equilibrium Jacobian eigenvalues versus tonic drive at the reference closure; the marker identifies the accepted crossing. *B:* Upward and downward peak-to-peak excitatory-rate amplitudes over the final 500 ms of each 2-s drive step. *C:* Onset frequency at the six reference-closure inhibitory decay times; light endpoint segments show the four low/high effective-noise and rate-relaxation corner checks. All curves are deterministic.],
          kind: image,
          supplement: [Figure],
        ) <fig:exp116-evidence>
      ],
    )
  ]

  #journal-methods(body: (
    method-card([Define the closure], [We used excitatory and inhibitory population rates plus the recurrent excitatory and inhibitory conductances. Each rate relaxed toward a stationary noisy-LIF gain; conductances followed exponential AMPA and GABA filters. Fixed driving forces and prescribed effective-noise and rate-relaxation scales make this a deterministic closure rather than an exact reduction of the spiking network.]),
    method-card([Locate oscillatory onset], [We continued equilibria across 401 tonic-drive values from 0 to 4 nA and refined the first stable-to-unstable complex-eigenvalue crossing. We repeated every continuation on an 801-point grid and required onset drive and frequency to agree within $10^(-7)$ nA and $10^(-4)$ Hz.]),
    method-card([Test sampled criticality], [At the reference closure, we integrated 25 ascending and descending drive steps from 0.10 nA below to 0.55 nA above onset. Each step lasted 2 s and carried its endpoint forward. We measured excitatory peak-to-peak amplitude over the final 500 ms. Consistency with supercritical onset required a branch gap below $10^(-4)$ $"ms"^(-1)$, positive amplitude-squared slope and $R^2_"fit">0.9$.]),
    method-card([Test timescale and robustness], [We repeated onset detection at inhibitory decay times 4.5, 6, 9, 12, 18 and 27 ms with the reference effective-noise scale and rate relaxation. We then tested only the two decay endpoints at the four low/high corners of those closure choices. The robustness criterion required resolved onset at both endpoints and lower frequency under slower inhibition in every corner.]),
  ))

  == Appendix A — Four-variable closure derivation <app:exp116-closure>

  The calculation starts from the reciprocal excitatory–inhibitory motif used
  in conductance-based PING models.#cite(1) It then applies three explicit
  closure operations: homogeneous population averaging, fixed synaptic driving
  forces, and relaxation toward a stationary noisy-LIF gain. These operations
  define a four-variable approximation; they are not an exact reduction of the
  finite spiking network.

  === Conductance-based neurons

  Let $P in {E,I}$ denote an excitatory or inhibitory population. A
  representative neuron has membrane voltage $V_m^P$, capacitance $C_(m,P)$,
  leak conductance $g_(L,P)$ and leak potential $E_L$. Replacing external spike
  input by the tonic current $I_"ext"$, the recurrent subthreshold currents are

  $ C_(m,E) dot(V)_m^E &= -g_(L,E)(V_m^E-E_L)
      -g_i^E(V_m^E-E_i)+I_"ext", \
    C_(m,I) dot(V)_m^I &= -g_(L,I)(V_m^I-E_L)
      -g_e^I(V_m^I-E_e). $ <eq:exp116-membranes>

  Here $E_e$ and $E_i$ are the excitatory and inhibitory reversal potentials;
  $g_e^I$ is excitatory conductance onto I and $g_i^E$ is inhibitory
  conductance onto E. Threshold $V_"th"$, reset $V_"reset"$ and refractory
  time $tau_("ref",P)$ complete each LIF neuron. Its membrane time constant is
  $tau_(m,P)=C_(m,P)/g_(L,P)$.

  For a pathway $P arrow.r Q$, conductance onto target neuron $k$ obeys

  $ dot(g)_(P arrow.r Q,k) = -frac(g_(P arrow.r Q,k), tau_(P arrow.r Q))
      + sum_(j=1)^(N_P) W_(k j)^(P arrow.r Q)
        sum_n delta(t-t_(j n)^P). $ <eq:exp116-event-filter>

  The matrix entry $W_(k j)^(P arrow.r Q)$ is the conductance increment from one
  presynaptic event, $N_P$ is the source-population size, $t_(j n)^P$ is event
  time $n$ from source neuron $j$, and $delta$ is the Dirac impulse.

  === Population averaging

  Define the smooth per-neuron population rate $r_P(t)$ and replace the weighted
  spike sum by its homogeneous mean. If $macron(w)_(P arrow.r Q)$ includes both
  present and absent edges, the summed incoming conductance increment is

  $ G_(P arrow.r Q)=N_P macron(w)_(P arrow.r Q). $ <eq:exp116-summed-coupling>

  Applying this approximation to E→I AMPA and I→E GABA pathways gives

  $ dot(g)_e^I &= -frac(g_e^I, tau_"AMPA")+G_(E arrow.r I)r_E, \
    dot(g)_i^E &= -frac(g_i^E, tau_"GABA")+G_(I arrow.r E)r_I. $ <eq:exp116-mean-conductances>

  Rates use inverse milliseconds, so $G r_P$ has conductance per millisecond.
  This averaging removes individual connectivity, finite-population noise and
  spike correlations. Because the exponential event filters are not normalized
  to unit area, changing $tau_"GABA"$ while holding $G_(I arrow.r E)$ fixed also
  changes stationary inhibitory conductance.

  === Fixed driving forces and rate closure

  Evaluating synaptic driving forces at the leak potential gives the positive
  magnitudes $Delta V_"exc"=E_e-E_L=65$ mV and
  $Delta V_"inh"=E_L-E_i=15$ mV. The currents supplied to the population gains
  are therefore

  $ I_E=I_"ext"-Delta V_"inh"g_i^E, quad
    I_I=Delta V_"exc"g_e^I. $ <eq:exp116-effective-currents>

  Freezing these voltage differences removes conductance-dependent shunting.
  For population $P$, the stationary noisy-LIF gain is the Siegert
  first-passage rate#cite(2, 3)

  $ Phi_P(I)=left[tau_("ref",P)+tau_(m,P)sqrt(pi)
      integral_(alpha_P(I))^(beta_P(I)) e^(u^2)(1+op("erf")(u)) dif u right]^(-1), $ <eq:exp116-siegert>

  where $mu_P(I)=E_L+I/g_(L,P)$,
  $alpha_P=(V_"reset"-mu_P)/sigma_V$ and
  $beta_P=(V_"th"-mu_P)/sigma_V$. The effective voltage-noise scale $sigma_V$
  is prescribed rather than derived from recurrent spike fluctuations.

  Finally, introduce phenomenological rate relaxation
  $tau_(r,P)=kappa tau_(m,P)$. Substitution of @eq:exp116-effective-currents and
  @eq:exp116-mean-conductances gives the implemented four-variable system

  $ tau_(r,E) dot(r)_E &= -r_E+Phi_E(I_"ext"-Delta V_"inh"g_i^E), \
    tau_(r,I) dot(r)_I &= -r_I+Phi_I(Delta V_"exc"g_e^I), \
    dot(g)_e^I &= -frac(g_e^I, tau_"AMPA")+G_(E arrow.r I)r_E, \
    dot(g)_i^E &= -frac(g_i^E, tau_"GABA")+G_(I arrow.r E)r_I. $ <eq:exp116-four-variable-model>

  Thus $x=(r_E,r_I,g_e^I,g_i^E)^T$ is the complete state. The closure retains
  reciprocal population feedback and synaptic filtering, but omits membrane
  voltage as a state, shunting, heterogeneous connectivity, coloured recurrent
  fluctuations and synchrony-dependent corrections. Any Hopf crossing found in
  this system belongs to the deterministic closure, not automatically to the
  conductance-based spiking network from which its structure was motivated.

  #journal-references((
    (text: [C. Börgers and N. Kopell. “Synchronization in Networks of Excitatory and Inhibitory Neurons with Sparse, Random Connectivity.” _Neural Computation_ *15*(3), 509–538 (2003).], doi: "10.1162/089976603321192059"),
    (text: [A. J. F. Siegert. “On the First Passage Time Probability Problem.” _Physical Review_ *81*, 617–623 (1951).], doi: "10.1103/PhysRev.81.617"),
    (text: [L. M. Ricciardi and L. Sacerdote. “The Ornstein–Uhlenbeck Process as a Model for Neuronal Activity. I. Mean and Variance of the Firing Time.” _Biological Cybernetics_ *35*, 1–9 (1979).], doi: "10.1007/BF01845839"),
  ))
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(data-file, inputs, [], preview-figures)
}

#let meta = meta + (assets: input-assets("exp116", inputs))
#let body = journal-article("exp116", inputs, report-body)
