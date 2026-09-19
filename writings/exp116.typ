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
  (path: "exp116/hopf-onset.svg", label: "Hopf onset"),
  (path: "exp116/sampled-criticality.svg", label: "sampled criticality"),
  (path: "exp116/frequency-vs-gaba.svg", label: "frequency versus inhibitory decay"),
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
      title: "Equilibrium loses stability",
      observation: [At the reference closure, one complex-conjugate eigenvalue pair crossed from negative to positive real part at #calc.round(reference.onset.drive_nA, digits: 6) nA while the remaining modes stayed damped (#result-figure-ref(<fig:exp116-onset>)). The equilibrium therefore lost stability through a numerically resolved Hopf crossing.],
      visual: [
        #figure(
          data-image(
            data-file("exp116/hopf-onset.svg"),
            width: 70%,
            alt: "Largest real part of the four-variable model's Jacobian eigenvalues crossing zero as tonic drive increases.",
          ),
          caption: [Largest real part of the equilibrium flow-Jacobian eigenvalues across tonic drive at $tau_"GABA"=6$ ms, $sigma_V=4$ mV and $kappa=1$. The horizontal line marks zero real part; the vertical line and marker identify the refined crossing.],
          kind: image,
          supplement: [Figure],
        ) <fig:exp116-onset>
      ],
    )

    #let frequency-table = table(
      columns: (1.4fr, 1fr),
      table.header([Quantity], [Reference value]),
      [Onset drive $I_"ext"^*$], [#calc.round(reference.onset.drive_nA, digits: 6) nA],
      [Angular frequency $omega_"Hopf"$], [#calc.round(reference.onset.omega_per_ms, digits: 6) rad/ms],
      [Hopf frequency $f_"Hopf"$], [#calc.round(reference.onset.frequency_Hz, digits: 2) Hz],
    )
    #journal-result-card(
      title: "Crossing predicts onset frequency",
      observation: [The imaginary part of the crossing eigenvalues corresponded to an onset frequency of #calc.round(reference.onset.frequency_Hz, digits: 2) Hz (@tab:exp116-frequency). This is an eigenvalue-derived frequency of the deterministic closure, not a measured spectral peak from the spiking network.],
      visual: [
        #figure(
          align(center, frequency-table),
          kind: table,
          caption: [Reference crossing quantities. Angular frequency is the positive imaginary part of the critical eigenvalue in radians per millisecond; $f_"Hopf"=1000 omega_"Hopf"/(2 pi)$ converts it to hertz.],
        ) <tab:exp116-frequency>
      ],
    )

    #journal-result-card(
      title: "Sampled onset is supercritical",
      observation: [The upward and downward amplitude branches nearly coincided, with maximum gap #calc.round(reference.criticality.branch_gap_per_ms, digits: 8) $"ms"^(-1)$, while the amplitude-squared fit had $R^2_"fit"=#calc.round(reference.criticality.amplitude_squared_r2, digits: 4)$ (#result-figure-ref(<fig:exp116-criticality>)). The sampled behaviour was therefore consistent with a supercritical Hopf; finite ramps cannot exclude a narrower bistable interval or unstable periodic orbit.],
      visual: [
        #figure(
          data-image(
            data-file("exp116/sampled-criticality.svg"),
            width: 70%,
            alt: "Upward and downward excitatory-rate oscillation amplitudes across tonic drive around the Hopf onset.",
          ),
          caption: [Peak-to-peak excitatory-rate amplitude over the final 500 ms of each 2-s drive step at the reference closure. Circles show the ascending sequence, open squares the descending sequence, and the vertical line the refined Hopf onset.],
          kind: image,
          supplement: [Figure],
        ) <fig:exp116-criticality>
      ],
    )

    #journal-result-card(
      title: "Slower inhibition lowers frequency",
      observation: [Across the reference inhibitory-decay sweep, predicted Hopf frequency fell from #calc.round(primary.first().onset.frequency_Hz, digits: 2) to #calc.round(primary.last().onset.frequency_Hz, digits: 2) Hz (#result-figure-ref(<fig:exp116-gaba-frequency>)). Every low/high effective-noise and rate-relaxation corner retained the same endpoint direction.],
      visual: [
        #figure(
          data-image(
            data-file("exp116/frequency-vs-gaba.svg"),
            width: 70%,
            alt: "Hopf onset frequency decreasing with inhibitory decay for the reference closure and four endpoint robustness checks.",
          ),
          caption: [Eigenvalue-derived Hopf frequency versus inhibitory decay. The black curve uses $sigma_V=4$ mV and $kappa=1$ at all six decay times. Grey endpoint segments use the four combinations of $sigma_V in {3,6}$ mV and $kappa in {0.5,2}$. These deterministic comparisons have no statistical uncertainty intervals.],
          kind: image,
          supplement: [Figure],
        ) <fig:exp116-gaba-frequency>
      ],
    )
  ]

  #journal-methods(body: (
    method-card([Define the closure], [We used excitatory and inhibitory population rates plus the recurrent excitatory and inhibitory conductances. Each rate relaxed toward a stationary noisy-LIF gain; conductances followed exponential AMPA and GABA filters. Fixed driving forces and prescribed effective-noise and rate-relaxation scales make this a deterministic closure rather than an exact reduction of the spiking network. #link(<app:exp116-closure>)[Appendix A] derives the four-variable system from the conductance-based neuron and synapse equations.]),
    method-card([Continue the equilibria], [We solved the equilibrium of the system derived in #link(<app:exp116-closure>)[Appendix A] at 401 evenly spaced tonic-drive values from 0 to 4 nA. Adjacent solutions initialized one another. We repeated the continuation on an 801-point grid and required the refined onset drive and frequency to agree within $10^(-7)$ nA and $10^(-4)$ Hz.]),
    method-card([Identify the Hopf crossing], [At each equilibrium, we calculated the four eigenvalues of the continuous-time flow Jacobian. We refined the first crossing at which one complex-conjugate pair changed from negative to positive real part while the other pair remained damped (#result-figure-ref(<fig:exp116-onset>)).]),
    method-card([Calculate onset frequency], [We took the positive imaginary part $omega_"Hopf"$ of the critical eigenvalue in radians per millisecond and calculated $f_"Hopf"=1000 omega_"Hopf"/(2 pi)$ in hertz (@tab:exp116-frequency).]),
    method-card([Test sampled criticality], [At the reference closure, we integrated 25 ascending and descending drive steps from 0.10 nA below to 0.55 nA above onset. Each step lasted 2 s and carried its endpoint forward. We measured excitatory peak-to-peak amplitude over the final 500 ms. Consistency with supercritical onset required a branch gap below $10^(-4)$ $"ms"^(-1)$, positive amplitude-squared slope and $R^2_"fit">0.9$ (#result-figure-ref(<fig:exp116-criticality>)).]),
    method-card([Vary inhibitory decay], [We repeated onset detection at $tau_"GABA"=4.5,6,9,12,18$ and $27$ ms with the reference effective-noise scale and rate relaxation. We then tested only the two decay endpoints at the four low/high corners of those closure choices. Robustness required resolved onset at both endpoints and lower frequency under slower inhibition in every corner (#result-figure-ref(<fig:exp116-gaba-frequency>)).]),
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

  $ Phi_P(I)=lr([tau_("ref",P)+tau_(m,P)sqrt(pi)
      integral_(alpha_P(I))^(beta_P(I)) e^(u^2)(1+op("erf")(u)) dif u])^(-1), $ <eq:exp116-siegert>

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
