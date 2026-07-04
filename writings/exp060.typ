#import "/demolab-engine/build/lib.typ": numbers-table, provenance-footer

#let meta = (
  title: "Strong vs Weak Coupling",
  date: "2026-06-26",
  description: "What strong coupling means: the per-synapse weight scales as 1/sqrt(K) with fan-in, forcing E-I balance and irregular cortex-like firing.",
  collection: "miscellaneous",
  status: "final",
)

#let run = json("/artifacts/data/exp060/numbers.json")

#let body = [
  What does it mean to say a recurrent network is _strongly coupled_? It is tempting to read it as "dense wiring" or "big synaptic weights," but neither alone is the definition. The real answer is about how the synaptic weight _scales_ as the network grows — and getting it right is what separates a network that fires like cortex from one that does not. This note derives that from scratch, and the two figures are computed straight from the scaling laws, not simulated.

  == The short version

  - *It is about scaling, not size.* Strong coupling means the per-synapse weight shrinks _slowly_ — as $1 \/ sqrt(K)$ — as the in-degree $K$ grows. Weak coupling shrinks it faster, as $1 \/ K$.
  - *Strong coupling forces balance.* The $1 \/ sqrt(K)$ scaling makes each population's gross input $O(sqrt(K))$ — enormous — so excitation and inhibition have no choice but to cancel. The small $O(1)$ residual that is left over, dominated by fluctuations, drives irregular ($C V approx 1$), cortex-like firing.
  - *Weak coupling kills the fluctuations.* The $1 \/ K$ scaling sends the input fluctuation to zero; the network goes smooth and deterministic — analytically convenient, but not cortex.
  - *Weight is not coupling.* The weight $w$ you set and the coupling $J = w sqrt(K)$ that fixes the regime differ by a factor of $sqrt(K)$; "strongly coupled" is always a statement about $J$, never $w$ alone.

  The rest of the note earns each of these.

  == What one neuron receives

  Take a single neuron in a large network. The input it integrates from one presynaptic population has a mean and a fluctuating part:

  $ mu approx K w r quad (1) $

  $ sigma approx sqrt(K w^2 r) = sqrt(K) w sqrt(r) quad (2) $

  where

  - $K$ — the number of presynaptic inputs onto the neuron (its in-degree), set by connection density times population size; here $K = (1-s) N_"pre"$, with $s$ the sparsity.
  - $w$ — the *per-synapse weight*: the mean of the recurrent weight matrix $W$, i.e. one synapse's peak conductance, the value you pass to the simulator (_--w-ei_ and friends).
  - $r$ — the mean firing rate of the presynaptic population.
  - $mu$, $sigma$ — the mean and standard deviation of the total input from that population (the $K$ spike trains treated as independent, so variances add).

  Equation (1) is "K inputs, each of size $w$, each at rate $r$." Equation (2) is the same sum's fluctuation. Everything that follows hinges on one question: *as the network grows, how does $w$ scale with $K$?* Two answers, two worlds.

  == Weak coupling — weights shrink as $1 \/ K$

  The classical mean-field choice keeps the mean input finite by shrinking each weight in step with the number of inputs:

  $ w = (w_0) / K quad ==> quad mu approx w_0 r = O(1), quad sigma approx (w_0 sqrt(r)) / sqrt(K) ->_(K -> oo) 0 quad (3) $

  Each input is a tiny nudge, the mean is whatever the drive makes it, and the fluctuations _vanish_ as the network grows. The neuron rides the population average, smoothly and deterministically — the regime behind classical rate models and Wilson–Cowan dynamics. Convenient, but it cannot produce the irregular, Poisson-like firing of cortex.

  == Strong coupling — weights shrink as $1 \/ sqrt(K)$

  The van Vreeswijk–Sompolinsky choice shrinks the weights more slowly:

  $ w = J / sqrt(K) quad ==> quad mu approx sqrt(K) J r = O(sqrt(K)), quad sigma approx J sqrt(r) = O(1) quad (4) $

  Now two things happen at once. The fluctuation stays finite, $O(1)$ — it does not wash out, and it is the seed of irregular firing. But the _mean_ input from a single population is now $O(sqrt(K))$, enormous for a realistic cell ($K tilde 10^3$–$10^4$). A current that large would pin the neuron far above threshold and saturate its rate. Something has to cancel it — which is the whole story, two sections down.

  == Weight is not coupling

  That constant $J$ in equation (4) is the *coupling*, and it is worth pinning down because it is constantly confused with the weight:

  - the *per-synapse weight* $w$ (the matrix mean $W$, what you set) is a number you choose;
  - the *coupling* $J = w sqrt(K)$ is _derived_ — it tells you which regime that choice puts you in.

  They differ by a factor of $sqrt(K)$, and that is the point. Two networks with the same weight but different fan-in are coupled differently; two with the same $J$ are in the same regime even if their weights and densities differ. So "is it strongly coupled?" is a question about $J$ (equivalently, about how $w$ compares to $1 \/ sqrt(K)$), never about $w$ alone. Holding $J$ fixed as $K -> oo$ _means_ letting the weight shrink as $w = J \/ sqrt(K)$ — that is the operating definition of strong coupling.

  (Cross-referencing note: #link("/exp058/")[exp058] currently writes this relation inverted, $J = w \/ sqrt(K)$. The version here — $J = w sqrt(K)$, weights shrinking as $1 \/ sqrt(K)$ — is the one that keeps $sigma$ at $O(1)$, so it is the convention to trust; the two are being reconciled.)

  #figure(
    image("/artifacts/data/exp060/input_scaling.png", width: 100%),
    caption: [Equations (3)–(4) made visual. *Left:* the mean recurrent input $mu$ — under strong coupling it climbs $prop sqrt(K)$ (so each population's drive is huge and must cancel); under weak coupling it is flat. *Right:* the fluctuation $sigma$, and the punchline — under strong coupling it stays $O(1)$ as the network grows (irregular firing survives), while under weak coupling it decays $prop 1 \/ sqrt(K)$ to zero (the network goes deterministic). Analytic; no simulation.],
  )

  == Why strong coupling forces balance

  The only thing that can cancel an $O(sqrt(K))$ excitatory current is an equally large $O(sqrt(K))$ inhibitory one. With both populations:

  $ mu_"net" approx sqrt(K) (J_E r_E - J_I r_I) quad (5) $

  For the neuron to fire at a finite rate, the bracket must be small — of order $1 \/ sqrt(K)$ — so that $mu_"net"$ returns to $O(1)$:

  $ J_E r_E - J_I r_I = O(1 / sqrt(K)) quad (6) $

  This is the *balance condition*, and the key point is that _nobody imposes it by hand_. You cannot tune external parameters to satisfy (6) at every $K$; instead the rates $r_E$ and $r_I$ adjust themselves until gross excitation and gross inhibition track and nearly cancel. The cancellation is self-consistent and emergent. What is left over — the small residual between two huge opposing currents — is dominated by the $O(1)$ fluctuations of (4), not a steady drive. So the membrane potential hovers near threshold and crosses it at irregular, fluctuation-driven moments. ($r_E, r_I$ are the self-organised population rates; $J_E, J_I$ the two couplings; $mu_"net"$ the net input, a small difference of two $O(sqrt(K))$ terms.)

  == What the balanced state gives you

  Because firing is driven by the residual rather than a mean overshoot, three cortically realistic things follow:

  - *Irregular spiking.* Inter-spike intervals look Poisson, $C V approx 1$ — the hallmark of cortical firing — with no injected noise. The irregularity is intrinsic and, in the deterministic limit, chaotic.
  - *Fast, linear tracking.* Balance is dynamic (rates re-cancel almost instantly), so the network responds on a single-synapse timescale and the population rate is roughly linear in the drive.
  - *Robustness, not fine-tuning.* Balance emerges from the strong-coupling scaling itself; it is not a knife-edge.

  This is why "strongly coupled, recurrent, balanced" travel together: strong coupling ($w tilde 1 \/ sqrt(K)$) _forces_ balance, and balance makes the state irregular and fast. The simulated counterpart of Figure 1 — irregularity ($C V$) surviving as $K$ grows under strong coupling but decaying under weak — is measured directly in #link("/exp058/")[exp058, Figure 2].

  == The conductance-based wrinkle

  Equations (1)–(6) treat inputs as currents. Real synapses — and this project's model — are conductance-based: a synapse opens a conductance $g$ and injects $g (V - E_"rev")$, which depends on the voltage and reverses at $E_"rev"$. Strong coupling then means the _total_ synaptic conductance is large, with a consequence the current picture misses — the effective membrane time constant shrinks,

  $ tau_"eff" = (C_m) / (g_L + g_"tot") quad (7) $

  so a strongly coupled conductance-based neuron is not just driven harder, it integrates _faster_ and its gain changes with the input level. That is why conductance-based balanced networks need their own theory: the convenient current-based scaling laws mispredict rates and correlations once $g_"tot"$ ($O(sqrt(K))$ here) is large. ($C_m$ membrane capacitance, $g_L$ leak conductance, $E_"rev"$ reversal potential.)

  == Putting it in this project's terms

  The model used throughout this lab lives in the strong-coupling world by construction, and four quantities chain together. Sparsity sets the fan-in; fan-in and weight set the coupling:

  $ K = (1-s) N_"pre", quad J = w sqrt(K) = w sqrt((1-s) N_"pre") quad (8) $

  You choose $s$ and $N_"pre"$ (which fix $K$) and you choose $w$; together $w$ and $K$ land you in regime $J$. So the two knobs you actually turn — $s$ and $w$ — are _not_ independent. Making the network sparser at fixed $w$ _weakens_ the coupling, and to stay in the same balanced regime you must raise the weight to compensate:

  $ w prop 1 / sqrt((1-s) N_"pre") quad "(to hold " J " fixed)" quad (9) $

  #figure(
    image("/artifacts/data/exp060/weight_vs_sparsity.png", width: 100%),
    caption: [Equation (9). To stay in the same coupling regime as the network is made sparser (right, smaller $K$), the per-synapse weight has to grow — gently at first, then steeply as $s -> 1$. Sparsity $s$ and weight $w$ are not independent: a change in one must be matched in the other to hold $J$ fixed.],
  )

  So $W$ is the weight you set, $s$ sets the fan-in $K$, and $J$ is the regime the pair lands you in. The balanced state itself — irregular, asynchronous, rhythmless firing with $C V approx 1$ — is built and measured in #link("/exp058/")[exp058], where _weights scaled to hold $J$ fixed_ versus _weights held fixed_ is exactly the strong-versus-weak distinction made here. PING then sits one step beyond: it keeps the strong, balanced E–I coupling but adds a _structured_ recurrent loop (E→I→E) that can turn the asynchronous balanced regime into a gamma rhythm — the onset studied in the #link("/ar009/")[manuscript]. Strong coupling is the substrate; the structured loop is what makes it tick.

  == Sources

  - #link("https://doi.org/10.1126/science.274.5293.1724")[van Vreeswijk & Sompolinsky (1996), _Chaos in Neuronal Networks with Balanced Excitatory and Inhibitory Activity_] — the origin of the $1 \/ sqrt(K)$ strong-coupling scaling and the balanced state.
  - #link("https://doi.org/10.1023/A:1008925309027")[Brunel (2000), _Dynamics of Sparsely Connected Networks of Excitatory and Inhibitory Spiking Neurons_] — the analytic phase diagram of where these regimes live and oscillate.
  - #link("https://doi.org/10.1126/science.1179850")[Renart et al. (2010), _The Asynchronous State in Cortical Circuits_] — how balance also decorrelates the population, not just individual cells.
  - #link("https://doi.org/10.1162/089976600300015899")[Gerstner (2000), _Population Dynamics of Spiking Neurons_] — the population-level companion.
]
