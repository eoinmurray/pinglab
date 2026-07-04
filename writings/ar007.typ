#let meta = (
  title: "Bayesian Literature Review",
  date: "2026-05-19",
  description: "A reading list on Bayesian and sampling-based theories of cortical computation, with a shared glossary of terms.",
  collection: "miscellaneous",
)

#let body = [
  #table(
    columns: (auto, auto),
    align: (left, left),
    [*Paper*], [*Year*],
    [#link("https://doi.org/10.1038/nn1790")[Ma, Beck, Latham & Pouget — _Bayesian Inference with Probabilistic Population Codes_]], [2006],
    [#link("https://doi.org/10.1146/annurev.neuro.26.041002.131112")[Pouget, Dayan & Zemel — _Inference and Computation with Population Codes_]], [2003],
    [#link("https://doi.org/10.1016/j.tics.2010.01.003")[Fiser, Berkes, Orbán & Lengyel — _Statistically Optimal Perception and Learning: From Behavior to Neural Representations_]], [2010],
    [#link("https://doi.org/10.1016/j.neuron.2016.09.038")[Orbán, Berkes, Fiser & Lengyel — _Neural Variability and Sampling-Based Probabilistic Representations in the Visual Cortex_]], [2016],
    [#link("https://doi.org/10.1371/journal.pcbi.1005186")[Aitchison & Lengyel — _The Hamiltonian Brain: Efficient Probabilistic Inference with Excitation-Inhibition Networks_]], [2016],
    [#link("https://doi.org/10.1038/s41593-020-0671-1")[Echeveste, Aitchison, Hennequin & Lengyel — _Cortical-like Dynamics in Recurrent Circuits Optimized for Sampling-Based Probabilistic Inference_]], [2020],
    [#link("https://doi.org/10.1016/j.neuron.2021.10.024")[Padamsey, Katsanevaki, Dupuy & Rochefort — _Neocortex Saves Energy by Reducing Coding Precision during Food Scarcity_]], [2022],
  )

  == Terms to know first

  The papers above share a small vocabulary that runs across all of them. If a term feels hazy, look it up before reading — they are not redefined in each paper.

  *Bayesian fundamentals.*

  - _Prior, likelihood, posterior._ $p(theta)$, $p(x | theta)$, $p(theta | x) prop p(x | theta) p(theta)$. The posterior is what every paper here claims the brain represents in some form.
  - _Generative model._ The forward model the brain is assumed to have inverted: latent causes $->$ sensory observations.
  - _Recognition / inference._ The inverse problem — recovering $p(theta | x)$ from $x$ — performed approximately by the circuit.
  - _Marginalisation._ Integrating out nuisance variables. Cortical circuits are claimed to do this in different ways depending on whether the representation is parametric or sample-based.

  *Two flavours of neural representation of probability.* This is the central split in the reading list and the two camps disagree on it.

  - _Parametric (probabilistic population code, PPC)._ The instantaneous population firing rate $bold(r)$ _encodes the parameters_ of a distribution over $theta$ — for an exponential-family code, $p(theta | x) prop exp(bold(h)(theta)^top bold(r))$. Posterior uncertainty is read off the population gain. Ma et al. 2006 and Pouget et al. 2003 are the canonical PPC papers.
  - _Sampling-based._ The instantaneous activity $bold(r)(t)$ _is_ a sample from the posterior, $bold(r)(t) tilde p(theta | x)$. Uncertainty is read off the trial-to-trial variability over time. Fiser et al. 2010, Orbán et al. 2016, and the Lengyel-group papers (Aitchison & Lengyel 2016, Echeveste et al. 2020) are the canonical sampling-camp papers.
  - _Variational inference (background)._ A third flavour — fit a parametric $q(theta)$ to the posterior by minimising KL divergence. Worth knowing as the ML reference point, though none of these papers commit to it as the brain's algorithm.

  *Population-code mechanics.*

  - _Tuning curve_ $f_i (theta)$ — neuron $i$'s mean rate as a function of stimulus $theta$.
  - _Fisher information_ $I_F (theta)$ — the precision a code carries about $theta$; for an unbiased estimator, $"Var"(hat(theta)) >= 1\/I_F (theta)$ (Cramér–Rao). Often discussed per spike, to compare codes at matched energy.
  - _Linear Fisher information._ The piece of $I_F$ accessible by a linear readout — what a downstream neuron with synaptic weights can actually extract.
  - _Noise correlations_ $c_(i j)$ — trial-to-trial covariance between neurons at fixed stimulus. Information-limiting correlations are the slice of $c_(i j)$ aligned with the tuning gradient $partial f \/ partial theta$; they cap $I_F$ no matter how many neurons you average.
  - _Poisson variability._ Spike-count variance ≈ mean. The baseline noise model PPCs are built on.

  *Sampling-camp specifics.*

  - _Langevin dynamics._ Stochastic gradient on $-log p(theta | x)$ plus white noise — the simplest dynamical system whose stationary distribution is the posterior. The recurrent-circuit papers are versions of this idea with biological constraints.
  - _Hamiltonian Monte Carlo (HMC)._ Sampling with momentum — propose moves along trajectories of a fictitious Hamiltonian, accept by Metropolis. Aitchison & Lengyel argue E/I circuits implement an HMC-like sampler with the inhibitory population playing the momentum role.
  - _Inhibition-stabilised network (ISN)._ A recurrent E/I circuit whose excitatory subnetwork is _unstable_ on its own and stabilised only by inhibition. Empirically common in cortex; the sampler papers exploit ISN dynamics for fast, well-mixed sampling.
  - _Mixing time, autocorrelation time._ How quickly a sampler decorrelates from its current state — sets how fast a circuit can revise its uncertainty estimate. Echeveste et al. argue cortical-like transients arise from optimising for fast mixing.
  - _Stochastic stability vs. transients._ A sampler must be globally stable but locally fluctuating. The non-trivial transient responses cortical circuits show are read as evidence of a sampler near, not at, equilibrium.

  *Coding-cost and metabolism.*

  - _Bits per spike._ Mutual information per spike between stimulus and response — the efficiency the energetics papers ask about.
  - _Coding precision._ The width of the represented posterior; can be widened to save metabolic cost without changing the represented mean. Padamsey et al. 2022 show this happens _in vivo_ under food scarcity — directly tying the Bayesian and metabolic stories together.
  - _Metabolic cost of a spike._ Sodium-pump ATP per action potential; the currency the precision–cost trade-off is denominated in.

  *Useful background that none of these papers stop to define.*

  - _Exponential family, sufficient statistics, natural parameters._ PPCs lean on this — the firing rates are the natural parameters of an exponential-family posterior.
  - _KL divergence._ The asymmetric distance between distributions, used to score how good an approximate posterior is.
  - _Stationary distribution of a Markov chain._ The distribution the chain converges to and samples from at long times. The sampling-camp claim is that the cortical chain's stationary distribution is the posterior.
]
