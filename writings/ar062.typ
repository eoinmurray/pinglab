#import "/.demolab/lib.typ": cite, reference-list

#let meta = (
  title: "Bayesian and sampling-based cortical computation",
  date: "2026-06-28",
  description: "A project-oriented reading guide to Bayesian population codes, neural sampling, recurrent E/I dynamics, and the background needed to read the core papers.",
  collection: "miscellaneous",
)

#let refs = (
  (text: [Pouget, Dayan & Zemel (2003), _Inference and computation with population codes_], doi: "10.1146/annurev.neuro.26.041002.131112"),
  (text: [Ma, Beck, Latham & Pouget (2006), _Bayesian inference with probabilistic population codes_], doi: "10.1038/nn1790"),
  (text: [Fiser, Berkes, Orbán & Lengyel (2010), _Statistically optimal perception and learning: from behavior to neural representations_], doi: "10.1016/j.tics.2010.01.003"),
  (text: [Aitchison & Lengyel (2016), _The Hamiltonian brain: efficient probabilistic inference with excitation-inhibition networks_], doi: "10.1371/journal.pcbi.1005186"),
  (text: [Orbán, Berkes, Fiser & Lengyel (2016), _Neural variability and sampling-based probabilistic representations in the visual cortex_], doi: "10.1016/j.neuron.2016.09.038"),
  (text: [Echeveste, Aitchison, Hennequin & Lengyel (2020), _Cortical-like dynamics in recurrent circuits optimized for sampling-based probabilistic inference_], doi: "10.1038/s41593-020-0671-1"),
  (text: [Padamsey, Katsanevaki, Dupuy & Rochefort (2022), _Neocortex saves energy by reducing coding precision during food scarcity_], doi: "10.1016/j.neuron.2021.10.024"),
)

#let body = [
  This guide connects the neural-Bayesian literature to our PING project. The central question is not merely whether cortical behaviour can look Bayesian. It is whether recurrent excitatory/inhibitory (E/I) dynamics can represent uncertainty by sampling, and whether the oscillatory and transient regimes we measure are computationally useful rather than incidental.

  == 1. Papers and their relation to the project

  + *Pouget, Dayan & Zemel (2003), population-code foundations.* A broad account of inference with population activity#cite(1). It supplies the coding language needed to distinguish a distribution represented by a population from a single decoded estimate. For this project, it is the conceptual baseline: before asking whether PING dynamics sample a posterior, we need to state what information the population activity carries and what a downstream readout could recover.

  + *Ma, Beck, Latham & Pouget (2006), probabilistic population codes.* This is the canonical parametric alternative to neural sampling#cite(2). Instantaneous population rates encode the parameters of an exponential-family distribution, and cue combination becomes addition of population activity. It gives us a competing hypothesis: uncertainty may be encoded in population gain rather than in temporal variability. Any sampling claim from our network should make predictions that separate these accounts.

  + *Fiser, Berkes, Orbán & Lengyel (2010), the sampling hypothesis.* This review develops the proposal that spontaneous and evoked cortical activity are samples from an internal generative model#cite(3). It connects trial-to-trial variability to probabilistic representation. For our project, it motivates treating fluctuations as signal-bearing dynamics and asking whether the stationary activity distribution, rather than only the mean firing rate or oscillation frequency, matches a target distribution.

  + *Aitchison & Lengyel (2016), E/I circuits as Hamiltonian samplers.* This paper is the closest theoretical bridge to our model#cite(4). It assigns different computational roles to excitatory and inhibitory populations and argues that their coupled dynamics can accelerate sampling. It should inform which E/I variables we analyse, how phase-lagged activity might act like position and momentum, and why balanced oscillatory dynamics could improve exploration rather than simply stabilize the circuit.

  + *Orbán, Berkes, Fiser & Lengyel (2016), neural variability as sampling evidence.* This paper tests sampling-based predictions in visual cortex#cite(5). It is useful for translating the theory into empirical diagnostics, especially the relation between stimulus-dependent uncertainty, response variability, and the distribution of population activity. Our analogue is to test whether trained PING networks change their variability in the way the represented posterior requires.

  + *Echeveste, Aitchison, Hennequin & Lengyel (2020), optimized recurrent samplers.* Recurrent networks optimized for sampling develop cortical-like transients and variability#cite(6). This is the most direct precedent for training a recurrent circuit toward a sampling objective. It motivates comparing task performance with mixing, autocorrelation, transient amplification, and stability, including our Δt-stability diagnostics. A network that fits the task but mixes poorly is not yet a convincing sampler.

  + *Padamsey, Katsanevaki, Dupuy & Rochefort (2022), precision under metabolic constraint.* Food scarcity reduces cortical coding precision and energy use#cite(7). This adds a cost axis to the project. If spike rate, inhibitory recruitment, or oscillatory precision has a metabolic price, the best circuit may trade posterior precision against energy rather than maximize accuracy without constraint.

  == 2. Reading order and guide

  Read the papers in four passes. The order below moves from representation, through the sampling hypothesis, to circuit implementation and finally energetic constraint.

  + *Establish the representational alternatives: Pouget et al. (2003), then Ma et al. (2006).* Ask what object neural activity represents, how uncertainty is encoded, and what operations a downstream circuit can perform. Keep the distinction between _encoding a distribution's parameters_ and _producing samples from a distribution_ explicit.

  + *Learn the sampling claim: Fiser et al. (2010), then Orbán et al. (2016).* Ask what counts as a neural sample, over what time or trial ensemble the distribution is defined, and which observations distinguish meaningful sampling variability from ordinary noise.

  + *Study the E/I mechanism: Aitchison & Lengyel (2016), then Echeveste et al. (2020).* Track the mapping from mathematical sampler variables to excitatory and inhibitory activity. Then look for measurable consequences in our networks: phase relations, autocorrelation time, mixing, transient amplification, stationary distributions, and sensitivity to the integration step $Delta t$.

  + *Add the resource constraint: Padamsey et al. (2022).* Ask whether coding precision, firing cost, and inhibitory stabilization can be treated as a joint objective. This paper is best read last because it changes the optimization question from “what is the most accurate code?” to “what precision is worth its metabolic cost?”

  On a first pass, read abstracts, figures, and discussion. On a second pass, reconstruct the representation and objective in equations. On a third pass, extract one table with four columns for each paper: represented quantity, circuit mechanism, observable prediction, and corresponding diagnostic in our PING model.

  == 3. Terms and background to know

  === Bayesian fundamentals

  A *generative model* is a forward account of how an unobserved cause $theta$ produces an observation $x$. *Recognition* or *inference* reverses that account and estimates the cause from the observation. Bayes' rule is

  $ p(theta | x) = (p(x | theta) p(theta))/(p(x)) prop p(x | theta) p(theta). $

  - $theta$ is the latent or unobserved cause.
  - $x$ is the observation.
  - $p(theta)$ is the *prior*, the distribution before observing $x$.
  - $p(x | theta)$ is the *likelihood*, the probability of $x$ under a proposed cause.
  - $p(theta | x)$ is the *posterior*, the updated distribution over causes.
  - $p(x) = integral p(x | theta) p(theta) dif theta$ is the *evidence* or marginal likelihood, which normalizes the posterior.

  *Marginalisation* integrates out variables that are not of interest. For a target $theta_1$ and nuisance variable $theta_2$,

  $ p(theta_1 | x) = integral p(theta_1, theta_2 | x) dif theta_2. $

  - $theta_1$ is the quantity to retain.
  - $theta_2$ is the nuisance variable to integrate out.
  - $x$ is the observation.

  High-dimensional marginalisation is usually intractable, which is why approximate inference is needed.

  === Probability representations

  - *Probabilistic population code (PPC).* Population firing rates encode parameters of a probability distribution. In an exponential-family PPC,

    $ p(theta | x) prop exp(bold(h)(theta)^top bold(r)). $

    Here $theta$ is the latent variable, $x$ is the observation, $bold(h)(theta)$ is a vector of basis functions or sufficient statistics, and $bold(r)$ is the population response. Posterior uncertainty is commonly expressed through population gain.

  - *Sampling-based code.* Instantaneous activity is treated as a draw from a distribution,

    $ bold(r)(t) tilde p(bold(r) | x). $

    Here $bold(r)(t)$ is population activity at time $t$, $x$ is the observation, and $p(bold(r) | x)$ is the target activity distribution. Uncertainty is expressed by variability across time or trials. A decoder estimates expectations from multiple samples rather than reading distribution parameters from a single activity vector.

  - *Variational inference (VI).* A tractable distribution $q_phi(theta)$ is fitted to the posterior by optimizing parameters $phi$. VI turns inference into optimization and is usually fast, but its approximation is limited by the chosen family for $q_phi$.

  === Exponential families and approximate inference

  An exponential-family distribution has the form

  $ p(x | eta) = h(x) exp(eta^top T(x) - A(eta)). $

  - $x$ is the random variable.
  - $eta$ is the vector of *natural parameters*.
  - $T(x)$ is the vector of *sufficient statistics*.
  - $A(eta)$ is the log-partition function.
  - $h(x)$ is the base measure.

  A prior and likelihood are *conjugate* when the posterior belongs to the same parametric family as the prior. Conjugacy makes some Bayesian updates analytic. PPCs use exponential-family structure because multiplying compatible distributions can become addition of their natural parameters.

  Three common approximation families are:

  - *Laplace approximation.* Fit a Gaussian around the posterior mode using local curvature. It is cheap but poor for strongly non-Gaussian or multimodal posteriors.
  - *Variational inference.* Optimize a tractable $q_phi(theta)$ to approximate the posterior.
  - *Markov chain Monte Carlo (MCMC).* Construct a Markov chain whose stationary distribution is the posterior. It can be asymptotically exact, but finite runs can be biased by burn-in and autocorrelation.

  The *Kullback–Leibler (KL) divergence* compares distributions:

  $ "KL"(q || p) = integral q(theta) log (q(theta))/(p(theta)) dif theta >= 0. $

  - $p(theta)$ is the target distribution.
  - $q(theta)$ is the approximation.
  - $theta$ is the variable being integrated over.

  The divergence is asymmetric. Minimizing $"KL"(q || p)$ is usually mode-seeking, while minimizing $"KL"(p || q)$ is usually mass-covering.

  === Population coding and information

  - *Tuning curve.* The mean response $f_i(theta)$ of neuron $i$ as a function of stimulus $theta$.
  - *Fisher information.* A measure of how precisely responses identify $theta$. Under regularity conditions, an unbiased estimator obeys the Cramér–Rao bound, $"Var"(hat(theta)) >= 1 / I_F(theta)$, where $hat(theta)$ is the estimator and $I_F(theta)$ is Fisher information.
  - *Linear Fisher information.* The information accessible to a linear decoder, which is a useful proxy for what a downstream neuron can extract through weighted synaptic input.
  - *Noise correlations.* Trial-to-trial covariance between neurons at fixed stimulus. Correlations aligned with the tuning gradient can cap information even as the population grows.
  - *Poisson variability.* A count model in which spike-count variance equals its mean. It is a common baseline rather than a universal biological law.

  === Sampling dynamics and E/I circuits

  - *Stationary distribution.* The distribution approached by a Markov process at long times. A neural-sampling claim requires the circuit's stationary distribution to match the intended posterior, not merely to display irregular activity.
  - *Langevin dynamics.* Noisy gradient dynamics designed to approach a target stationary distribution. They provide the simplest bridge from a log-posterior landscape to recurrent stochastic dynamics.
  - *Hamiltonian Monte Carlo (HMC).* A sampler that augments the represented variable with momentum, allowing longer, less diffusive moves through the target distribution. The Hamiltonian-brain account associates excitatory and inhibitory populations with complementary state variables.
  - *Inhibition-stabilized network (ISN).* A recurrent E/I network whose excitatory subnetwork is unstable in isolation and stabilized by feedback inhibition.
  - *Mixing time.* The time required for a chain to lose dependence on its initial state and explore the target distribution.
  - *Autocorrelation time.* The time over which successive samples remain correlated. Shorter autocorrelation generally means more effective samples per unit time.
  - *Transient amplification.* Temporary growth of selected activity patterns in an asymptotically stable recurrent network. It can accelerate movement through state space without requiring sustained instability.
  - *Δt-stability.* In this project, robustness of the simulated or trained dynamics to the numerical integration step $Delta t$. A purported computational regime that disappears under a smaller step may be a discretization artefact rather than a property of the continuous-time circuit.

  === Posterior readouts and metabolic cost

  - *Maximum a posteriori (MAP) estimate.* The posterior mode, $theta_"MAP" = arg max_theta p(theta | x)$.
  - *Posterior mean.* $bb(E)[theta | x]$, the Bayes-optimal point estimate under squared-error loss.
  - *Posterior variance or credible interval.* The width of the posterior, representing uncertainty rather than only a best estimate.
  - *Bits per spike.* Mutual information between stimulus and response divided by spike count, used as a measure of coding efficiency.
  - *Coding precision.* The inverse width of the represented distribution. A system can reduce precision to save energy while retaining a similar posterior mean.
  - *Metabolic cost of a spike.* The energetic cost, largely paid by ion pumps, of restoring ionic gradients after an action potential.

  #reference-list(refs)
]
