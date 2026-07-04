#let meta = (
  title: "Bayesian Fundamentals",
  date: "2026-06-28",
  description: "A short primer fixing the notation and vocabulary the neural-Bayesian reading list leans on: Bayes' rule, marginalisation, conjugacy, VI/MCMC, KL, and posterior summaries.",
  collection: "miscellaneous",
)

#let body = [
  A short primer for the vocabulary the neural-Bayesian papers in the #link("/ar007/")[Bayesian Literature Review] lean on without redefining. The aim is not to teach Bayesian statistics from scratch — it is to fix notation, name the moves, and flag where the hard parts are, so a reader recognises each term on contact when they hit it in Ma, Fiser, Lengyel, or Echeveste.

  == The core equation

  Given a generative model — a forward story for how unobserved causes $theta$ produce observations $x$ — Bayes' rule inverts the story:

  $ p(theta | x) = (p(x | theta) p(theta))/(p(x)) prop p(x | theta) p(theta) quad (1) $

  - $theta$ — the latent (unobserved) cause to be inferred (a tilt angle, a phoneme, a depth);
  - $x$ — the observation (the retinal image, the cochlear input);
  - $p(theta)$ — the *prior*, what was believed about $theta$ before seeing $x$;
  - $p(x | theta)$ — the *likelihood*, how probable this $x$ would be if $theta$ were the cause;
  - $p(theta | x)$ — the *posterior*, the updated belief about $theta$ after seeing $x$;
  - $p(x) = integral p(x | theta) p(theta) dif theta$ — the *evidence* (or marginal likelihood), the normaliser.

  The proportionality at the right is doing most of the work in practice: every algorithm in the rest of this article exists to avoid computing $p(x)$ exactly.

  == Generative model vs recognition

  A *generative model* is the brain's (or the modeller's) story about how the world produces the data: causes $->$ observations. *Recognition* (or *inference*) is the reverse — recover $p(theta | x)$ from $x$. These are different computations and the brain need not implement them the same way; a sensory area can run a fast amortised recognition map even if the underlying generative model is much richer. When neural-Bayesian papers say "the cortex represents the posterior", they mean the recognition output.

  == Marginalisation, and why it's the hard part

  Anything you don't care about you must integrate out:

  $ p(theta_1 | x) = integral p(theta_1, theta_2 | x) dif theta_2 quad (2) $

  - $theta_1$ — the quantity of interest (the depth of a surface);
  - $theta_2$ — the nuisance variables (lighting, surface reflectance) you marginalise over.

  For high-dimensional $theta$ this integral is intractable. Every approximate-inference algorithm below is, at heart, a different way to dodge it.

  == Conjugacy and the exponential family

  A likelihood–prior pair is *conjugate* if the posterior has the same parametric form as the prior. Conjugate pairs (Gaussian–Gaussian, beta–Bernoulli, Dirichlet–multinomial) give closed-form posterior updates and are the only place Bayes is genuinely cheap. They live inside the *exponential family*:

  $ p(x | eta) = h(x) exp(eta^top T(x) - A(eta)) quad (3) $

  - $eta$ — the *natural parameters*;
  - $T(x)$ — the *sufficient statistics* (the only function of $x$ the posterior cares about);
  - $A(eta)$ — the *log-partition* (normaliser), whose derivatives give the moments of $T$;
  - $h(x)$ — a base measure.

  This matters for cortex because the Probabilistic Population Code framework (Ma et al. 2006, in #link("/ar007/")[ar007]) writes posteriors as exponential families whose natural parameters are population firing rates: $eta eq.triple bold(r)$. Multiplying two PPCs (combining cues) becomes adding their firing rates — the framework's central, very pretty claim.

  == Three flavours of approximate inference

  When the posterior is not conjugate, you approximate. Three families dominate; the neural-Bayesian papers each pick one:

  - *Laplace approximation.* Find the posterior mode $hat(theta)$, fit a Gaussian whose covariance is $(-nabla^2 log p(theta | x))^(-1)$ at the mode. Cheap, local, breaks down when the posterior is multimodal.
  - *Variational inference (VI).* Pick a tractable family $q_phi (theta)$ and minimise $"KL"(q_phi (theta) || p(theta | x))$ in $phi$. Turns inference into optimisation. Fast, scalable, but biased — the bias is whatever $p$ has that $q$ cannot represent.
  - *Markov chain Monte Carlo (MCMC).* Construct a Markov chain whose stationary distribution _is_ the posterior, then run it. Unbiased in the limit, slow to converge. Variants: Metropolis–Hastings, Gibbs sampling, Langevin dynamics (gradient + noise), Hamiltonian Monte Carlo (gradient + momentum). The Lengyel-group "sampling cortex" papers in #link("/ar007/")[ar007] argue that cortical dynamics are a biological MCMC chain — Aitchison & Lengyel cast E/I circuits as a Hamiltonian sampler, with the inhibitory population playing the momentum role.

  == KL divergence

  The asymmetric "distance" between two distributions:

  $ "KL"(q || p) = integral q(theta) log (q(theta))/(p(theta)) dif theta >= 0, quad "with equality iff " q = p. quad (4) $

  Asymmetric because $"KL"(q || p) != "KL"(p || q)$ in general. VI minimises $"KL"(q || p)$ (forward KL) which makes $q$ _mode-seeking_ (it concentrates on a single mode rather than averaging across modes); some methods minimise $"KL"(p || q)$ (reverse KL) which makes $q$ _mass-covering_. The choice shows up in what the approximate posterior looks like.

  == Posterior summaries (the things a brain might actually read out)

  A posterior is a distribution; downstream computation usually wants a number or two. The standard summaries:

  - *MAP estimate.* The mode: $theta_"MAP" = arg max_theta p(theta | x)$. A single best guess.
  - *Posterior mean.* $bb(E)[theta | x]$. The Bayes-optimal point estimate under squared-error loss.
  - *Posterior variance / credible interval.* The width — how _uncertain_ the inference is. The whole point of being Bayesian rather than maximum-likelihood; if downstream computation needs to know "should I bet on this?" it needs the spread, not just the mean.

  The neural-Bayesian split is partly about how the cortex represents this last one — as a population gain (PPC) or as trial-to-trial variability (sampling).

  == How this lands in the cortex

  Three claims recur in the reading list, in roughly this order of strength:

  + *Cortical responses look Bayesian behaviourally.* Cue integration, prior-weighted perception, multisensory fusion — all match Bayes-optimal predictions in many tasks. The papers in #link("/ar007/")[ar007] take this as given.
  + *The cortical _representation_ of probability is one of two flavours.* Either parametric (firing rates encode posterior parameters — PPC) or sample-based (instantaneous activity is itself a sample). These are different empirical predictions about neural variability, and the camps disagree.
  + *The cortical _dynamics_ implement the inference algorithm.* The sampling camp claims recurrent E/I circuits run Langevin or HMC; the PPC camp claims linear combinations of population activity perform the algebra. Echeveste et al. 2020 is the cleanest version of the dynamical claim.

  This article stops short of those claims — see #link("/ar007/")[ar007] for the reading list that develops them.
]
