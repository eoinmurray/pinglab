#let meta = (
  title: "Weekly Feed — 2 July 2026",
  date: "2026-07-02",
  description: "A thinner week's triage of gamma-gated-sparsity-relevant papers: mean-field oscillations, criticality-aware pruning, CA1 theta–gamma, and the older work they build on.",
  collection: "weekly-feed",
)

#let paper(title, href, meta, body) = block(breakable: false, below: 1.2em)[
  #link(href)[*#title*] #h(0.4em) #text(size: 9pt, fill: gray)[#meta]

  #body
]

#let body = [
  A weekly triage of papers relevant to the gamma-gated-sparsity project — one entry per week, deduped against prior weeks. *New papers* are this week's catch from the fetcher (arXiv, bioRxiv, journal tables of contents, PubMed, and Semantic Scholar recommendations seeded from the project bibliography) — all published _after_ the January 2026 knowledge cutoff, so genuinely new. *Old papers* are established work from _before_ the cutoff that this week's papers build on, surfaced from background knowledge as context rather than from the fetch. Sources are configured in src/docs/scripts/paperfeed/sources.ts.

  This was a thinner week than the last one: the fetch skewed clinical and molecular (deep-brain-stimulation trials, single-cell atlases, metabolic optimisers), and the keyword scorer over-weighted _energy_ and _metabolic_ on papers that have nothing to do with rhythmic spiking. After stripping those out, four candidates are genuinely load-bearing — one of them squarely on the mean-field and chaos questions still open in #link("/nb058/")[nb058].

  == New papers

  Published after the cutoff — caught this week from the feeds and recommendations.

  #paper(
    "Mean-field theory of rich oscillatory dynamics in low-rank recurrent networks with activity-dependent adaptation",
    "https://arxiv.org/abs/2606.30366",
    "arXiv q-bio.NC · Jun 2026",
  )[
    This is the standout of the week. The authors build a dynamical mean-field theory for random recurrent networks that carry a low-rank structural component and a firing-rate-driven adaptation current. As adaptation strength increases, the network walks through four regimes — a static coherent state, noise-sustained oscillations that go from regular to irregular, stochastic switching between two symmetric wells, and finally a clean global limit cycle. Crucially, they separate two distinct instabilities that both produce complex activity: chaos onset driven by the strong _random_ connectivity, and a _Hopf bifurcation_ of the coherent (mean) mode. Adaptation shapes both through the frequency-dependent single-neuron transfer function, and the whole thing collapses to a reduced three-dimensional system. It is a rare paper that treats "chaotic irregularity" and "coherent oscillation" as separable mechanisms with their own onsets rather than lumping them together.

    *Relevance:* the closest methodological neighbour to our own analysis this week. The Hopf-of-the-coherent-mode is exactly the object behind our gamma onset (#link("/ar009/#22-gamma-onset-across-the-weiwiewei-times-wieweiwie-plane")[§2.2]) and the #link("/ar009/#53-mean-field-reduction")[4D mean-field reduction]. More pointedly, their clean split between _chaos-from-random-connectivity_ and _Hopf-from-the-coherent-mode_ is the language we need for the unresolved #link("/nb058/")[nb058] thread — where a stable integrator suppressed the chaotic ($sqrt(K)$) regime while leaving the coherent rhythm intact. This paper says those are different bifurcations; that is a testable claim against our integrator choice.
  ]

  #paper(
    "Criticality-Constrained Iterative Pruning for Energy-Efficient Spiking Neural Networks via Combined Importance Scoring",
    "https://arxiv.org/abs/2606.30676",
    "arXiv cs.NE · Jun 2026",
  )[
    Pruning a spiking network for neuromorphic hardware is combinatorial, and the usual convex-relaxation tricks produce fractional connection masks that fall apart when you binarise them at high sparsity. This paper fuses two importance signals — plain weight magnitude and a surrogate-gradient measure of each neuron's _criticality_ — into a single analytically exact score, then prunes iteratively without ever solving the relaxed problem, sidestepping the rounding artefacts. The framing treats sparsity and energy as the primary design currency and asks how much connectivity you can strip before temporal computation degrades. It is a learned, subtractive route to sparsity: start dense, remove what a criticality score says is dispensable.

    *Relevance:* the clean methodological contrast to our sparsity story. They _arrive at_ sparsity by pruning a trained network; we _start_ sparse because the gamma gate lets each cell fire ≈ once per cycle — the sparsity is supplied by the architecture, not carved out afterwards (#link("/ar009/#24-the-firing-rate-reduction-does-not-require-trained-loop-weights")[§2.4]). Their surrogate-gradient criticality metric is also a candidate lens on _which_ connections in our PING loop are load-bearing for the ≈ 10× spike reduction (#link("/ar009/#23-trained-ping-attains-coba-accuracy-at-10approx-10times10-fewer-spikes")[§2.3]).
  ]

  #paper(
    "A reduced multicompartment network model of CA1 theta–gamma oscillations under extracellular stimulation",
    "https://doi.org/10.64898/2026.06.22.733913",
    "Preprint (Semantic Scholar) · Jun 2026",
  )[
    A reduced multicompartment model of hippocampal CA1 that reproduces theta-nested gamma — the fast gamma rhythm riding on the slower theta cycle — and then asks how extracellular (deep-brain-stimulation-like) fields perturb the phase-amplitude coupling between the two. The motivation is clinical, aimed at the disrupted theta-gamma coupling seen in Alzheimer's, but the modelling contribution is a tractable circuit that keeps enough biophysics to place the gamma generator while staying small enough to sweep stimulation protocols. It sits on the interneuron-gamma / cross-frequency-coupling side of the rhythm literature rather than the PING side.

    *Relevance:* a contrast case for how gamma is generated. Their gamma is nested inside theta and shaped by an external field; ours is a stand-alone rhythm generated by the E→I→E loop (#link("/ar009/#22-gamma-onset-across-the-weiwiewei-times-wieweiwie-plane")[§2.2]). Useful mainly as a reminder that "a reduced model of gamma" need not mean a PING model — and as a pointer to cross-frequency coupling, a direction our single-band model does not yet touch.
  ]

  #paper(
    "Data-Constrained Recurrent Network Neural Model Uncovers the Circuit Mechanism of Olfactory OFF Responses",
    "https://doi.org/10.64898/2026.05.22.727331",
    "Preprint (Semantic Scholar) · May 2026",
  )[
    Projection neurons in the insect antennal lobe fire a transient burst when an odour _ends_ — an OFF response — and the circuit origin of that has been unclear. The authors fit a biologically constrained firing-rate recurrent network directly to in-vivo recordings from 110 projection neurons across five odorants, reproduce the measured temporal dynamics, and then read the mechanism off the trained connectivity: the OFF response falls out of recurrent excitatory-inhibitory interactions rather than any single-cell property. It is a worked example of using a trained, data-constrained E–I recurrent network as a hypothesis-generator for a specific circuit computation.

    *Relevance:* company for the trained-recurrent-E/I corner of the design space we occupy, from the systems-neuroscience side rather than the machine-learning side. The methodology — constrain an E–I recurrent net, train it, then interrogate the connectivity for mechanism — is the same move we make when we ask whether the #link("/ar009/#24-the-firing-rate-reduction-does-not-require-trained-loop-weights")[firing-rate reduction survives without trained loop weights]. Their answer (mechanism lives in the recurrence, not the cells) rhymes with ours.
  ]

  == Old papers

  Published before the cutoff — established work this week's papers build on, surfaced as background.

  #paper(
    "Sompolinsky, Crisanti & Sommers (1988) — Chaos in random neural networks",
    "https://doi.org/10.1103/PhysRevLett.61.259",
    "Phys. Rev. Lett. · 1988",
  )[
    The founding result of dynamical mean-field theory for neural networks. For a fully connected random network with strong enough coupling, the authors show — analytically, via a self-consistent field derived from the disorder-averaged dynamics — that the network undergoes a sharp transition from a silent fixed point to a state of sustained, deterministic _chaos_. The order parameter is the coupling gain; past a critical value the trivial state destabilises and autonomous chaotic fluctuations appear with no external noise. Every later mean-field treatment of random recurrent dynamics, including this week's low-rank-plus-adaptation paper, is a descendant of this calculation.

    *Background for:* _Mean-field theory of rich oscillatory dynamics in low-rank recurrent networks_ above — its "chaos onset from the random connectivity" is precisely this 1988 transition, now running alongside a coherent-mode Hopf. It is also the reference state behind the chaotic ($sqrt(K)$) regime we chased in #link("/nb058/")[nb058].
  ]

  #paper(
    "Mastrogiuseppe & Ostojic (2018) — Linking connectivity, dynamics, and computations in low-rank recurrent neural networks",
    "https://doi.org/10.1016/j.neuron.2018.07.003",
    "Neuron · 2018",
  )[
    This paper established the low-rank recurrent network as a tractable object: split the connectivity into a strong random part plus a low-rank structured part, and the structured part carries the computation while the random part sets the dynamical regime. Using mean-field theory they show how a rank-one or rank-two component reshapes the network's fixed points, oscillations, and chaotic activity, and how those in turn support concrete computations. It is the framework that makes "low-rank recurrent network" a phrase with a precise dynamical meaning rather than a modelling convenience.

    *Background for:* the low-rank half of this week's _Mean-field theory of rich oscillatory dynamics_ paper. The structured-plus-random decomposition it introduced is exactly the setting that paper extends by adding activity-dependent adaptation.
  ]

  #paper(
    "Börgers & Kopell (2003) — Synchronization in networks of excitatory and inhibitory neurons with sparse, random connectivity",
    "https://doi.org/10.1162/089976603321192059",
    "Neural Comput. · 2003",
  )[
    The canonical theory of PING — pyramidal-interneuron network gamma. Börgers and Kopell work out the conditions under which a network of excitatory and inhibitory cells with sparse random connectivity locks into a coherent gamma rhythm: the excitatory cells recruit the interneurons, the resulting inhibition silences the network for a GABA time constant, and the cycle repeats. They pin down when this E↔I loop produces tight synchrony versus when it fails, and how drive and connection density set the boundary. This is the mechanistic backbone our whole model rests on — gamma as a property of the E→I→E loop, not of interneurons alone.

    *Background for:* _A reduced multicompartment model of CA1 theta–gamma_ above (the interneuron-gamma contrast) and, more broadly, for our #link("/ar009/#22-gamma-onset-across-the-weiwiewei-times-wieweiwie-plane")[gamma onset diagram]. The E→I→E synchronisation logic it formalises is the exact mechanism whose onset we sweep across the $W^(E I) times W^(I E)$ plane.
  ]

  #paper(
    "Frankle & Carbin (2019) — The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks",
    "https://arxiv.org/abs/1803.03635",
    "ICLR · 2019",
  )[
    The paper that reframed pruning. Frankle and Carbin show that a dense network contains a sparse subnetwork — a "winning ticket" — which, trained in isolation from the same initialisation, matches the full network's accuracy. Sparsity, on this view, is not a lossy compression applied after the fact but a structure that was trainable all along and merely had to be found. It launched the modern literature on when and how aggressively a network can be pruned without losing what it learned.

    *Background for:* this week's _Criticality-Constrained Iterative Pruning_ paper, which is a spiking, criticality-aware instance of exactly this find-the-sparse-subnetwork program. It also sharpens our own framing by contrast: a lottery ticket is a sparse subnetwork _discovered by pruning_, whereas our sparsity is _imposed by the gamma gate_ before any training (#link("/ar009/#24-the-firing-rate-reduction-does-not-require-trained-loop-weights")[§2.4]).
  ]

  #v(1em)

  _Not shown: ≈ 140 lower-relevance candidates this week (clinical DBS, single-cell atlases, metabolic optimisers, off-topic ML). A leaner haul than last week — worth watching whether the energy/metabolic keyword weighting in the scorer keeps surfacing noise. Next week gets its own dated entry, deduped against this one (bun run feed, from src/docs, then curate). To change coverage, edit src/docs/scripts/paperfeed/sources.ts; per-week candidate sets are archived under src/docs/scripts/paperfeed/archive/._
]
