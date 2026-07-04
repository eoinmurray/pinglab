#let meta = (
  title: "Shadlen & Movshon (1999) — Synchrony Unbound: A Critical Evaluation of the Temporal Binding Hypothesis",
  date: "2026-06-25",
  description: "A reading of the rate-coding case against synchrony as a code: the flooded, balanced cortical neuron integrates a rate and cannot read coincidences.",
  collection: "literature",
)

#let body = [
  A reading of #link("https://doi.org/10.1016/S0896-6273(00)80822-3")[Shadlen & Movshon (1999), _Synchrony Unbound: A Critical Evaluation of the Temporal Binding Hypothesis_] (Neuron 24:67–77). It is the rate-coding skeptic's counterpart to the rhythm-and-synchrony story: where #link("/ar059/")[ar059] (Gerstner) and #link("/exp050/")[exp050] ask _when_ a population locks into a gamma rhythm, this paper asks whether that synchrony could ever be a _code_ — and argues, on a back-of-envelope count of cortical inputs, that it cannot. Its engine is the same flooded, balanced, high-input regime we simulate in #link("/exp058/")[exp058].

  *In plain English.* The _binding problem_: your visual cortex represents an object's pieces — edges, colours, motions — in scattered neurons, so how does the brain know they belong to one object? The _temporal binding hypothesis_ answers: the neurons coding one object fire _in synchrony_ (often riding a gamma rhythm), and that synchrony is the "we go together" tag. Shadlen & Movshon say this can't work in real cortex, for one blunt reason — a cortical neuron is buried under input. It receives thousands of synapses and _hundreds of spikes for every spike it emits_, so in any few-millisecond window a flood of inputs always arrives "together." If everything is synchronous, synchrony tags nothing: a neuron drowning in input integrates a _rate_, it cannot pick out a special coincident subset. They conclude cortex is a rate coder, and that binding is a higher-level, action-driven computation.

  == The binding problem and the temporal solution

  A scene must be assembled from local features computed by neurons with small receptive fields. The *binding problem* is how the brain marks which features belong to the same object — without a combinatorial explosion of dedicated "this-red-vertical-bar-here" neurons. The *temporal binding hypothesis* (von der Malsburg, 1981; Gray et al., 1989; Singer & Gray, 1995) proposes a dynamic tag: neurons representing one object *synchronise* their spikes — typically within a gamma cycle — while neurons coding different objects fire out of phase. Synchrony is the binding label, transient and reassignable, sidestepping the need for cardinal neurons.

  The paper's first objections are conceptual: the hypothesis names a _signature_ of binding (synchrony) without saying how binding is _computed_; and the perceptual and lesion evidence places object binding in high-level (parietal/frontal) cortex, not the early visual areas where the synchrony is recorded. But the argument that endures — and that ties to our work — is quantitative.

  == When synchrony could work: the sparse coincidence detector

  For synchrony to _mean_ something, a downstream neuron must act as a *coincidence detector* — fire when enough inputs arrive within a short window $tau$, and stay quiet otherwise. That requires chance coincidences to be rare. Take a neuron reading 10 inputs, each firing as a Poisson process at 10 spikes/s, so the ensemble rate is $R = 100$ spikes/s. The rate at which $m$ inputs fall within $tau$ of one another (an $m$-fold coincidence) is approximately

  $ r_"coinc" approx R ((R tau)^(m-1))/((m-1)!), quad (1) $

  where:
  - $R$ — total spike rate of the input ensemble;
  - $tau$ — coincidence window;
  - $m$ — number of near-simultaneous spikes the detector demands.

  For $m = 4$, $tau = 5$ ms, $R = 100$ spikes/s this gives a chance rate of ≈ 1 spike/s — low enough that a spike from such a detector genuinely signals "≥ 4 of the 10 inputs were active together." So coincidence detection _can_ carry information — *but only in a sparse, low-rate regime*, where few inputs arrive between the neuron's own spikes. And Shadlen & Movshon note the trap: in exactly that regime there is no binding problem to solve — with few effective inputs and low rates, there is nothing to tag, select among, or multiplex. The coincidence code works only where it is not needed.

  == The knockout: real cortex is flooded

  What regime is a cortical neuron actually in? The anatomy answers. A pyramidal cell receives 3,000–10,000 synapses, ≈ 85% excitatory; roughly 1,000 excitatory inputs come from the ≈ 1,000 neurons in its own local column (neuron density ≈ $10^5$/mm³, connection probability ≈ 0.09). The number that matters is the *effective excitatory input per output spike*,

  $ n_"in" = N_"eff" r_"in"/r_"out", quad (2) $

  where $N_"eff"$ is the effective count of excitatory inputs, $r_"in"$ their mean rate, and $r_"out"$ the cell's own rate. With $N_"eff" tilde 10^3$ and comparable in/out rates, a neuron integrates *several hundred excitatory spikes between one output spike and the next* (Shadlen & Newsome, 1994, 1998).

  Now the argument is immediate. With hundreds of input spikes per interspike interval, _any_ 5–10 ms window during active cortex contains a flood of inputs arriving in apparent synchrony — not because anything special happened, but because that is what a recurrent, highly convergent network does. So what could "synchronous" spikes signify to such a neuron? In effect, *all spikes are synchronous with other spikes*; there is no quiet background against which a coincident subset stands out. A neuron in this regime is forced to be an *integrator* — it sums its drenched input and reports a rate — not a coincidence detector that reads timing.

  == The precision escape, and why it fails

  One could rescue synchrony by demanding a much narrower window — sub-millisecond coincidences would be rare even in the flood (set $tau ->$ 1 ms in equation 1 and $r_"coinc"$ collapses). Shadlen & Movshon close that door twice. First, *biophysics*: unlike specialised auditory-brainstem neurons, cortical cells lack the machinery for millisecond coincidence detection (Reyes et al.; Koch, 1999) — their membrane time constants and dendritic integration smear inputs over many milliseconds. Second, *the data*: reported "synchrony" is broad — correlogram peaks of 10–20 ms, sometimes 50 ms — far coarser than the precision a usable timing code would need. The window where synchrony is both rare-by-chance and biophysically readable does not coincide with the window where cortex actually correlates.

  == Their alternative: rate, and binding through action

  The positive proposal is deflationary. Rate-modulated population activity already does the perceptual work; no new kind of signal is needed. And binding, properly construed, is a *high-level, action-oriented* operation: in parietal cortex (e.g. LIP), stimuli are co-represented to the extent they bear on a common action or saccade — a priority/saliency map. "Features are bound together to the extent that one feature can be viewed as an instruction to act in some way upon another." This tames the combinatorial explosion (groupings are constrained by possible actions) and fits the neurology of parietal damage. Their verdict: the temporal binding hypothesis _as proposed_ is untenable — a signature without a mechanism, at the wrong level, biophysically undecodable, and weakly supported — though it productively forced the field to take neural timing seriously.

  == The tie to the balanced state

  Here is why this 1999 polemic belongs next to our balanced-network work. *The knockout argument is the same physics as exp058.* "Several hundred balanced excitatory and inhibitory inputs per output spike, fluctuation-driven, irregular firing" is precisely the Vreeswijk–Sompolinsky asynchronous-irregular state — the regime where each cell's net drive is a near-cancelling sum whose residual fluctuations decide its spikes (CV ≈ 1). The two literatures read the _same_ regime through different lenses:

  - *V&S / exp058* asks _why_ that state is self-consistent and finds it is deterministically chaotic — the network manufactures its own noise.
  - *Shadlen & Movshon* take that flooded, irregular regime as given and draw the _coding_ moral: a neuron there integrates a rate; it cannot read synchrony.

  So exp058 supplies the mechanism beneath their argument. The same picture threads the rest of the #link("/ar008/")[reading list]: #link("https://doi.org/10.1038/nature09086")[London et al. (2010)] shows a single spike perturbs cortical activity — high sensitivity, which they read as high noise and hence rate coding — and our exp058 chaos result (a positive Lyapunov exponent, a perturbation that amplifies) is the _source_ of that sensitivity. #link("https://doi.org/10.1126/science.1179850")[Renart et al. (2010)] is the asynchronous state measured directly.

  And the debate is not settled by fiat. This paper is the *rate* pole; #link("https://doi.org/10.1016/j.neuron.2015.09.034")[Fries' Communication-through-Coherence] is the modern *timing/rhythm* pole, where gamma coherence gates _which_ signals get through. Our own pair of entries refuses the dichotomy: #link("/exp058/")[exp058] is the integrator/asynchronous regime Shadlen & Movshon favour, #link("/exp050/")[exp050] is PING gamma — the synchrony regime the binding camp favours — and they are the _same network_ on either side of one control parameter (the correlation of the input). The interesting question our model poses is not "rate or timing?" but "what makes cortex sit on one side of that bifurcation, and can it switch?"

  == Honest caveats

  It is a 1999 polemic, and worth reading as one. Its target is the _strong_ binding-by-synchrony claim, and it lands; but precise spike timing does carry information in some systems and tasks, gamma coherence does modulate effective connectivity in later work, and "rate vs timing" is now understood as a spectrum rather than a verdict. The paper's durable contribution is not "synchrony is dead" — it is the *integrator argument*: a quantitative, anatomy-grounded reason that a balanced, high-input cortical neuron reports rates. That is exactly the cortex we build in exp058, which is why the argument has outlived the debate that prompted it.

  == One-paragraph summary

  Counting cortical inputs settles the matter: a pyramidal neuron integrates several hundred balanced spikes per output spike (equation 2), so in any few-millisecond window inputs are always "synchronous" by design — synchrony cannot stand out as a code. Coincidence detection works only in a sparse, low-rate regime (equation 1) where there is no binding problem to solve, and cortex lacks both the biophysics and the measured precision for a millisecond timing code anyway. The conclusion — cortex rate-codes, binding is high-level — rests on the same flooded, fluctuation-driven balanced state that exp058 simulates and V&S explains. Two lenses on one regime: V&S asks why it is chaotic; Shadlen & Movshon ask what it can compute.
]
