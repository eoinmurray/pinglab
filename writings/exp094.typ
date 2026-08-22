#let meta = (
  title: "What does a decoder believe?",
  date: "2026-08-22",
  description: "An illustrated guide to temporal evidence and output mappings for frozen MNIST COBA and PING networks.",
  collection: "gamma-gated-sparsity",
  status: "complete",
)

#let body = [
  == Introduction

  A decoder converts neural activity into a class decision. Here the learned output projection remains frozen while two later choices change: $z$ decides what temporal activity counts as evidence, and $p$ decides how that evidence is displayed as relative confidence.

  One preselected MNIST digit and one Poisson spike tensor were replayed through the canonical seed-42 COBA and PING final-epoch checkpoints represented in the latest gold-star publication view. These are the full-MNIST reference networks, not exp022's 10%-MNIST activity-frontier (`off`) condition. Their trained decoder is `mem-mean`: a non-spiking output membrane whose temporal mean supplies the class logits. To compare spike-based alternatives without retraining, the same frozen hidden activity and learned output weights also drive a counterfactual output-LIF `spike-count` decoder at inference time. Every spike-based $z$ reads that one shared intervention trajectory, and every $p$ reads the same cumulative counts. The comparison therefore explains decoder choices; it does not estimate the accuracy each alternative would achieve if trained end to end. Each subsection pairs a qualitative mechanism diagram with its measured mirror using the same visual language.

  #figure(
    image("/artifacts/data/exp094/decoder_pipeline.svg", width: 100%, alt: "Frozen COBA and PING activity passing through temporal evidence, output mapping, and decision stages."),
    caption: [Decoder decomposition. Frozen COBA and PING networks produce output spikes and voltages. A temporal evidence function $z$ compresses that activity, an output mapping $p$ displays the resulting competition, and the decision selects the largest final score.],
  )

  == Temporal evidence functions z

  Each subsection changes only $z$. Red denotes the true class, black its strongest competitor, and time runs horizontally. Mean voltage uses the native non-spiking readout. The four spike-based alternatives use the single counterfactual output-LIF trajectory driven by the same frozen hidden spikes and output weights.

  === Mean output voltage

  $ z_c^"mean"(t) = 1/(t + 1) sum_(tau=0)^t v_c^"pre"(tau). $

  Here $c$ identifies a class, $t$ is the current timestep, $tau$ indexes earlier timesteps, and $v_c^"pre"(tau)$ is pre-reset output voltage. Mean voltage retains subthreshold evidence but removes temporal order. It is the decoder used to train the canonical exp022 COBA and PING checkpoints.

  #figure([
    #image("/artifacts/data/exp094/z_mean.svg", width: 100%, alt: "Illustrative output voltages becoming mean-voltage evidence.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_mean.svg", width: 100%, alt: "Measured running mean-voltage evidence for frozen COBA and PING trajectories.")
  ], caption: [Mean-voltage decoder. The diagram shows pre-reset voltages entering a running temporal mean; the measured mirror shows the resulting frozen COBA and PING trajectories.])

  The native decoder selects the true class in both networks. COBA settles rapidly, whereas PING retains a visible early competition before the true-class mean becomes consistently larger.

  === Cumulative spike count

  $ z_c^"cum"(t) = sum_(tau=0)^t s_c(tau). $

  Here $s_c(tau)$ is one when output neuron $c$ spikes at timestep $tau$ and zero otherwise. Every spike remains evidence until reset. Timing and order disappear; only accumulated count survives.

  #figure([
    #image("/artifacts/data/exp094/z_cumulative.svg", width: 100%, alt: "Illustrative output spikes becoming cumulative staircase evidence.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_cumulative.svg", width: 100%, alt: "Measured cumulative output-spike evidence for frozen COBA and PING trajectories.")
  ], caption: [Cumulative-count decoder. Each output spike increments one class trace and no evidence decays; the staircase exposes early leads and later reversals.])

  The counterfactual spike decoder changes the outcome. PING eventually gives the true class the largest cumulative count, while COBA does not. Frozen weights therefore do not make a new temporal evidence rule equivalent to the rule used during training.

  === Leaky spike count

  $ z_c^"leak"(t) = lambda z_c^"leak"(t - 1) + s_c(t), quad 0 < lambda < 1. $

  Here $lambda$ is the fraction of previous evidence retained for one timestep. Spikes add evidence; between spikes it decays. The decoder can follow changing input, but its meaning depends on the chosen memory timescale.

  #figure([
    #image("/artifacts/data/exp094/z_leaky.svg", width: 100%, alt: "Illustrative output spikes becoming decaying leaky evidence.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_leaky.svg", width: 100%, alt: "Measured leaky spike evidence for frozen COBA and PING trajectories.")
  ], caption: [Leaky-count decoder. Spikes create upward steps and evidence decays smoothly between them, using the same leak factor for COBA and PING.])

  Leakage exposes recent differences in PING rather than preserving its entire history. COBA's much denser counterfactual output keeps both displayed classes near a steady ceiling, so forgetting adds little useful separation.

  === Sliding-window spike count

  $ z_c^"win"(t) = sum_(tau=max(0, t - W + 1))^t s_c(tau). $

  Here $W$ is the number of timesteps in the window. A spike contributes while it remains inside the window and disappears at the trailing boundary. Memory is finite and explicit rather than gradual.

  #figure([
    #image("/artifacts/data/exp094/z_window.svg", width: 100%, alt: "Illustrative time window selecting which output spikes remain evidence.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_window.svg", width: 100%, alt: "Measured sliding-window spike evidence for frozen COBA and PING trajectories.")
  ], caption: [Sliding-window decoder. Evidence persists for 25 ms, then disappears abruptly when its spike leaves the window.])

  The finite window makes PING's changing local competition explicit. COBA again remains saturated: nearly every window is full, showing that a memory rule cannot rescue an output representation whose event rate is poorly matched to it.

  === Cycle or matched-bin voting

  $ z_c^"vote"(K) = sum_(k=1)^K 1_(c = arg max_j n_(j,k)). $

  Here $K$ is the number of completed bins, $k$ identifies one bin, $j$ indexes classes, and $n_(j,k)$ is class $j$'s spike count inside bin $k$. The subscripted one is one when its condition is true and zero otherwise. PING bins follow measured cycles; COBA uses equal-duration controls. Each bin contributes one vote, discarding within-bin burst size.

  #figure([
    #image("/artifacts/data/exp094/z_vote.svg", width: 100%, alt: "Illustrative interval winners becoming cumulative class votes.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_vote.svg", width: 100%, alt: "Measured cumulative cycle or matched-bin votes for frozen COBA and PING trajectories.")
  ], caption: [Cycle or matched-bin voting. Each interval names one winner before adding one vote. PING votes follow detected cycles and COBA votes use duration-matched controls.])

  PING's true class wins more detected cycles and pulls away in cumulative votes. The matched COBA bins consistently favour another class. Cycle voting is therefore meaningful here because PING supplies distinct rhythmic evidence packets, not because voting itself guarantees a better decision.

  == Output mappings p

  #block(inset: 10pt, fill: rgb("f3f0e8"), radius: 3pt)[
    *Controlled comparison.* All three $p$ mappings receive the identical ten-class cumulative-count vector

    $ bold(z)^"cum"(t) = (z_0^"cum"(t), dots, z_9^"cum"(t)). $

    The counterfactual output-LIF trajectory is generated once from the shared frozen hidden drive, and $bold(z)^"cum"(t)$ is calculated once. It is then passed unchanged to ordinary softmax, temperature-softened softmax, and independent sigmoid. We do not select or recompute a different $z$ for any $p$. Therefore, every difference between these three displays is caused by $p$ alone.
  ]

  Here $bold(z)^"cum"(t)$ is the ten-element cumulative-count vector at time $t$, and its entries $z_0^"cum"(t)$ through $z_9^"cum"(t)$ correspond to MNIST classes zero through nine.

  === Ordinary softmax

  *Fixed input to $p$:* the shared cumulative-count vector $bold(z)^"cum"(t)$.

  $ p_c(t) = frac(exp(z_c^"cum"(t)), sum_j exp(z_j^"cum"(t))). $

  Here $p_c(t)$ is class $c$'s displayed share, $exp$ is the exponential function, and $j$ ranges over all ten classes. Scores sum to one and preserve the winner. Count differences are exponentially sharpened.

  #figure([
    #image("/artifacts/data/exp094/p_softmax.svg", width: 100%, alt: "Illustrative cumulative counts becoming ordinary-softmax shares.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_p_softmax.svg", width: 100%, alt: "Measured ordinary softmax of cumulative counts for frozen COBA and PING trajectories.")
  ], caption: [Ordinary softmax applied to cumulative spike counts. The measured mirror shows how an accumulating count margin becomes apparent confidence.])

  Softmax hides absolute activity and displays only count margins. PING's early leader changes produce abrupt transfers of displayed share before the true class dominates; COBA's different final winner remains unchanged by the mapping.

  === Temperature-softened softmax

  *Fixed input to $p$:* the same $bold(z)^"cum"(t)$ used by ordinary softmax.

  $ p_c(t; T) = frac(exp(z_c^"cum"(t) / T), sum_j exp(z_j^"cum"(t) / T)), quad T > 1. $

  Here $T$ is temperature. Dividing all counts by the same positive $T$ preserves the winner but reduces visual sharpening. This separates accumulated evidence from its apparent decisiveness.

  #figure([
    #image("/artifacts/data/exp094/p_softened.svg", width: 100%, alt: "Illustrative cumulative counts becoming temperature-softened shares.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_p_softened.svg", width: 100%, alt: "Measured temperature-softened softmax of the same cumulative counts.")
  ], caption: [Temperature-softened softmax applied to the same counts. It differs from ordinary softmax only in temperature, isolating the effect of $p$.])

  Temperature preserves every leader change and the final winner while making the competing PING classes easier to see. It changes how decisive the same evidence appears, not which evidence was accumulated.

  === Independent sigmoid

  *Fixed input to $p$:* the same $bold(z)^"cum"(t)$ used by both softmax mappings.

  $ p_c^"sig"(t) = 1 / (1 + exp(-z_c^"cum"(t))). $

  Here $p_c^"sig"(t)$ is an independent bounded score. One class does not suppress the others and the ten scores need not sum to one. Since cumulative counts are non-negative and grow, several classes may approach one together. This exposes why independent sigmoid is poorly matched to mutually exclusive MNIST labels when applied directly to raw counts.

  #figure([
    #image("/artifacts/data/exp094/p_sigmoid.svg", width: 100%, alt: "Illustrative cumulative counts becoming independent sigmoid scores that rise together.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_p_sigmoid.svg", width: 82%, alt: "Measured independent sigmoid scores from the same cumulative counts.")
  ], caption: [Independent sigmoid applied to cumulative spike counts. It retains the same $z$ used by both softmax variants and reveals what is lost when classes no longer compete.])

  Because every cumulative count grows, all ten sigmoid scores saturate near one almost immediately. The mapping preserves neither useful relative separation nor a meaningful class decision; its failure follows from applying an independent bounded transform to non-negative accumulating counts.
]
