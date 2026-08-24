#import "/.demolab/lib.typ": cite, reference-list

#let meta = (
  title: "What does gamma timing buy?",
  date: "2026-08-23",
  description: "Separate temporal information, native readout use, and resource-dependent benefit in frozen COBA and PING representations.",
  collection: "gamma-gated-sparsity",
  status: "ExpScout",
)

#let body = [
  == Abstract

  Gamma rhythms create periodic windows for neuronal firing and can shape sensory responses and signal transmission#cite(1, 2). Whether those windows carry an operative code or merely regulate activity remains unresolved. We compare frozen COBA and PING networks trained on rate-Poisson MNIST and change only how much hidden-spike timing is retained. Three measurements follow in order: what timing makes externally decodable, what timing the trained output circuit actually uses, and whether that timing permits decisions with less time or fewer spikes. The analysis varies temporal resolution continuously rather than choosing between a rate code and a temporal code.

  == Introduction

  A hidden response is a binary array $H$, with one row per excitatory neuron and one column per simulation timestep. Entry $H_(i t)$ is one when neuron $i$ spikes at time $t$ and zero otherwise. The complete array contains every hidden spike, but a decoder need not read it at full resolution.

  Here one number, the bin width $Delta$, decides how much timing survives. Divide the 200 ms presentation into adjacent windows of duration $Delta$ and count each neuron's spikes inside each window:

  $ z_(i b)^(Delta) = sum_(t in [b Delta, (b + 1) Delta)) H_(i t). $

  Here $i$ identifies a hidden E neuron, $b$ identifies a time bin, and $t$ indexes simulation timesteps. At $Delta = 200$ ms, each neuron contributes one presentation-wide count and all temporal order disappears. At 50 ms there are four counts per neuron; at 25 ms there are eight; at 5 ms there are forty. Narrower bins preserve finer timing.

  This gives the experiment one common control knob:

  $ "presentation count" arrow.r.long z^(Delta) arrow.r.long "fine timing". $

  The three questions are deliberately different. An external probe may decode timing that the trained network ignores. The trained readout may use timing without gaining robustness from it. Only the final comparison asks what computational advantage timing provides.

  == Shared activity

  #block(inset: 10pt, fill: rgb("f3f0e8"), radius: 3pt)[
    *Nothing in this experiment has been run.* Every curve below is a qualitative design schematic, not data. Measured figures will replace the schematic curves while preserving their axes and visual language.
  ]

  Use the canonical COBA and PING final-epoch checkpoints produced by #link("/exp022/")[exp022] and characterised by #link("/exp025/")[exp025]. Freeze every parameter. Present identical MNIST images and identical Poisson input tensors to each paired checkpoint, including every available independently trained network seed. Record the 1,024 hidden E spike trains and the native output state.

  External probes use the original training partition for fitting, a fixed validation partition for selection, and the untouched official test partition once for final evaluation. Before accepting any result, a compact validation table must confirm checkpoint provenance, paired trial identity, duration, neuron count, firing rate, native accuracy, and exact reproduction of the unaltered output replay.

  == Methods

  + *Information available at resolution $Delta$.*

    Give an external digit decoder the same hidden activity at successively narrower bin widths:

    $ Delta in {200, 100, 50, 25, 20, 10, 5, 2, 1} " ms". $

    Fixed bins begin at presentation onset. COBA and PING receive separately fitted probes with identical data splits, optimization, regularisation, initialization count, and matched trainable capacity. A shuffled-label probe must remain at chance.

    Gamma alignment is a separate comparison, not another point on the fixed-bin axis. For PING, detect the actual cycle boundaries and count spikes cycle by cycle. Compare that representation with ordinary fixed bins having the same median duration. Equal-duration fixed bins provide the COBA control. If cycle alignment helps, it must beat resolution alone rather than benefiting merely because a gamma cycle is short.

    #figure(
      image("/artifacts/data/exp096/result_1_information_schematic.svg", width: 100%, alt: "Qualitative decoder-accuracy curve across bin widths, with a gamma-aligned point compared with a duration-matched fixed-bin point."),
      caption: [Design schematic for information available. Moving right narrows $Delta$ and exposes finer timing. The purple cycle mark tests alignment against a fixed bin of the same duration. These curves are illustrative, not predictions or measurements.],
    )

    Generates #link("#result-1-information")[Result 1].

  + *Information used by the native readout.*

    An external decoder answers what _could_ be read. The native readout answers what the trained network _does_ read. Let $cal(S)_Delta(H)$ denote the response obtained by redistributing each neuron's spikes inside each $Delta$-wide bin.

    #block(inset: 10pt, fill: rgb("f3f0e8"), radius: 3pt)[
      *Controlled comparison.* Original $H$ and shuffled $cal(S)_Delta(H)$ contain exactly the same spike count for every neuron inside every $Delta$-wide window. They differ only in timing finer than $Delta$. At 200 ms, only each neuron's presentation-wide count survives. At smaller $Delta$, progressively finer structure is protected from the shuffle.
    ]

    Replay $H$ and $cal(S)_Delta(H)$ through the same frozen $W_"out"$ and exact #raw("mem-mean") output dynamics from identical initial states. First require intact replay to reproduce the stored native output. Then measure class decisions and true-class logit margins across $Delta$.

    Two interventions ask what kind of gamma-relative timing matters. Cycle permutation preserves each complete volley but changes its order. Within-cycle phase shuffling preserves cycle participation but changes fine phase. Duration-matched interventions provide the COBA controls.

    #figure(
      image("/artifacts/data/exp096/result_2_use_schematic.svg", width: 100%, alt: "Qualitative native-readout accuracy and logit-margin curves as progressively finer temporal structure is preserved."),
      caption: [Design schematic for native use. A flat trace means presentation counts suffice. A low coarse-resolution score that recovers toward intact replay as $Delta$ narrows means the trained output depends on timing at the recovery scale.],
    )

    Generates #link("#result-2-use")[Result 2].

  + *Computational benefit under constraint.*

    Timing may be unnecessary after a full 200 ms presentation yet useful when the decision must be made earlier or with fewer spikes. Compare intact PING with its presentation-count control $cal(S)_(200 " ms")(H)$ at prefixes of 25, 50, 80, 100, 150, and 200 ms.

    For the spike-budget comparison, create nested retained-spike fractions by reproducible per-neuron thinning. Apply the same retention mask before temporal redistribution so intact and shuffled conditions retain the same neuron-specific spike counts. COBA receives the same prefix and thinning controls.

    Let the timing benefit be

    $ Delta A(c) = A_"intact PING"(c) - A_"timing-disrupted PING"(c), $

    where $A$ is held-out accuracy and $c$ is the available resource: presentation time or retained-spike fraction. A positive $Delta A$ under scarcity means timing makes the existing population code usable with fewer resources; convergence at 200 ms or 100% retained spikes means the advantage is conditional rather than a baseline accuracy difference.

    #figure(
      image("/artifacts/data/exp096/result_3_benefit_schematic.svg", width: 100%, alt: "Qualitative accuracy plots against presentation time and retained-spike fraction for COBA, intact PING, and count-matched temporally disrupted PING."),
      caption: [Design schematic for computational benefit. The illustrated positive interaction is a gap between intact and timing-disrupted PING when time or spikes are scarce, followed by convergence when resources are ample. Overlapping PING traces are the registered null.],
    )

    Generates #link("#result-3-benefit")[Result 3].

  == Planned Results

  + <result-1-information> *What timing makes decodable.*

    *Axes.* Bin width $Delta$ in milliseconds on the horizontal logarithmic axis, ordered from the 200 ms presentation count toward fine timing; official-test digit accuracy on the vertical axis.

    *Traces or marks.* Fixed-window COBA and PING curves, paired network-seed marks, uncertainty across probe initializations, chance control, and a PING gamma-aligned mark beside its duration-matched fixed-window mark.

    *Purpose.* Locate the temporal resolution at which additional digit information first becomes externally accessible, and test whether gamma-relative coordinates outperform an ordinary clock.

    *Expected observation.* Presentation counts probably retain most information, giving shallow fixed-bin curves. A PING-specific rise would show information beyond counts. A cycle-aligned mark above its matched fixed-bin mark would show a gamma-coordinate advantage.

  + <result-2-use> *What timing the trained output uses.*

    *Axes.* Preserved bin width $Delta$ on the horizontal axis; native-readout accuracy and change in true-class logit margin on paired vertical panels.

    *Traces or marks.* COBA and PING replay curves, intact references, cycle-permutation marks, within-cycle-phase marks, and duration-matched COBA controls.

    *Purpose.* Distinguish information available to an external observer from temporal structure required by the actual trained output circuit.

    *Expected observation.* If counts suffice, even the 200 ms shuffle remains near intact replay. If timing matters, performance begins below intact at coarse $Delta$ and recovers when the required temporal scale is preserved. Cycle or phase interventions then identify the gamma-relative component.

  + <result-3-benefit> *What timing buys.*

    *Axes.* Available presentation time against accuracy in the left panel; retained-spike fraction against accuracy in the right panel.

    *Traces or marks.* Matched COBA, intact PING, and presentation-count-shuffled PING curves with one mark per network seed and resource level. Annotations report decision latency, spikes to stable correct decision, and agreement across independent Poisson realizations.

    *Purpose.* Test whether temporal organisation makes the representation readable sooner or with fewer spikes.

    *Expected observation.* The positive result is an interaction: intact PING exceeds count-matched disrupted PING under scarcity, then converges as resources become ample. Overlap throughout is the null result and assigns gamma's demonstrated contribution to regulation and spike economy rather than an additional temporal code.

  == Reading the three results together

  #table(
    columns: (1.2fr, 1.2fr, 1.2fr, 2fr),
    [*Resolution gain*], [*Native replay loss*], [*Constraint interaction*], [*Interpretation*],
    [No], [No], [No], [Timing adds no detected representational or computational benefit; PING regulates activity.],
    [Yes], [No], [Either], [Timing is externally decodable but is not an operative code for the trained readout.],
    [Either], [Yes], [No], [The readout uses timing under ordinary inference without a demonstrated resource-dependent advantage.],
    [Either], [Yes], [Yes], [PING timing is operative and improves rapid or spike-limited computation.],
  )

  #reference-list((
    (
      text: [Cardin, J. A., Carlén, M., Meletis, K., Knoblich, U., Zhang, F., Deisseroth, K., Tsai, L.-H., and Moore, C. I. (2009). Driving fast-spiking cells induces gamma rhythm and controls sensory responses.],
      doi: "10.1038/nature08002",
    ),
    (
      text: [Lewis, C. M., Bosman, C. A., Womelsdorf, T., and Fries, P. (2021). Cortical gamma-band resonance preferentially transmits coherent input.],
      doi: "10.1016/j.celrep.2021.109083",
    ),
  ))
]
