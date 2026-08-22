#let meta = (
  title: "How hard is a neural code to read?",
  date: "2026-08-22",
  description: "Measure how decoder capacity and temporal access change the legibility of frozen COBA and PING representations on MNIST.",
  collection: "gamma-gated-sparsity",
  status: "draft",
)

#let body = [
  == Abstract

  Exp094 found that simple spike-based decoders preserved much more of PING's native MNIST decision than COBA's. That result does not tell us whether COBA discarded the class information or merely expressed it in a form that a simple decoder could not read. This experiment freezes matched COBA and PING networks, gives downstream decoders access only to their hidden excitatory spikes, and increases decoder capacity and temporal access systematically. It asks how much learned machinery and labelled data each representation requires before it supports accurate classification.

  == Methods

  #block(inset: 10pt, fill: rgb("f3f0e8"), radius: 3pt)[
    *Run status.* Nothing in this experiment has been run. All curves, surfaces, and numerical comparisons below are planned outputs. The diagrams are qualitative design schematics, not simulated data.
  ]

  + *Create one matched frozen-activity bank.* Use the canonical seed-42 COBA and PING final-epoch checkpoints from the latest gold-star publication view. Present identical Poisson-encoded MNIST images and seeds to both networks. Freeze every network parameter. Record only the 1,024 excitatory hidden-neuron spike trains, binned into 5 ms counts over the 200 ms presentation. A decoder therefore receives a sequence $bold(x)_(1:B)$ with $B = 40$ bins and $bold(x)_b in RR^1024$. It never receives pixels, labels, hidden voltages, or native output activity. Use the original MNIST training partition to fit decoders, a fixed held-out part of that partition for decoder selection, and the untouched official test partition once for final evaluation. Generates #link("#result-1-interface")[Result 1].

    #figure(
      image(
        "/artifacts/data/exp095/frozen_access_pipeline.svg",
        width: 100%,
        alt: "Identical MNIST spike inputs passing through frozen COBA and PING networks into matched decoder ladders.",
      ),
      caption: [Experimental interface. Both networks receive the same input and expose the same 40 by 1,024 binned excitatory-spike representation. Only the separately fitted downstream decoder may learn.],
    )

  + *Increase decoder capacity without changing its input.* Fit the same decoder family separately to COBA and PING activity. Begin with a linear classifier on presentation-mean spike counts. Add a linear temporal classifier that may weight bins differently. Then fit gated recurrent units with hidden widths $H in {2, 4, 8, 16, 32, 64, 128}$. For a decoder with parameters $theta$, let $P = |theta|$ be its trainable parameter count. Compare models by measured $P$, not by their informal family names. Use identical optimization budgets, early-stopping rules, data order, and five decoder initializations for both networks. A shuffled-label control must remain at chance. Generates #link("#result-2-capacity")[Result 2].

    #figure(
      image(
        "/artifacts/data/exp095/capacity_curve_schematic.svg",
        width: 100%,
        alt: "Design schematic of accuracy against decoder parameter count, with an earlier-rising PING curve and a later-rising COBA curve converging at high capacity.",
      ),
      caption: [Design schematic for the decoder-accessibility curve. The illustrated values are not predictions. The measured result will retain this panel layout and visual language.],
    )

  + *Separate computational capacity from temporal access.* Repeat selected decoder widths while limiting each decoder to the first 5, 10, 25, 50, 100, or 200 ms of activity. At 200 ms, add two controls: collapse all bins into one rate vector, and independently permute time bins within each presentation while preserving every neuron's total spike count. These interventions distinguish information that is linearly available, nonlinearly available, or dependent on temporal order. Generates #link("#result-3-temporal")[Result 3].

    #figure(
      image(
        "/artifacts/data/exp095/temporal_access_schematic.svg",
        width: 100%,
        alt: "Matched COBA and PING heatmap schematics with decoder capacity horizontally, temporal access vertically, and accuracy represented by shade.",
      ),
      caption: [Design schematic for the temporal-accessibility surface. Matched COBA and PING panels vary decoder capacity horizontally and available activity duration vertically.],
    )

  + *Measure the supervision needed to learn each code.* For the linear rate probe and GRU widths 16 and 64, train on nested, class-balanced subsets of 100, 300, 1,000, 3,000, 10,000, and all available decoder-training images. Keep validation and test sets fixed. Report test accuracy only after selecting the decoder with validation data. This measures sample efficiency rather than raw capacity alone. The seed-42 checkpoint pair is sufficient for this first experiment, but any architectural claim must later be repeated across independently trained network seeds. Generates #link("#result-4-samples")[Result 4].

    #figure(
      image(
        "/artifacts/data/exp095/sample_efficiency_schematic.svg",
        width: 100%,
        alt: "Design schematic of held-out accuracy against labelled decoder-training images for linear, medium recurrent, and larger recurrent decoders.",
      ),
      caption: [Design schematic for sample efficiency. Each decoder capacity receives the same nested labelled subsets for COBA and PING.],
    )

  == Results

  + <result-1-interface> *Native reference and activity-bank validation.*

    *Axes.* Horizontal axis: presentation time from 0 to 200 ms. Vertical axes: population firing rate and cumulative spike count.

    *Traces or marks.* Matched COBA and PING population summaries, plus their native mean-voltage test accuracy as reference lines in an adjacent compact panel.

    *Why.* The decoder comparison is interpretable only if both activity banks contain the intended trials, labels, duration, neuron count, and shared input seeds.

    *Expectation.* Both native decoders should reproduce their checkpoint performance. COBA and PING may differ in firing rate and temporal organization, but neither activity bank should contain missing or duplicated presentations.

  + <result-2-capacity> *Decoder-accessibility curve.*

    *Axes.* Horizontal axis: trainable decoder parameters $P$ on a logarithmic scale. Vertical axis: official-test MNIST accuracy.

    *Traces or marks.* COBA and PING traces; one mark per decoder family and width; uncertainty bands across five decoder initializations; horizontal references for native accuracy and chance.

    *Why.* This is the primary test of whether PING exposes class information to smaller downstream computations.

    *Expectation.* If PING and COBA approach the same high-capacity ceiling but PING rises earlier, they retain comparable information and differ in accessibility. A persistent high-capacity gap instead suggests a difference in retained task information. The schematic does not assume either outcome.

  + <result-3-temporal> *Capacity-by-temporal-access surface.*

    *Axes.* Horizontal axis: decoder parameter count or width. Vertical axis: available activity duration in milliseconds. Colour axis: official-test accuracy.

    *Traces or marks.* Matched COBA and PING heatmaps, supplemented by rate-collapse and time-permutation marks at 200 ms.

    *Why.* Parameter count alone cannot distinguish a nonlinear code from one that specifically requires temporal memory.

    *Expectation.* If PING's advantage depends on cycle-organized activity, it should appear at modest capacity with sufficient temporal context and weaken after time permutation. If COBA catches up with a memoryless nonlinear decoder, its obstacle is nonlinear accessibility rather than temporal reconstruction.

  + <result-4-samples> *Decoder sample efficiency.*

    *Axes.* Horizontal axis: labelled decoder-training images on a logarithmic scale. Vertical axis: official-test accuracy.

    *Traces or marks.* COBA and PING traces for the linear rate probe, GRU-16, and GRU-64; uncertainty bands across decoder initializations.

    *Why.* A representation may support the same final accuracy yet require much more supervision to learn how to read it.

    *Expectation.* Earlier-rising PING curves would show that its code is not only readable by smaller decoders but easier to learn from fewer labels. Convergence at large sample sizes would indicate an accessibility difference rather than absent information.
]
