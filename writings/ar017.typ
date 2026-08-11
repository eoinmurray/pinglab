#let meta = (
  title: "Introduction",
  date: "2026-08-11",
  description: "A short guide to the gamma-gated sparsity collection: its central question, experiment sequence, and data dependencies.",
  collection: "gamma-gated-sparsity",
  order: 1,
  status: "draft",
)

#let body = [
This collection asks whether a pyramidal–interneuron network gamma (PING) loop can act as a structural sparsity constraint in a task-trained spiking network. The experiments separate three questions that are easy to muddle: what generates the rhythm, what sets the firing-rate floor, and whether the resulting sparse activity remains useful for classification. They also distinguish a genuine experimental dependency—one entry consuming another entry's checkpoints or measurements—from a looser conceptual dependency.

== Experiments and dependencies

An experiment is a _root_ below when it does not consume another experiment's checkpoints or recorded measurements. Children are grouped beneath the root that supplies their hard execution dependency. Where a child also uses another experiment, that secondary dependency is stated explicitly.

=== exp022 — Training-run guide

+ #link("/exp022/")[Training-run guide (exp022)]
+ The collection's operational hub: it owns the canonical training registry and produces the shared COBA and PING checkpoint banks.
+ *Child experiments*
  + *exp024*
    + #link("/exp024/")[Accuracy converges, firing rate does not]
    + Audits exp022's per-epoch baseline histories, comparing convergence of classification accuracy and firing rate.
  + *exp025*
    + #link("/exp025/")[PING locks E rate ≈10× below COBA]
    + Uses exp022's trained cells for the central comparison between loop-free COBA and gamma-gated PING. Its child, #link("/exp042/")[exp042], perturbs this canonical operating point and also consumes exp041's timing measurements.
  + *exp037*
    + #link("/exp037/")[PING tolerates 80% dropped spikes but collapses on added noise]
    + Applies spike-deletion and spike-addition perturbations to exp022's trained cells.
  + *exp038*
    + #link("/exp038/")[Switching the loop on at inference cuts E rate ≈15×]
    + Transfers a trained loop-free baseline into a fresh inhibitory architecture, separating the loop's inference-time effect from training history.
  + *exp041*
    + #link("/exp041/")[E rate is affine in gamma frequency]
    + Uses exp022's $tau_"GABA"$ family to measure how inhibitory decay changes gamma frequency and excitatory rate. Its children are #link("/exp033/")[exp033], which compares the sweep with a mean-field model, and #link("/exp046/")[exp046], which counts spikes within the measured cycles.
  + *exp044*
    + #link("/exp044/")[Rate floor stable across a 20× Δt sweep]
    + Reads exp022's timestep-sweep cells to test whether the rate floor is physical rather than a step-count artefact.
  + *exp048*
    + #link("/exp048/")[Temporal and spatial evidence limits of trained PING]
    + Loads the canonical trained PING checkpoint and varies presentation time, input rate, and evidence distribution.
  + *exp049*
    + #link("/exp049/")[Gradient descent does not preserve a trainable PING loop]
    + Uses exp022's initialisation-family cells to test whether training retains or dismantles the recurrent loop.
  + *exp082*
    + #link("/exp082/")[Variable-rate streaming with a spike-rate readout]
    + Consumes exp022's variable-rate checkpoint bank. Exp080 and exp081 support its design conceptually but are not execution dependencies.

=== exp023 — PING fundamentals

+ #link("/exp023/")[PING fundamentals (exp023)]
+ Establishes the free-running excitatory–inhibitory gamma mechanism without training.
+ *Child experiments*
  + None. Later experiments use it as a conceptual foundation, not as an artifact source.

=== exp047 — Pool-size invariance requires inverse synaptic scaling

+ #link("/exp047/")[Pool-size invariance requires inverse synaptic scaling (exp047)]
+ Separates nominal total coupling from realised per-synapse strength while varying inhibitory-pool size.
+ *Child experiments*
  + None.

=== exp054 — A PING rhythmicity metric

+ #link("/exp054/")[A PING rhythmicity metric (exp054)]
+ Calibrates a rhythmicity statistic and identifies its low-rate and shared-input failure modes using independent simulations.
+ *Child experiments*
  + None by hard dependency.

=== exp080 — Empirical input-rate calibration for variable-rate PING training

+ #link("/exp080/")[Empirical input-rate calibration for variable-rate PING training (exp080)]
+ Calibrates transformations of sparse pixel intensities into variable Poisson input rates.
+ *Child experiments*
  + None by hard dependency. It conceptually informs exp082.

=== exp081 — Linear-filter analysis of sparse conductance-driven pixel features

+ #link("/exp081/")[Linear-filter analysis of sparse conductance-driven pixel features (exp081)]
+ Analyses how sparse conductance-driven pixel streams are filtered before and within the PING network.
+ *Child experiments*
  + None by hard dependency. It conceptually informs exp082.

== Articles around the experiments

#link("/ar009/")[The manuscript (ar009)] assembles the experimental claims into one argument. #link("/ar010/")[The literature companion (ar010)] records the external evidence behind that argument. Those articles cite and interpret the experiments; they do not produce inputs required to run them.

When following or rerunning the collection, begin at exp022 for provenance and checkpoint ownership, but begin at exp023 for the science. That distinction is the whole map in one sentence.
]
