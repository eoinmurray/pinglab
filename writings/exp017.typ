#let meta = (
  title: "Introduction",
  date: "2026-08-11",
  description: "A short guide to the gamma-gated sparsity collection: its central question, experiment sequence, and data dependencies.",
  collection: "gamma-gated-sparsity",
  order: 1,
)

#let body = [
This collection asks whether a pyramidal–interneuron network gamma (PING) loop can act as a structural sparsity constraint in a task-trained spiking network. The experiments separate three questions that are easy to muddle: what generates the rhythm, what sets the firing-rate floor, and whether the resulting sparse activity remains useful for classification. They also distinguish a genuine experimental dependency—one entry consuming another entry's checkpoints or measurements—from a looser conceptual dependency.

=== exp022 — Training-run guide

#link("exp022.html")[Link to exp022.] The collection's operational hub owns the canonical training registry and produces the shared COBA and PING checkpoint banks.

*Child experiments*

+ *exp024 — Accuracy converges, firing rate does not* \
  #link("exp024.html")[Link to exp024.] Audits exp022's per-epoch baseline histories, comparing convergence of classification accuracy and firing rate. *Manuscript:* supports the rate-attractor interpretation of Figure 3; it is not the plotted image source.

+ *exp025 — PING locks E rate ≈10× below COBA* \
  #link("exp025.html")[Link to exp025.] Uses exp022's trained cells for the central comparison between loop-free COBA and gamma-gated PING. *Manuscript:* Figure 3.

+ *exp037 — PING tolerates 80% dropped spikes but collapses on added noise* \
  #link("exp037.html")[Link to exp037.] Applies spike-deletion and spike-addition perturbations to exp022's trained cells. *Manuscript:* Figure 8.

+ *exp038 — Switching the loop on at inference cuts E rate ≈15×* \
  #link("exp038.html")[Link to exp038.] Transfers a trained loop-free baseline into a fresh inhibitory architecture, separating the loop's inference-time effect from training history. *Manuscript:* Figure 4.

+ *exp041 — E rate is affine in gamma frequency* \
  #link("exp041.html")[Link to exp041.] Uses exp022's $tau_"GABA"$ family to measure how inhibitory decay changes gamma frequency and excitatory rate. *Manuscript:* supplies the spiking comparison in Figure 2 and the complete Figure 6.
  + *exp033 — Mean-field PING* \
    #link("exp033.html")[Link to exp033.] Compares the measured sweep with the mean-field model. *Manuscript:* mean-field panels in Figure 2.
  + *exp042 — Timing perturbations* \
    #link("exp042.html")[Link to exp042.] Uses exp022 baseline checkpoints and exp041 timing measurements to separate inhibitory timing from inhibitory level. *Manuscript:* Figure 9.
  + *exp046 — Per-cycle E-spike count* \
    #link("exp046.html")[Link to exp046.] Counts excitatory spikes within cycles measured by exp041. *Manuscript:* Figure 7.

+ *exp044 — Rate floor stable across a 20× Δt sweep* \
  #link("exp044.html")[Link to exp044.] Reads exp022's timestep-sweep cells to test whether the rate floor is physical rather than a step-count artefact. *Manuscript:* Figure 10.

+ *exp048 — Temporal and spatial evidence limits of trained PING* \
  #link("exp048.html")[Link to exp048.] Loads the canonical trained PING checkpoint and varies presentation time, input rate, and evidence distribution. *Manuscript:* current source for Figures 11 and 12, pending replacement by exp082.

+ *exp049 — Gradient descent does not preserve a trainable PING loop* \
  #link("exp049.html")[Link to exp049.] Uses exp022's initialisation-family cells to test whether training retains or dismantles the recurrent loop. *Manuscript:* Figure 5.

+ *exp082 — Variable-rate streaming with a spike-rate readout* \
  #link("exp082.html")[Link to exp082.] Consumes exp022's variable-rate checkpoint bank. Exp080 and exp081 support its design conceptually but are not execution dependencies. *Manuscript:* planned replacement source for Figures 11 and 12 after results are complete.

=== exp023 — PING fundamentals

#link("exp023.html")[Link to exp023.] Establishes the free-running excitatory–inhibitory gamma mechanism without training. *Manuscript:* Figure 1.

=== exp047 — Pool-size invariance requires inverse synaptic scaling

#link("exp047.html")[Link to exp047.] Separates nominal total coupling from realised per-synapse strength while varying inhibitory-pool size.

=== exp054 — A PING rhythmicity metric

#link("exp054.html")[Link to exp054.] Calibrates a rhythmicity statistic and identifies its low-rate and shared-input failure modes using independent simulations. *Manuscript:* coupling-plane and spiking panels in the Figure 2 composite.

=== exp080 — Empirical input-rate calibration for variable-rate PING training

#link("exp080.html")[Link to exp080.] Calibrates transformations of sparse pixel intensities into variable Poisson input rates. It conceptually informs exp082 but supplies no runtime artifact.

=== exp081 — Linear-filter analysis of sparse conductance-driven pixel features

#link("exp081.html")[Link to exp081.] Analyses how sparse conductance-driven pixel streams are filtered before and within the PING network. It conceptually informs exp082 but supplies no runtime artifact.

]
