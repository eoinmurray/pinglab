// Author-approved exception to Writing Guide 33.0.0 section 7: this manuscript
// intentionally omits Results card wrappers and allows each thematic Results
// subsection to contain multiple ordinary figures while its narrative is written.
#import "contents.typ": contents-here, with-contents, with-numbered-equations, with-result-sections
#import "/.demolab/lib.typ": data-image
#import "run-inputs.typ": data-file, input-assets, inputs-ready, pending-report
#import "run-view.typ": with-datasets
#let data-file = data-file.with(article: "exp110")

#let meta = (
  status: "[▦ DATA | v33.0.0]",
  title: "Manuscript",
  created_at: "2026-09-02T00:00:00Z",
  updated_at: "2026-09-02",
  description: "A manuscript scaffold connecting PING circuit dynamics, low-rate task performance, cycle participation, perturbation sensitivity and continuous-stream classification.",
  collection: "gamma-gated-sparsity",
)

#let inputs = (
  "exp023",
  "exp042",
  "exp049",
  "exp110",
  "exp082",
)

#let preview-figures = (
  (path: "exp023/overview_compound.png", label: "COBA and PING overview"),
  (path: "exp110/onset_super_compound.png", label: "gamma onset"),
  (path: "exp110/performance_transfer_compound.png", label: "accuracy, firing rate and loop transfer"),
  (path: "exp049/training_curves.svg", label: "frozen and trainable loop weights"),
  (path: "exp110/cycle_participation_compound.png", label: "rate, gamma frequency and cycle participation"),
  (path: "exp110/robustness_compound.png", label: "spike perturbation and timestep robustness"),
  (path: "exp042/rhythm_compound.png", label: "inhibitory replay perturbations"),
  (path: "exp082/continuous_stream_compound.png", label: "continuous-stream capability and operating range"),
)

#let render-report(data-file) = [
  == Abstract

  A fixed excitatory–inhibitory PING loop produced a low-rate rhythmic regime
  compatible with MNIST classification, linked excitatory firing to gamma-cycle
  participation, showed distinct sensitivity to spike and timing perturbations,
  and continued to support classification when inputs were presented as a
  continuous stream.

  #contents-here()

  == Results

  #with-result-sections[

    === Reciprocal coupling creates a gamma-rhythmic low-rate regime

    #figure(
      data-image(
        data-file("exp023/overview_compound.png"),
        width: 92%,
        alt: "Matched-drive comparison of loop-disabled COBA and recurrent PING networks using wiring diagrams, spike rasters, power spectra and excitatory firing-rate responses.",
      ),
      caption: [Loop-disabled COBA and recurrent PING configurations under
        matched Poisson drive: *(A–B)* architecture, *(C–D)* representative E/I
        spiking, *(E–F)* excitatory spectra and *(G–H)* firing-rate–input curves;
        COBA is the first panel and PING the second in each pair. Source writing:
        #link("/exp023/")[_Turning the PING Loop On_].],
    )

    #figure(
      data-image(
        data-file("exp110/onset_super_compound.png"),
        width: 92%,
        alt: "Coupling-plane maps, representative spike rasters and mean-field analyses of oscillatory onset in the recurrent excitatory-inhibitory circuit.",
      ),
      caption: [*(A–C)* Mean E rate, mean I rate and autocorrelation lobe–trough
        contrast across reciprocal coupling; *(D–F)* representative rasters;
        *(G–I)* mean-field eigenvalue, amplitude and frequency analyses. Source
        writing: #link("/exp054/")[_Pinglab Rythmicity Metric_], incorporating
        #link("/exp033/")[_Gamma Emerges at a
        Hopf Bifurcation_] and frequencies from #link("/exp041/")[_Firing Rate
        Tracks Gamma Frequency_].],
    )

    === The fixed PING loop preserves accuracy at lower excitatory rates

    #figure(
      data-image(
        data-file("exp110/performance_transfer_compound.png"),
        width: 92%,
        alt: "COBA and PING activity, validation and accuracy-rate results followed by inference-time activation of reciprocal excitation and inhibition in trained loop-disabled networks.",
      ),
      caption: [*(A–B)* Representative COBA and PING activity, *(C)* validation
        accuracy and *(D)* the test-accuracy–E-rate frontier across activity
        ceilings; frontier points show three-training-replicate means and SEM.
        Reciprocal loop strength was then varied during inference in three trained
        COBA networks: *(E)* loop-off and *(F)* loop-on rasters, *(G)* population
        rates and *(H)* accuracy; bands show sample SD across training replicates.
        Source writings: #link("/exp025/")[_Accuracy and Firing Rate With and
        Without Inhibition_] and #link("/exp038/")[_Switching On the Inhibitory
        Loop_]; related convergence analysis: #link("/exp024/")[_Accuracy
        Plateaus While Firing Rate Rises_].],
    )

    #figure(
      data-image(
        data-file("exp049/training_curves.svg"),
        width: 92%,
        alt: "Training trajectories for accuracy, excitatory rate, inhibitory rate and rhythmicity with frozen or trainable recurrent weights.",
      ),
      caption: [Per-epoch *(A)* accuracy, *(B)* E rate, *(C)* I rate and *(D)*
        lobe–trough contrast for three trainable recurrent initialisations and a frozen-loop control; lines show
        three-training-replicate means and shading shows their range. Source
        writing: #link("/exp049/")[_Training Recurrent Weights Weakens PING
        Rhythmicity_].],
    )

    === Excitatory firing is organised by gamma-cycle participation

    #figure(
      data-image(
        data-file("exp110/cycle_participation_compound.png"),
        width: 92%,
        alt: "Post-training excitatory firing rate and accuracy across gamma frequencies, followed by distributions of excitatory spikes per neuron and inferred inhibitory-burst cycle.",
      ),
      caption: [*(A)* Mean post-training E rate and *(B)* test accuracy across six
        inhibitory-decay conditions; markers show three-training-replicate means,
        error bars show SEM and the line is an affine rate–frequency fit.
        *(C–H)* Distributions of E spikes per neuron–cycle pair, with cycles
        inferred from inhibitory population-burst peaks, at inhibitory decay
        times 4.5, 6, 9, 12, 18 and 27 ms, respectively. Source writings:
        #link("/exp041/")[_Firing Rate Tracks Gamma Frequency_] and
        #link("/exp046/")[_One Spike per Gamma Cycle_].],
    )

    === The operating regime has asymmetric perturbation sensitivity

    #figure(
      data-image(
        data-file("exp110/robustness_compound.png"),
        width: 92%,
        alt: "COBA and PING accuracy under spike deletion and addition, followed by firing rate and accuracy across integration timesteps.",
      ),
      caption: [Mean test accuracy under *(A)* random hidden-spike deletion and
        *(B)* Poisson spike addition; lines show means across three training replicates
        and shading shows SEM. *(C)* Post-training E rate and test accuracy across
        matched training-and-inference integration timesteps from 0.05 to 1.0 ms.
        Source writings: #link("/exp037/")[_Dropped Spikes vs Added Noise_] and
        #link("/exp044/")[_Firing Rate Across the Timestep Sweep_].],
    )

    #figure(
      data-image(
        data-file("exp042/rhythm_compound.png"),
        width: 92%,
        alt: "Excitatory and inhibitory rasters, excitatory rate, accuracy and realised inhibitory rate under two inhibitory replay-jitter manipulations.",
      ),
      caption: [Representative E/I rasters under *(A)* independent-spike and
        *(B)* fixed-window inhibitory replay jitter; *(C–D)* show the corresponding
        E-rate, accuracy and realised-I-rate sweeps. Source writing:
        #link("/exp042/")[_Inhibitory Replay Perturbations Change Excitatory
        Firing_].],
    )

    === PING networks classify continuously presented inputs

    #figure(
      data-image(
        data-file("exp082/continuous_stream_compound.png"),
        width: 92%,
        alt: "A correctly classified five-digit continuous stream at varying input rates, alongside accuracy across presentation duration and input rate.",
      ),
      caption: [One correctly classified five-digit continuous stream with
        200 ms presentations at maximum-pixel input rates of 5, 7.5, 10, 15 and
        25 Hz: *(A)* input thumbnails, *(B)* E spikes, *(C)* I spikes and *(D)*
        output-count evidence. *(E)* Mean accuracy across presentation duration
        and input rate; *(F)* the 200 ms input-rate curve. Summary values are
        means across three training replicates and curve error bars are SEM.
        Hidden neuronal state continued while output counts reset at known
        boundaries. Source writing:
        #link("/exp082/")[_Spike-Count Classification in a Continuous Stream_].],
    )

  ]
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file,
    inputs,
    [How does a fixed PING loop shape excitatory firing, task performance and continuous-stream classification?],
    preview-figures,
  )
}

#let meta = meta + (assets: input-assets("exp110", inputs))
#let body = with-datasets("exp110", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-numbered-equations(body)
#let body = with-contents(body)
