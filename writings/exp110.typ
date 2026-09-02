// Author-approved exception to Writing Guide 34.0.1 section 7: this manuscript
// intentionally omits Results card wrappers and allows each thematic Results
// subsection to contain multiple ordinary figures while its narrative is written,
// with its nested thematic headings numbered within each subsection.
#import "contents.typ": contents-here, with-contents, with-numbered-equations, with-result-sections
#import "/.demolab/lib.typ": data-image
#import "run-inputs.typ": data-file, input-assets, inputs-ready, pending-report
#import "run-view.typ": with-datasets
#let data-file = data-file.with(article: "exp110")

#let meta = (
  status: "[▦ DATA | v34.0.1]",
  title: "Manuscript",
  created_at: "2026-09-02T00:00:00Z",
  updated_at: "2026-09-02",
  description: "A manuscript scaffold connecting PING circuit dynamics, low-rate task performance, cycle participation, perturbation sensitivity and continuous-stream classification.",
  collection: "gamma-gated-sparsity",
)

#let inputs = (
  "exp023",
  "exp025",
  "exp038",
  "exp042",
  "exp049",
  "exp110",
  "exp082",
)

#let preview-figures = (
  (path: "exp023/overview_compound.png", label: "COBA and PING overview"),
  (path: "exp110/onset_super_compound.png", label: "gamma onset"),
  (path: "exp025/results_compound.png", label: "accuracy and firing rate"),
  (path: "exp038/loop_transfer_compound.png", label: "loop transfer"),
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

  #with-result-sections(number-subsections: true)[

    === Reciprocal coupling creates a gamma-rhythmic low-rate regime

    ==== Circuit architecture and illustrative activity

    #emph[Figure reference: @fig:matched-drive (A–D).]

    We first compared two conductance-based spiking circuits that differed only
    in their reciprocal population coupling (Fig. 1A,B). Both contained 1,024
    excitatory and 256 inhibitory neurons, with 1,024 Poisson input channels
    projecting to the excitatory population. The COBA control had recurrent
    coupling disabled, whereas the PING circuit included E→I excitation and I→E
    inhibition, without E→E or I→I connections. Representative 400-ms rasters
    introduce the resulting loop-off and loop-on activity regimes (Fig. 1C,D).
    These examples were generated at separately chosen input rates—5 and 45 Hz
    per channel, respectively.

    #figure(
      data-image(
        data-file("exp023/overview_compound.png"),
        width: 92%,
        alt: "Loop-disabled COBA and recurrent PING networks shown through wiring diagrams, illustrative spike rasters collected under different Poisson drives, power spectra and matched firing-rate–input sweeps.",
      ),
      caption: [*Architecture and population activity of loop-disabled COBA and
        recurrent PING circuits.* *(A, B)* Schematics of the loop-disabled COBA
        control and the PING circuit with reciprocal E→I and I→E coupling.
        *(C, D)* Representative 400-ms spike rasters from networks containing
        1,024 excitatory (E; black) and 256 inhibitory (I; red) neurons. The
        illustrative COBA and PING trials received independent Poisson input
        through 1,024 channels at 5 and 45 Hz per channel, respectively.
        *(E, G)* Welch power spectral densities of the mean-subtracted
        E-population spike traces from the corresponding raster trials. The
        dashed line in *(G)* marks the interpolated 41.4-Hz spectral peak; the
        absence of a marker in *(E)* is not a statistical test for rhythmicity.
        *(F, H)* Mean per-neuron E and I firing rates across matched 2–100-Hz
        Poisson-drive sweeps using 784 input channels. Each point represents one
        trial from one stochastic seed; no uncertainty estimate is shown.
        Vertical rate scales differ between *(F)* and *(H)*.],
    ) <fig:matched-drive>

    ==== Recurrent coupling organises activity into gamma cycles

    #emph[Figure reference: @fig:matched-drive (C–E, G).]

    These architectural differences were accompanied by distinct temporal
    patterns. With recurrent coupling disabled, excitatory spikes were dispersed
    across the 400-ms trial and the inhibitory population remained silent (Fig.
    1C). In the PING circuit, excitatory and inhibitory spikes instead formed
    recurring population volleys (Fig. 1D), and the excitatory-population
    spectrum contained a 41.4-Hz peak with higher-frequency harmonics (Fig. 1G);
    the loop-off spectrum lacked this regular harmonic structure (Fig. 1E).
    These single-trial examples show that the recurrent circuit supported
    gamma-periodic organisation.

    ==== PING maintains lower excitatory rates across input drive

    #emph[Figure reference: @fig:matched-drive (F, H).]

    The temporal reorganisation was accompanied by a marked change in the
    circuit’s input–output response. In the loop-off control, mean excitatory
    firing increased from 2.9 to 481.5 Hz as the per-channel input rate rose
    from 2 to 100 Hz, while the disconnected inhibitory population remained
    silent (Fig. 1F). With the PING loop active, excitatory firing remained
    between 2.6 and 8.7 Hz across the same drive sweep, whereas inhibitory
    firing increased to 72.1 Hz (Fig. 1H). Thus, above the lowest drive
    condition, reciprocal coupling strongly constrained excitatory recruitment.
    Each condition comprised one 400-ms trial from one stochastic seed.

    ==== The rhythmic low-rate regime spans the coupling plane

    #emph[Figure reference: @fig:coupling-plane (A–C).]

    We next tested how the low-rate rhythmic state depended on reciprocal
    coupling strength. We varied the E→I and I→E initialization means across an
    11 × 11 coupling plane under fixed 100-Hz Poisson drive (Fig. 2A–C). When
    either pathway was absent, excitatory firing remained near 94 Hz and
    lobe–trough contrast was zero. With both pathways present, stronger reciprocal
    coupling progressively reduced excitatory firing into the single-digit range,
    recruited sustained inhibitory firing and increased lobe–trough contrast
    towards one across a broad region of the plane. Thus, low-rate, strongly
    structured activity emerged over an extended range of reciprocal coupling
    values rather than at an isolated operating point. Each grid condition
    contained one network from one stochastic seed.

    #figure(
      data-image(
        data-file("exp110/onset_super_compound.png"),
        width: 92%,
        alt: "Coupling-plane maps, representative spike rasters and mean-field analyses of oscillatory onset in the recurrent excitatory-inhibitory circuit.",
      ),
      caption: [*Reciprocal-coupling sweep and mean-field onset comparison.*
        *(A–C)* Mean per-neuron excitatory (E) firing rate, inhibitory (I)
        firing rate, and E-population autocorrelation lobe–trough contrast across
        an 11×11 grid of E→I ($W_(E I)$) and I→E ($W_(I E)$) coupling strengths.
        The coupling values are initialization parent means on the
        fan-in-normalized summed-conductance scale. Each condition comprised one
        untrained network of 256 E and 256 I neurons receiving private 100-Hz
        Poisson input, evaluated for 0.9 s after a 0.1-s burn-in. Cell
        annotations give the measured values; the grayscale in *(B)* is clipped
        at its 92nd percentile, while annotations retain the unclipped I rates.
        Lobe–trough contrast is
        $(A_"lobe" - A_"trough") / (A_"lobe" + A_"trough")$, calculated from
        the 1-ms-binned E autocorrelogram. *(D–F)* Representative 200-ms rasters
        at the conditions marked in *(C)*: $(W_(E I), W_(I E)) = (0, 0)$,
        $(0.6, 1.2)$, and $(3, 6)$ µS, respectively. Black and red marks show
        the first 160 E and 48 I neurons. One seed was used per grid condition;
        no uncertainty estimate is shown. *(G–I)* A separate four-variable
        mean-field conductance model with 4-mV effective voltage noise. *(G)*
        Fixed-point eigenvalues over external drive $I_"ext" = 0$–4 nA; colour
        denotes drive and cyan circles mark the leading conjugate pair at
        $I_"ext"^* = 0.596$ nA. *(H)* Peak-to-peak E-rate amplitude measured
        over the final 500 ms of 2-s upward and downward drive integrations; the
        dotted line marks $I_"ext"^*$. *(I)* Black circles and the solid line
        show the mean-field prediction: the onset frequency calculated from the
        leading eigenvalue at the Hopf crossing for each inhibitory decay time.
        Red squares and the dashed line show the simulator result: the median
        E-population spectral-peak frequency across three separately trained
        spiking classifiers at each decay time. No uncertainty interval is
        shown. The predicted onset frequencies and simulated finite-drive
        spectral peaks are distinct estimators; the mean-field model was not
        calibrated to the spiking networks.],
    ) <fig:coupling-plane>

    ==== Representative networks confirm the mapped transition

    #emph[Figure reference: @fig:coupling-plane (D–F).]

    Representative rasters linked the coupling-plane summaries to the underlying
    population spike patterns. Without reciprocal coupling, excitatory neurons
    fired densely while the inhibitory population remained silent, and
    E-population lobe–trough contrast was near zero (0.0017; displayed as 0.00;
    Fig. 2D). At intermediate coupling ($W_(E I)=0.6$, $W_(I E)=1.2$ µS),
    recurring inhibitory volleys appeared alongside sparser excitatory firing,
    with contrast increasing to 0.27 (Fig. 2E). Under strong coupling
    ($W_(E I)=3$, $W_(I E)=6$ µS), inhibitory volleys were highly regular and
    excitatory firing was sparse, while contrast reached 0.98 (Fig. 2F).
    Together, the rasters and summary metric show a transition from dense, weakly
    structured activity to sparse, temporally clustered population firing. These
    are selected conditions from the same single-seed sweep.

    ==== Oscillations emerge through a Hopf-like transition

    #emph[Figure reference: @fig:coupling-plane (G–H).]

    ==== Inhibitory timescale controls oscillation frequency

    #emph[Figure reference: @fig:coupling-plane (I).]

    === The fixed PING loop preserves accuracy at lower excitatory rates

    ==== PING shifts the accuracy–firing-rate relationship

    #emph[Figure reference: @fig:accuracy-rate (A–D).]

    #figure(
      data-image(
        data-file("exp025/results_compound.png"),
        width: 92%,
        alt: "COBA and PING single-trial activity, validation accuracy and test accuracy against excitatory firing rate.",
      ),
      caption: [*(A–B)* Representative COBA and PING activity, *(C)* validation
        accuracy and *(D)* the test-accuracy–E-rate frontier across activity
        ceilings; frontier points show three-training-replicate means and SEM.
        Source writing: #link("/exp025/")[_Accuracy and Firing Rate With and
        Without Inhibition_]; related convergence analysis:
        #link("/exp024/")[_Accuracy
        Plateaus While Firing Rate Rises_].],
    ) <fig:accuracy-rate>

    ==== Activating inhibition after training reproduces the regime

    #emph[Figure reference: @fig:loop-transfer (A–D).]

    #figure(
      data-image(
        data-file("exp038/loop_transfer_compound.png"),
        width: 92%,
        alt: "Loop-off and loop-on rasters followed by population firing rates and test accuracy across reciprocal loop strength.",
      ),
      caption: [Reciprocal loop strength varied during inference in three trained
        COBA networks without retraining: *(A)* loop-off and *(B)* loop-on
        rasters, *(C)* population rates and *(D)* accuracy; curves show means and
        bands show sample SD across training replicates. Source writing:
        #link("/exp038/")[_Switching On the Inhibitory Loop_].],
    ) <fig:loop-transfer>

    ==== Training recurrent weights weakens rhythmic organisation

    #emph[Figure reference: @fig:trainable-loop (A–D).]

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
    ) <fig:trainable-loop>

    === Excitatory firing is organised by gamma-cycle participation

    ==== Excitatory rate scales with gamma frequency

    #emph[Figure reference: @fig:cycle-participation (A–B).]

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
    ) <fig:cycle-participation>

    ==== Active excitatory neurons usually fire once per cycle

    #emph[Figure reference: @fig:cycle-participation (C–H).]

    === The operating regime has asymmetric perturbation sensitivity

    ==== Added and deleted spikes affect accuracy differently

    #emph[Figure reference: @fig:robustness (A–B).]

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
    ) <fig:robustness>

    ==== Performance persists across integration timesteps

    #emph[Figure reference: @fig:robustness (C).]

    ==== Inhibitory replay perturbations alter recruitment

    #emph[Figure reference: @fig:replay-perturbations (A–D).]

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
    ) <fig:replay-perturbations>

    === PING networks classify continuously presented inputs

    ==== Classification survives without resetting hidden state

    #emph[Figure reference: @fig:continuous-stream (A–D).]

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
    ) <fig:continuous-stream>

    ==== Duration and input rate define the operating range

    #emph[Figure reference: @fig:continuous-stream (E–F).]

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
