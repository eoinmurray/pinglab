#import "templates/abstract.typ": journal-abstract
// Author-approved exception to Writing Guide 34.0.1 section 7: this manuscript
// intentionally omits Results card wrappers and allows each thematic Results
// subsection to contain multiple ordinary figures while its narrative is written,
// with its nested thematic headings numbered within each subsection.
#import "templates/article-layout.typ": journal-article
#import "templates/result-card.typ": with-result-sections
#import "/.demolab/lib.typ": data-image
#import "templates/dataset.typ": data-file, input-assets, inputs-ready, pending-report
#let data-file = data-file.with(article: "exp110")

#let meta = (
  tags: ("data", "v36.0.0"),
  title: "Manuscript",
  created_at: "2026-09-02T00:00:00Z",
  updated_at: "2026-09-07",
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
  (path: "exp049/training_summary.svg", label: "frozen and trainable loop weights"),
  (path: "exp110/cycle_participation_compound.png", label: "rate, gamma frequency and cycle participation"),
  (path: "exp110/robustness_compound.png", label: "spike perturbation and timestep robustness"),
  (path: "exp042/rhythm_compound.png", label: "inhibitory replay perturbations"),
  (path: "exp082/continuous_stream_compound.png", label: "continuous-stream capability and operating range"),
)

#let render-report(data-file) = [
  #journal-abstract(body: [
  A fixed excitatory–inhibitory PING loop produced a low-rate rhythmic regime
  compatible with MNIST classification, linked excitatory firing to gamma-cycle
  participation, showed distinct sensitivity to spike and timing perturbations,
  and continued to support classification when inputs were presented as a
  continuous stream.
  ])

  == Results

  #with-result-sections(number-subsections: true)[

    === Reciprocal coupling creates a gamma-rhythmic low-rate regime

    We compared two conductance-based spiking circuits differing only in their
    reciprocal population coupling (Fig. 1A,B). Both contained 1,024
    excitatory and 256 inhibitory neurons, with 1,024 Poisson input channels
    projecting to the excitatory population. The COBA control had recurrent
    coupling disabled, whereas the PING circuit included E→I excitation and I→E
    inhibition, without E→E or I→I connections. The 400-ms rasters illustrate
    the resulting loop-off and loop-on activity regimes (Fig. 1C,D).
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
        Vertical rate scales differ between *(F)* and *(H)*. Source experiment:
        #link("/exp023/")[exp023] — #link("/exp023/")[_Turning the PING Loop On._]],
    ) <fig:matched-drive>

    Reciprocal coupling substantially reorganised the temporal pattern of
    spiking. With recurrent coupling disabled, excitatory spikes were dispersed
    across the 400-ms trial and the inhibitory population remained silent (Fig.
    1C). In the PING circuit, excitatory and inhibitory spikes instead formed
    recurring population volleys (Fig. 1D), and the excitatory-population
    spectrum contained a 41.4-Hz peak with higher-frequency harmonics (Fig. 1G);
    the loop-off spectrum lacked this regular harmonic structure (Fig. 1E).
    These single-trial examples show that the recurrent circuit supported
    gamma-periodic organisation.

    The same circuit change also reshaped the input–output response. In the
    loop-off control, mean excitatory firing increased from 2.9 to 481.5 Hz as
    the per-channel input rate rose from 2 to 100 Hz, while the disconnected
    inhibitory population remained silent (Fig. 1F). With the PING loop active,
    excitatory firing remained
    between 2.6 and 8.7 Hz across the same drive sweep, whereas inhibitory
    firing increased to 72.1 Hz (Fig. 1H). Thus, above the lowest drive
    condition, reciprocal coupling strongly constrained excitatory recruitment.
    Each condition comprised one 400-ms trial from one stochastic seed.

    To determine whether this regime required a narrow parameter choice, we
    mapped the E→I and I→E initialization means across an 11 × 11 coupling plane
    under fixed 100-Hz Poisson drive (Fig. 2A–C). When either pathway was absent,
    excitatory firing remained near 94 Hz and lobe–trough contrast was zero. With
    both pathways present, stronger reciprocal
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
        the 1-ms-binned E autocorrelogram.

        *(D–F)* Representative 200-ms rasters
        at the conditions marked in *(C)*: $(W_(E I), W_(I E)) = (0, 0)$,
        $(0.6, 1.2)$, and $(3, 6)$ µS, respectively. Black and red marks show
        the first 160 E and 48 I neurons. One seed was used per grid condition;
        no uncertainty estimate is shown.

        *(G–I)* A separate four-variable
        mean-field conductance model with 4-mV effective voltage noise. *(G)*
        Fixed-point eigenvalues over external drive $I_"ext" = 0$–4 nA; colour
        denotes drive and cyan circles mark the leading conjugate pair at
        $I_"ext"^* = 0.596$ nA. *(H)* Peak-to-peak E-rate amplitude measured
        over the final 500 ms of 2-s upward and downward drive integrations; the
        dotted line marks $I_"ext"^*$.

        *(I)* Black circles and the solid line
        show the mean-field prediction: the onset frequency calculated from the
        leading eigenvalue at the Hopf crossing for each inhibitory decay time.
        Red squares and the dashed line show the simulator result: the median
        E-population spectral-peak frequency across three separately trained
        spiking classifiers at each decay time. No uncertainty interval is
        shown. The predicted onset frequencies and simulated finite-drive
        spectral peaks are distinct estimators; the mean-field model was not
        calibrated to the spiking networks. Source experiments:
        #link("/exp054/")[exp054] — #link("/exp054/")[_Pinglab Rythmicity Metric_],
        #link("/exp033/")[exp033] — #link("/exp033/")[_Gamma Emerges at a Hopf Bifurcation_], and
        #link("/exp041/")[exp041] — #link("/exp041/")[_Firing Rate Tracks Gamma Frequency._]],
    ) <fig:coupling-plane>

    The selected rasters make this progression concrete. Without reciprocal
    coupling, excitatory neurons fired densely while the inhibitory population
    remained silent, and E-population lobe–trough contrast was near zero (0.0017;
    displayed as 0.00;
    Fig. 2D). At intermediate coupling ($W_(E I)=0.6$, $W_(I E)=1.2$ µS),
    recurring inhibitory volleys appeared alongside sparser excitatory firing,
    with contrast increasing to 0.27 (Fig. 2E). Under strong coupling
    ($W_(E I)=3$, $W_(I E)=6$ µS), inhibitory volleys were highly regular and
    excitatory firing was sparse, while contrast reached 0.98 (Fig. 2F).
    Across these conditions, dense, weakly structured activity gave way to sparse
    excitatory firing clustered around increasingly regular inhibitory volleys.
    These are selected conditions from the same single-seed sweep.

    To examine a possible dynamical basis for this oscillatory onset, we analysed
    a separate four-variable mean-field conductance model. As external drive
    $I_"ext"$ increased, the leading complex-conjugate eigenvalues crossed from
    negative to positive real parts at $I_"ext"^* = 0.596$ nA, with an imaginary
    component corresponding to 27.6 Hz
    (Fig. 2G). Above this crossing, peak-to-peak excitatory-rate amplitude
    increased continuously from near zero, while upward and downward drive
    sweeps nearly coincided with no resolved hysteresis (Fig. 2H). The positive
    amplitude-squared slope and its $R^2 = 0.999$ fit therefore met the predefined
    numerical criteria for a supercritical Hopf onset. This classifies the
    sampled mean-field transition numerically; criticality was not established
    analytically.

    Finally, we asked whether the mean-field and spiking models shared a
    dependence on the inhibitory timescale. In the four-variable mean-field
    formulation, increasing the inhibitory decay time $tau_"GABA"$ from 4.5 to
    27 ms reduced the numerically calculated Hopf-onset frequency from 30.2 to
    17.9 Hz (Fig. 2I). In simulations of three
    trained spiking classifiers, the median E-population spectral-peak frequency
    likewise decreased, from 67.3 to 12.2 Hz. Thus, slower inhibition reduced
    oscillation frequency in both the analytical and simulated descriptions.
    Their numerical values did not coincide because the mean-field calculation
    measures an onset eigenfrequency, whereas the simulations measure
    finite-drive spectral peaks in an uncalibrated spiking model; the comparison
    therefore supports a shared timescale dependence rather than pointwise
    quantitative agreement.

    === The fixed PING loop preserves accuracy at lower excitatory rates

    The rhythmic, low-rate regime also supported learned MNIST classification.
    Illustrative responses retained dense excitatory firing in
    COBA and recurring population volleys in PING (Fig. 3A,B), while both
    unpenalised conditions reached approximately 90% validation accuracy during
    training (Fig. 3C). Across three independent training replicates,
    unpenalised PING achieved 89.8% test accuracy at a mean excitatory firing
    rate of 16.6 Hz, compared with 91.1% at 113.9 Hz for COBA—a 6.9-fold
    reduction in firing rate accompanied by a 1.3-percentage-point reduction in
    accuracy (Fig. 3D). Activity penalties reduced firing in both families, but
    PING retained higher accuracy at comparable low rates: with a 10-Hz training
    ceiling, both averaged approximately 9.1 Hz, while PING achieved 88.6%
    accuracy and COBA 86.0%. These results demonstrate that PING supported
    classification with substantially reduced excitatory activity and shifted
    the accuracy–rate relationship at lower firing rates.

    #figure(
      data-image(
        data-file("exp025/results_compound.png"),
        width: 92%,
        alt: "COBA and PING single-trial activity, validation accuracy and test accuracy against excitatory firing rate.",
      ),
      caption: [*Accuracy and hidden-excitatory firing in simulated COBA and
        PING classifiers.* *(A–B)* Illustrative 400-ms spike rasters from the
        final checkpoints of the unpenalised COBA (E↔I loop disabled) and PING
        (fixed E↔I loop enabled) training replicates generated with seed 42 for
        MNIST digit-0 sample 0. Each of the 784 normalized pixels drove an
        independent Poisson channel at its intensity multiplied by the 25-Hz
        maximum-pixel rate; black and red marks denote excitatory and inhibitory
        spikes, respectively.

        Throughout *(C–D)*, red squares denote COBA and
        black diamonds denote PING. *(C)* Mean validation accuracy across epochs
        for the unpenalised conditions, averaged across three independent
        training replicates; no uncertainty interval is displayed.

        *(D)*
        Final-checkpoint test accuracy versus mean per-neuron hidden-E firing
        rate over 200-ms presentations for activity ceilings of 1, 2.5, 5, 10
        and 25 Hz and the unpenalised condition. Points are means across three
        independent training replicates generated with seeds 42–44, each evaluated
        on the same 1,000 official MNIST test images; horizontal and vertical
        bars show SEM across training replicates, and stars mark unpenalised
        conditions. Source experiment: #link("/exp025/")[exp025] — #link("/exp025/")[_Accuracy and Firing Rate With and Without Inhibition._]],
    ) <fig:accuracy-rate>

    To test whether this activity pattern required inhibition during learning,
    we added reciprocal E→I and I→E coupling to three independently trained
    COBA classifiers, holding their learned input and readout weights fixed
    without further optimisation. In
    the illustrative response, enabling the loop replaced dense excitatory
    firing with recurring excitatory and inhibitory volleys (Fig. 4A,B). Across
    the three classifiers, increasing coupling from zero to the strongest
    tested value reduced mean excitatory firing from 112.1 to 8.1 Hz, a
    13.8-fold reduction, while inhibitory firing reached 45.0 Hz (Fig. 4C).
    Test accuracy, however, declined from 91.0% to 42.9% across the same
    intervention (Fig. 4D). Thus, the loop could impose sparse, temporally
    grouped firing on previously trained classifiers, demonstrating that this
    activity pattern did not require learning in the presence of inhibition.
    Entering that regime after training nevertheless incurred a substantial
    classification cost.

    #figure(
      data-image(
        data-file("exp038/loop_transfer_compound.png"),
        width: 92%,
        alt: "Loop-off and loop-on rasters followed by population firing rates and test accuracy across reciprocal loop strength.",
      ),
      caption: [*Post-training insertion of reciprocal inhibition in simulated
        spiking classifiers.* *(A–B)* Illustrative 200-ms rasters from the
        validation-selected checkpoint of the unpenalised, loop-disabled COBA
        training replicate generated with seed 42, evaluated on the same MNIST
        digit-7 source image at $s = 0$ and $s = 1$. Each of the 784 normalized
        pixels drove an independent Poisson channel at its intensity multiplied
        by the 25-Hz maximum-pixel rate. The networks contained 1,024 excitatory
        (E) and 256 inhibitory (I) neurons; black and red marks show a fixed
        pseudorandom subset of 200 E and 64 I neurons, respectively.

        Learned
        input and readout weights were held fixed; recurrent E→I and I→E weights
        were newly initialized at each $s$, without retraining. Before
        normalization, their lower-clamped Gaussian parent means were $s$ and
        $2s$, respectively, with each parent SD equal to 10% of its mean. Each
        matrix was then divided by its number of presynaptic neurons: $1 / 1024$
        for E→I and $1 / 256$ for I→E—a $1 / N_"pre"$, not
        $1 / sqrt(N_"pre")$, scaling.

        *(C–D)* The intervention was swept from
        $s = 0$ to $1$ in increments of 0.1 for three independently trained COBA
        classifiers generated with seeds 42–44. *(C)* Mean per-neuron E rate
        (black circles) and I rate (red squares), calculated over every neuron
        and evaluated presentation. *(D)* Official-test accuracy (red circles);
        the dashed line marks mean accuracy at $s = 0$. Each classifier was
        evaluated on the same 1,000 test images using 200-ms presentations;
        curves show means and bands show sample SD across training replicates.
        Source experiment: #link("/exp038/")[exp038] — #link("/exp038/")[_Switching On the Inhibitory Loop._]],
    ) <fig:loop-transfer>

    Finally, we examined whether the temporal organisation associated with the
    fixed PING loop was retained when its recurrent weights were trained. Input and
    readout weights were trained in all conditions, while E→I and I→E weights
    were either held fixed or trained from standard, one-tenth-standard or zero
    initialisation. After 50 epochs, mean excitatory-population autocorrelation
    lobe–trough contrast was 0.999 with fixed recurrence, compared with 0.094
    when recurrent weights were trained from the same standard initialisation
    (Fig. 5C). The one-tenth-standard and zero-initialised conditions yielded
    contrasts of 0.095 and 0.018, respectively. These measurements averaged
    three independent training replicates responding to the same fixed Poisson
    encoding of a single reference digit, rather than responses across the test
    set. Thus, under this training protocol, allowing recurrent conductances to
    adapt did not preserve the strong temporal contrast observed with the fixed
    PING loop.

    #figure(
      data-image(
        data-file("exp049/training_summary.svg"),
        width: 92%,
        alt: "Final accuracy, E/I rates and reference-image contrast across four conditions, with initial and final recurrent nonzero fractions and mean weights.",
      ),
      caption: [
      *Endpoint activity, probe-response and recurrent-weight measurements across
      training conditions.* Four PING conditions were compared at epoch 50. In
      Frozen, E→I and I→E weights remained fixed while input and readout weights
      trained; Std., 10% and Zero denote recurrent weights trained from standard,
      one-tenth-standard and zero initialization, respectively. Standard
      initialization used lower-clamped Gaussian parent means of 1 µS for E→I
      and 2 µS for I→E, followed by $1 / N_"pre"$ scaling: $1 / 1024$ and
      $1 / 256$, respectively.

      *(A)* Official-test accuracy. *(B)* Mean
      per-neuron E rate (black) and I rate (red). Within each training replicate,
      accuracy and rates were calculated across the same 1,000 MNIST test images;
      rates additionally average over all presentations and all 1,024 E or 256 I
      neurons. Images were presented for 200 ms through 784 independent Poisson
      channels with pixel-proportional rates and a 25-Hz maximum-pixel rate.

      *(C)* Final, non-epoch-smoothed reference-image contrast
      $R_"contrast" = (A_"lobe" - A_"trough") / (A_"lobe" + A_"trough")$,
      calculated from the 1-ms-binned E-population autocorrelation. The same
      single digit-0, sample-0 Poisson encoding was passed through each
      independently trained network; this is a fixed-input diagnostic, not a
      test-set average or estimate of trial-to-trial variability. Bars and error
      bars in *(A–C)* show means ± SEM across three training replicates generated
      with seeds 42–44.

      *(D–E)* Pooled fractions of E→I and I→E matrix entries
      greater than zero. *(F–G)* Pooled arithmetic means of the corresponding
      weights, including zeros, reported in $10^(-3)$ µS. Each weight statistic
      combines 786,432 entries—three complete 262,144-entry matrices—per direction
      and condition. Wide grey and narrow red bars show initialization and epoch
      50, respectively; no weight uncertainty is displayed. Arrows mark relative
      changes of at least 5%, not statistical significance.
        Source experiment: #link("/exp049/")[exp049] — #link("/exp049/")[_Training Recurrent Weights Weakens PING Rhythmicity._]],
    ) <fig:trainable-loop>

    === Excitatory firing is organised by gamma-cycle participation

    Excitatory firing closely tracked population rhythm frequency across
    networks trained at different inhibitory decay times. As the decay time
    increased from 4.5 to 27 ms, mean excitatory-population spectral-peak
    frequency decreased from 67.5 to 11.7 Hz, while mean excitatory firing fell
    from 18.3 to 2.8 Hz (Fig. 6A). An affine fit to the six condition means
    yielded a slope of 0.285 Hz/Hz and an intercept of −0.70 Hz, with
    $R^2 = 0.997$. Each condition comprised three independently trained PING
    classifiers, evaluated at epoch 50 on the same 1,000 MNIST test images using
    200-ms presentations. Mean test accuracy declined from 91.2% to 81.9%
    across the same sweep (Fig. 6B), showing that slower rhythms accompanied
    lower excitatory activity but also a classification cost. This close
    rate–frequency association is consistent with a cycle-participation account,
    although the fitted slope alone does not establish how many neurons fired
    during each cycle.

    #figure(
      data-image(
        data-file("exp110/cycle_participation_compound.png"),
        width: 92%,
        alt: "Post-training excitatory firing rate and accuracy across gamma frequencies, followed by distributions of excitatory spikes per neuron and inferred inhibitory-burst cycle.",
      ),
      caption: [*Excitatory firing and spike counts per cycle across inhibitory
        decay times.* Eighteen PING classifiers comprised three independent
        training replicates (seeds 42–44) at each of six inhibitory decay times.
        All measurements used epoch-50 checkpoints, with 1,024 excitatory (E)
        and 256 inhibitory (I) neurons. Each network received the same 1,000
        MNIST test images for 200 ms per image through 784 independent Poisson
        channels, with pixel-proportional rates and a 25-Hz maximum-pixel rate.

        *(A–B)* Mean per-neuron E firing rate $r_E$ and test accuracy versus
        measured spectral-peak frequency $f_gamma$. Within each network, firing
        rate averages all E neurons and presentations; accuracy counts correct
        classifications across the test images. Frequency was estimated by
        averaging the trials’ E-population Welch power spectra, then locating
        the largest peak within 5–150 Hz using parabolic interpolation. Black
        points and horizontal and vertical error bars show means ± SEM across
        three training replicates per condition. Labels in A identify inhibitory
        decay times. The red dashed line is the equal-weight least-squares fit
        $r_E = a + p f_gamma$ to the six condition means, with fitted intercept
        $a$, slope $p$, and coefficient of determination $R^2$. The dotted line
        in B marks 10% chance accuracy.

        *(C–H)* Fractions of E neuron–cycle pairs containing 0, 1, 2 or ≥3 spikes
        at inhibitory decay times of 4.5, 6, 9, 12, 18 and 27 ms, respectively.
        Inhibitory bursts were detected after Gaussian smoothing of population-I
        counts ($sigma = 1$ ms), using a 5%-of-maximum height threshold and
        minimum separation approximately half the network’s measured period.
        Cycle boundaries were midpoints between successive burst peaks; the
        first and last intervals extended to the presentation boundaries.
        Trials without detected bursts were excluded. Each distribution pools
        counts across all E neurons, detected cycles and three training
        replicates within that condition; no uncertainty bars are displayed.
        Source experiments:
        #link("/exp041/")[exp041] — #link("/exp041/")[_Firing Rate Tracks Gamma Frequency_] and
        #link("/exp046/")[exp046] — #link("/exp046/")[_One Spike per Gamma Cycle._]],
    ) <fig:cycle-participation>

    Direct spike counts clarified how excitatory neurons participated in the
    population rhythm. We counted each neuron’s spikes within cycles defined
    around inhibitory population bursts (Fig. 6C–H). Across 167.2 million
    neuron–cycle pairs pooled from the 18 trained classifiers, 75.2% contained
    no spikes, 23.6% contained exactly one spike, and 1.15% contained two or
    more. Among pairs with at least one spike, 95.4% contained exactly one.
    This predominance of single-spike participation held across all six
    inhibitory-decay conditions, although it weakened as inhibition slowed:
    exactly one spike accounted for 98.9% of active pairs at 4.5 ms, compared
    with 83.8% at 27 ms. Thus, population activity combined widespread silence
    within individual cycles with predominantly single-spike participation
    when neurons were active. These distributions support an approximate
    one-spike-per-active-cycle description, rather than a strict firing ceiling
    or a constant participating fraction.

    === The operating regime has asymmetric perturbation sensitivity

    + Added and deleted spikes affect accuracy differently
      (@fig:robustness, panels A–B).

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
        Source experiments: #link("/exp037/")[exp037] — #link("/exp037/")[_Dropped Spikes vs Added Noise_] and #link("/exp044/")[exp044] — #link("/exp044/")[_Firing Rate Across the Timestep Sweep._]],
    ) <fig:robustness>

    + Performance persists across integration timesteps
      (@fig:robustness, panel C).

    + Inhibitory replay perturbations alter recruitment
      (@fig:replay-perturbations, panels A–D).

    #figure(
      data-image(
        data-file("exp042/rhythm_compound.png"),
        width: 92%,
        alt: "Excitatory and inhibitory rasters, excitatory rate, accuracy and realised inhibitory rate under two inhibitory replay-jitter manipulations.",
      ),
      caption: [Representative E/I rasters under *(A)* independent-spike and
        *(B)* fixed-window inhibitory replay jitter; *(C–D)* show the corresponding
        E-rate, accuracy and realised-I-rate sweeps. Source experiment:
        #link("/exp042/")[exp042] — #link("/exp042/")[_Inhibitory Replay Perturbations Change Excitatory Firing._]],
    ) <fig:replay-perturbations>

    === PING networks classify continuously presented inputs

    + Classification survives without resetting hidden state
      (@fig:continuous-stream, panels A–D).

    #figure(
      data-image(
        data-file("exp082/continuous_stream_compound.png"),
        width: 92%,
        alt: "A correctly classified five-digit continuous stream with per-digit durations and input rates labelled, alongside accuracy across presentation duration and input rate.",
      ),
      caption: [One correctly classified five-digit continuous stream with
        per-digit duration and maximum-pixel input rate labelled above each segment: *(A)* input thumbnails, *(B)* E spikes, *(C)* I spikes and *(D)*
        output-count evidence. *(E)* Mean accuracy across presentation duration
        and input rate; *(F)* the 200 ms input-rate curve. Summary values are
        means across three training replicates and curve error bars are SEM.
        Hidden neuronal state continued while output counts reset at known
        boundaries. Source experiment: #link("/exp082/")[exp082] — #link("/exp082/")[_Spike-Count Classification in a Continuous Stream._]],
    ) <fig:continuous-stream>

    + Duration and input rate define the operating range
      (@fig:continuous-stream, panels E–F).

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
#let body = journal-article("exp110", inputs, report-body, dataset-placed: inputs-ready(data-file, inputs))
