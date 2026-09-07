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
        Poisson input, evaluated for 0.9 s after a 0.1-s burn-in. Colour bars
        give the measurement scales; the grayscale in *(B)* is clipped
        at its 92nd percentile.
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
    remained silent, and E-population lobe–trough contrast was near zero (0.0017; Fig. 2D). At intermediate coupling ($W_(E I)=0.6$, $W_(I E)=1.2$ µS),
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
        and evaluated presentation. *(D)* Official-test accuracy (grey squares);
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

    The trained circuits tolerated substantial spike deletion, but diverged
    sharply under spike insertion. Independently deleting each naturally
    generated excitatory or inhibitory spike with 80% probability left mean
    test accuracy at 89.3% for COBA and 89.2% for PING; complete deletion
    reduced both to 10.6% (Fig. 7A). In contrast, inserting independent spikes
    into both populations at a nominal per-neuron rate equal to each network’s
    own unperturbed test-set excitatory firing rate reduced accuracy to 82.8%
    for COBA and 38.5% for PING. At twice that baseline rate, accuracy was
    72.6% and 11.4%, respectively (Fig. 7B). These values average three
    independently trained classifiers per family, each evaluated on the same
    1,000 test images without retraining. Thus, PING tolerated substantial
    deletion but was more sensitive than COBA to strong spike insertion at
    matched baseline-relative doses. Because both perturbations affected
    recurrent feedback and readout input, this comparison does not isolate
    disruption of rhythmic organisation as the cause of classification failure.

    #figure(
      data-image(
        data-file("exp110/robustness_compound.png"),
        width: 92%,
        alt: "COBA and PING accuracy under spike deletion and addition, followed by firing rate and accuracy across integration timesteps.",
      ),
      caption: [*Spike-perturbation and integration-timestep comparisons in
        trained spiking classifiers.* Networks contained 1,024 excitatory (E)
        and 256 inhibitory (I) neurons. Each network was evaluated on 1,000
        MNIST test images presented for 200 ms through 784 independent Poisson
        channels, with pixel-proportional rates and a 25-Hz maximum-pixel rate.

        *(A–B)* Test accuracy under inference-time perturbations of unpenalised
        COBA and PING classifiers, without retraining. Checkpoints were selected
        by minimum validation loss over 50 training epochs. Red squares denote
        COBA and black diamonds denote PING. Lines show means across three
        independent training replicates per model, generated with seeds 42–44;
        shaded bands show ±1 sample SD. Dashed horizontal lines mark 10% chance
        accuracy. The integration timestep was 0.1 ms.

        *(A)* Each naturally generated E or I spike was independently deleted
        with the indicated probability, spanning 0–100% in 10-percentage-point
        increments. Deletion affected transmitted spikes without undoing their
        associated neuronal resets.

        *(B)* Independent Bernoulli spike insertion at nominal rates spanning
        0–200% of each network’s own unperturbed test-set E firing rate, in
        10-percentage-point increments. This baseline averages all E neurons and
        test presentations. The same nominal per-neuron insertion rate was
        applied independently to E and I; accuracy was averaged across training
        replicates at matched percentages, not matched rates in hertz.
        Transmitted activity was capped at one spike per neuron per timestep,
        so insertions coinciding with existing spikes added nothing. Inserted
        events did not trigger neuronal resets or refractory periods. Both
        perturbations acted before readout and subsequent recurrent transmission.

        *(C)* Mean per-neuron E firing rate (black diamonds, left axis) and test
        accuracy (grey squares, right axis) for PING networks trained separately
        at integration timesteps of 0.05, 0.1, 0.25, 0.5 and 1 ms. Each final
        epoch-50 checkpoint was evaluated at its training timestep. Rates
        average all E neurons and test presentations; points and error bars show
        means ± SEM across three independent training replicates per timestep,
        generated with seeds 42–44.

        Source experiments: #link("/exp037/")[exp037] — #link("/exp037/")[_Dropped Spikes vs Added Noise_] and
        #link("/exp044/")[exp044] — #link("/exp044/")[_Firing Rate Across the Timestep Sweep._]],
    ) <fig:robustness>

    Classification performance remained near 90% across a twentyfold range of
    integration timesteps. We compared five timestep settings from 0.05 to 1 ms,
    with three independently trained PING classifiers per setting. Each network
    was evaluated at epoch 50 using its training timestep, the same 1,000 MNIST
    test images and 200-ms presentations. Mean test accuracy ranged from 88.8%
    to 90.1%, a span of 1.3 percentage points, whereas mean excitatory firing
    increased from 13.8 to 20.1 Hz between the finest and coarsest timesteps
    (Fig. 7C). Thus, classification performance persisted across the tested
    numerical resolutions, although excitatory activity remained
    timestep-dependent. Because networks were trained separately at each
    timestep, this comparison establishes robustness of the matched
    training-and-evaluation procedure rather than numerical convergence for a
    fixed set of weights.

    Rearranging inhibitory spike timing produced opposite effects on excitatory
    recruitment despite preserving every inhibitory neuron’s spike count. We
    replayed recorded inhibitory streams after either shifting individual spikes
    independently or applying a shared shift to spikes within fixed 22.8-ms
    windows. At a proposed jitter standard deviation of 14 ms, independent-spike
    shifts reduced mean excitatory firing from 16.6 Hz in the zero-jitter replay
    control to 0.008 Hz, while test accuracy fell from 89.8% to 11.9%
    (Fig. 8A,C). Fixed-window group shifts instead increased excitatory firing
    to 68.3 Hz, while accuracy declined to 82.5% (Fig. 8B,D). Mean replayed
    inhibitory firing remained unchanged at 108.3 Hz throughout both sweeps.
    These values average three independently trained classifiers, each evaluated
    on the same 1,000 test images. Thus, the temporal distribution of a fixed
    inhibitory spike count strongly influenced excitatory recruitment. This
    result concerns inhibitory replay rather than the intact feedback circuit,
    because delivered inhibition could no longer respond to ongoing excitatory
    activity.

    #figure(
      data-image(
        data-file("exp042/rhythm_compound.png"),
        width: 92%,
        alt: "Excitatory and inhibitory rasters, excitatory rate, accuracy and realised inhibitory rate under two inhibitory replay-jitter manipulations.",
      ),
      caption: [*Excitatory responses to count-preserving inhibitory replay
        perturbations.* Three unpenalised PING classifiers, generated with seeds
        42–44, were evaluated at their final epoch-50 checkpoints. Networks
        contained 1,024 excitatory (E) and 256 inhibitory (I) neurons. Each
        condition used the same 1,000 MNIST test images, presented for 200 ms at
        a 0.1-ms timestep through 784 independent Poisson channels with
        pixel-proportional rates and a 25-Hz maximum-pixel rate. Recorded
        unperturbed I spike streams were shifted and replayed in place of
        naturally generated I outputs, while E activity and readout responses
        were recomputed without changing weights.

        *(A–B)* Illustrative responses to the same digit-7 test image, sample 0,
        from the seed-42 training replicate at $sigma = 14$ ms, where $sigma$
        denotes the standard deviation of the proposed Gaussian time shifts.
        Black marks show simulated E spikes and red marks show replayed I
        spikes. Both panels display the same sampled 200 E and 64 I neurons
        over the complete presentation.

        *(A, C)* Independent-spike jitter assigned each inhibitory event its
        own zero-mean Gaussian offset. *(B, D)* Fixed-window group jitter
        assigned one shared offset to all inhibitory events originating within
        each 22.8-ms clock window of a presentation; these windows were not
        detected gamma cycles. Offsets were rounded to the simulation grid.
        Boundary reflection and nearest-free-timestep collision resolution
        preserved each inhibitory neuron’s spike count within every presentation.

        *(C–D)* Black circles show mean per-neuron E firing rate and red squares
        show mean per-neuron replayed I rate on the left axes; grey squares show
        test accuracy on the right axes. Rates average all neurons in the
        corresponding population and all test presentations. Curves show means
        across three training replicates; no uncertainty intervals are
        displayed. Visible jitter values are 0, 0.5, 1, 2, 5, 9 and 14 ms in C,
        and 0, 1, 3, 7 and 14 ms in D. Both arms share the same zero-jitter
        replay control. Replay fixes inhibitory delivery independently of
        ongoing E activity, interrupting effective online E→I→E feedback.

        Source experiment:
        #link("/exp042/")[exp042] — #link("/exp042/")[_Inhibitory Replay Perturbations Change Excitatory Firing._]],
    ) <fig:replay-perturbations>

    === PING networks classify continuously presented inputs

    A PING classifier trained across variable input rates correctly classified
    successive MNIST digits while retaining hidden neuronal state. The
    illustrative stream contained digits 1, 7, 9, 5 and 2, presented without
    gaps over 650 ms, with individual durations of 50–200 ms and maximum-pixel
    input rates of 5–25 Hz (Fig. 9A). Sparse excitatory firing and recurring
    inhibitory volleys continued through the sequence (Fig. 9B,C), and the
    correct class accumulated the largest output-spike count by the end of
    every presentation (Fig. 9D). Hidden excitatory and inhibitory states
    continued between digits, whereas output-neuron state and spike counts
    reset at externally supplied boundaries. The displayed stream was selected
    using a predefined five-correct criterion, which the first candidate
    satisfied. This example therefore demonstrates successful classification
    without hidden-state resets under changing input conditions, but does not
    establish reliability across arbitrary streams or autonomous detection of
    digit boundaries.

    #figure(
      data-image(
        data-file("exp082/continuous_stream_compound.png"),
        width: 92%,
        alt: "A correctly classified five-digit continuous stream with per-digit durations and input rates labelled, alongside accuracy across presentation duration and input rate.",
      ),
      caption: [*Spike-count classification in continuous MNIST streams.*
        Three independently trained PING classifiers (seeds 42–44) contained
        1,024 excitatory (E), 256 inhibitory (I) and ten output leaky
        integrate-and-fire neurons. Networks were trained with variable
        maximum-pixel input rates spanning 0.5–25 Hz; checkpoints were selected
        by minimum validation cross-entropy over 50 epochs, with validation
        accuracy breaking ties. Inference used unchanged weights and 784
        independent, pixel-proportional Bernoulli spike channels at 0.1-ms
        resolution. Digits followed without gaps. Hidden E/I state continued
        between digits within each stream, while output membrane voltage and
        accumulated spike counts reset at externally supplied boundaries.

        *(A–C)* Illustrative 650-ms stream from the seed-42 network. Digits 1,
        7, 9, 5 and 2 were presented with duration–maximum-pixel-rate pairs of
        (100 ms, 5 Hz), (200 ms, 7.5 Hz), (50 ms, 25 Hz), (100 ms, 15 Hz) and
        (200 ms, 10 Hz), respectively. The stream was selected as the first
        candidate satisfying a predefined five-correct criterion; the first
        candidate qualified. *(A)* Input images with true→predicted labels and
        presentation conditions. *(B)* Spikes from the first 200 E neurons,
        shown in black. *(C)* Spikes from the first 64 I neurons, shown in red.
        Vertical dotted lines mark supplied digit boundaries.

        *(D)* Softmax-normalized cumulative output-spike counts within each
        presentation, with the true class highlighted in red and other classes
        in grey. These shares are not calibrated probabilities. Classification
        used the largest final output-spike count, equivalently the largest
        final share; ties selected the lowest class index. The dashed
        horizontal line marks a share of 0.5, not a decision threshold.

        *(E–F)* Test accuracy across presentation duration and maximum-pixel
        input rate. Each network–condition combination comprised 40 five-digit
        streams, giving 200 digit decisions sampled from the official MNIST
        test partition; duration and rate remained fixed within each stream.
        *(E)* Colours and integer annotations show mean percentage accuracy
        across three training replicates at durations of 25, 50, 100 and
        200 ms and the eleven indicated input rates. No uncertainty intervals
        are displayed in the map. *(F)* The same 200-ms measurements plotted
        against input rate on a logarithmic axis; black circles and error bars
        show means ± SEM across training replicates.

        Source experiment:
        #link("/exp082/")[exp082] — #link("/exp082/")[_Spike-Count Classification in a Continuous Stream._]],
    ) <fig:continuous-stream>

    Beyond the illustrative stream, classification accuracy depended on both
    presentation duration and input strength. We evaluated four durations from
    25 to 200 ms and eleven maximum-pixel input rates from 0.5 to 25 Hz using
    three frozen PING classifiers. Each network–condition combination comprised
    40 five-digit streams, yielding 200 decisions, with duration and input rate
    held constant within each stream. At a maximum-pixel rate of 25 Hz,
    increasing presentation duration from 25 to 200 ms raised mean accuracy
    from 72.3% to 88.5% (Fig. 9E). At 200 ms, increasing the input rate from
    0.5 to 5 Hz raised accuracy from 26.0% to 82.7%; accuracy ranged from
    82.7% to 88.7% over 5–25 Hz, without a strictly monotonic increase
    (Fig. 9F). Thus, longer presentations and stronger inputs generally
    supported more accurate continuous classification, whereas brief
    presentations and weak drive constrained performance.

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
