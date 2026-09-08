// Author-approved standalone manuscript: no shared writing templates.
#import "/.demolab/lib.typ": data-image

// Resolve only the validated presentation inputs supplied by the publishing engine.
#let prepared = "demolab-url-render" in sys.inputs or "demolab-bundle-root" in sys.inputs
#let preview = "demolab-preview-file" in sys.inputs
#let catalogue = if prepared { json("/.demolab/pinglab-inputs.json") } else {
  (articles: (:), defaults: (:), runs: ())
}
#let inventory = if not preview and "demolab-data-inputs" in sys.inputs {
  json(sys.inputs.at("demolab-data-inputs"))
} else { (:) }
#let selections = if preview {
  json(sys.inputs.at("demolab-preview-file"))
} else { inventory.at("sources", default: (:)) }
#let data-file(rel) = {
  let parts = rel.split("/")
  assert(
    parts.all(part => part not in ("", ".", "..")) and not rel.contains("\\"),
    message: "unsafe manuscript input path",
  )
  let key = parts.first()
  if prepared and not preview {
    assert(key in catalogue.articles.at("exp110", default: ()), message: "undeclared manuscript input: " + key)
    let available = catalogue.runs.filter(run => run.experiment == key.split(".").last())
    let parameter = "source." + key
    let run = if sys.inputs.at("demolab-url-article", default: "") == "exp110" and parameter in sys.inputs {
      let matches = available.filter(run => run.basepath == sys.inputs.at(parameter))
      assert(matches.len() == 1, message: "input is not a validated presentation: " + key)
      matches.first()
    } else {
      let pinned = catalogue.defaults.at("exp110", default: (:)).at(key, default: none)
      if pinned != none {
        let matches = available.filter(run => run.id == pinned)
        assert(matches.len() == 1, message: "unavailable presentation: " + key)
        matches.first()
      } else if available.len() > 0 { available.first() } else { none }
    }
    if run == none { return none }
    if parts.len() == 1 { return run.basepath }
    let filename = parts.slice(1).join("/")
    assert(filename in run.files, message: "missing presentation file: " + rel)
    return run.basepath + "/" + filename
  }
  let selected = selections.at("exp110", default: (:))
  if not preview and "exp110" in selections {
    assert(key in selected, message: "missing manuscript input selection: " + key)
  }
  let directory = selected.at(key, default: none)
  if directory == none { return none }
  if parts.len() == 1 { return directory }
  let path = directory + rel.slice(key.len())
  if not preview {
    assert(path in inventory.at("files", default: ()), message: "missing presentation file: " + path)
  }
  path
}

#let meta = (
  tags: ("data", "v36.0.0"),
  title: "Manuscript",
  created_at: "2026-09-02T00:00:00Z",
  updated_at: "2026-09-08",
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

#let render-report(data-file) = [
  #set heading(numbering: none)
  #metadata("exp110-start")

  == Abstract
  A fixed excitatory–inhibitory PING loop produced a low-rate rhythmic regime
  compatible with MNIST classification, linked excitatory firing to gamma-cycle
  participation, showed distinct sensitivity to spike and timing perturbations,
  and continued to support classification when inputs were presented as a
  continuous stream.

  #context {
    let start = query(metadata.where(value: "exp110-start").before(here())).last().location()
    let end = query(metadata.where(value: "exp110-end").after(here())).first().location()
    let sections = query(heading.where(level: 2).after(start).before(end))
    let entries = sections.enumerate().map(((index, section)) => {
      let stop = if index + 1 < sections.len() { sections.at(index + 1).location() } else { end }
      let children = query(heading.where(level: 3).after(section.location()).before(stop))
      [
        #link(section.location(), section.body)
        #if children.len() > 0 {
          list(tight: true, ..children.map(child => link(child.location(), child.body)))
        }
      ]
    })
    let contents = [*Contents* #list(tight: true, ..entries)]
    if target() == "html" {
      html.elem("nav", attrs: ("aria-label": "Table of Contents"), contents)
    } else { contents }
  }

  == Results

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
    caption: [*COBA and PING circuit architecture and activity.*
      *(A, B)* Schematics of loop-disabled COBA and PING with reciprocal E→I
      and I→E coupling. *(C, D)* Illustrative 400-ms rasters from networks
      of 1,024 excitatory (E; black) and 256 inhibitory (I; red) neurons.
      Poisson input used 1,024 independent channels at 5 Hz (COBA) and
      45 Hz (PING) per channel.

      *(E, G)* Welch power spectral densities of the mean-subtracted
      E-population spike traces from C and D. The dashed line in G marks the
      interpolated spectral peak; the missing marker in E is not a
      statistical test for rhythmicity. *(F, H)* Mean per-neuron E and I
      firing rates across matched 2–100-Hz Poisson-drive sweeps with 784 input
      channels. Each point is one trial from one seed, without uncertainty
      estimates; vertical rate scales differ. Source experiment:
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

  The same circuit change also reshaped the input–output response. These
  drive sweeps used 784 input channels, rather than the 1,024 used for the
  illustrative rasters. In the loop-off control, mean excitatory firing
  increased from 2.9 to 481.5 Hz as the per-channel input rate rose from 2
  to 100 Hz, while the disconnected inhibitory population remained silent
  (Fig. 1F). With the PING loop active, excitatory firing remained between
  2.6 and 8.7 Hz across the same drive sweep, whereas inhibitory firing
  increased to 72.1 Hz (Fig. 1H). Thus, above the lowest drive condition,
  reciprocal coupling strongly constrained excitatory recruitment. Each
  condition comprised one 400-ms trial from one stochastic seed.

  To determine whether this regime required a narrow parameter choice, we
  mapped the E→I and I→E initialization means across an 11 × 11 coupling
  plane in untrained networks containing 256 excitatory and 256 inhibitory
  neurons under fixed 100-Hz Poisson drive (Fig. 2A–C). We measured
  temporal structure using lobe–trough contrast, the normalized difference
  between the autocorrelation lobe and trough. When either pathway was
  absent, excitatory firing remained near 94 Hz and lobe–trough contrast
  was zero. With both pathways present, stronger reciprocal coupling
  progressively reduced excitatory firing into the single-digit range,
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
    caption: [*Reciprocal-coupling sweep and mean-field onset.*
      *(A–C)* Mean per-neuron E rate, I rate and E-population autocorrelation
      lobe–trough contrast across an 11×11 grid of E→I ($W_(E I)$) and I→E
      ($W_(I E)$) initialization parent means on the fan-in-normalized
      summed-conductance scale. Each condition used one untrained network
      (256 E, 256 I; one seed) with private 100-Hz Poisson input, evaluated
      for 0.9 s after a 0.1-s burn-in. B's grayscale is clipped at its 92nd
      percentile. Contrast is
      $(A_"lobe" - A_"trough") / (A_"lobe" + A_"trough")$ from the 1-ms-binned
      E autocorrelogram.

      *(D–F)* Illustrative 200-ms rasters at the conditions marked in C:
      $(W_(E I), W_(I E)) = (0, 0)$, $(0.6, 1.2)$ and $(3, 6)$ µS,
      respectively.
      Black and red marks show the first 160 E and 48 I neurons.
      A–F show no uncertainty estimates.

      *(G–H)* Separate four-variable mean-field conductance model with
      4-mV effective voltage noise. *(G)* Fixed-point eigenvalues over
      external drive $I_"ext" = 0$–4 nA (colour); cyan circles mark the leading
      conjugate pair at $I_"ext"^* = 0.596$ nA. *(H)* Peak-to-peak E-rate
      amplitude over the final 500 ms of 2-s upward/downward drive
      integrations; the dotted line marks $I_"ext"^*$.

      *(I)* Mean-field Hopf-onset frequencies from leading eigenvalues
      (black circles, solid line) and median finite-drive E-population
      spectral peaks across three separately trained spiking classifiers
      (red squares, dashed line), at each inhibitory decay time. No
      uncertainty interval is shown. Source experiments:
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
    caption: [*Accuracy and excitatory firing in COBA and PING classifiers.*
      *(A–B)* Illustrative 400-ms rasters from final, unpenalised COBA
      (loop disabled) and PING (fixed loop enabled) checkpoints, seed 42,
      for MNIST digit-0 sample 0. Each of 784 normalized pixels drove an
      independent Poisson channel at its intensity times 25 Hz. Black and
      red marks denote excitatory and inhibitory spikes.

      *(C–D)* Red squares denote COBA; black diamonds, PING. Values are
      means across three independent training replicates (seeds 42–44).
      *(C)* Validation accuracy across epochs for unpenalised conditions,
      without uncertainty intervals. *(D)* Final-checkpoint test accuracy
      versus mean per-neuron hidden-E firing rate over 200-ms presentations
      of the same 1,000 official MNIST test images. Activity ceilings were
      1, 2.5, 5, 10 and 25 Hz; stars mark the unpenalised condition.
      Horizontal and vertical bars show SEM across training replicates.
      Source experiment: #link("/exp025/")[exp025] — #link("/exp025/")[_Accuracy and Firing Rate With and Without Inhibition._]],
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
    caption: [*Post-training insertion of reciprocal inhibition.*
      *(A–B)* Illustrative 200-ms rasters from the validation-selected,
      unpenalised, loop-disabled COBA checkpoint (seed 42), showing the same
      MNIST digit-7 image at $s = 0$ and $s = 1$. Networks contained 1,024
      excitatory (E) and 256 inhibitory (I) neurons; black and red marks show
      fixed pseudorandom subsets of 200 E and 64 I neurons. Each of 784
      normalized pixels drove an independent Poisson channel at its intensity
      times 25 Hz.

      With learned input/readout weights frozen, E→I and I→E weights were
      newly initialized at each $s$, without retraining. Lower-clamped
      Gaussian parent means were $s$ and $2s$, respectively, with SDs equal
      to 10% of each mean. Matrices were divided by the number of presynaptic
      neurons, $N_"pre"$: $1 / 1024$ for E→I and $1 / 256$ for I→E,
      rather than $1 / sqrt(N_"pre")$ scaling.

      *(C–D)* Coupling sweep ($s = 0$–1, step 0.1) across three independently
      trained COBA classifiers (seeds 42–44), each evaluated on the same
      1,000 official MNIST test images for 200 ms each. *(C)* Per-neuron
      E rate (black circles) and I rate (red squares), averaged over all
      neurons and presentations. *(D)* Test accuracy (grey squares);
      dashed line, mean accuracy at $s = 0$. Curves and bands show means
      and sample SD across training replicates. Source experiment:
      #link("/exp038/")[exp038] — #link("/exp038/")[_Switching On the Inhibitory Loop._]],
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
    caption: [*Epoch-50 activity, probe responses and recurrent weights.*
      Input and readout weights were trained in all four PING conditions;
      recurrent E→I and I→E weights were fixed (Frozen) or trained from
      standard (Std.), one-tenth-standard (10%) or zero (Zero) initialization.
      Standard lower-clamped Gaussian parent means were 1 µS for E→I and
      2 µS for I→E, followed by $1 / N_"pre"$ scaling ($1 / 1024$ and
      $1 / 256$, respectively).

      *(A)* Official-test accuracy. *(B)* Per-neuron E rate (black) and
      I rate (red). Each replicate received the same 1,000 MNIST test images
      for 200 ms each through 784 independent, pixel-proportional Poisson
      channels (maximum-pixel rate, 25 Hz). Rates average all presentations
      and all 1,024 E or 256 I neurons.

      *(C)* Final, non-epoch-smoothed contrast
      $R_"contrast" = (A_"lobe" - A_"trough") / (A_"lobe" + A_"trough")$
      from the 1-ms-binned E-population autocorrelation. Every network
      received the same fixed digit-0, sample-0 Poisson encoding; this
      diagnostic does not estimate trial-to-trial variability. Bars and
      error bars in A–C show means ± SEM across three independent training
      replicates (seeds 42–44).

      *(D–E)* Pooled fractions of positive E→I and I→E weights.
      *(F–G)* Corresponding pooled arithmetic means, including zeros,
      in $10^(-3)$ µS. Each statistic combines 786,432 entries (three complete
      262,144-entry matrices) per direction and condition. Wide grey and
      narrow red bars denote initialization and epoch 50; no weight
      uncertainty is shown. Arrows denote relative changes ≥5%, not
      statistical significance. Source experiment:
      #link("/exp049/")[exp049] — #link("/exp049/")[_Training Recurrent Weights Weakens PING Rhythmicity._]],
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
    caption: [*Excitatory firing and spike counts per cycle across
      inhibitory decay times.* Eighteen epoch-50 PING classifiers comprised
      three independent training replicates (seeds 42–44) per decay time,
      each with 1,024 excitatory (E) and 256 inhibitory (I) neurons. All
      received the same 1,000 MNIST test images for 200 ms each through 784
      independent, pixel-proportional Poisson channels (maximum-pixel rate,
      25 Hz).

      *(A–B)* Mean per-neuron E rate $r_E$ and test accuracy versus
      spectral-peak frequency $f_gamma$. Rates average all E neurons and
      presentations; accuracy is the fraction of correctly classified test
      images. Frequency is the parabolically interpolated largest 5–150-Hz
      peak of the trial-averaged E-population Welch spectrum. Black points
      and horizontal/vertical bars show means ± SEM across three training
      replicates per condition. Labels in A give inhibitory decay times;
      the red dashed line is the equal-weight least-squares fit
      $r_E = a + p f_gamma$ to the six condition means ($a$, intercept;
      $p$, slope; $R^2$, coefficient of determination). B's dotted line
      marks 10% chance accuracy.

      *(C–H)* Fractions of E neuron–cycle pairs with 0, 1, 2 or ≥3 spikes at
      decay times of 4.5, 6, 9, 12, 18 and 27 ms, respectively. I-population
      bursts were detected after Gaussian smoothing of spike counts
      ($sigma = 1$ ms), with a 5%-of-maximum height threshold and minimum
      separation approximately half the network's measured period. Cycle
      boundaries were midpoints between successive burst peaks, with
      first/last intervals extending to presentation boundaries; trials
      without detected bursts were excluded. Distributions pool all E
      neurons, detected cycles and three training replicates per condition;
      no uncertainty bars are shown. Source experiments:
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

  The trained circuits tolerated substantial deletion of transmitted spike
  events, but diverged sharply under insertion. Independently deleting each
  naturally generated excitatory or inhibitory spike with 80% probability
  left mean test accuracy at 89.3% for COBA and 89.2% for PING; complete
  deletion reduced both to 10.6% (Fig. 7A). In contrast, inserting
  independent spike events into both populations’ transmitted activity at a
  nominal per-neuron rate equal to each network’s own unperturbed test-set
  excitatory firing rate reduced accuracy to 82.8% for COBA and 38.5% for
  PING. At twice that baseline rate, accuracy was 72.6% and 11.4%,
  respectively (Fig. 7B). These values average three independently trained
  classifiers per family, each evaluated on the same 1,000 test images
  without retraining. Thus, PING tolerated substantial deletion but was
  more sensitive than COBA to strong spike insertion at matched
  baseline-relative doses. Because both perturbations affected recurrent
  feedback and readout input, this comparison does not isolate disruption
  of rhythmic organisation as the cause of classification failure.

  #figure(
    data-image(
      data-file("exp110/robustness_compound.png"),
      width: 92%,
      alt: "COBA and PING accuracy under spike deletion and addition, followed by firing rate and accuracy across integration timesteps.",
    ),
    caption: [*Spike perturbations and integration timesteps in trained
      classifiers.* Networks contained 1,024 excitatory (E) and 256
      inhibitory (I) neurons. Tests used 1,000 MNIST test images, 200-ms
      presentations and 784 independent, pixel-proportional Poisson channels
      (25-Hz maximum-pixel rate). All summaries use three independent
      training replicates per model/condition (seeds 42–44); E rates average
      all E neurons and test presentations.

      *(A–B)* Inference-time perturbations of unpenalised COBA (red squares)
      and PING (black diamonds), without retraining, at a 0.1-ms timestep.
      Checkpoints were selected by minimum validation loss over 50 epochs. Lines
      show mean accuracy, shading ±1 sample SD and dashed lines 10% chance
      accuracy. *(A)* Independent deletion of naturally generated E/I spikes
      (0–100%, in 10-percentage-point steps) suppressed transmission without
      reversing neuronal resets. *(B)* Independent Bernoulli insertion into
      both populations (0–200% of each network’s unperturbed test-set E rate,
      in 10-percentage-point steps). The same nominal per-neuron insertion
      rate applied to E and I; replicate accuracy was averaged at matched
      percentages, not matched hertz. Transmission was capped at one spike
      per neuron per timestep, so collisions with existing spikes added
      nothing. Inserted events triggered neither resets nor refractory
      periods. Both perturbations preceded readout and subsequent recurrent
      transmission.

      *(C)* Mean E firing rate (black diamonds, left axis) and test accuracy
      (grey squares, right axis), with means ± SEM. PING networks were trained
      separately at timesteps of 0.05, 0.1, 0.25, 0.5 and 1 ms and evaluated
      at their final epoch-50 checkpoints using their training timestep.

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

  Rearranging inhibitory spike timing produced opposite effects on
  excitatory recruitment despite preserving every inhibitory neuron’s spike
  count. We replayed recorded inhibitory streams after either shifting
  individual spikes independently or applying a shared shift to spikes
  within fixed 22.8-ms clock windows, rather than detected gamma cycles. At
  a proposed jitter standard deviation of 14 ms, independent-spike shifts
  reduced mean excitatory firing from 16.6 Hz in the zero-jitter replay
  control to 0.008 Hz, while test accuracy fell from 89.8% to 11.9% (Fig.
  8A,C). Fixed-window group shifts instead increased excitatory firing to
  68.3 Hz, while accuracy declined to 82.5% (Fig. 8B,D). Mean replayed
  inhibitory firing remained unchanged at 108.3 Hz throughout both sweeps.
  These values average three independently trained classifiers, each
  evaluated on the same 1,000 test images. Thus, the temporal distribution
  of a fixed inhibitory spike count strongly influenced excitatory
  recruitment. This result concerns inhibitory replay rather than the
  intact feedback circuit, because delivered inhibition could no longer
  respond to ongoing excitatory activity.

  #figure(
    data-image(
      data-file("exp042/rhythm_compound.png"),
      width: 92%,
      alt: "Excitatory and inhibitory rasters, excitatory rate, accuracy and realised inhibitory rate under two inhibitory replay-jitter manipulations.",
    ),
    caption: [*Count-preserving inhibitory replay perturbations.*
      Three unpenalised PING classifiers (seeds 42–44; final epoch-50
      checkpoints) contained 1,024 excitatory (E) and 256 inhibitory (I)
      neurons. Conditions used the same 1,000 MNIST test images, 200-ms
      presentations, a 0.1-ms timestep and 784 independent, pixel-proportional
      Poisson channels (25-Hz maximum-pixel rate). Shifted recordings of
      unperturbed I spikes replaced naturally generated I outputs; E activity
      and readout responses were recomputed with unchanged weights.

      *(A–B)* Illustrative digit-7 responses (test sample 0, seed-42
      replicate) at $sigma = 14$ ms, where $sigma$ is the standard deviation
      of proposed Gaussian time shifts. Both panels show the same sampled
      200 E neurons (black) and 64 replayed I neurons (red) over the full
      presentation. *(A, C)* Independent-spike jitter assigned each I event
      a zero-mean Gaussian offset. *(B, D)* Fixed-window group jitter assigned
      a shared offset to all I events originating within each 22.8-ms clock
      window. Offsets were rounded to the simulation grid; boundary
      reflection and nearest-free-timestep collision resolution preserved
      each I neuron’s spike count within every presentation.

      *(C–D)* Mean per-neuron E rate (black circles) and replayed I rate
      (red squares), left axes; test accuracy (grey squares), right axes.
      Rates average all neurons in the respective population and all test
      presentations. Curves show means across training replicates without
      uncertainty intervals. Displayed jitter values are 0, 0.5, 1, 2, 5, 9
      and 14 ms in C, and 0, 1, 3, 7 and 14 ms in D; both share the same
      zero-jitter replay control. Source experiment:
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
      integrate-and-fire neurons. Training used variable maximum-pixel rates
      of 0.5–25 Hz; checkpoints were selected by minimum validation
      cross-entropy over 50 epochs, with validation accuracy breaking ties. Inference used
      unchanged weights and 784 independent, pixel-proportional Bernoulli
      spike channels at 0.1-ms resolution. Digits followed without gaps:
      hidden E/I state continued within each stream, while output membrane
      voltage and spike counts reset at supplied boundaries.

      *(A–C)* Illustrative 650-ms stream from the seed-42 network, selected
      by a predefined five-correct criterion. Digits 1, 7, 9, 5 and 2 used
      duration–maximum-pixel-rate pairs of (100 ms, 5 Hz), (200 ms, 7.5 Hz),
      (50 ms, 25 Hz), (100 ms, 15 Hz) and (200 ms, 10 Hz), respectively.
      *(A)* Input images, true→predicted labels and presentation conditions.
      *(B)* First 200 E neurons (black). *(C)* First 64 I neurons (red).
      Dotted vertical lines mark supplied digit boundaries.

      *(D)* Softmax-normalized cumulative output-spike counts per presentation
      (true class red; others grey), not calibrated probabilities.
      Classification selected the largest final count, equivalently the
      largest final share; ties selected the lowest class index. The dashed
      0.5-share line is not a decision threshold.

      *(E–F)* Accuracy on official MNIST test images: 40 five-digit streams
      per network/condition (200 decisions), with fixed duration and rate
      within each stream. *(E)* Mean percentage accuracy across three
      training replicates (colours and integer annotations) at durations
      of 25, 50, 100 and 200 ms and eleven indicated input rates; no
      uncertainty intervals are shown. *(F)* The same 200-ms measurements
      on a logarithmic input-rate axis: black circles and error bars show
      means ± SEM across training replicates. Source experiment:
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

  == Methods

  === Experimental design

  We combined untrained circuit simulations, trained MNIST classifiers,
  interventions on those classifiers and a separate mean-field model
  (Table 1). The inhibitory-timescale classifiers supplied both frequency
  and cycle-participation measurements; unpenalised classifiers from the
  accuracy–rate comparison were reused for loop insertion and spike
  perturbations.

  #let design-table = table(
    columns: (auto, auto, auto),
    table.header([Figure], [Design], [Checkpoint]),
    [1], [Untrained; 1,024 E, 256 I], [—],
    [2A–F], [Coupling grid; 256 E, 256 I], [—],
    [2G–I¹], [Mean-field model], [—],
    [3], [2 architectures × 6 activity conditions], [Final],
    [4], [Loop insertion; reused COBA], [Best validation],
    [5], [4 recurrent-training conditions], [Final],
    [6; 2I²], [6 inhibitory decay times], [Final],
    [7A–B], [Spike perturbations; reused COBA/PING], [Best validation],
    [7C], [5 timesteps], [Final],
    [8], [Inhibitory replay; reused PING], [Final],
    [9], [Variable-rate, spike-count training], [Best validation],
  )
  #counter(figure.where(kind: table)).update(0)
  #context figure(
    if target() == "html" {
      html.elem("div", attrs: (style: "display: flex; justify-content: center; overflow-x: auto;"), design-table)
    } else {
      align(center, design-table)
    },
    kind: table,
    caption: [*Experimental design and checkpoint policy.*
      E/I: excitatory/inhibitory neurons. Each training condition comprised
      three replicates. Final: epoch 50. Best validation: lowest mean
      validation cross-entropy across three encoding draws, with ties
      resolved by higher accuracy, then earlier epoch.
      ¹ Mean-field results. ² Spiking-model results.],
  ) <tab:experimental-design>

  === Spiking networks

  Hidden excitatory and inhibitory neurons followed conductance-based leaky
  integrate-and-fire dynamics:

  #math.equation(
    block: true,
    numbering: "(1)",
    $ C_m (dif V)/(dif t) = -g_L (V - E_L) - g_E (V - E_E) - g_I (V - E_I). $,
  ) <eq:hidden-neuron-dynamics>

  Here, $V$ is membrane voltage and $t$ is time. Membrane capacitance $C_m$
  was 1 nF for excitatory neurons and 0.5 nF for inhibitory neurons; leak
  conductance $g_L$ was 0.05 and 0.1 µS, respectively. The leak reversal
  potential $E_L$ was −65 mV, while excitatory and inhibitory reversal
  potentials $E_E$ and $E_I$ were 0 and −80 mV. Synaptic conductances $g_E$
  and $g_I$ varied with incoming spikes, as described below. At the spike
  threshold $V_"th" = -50$ mV, neurons emitted a spike and returned to the
  reset potential $V_"reset" = -65$ mV. Voltage remained at reset for an
  absolute refractory period $tau_"ref"$ of 3 ms in excitatory neurons and
  1.5 ms in inhibitory neurons.

  *P3 — Synapses and numerical updates.* Describe exponentially decaying
  conductances and event-triggered increments, with exponential-Euler membrane
  integration under conductances held constant during each step. AMPA decay
  was 2 ms; standard classifier GABA decay was 6 ms, while Figure 1 used
  9 ms. State the usual 0.1-ms timestep and distinguish the coupling-grid and
  timestep-sweep exceptions.

  *P4 — Connectivity and initialization.* Describe input→E, E→I, I→E and
  E→output projections; E→E and I→I remained disabled. Standard classifier
  recurrent parent means were 1 and 2 µS with 10% SD, lower-clamped at zero
  and divided by presynaptic population size. Input parent mean/SD were
  0.9/0.09, with 95% initial zeroing and survivor rescaling. Initial zeros
  in trainable projections could regrow.

  *Table 2 placement — Shared parameters and explicit exceptions.* Include
  the neuronal values above, initialization distributions, synaptic times and
  training settings. Distinguish Figure 1's input initialization of 1.5/0.3
  and recurrent means of 1.5/3 from the classifier defaults. Allocate exact
  discrete updates and initialization algorithms to Appendix A.

  === Classifier training

  *P5 — Images and data partitioning.* Describe flattening MNIST images to
  784 pixels and dividing intensities by 255. A seed-42 sample of 7,000
  official-training images was split, stratified by class, into 6,300
  optimization and 700 validation images. The official test partition
  remained separate. For the 1,000-image endpoint evaluations in Figs. 3–8,
  specify uniform sampling without replacement from the 10,000 official test
  images using fixed seed 42, without class stratification. The common subset
  and its ordering were independent of training seed. Distinguish this subset
  from single-image probes, illustrative rasters and the separately sampled
  streams in P27. Confirm selection against the executed revisions before
  converting this scaffold to final prose. Figs. 3–9.

  *P6 — Spike encoding and independent presentations.* Describe independent
  Bernoulli events with probability equal to normalized pixel intensity ×
  maximum-pixel rate × timestep in seconds. Standard presentations lasted
  200 ms at a 25-Hz maximum-pixel rate. Independent presentations started
  with fresh network state; continuous streams used the different boundary
  handling described below. Specify the separate, fixed evaluation-encoding
  random stream: the shared inference procedure restarted it for each
  evaluation and processed images in fixed order. Identical images and
  encoding settings used matched input draws across networks; do not imply
  identical spike trains across different timesteps, durations or input rates.
  Distinguish encoding randomness from separate perturbation draws in P23
  and P25. Confirm these settings against the executed revisions.

  *P7 — Output dynamics and ordinary classification.* Describe the output
  layer's dimensionless LIF state, 2-ms decay time, threshold 1 and
  subtractive reset. Ordinary classifiers used the mean pre-reset output
  state over the presentation as class logits. These output units should not
  be assigned the hidden neurons' millivolt parameters or refractory rules.
  Figs. 3–8.

  *P8 — Optimization and gradients.* State 50 epochs, AdamW, learning rate
  0.0004, zero weight decay, batch size 256 and gradient-norm clipping at 1.
  Training used a fast-sigmoid surrogate with slope 1. Voltage-gradient
  damping divided the hidden membrane-increment gradient by 1,000 in PING
  and by 1 in COBA; it did not change the forward voltage update. Describe
  nonnegative weight projection. Allocate full backward equations to Appendix A.

  *P9 — Activity regularization.* Give the one-sided quadratic penalty on
  each presentation's population-mean E firing rate, averaged over the
  minibatch. The coefficient was 0.041 Hz⁻²; ceilings were 1, 2.5, 5, 10
  and 25 Hz, plus an unpenalised condition. Only input and readout weights
  trained in this comparison. Fig. 3D.

  *P10 — Validation and checkpoint choice.* Validation averaged three
  encoding draws. Selection minimized mean validation cross-entropy, broke
  ties by higher mean accuracy and then earliest epoch. Explain that the
  endpoint analyses nevertheless used epoch 50, as specified in Table 1.
  Preserve this distinction when describing reused networks.

  === Untrained circuit experiments

  *P11 — Loop-off/on experiment.* Describe the 1,024-E/256-I circuits and
  400-ms trials. Illustrative rasters used 1,024 input channels at 5 Hz for
  COBA and 45 Hz for PING. The matched sweep instead used 784 channels at
  2, 5, 10, 20, 40, 70 and 100 Hz. Each condition used one seed-42 trial,
  without discarded burn-in. Fig. 1.

  *P12 — Coupling plane.* Describe 256 E and 256 I neurons, private
  100-Hz input with input weight 0.5, and the 11×11 grid spanning E→I
  means 0–3 and I→E means 0–6 µS. Simulations lasted 1 s at 0.25-ms
  resolution; measurements excluded the first 100 ms. There was one seeded
  network per condition. Fig. 2A–F.

  === Activity measurements

  *P13 — Firing rates and spectral peaks.* Define rates as spike count
  divided by neuron number and observation duration. For the
  inhibitory-timescale classifiers, demean each trial's E trace, compute a
  full-trial Hann-window Welch spectrum, average spectra over test
  presentations, then find the 5–150-Hz maximum with bounded parabolic
  interpolation. The 200-ms window gives 5-Hz bin spacing.
  Figs. 1E/G, 2I and 6A–B.

  *P14 — Autocorrelation contrast.* Describe 1-ms population-count bins,
  finite-overlap and squared-mean normalization, replacement of zero lag by
  its neighbor and three-point smoothing. Define the first trough from 2 ms
  onward, the preceding positive-lag maximum and their normalized difference.
  Figs. 2C and 5C. Allocate exact estimator edge cases to Appendix B.

  === Mean-field calculation

  *P15 — Four-variable model.* Present the E/I rate and two conductance
  equations. Rates relaxed toward noisy-LIF gains with 20/5-ms timescales;
  driving-force magnitudes were fixed at 65/15 mV, summed couplings at
  1/2 µS and effective voltage noise at 4 mV. Explicitly identify this as
  a separate, uncalibrated closure. Fig. 2G–I.

  *P16 — Oscillatory onset.* Describe fixed-point continuation over 401
  drives from 0 to 4 nA, centered-difference Jacobians with increment
  $10^(-6)$, and Brent refinement of the leading complex-pair crossing.
  Explain conversion of the eigenvalue's imaginary component to hertz.
  Fig. 2G.

  *P17 — Numerical criticality and decay dependence.* Describe
  upward/downward sweeps through 25 drives from onset minus 0.1 to onset
  plus 0.55 nA. LSODA integrations lasted 2 s per drive, carrying endpoint
  states forward; amplitude used the final 500 ms. Classification required
  a small branch gap, positive amplitude-squared slope and coefficient of
  determination greater than 0.9. Compare onset frequencies across six
  inhibitory decay times with medians across three classifier frequencies.
  Fig. 2H–I.

  *Appendix C allocation.* Include the gain integral, derivation, solver
  tolerances and numerical limitations. Do not imply that a first Lyapunov
  coefficient was calculated.

  === Classifier comparisons and recurrent coupling

  *P18 — Accuracy–rate comparison.* Describe 36 networks: two architectures,
  six activity conditions and three training replicates. Evaluate final
  checkpoints on the same 1,000 test images; distinguish these 200-ms
  measurements from the 400-ms illustrative rasters. Summaries use means
  and SEM across training replicates. Fig. 3.

  *P19 — Loop insertion.* Start from three validation-selected, unpenalised
  COBA classifiers. Hold input/readout weights fixed and initialize
  reciprocal matrices at strength values 0–1 in 0.1 increments, with parent
  means equal to strength and twice strength. No retraining occurred.
  Curves use means and sample SD. Fig. 4.

  *P20 — Trainable recurrence.* Describe four conditions: frozen recurrence
  and trainable recurrence initialized at standard, one-tenth-standard or
  zero strength, each with three training replicates. Epoch-50 accuracy/rates
  used 1,000 test images; contrast used one fixed digit-0 encoding. Weight
  statistics pooled complete matrices, including zeros. Fig. 5.

  === Inhibitory timescale and cycle participation

  *P21 — Inhibitory-timescale experiment.* Describe 18 separately trained
  networks at 4.5, 6, 9, 12, 18 and 27 ms. Evaluate final checkpoints on
  1,000 images. Fit an affine rate–frequency relationship to the six
  across-replicate means with equal weights. Fig. 6A–B. This uses means,
  unlike Figure 2I's medians.

  *P22 — Cycle counting.* Smooth I-population counts with a 1-ms Gaussian,
  detect peaks above 5% of the trial maximum with minimum separation
  approximately half the measured period, and place boundaries at integer
  midpoints. Extend outer intervals to presentation boundaries; exclude
  zero-peak trials. Pool E-neuron counts into 0, 1, 2 and ≥3 categories.
  Fig. 6C–H. Allocate edge conventions to Appendix B.

  === Spike perturbations and numerical resolution

  *P23 — Deletion and insertion.* Use validation-selected unpenalised
  classifiers, without retraining. Delete transmitted E/I events with
  probabilities 0–1 in 0.1 increments. Insert independent events at 0–200%
  of each network's baseline E rate in 10-percentage-point increments,
  applying that rate to both populations. Explain reset preservation,
  insertion without reset and event capping. Fig. 7A–B.

  *P24 — Timestep experiment.* Describe 15 independently trained networks:
  three at each of 0.05, 0.1, 0.25, 0.5 and 1 ms. Evaluate epoch-50
  weights at their training timestep, keeping presentations at 200 ms.
  Fig. 7C. No fixed-weight timestep-convergence experiment was performed.

  *P25 — Inhibitory replay.* Describe replacing recorded I outputs while
  recomputing E and readout responses. Independent jitter used 0, 0.5, 1,
  2, 5, 9 and 14 ms; group jitter used 0, 1, 3, 7 and 14 ms. Groups
  were fixed 228-step/22.8-ms windows. Reflection and nearest-free-step
  collision resolution preserved per-neuron counts. Both arms shared
  zero-jitter replay. Fig. 8.

  *Appendix D allocation.* Include exact replay quantization, reflection and
  collision-resolution rules. Retain in the
  main paragraph that replay interrupts responsive feedback.

  === Continuous streams

  *P26 — Streaming-specific training.* Describe three separately trained
  classifiers using spike-count logits and uniformly sampled per-presentation
  rates from 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15 and 25 Hz.
  Readout initialization mean/SD were 0.05/0.04, unlike the ordinary
  classifiers' approximately 1.1206/0.8350. Use validation-selected
  checkpoints. Fig. 9.

  *P27 — Stream evaluation.* Describe 40 five-digit streams per
  network–duration–rate condition, processed in batches of five streams.
  Durations were 25, 50, 100 and 200 ms. Images were sampled without
  replacement within each stream from the full 10,000-image official test
  partition, without class stratification; images could recur across streams.
  Sampling depended on network seed, duration and input rate, so conditions
  did not share one fixed 1,000-image subset. Encoding used a separate random
  stream indexed by network seed and stream number; its seed was reused
  across conditions without implying identical spike trains when duration
  or rate changed. Hidden state continued; output state/counts reset at
  supplied boundaries. Fig. 9E–F.

  *P28 — Decisions and illustration selection.* Classify by largest final
  output count; ties choose the lowest class index. Explain the plotted
  softmax shares. The illustrative sequence used predefined duration–rate
  pairs and the first candidate with five correct decisions; the first
  candidate qualified. Fig. 9A–D. Keep the sampling and selection account in Methods.

  === Statistical reporting and reproducibility

  *P29 — Replication, aggregation and exclusions.* Define the independent
  unit for across-network summaries as one independently initialized and
  trained network, with three training replicates per condition (seeds
  42–44). Describe which stochastic processes varied with training seed:
  initialization, minibatch ordering and training encodings, plus sampled
  input rates for streaming training. The data split and evaluation-image
  subset were fixed separately. Repeated interventions on one network were
  repeated measurements, not additional training replicates; neurons,
  presentations and neuron–cycle pairs were not independent network
  replicates. Distinguish these designs from the single-seed untrained
  probes and deterministic mean-field calculations in Table 1.

  State explicitly: SEM for Figs. 3, 5, 6A–B, 7C and 9F; sample SD for
  Figs. 4 and 7A–B; pooled counts without intervals for Fig. 6C–H; means
  without displayed intervals for Fig. 8. Define sample SD using the
  replicate-count-minus-one denominator and SEM as sample SD divided by
  the square root of the training-replicate count. Compute each network's
  summary before aggregating across networks; distinguish the pooled
  neuron–cycle distributions, which weight networks by their available
  pair counts. Describe these summaries and fitted relationships as
  descriptive; do not add unperformed significance tests or confidence
  intervals.

  Report the basis for choosing three training replicates and the evaluation
  sample counts if documented; the rationale remains to be established,
  and a power calculation must not be invented. State any failed or excluded
  training runs, evaluations or undefined measurements and their handling,
  checking execution records before claiming that none occurred. Retain the
  zero-detected-burst exclusion in P22 and distinguish measurement exclusions
  from the illustrative-stream selection in P28. Report the resulting
  denominators where exclusions affect summaries.

  *P30 — Reproducibility.* Identify the executed source revisions and
  environments rather than today's defaults. The inspected training
  configurations record CUDA execution and PyTorch 2.11.0+cu128; the
  mean-field calculation uses SciPy. Allocate per-execution software and
  provenance details to the accompanying reproducibility record rather than
  claiming one environment covered every stage.

  == Appendix A — Discrete dynamics and gradient calculations

  *A1 — Discrete neuronal and synaptic updates.* Give the actual recurrence
  equations implementing the continuous equations in Methods: exponential
  conductance decay, event increments, effective membrane equilibrium and
  exponential-Euler update. Specify the ordering of recurrent transmission,
  threshold evaluation, refractory-counter updates and reset. Define the
  timestep indexing so the transmission delay is unambiguous.
  Supports Methods P3; Figs. 1–8.

  *A2 — Output-state recurrence.* Give the output update, including its
  timestep-dependent input factor, pre-reset evidence accumulation and
  subtractive reset. Show precisely where mean-voltage and spike-count
  accumulation diverge. Supply the implementation equations without repeating
  the readout definitions or parameter values. Supports P7 and P26; Figs. 3–9.

  *A3 — Surrogate and damped derivatives.* Write the fast-sigmoid surrogate
  derivative and the forward-preserving gradient-scaling operation. Show which
  membrane-increment derivatives receive damping, which direct state paths
  remain, and how hidden hard resets differ from differentiable output
  subtractive resets. Do not repeat optimizer settings. Supports P8.

  *A4 — Initialization transformation.* Express the executed sequence
  algebraically: Gaussian draw, lower clamp, independent initial-zero mask,
  survivor rescaling and division by presynaptic population size. Explain why
  the parent mean differs from the realized matrix mean. Distinguish the
  output initializer, which uses stored-weight parameters directly. Refer to
  Table 2 for values. Supports P4 and P26.

  == Appendix B — Measurement algorithms and boundary cases

  *B1 — Spectral interpolation.* Give the three-bin parabolic interpolation
  formula on linear spectral power. Specify the half-bin correction limit and
  zero-offset behavior at spectrum boundaries or zero curvature. Explain
  treatment of unusable traces where applicable. Leave window lengths,
  frequency ranges and averaging order in Methods.
  Supports P13; Figs. 1E/G, 2I and 6A–B.

  *B2 — Autocorrelation implementation.* Give the FFT construction and
  finite-overlap normalization explicitly. Specify discarded incomplete bins,
  treatment of zero lag, smoothing at array boundaries, the asymmetric
  inequalities used to identify a trough, and first-maximum tie handling.
  State when contrast is undefined rather than replaced by zero.
  Supports P14; Figs. 2C and 5C.

  *B3 — Cycle-boundary discretization.* Specify the Gaussian kernel's
  ±4-standard-deviation support, integer rounding of minimum peak separation,
  floor-rounded midpoint boundaries and half-open counting intervals. Explain
  the single-detected-burst case: its counting interval spans the entire
  presentation. Clarify these consequences of the cycle definition without
  repeating the detection protocol or pooled results.
  Supports P22; Fig. 6C–H.

  == Appendix C — Mean-field closure and numerical specification

  *C1 — Gain function and closure assumptions.* Give the noisy-LIF gain
  integral, its reset/threshold integration bounds and conversion from mean
  current to effective mean voltage. Explain the substitutions leading from
  voltage-dependent synaptic currents to fixed driving forces and
  population-rate relaxation. Distinguish assumptions from derivations; do not
  repeat the four final equations or their parameter table.
  Supports P15; Fig. 2G–I.

  *C2 — Numerical solution details.* Specify initialization of the fixed-point
  solver and continuation between neighboring drives, failed-solution handling,
  numerical quadrature settings and the selection of the leading complex
  eigenvalue. Put the precise Brent and LSODA tolerances and maximum
  integration step in Appendix Table C1, rather than repeating the drive grids
  and integration durations. Supports P16–P17.

  *Appendix Table C1 placement — Numerical solver settings.* Record the
  quadrature, root-refinement and integration settings needed to reproduce
  the displayed mean-field calculation; omit protocol values already in Methods.

  *C3 — Criticality calculation.* Define the branch gap as the maximum
  absolute difference between matched upward/downward amplitudes. State its
  threshold of $10^(-4)$ inverse milliseconds. Specify that the
  amplitude-squared regression used upward-branch points above onset, included
  an intercept and used centered total sum of squares. Explain handling of
  insufficient points or zero variance. Supports P17; Fig. 2H.

  == Appendix D — Exact replay transformations

  *D1 — Quantization and boundary reflection.* Give the conversion of
  Gaussian offsets to integer timestep shifts and the repeated-reflection
  mapping into a bounded interval. For independent jitter, apply reflection
  to event destinations. For group jitter, derive the allowable shared
  displacement from the earliest and latest events in the source window.
  Explain the degenerate interval case. Supports P25; Fig. 8.

  *D2 — Collision resolution.* Describe stable event ordering and resolution
  of duplicate same-neuron destinations by searching offsets in the order
  +1, −1, +2, −2, continuing outward and skipping out-of-range candidates.
  Explain that count preservation survives this operation, although collision
  adjustment can change within-group relative timing. Supports P25; Fig. 8.
  #metadata("exp110-end")
]

#let body = if inputs.all(key => data-file(key) != none) {
  render-report(data-file)
} else [
  A required run is unavailable, so there is no content to display yet.
]
