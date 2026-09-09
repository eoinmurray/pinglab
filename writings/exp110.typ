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
  updated_at: "2026-09-09",
  description: "A manuscript scaffold connecting PING circuit dynamics, low-rate task performance, cycle participation, perturbation sensitivity and continuous-stream classification.",
  collection: "gamma-gated-sparsity",
)

#let inputs = (
  "exp023",
  "exp025",
  "exp037",
  "exp038",
  "exp042",
  "exp044",
  "exp049",
  "exp110",
  "exp082",
)

// Editing-only paragraph-group labels. Set to false for clean output.
#let show-editing-paragraph-labels = true
#let editing-paragraph-label(label) = if show-editing-paragraph-labels {
  context {
    if target() == "html" {
      html.elem(
        "div",
        attrs: (style: "color: #b42318; font-size: 0.9em; font-weight: 400; line-height: 1; margin: 0.45rem 0 -0.7rem;"),
        label,
      )
    } else {
      block(above: 0.4em, below: 0.05em)[
        #text(size: 9pt, weight: "regular", fill: rgb("#b42318"))[#label]
      ]
    }
  }
}

#let render-report(data-file) = [
  #set heading(numbering: none)
  #set math.equation(numbering: "(1)")
  #counter(math.equation).update(0)
  #show math.equation.where(block: true): equation => context {
    if target() == "html" {
      html.elem("div", attrs: (
        class: "pinglab-numbered-equation",
        style: "display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;gap:1em",
      ), {
        html.elem("div", attrs: (style: "min-width:0;overflow-x:auto;overflow-y:hidden"), equation)
        html.elem(
          "span",
          attrs: (class: "pinglab-equation-number"),
          numbering(equation.numbering, ..counter(math.equation).at(equation.location())),
        )
      })
    } else {
      equation
    }
  }
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

  #editing-paragraph-label("P1")
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

  #editing-paragraph-label("P2")
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

  #editing-paragraph-label("P3")
  To determine whether this regime required a narrow parameter choice, we
  mapped the E→I and I→E initialization means across an 11 × 11 coupling
  plane in untrained networks containing 1,024 excitatory and 256 inhibitory
  neurons under fixed 100-Hz Poisson drive (Fig. 2A–C). We measured
  temporal structure using lobe–trough contrast, the normalized difference
  between the autocorrelation lobe and trough. When either pathway was
  absent, excitatory firing remained near 169 Hz and lobe–trough contrast
  was near zero. With both pathways present, stronger reciprocal coupling
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
      (1,024 E, 256 I; one seed) with private 100-Hz Poisson input, a
      0.1-ms timestep and 6-ms GABA decay, evaluated for 0.9 s after a
      0.1-s burn-in. B's grayscale is clipped at its 92nd
      percentile. Contrast is
      $(A_"lobe" - A_"trough") / (A_"lobe" + A_"trough")$ from the 1-ms-binned
      E autocorrelogram.

      *(D–F)* Illustrative 200-ms rasters at the conditions marked in C:
      $(W_(E I), W_(I E)) = (0, 0)$, $(0.6, 1.2)$ and $(3, 6)$ µS,
      respectively.
      Black and red marks show the first 160 E and 48 I neurons.
      A–F show no uncertainty estimates.

      *(G–H)* Separate four-variable mean-field conductance model with
      4-mV effective voltage noise, 6-ms GABA decay and 1.2/0.6-ms E/I
      refractory periods. *(G)* Fixed-point eigenvalues over
      external drive $I_"ext" = 0$–4 nA (colour); cyan circles mark the leading
      conjugate pair at $I_"ext"^* = 0.594$ nA. *(H)* Peak-to-peak E-rate
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

  #editing-paragraph-label("P4")
  The selected rasters make this progression concrete. Without reciprocal
  coupling, excitatory neurons fired densely while the inhibitory population
  remained silent, and E-population lobe–trough contrast was near zero (0.00030; Fig. 2D). At intermediate coupling ($W_(E I)=0.6$, $W_(I E)=1.2$ µS),
  recurring inhibitory volleys appeared alongside sparser excitatory firing,
  with contrast increasing to 0.268 (Fig. 2E). Under strong coupling
  ($W_(E I)=3$, $W_(I E)=6$ µS), inhibitory volleys were highly regular and
  excitatory firing was sparse, while contrast reached 0.989 (Fig. 2F).
  Across these conditions, dense, weakly structured activity gave way to sparse
  excitatory firing clustered around increasingly regular inhibitory volleys.
  These are selected conditions from the same single-seed sweep.

  #editing-paragraph-label("P5")
  To examine a possible dynamical basis for this oscillatory onset, we analysed
  a separate four-variable mean-field conductance model. As external drive
  $I_"ext"$ increased, the leading complex-conjugate eigenvalues crossed from
  negative to positive real parts at $I_"ext"^* = 0.594$ nA, with an imaginary
  component corresponding to 27.6 Hz
  (Fig. 2G). Above this crossing, peak-to-peak excitatory-rate amplitude
  increased continuously from near zero, while upward and downward drive
  sweeps nearly coincided with no resolved hysteresis (Fig. 2H). The positive
  amplitude-squared slope and its $R^2 = 0.999$ fit therefore met the predefined
  numerical criteria for a supercritical Hopf onset. This classifies the
  sampled mean-field transition numerically; criticality was not established
  analytically.

  #editing-paragraph-label("P6")
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

  #editing-paragraph-label("P7")
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

  #editing-paragraph-label("P8")
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

  #editing-paragraph-label("P9")
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

  Training from standard or one-tenth-standard initialization eliminated most
  E→I connections, while most I→E connections remained positive and their
  mean strength increased (Fig. 5D–G). Mean E→I strength fell from standard
  initialization but rose from one-tenth-standard initialization;
  zero-initialized recurrence remained zero. Thus, weak temporal contrast
  accompanied asymmetric recurrent reorganization, without isolating any
  weight change as its cause.

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

  #editing-paragraph-label("P10")
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

  #editing-paragraph-label("P11")
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

  #editing-paragraph-label("P12")
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
      data-file("exp037/perturbation_curves.svg"),
      width: 92%,
      alt: "COBA and PING test accuracy under spike deletion and baseline-relative spike insertion.",
    ),
    caption: [*Spike perturbations in trained classifiers.* Networks contained
      1,024 excitatory (E) and 256 inhibitory (I) neurons. Tests used 1,000
      MNIST test images, nominally 200-ms
      presentations and 784 independent, pixel-proportional Poisson channels
      (25-Hz maximum-pixel rate). All summaries use three independent
      training replicates per model/condition (seeds 42–44).

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

      Source experiment: #link("/exp037/")[exp037] — #link("/exp037/")[_Dropped Spikes vs Added Noise._]],
  ) <fig:robustness>

  #editing-paragraph-label("P13")
  Classification accuracy remained near 90% across tested timesteps, although
  excitatory firing varied. This supports robustness after separate training
  at each timestep, without establishing fixed-weight convergence (Appendix A5).

  #editing-paragraph-label("P14")
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

  #editing-paragraph-label("P15")
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

  #editing-paragraph-label("P16")
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

  #editing-paragraph-label("P17")
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
    [2A–F], [Coupling grid; 1,024 E, 256 I], [—],
    [2G–I¹], [Mean-field model], [—],
    [3], [2 architectures × 6 activity conditions], [Final],
    [4], [Loop insertion; reused COBA], [Best validation],
    [5], [4 recurrent-training conditions], [Final],
    [6; 2I²], [6 inhibitory decay times], [Final],
    [7A–B], [Spike perturbations; reused COBA/PING], [Best validation],
    [App. A1], [5 timesteps; 0.05–0.6 ms], [Final],
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

  #editing-paragraph-label("P18")
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
  absolute refractory period $tau_"ref"$ of 1.2 ms in excitatory neurons and
  0.6 ms in inhibitory neurons.

  #editing-paragraph-label("P19")
  Synaptic conductances decayed exponentially and increased upon incoming
  spikes:

  #math.equation(
    block: true,
    numbering: "(1)",
    $ g_x[k+1] = g_x[k] exp(- (Delta t_"sim") / tau_x)
      + sum_j w_(x j) s_j[k], quad x in {E, I}. $,
  ) <eq:synaptic-conductance-update>

  Here, $g_x[k]$ is the excitatory or inhibitory conductance at timestep $k$,
  $w_(x j)$ is the conductance increment from presynaptic source $j$, and
  $s_j[k] in {0, 1}$ indicates a spike delivered during that update. The
  excitatory decay time $tau_E = tau_"AMPA"$ was 2 ms; the inhibitory decay
  time $tau_I = tau_"GABA"$ was 6 ms, except in the inhibitory-timescale
  experiment, which used 4.5, 6, 9, 12, 18 and 27 ms. Membrane voltage
  was integrated using exponential Euler; update ordering is specified in
  Appendix A1. The integration timestep $Delta t_"sim"$ was
  ordinarily 0.1 ms. The timestep experiment used 0.05, 0.1, 0.2,
  0.3 and 0.6 ms, each representing both refractory periods exactly as
  integer numbers of steps. Exact updates and recurrent transmission timing
  are specified in Appendix A.

  We reused the fixed-0.1-ms spiking measurements, whose executed E/I
  refractory holds were already 1.2/0.6 ms. We recomputed the timestep
  comparison and the separate mean-field calculation with these same
  refractory durations; the remaining spiking measurements were reused.

  #editing-paragraph-label("P20")
  Classifier networks contained input→E, E→I, I→E and E→output projections,
  with E→E and I→I connections disabled; COBA additionally disabled reciprocal
  E–I coupling. Input and recurrent weights used lower-clamped Gaussian draws
  with fan-in normalization (Appendix A4; Table A1). Here, $N_"pre"$ is the
  presynaptic population size; input weights were initially zeroed with probability
  $q_"zero" = 0.95$ but remained trainable. Readout weights were
  dimensionless and initialized directly without fan-in normalization.

  === Classifier training

  #editing-paragraph-label("P21")
  MNIST images were flattened into 784-element vectors and their pixel
  intensities divided by 255. We sampled 7,000 images uniformly without
  replacement from the official training partition, then divided this subset
  into 6,300 optimization images and 700 validation images using a
  class-stratified split. Both sampling and splitting used seed 42, giving
  the same data partitions across training replicates. The official
  10,000-image test partition remained separate from optimization and
  checkpoint selection. Endpoint evaluations in Figs. 3–8 and Appendix
  Fig. A1 used a common subset of 1,000 test images sampled uniformly without
  replacement using
  seed 42, without class stratification. This subset and its presentation
  order were fixed independently of training seed. Single-image probes and
  illustrative rasters used separately specified examples; continuous-stream
  evaluations sampled from the full official test partition, as described
  below.

  #editing-paragraph-label("P22")
  Each normalized pixel drove an independent Bernoulli spike channel:

  #math.equation(
    block: true,
    numbering: "(1)",
    $ s_j[k] tilde.op "Bernoulli"(p_j), quad
      p_j = a_j r_("input,max") Delta t_"sim". $,
  ) <eq:input-spike-encoding>

  Here, $s_j[k] in {0, 1}$ indicates an input spike from pixel $j$ at
  timestep $k$, $a_j in [0, 1]$ is its normalized intensity, and $p_j$ is
  its per-step spike probability. The maximum-pixel rate $r_("input,max")$
  was ordinarily 25 Hz, and the integration timestep $Delta t_"sim"$ is
  expressed in seconds in this equation. Draws were independent across
  pixels, timesteps and presentations, with at most one spike per channel
  per step.

  Independent image presentations lasted nominally 200 ms and began with
  freshly initialized hidden and output states. Training generated fresh
  encoding draws on each presentation. Endpoint encodings were matched across
  networks at common input settings, with independent perturbation randomness
  (Appendix B5). Variable input rates and state continuity between successive
  images in the streaming experiment are described below.

  #editing-paragraph-label("P23")
  The output layer contained ten leaky integrate-and-fire units, one per
  digit class. Each unit had a dimensionless state $u_c$, where
  $c in {0, dots, 9}$ denotes the class, and an output decay time
  $tau_"out" = 2$ ms. Weighted excitatory spikes drove this state. At each
  timestep, a pre-reset state at or above the output threshold
  $theta_"out" = 1$ produced one spike, after which the threshold was
  subtracted from the state. Output units had no refractory period and
  began each independent presentation at zero.

  For ordinary classification (Figs. 3–8 and Appendix Fig. A1), class logits
  and predictions were

  #math.equation(
    block: true,
    numbering: "(1)",
    $ z_c = 1 / N_t sum_(k=1)^(N_t) tilde(u)_c[k], quad
      hat(y) = #math.op("arg max", limits: true)_c z_c, $,
  ) <eq:output-classification>

  where $tilde(u)_c[k]$ is the output state before subtractive reset at
  timestep $k$, $N_t$ is the number of simulation steps in the presentation,
  $z_c$ is the class logit, and $hat(y)$ is the predicted digit. Thus,
  classification used the mean pre-reset state across the full presentation.
  Exact output updates are given in Appendix A; the streaming experiment’s
  spike-count readout is described below.

  #editing-paragraph-label("P24")
  We trained classifiers for 50 epochs using AdamW with learning rate
  $eta = 0.0004$, zero weight decay and minibatches of 256 images. The
  objective was mean cross-entropy, supplemented by the activity penalty
  described below where applicable. We used backpropagation through time
  over each presentation and clipped the global Euclidean gradient norm to
  a maximum of 1 before each optimizer update. Trainable connection weights
  were subsequently clamped below at zero.

  Spikes remained binary during forward simulation. During backpropagation,
  the threshold derivative was replaced by the fast-sigmoid surrogate

  #math.equation(
    block: true,
    numbering: "(1)",
    $ (∂ s)/(∂ xi) approx beta / (1 + beta abs(xi))^2, $,
  ) <eq:surrogate-gradient>

  where $s$ is the spike indicator, $xi$ is the numerical distance from
  threshold—expressed in millivolts for hidden neurons and normalized units
  for output neurons—and the surrogate slope $beta = 1$. Gradients through
  each hidden excitatory and inhibitory membrane increment were additionally
  divided by the voltage-gradient damping factor $d_"grad" = 1,000$ in PING
  and $d_"grad" = 1$ in COBA. Forward dynamics were unchanged; output updates
  were undamped. Exact gradient paths are specified in Appendix A3.

  #editing-paragraph-label("P25")
  For the accuracy–rate comparison (Fig. 3D), we added a one-sided quadratic
  penalty to the classification loss:

  #math.equation(
    block: true,
    numbering: "(1)",
    $ L_"total" = L_"CE" + lambda_"rate" / B sum_(b=1)^B
      [(max(0, r_(E,b) - r_(E,"ceil"))) / (1 thin "Hz")]^2. $,
  ) <eq:activity-penalty>

  Here, $L_"total"$ is the training objective, $L_"CE"$ is mean
  classification cross-entropy, $B$ is the minibatch size and $b$ indexes
  image presentations. The mean excitatory firing rate $r_(E,b)$ was
  calculated separately for each presentation by dividing its total
  excitatory spike count by the excitatory population size and presentation
  duration in seconds. Division by $1 thin "Hz"$ expresses the rate excess
  as a dimensionless numerical value. The dimensionless penalty coefficient
  $lambda_"rate" = 0.041$ was fixed across penalised conditions, and
  the activity ceiling $r_(E,"ceil")$ took values of 1, 2.5, 5, 10 and
  25 Hz. An additional unpenalised condition used $lambda_"rate" = 0$.

  The penalty was applied to each presentation’s population-mean rate before
  averaging over the minibatch. Rates at or below the ceiling incurred no
  penalty; exceeding it increased the loss quadratically, without imposing
  a hard firing-rate limit. Only input and readout weights were trained in
  this comparison; recurrent weights remained fixed, with reciprocal
  coupling enabled in PING and disabled in COBA.

  #editing-paragraph-label("P26")
  After each training epoch, we evaluated the 700 validation images using
  three independently seeded spike-encoding draws. The same encoding seeds
  were reused across epochs, keeping stochastic inputs fixed for checkpoint
  comparisons. For variable-rate classifiers, validation input-rate draws
  were also fixed across epochs. Cross-entropy and accuracy were averaged
  over validation images and encoding draws; the activity penalty was
  excluded from the validation loss.

  For each training replicate, the best-validation checkpoint was the saved
  network with the lowest mean validation cross-entropy among epochs 1–50.
  Ties were resolved by higher mean validation accuracy and then the earliest
  epoch. We also retained the final checkpoint from epoch 50, irrespective
  of its validation ranking. Subsequent analyses used the checkpoint
  specified in Table 1: best-validation checkpoints for loop insertion,
  spike deletion/insertion and continuous-stream classification, and final
  checkpoints for the accuracy–rate, trainable-recurrence,
  inhibitory-timescale, timestep and inhibitory-replay analyses. Reusing a
  trained network did not imply that every analysis used the same checkpoint.

  === Untrained circuit experiments

  #editing-paragraph-label("P27")
  We compared untrained loop-disabled COBA and PING circuits containing
  1,024 excitatory and 256 inhibitory neurons (Fig. 1). Reciprocal E→I
  and I→E coupling was absent in COBA and enabled in PING, using the
  initialization parameters specified in Table A1. Each simulation lasted
  400 ms at a 0.1-ms timestep, beginning with membrane voltages of −65 mV
  and zero synaptic conductances; no burn-in interval was discarded.
  Illustrative rasters used 1,024 independent Poisson input channels at
  5 Hz per channel for COBA and 45 Hz for PING. The matched input–output
  sweeps instead used 784 channels, testing both architectures at 2, 5,
  10, 20, 40, 70 and 100 Hz per channel. Each architecture–input condition
  comprised one trial with seed 42. Population firing rates were measured
  over the full trial, and spectra were calculated from the illustrative
  raster trials as described below.

  #editing-paragraph-label("P28")
  We mapped reciprocal coupling in untrained networks containing 1,024
  excitatory and 256 inhibitory neurons (Fig. 2A–F). Each excitatory neuron
  received its own independent 100-Hz input channel through a fixed input
  weight of 0.5 µS. The E→I initialization parent mean $mu_(E arrow.r I)$
  ranged from 0 to 3 µS in 0.3-µS increments, and the I→E parent mean
  $mu_(I arrow.r E)$ ranged from 0 to 6 µS in 0.6-µS increments, forming
  an $11 times 11$ grid. Recurrent weights used Gaussian draws with
  standard deviations equal to 10% of their respective means, followed by
  lower clamping at zero and division by presynaptic population size.
  Each grid condition comprised one network and one trial with seed 42.
  Simulations lasted 1 s at a 0.1-ms timestep, with an inhibitory decay
  time of 6 ms. We discarded the first 100 ms and measured population
  firing rates and excitatory-population autocorrelation contrast over the
  remaining 900 ms. Illustrative rasters show the first 200 ms after
  burn-in at three selected grid conditions.

  === Activity measurements

  #editing-paragraph-label("P29")
  Mean per-neuron firing rates were calculated as

  #math.equation(
    block: true,
    numbering: "(1)",
    $ r_P = n_("spike",P) / (N_P T_"obs"), quad P in {E, I}, $,
  ) <eq:population-firing-rate>

  where $r_P$ is the firing rate of population $P$, $n_("spike",P)$ is
  its total spike count, $N_P$ is its neuron count and $T_"obs"$ is the
  observation duration in seconds, summed across presentations when pooling
  trials.

  For spectral analysis, we formed each presentation’s excitatory-population
  trace by averaging binary spike indicators across neurons at each
  simulation step. We subtracted the trace mean and estimated its power
  spectral density using Welch’s method with a single Hann window spanning
  the full presentation. For the inhibitory-timescale classifiers, spectra
  were averaged across the 1,000 test presentations within each network
  before locating the largest peak between 5 and 150 Hz. The spectral-peak
  frequency $f_gamma$ was refined by parabolic interpolation (Appendix B1).
  The 200-ms presentations gave a frequency-bin spacing of 5 Hz.

  The illustrative spectra in Fig. 1E,G instead used individual 400-ms
  trials, giving 2.5-Hz bin spacing. Their peak annotation additionally
  required inhibitory spiking; absence of a marker was not a statistical
  test for rhythmicity. Interpolation formulas and boundary cases are
  specified in Appendix B.

  #editing-paragraph-label("P30")
  We quantified excitatory-population temporal structure using
  autocorrelation lobe–trough contrast (Figs. 2C and 5C). Spikes were pooled
  across excitatory neurons into 1-ms count bins. We normalized and smoothed
  the uncentered autocorrelogram over lags up to 100 ms as specified in
  Appendix B2.

  We identified the first local minimum at a lag of at least 2 ms and the
  maximum at positive lags preceding that minimum. Contrast was defined as

  #math.equation(
    block: true,
    numbering: "(1)",
    $ R_"contrast" = (A_"lobe" - A_"trough") / (A_"lobe" + A_"trough"), $,
  ) <eq:autocorrelation-contrast>

  where $A_"lobe"$ and $A_"trough"$ are the smoothed autocorrelation
  values at the preceding maximum and first minimum, respectively. The
  dimensionless contrast $R_"contrast"$ measures their relative separation.
  Coupling-grid measurements used the 900-ms post-burn-in interval;
  classifier measurements used the fixed reference-image presentation
  described below.

  The coupling-grid analysis retained undefined contrast values as missing,
  whereas the training diagnostic supplying Fig. 5C assigned zero when
  contrast could not be calculated. Exact feature-selection rules,
  undefined-value conditions and boundary conventions are specified in
  Appendix B.

  === Mean-field calculation

  #editing-paragraph-label("P31")
  We examined oscillatory onset using a separate four-variable mean-field
  model (Fig. 2G–I). Its state comprised excitatory and inhibitory population
  rates $r_E$ and $r_I$, excitatory conductance onto inhibitory neurons
  $g_E^I$, and inhibitory conductance onto excitatory neurons $g_I^E$:

  #math.equation(
    block: true,
    numbering: "(1)",
    $ tau_(r,E) dot(r)_E &= -r_E + Phi_E (I_"ext" - Delta V_"inh" g_I^E), \
      tau_(r,I) dot(r)_I &= -r_I + Phi_I (Delta V_"exc" g_E^I), \
      dot(g)_E^I &= -g_E^I / tau_"AMPA" + G_(E arrow.r I) r_E, \
      dot(g)_I^E &= -g_I^E / tau_"GABA" + G_(I arrow.r E) r_I. $,
  ) <eq:mean-field-model>

  Here, dots denote derivatives with respect to time in milliseconds;
  rates were expressed in $"ms"^(-1)$ and conductances in µS. The external
  current $I_"ext"$, measured in nA, drove only the excitatory population.
  The functions $Phi_E$ and $Phi_I$ mapped mean input current to steady-state
  firing rate using noisy leaky integrate-and-fire gains.

  The rate-relaxation constants $tau_(r,E) = 20$ ms and $tau_(r,I) = 5$ ms
  matched the respective membrane time constants. Synaptic decay constants
  were $tau_"AMPA" = 2$ ms and, unless varied, $tau_"GABA" = 6$ ms.
  Fixed driving-force magnitudes were $Delta V_"exc" = 65$ mV and
  $Delta V_"inh" = 15$ mV; summed conductance increments were
  $G_(E arrow.r I) = 1$ µS and $G_(I arrow.r E) = 2$ µS. The gains used
  the leak, reset and threshold parameters specified above, refractory
  periods $tau_("ref",E) = 1.2$ ms and $tau_("ref",I) = 0.6$ ms, and an
  effective voltage-noise scale $sigma_V = 4$ mV.

  This phenomenological closure fixed synaptic driving forces at rest,
  omitted conductance-dependent shunting and assumed a noise scale that was
  neither measured nor fitted to spiking activity. The population equations
  were deterministic; noise entered through the gain functions. The gain
  integral and closure assumptions are specified in Appendix C.

  #context {
    let note = [*Draft note:* Double-check the effective voltage-noise scale
      $sigma_V = 4$ mV.]
    if target() == "html" {
      html.elem("div", attrs: (style: "color: red;"), note)
    } else {
      text(fill: red, note)
    }
  }

  #editing-paragraph-label("P32")
  We continued fixed points across external currents $I_"ext" = 0$–4 nA
  and located onset at $I_"ext"^*$, where the leading complex-conjugate
  eigenvalue pair first crossed from negative to nonnegative real part
  (Fig. 2G). Eigenvalues came from the continuous-time Jacobian $J_"flow"$
  of the four-variable model; continuation and refinement details are given
  in Appendix C2. The onset frequency was

  #math.equation(
    block: true,
    numbering: "(1)",
    $ f_"Hopf" = (1000 abs(op("Im") lambda_J^*)) / (2 pi), $,
  ) <eq:mean-field-onset-frequency>

  where $lambda_J^*$ is either eigenvalue of the crossing pair at the
  refined onset, $op("Im")$ denotes its imaginary component, and
  $f_"Hopf"$ is the frequency in hertz. Because time was expressed in
  milliseconds, $abs(op("Im") lambda_J^*)$ gives angular frequency in
  radians per millisecond; the factor 1000 converts the resulting cycles
  per millisecond to hertz. Solver initialization, eigenvalue-selection
  thresholds and numerical tolerances are specified in Appendix C.

  #editing-paragraph-label("P33")
  We assessed onset criticality using upward and downward drive sweeps,
  integrating for 2 s at each current (Fig. 2H). Peak-to-peak excitatory-rate
  amplitude $A_"pp"$, in $"ms"^(-1)$, was the maximum minus the minimum
  over the final 500 ms. A numerically supercritical classification required
  no resolved hysteresis and a positive amplitude-squared slope above onset,
  using the predefined branch-gap and fit thresholds in Appendix C3.
  Sweep settings are specified in Appendix C2. No first Lyapunov coefficient
  was calculated.

  We assessed inhibitory-timescale dependence by repeating the fixed-point
  continuation and onset refinement at inhibitory decay times
  $tau_"GABA" = 4.5, 6, 9, 12, 18$ and $27$ ms, with other mean-field
  parameters fixed (Fig. 2I). At each decay, we compared the mean-field
  onset frequency $f_"Hopf"$ with the median finite-drive spectral-peak
  frequency $f_gamma$ across three independently trained classifiers
  evaluated at their final checkpoints, using the spectral protocol
  described above.

  === Classifier comparisons and recurrent coupling

  #editing-paragraph-label("P34")
  We compared COBA and PING classifiers under six activity conditions: one
  unpenalised condition and five activity ceilings of 1, 2.5, 5, 10 and
  25 Hz, using the penalty defined above. Each architecture–condition
  combination comprised three independent training replicates (seeds 42–44),
  giving 36 networks (Fig. 3).

  We evaluated each network’s epoch-50 checkpoint on the common subset of
  1,000 official MNIST test images, presenting each image for 200 ms. Test
  accuracy and mean per-neuron excitatory firing rate were calculated across
  the complete subset for each network, then summarized within each
  condition as the mean and standard error of the mean (SEM) across the
  three training replicates. SEM was calculated as the sample standard
  deviation divided by $sqrt(3)$.

  For the unpenalised conditions, validation learning curves averaged accuracy
  across the three replicates at each epoch. Illustrative-example selection
  is specified in Appendix B4.

  #editing-paragraph-label("P35")
  We examined the effect of adding reciprocal inhibition after learning by
  reusing the three unpenalised COBA classifiers from the accuracy–rate
  comparison (seeds 42–44), taking each network’s best-validation checkpoint
  (Fig. 4). Learned input and readout weights were held fixed throughout;
  no further training occurred.

  The dimensionless loop strength $s$ ranged from 0 to 1 in increments of
  0.1. At each strength, we newly initialized the E→I and I→E matrices from
  Gaussian distributions with parent means $mu_(E arrow.r I) = s$ µS and
  $mu_(I arrow.r E) = 2s$ µS, respectively, and standard deviations equal
  to 10% of each mean. We clamped negative draws to zero and divided weights
  by the presynaptic population size: 1,024 for E→I and 256 for I→E. Thus,
  $s = 0$ disabled the loop and $s = 1$ used the standard recurrent
  initialization.

  At every strength, each classifier received the common subset of 1,000
  official MNIST test images for 200 ms per image. Test accuracy and mean
  per-neuron E and I firing rates were summarized as means and sample
  standard deviations across the three training replicates.

  #editing-paragraph-label("P36")
  We compared fixed reciprocal E→I and I→E weights at standard strength
  $s = 1$ with trainable reciprocal weights initialized at $s = 1$, $0.1$
  or $0$, using the initialization defined above (Fig. 5). Input and
  readout weights were trained in all conditions for 50 epochs without an
  activity penalty. Each condition comprised three independent training
  replicates (seeds 42–44), giving 12 networks.

  Epoch-50 test accuracy and mean per-neuron E and I firing rates were
  measured across the common subset of 1,000 official MNIST test images,
  presented for 200 ms each. Autocorrelation contrast was obtained from the
  epoch-50 training diagnostic using a single 200-ms Poisson encoding of
  the first digit-0 image in the official test set. This encoding was
  generated with seed 0 and held fixed across networks. We used the final
  contrast directly, without smoothing across epochs. Accuracy, firing
  rates and contrast were summarized as means and SEM across the three
  training replicates.

  For recurrent-weight comparisons, initial matrices were reconstructed
  using the original initialization seeds and parameters. At initialization
  and epoch 50, we pooled the three complete matrices separately for each
  connection direction and condition, giving 786,432 entries per pooled
  distribution. We calculated the fraction of strictly positive weights and
  the arithmetic mean weight across all entries, including zeros.

  === Inhibitory timescale and cycle participation

  #editing-paragraph-label("P37")
  We examined inhibitory-timescale dependence by separately training PING
  classifiers at inhibitory decay times $tau_"GABA" = 4.5, 6, 9, 12, 18$
  and $27$ ms (Fig. 6A–B). Each decay time comprised three independent
  training replicates (seeds 42–44), giving 18 networks. Input and readout
  weights were trained for 50 epochs without an activity penalty, while
  reciprocal weights remained fixed at standard strength. Each network
  retained its assigned inhibitory decay time during evaluation.

  We evaluated epoch-50 checkpoints on the common subset of 1,000 official
  MNIST test images, presented for 200 ms each. For each network, we
  measured test accuracy, mean per-neuron excitatory firing rate and
  spectral-peak frequency using the protocols above. Each measure was
  summarized as the mean and SEM across the three replicates at each
  decay time.

  We fitted the affine relationship

  #math.equation(
    block: true,
    numbering: "(1)",
    $ macron(r)_E = a + p macron(f)_gamma $,
  ) <eq:rate-frequency-fit>

  by ordinary least squares, giving equal weight to each of the six
  condition means. Here, $macron(r)_E$ and $macron(f)_gamma$ are the
  across-replicate mean excitatory firing rate and spectral-peak frequency,
  both in hertz; $a$ is the fitted intercept in hertz and $p$ is the
  dimensionless fitted slope. Fit quality was quantified by the coefficient
  of determination $R_"fit"^2$, using the centered total sum of squares.
  The mean-field comparison in Fig. 2I used medians of the same three
  per-network frequencies at each decay time.

  #editing-paragraph-label("P38")
  We quantified excitatory spike counts per cycle using the same 18
  epoch-50 classifiers and common subset of 1,000 test images, presented
  for 200 ms each (Fig. 6C–H). Inhibitory bursts were detected from
  Gaussian-smoothed population spike counts, with peak separation scaled to
  each network's spectral period (Appendix B3).

  We placed cycle boundaries at the midpoints between consecutive
  inhibitory-burst peaks, extending the first and last intervals to the
  presentation boundaries. Presentations without detected bursts were
  excluded. Within each retained interval, we counted spikes separately
  for every excitatory neuron and assigned each neuron–cycle pair to one
  of four categories: 0, 1, 2 or at least 3 spikes.

  For each inhibitory decay time, category fractions were calculated by
  pooling counts across all retained presentations and the three training
  replicates, with each neuron–cycle pair contributing equally. Overall
  fractions pooled counts across all six decay times. For summaries
  restricted to active pairs, defined as pairs containing at least one
  spike, zero-spike pairs were excluded from the denominator. Discrete
  peak-separation, boundary-rounding and single-burst conventions are
  specified in Appendix B.

  === Spike perturbations and numerical resolution

  #editing-paragraph-label("P39")
  We applied spike deletion and insertion separately to the best-validation
  checkpoints of unpenalised COBA and PING classifiers, using three training
  replicates per architecture (seeds 42–44), without retraining (Fig. 7A–B).
  Each condition used the common subset of 1,000 test images, presented for
  200 ms at a 0.1-ms timestep. Input encodings were held fixed across
  perturbation conditions.

  Deletion independently suppressed each naturally generated excitatory or
  inhibitory spike with probability $p_"del" = 0, 0.1, dots, 1$,
  preserving the neuronal reset and refractory period associated with that
  spike.

  For insertion, we first measured each network's unperturbed mean
  per-neuron excitatory firing rate, $r_(E,0)$, across the complete test
  subset. The nominal per-neuron insertion rate was
  $r_"add" = alpha r_(E,0)$, where the dimensionless dose
  $alpha = 0, 0.1, dots, 2$. Independent Bernoulli events were sampled for
  every excitatory and inhibitory neuron at each timestep, with probability
  equal to the insertion rate in hertz multiplied by the timestep in
  seconds. The same nominal insertion rate applied to both populations.
  Transmission was capped at one spike per neuron per timestep, so
  insertion coinciding with an existing spike added no event. Inserted
  events triggered neither resets nor refractory periods.

  Modified spike events supplied the readout and subsequent recurrent
  transmission. Accuracy was calculated across all 1,000 images for each
  network and summarized as the mean and sample standard deviation across
  the three training replicates. Insertion conditions were aggregated at
  matched baseline-relative doses.

  #editing-paragraph-label("P40")
  We evaluated separately trained PING classifiers at matched training and
  evaluation timesteps while holding refractory durations fixed. The full
  protocol and results are given in Appendix A5.

  #editing-paragraph-label("P41")
  We applied two count-preserving inhibitory replay perturbations to three
  unpenalised PING classifiers (seeds 42–44), using their epoch-50
  checkpoints (Fig. 8). We recorded unperturbed inhibitory spikes for the
  common subset of 1,000 test images, presented for 200 ms at a 0.1-ms
  timestep. Shifted recordings replaced naturally generated inhibitory
  outputs while excitatory activity and readout responses were recomputed
  using unchanged weights and identical input encodings.

  Independent-spike jitter assigned each inhibitory event a zero-mean
  Gaussian offset with standard deviation
  $sigma = 0, 0.5, 1, 2, 5, 9, 14, 21$ or $50$ ms. Group jitter assigned
  one shared Gaussian offset to all inhibitory events originating within
  each fixed 22.8-ms window of a presentation, using
  $sigma = 0, 1, 3, 7, 14, 21, 28, 42, 60$ or $100$ ms. These windows
  comprised 228 simulation steps and were defined independently of detected
  bursts. Figure 8 displays both sweeps through 14 ms.

  Each inhibitory neuron's spike count was preserved within every
  presentation. The parameter $sigma$ describes proposed offsets before
  boundary and collision adjustments; both arms shared the zero-jitter
  replay control. Exact transformations are specified in Appendix D.

  Accuracy and mean per-neuron excitatory and replayed inhibitory rates
  were calculated across the complete test subset, then averaged across
  the three training replicates. Figure 8 shows these means without
  uncertainty intervals. Replay interrupted responsive feedback because
  delivered inhibition could no longer respond to ongoing excitatory activity.

  === Continuous streams

  #editing-paragraph-label("P42")
  For continuous-stream experiments (Fig. 9), we trained three separate
  PING classifiers (seeds 42–44) for 50 epochs using the common MNIST split
  and optimization procedure described above, without an activity penalty.
  Input and readout weights were trained, while recurrent weights remained
  fixed at standard strength. Training used isolated 200-ms image
  presentations at a 0.1-ms timestep, with hidden and output states
  reinitialized for every presentation.

  For each presentation, the maximum-pixel input rate,
  $r_("input,max")$, was sampled uniformly from 0.5, 0.75, 1, 1.5, 2, 3,
  5, 7.5, 10, 15 and 25 Hz and held constant throughout that presentation.
  Rates were sampled independently for individual images within each
  minibatch and resampled on subsequent presentations.

  The output layer comprised ten spiking leaky integrate-and-fire units,
  one per digit class, with decay time $tau_"out" = 2$ ms, dimensionless
  threshold $theta_"out" = 1$ and no refractory period. After each spike,
  $theta_"out"$ was subtracted from the unit's state. For continuous-stream
  classifiers, class logits were accumulated output-spike counts rather
  than the mean pre-reset states used for ordinary classification:

  #math.equation(
    block: true,
    numbering: "(1)",
    $ z_c = sum_(k=1)^(N_t) s_c^"out"[k]. $,
  ) <eq:stream-spike-count-logits>

  Here, $z_c$ is the dimensionless logit for digit class
  $c in {0, dots, 9}$, $s_c^"out"[k] in {0, 1}$ indicates an output
  spike at timestep $k$, and $N_t$ is the number of steps in the
  presentation. Spike counts accumulated without decay or duration
  normalization. These logits entered the cross-entropy loss directly,
  with gradients propagated using the surrogate derivative described above.
  Classification selected the largest final count; ties selected the
  lowest class index.

  Readout weights were drawn from a Gaussian distribution with dimensionless
  mean $mu_"out" = 0.05$ and standard deviation $sigma_"out" = 0.04$, then
  clamped below at zero, without fan-in normalization. Evaluations used
  each network's best-validation checkpoint with weights held fixed.
  During continuous streams, output state and accumulated counts were reset
  to zero at each supplied image boundary, while hidden neuronal and
  synaptic states continued between images.

  #editing-paragraph-label("P43")
  For quantitative streaming evaluation (Fig. 9E–F), we crossed presentation
  durations of 25, 50, 100 and 200 ms with the eleven maximum-pixel input
  rates used during training. Each of the three frozen classifiers
  processed 40 five-image streams per duration–rate combination, in batches
  of five streams. This gave 200 classification decisions per network and
  condition, and 26,400 decisions overall. Duration and input rate remained
  constant within each stream, and images followed without gaps. Each
  stream began with freshly initialized hidden and output states;
  subsequent image boundaries used the state-handling procedure described
  above.

  For each stream, five images were sampled uniformly without replacement
  from the full 10,000-image official MNIST test partition, without class
  stratification. Digit labels could repeat within a stream, and images
  could recur across streams. Image samples varied across networks and conditions;
  encoding used an independent random stream (Appendix B5).

  For each network and condition, accuracy was the fraction of correct
  decisions across all 200 presentations, including presentations with no
  output spikes. Accuracies were averaged across the three training
  replicates at each duration–rate combination. The 200-ms curve in Fig. 9F
  reused the corresponding evaluations from Fig. 9E and reported the mean
  and SEM across those same three replicates.

  #editing-paragraph-label("P44")
  The illustrative stream was selected independently of quantitative
  evaluation using a predefined five-correct criterion, met by the first
  candidate (Appendix B4).

  To visualize the evolving readout within each presentation, we
  transformed cumulative output-spike counts into softmax shares:

  #math.equation(
    block: true,
    numbering: "(1)",
    $ q_c[k] = exp(n_c[k]) / (sum_(d=0)^9 exp(n_d[k])). $,
  ) <eq:stream-count-shares>

  Here, $n_c[k]$ is the cumulative spike count of output neuron $c$ through
  timestep $k$ of the current presentation, and $q_c[k]$ is its
  dimensionless softmax share; $c$ and $d$ index the ten digit classes.
  Counts restarted at each image boundary. These shares were used for
  visualization and were not calibrated probabilities. Classification
  followed the largest-final-count rule described above; the plotted
  0.5-share line was not a decision threshold.

  === Statistical reporting and reproducibility

  #editing-paragraph-label("P45")
  For trained-network comparisons, the unit of replication was one
  separately initialized and trained network, with three replicates per
  condition (seeds 42–44). These seeds controlled network initialization
  and, for variable-rate training, the sampling of input rates. The
  training-data split and endpoint test-image subset were fixed separately.
  Repeated interventions on a network were repeated measurements;
  individual neurons, presentations and neuron–cycle pairs were not
  additional network replicates. Untrained circuit probes used a single
  seed, and mean-field calculations were deterministic.

  Each network's outcome was calculated before aggregation across
  networks. Displayed error bars or bands represent SEM in Figs. 3,
  5A–C, 6A–B and 9F and Appendix Fig. A1, and sample SD in Figs. 4 and 7A–B.
  Sample SD used the denominator $n - 1$, and SEM was calculated as
  $"SD" / sqrt(n)$, where $n = 3$ is the number of training replicates.
  Figures 8 and 9E show means without uncertainty intervals. Pooled
  weight summaries and neuron–cycle distributions were calculated directly
  from their constituent observations; consequently, networks contributed
  to cycle distributions in proportion to their available neuron–cycle
  pairs. These summaries and fitted relationships were descriptive.

  Cycle-participation analysis excluded presentations without a detected
  inhibitory burst. Twelve of the 18,000 network–image presentations met
  this criterion, all from the 27-ms inhibitory-timescale condition at
  seed 43. The resulting distributions comprised 17,988 contributing
  presentations and 167,178,240 neuron–cycle pairs. This exclusion affected
  cycle-participation measurements, not test accuracy or whole-presentation
  firing rates. Selection of the illustrative continuous stream was handled
  separately, as described above.

  All 84 included networks completed 50 training epochs; their retained
  training records reported no skipped optimizer updates or batches with
  NaN outputs.

  #context {
    let note = [*Draft note:* Establish the rationale for the replicate and
      evaluation sample sizes, and confirm whether failed or discarded
      execution attempts preceded the retained runs.]
    if target() == "html" {
      html.elem("div", attrs: (style: "color: red;"), note)
    } else {
      text(fill: red, note)
    }
  }

  #editing-paragraph-label("P46")
  Spiking-network simulation, training and evaluation used custom
  Python/PyTorch code. All 84 classifiers were trained with PyTorch
  2.11.0+cu128 on CUDA devices. Mean-field calculations used Python 3.10.19,
  NumPy 2.2.6 and SciPy 1.15.3. Saved model checkpoints, simulation outputs
  and analysis results were linked through experiment records containing
  execution parameters, random seeds and recorded source revisions. Reused
  classifiers retained their original training provenance, distinguishing
  training from subsequent evaluation. Source revisions and dependency
  information were recorded separately for computation, analysis and figure
  generation.

  #context {
    let note = [*Draft note:* Add a permanent code/data archive and a
      figure-specific source and software record. Resolve the uncommitted
      changes recorded for some executions by including the executed source
      or patch; a commit identifier alone does not fully specify those
      executions.]
    if target() == "html" {
      html.elem("div", attrs: (style: "color: red;"), note)
    } else {
      text(fill: red, note)
    }
  }

  == Appendix A — Discrete dynamics and gradient calculations

  *A1 — Discrete neuronal and synaptic updates.* Existing conductances
  decayed before spike increments were added. Membrane voltage was then
  integrated using exponential Euler, holding the updated conductances
  constant over each step.

  *Remaining scaffold:* Give the actual recurrence equations implementing
  the continuous equations in Methods: exponential conductance decay, event
  increments, effective membrane equilibrium and exponential-Euler update.
  Specify the ordering of recurrent transmission, threshold evaluation,
  refractory-counter updates and reset. Define the timestep indexing so the
  transmission delay is unambiguous. Supports Methods P19; Figs. 1–8 and A1.

  *A2 — Output-state recurrence.* Give the output update, including its
  timestep-dependent input factor, pre-reset evidence accumulation and
  subtractive reset. Show precisely where mean-voltage and spike-count
  accumulation diverge. Supply the implementation equations without repeating
  the readout definitions or parameter values. Supports P23 and P42; Figs. 3–9.

  *A3 — Surrogate and damped derivatives.* Voltage-gradient damping preserved
  the forward voltage update and left the direct gradient path through the
  previous membrane voltage unchanged. Output-neuron updates did not receive
  this damping.

  *Remaining scaffold:* Starting from the fast-sigmoid surrogate in Equation 5,
  write the forward-preserving gradient-scaling operation. Show which
  membrane-increment derivatives receive damping, which direct state paths
  remain, and how hidden hard resets differ from differentiable output
  subtractive resets. Do not repeat optimizer settings. Supports P24.

  *A4 — Initialization transformation.* Input and recurrent weights were
  drawn from Gaussian distributions with the parent parameters in Table A1,
  clamped below at zero and divided by the presynaptic population size
  $N_"pre"$. Before
  this division, each input weight was independently zeroed with probability
  $q_"zero" = 0.95$. Surviving weights were multiplied by
  $1 / (1 - q_"zero") = 20$, preserving the expected summed connection
  strength despite the zeroing. This imposed no permanent connectivity mask:
  initially zero trainable weights could become positive during optimization.
  Readout weights were initialized directly, without fan-in normalization.
  These weights were dimensionless because they drove a normalized output
  state with threshold 1, rather than a synaptic conductance or a membrane
  voltage in millivolts.

  *Remaining scaffold:* Express this sequence algebraically and explain why
  the parent mean differs from the realized matrix mean. Refer to Table A1
  for values. Supports P20 and P42.

  #let parameter-table = table(
    columns: (auto, auto),
    table.header([Parameter], [Standard value]),
    [Population sizes $N_E, N_I$], [1,024; 256],
    [Membrane capacitance $C_m$, E/I], [1/0.5 nF],
    [Leak conductance $g_L$, E/I], [0.05/0.1 µS],
    [Leak/reset potentials $E_L, V_"reset"$], [−65 mV],
    [Spike threshold $V_"th"$], [−50 mV],
    [Reversal potentials $E_E, E_I$], [0; −80 mV],
    [Refractory period $tau_"ref"$, E/I], [1.2/0.6 ms],
    [Synaptic decay $tau_"AMPA", tau_"GABA"$], [2; 6 ms],
    [Integration timestep $Delta t_"sim"$], [0.1 ms],
    [Input parent mean/SD $mu_"in", sigma_"in"$], [0.9/0.09 µS],
    [Recurrent parent mean/SD, E→I], [1/0.1 µS],
    [Recurrent parent mean/SD, I→E], [2/0.2 µS],
    [Input-zeroing probability $q_"zero"$], [0.95],
    [Readout initialization mean/SD (dimensionless)], [1.12/0.835],
    [Output decay time; threshold], [2 ms; 1],
    [Presentation duration; maximum-pixel rate], [200 ms; 25 Hz],
    [Optimizer; learning rate], [AdamW; 0.0004],
    [Epochs; minibatch size], [50; 256],
    [Weight decay; gradient-norm limit], [0; 1],
    [Surrogate slope; voltage-gradient damping], [1; 1,000 (PING), 1 (COBA)],
  )
  #context figure(
    if target() == "html" {
      html.elem("div", attrs: (style: "display: flex; justify-content: center; overflow-x: auto;"), parameter-table)
    } else {
      align(center, parameter-table)
    },
    kind: table,
    numbering: n => "A1",
    caption: [*Standard spiking-classifier parameters and exceptions.*
      Values describe standard classifiers, not the separate mean-field model.
      Figure 1 used input parent mean/SD 1.5/0.3 µS and PING recurrent means
      1.5/3 µS. Coupling strengths, inhibitory decay times and timesteps varied
      in their respective sweeps. Figure 9 used readout mean/SD 0.05/0.04 and
      variable input rates; stream durations varied during evaluation.
      Readout initialization values are rounded.],
  ) <tab:shared-parameters>

  *A5 — Numerical-resolution validation.*

  We examined timestep dependence using PING classifiers trained at
  integration timesteps $Delta t_"sim" = 0.05, 0.1, 0.2, 0.3$ and $0.6$ ms,
  with three independent training replicates per timestep (seeds 42–44),
  giving 15 networks (Appendix Fig. A1). All networks were trained for 50 epochs
  without an activity penalty, with recurrent weights fixed at standard
  strength. Excitatory and inhibitory refractory periods remained 1.2 and
  0.6 ms, respectively, represented exactly by integer step counts at every
  timestep. We reused the three existing 0.1-ms classifiers and trained
  twelve networks at the remaining timesteps.

  Each network's epoch-50 checkpoint was evaluated at its training timestep
  on the common subset of 1,000 test images. Presentations lasted nominally
  200 ms, truncated to complete simulation steps: the realised duration was
  199.8 ms at 0.3 and 0.6 ms and 200 ms otherwise. Mean per-neuron
  excitatory firing rates were calculated across all neurons and test
  presentations using the realised duration. Accuracy and excitatory firing
  rate were summarized as means and SEM across the three training replicates
  at each timestep. Because weights were trained separately at each
  timestep, this experiment assessed timestep dependence after training and
  did not establish numerical convergence with weights held fixed.

  Across this twelvefold timestep range, mean test accuracy ranged from
  88.4% to 89.6%, a span of 1.27 percentage points, whereas mean excitatory
  firing increased from 14.3 to 17.0 Hz between the finest and coarsest
  timesteps (Appendix Fig. A1). Thus, classification performance persisted
  across the tested numerical resolutions, although excitatory activity
  remained timestep-dependent.

  #figure(
    data-image(
      data-file("exp044/dt_sweep.svg"),
      width: 80%,
      alt: "Hidden excitatory firing rate and test accuracy across five integration timesteps, with SEM across independently trained classifiers.",
    ),
    numbering: n => "A1",
    caption: [*Timestep dependence after separate training.*
      Mean per-neuron E firing rate (black diamonds, left axis) and test
      accuracy (red squares, right axis), with means ± SEM across three
      independently trained PING classifiers per timestep (seeds 42–44).
      Networks contained 1,024 E and 256 I neurons and used the same
      1,000 MNIST test images, 784 independent pixel-proportional Poisson
      channels and a 25-Hz maximum-pixel rate. Each epoch-50 checkpoint was
      evaluated at its training timestep; E/I refractory holds remained
      1.2/0.6 ms. Presentations lasted 199.8 ms at 0.3 and 0.6 ms and
      200 ms otherwise; rates use the realised durations.
      Source experiment: #link("/exp044/")[exp044] — #link("/exp044/")[_Firing Rate Across the Timestep Sweep._]],
  ) <fig:timestep-validation>

  == Appendix B — Measurement algorithms and boundary cases

  *B1 — Spectral interpolation.* The spectral-peak frequency $f_gamma$ was
  refined by parabolic interpolation through the peak and its two neighboring
  bins, with the adjustment limited to half a bin.

  *Remaining scaffold:* Give the three-bin interpolation formula on linear
  spectral power. Specify zero-offset behavior at spectrum boundaries or
  zero curvature and treatment of unusable traces where applicable. Leave
  window lengths, frequency ranges and averaging order in Methods.
  Supports P29; Figs. 1E/G, 2I and 6A–B.

  *B2 — Autocorrelation implementation.* We calculated the
  uncentered autocorrelogram up to 100-ms lag, dividing each lag’s summed
  count products by the number of contributing bin pairs and the squared
  mean bin count. Before extracting features, we replaced the zero-lag
  value with its 1-ms neighbor and applied three-point smoothing with
  weights 0.25, 0.5 and 0.25.

  *Remaining scaffold:* Give the FFT construction and finite-overlap
  normalization explicitly. Specify discarded incomplete bins, smoothing at
  array boundaries, the asymmetric inequalities used to identify a trough,
  and first-maximum tie handling. State when contrast is undefined and
  distinguish missing values in the coupling-grid analysis from zero
  substitution in the training diagnostic. Supports P30; Figs. 2C and 5C.

  *B3 — Burst detection and cycle-boundary discretization.* For each
  presentation, we summed inhibitory spikes across neurons at each 0.1-ms
  simulation step and smoothed the resulting trace with a normalized Gaussian
  kernel of standard deviation
  $sigma_"smooth" = 1$ ms. We detected peaks with heights at least 5% of
  the maximum smoothed value within that presentation, requiring a minimum
  separation of approximately half the period corresponding to that
  network's spectral-peak frequency $f_gamma$.

  *Remaining scaffold:* Specify the Gaussian kernel's ±4-standard-deviation
  support, integer rounding of minimum peak separation, floor-rounded
  midpoint boundaries and half-open counting intervals. Explain the
  single-detected-burst case: its counting interval spans the entire
  presentation. Clarify these consequences of the cycle definition without
  repeating pooled results. Supports P38; Fig. 6C–H.

  *B4 — Illustrative-example selection.*

  For Fig. 3, illustrative rasters used the seed-42, epoch-50 checkpoint
  from each architecture and the first digit-0 image in the official test
  set, presented for 400 ms.

  For Fig. 4, illustrative rasters used the seed-42 classifier and the first
  image in the official test set, a digit 7, at $s = 0$ and $s = 1$. Both
  presentations lasted 200 ms. Fixed subsets of 200 E and 64 I neurons
  were selected uniformly without replacement using seed 0 for display.

  For Fig. 8, illustrative rasters used the seed-42 classifier
  and official test image 0, a digit 7, at $sigma = 14$ ms.

  For the illustrative stream (Fig. 9A–D), we used the seed-42 classifier
  with five predefined presentation-duration–maximum-pixel-rate pairs:
  (100 ms, 5 Hz), (200 ms, 7.5 Hz), (50 ms, 25 Hz), (100 ms, 15 Hz) and
  (200 ms, 10 Hz). Each candidate contained five distinct digit classes
  drawn in random order, with one image sampled uniformly from each class
  in the official MNIST test partition.

  Candidates were evaluated in a predetermined order, and the first
  yielding five correct classifications was retained for the illustration.
  The first candidate met this criterion, producing the sequence 1, 7, 9,
  5 and 2. Its image-selection and spike-encoding seeds were 820000 and
  830000, respectively. This illustrative selection was separate from the
  quantitative streaming evaluation. Rasters displayed the first 200
  excitatory and 64 inhibitory neurons.

  *B5 — Random-stream implementation.* Endpoint evaluation restarted a
  separate encoding random stream with fixed seed 20260415, providing
  matched input spike trains across networks evaluated with the same images,
  ordering, batch grouping and encoding settings. Perturbation randomness
  was generated separately.

  For quantitative continuous-stream evaluation, image-sampling seeds depended
  on network seed, presentation duration and input rate, so these evaluations
  did not use the fixed 1,000-image endpoint subset. Bernoulli encoding used a
  separate random stream indexed by network seed and stream number.
  Encoding seeds were reused across conditions, without imposing identical
  spike trains when images, durations or rates differed.

  == Appendix C — Mean-field closure and numerical specification

  *C1 — Gain function and closure assumptions.* Give the noisy-LIF gain
  integral, its reset/threshold integration bounds and conversion from mean
  current to effective mean voltage. Explain the substitutions leading from
  voltage-dependent synaptic currents to fixed driving forces and
  population-rate relaxation. Distinguish assumptions from derivations; do not
  repeat the four final equations or their parameter table.
  Supports P31; Fig. 2G–I.

  *C2 — Numerical solution details.*

  We located the mean-field oscillatory onset by continuing fixed points
  across 401 equally spaced external currents $I_"ext"$ from 0 to 4 nA,
  in 0.01-nA increments (Fig. 2G). We solved the steady-state equations by
  nonlinear root finding, using the preceding solution to initialize each
  subsequent solve. At each equilibrium, we calculated the continuous-time
  Jacobian $J_"flow"$ of the four-variable model using centered finite
  differences, with perturbations of $10^(-6) thin "ms"^(-1)$ for rate
  coordinates and $10^(-6)$ µS for conductance coordinates.

  Among the complex eigenvalues, we selected the conjugate pair with the
  largest real part and identified its first crossing from negative to
  nonnegative real part as current increased. We refined the onset current
  $I_"ext"^*$ using Brent’s method to locate zero real part, recalculating
  the equilibrium and Jacobian at each trial current.

  We assessed onset criticality numerically using upward and downward sweeps
  through 25 equally spaced external currents from $I_"ext"^* - 0.1$ nA
  to $I_"ext"^* + 0.55$ nA (Fig. 2H). We initialized the lowest-current
  equilibrium with its excitatory rate increased by $10^(-3) thin "ms"^(-1)$.
  At each current, we integrated the four-variable equations for 2 s using
  LSODA, carrying the final state into the next integration and from the
  upward sweep into the downward sweep. The peak-to-peak excitatory-rate
  amplitude $A_"pp"$, measured in $"ms"^(-1)$, was the maximum minus the
  minimum rate over the final 500 ms.

  For negative integration arguments $u$, the gain integrand was evaluated
  as $"erfcx"(-u)$, the scaled complementary error function, rather than
  $exp(u^2)(1 + "erf"(u))$, to avoid cancellation. This numerical correction
  accompanied the refractory change, so differences from the earlier
  mean-field calculation cannot be attributed to refractoriness alone.

  *Remaining scaffold:* Specify initialization of the first fixed-point
  solve, failed-solution handling, numerical quadrature settings and
  eigenvalue-selection thresholds. Put the precise Brent and LSODA tolerances
  and maximum integration step in Appendix Table C1. Supports P32–P33.

  *Appendix Table C1 placement — Numerical solver settings.* Record the
  quadrature, root-refinement and integration settings needed to reproduce
  the displayed mean-field calculation; omit protocol values already in Methods.

  *C3 — Criticality calculation.*

  We classified the sampled onset as consistent with a supercritical
  transition when the maximum absolute difference between upward and
  downward amplitudes at matched currents was below $10^(-4) thin "ms"^(-1)$,
  and a linear regression of $A_"pp"^2$ against excess current
  $I_"ext" - I_"ext"^*$ had a positive slope and coefficient of determination
  $R_"fit"^2 > 0.9$. The regression included an intercept and used
  upward-sweep points above onset. This was a numerical classification of
  the sampled trajectories; no first Lyapunov coefficient was calculated.

  *Remaining scaffold:* Specify the centered total sum of squares and
  handling of insufficient points or zero variance. Supports P33; Fig. 2H.

  == Appendix D — Exact replay transformations

  *D1 — Quantization and boundary reflection.* Offsets were rounded to the
  simulation grid. Boundary reflection kept events within the presentation.

  *Remaining scaffold:* Give the conversion of Gaussian offsets to integer
  timestep shifts and the repeated-reflection mapping into a bounded interval.
  For independent jitter, apply reflection to event destinations. For group
  jitter, derive the allowable shared displacement from the earliest and
  latest events in the source window. Explain the degenerate interval case.
  Supports P41; Fig. 8.

  *D2 — Collision resolution.* Same-neuron collisions were resolved by moving
  events to the nearest unoccupied timestep. Each inhibitory neuron's spike
  count was preserved exactly within every presentation.

  *Remaining scaffold:* Describe stable event ordering and resolution of
  duplicate same-neuron destinations by searching offsets in the order
  +1, −1, +2, −2, continuing outward and skipping out-of-range candidates.
  Explain that count preservation survives this operation, although collision
  adjustment can change within-group relative timing. Supports P41; Fig. 8.
  #metadata("exp110-end")
]

#let body = if inputs.all(key => data-file(key) != none) {
  render-report(data-file)
} else [
  A required run is unavailable, so there is no content to display yet.
]
