// Author-approved standalone manuscript: no shared writing templates.
#import "/.demolab/lib.typ": data-image, cite, reference-list

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
  updated_at: "2026-09-11",
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
  "exp046",
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

#let manuscript-note(body) = context {
  if target() == "html" {
    html.elem("div", attrs: (style: "color: red;"), body)
  } else {
    text(fill: red, body)
  }
}

#let render-report(data-file) = [
  #let stream-data = json(data-file("exp082/numbers.json"))
  #let stream-image-policy = stream-data.config.at("image_stream_policy", default: none)
  #assert(stream-image-policy in (none, "shared-across-training-seeds-durations-rates/v1"),
    message: "unsupported continuous-stream image policy")
  #let shared-stream-images = stream-image-policy != none
  #let stream-mean(duration, rate) = {
    let values = stream-data.grid_per_seed
      .filter(row => row.duration_ms == duration and row.rate_hz == rate)
      .map(row => row.accuracy)
    values.sum() / values.len()
  }
  #let stream-pct(value) = str(calc.round(100 * value, digits: 1)) + "%"
  #let stream-accuracy(duration, rate) = stream-pct(stream-mean(duration, rate))
  #let stream-upper-means = (stream-data.config.psychometric_rates_hz
    .filter(rate => rate >= 5)
    .map(rate => stream-mean(200, rate)))
  #let stream-upper-increasing = (range(1, stream-upper-means.len())
    .all(i => stream-upper-means.at(i) > stream-upper-means.at(i - 1)))
  #set heading(numbering: none)
  #set math.equation(numbering: "(1)")
  #counter(math.equation).update(0)
  #let manuscript-figure-ref(target, panel: none, appendix: false) = context {
    let numbered = ref(
      target,
      supplement: if appendix { [Appendix Fig.] } else { [Fig.] },
    )
    if panel == none { numbered } else { [#numbered#panel] }
  }
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

  == Meta

  // Keep these word counts up to date after manuscript edits. Counts use
  // rendered text: section totals exclude headings, figures and tables,
  // captions, editing labels and red editorial notes; the caption total
  // includes all figure and table captions but excludes generated labels.
  #context {
    let counts = [
      - *Results:* 1,290 words
      - *Methods:* 4,343 words
      - *Captions:* 1,664 words
      - *Appendix A:* 1,540 words
      - *Appendix B:* 1,248 words
      - *Appendix C:* 832 words
      - *Appendix D:* 404 words
    ]
    if target() == "html" {
      html.elem("div", attrs: (style: "color: red;"), counts)
    } else {
      text(fill: red, counts)
    }
  }

  == Results

  === Reciprocal coupling creates a gamma-rhythmic low-rate regime

  #editing-paragraph-label("P1")
  We compared loop-disabled COBA with PING circuits containing reciprocal
  coupling between excitatory (E) and inhibitory (I) neurons (Fig. 1A,B).
  Illustrative 400-ms responses used separately chosen Poisson drives of 5
  and 45 Hz per channel, respectively (Fig. 1C,D).

  #figure(
    data-image(
      data-file("exp023/overview_compound.png"),
      width: 92%,
      alt: "Loop-disabled COBA and recurrent PING networks shown through wiring diagrams, illustrative spike rasters collected under different Poisson drives, power spectra and matched firing-rate–input sweeps.",
    ),
    caption: [*COBA and PING circuit architecture and activity.*
      *(A–B)* Loop-disabled COBA and reciprocal E→I/I→E PING.
      *(C–D)* Illustrative 400-ms rasters: 1,024 E neurons (black), 256 I
      (red), and 1,024 Poisson channels at 5 Hz (COBA) or 45 Hz (PING).
      *(E, G)* Mean-subtracted E-population Welch spectra; G's dashed line
      marks the interpolated peak. A missing marker is not a rhythmicity
      test.
      *(F, H)* Mean per-neuron E/I rates over matched 2–100-Hz drive sweeps
      using 784 channels. Each point is one single-seed trial, without
      uncertainty estimates; vertical scales differ. Source experiment:
      #link("/exp023/")[exp023] — #link("/exp023/")[_Turning the PING Loop On._]],
  ) <fig:matched-drive>

  #editing-paragraph-label("P2")
  Loop-disabled excitatory spikes were dispersed, with silent inhibitory
  neurons (Fig. 1C); PING instead produced recurring E/I volleys and a
  56.0-Hz spectral peak with higher-frequency harmonics. The control lacked
  this regular harmonic structure (Fig. 1D,E,G). These single-trial examples
  support gamma-periodic organisation.

  In matched 2–100-Hz drive sweeps, using 784 input channels rather than the
  illustrative rasters' 1,024, COBA excitatory firing increased from 2.9 to
  481.5 Hz while inhibition remained silent. PING excitatory rates stayed
  between 2.8 and 11.2 Hz, while inhibitory firing reached 124.4 Hz (Fig.
  1F,H). Thus, reciprocal coupling constrained excitatory recruitment above
  the lowest drive condition; each condition contained one trial.

  #editing-paragraph-label("P3")
  Across an 11×11 reciprocal-coupling grid, absent E→I or I→E coupling left
  excitatory firing near 169 Hz and autocorrelation lobe–trough contrast
  near zero (Fig. 2A–C). With both pathways present, stronger coupling
  reduced excitatory firing to single-digit rates, sustained inhibitory
  firing and increased contrast towards one across a broad region. The
  low-rate, structured regime therefore extended beyond an isolated
  operating point. Each condition used one untrained network and seed.

  #figure(
    data-image(
      data-file("exp110/onset_super_compound.png"),
      width: 92%,
      alt: "Coupling-plane maps, representative spike rasters and mean-field analyses of oscillatory onset in the recurrent excitatory-inhibitory circuit.",
    ),
    caption: [*Reciprocal-coupling sweep and mean-field onset.*
      *(A–C)* Mean E rate, I rate and autocorrelation lobe–trough contrast
      across an 11×11 grid of E→I ($W_(E I)$) and I→E ($W_(I E)$)
      initialization means on the fan-in-normalized summed-conductance
      scale. Each point represents one untrained network and seed; B's
      grayscale is clipped at its 92nd percentile.
      *(D–F)* Illustrative 200-ms rasters at $(W_(E I), W_(I E)) = (0, 0)$,
      $(0.6, 1.2)$ and $(3, 6)$ µS; black/red marks show 160 E/48 I neurons.
      A–F have no uncertainty estimates.
      *(G–H)* Separate four-variable mean-field model. G shows fixed-point
      eigenvalues across $I_"ext" = 0$–4 nA (colour), with the leading pair
      at $I_"ext"^* = 0.594$ nA circled in cyan. H shows upward/downward
      peak-to-peak E-rate amplitudes; the dotted line marks onset.
      *(I)* Mean-field onset frequencies (black circles, solid) and median
      finite-drive spectral peaks from three separately trained classifiers
      (red squares, dashed), without uncertainty intervals. Methods define
      measurements and protocols. Source experiments:
      #link("/exp054/")[exp054] — #link("/exp054/")[_Pinglab Rythmicity Metric_],
      #link("/exp033/")[exp033] — #link("/exp033/")[_Gamma Emerges at a Hopf Bifurcation_], and
      #link("/exp041/")[exp041] — #link("/exp041/")[_Firing Rate Tracks Gamma Frequency._]],
  ) <fig:coupling-plane>

  #editing-paragraph-label("P4")
  Selected rasters illustrate the same progression (Fig. 2D–F):
  autocorrelation contrast increased from 0.00030 without coupling to 0.268
  at intermediate coupling and 0.989 at strong coupling, as excitatory
  firing became sparse and inhibitory volleys increasingly regular.

  #editing-paragraph-label("P5")
  In the separate four-variable mean-field model, leading complex-conjugate
  eigenvalues crossed zero real part at $I_"ext"^* = 0.594$ nA,
  corresponding to 27.6 Hz (Fig. 2G). Above onset, excitatory-rate amplitude
  increased continuously, with nearly coincident upward/downward sweeps and
  no resolved hysteresis (Fig. 2H). A positive amplitude-squared slope and
  $R^2 = 0.999$ met the predefined numerical criteria for supercritical
  onset; criticality was not established analytically.

  #editing-paragraph-label("P6")
  Increasing inhibitory decay from 4.5 to 27 ms reduced mean-field onset
  frequency from 30.2 to 17.9 Hz and median spiking-classifier spectral
  frequency from 67.3 to 12.2 Hz (Fig. 2I). This shared timescale dependence
  does not imply quantitative agreement between mean-field onset
  eigenfrequencies and finite-drive spectral peaks of the separate,
  uncalibrated spiking model.

  === The fixed PING loop preserves accuracy at lower excitatory rates

  #editing-paragraph-label("P7")
  Both unpenalised architectures reached approximately 90% validation
  accuracy, with dense COBA firing and recurring PING volleys (Fig. 3A–C).
  PING achieved 89.8% mean test accuracy at a mean excitatory firing rate of
  16.6 Hz, versus 91.1% at 113.9 Hz for COBA: a 6.9-fold firing-rate
  reduction and 1.3-percentage-point accuracy reduction (Fig. 3D). Activity
  penalties lowered firing in both families.
  At a 10-Hz training ceiling, both averaged approximately 9.1 Hz, but PING
  retained higher accuracy (88.6% versus 86.0%), shifting the accuracy–rate
  relationship favourably at low rates.

  #figure(
    data-image(
      data-file("exp025/results_compound.png"),
      width: 92%,
      alt: "COBA and PING single-trial activity, validation accuracy and test accuracy against excitatory firing rate.",
    ),
    caption: [*Accuracy and excitatory firing in COBA and PING classifiers.*
      *(A–B)* Illustrative 400-ms responses to digit-0 sample 0 from final
      unpenalised checkpoints, seed 42; E spikes are black, I red.
      *(C–D)* COBA: red squares; PING: black diamonds. Means use three
      independent training replicates. C shows unpenalised validation
      accuracy without uncertainty intervals. D shows final-checkpoint test
      accuracy versus mean per-neuron E rate under the common endpoint
      protocol; bars show the standard error of the mean (SEM) on both axes.
      Ceilings were 1, 2.5, 5, 10 and 25 Hz; stars denote unpenalised
      conditions.
      Source experiment: #link("/exp025/")[exp025] — #link("/exp025/")[_Accuracy and Firing Rate With and Without Inhibition._]],
  ) <fig:accuracy-rate>

  #editing-paragraph-label("P8")
  Adding reciprocal inhibition to trained COBA classifiers without
  retraining replaced dense excitatory activity with recurring E/I volleys
  (Fig. 4A,B). Across the coupling sweep, mean excitatory firing fell from
  112.1 to 8.1 Hz (13.8-fold), inhibitory firing reached 45.0 Hz, and
  accuracy fell from 91.0% to 42.9% (Fig. 4C,D). The activity pattern
  therefore did not require learning with inhibition, but inserting the loop
  after training incurred a substantial classification cost.

  #figure(
    data-image(
      data-file("exp038/loop_transfer_compound.png"),
      width: 92%,
      alt: "Loop-off and loop-on rasters followed by population firing rates and test accuracy across reciprocal loop strength.",
    ),
    caption: [*Post-training insertion of reciprocal inhibition.*
      *(A–B)* Illustrative 200-ms responses to the same digit-7 image from
      one validation-selected COBA classifier (seed 42), at $s = 0$ and $s =
      1$; fixed pseudorandom subsets show 200 E neurons (black) and 64 I
      (red). Learned weights remained frozen; recurrent matrices were newly
      initialized at each strength without retraining (Methods).
      *(C–D)* Mean ± sample standard deviation (SD) across three independent
      classifiers, evaluated using the common endpoint protocol at $s = 0$–1
      in 0.1 steps. C shows per-neuron E/I rates (black circles/red
      squares); D shows accuracy (grey squares), with the dashed line
      marking mean accuracy at $s = 0$. Source experiment:
      #link("/exp038/")[exp038] — #link("/exp038/")[_Switching On the Inhibitory Loop._]],
  ) <fig:loop-transfer>

  #editing-paragraph-label("P9")
  Training recurrent weights reduced autocorrelation contrast relative to
  fixed PING recurrence. Final autocorrelation lobe–trough contrast was
  0.999 with fixed recurrence, versus 0.094, 0.095 and 0.018 when recurrence
  was trained from standard, one-tenth-standard and zero initialization
  (Fig. 5C). These means describe three networks per condition responding to
  one fixed encoding of a reference digit, not test-set variability.

  In all three networks in each condition, standard and one-tenth-standard
  training eliminated most E→I connections; most I→E connections remained
  positive and their mean strength across all entries increased (Fig. 5D–G).
  Mean E→I strength fell from standard but rose from
  one-tenth-standard initialization; zero-initialized recurrence remained
  zero. Weak contrast accompanied asymmetric reorganization in these
  three-replicate comparisons, without identifying a causal weight change.

  #figure(
    data-image(
      data-file("exp049/training_summary.svg"),
      width: 92%,
      alt: "Final accuracy, E/I rates and reference-image contrast across four conditions, with initial and final recurrent nonzero fractions and mean weights.",
    ),
    caption: [*Epoch-50 activity, probe responses and recurrent weights.*
      Input/readout weights were trained in four conditions: fixed
      recurrence (Frozen), or recurrence trained from standard (Std.),
      one-tenth-standard (10%) or zero (Zero) initialization.
      *(A–B)* Test accuracy and mean per-neuron E/I rates (black/red) under
      the common endpoint protocol.
      *(C)* Final, non-epoch-smoothed autocorrelation lobe–trough contrast
      from the same fixed digit-0, sample-0 encoding; this probe does not
      estimate trial-to-trial variability. A–C show means ± SEM across three
      independent training replicates.
      *(D–E)* Pooled positive-weight fractions for E→I/I→E.
      *(F–G)* Pooled arithmetic means including zeros, in $10^(-3)$ µS. For each
      direction, condition and time point, the pool contains 786,432 entries
      from three equal-sized complete matrices, giving each network equal
      weight. Wide grey/narrow red bars denote initialization/epoch 50.
      Error bars show ±1 SEM across the three per-network statistics (sample
      SD divided by $sqrt(3)$), calculated separately before and after training;
      they do not describe uncertainty in the paired change. Individual weights
      are not independent training replicates. Arrows mark pooled relative
      changes ≥5%, not statistical significance. Source experiment:
      #link("/exp049/")[exp049] — #link("/exp049/")[_Training Recurrent Weights Weakens PING Rhythmicity._]],
  ) <fig:trainable-loop>

  === Excitatory firing is organised by gamma-cycle participation

  #editing-paragraph-label("P10")
  Across inhibitory decay times of 4.5–27 ms, mean spectral frequency fell
  from 67.5 to 11.7 Hz and excitatory firing from 18.3 to 2.8 Hz (Fig. 6A).
  The affine fit to six condition means had slope 0.285 Hz/Hz, intercept
  −0.70 Hz and $R^2 = 0.997$. Mean test accuracy fell from 91.2% to 81.9%
  (Fig. 6B). Slower rhythms thus accompanied lower activity and a
  classification cost; the fitted slope alone does not establish neurons'
  participation per cycle.

  #figure(
    data-image(
      data-file("exp110/cycle_participation_compound.png"),
      width: 92%,
      alt: "Post-training excitatory firing rate and accuracy across gamma frequencies, followed by distributions of excitatory spikes per neuron and inferred inhibitory-burst cycle.",
    ),
    caption: [*Excitatory firing and spike counts per cycle across inhibitory decay
      times.* Eighteen epoch-50 classifiers comprise three independent
      replicates per decay time, evaluated with the common endpoint
      protocol.
      *(A–B)* Mean per-neuron E rate $r_E$ and test accuracy versus
      spectral-peak frequency $f_gamma$, estimated from the largest
      interpolated 5–150-Hz peak of each network's trial-averaged Welch
      spectrum. Points and both-axis bars show means ± SEM across
      replicates. Labels give decay times; A's red dashed line is the
      equal-weight affine fit to six condition means. B's dotted line marks
      10% chance accuracy.
      *(C–H)* Fractions of E neuron–cycle pairs with 0, 1, 2 or ≥3 spikes at
      decay times 4.5, 6, 9, 12, 18 and 27 ms. Cycles span midpoints between
      detected inhibitory bursts, with edge intervals extending to
      presentation boundaries (Methods). Presentations without bursts are
      excluded. These opportunity-pooled distributions weight each pair
      equally, giving networks weight proportional to their detected cycle
      totals; they have no uncertainty bars. Equal-network distributions are
      shown in #manuscript-figure-ref(<fig:equal-network-cycles>, appendix: true).
      Source experiments:
      #link("/exp041/")[exp041] — #link("/exp041/")[_Firing Rate Tracks Gamma Frequency_] and
      #link("/exp046/")[exp046] — #link("/exp046/")[_One Spike per Gamma Cycle._]],
  ) <fig:cycle-participation>

  #editing-paragraph-label("P11")
  Across 167.2 million neuron–cycle pairs, 75.2% contained no spikes, 23.6%
  one spike and 1.15% two or more (Fig. 6C–H). Among active pairs, 95.4%
  contained exactly one spike; this fraction declined from 98.9% at 4.5-ms
  inhibitory decay to 83.8% at 27 ms. These distributions support
  predominantly one-spike participation among active neurons alongside
  widespread within-cycle silence, without establishing a strict firing
  ceiling or constant participating fraction.

  With equal network weighting, the largest within-condition change was
  0.191 percentage points in the one-spike fraction at 27 ms
  (#manuscript-figure-ref(<fig:equal-network-cycles>, appendix: true)).
  Across all 18 equally weighted networks, 76.35% of pairs contained no
  spikes, 22.09% one and 1.56% two or more: 98.44% contained at most one,
  compared with 98.85% under opportunity pooling. The latter gives greater
  weight to faster rhythms with more detected cycles; equal-network averaging
  also balances the six decay conditions. The predominance of zero- and
  one-spike pairs therefore persisted under both weighting choices in this
  three-replicate design.

  === The operating regime has asymmetric perturbation sensitivity

  #editing-paragraph-label("P12")
  Deleting 80% of naturally generated E/I transmitted spikes left mean
  accuracy at 89.3% for COBA and 89.2% for PING; complete deletion reduced
  both to 10.6% (Fig. 7A). Inserting events into both populations at each
  network's baseline E firing rate reduced accuracy to 82.8% and 38.5%,
  respectively; at twice baseline, accuracies were 72.6% and 11.4% (Fig.
  7B). PING therefore tolerated substantial deletion but was more sensitive
  to insertion at matched baseline-relative doses. Both interventions
  affected feedback and readout input, so they do not isolate rhythmic
  disruption as the cause of failure.

  #figure(
    data-image(
      data-file("exp037/perturbation_curves.svg"),
      width: 92%,
      alt: "COBA and PING test accuracy under spike deletion and baseline-relative spike insertion.",
    ),
    caption: [*Spike perturbations in trained classifiers.*
      *(A–B)* Inference-time perturbations of validation-selected
      unpenalised COBA (red squares) and PING (black diamonds), without
      retraining, using the common endpoint protocol. Lines and shading show
      mean ± sample SD across three independent training replicates; dashed
      lines mark 10% chance accuracy.
      *(A)* Independent deletion of transmitted E/I spikes, preserving
      neuronal resets.
      *(B)* Independent insertion into both populations at 0–200% of each
      network's unperturbed E firing rate; aggregation matches relative
      doses, not hertz. The same nominal per-neuron insertion rate applies
      to E and I. Collisions add no event; inserted events cause no reset or
      refractory period. Both interventions affect readout and recurrent
      transmission. Exact dose grids and operations are given in Methods.
      Source experiment: #link("/exp037/")[exp037] — #link("/exp037/")[_Dropped Spikes vs Added Noise._]],
  ) <fig:robustness>

  #editing-paragraph-label("P13")
  Classification accuracy remained near 90% across tested timesteps, although
  excitatory firing varied. This supports robustness after separate training
  at each timestep, without establishing fixed-weight convergence (Appendix A5).

  #editing-paragraph-label("P14")
  Count-preserving inhibitory replay produced opposite effects under
  independent-spike shifts and shared shifts within fixed 22.8-ms clock
  windows (Fig. 8). At proposed jitter SD 14 ms, independent shifts reduced
  mean excitatory firing from 16.6 to 0.0075 Hz and accuracy from 89.8% to
  11.9%; group shifts increased firing to 68.3 Hz while accuracy declined to
  82.5%. Replayed inhibitory firing remained 108.3 Hz. Thus, timing altered
  excitatory recruitment despite preserved inhibitory counts. The group
  windows were not detected cycles, and replay prevented inhibition from
  responding to ongoing excitatory activity.

  #figure(
    data-image(
      data-file("exp042/rhythm_compound.png"),
      width: 92%,
      alt: "Excitatory and inhibitory rasters, excitatory rate, accuracy and realised inhibitory rate under two inhibitory replay-jitter manipulations.",
    ),
    caption: [*Count-preserving inhibitory replay perturbations.* Recorded
      inhibitory spikes replace naturally generated outputs; excitatory
      activity and readout responses are recomputed with unchanged weights
      and inputs.
      *(A–B)* Illustrative digit-7 responses (test sample 0, seed 42) at
      proposed Gaussian-shift SD $sigma = 14$ ms, showing the same 200 E
      neurons (black) and 64 replayed I neurons (red).
      *(A, C)* Independent-spike jitter.
      *(B, D)* Shared shifts within fixed 22.8-ms clock windows, defined
      independently of detected bursts. Grid rounding, reflection and
      collision handling preserve each neuron's count (Appendix D).
      *(C–D)* Per-neuron E/I rates (black circles/red squares, left axes)
      and test accuracy (grey squares, right axes), averaged across three
      independently trained final-checkpoint classifiers using the common
      endpoint protocol; no uncertainty intervals. Displayed jitter values
      are 0, 0.5, 1, 2, 5, 9, 14 ms (C) and 0, 1, 3, 7, 14 ms (D); both
      share the zero-jitter replay control. Source experiment:
      #link("/exp042/")[exp042] — #link("/exp042/")[_Inhibitory Replay Perturbations Change Excitatory Firing._]],
  ) <fig:replay-perturbations>

  === PING networks classify continuously presented inputs

  #editing-paragraph-label("P15")
  A PING classifier trained across variable input rates correctly classified
  five successive digits while retaining hidden state (Fig. 9A–D). Sparse
  excitatory firing and inhibitory volleys persisted across changing
  durations and input rates. The first candidate satisfied the predefined
  five-correct selection criterion; the example therefore demonstrates
  feasibility, not reliability across arbitrary streams. Output state and
  counts reset at supplied boundaries, so this result does not establish
  autonomous boundary detection.

  #figure(
    data-image(
      data-file("exp082/continuous_stream_compound.png"),
      width: 92%,
      alt: "A correctly classified five-digit continuous stream with per-digit durations and input rates labelled, alongside accuracy across presentation duration and input rate.",
    ),
    caption: [*Spike-count classification in continuous MNIST streams.* Three
      independently trained, validation-selected PING classifiers used
      variable-rate training (0.5–25 Hz). During inference, hidden E/I state
      continued between digits; output voltage/counts reset at supplied
      boundaries.
      *(A–C)* Reused illustrative seed-42 stream selected by a predefined
      five-correct criterion: digits 1, 7, 9, 5, 2 at duration–rate pairs
      (100 ms, 5 Hz), (200 ms, 7.5 Hz), (50 ms, 25 Hz), (100 ms, 15 Hz),
      (200 ms, 10 Hz). A shows images, labels and conditions; B/C show the
      first 200 E/64 I neurons (black/red). Dotted lines mark supplied
      boundaries.
      *(D)* Softmax-normalized cumulative output-spike counts: true class
      red, others grey. Shares are not calibrated probabilities; the dashed
      0.5 line is not a decision threshold. Classification uses the largest
      final count (Methods).
      *(E–F)* #if shared-stream-images [Repeated quantitative evaluation on the
      same 40 ordered five-image streams for every network, duration and rate,
      with separate spike-encoding draws.] else [Quantitative evaluation with
      image samples only partly paired across duration–rate conditions.]
      Accuracy across 40 five-digit test streams per
      network/condition (200 decisions), with duration/rate fixed within
      each stream. E shows replicate-mean percentage accuracy at four
      durations and eleven rates, without uncertainty intervals. F reuses
      the 200-ms evaluations: mean ± SEM across three replicates on a
      logarithmic rate axis. The illustrative recording was selected separately
      from this grid. Source experiment:
      #link("/exp082/")[exp082] — #link("/exp082/")[_Spike-Count Classification in a Continuous Stream._]],
  ) <fig:continuous-stream>

  #editing-paragraph-label("P16")
  Quantitative streaming evaluation showed dependence on duration and input
  strength (Fig. 9E,F). At 25-Hz maximum-pixel input, increasing duration
  from 25 to 200 ms raised mean accuracy from #stream-accuracy(25, 25) to
  #stream-accuracy(200, 25). At 200 ms, increasing input from 0.5 to 5 Hz
  raised accuracy from #stream-accuracy(200, 0.5) to #stream-accuracy(200, 5).
  #if stream-upper-increasing [Mean accuracy then increased across the tested
  rates from 5 to 25 Hz, reaching #stream-accuracy(200, 25).] else [Accuracy
  ranged from #stream-pct(calc.min(..stream-upper-means)) to
  #stream-pct(calc.max(..stream-upper-means)) across 5–25 Hz without a strictly
  monotonic increase.] Brief presentations and weak drive constrained
  performance. #if shared-stream-images [The repeated grid paired image identity
  and order across conditions; its uncertainty remains conditional on one
  sampled image bank.] else [These comparisons also contain image-sampling
  variation because conditions were only partly paired.]

  == Methods

  === Experimental design

  #editing-paragraph-label("P17")
  We combined untrained circuit simulations, trained MNIST classifiers,
  interventions on those classifiers and a separate mean-field model (Table
  1). Inhibitory-timescale classifiers supplied frequency and
  cycle-participation measurements; unpenalised accuracy–rate classifiers
  supplied loop-insertion and spike-perturbation experiments. Unless
  specified otherwise, trained-network analyses used the common endpoint
  protocol below; Table 1 and Statistical reporting define checkpoints and
  replication.

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

  Here, $g_x[k]$ is conductance at step $k$, $w_(x j)$ the conductance
  increment from source $j$, and $s_j[k] in {0, 1}$ its delivered spike
  indicator. Decay times were $tau_E = tau_"AMPA" = 2$ ms and ordinarily
  $tau_I = tau_"GABA" = 6$ ms. Exponential-Euler voltage integration used
  $Delta t_"sim" = 0.1$ ms, except in the timestep study (Appendix A5). The
  inhibitory-timescale grid is specified below; exact updates and
  transmission ordering are in Appendix A1.

  We reused the fixed-0.1-ms spiking measurements, whose executed E/I
  refractory holds were already 1.2/0.6 ms. We recomputed the timestep
  comparison and the separate mean-field calculation with these same
  refractory durations. #if shared-stream-images [We subsequently repeated the
  quantitative continuous-stream evaluation with a shared image bank and
  explicitly specified the same refractory durations; its earlier selected
  showcase and the other spiking measurements were reused.] else [The remaining
  spiking measurements were reused.]

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

  Here, $tilde(u)_c[k]$ is pre-reset output state at step $k$, $N_t$ the
  presentation's step count, $z_c$ the class logit and $hat(y)$ the
  predicted digit. Exact output updates are in Appendix A; streaming uses
  the spike-count readout below.

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

  Here, $L_"total"$ is the objective, $L_"CE"$ mean classification
  cross-entropy, $B$ the minibatch size and $b$ the presentation index. The
  excitatory rate $r_(E,b)$ is that presentation's excitatory spike count
  divided by excitatory population size and duration in seconds. Division by
  $1 thin "Hz"$ makes the excess dimensionless. We used dimensionless penalty coefficient
  $lambda_"rate" = 0.041$ at ceilings $r_(E,"ceil")$ of 1, 2.5, 5, 10 and 25
  Hz; the unpenalised condition used $lambda_"rate" = 0$. The penalty acts
  on each presentation before minibatch averaging and imposes no hard rate
  limit. Only input/readout weights were trained; recurrence stayed fixed,
  enabled in PING and disabled in COBA.

  #editing-paragraph-label("P26")
  After each training epoch, we evaluated the 700 validation images using
  three independently seeded spike-encoding draws. The same encoding seeds
  were reused across epochs, keeping stochastic inputs fixed for checkpoint
  comparisons. For variable-rate classifiers, validation input-rate draws
  were also fixed across epochs. Cross-entropy and accuracy were averaged
  over validation images and encoding draws; the activity penalty was
  excluded from the validation loss.

  Best-validation checkpoints minimized mean validation cross-entropy over
  epochs 1–50, with ties resolved by higher accuracy, then earlier epoch.
  Final checkpoints were from epoch 50. Table 1 specifies the checkpoint
  used for each analysis.

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

  Here, $A_"lobe"$ and $A_"trough"$ are the preceding maximum and first
  minimum of the smoothed autocorrelation. Coupling-grid contrast used the
  900-ms post-burn-in interval; classifier contrast used the fixed
  reference-image presentation below. Undefined values remained missing for
  the coupling grid but became zero in the training diagnostic. Appendix B
  specifies feature-selection and boundary rules.

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

  #manuscript-note([*Note:* Double-check the effective voltage-noise scale
    $sigma_V = 4$ mV.])

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

  Here, $lambda_J^*$ is either eigenvalue of the refined crossing pair and
  $op("Im")$ its imaginary part in radians per millisecond; $f_"Hopf"$ is in
  hertz. The factor 1000 converts cycles per millisecond to hertz. Numerical
  settings are specified in Appendix C.

  #editing-paragraph-label("P33")
  We assessed criticality with 2-s upward/downward drive integrations,
  measuring peak-to-peak E-rate amplitude $A_"pp"$ in $"ms"^(-1)$ over the
  final 500 ms (Fig. 2H). Numerical supercriticality required no resolved
  hysteresis and a positive amplitude-squared slope, using Appendix C's
  predefined thresholds; no first Lyapunov coefficient was calculated. We
  repeated onset refinement at the six inhibitory decay times used for
  spiking classifiers, holding other mean-field parameters fixed. Figure 2I
  compares $f_"Hopf"$ with the median final-checkpoint spectral frequency
  $f_gamma$ across three classifiers per decay time.

  === Classifier comparisons and recurrent coupling

  #editing-paragraph-label("P34")
  We compared COBA and PING under the six activity conditions defined above,
  giving 36 networks (Fig. 3). Final-checkpoint test accuracy and excitatory
  firing rates used the common endpoint protocol. Unpenalised validation
  learning curves averaged replicate accuracies at each epoch;
  illustrative-example selection is specified in Appendix B4.

  #editing-paragraph-label("P35")
  We inserted reciprocal inhibition into the unpenalised COBA classifiers
  from the accuracy–rate comparison, holding learned input and readout
  weights fixed without retraining (Fig. 4).

  The dimensionless loop strength $s$ ranged from 0 to 1 in increments of
  0.1. At each strength, we newly initialized the E→I and I→E matrices from
  Gaussian distributions with parent means $mu_(E arrow.r I) = s$ µS and
  $mu_(I arrow.r E) = 2s$ µS, respectively, and standard deviations equal
  to 10% of each mean. We clamped negative draws to zero and divided weights
  by the presynaptic population size: 1,024 for E→I and 256 for I→E. Thus,
  $s = 0$ disabled the loop and $s = 1$ used the standard recurrent
  initialization.

  At each strength, we measured endpoint accuracy and E/I firing rates.

  #editing-paragraph-label("P36")
  We compared fixed recurrence at standard strength $s = 1$ with trainable
  recurrence initialized at $s = 1$, $0.1$ or $0$ (Fig. 5), giving 12
  networks. All conditions trained input and readout weights without an
  activity penalty. Endpoint accuracy and E/I firing rates used the common
  protocol. Contrast used the final training diagnostic from one 200-ms
  encoding of the first official-test digit-0 image, generated with seed 0
  and fixed across networks, without smoothing across epochs.

  For recurrent-weight comparisons, initial matrices were reconstructed
  using the original initialization seeds and parameters. At initialization
  and epoch 50, we pooled the three complete matrices separately for each
  connection direction and condition, giving 786,432 entries per pooled
  distribution. We calculated the fraction of strictly positive weights and
  the arithmetic mean weight across all entries, including zeros. Because the
  three matrices had equal size, each network contributed equally to these
  pooled summaries. We also calculated the same two statistics separately for
  each network and calculated their SEM across the three networks, separately
  at initialization and epoch 50. These were descriptive summaries, without
  significance tests of training-induced weight changes. Jointly trained
  weights within a matrix were not independent training replicates.

  === Inhibitory timescale and cycle participation

  #editing-paragraph-label("P37")
  We separately trained 18 PING classifiers at inhibitory decay times
  $tau_"GABA" = 4.5, 6, 9, 12, 18$ and $27$ ms (Fig. 6A–B). Input and
  readout weights were trained without an activity penalty; recurrence
  remained fixed at standard strength. Evaluation retained each assigned
  decay time and measured endpoint accuracy, excitatory firing rate and
  spectral-peak frequency using the common protocols.

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
  We reused the same classifiers' recorded spikes to measure excitatory
  counts per cycle (Fig. 6C–H) and compare network weighting
  (#manuscript-figure-ref(<fig:equal-network-cycles>, appendix: true)), without
  new inference or training. Inhibitory bursts were detected from smoothed
  population counts with peak separation scaled to each network's spectral
  period (Appendix B3).

  Cycle boundaries were midpoints between inhibitory-burst peaks, with edge
  intervals extending to presentation boundaries. Presentations without
  bursts were excluded. Each excitatory neuron–cycle pair was classified as
  containing 0, 1, 2 or ≥3 spikes. Opportunity-pooled fractions normalized
  summed counts within each decay condition across presentations and training
  replicates; overall fractions pooled all six decays. For equal-network
  estimates, we normalized each network's four counts separately and averaged
  its fractions equally with the other two networks in its condition, or
  across all 18 networks for the overall summary. Active-pair fractions
  excluded zero-spike pairs. Appendix B specifies discretization and
  single-burst conventions.

  === Spike perturbations and numerical resolution

  #editing-paragraph-label("P39")
  We separately applied deletion and insertion to unpenalised COBA and PING
  classifiers without retraining (Fig. 7A–B), using the common endpoint
  protocol and matched input encodings across perturbation conditions.

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

  Modified events supplied the readout and subsequent recurrent
  transmission. Insertion conditions were aggregated at matched
  baseline-relative doses.

  #editing-paragraph-label("P40")
  We evaluated separately trained PING classifiers at matched training and
  evaluation timesteps while holding refractory durations fixed. The full
  protocol and results are given in Appendix A5.

  #editing-paragraph-label("P41")
  We applied two count-preserving inhibitory replay perturbations to
  unpenalised PING classifiers (Fig. 8). Unperturbed inhibitory spikes
  recorded under the common endpoint protocol were shifted and replayed,
  replacing naturally generated inhibitory outputs while excitatory activity
  and readout responses were recomputed with unchanged weights and identical
  inputs.

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

  We measured endpoint accuracy and excitatory and replayed inhibitory
  rates. Delivered inhibition could not respond to ongoing excitatory
  activity.

  === Continuous streams

  #editing-paragraph-label("P42")
  Continuous-stream classifiers used the common isolated-image training
  protocol without an activity penalty, training input and readout weights
  while keeping recurrence fixed at standard strength (Fig. 9).

  For each presentation, the maximum-pixel input rate,
  $r_("input,max")$, was sampled uniformly from 0.5, 0.75, 1, 1.5, 2, 3,
  5, 7.5, 10, 15 and 25 Hz and held constant throughout that presentation.
  Rates were sampled independently for individual images within each
  minibatch and resampled on subsequent presentations.

  Continuous-stream classifiers used the output units defined above,
  with class logits given by accumulated spike counts:

  #math.equation(
    block: true,
    numbering: "(1)",
    $ z_c = sum_(k=1)^(N_t) s_c^"out"[k]. $,
  ) <eq:stream-spike-count-logits>

  Here, $s_c^"out"[k] in {0, 1}$ indicates an output spike; class index $c$,
  logit $z_c$ and presentation length $N_t$ are defined above. Counts
  entered cross-entropy directly without decay or duration normalization,
  using the same surrogate derivative. Classification selected the largest
  final count, breaking ties by lowest class index.

  Readout weights were drawn from a Gaussian distribution with dimensionless
  mean $mu_"out" = 0.05$ and standard deviation $sigma_"out" = 0.04$, then
  clamped below at zero, without fan-in normalization. Evaluations used
  each network's best-validation checkpoint with weights held fixed.
  #if shared-stream-images [The paired evaluation reused the same three trained
  classifiers; we did not retrain them for the change in sampling.]
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
  could recur across streams. #if shared-stream-images [We prespecified one bank
  of 40 ordered streams and reused the same image indices for every network,
  duration and rate. Thus, conditions shared both the target images and the
  preceding-image order. Spike encoding used separate generators with distinct
  seeds for all 5,280 network–condition–stream combinations (Appendix B5).
  The 26,400 decisions are repeated evaluations of this image bank, not that
  many independently sampled images.] else [Image samples varied across networks
  and most conditions, but ten duration–rate pairs accidentally shared streams
  within each network. Encoding used a separate random generator (Appendix B5).]

  #if shared-stream-images [This evaluation replaced an earlier grid whose
  arithmetic image-seed formula mapped 44 duration–rate combinations onto
  34 seeds per network. Ten condition pairs shared streams accidentally while
  the remainder were unpaired. Image-sampling and encoding seeds both changed
  in the repeated evaluation; differences from the earlier estimates cannot
  isolate an effect of pairing alone.]

  Accuracy included all 200 decisions per network and condition, including
  presentations with no output spikes. Figure 9F reuses the 200-ms
  evaluations from Fig. 9E; aggregation and uncertainty follow the reporting
  protocol below. Presentation and count-accumulation windows had equal duration,
  so their effects were not separated. We did not compare continuing hidden
  state with a hidden-state-reset control or test autonomous image segmentation.

  #editing-paragraph-label("P44")
  We reused the illustrative recording selected independently of quantitative
  evaluation using a predefined five-correct criterion, met by the first
  candidate (Appendix B4). #if shared-stream-images [It was not a stream from
  the shared quantitative image bank.]

  To visualize the evolving readout within each presentation, we
  transformed cumulative output-spike counts into softmax shares:

  #math.equation(
    block: true,
    numbering: "(1)",
    $ q_c[k] = exp(n_c[k]) / (sum_(d=0)^9 exp(n_d[k])). $,
  ) <eq:stream-count-shares>

  Here, $n_c[k]$ counts class-$c$ output spikes through step $k$ since the
  supplied boundary; $q_c[k]$ is its dimensionless share, with $d$ indexing
  classes. These uncalibrated shares visualize evidence; the 0.5 line is not
  a decision threshold.

  === Statistical reporting and reproducibility

  #editing-paragraph-label("P45")
  For trained-network comparisons, the unit of replication was one
  separately initialized and trained network, with three replicates per
  condition (seeds 42–44). These seeds controlled network initialization
  and, for variable-rate training, the sampling of input rates. The
  training-data split and endpoint test-image subset were fixed separately.
  Repeated interventions on a network were repeated measurements;
  individual weights, neurons, presentations and neuron–cycle pairs were not
  additional network replicates. Untrained circuit probes used a single
  seed, and mean-field calculations were deterministic.

  Each network's outcome was calculated before aggregation across
  networks. Displayed error bars or bands represent SEM in Figs. 3,
  5A–G, 6A–B and 9F and Appendix Fig. A1, and sample SD in Figs. 4 and 7A–B.
  Sample SD used the denominator $n - 1$, and SEM was calculated as
  $"SD" / sqrt(n)$, where $n = 3$ is the number of training replicates.

  #if shared-stream-images [For continuous-stream evaluation, all networks
  shared one image bank but used different encoding draws. Its SEM therefore
  summarizes variation across these trained networks and their encodings,
  conditional on the sampled images. One bank and one encoding draw per
  network, condition and stream do not separately estimate image-sampling,
  encoding and training variability. Image pairing does not make the 200
  sequential decisions per condition independent network replicates.]

  #manuscript-note([*Note:* Training cost limited each condition to three
    replicates; consequently, both sample SD and SEM are unstable estimates of
    between-network variability and should be interpreted cautiously.])

  Figures 8 and 9E show means without uncertainty intervals. Pooled weight
  summaries gave equal weight to each network because the matrices were
  equal-sized; their SEM used the three network-level summaries.
  Opportunity-pooled neuron–cycle distributions weighted networks in proportion
  to their available pairs; equal-network distributions instead averaged
  separately normalized network fractions. Both were descriptive, with
  individual-network values shown for the latter and no uncertainty intervals.
  A network with no detected cycles would have an undefined distribution and
  abort the equal-network analysis; none did. These summaries and fitted
  relationships do not treat neuron–cycle pairs as independent replicates.

  Cycle-participation analysis excluded presentations without a detected
  inhibitory burst. Twelve of the 18,000 network–image presentations met
  this criterion, all from the 27-ms inhibitory-timescale condition at
  seed 43. The resulting distributions comprised 17,988 contributing
  presentations and 167,178,240 neuron–cycle pairs. This exclusion affected
  cycle-participation measurements, not test accuracy or whole-presentation
  firing rates. Selection of the illustrative continuous stream was handled
  separately, as described above.

  #manuscript-note([*Note:* Re-investigate the cause of the 12 presentations
    without a detected inhibitory burst.])

  All 84 included networks completed 50 training epochs; their retained
  training records reported no skipped optimizer updates or batches with
  NaN outputs.

  #manuscript-note([*Note:* Establish the rationale for the replicate and
    evaluation sample sizes, and confirm whether failed or discarded
    execution attempts preceded the retained runs.])

  #editing-paragraph-label("P46")
  Spiking simulation, training and evaluation used custom Python/PyTorch
  code; all 84 classifiers were trained with PyTorch 2.11.0+cu128 on CUDA
  devices. Mean-field calculations used Python 3.10.19, NumPy 2.2.6 and
  SciPy 1.15.3. Experiment records retain stage-specific parameters, seeds,
  software/source revisions and original provenance for reused classifiers.

  #manuscript-note([*Note:* Add a permanent code/data archive and a
    figure-specific source and software record. Resolve the uncommitted
    changes recorded for some executions by including the executed source
    or patch; a commit identifier alone does not fully specify those
    executions.])

  == Appendix A — Discrete dynamics and gradient calculations

  *A1 — Discrete neuronal and synaptic updates.* These updates governed the
  spiking circuits in this study; their standard parameters are listed in
  Table A1. Let $k$ denote the number of completed
  updates. We updated conductances using Equation 2, first
  applying exponential decay and then adding the full weighted spike
  increments. Both recurrent pathways used spikes emitted during the
  preceding update; current external input entered the excitatory
  conductance before voltage integration. Consequently, newly emitted E
  or I spikes first affected the partner population in the following update.

  For each neuron, the updated conductances defined the total conductance
  and effective equilibrium voltage:

  #math.equation(
    block: true,
    numbering: "(1)",
    $ g_("tot")[k+1] &= g_L + g_(E)[k+1] + g_(I)[k+1], \
      V_(infinity)[k+1] &=
        (g_L E_L + g_(E)[k+1] E_E + g_(I)[k+1] E_I) / (g_("tot")[k+1]). $,
  ) <eq:hidden-effective-equilibrium>

  Holding these conductances constant over the step, we calculated the
  trial voltage

  #math.equation(
    block: true,
    numbering: "(1)",
    $ V^*[k+1] = max lr({
      V_"floor",
      V[k] + (V_(infinity)[k+1] - V[k])
        [1 - exp(- (Delta t_"sim" g_("tot")[k+1]) / C_m)]
    }). $,
  ) <eq:hidden-discrete-voltage>

  Here, $g_"tot"$ is in µS, and $V_infinity$ and $V^*$ are voltages in
  mV. The numerical lower bound $V_"floor" = -200$ mV guarded against
  extreme negative values: any calculated voltage below this bound was
  replaced by −200 mV before spike evaluation. Membrane capacitance $C_m$,
  timestep $Delta t_"sim"$, leak conductance $g_L$, synaptic conductances
  $g_E, g_I$, and reversal potentials $E_L, E_E, E_I$ retain their Methods
  definitions.

  After calculating the trial voltage, we decremented the integer refractory
  counter by one, with a lower bound of zero. A neuron emitted a spike only
  when the decremented counter was zero and $V^*[k+1] >= V_"th"$.
  Spiking or still-refractory neurons were assigned the reset voltage
  $V_"reset" = -65$ mV; otherwise, the trial voltage was retained. Each
  spike reloaded the counter with
  $N_"ref" = tau_"ref" / (Delta t_"sim")$, where $tau_"ref"$ is the
  population-specific refractory duration and $N_"ref"$ was an integer
  at every tested timestep. Spike eligibility therefore resumed on the
  $N_"ref"$-th subsequent update. Conductances continued evolving during
  refractory holds.

  *A2 — Output-state recurrence.* We updated each output unit after the
  hidden population, using excitatory spikes delivered during the same
  timestep. With the indexing of Appendix A1, the pre-reset state, output
  spike and retained state were

  #math.equation(
    block: true,
    numbering: "(1)",
    $ tilde(u)_(c)[k+1] &= beta_"out" u_(c)[k]
        + (1 - beta_"out") / h sum_j W_(j c)^"out" s_(j)^(E)[k+1], \
      s_(c)^("out")[k+1] &= bold(1) lr([
        tilde(u)_(c)[k+1] >= theta_"out"
      ]), \
      u_(c)[k+1] &= tilde(u)_(c)[k+1]
        - theta_"out" s_(c)^("out")[k+1]. $,
  ) <eq:output-state-recurrence>

  Here, $W_(j c)^"out"$ is the dimensionless weight from hidden excitatory
  neuron $j$ to output class $c$, and $s_(j)^(E)[k+1]$ denotes its
  delivered spike. The decay factor was
  $beta_"out" = exp(- (Delta t_"sim") / tau_"out")$, where
  $tau_"out"$ is the output decay time. The dimensionless quantity
  $h = (Delta t_"sim") / (1 thin "ms")$ expresses the timestep numerically
  in milliseconds and specifies the input-scaling convention. The indicator
  $bold(1)[dot.op]$ equals one when its condition holds and zero otherwise;
  $theta_"out"$ is the output threshold. Only one threshold was subtracted
  per update, so the retained state could remain above threshold after a
  spike.

  The readout modes differed in the quantity accumulated before this
  subtractive reset. Mean-state readout accumulated the pre-reset state for
  Figs. 3–8 and #manuscript-figure-ref(<fig:timestep-validation>, appendix: true), whereas spike-count readout accumulated
  the emitted spike for #manuscript-figure-ref(<fig:continuous-stream>):

  #math.equation(
    block: true,
    numbering: "(1)",
    $ M_(c)[k+1] &= M_(c)[k] + tilde(u)_(c)[k+1], \
      n_(c)[k+1] &= n_(c)[k] + s_(c)^("out")[k+1]. $,
  ) <eq:output-state-accumulators>

  Here, $M_c$ is the cumulative pre-reset state and $n_c$ the cumulative
  output-spike count. Neither accumulator decayed. The final logits were
  $(M_(c)[N_t]) / N_t$ or $n_(c)[N_t]$, respectively, where $N_t$ is the
  presentation's step count. Output state and accumulators began at zero;
  supplied image-boundary resets cleared them before that step's output
  drive was applied.

  *A3 — Surrogate and damped derivatives.* These derivatives governed the
  classifier training underlying Figs. 3–9 and #manuscript-figure-ref(<fig:timestep-validation>, appendix: true). Spikes used the binary forward
  threshold and surrogate backward derivative in Equation 5. To implement
  voltage-gradient damping, we used a stop-gradient operator $op("sg")(x)$,
  which returned $x$ but contributed zero derivative during backpropagation.
  We defined

  #math.equation(
    block: true,
    numbering: "(1)",
    $ cal(D)_(alpha_"grad")(x) = alpha_"grad" x
        + (1 - alpha_"grad") op("sg")(x), quad
      alpha_"grad" = 1 / d_"grad", $,
  ) <eq:voltage-gradient-scaling>

  where $d_"grad"$ is the damping divisor. We used voltage-gradient damping
  only in PING ($d_"grad" = 1,000$); COBA was undamped ($d_"grad" = 1$).
  This operation preserved the forward value while multiplying its backward
  derivative by $alpha_"grad"$.

  #manuscript-note([*Note:* Double-check whether we should use the same
  voltage-gradient damping in both PING and COBA.])

  For both hidden populations, we applied this operation to the
  exponential-Euler membrane increment $F[k]$ from Appendix A1, before
  voltage clipping and reset:

  #math.equation(
    block: true,
    numbering: "(1)",
    $ macron(V)[k+1] = V[k] + cal(D)_(alpha_"grad")(F[k]). $,
  ) <eq:damped-hidden-voltage>

  Here, $macron(V)$ denotes voltage before clipping. Damping scaled every
  derivative through $F[k]$, including its dependence on conductances and
  previous voltage, while the additive $V[k]$ path retained unit derivative.
  At fixed updated conductances, the resulting backward rule was

  #math.equation(
    block: true,
    numbering: "(1)",
    $ lr(( (∂ macron(V)[k+1]) / (∂ V[k]) ))_("BP")
      = 1 - alpha_"grad"
        [1 - exp(- (Delta t_"sim" g_("tot")[k+1]) / C_m)]. $,
  ) <eq:hidden-voltage-backward-rule>

  The subscript BP identifies the derivative used by backpropagation;
  timestep, total conductance and capacitance retain their Appendix A1
  definitions. Conductance recurrences and the surrogate spike function
  received no separate damping factor.

  Hidden reset and refractory decisions used Boolean masks and integer
  counters that were not differentiated. The retained-voltage branch carried
  zero derivative when a neuron spiked or remained refractory, while the
  spike-output branch retained surrogate gradients whenever the refractory
  mask permitted spiking. Voltage clipping also blocked gradients below the
  numerical floor.

  Output updates received no voltage-gradient damping. Their subtractive
  reset retained the derivative through the output spike, giving

  #math.equation(
    block: true,
    numbering: "(1)",
    $ lr(( (∂ u_(c)[k+1]) / (∂ tilde(u)_(c)[k+1]) ))_("BP")
      = 1 - theta_"out" psi (tilde(u)_(c)[k+1] - theta_"out"), $,
  ) <eq:output-reset-backward-rule>

  where $psi$ denotes the surrogate derivative in Equation 5 and the
  output-state symbols retain their Appendix A2 definitions. The current-step
  accumulator increment contributed a direct derivative for mean-state
  readout and a surrogate derivative for spike-count readout.

  *A4 — Initialization transformation.* For each input or recurrent
  projection, we generated independent Gaussian draws and Bernoulli retention
  indicators, then formed the initial weights:

  #math.equation(
    block: true,
    numbering: "(1)",
    $ X_(j i) &tilde cal(N)(mu_"init", sigma_"init"^2), quad
      M_(j i) tilde op("Bernoulli")(1 - q_"zero"), \
      W_(j i) &= (M_(j i) max(0, X_(j i))) /
        ((1 - q_"zero") N_"pre"). $,
  ) <eq:initial-weight-transformation>

  Here, rows $j$ index presynaptic neurons, columns $i$ index postsynaptic
  neurons, and $N_"pre"$ is the presynaptic population size. The parent
  mean $mu_"init"$ and standard deviation $sigma_"init"$ follow the
  projection-specific parameters in Table A1; $X_(j i)$ and $W_(j i)$ are
  conductance increments in µS. Input connections used $q_"zero" = 0.95$,
  giving a retention-compensation factor of 20; recurrent connections used
  $q_"zero" = 0$. Negative Gaussian draws produced additional zeros through
  lower clamping.

  Writing $mu_(+) = bb(E)[max(0, X_(j i))]$ for the mean after lower
  clamping, this transformation gave

  #math.equation(
    block: true,
    numbering: "(1)",
    $ bb(E)[W_(j i)] = mu_(+) / N_"pre", quad
      bb(E) lr([sum_(j=1)^(N_"pre") W_(j i)]) = mu_(+). $,
  ) <eq:initial-weight-expectations>

  Thus, compensation for zeroing preserved the expected summed incoming
  weight after lower clamping. Parent parameters described the Gaussian
  before these transformations; realised matrix means and incoming sums
  additionally depended on finite sampling and Bernoulli zeroing. The
  retention mask was used only during initialization, so initially zero
  trainable weights could become positive during optimization.

  Readout weights followed a separate initialization:

  #math.equation(
    block: true,
    numbering: "(1)",
    $ Y_(j c) tilde cal(N)(mu_"out", sigma_"out"^2), quad
      W_(j c)^"out" = max(0, Y_(j c)). $,
  ) <eq:initial-readout-weights>

  Here, $c$ indexes output classes. These weights were dimensionless and
  used no additional Bernoulli zeroing or fan-in normalization. Their parent
  mean and standard deviation follow Table A1 for ordinary classification
  and the continuous-stream Methods for streaming classifiers.

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
      #manuscript-figure-ref(<fig:matched-drive>) used input parent mean/SD 1.5/0.3 µS and PING recurrent means
      1.5/3 µS. Coupling strengths, inhibitory decay times and timesteps varied
      in their respective sweeps. #manuscript-figure-ref(<fig:continuous-stream>) used readout mean/SD 0.05/0.04 and
      variable input rates; stream durations varied during evaluation.
      Readout initialization values are rounded.],
  ) <tab:shared-parameters>

  *A5 — Timestep dependence after separate training.*

  We assessed timestep dependence using 15 PING classifiers: three training
  replicates (seeds 42–44) at each integration timestep of 0.05, 0.1, 0.2,
  0.3 and 0.6 ms. All networks underwent 50 training epochs without an
  activity penalty, with recurrent weights fixed at standard strength.
  Three existing 0.1-ms classifiers were reused; twelve networks were
  trained at the remaining timesteps. Excitatory and inhibitory refractory
  periods remained 1.2 and 0.6 ms, respectively, corresponding to integer
  step counts throughout.

  Each epoch-50 checkpoint was evaluated at its training timestep on the
  common 1,000-image test subset. Nominally 200-ms presentations were
  truncated to complete simulation steps, giving realised durations of
  199.8 ms at 0.3 and 0.6 ms and 200 ms otherwise. Mean per-neuron excitatory
  firing rates included all excitatory neurons and test presentations,
  using the realised duration. Accuracy and firing rate were summarised
  as means and SEM across the three training replicates at each timestep.

  Mean test accuracy ranged from 88.37% to 89.63% across the twelvefold
  timestep range (#manuscript-figure-ref(<fig:timestep-validation>, appendix: true)). Mean excitatory firing was 14.27 Hz at
  the finest timestep and 17.02 Hz at the coarsest. Classification
  performance therefore persisted across the tested resolutions, while
  excitatory activity remained timestep-dependent. Because each network
  was trained and evaluated at the same timestep, this comparison combines
  timestep effects during training and evaluation; it does not establish
  numerical convergence with fixed weights.

  #figure(
    data-image(
      data-file("exp044/dt_sweep.svg"),
      width: 80%,
      alt: "Hidden excitatory firing rate and test accuracy across five integration timesteps, with SEM across independently trained classifiers.",
    ),
    numbering: n => "A1",
    caption: [*Timestep dependence after separate training.*
      Mean per-neuron excitatory firing rate (black diamonds, left axis)
      and test accuracy (red squares, right axis) versus integration timestep
      on a logarithmic horizontal axis. Points show means across three
      independently trained PING classifiers per timestep (seeds 42–44);
      error bars indicate ±1 SEM. Epoch-50 checkpoints were evaluated at
      their training timestep on the same 1,000 MNIST test images. Firing
      rates use realised presentation durations: 199.8 ms at 0.3 and 0.6 ms,
      and 200 ms otherwise.
      Source experiment: #link("/exp044/")[exp044] — #link("/exp044/")[_Firing Rate Across the Timestep Sweep._]],
  ) <fig:timestep-validation>

  == Appendix B — Measurement algorithms and boundary cases

  *B1 — Spectral interpolation.* This refinement supplied the spectral
  frequencies in #manuscript-figure-ref(<fig:matched-drive>, panel: "E,G"),
  #manuscript-figure-ref(<fig:coupling-plane>, panel: "I") and
  #manuscript-figure-ref(<fig:cycle-participation>, panel: "A"). We refined the spectral maximum using
  a parabola through three neighbouring bins of linear power spectral density.
  Let $m$ index the largest-power bin within the search interval, choosing
  the lowest-frequency bin when maxima were tied. The interpolated
  spectral-peak frequency was

  #math.equation(
    block: true,
    numbering: "(1)",
    $ delta_m &= op("clip")_([-1/2, 1/2])
        [ (S_(m-1) - S_(m+1)) / (2 (S_(m-1) - 2 S_m + S_(m+1))) ], \
      f_gamma &= f_m + delta_m Delta f_"bin", $,
  ) <eq:spectral-peak-interpolation>

  Here, $S_j$ is the power spectral density at frequency bin $j$, $f_m$
  is the selected bin's frequency, $Delta f_"bin"$ is the frequency-bin
  spacing in hertz, and $delta_m$ is a dimensionless offset. Clipping limits
  the displacement to half a bin.

  We set the offset to zero when the denominator vanished or the selected
  bin lay at either end of the full spectrum. Neighbouring bins could lie
  outside the search interval, and the interpolated frequency was not
  subsequently restricted to that interval. Empty search intervals or
  zero in-band power yielded undefined estimates. The circuit estimator
  also returned no peak for fewer than two time samples, an empty neuronal
  population or entirely nonfinite in-band power. For inhibitory-timescale
  classifiers, analyses rejected nonfinite input traces and required finite
  network-level frequency estimates; nonfinite single-trial diagnostic
  estimates were omitted.

  #manuscript-note([*Note:* Double-check the frequency interpolation.])

  *B2 — Autocorrelation implementation.* This algorithm supplied the
  lobe–trough contrasts in #manuscript-figure-ref(<fig:coupling-plane>, panel: "C") and
  #manuscript-figure-ref(<fig:trainable-loop>, panel: "C"). We pooled excitatory spikes into
  complete 1-ms count bins, discarding any final incomplete bin. The
  dimensionless, uncentered autocorrelogram was

  #math.equation(
    block: true,
    numbering: "(1)",
    $ A_("corr")[ell] =
      (sum_(b=0)^(N_"bin" - ell - 1) n_(E)[b] n_(E)[b+ell])
      / ((N_"bin" - ell) macron(n)_E^2), quad ell = 1, dots, 100. $,
  ) <eq:autocorrelation-normalization>

  Here, $n_(E)[b]$ is the excitatory population spike count in bin $b$,
  $N_"bin"$ is the number of complete bins, $macron(n)_E$ is their mean count,
  and lag index $ell$ corresponds to $ell$ milliseconds. We computed the
  numerator by zero-padding the count sequence to the smallest power of two
  at least $2 N_"bin"$, then inverse-transforming the squared magnitude of
  its fast Fourier transform. This gives linear autocorrelation; division
  by $N_"bin" - ell$ accounts for the decreasing number of contributing
  bin pairs.

  Before feature extraction, we replaced the undefined zero-lag entry with
  its 1-ms neighbour and convolved the resulting array with weights
  $(0.25, 0.5, 0.25)$, using zero padding beyond both ends. Denoting the
  smoothed values by $tilde(A)_ell$, we selected the first lag
  $ell = 2, dots, 99$ satisfying $tilde(A)_ell <= tilde(A)_(ell-1)$ and
  $tilde(A)_ell < tilde(A)_(ell+1)$. The lobe was the maximum at positive
  lags preceding this trough, with ties resolved by the earliest lag.
  These values entered the contrast equation in Methods.

  Contrast was undefined for recordings with at most 101 complete bins,
  zero mean count, nonfinite autocorrelation values remaining after
  zero-lag replacement, no qualifying trough, or a nonpositive lobe–trough
  sum. Undefined values remained missing in the coupling-grid analysis
  and were replaced by zero in training diagnostics.

  #manuscript-note([*Note:* Double-check the autocorrelation implementation.])

  *B3 — Burst detection and cycle-boundary discretization.* This procedure
  supplied the pooled neuron–cycle distributions in
  #manuscript-figure-ref(<fig:cycle-participation>, panel: "C–H") and the
  equal-network comparison in
  #manuscript-figure-ref(<fig:equal-network-cycles>, appendix: true). For each
  presentation, we summed inhibitory spikes across neurons at each 0.1-ms
  simulation step and convolved the resulting trace with a unit-sum Gaussian
  kernel of standard deviation 1 ms. The kernel extended ±4 ms (81 samples),
  with zero padding beyond the presentation boundaries. Candidate bursts
  were interior local maxima reaching at least 5% of the maximum smoothed
  value within that presentation. Flat maxima were represented by their
  middle sample, rounded down for even-length plateaus. Minimum peak
  separation was half the period corresponding to the network's
  spectral-peak frequency $f_gamma$, converted to simulation steps and
  rounded down, with a minimum of one step. Smaller competing peaks were
  removed first. The same network-level frequency set this separation for
  every presentation.

  For consecutive detected peak indices $p_i$ and $p_(i+1)$, we placed
  the intervening boundary at

  #math.equation(
    block: true,
    numbering: "(1)",
    $ b_i = floor((p_i + p_(i+1)) / 2), $,
  ) <eq:cycle-boundary-discretization>

  where $b_i$ is an integer simulation-step index and the floor operation
  rounds downward. The outer boundaries were zero and $N_t$, the
  presentation's number of simulation steps. Excitatory spikes were counted
  separately for each neuron in the resulting half-open intervals: the
  starting boundary was included and the ending boundary excluded. A spike
  at an internal boundary therefore belonged to the following interval.
  The first and last intervals extended to the presentation boundaries and
  could contain partial cycles. When only one burst was detected, its
  counting interval covered the entire presentation; presentations without
  detected bursts contributed no neuron–cycle pairs.

  #manuscript-note([*Note:* Double-check burst detection and
    cycle-boundary discretization.])

  #figure(
    data-image(
      data-file("exp046/spikes_per_cycle_distribution_equal_network.svg"),
      width: 92%,
      alt: "Six distributions of excitatory spikes per neuron and inhibitory-burst cycle, showing equal-network means and three individual network values per spike-count category.",
    ),
    numbering: n => "B1",
    caption: [*Cycle-count distributions with equal network weighting.*
      *(A–F)* Inhibitory decay times 4.5, 6, 9, 12, 18 and 27 ms. Bars show
      arithmetic means of the fractions of neuron–cycle pairs containing
      0, 1, 2 or ≥3 excitatory spikes, after each network's counts were
      normalized separately. Black points show the three independently
      trained networks per condition, irrespective of their cycle totals;
      no uncertainty intervals are shown. The analysis reused the same
      epoch-50 classifiers and spike recordings as
      #manuscript-figure-ref(<fig:cycle-participation>, panel: "C–H"), including
      the same burst detection, edge intervals and zero-burst exclusions.
      Source experiment: #link("/exp046/")[exp046] — #link("/exp046/")[_One Spike per Gamma Cycle._]],
  ) <fig:equal-network-cycles>

  *B4 — Illustrative-example selection.*

  For #manuscript-figure-ref(<fig:accuracy-rate>), illustrative rasters used the seed-42, epoch-50 checkpoint
  from each architecture and the first digit-0 image in the official test
  set, presented for 400 ms.

  For #manuscript-figure-ref(<fig:loop-transfer>), illustrative rasters used the seed-42 classifier and the first
  image in the official test set, a digit 7, at $s = 0$ and $s = 1$. Both
  presentations lasted 200 ms. Fixed subsets of 200 E and 64 I neurons
  were selected uniformly without replacement using seed 0 for display.

  For #manuscript-figure-ref(<fig:replay-perturbations>), illustrative rasters used the seed-42 classifier
  and official test image 0, a digit 7, at $sigma = 14$ ms.

  For the illustrative stream (#manuscript-figure-ref(<fig:continuous-stream>, panel: "A–D")), we used the seed-42 classifier
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

  *B5 — Random-stream implementation.* These streams supplied the endpoint
  comparisons in Figs. 3–8 and the continuous-stream evaluation in
  #manuscript-figure-ref(<fig:continuous-stream>).
  Endpoint evaluation restarted a
  dedicated CPU encoding generator with seed 20260415, advancing it through
  successive batches of 64 images without reseeding between batches. Input
  spike trains were matched across networks when images, ordering, batch
  grouping, timestep, presentation duration and encoding rates agreed.
  Deletion and insertion used a separate generator with seed 20260416.
  Inhibitory replay shifts also used separate generators, with seeds
  determined by training replicate, jitter type and magnitude. Perturbation
  draws therefore did not advance the input-encoding stream.

  Quantitative continuous-stream evaluation sampled images from the full
  official test partition rather than the fixed endpoint subset.

  #if shared-stream-images [
    One NumPy image-sampling generator, initialized with seed 82000, advanced
    through 40 streams, selecting five distinct image indices uniformly
    without replacement within each stream. Images could recur between streams;
    neither digit classes nor the complete bank were stratified. We used the
    same ordered indices for all networks and conditions. We recorded the
    sampled indices and labels and checked every condition's labels against
    this bank.

    For spike encoding, we used an integer-pairing map to assign a distinct
    seed to each network, duration, rate and stream:

    #math.equation(
      block: true,
      numbering: "(1)",
      $ C(a,b) &= ((a+b)(a+b+1))/2 + b, \
        eta_"encode" &= 830000 + C(C(C(i_s,i_d),i_r),j). $,
    ) <eq:stream-random-seeds>

    Here, $C$ maps two non-negative integers $a,b$ to one integer;
    $i_s in {0,1,2}$ indexes training seeds 42–44,
    $i_d in {0, dots, 3}$ indexes durations 25, 50, 100 and 200 ms,
    $i_r in {0, dots, 10}$ indexes the eleven input rates in ascending order,
    and $j in {0, dots, 39}$ indexes streams. The seed $eta_"encode"$ is unique
    within the 5,280 quantitative network–condition–stream combinations.
    A dedicated CPU PyTorch generator was initialized for each combination
    and advanced across its five presentations. Image selection did not
    consume encoding draws. Matching images therefore did not match their
    realized spike trains across conditions or networks.

    The earlier image-seed formula combined scaled duration and rate by
    addition, causing the ten condition-pair collisions described in Methods.
    Its encoding seeds also omitted duration and rate. The repeated grid
    replaces both constructions; it does not retroactively pair the earlier
    measurements or establish how much each change contributed to differences
    in accuracy.
  ] else [
    For each network and duration–rate condition, one image-sampling generator
    advanced through 40 streams, selecting five distinct images without
    replacement within each stream; images could recur between streams.
    The image-sampling and encoding seeds were

    #math.equation(
      block: true,
      numbering: "(1)",
      $ eta_"images" &= 82000 + q + floor(10 d) + floor(100 r), \
        eta_"encode" &= 82000 + 100 q + j, $,
    ) <eq:stream-random-seeds>

    where $q in {42, 43, 44}$ is the training seed, $d$ is the numerical
    presentation duration in milliseconds, $r$ is the numerical maximum-pixel
    rate in hertz, and $j = 0, dots, 39$ indexes streams. The floor operation
    takes the integer part. The image formula assigned only 34 distinct seeds
    to 44 duration–rate combinations per network, accidentally pairing ten
    condition pairs. A separate encoding generator restarted for each stream
    and advanced across its five presentations. Encoding seeds were reused
    across duration–rate conditions, but images, Bernoulli probabilities and
    the number of random draws could differ; seed reuse did not impose
    identical spike trains.
  ]

  == Appendix C — Mean-field closure and numerical specification

  *C1 — Gain function and population-model assumptions.* This closure defined
  the mean-field calculations in #manuscript-figure-ref(<fig:coupling-plane>, panel: "G–I"). For each population
  $P in {E, I}$, denoting excitatory and inhibitory neurons, we used the
  stationary noisy-LIF gain:

  #math.equation(
    block: true,
    numbering: "(1)",
    $ Phi_P (I) = [tau_("ref",P) + tau_(m,P) sqrt(pi)
      integral_(a_P)^(b_P) e^(u^2) (1 + "erf"(u)) dif u]^(-1), $,
  ) <eq:noisy-lif-gain>

  #math.equation(
    block: true,
    numbering: "(1)",
    $ mu_(V,P) &= E_L + I / g_(L,P), \
      a_P &= (V_"reset" - mu_(V,P)) / sigma_V, quad
      b_P = (V_"th" - mu_(V,P)) / sigma_V. $,
  ) <eq:noisy-lif-bounds>

  Here, $I$ is mean input current in nA; $g_(L,P)$ is leak conductance in µS;
  and $mu_(V,P)$ is the corresponding effective mean voltage. The leak
  reversal, reset and threshold voltages are $E_L$, $V_"reset"$ and
  $V_"th"$, respectively, in mV. Membrane and refractory times,
  $tau_(m,P)$ and $tau_("ref",P)$, are in milliseconds, giving $Phi_P$ in
  $"ms"^(-1)$. The integration variable $u$ is dimensionless, and
  $"erf"$ is the error function. This expression gives the stationary
  firing rate of a LIF neuron driven by Gaussian white current noise
  (Eq. 21 in Brunel, 2000).#cite(1)

  To construct the population model, we approximated each synaptic current
  as $g(E_"rev" - V_m) approx g(E_"rev" - E_L)$, where $g$ is synaptic
  conductance, $E_"rev"$ its reversal potential and $V_m$ membrane voltage.
  This fixed the driving forces at rest and omitted conductance-dependent
  shunting. We replaced presynaptic spike trains by population rates while
  retaining exponential conductance decay. Each population rate then
  relaxed towards the steady-state rate predicted for its current input,
  with relaxation time $tau_(r,P) = tau_(m,P)$. This relaxation law was an
  additional modelling assumption. The effective noise scale $sigma_V = 4$
  mV was prescribed, rather than estimated from voltage recordings or
  calculated from population activity; the population equations themselves
  were deterministic.

  #manuscript-note([*Note:* Double-check the effective noise scale
    $sigma_V = 4$ mV.])

  *C2 — Numerical solution details.*

  We solved the two population-rate self-consistency equations, recovering
  synaptic conductances from their stationary relations. Each continuation
  used 401 equally spaced currents over the range specified in Methods.
  The first solve used the initial rates in Table C1; subsequent solves
  started from the last successful solution. Negative trial rates
  contributed zero conductance during root finding. Unsuccessful solves
  were skipped without updating the initial guess.

  We calculated the four-variable continuous-time Jacobian by centred
  finite differences. Among eigenvalues passing the imaginary-part
  threshold in Table C1, we selected the pair with the largest real part.
  The first change from negative to nonnegative real part bracketed onset.
  Brent’s method refined the crossing, recomputing the equilibrium and
  Jacobian at each trial current. Every refinement solve started from the
  lower bracket’s equilibrium. A failed solve or absence of an eligible
  complex pair terminated refinement.

  The upward amplitude sweep used 25 equally spaced currents from
  $I_"ext"^* - 0.1$ to $I_"ext"^* + 0.55$ nA, where $I_"ext"^*$ is the
  refined onset current. We initialized the lowest-current equilibrium
  with its excitatory rate increased by $10^(-3) thin "ms"^(-1)$, then
  carried each integration’s final state into the next, including the
  transition to the descending sweep. Amplitudes used the solver’s
  adaptive output times within the measurement window specified in
  Methods. Fewer than ten samples yielded zero amplitude. Failed
  integrations, incomplete trajectories or nonfinite states were rejected.

  For negative values of the dimensionless gain-integration variable $u$,
  we evaluated the scaled complementary error function $"erfcx"(-u)$ to
  avoid cancellation. For nonnegative arguments, the exponent $u^2$ was
  capped at 700. This numerical correction accompanied the
  refractory-period change, so differences from earlier calculations
  cannot be attributed to refractoriness alone.

  #manuscript-note([*Note:* Double-check Brent’s method and the
    handling of failed integrations.])

  #let numerical-settings-table = table(
    columns: (1fr, 2fr),
    table.header([Numerical operation], [Method and setting]),
    [Gain quadrature],
    [Adaptive QUADPACK integration; absolute and relative tolerances both
      $1.49 times 10^(-8)$; maximum 200 subintervals],
    [Equilibrium root finding],
    [MINPACK hybrid method; internally estimated Jacobian;
      relative-iterate tolerance $1.49012 times 10^(-8)$],
    [Initial equilibrium guess],
    [Excitatory and inhibitory rates $0.005$ and $0.002 thin "ms"^(-1)$,
      respectively],
    [Continuous-time Jacobian],
    [Centred differences with perturbations $10^(-6) thin "ms"^(-1)$ for
      rates and $10^(-6)$ µS for conductances],
    [Complex-eigenvalue eligibility],
    [Imaginary-part magnitude greater than $10^(-6) thin "ms"^(-1)$],
    [Onset refinement],
    [Brent’s method; absolute current tolerance $10^(-10)$ nA;
      relative tolerance $10^(-12)$; maximum 100 iterations],
    [Upward/downward integration],
    [LSODA; relative tolerance $10^(-7)$; scalar absolute tolerance
      $10^(-10)$; maximum step 1 ms; no supplied Jacobian],
  )
  #context figure(
    if target() == "html" {
      html.elem("div", attrs: (style: "display: flex; justify-content: center; overflow-x: auto;"), numerical-settings-table)
    } else {
      align(center, numerical-settings-table)
    },
    kind: table,
    numbering: n => "C1",
    caption: [*Numerical settings for mean-field continuation, onset
      refinement and amplitude sweeps.* Calculations used SciPy 1.15.3.
      Quadrature tolerances, the equilibrium stopping tolerance and
      Brent’s iteration limit used library defaults. The absolute
      integration tolerance applied to rates in $"ms"^(-1)$ and
      conductances in µS; relative tolerances were dimensionless. Unlisted
      solver options retained their defaults.],
  ) <tab:mean-field-numerics>

  *C3 — Criticality calculation.* This classification supplied the transition
  assessment in #manuscript-figure-ref(<fig:coupling-plane>, panel: "H").

  We defined the hysteresis gap as the maximum absolute difference between
  upward and downward peak-to-peak excitatory-rate amplitudes at matched
  currents. For the amplitude-squared regression, we included all
  upward-sweep points satisfying $I_"ext" > I_"ext"^* + 10^(-9)$ nA,
  where $I_"ext"^*$ is the refined onset current. Membership depended only
  on current, without an amplitude cutoff.

  For each included point $j$, we defined excess current
  $x_j = I_("ext",j) - I_"ext"^*$ and squared amplitude
  $y_j = (A_("pp",j)^(arrow.t))^2$, where the upward arrow identifies the
  ascending sweep. We fitted $y_j approx m x_j + c$ by unweighted least
  squares, with slope $m$ and freely fitted intercept $c$. The coefficient
  of determination was

  #math.equation(
    block: true,
    numbering: "(1)",
    $ R_"fit"^2 = 1 - (sum_j [y_j - (m x_j + c)]^2)
      / (sum_j (y_j - macron(y))^2), $,
  ) <eq:criticality-fit-r-squared>

  where $macron(y)$ is the mean squared amplitude over the included points.
  The numerator is the residual sum of squares, and the denominator is the
  centred total sum of squares. With fewer than two eligible points, both
  $m$ and $R_"fit"^2$ were set to zero; with zero total variance,
  $R_"fit"^2$ was set to zero.

  We classified the sampled onset as consistent with a supercritical
  transition only when the hysteresis gap was below
  $10^(-4) thin "ms"^(-1)$, $m > 0$, and $R_"fit"^2 > 0.9$. All other
  outcomes were labelled “subcritical/inconclusive”. This was a numerical
  classification of finite-duration sweeps; no first Lyapunov coefficient
  was calculated.

  == Appendix D — Exact replay transformations

  *D1 — Quantization and boundary reflection.* These transformations defined
  the inhibitory replay perturbations in #manuscript-figure-ref(<fig:replay-perturbations>).

  We converted Gaussian timing offsets to integer simulation-step
  displacements:

  #math.equation(
    block: true,
    numbering: "(1)",
    $ q = op("round")_"even" ((sigma Z) / (Delta t_"sim")), quad
      Z tilde cal(N)(0, 1). $,
  ) <eq:replay-offset-quantization>

  Here, $q$ is the integer displacement, $Z$ is a standard-normal draw,
  and $sigma$ and $Delta t_"sim"$ are the proposed jitter standard
  deviation and simulation timestep, respectively, in milliseconds.
  Rounding selected the nearest integer, with halfway ties resolved
  towards the even integer. Independent-spike jitter used one draw per
  event; group jitter used one shared draw per presentation and original
  source window.

  For an integer $z$, reflection into inclusive integer bounds $[L,U]$ was

  #math.equation(
    block: true,
    numbering: "(1)",
    $ cal(R)_(L,U) (z) = cases(
        L & U = L,
        L + w - abs(((z - L) op("mod") (2 w)) - w) & U > L,
      ), quad w = U - L, $,
  ) <eq:replay-boundary-reflection>

  where $w$ is the interval span and modulo returns a nonnegative
  remainder. This mapping implements repeated reflection, including
  offsets spanning multiple boundary crossings.

  For independent jitter, an event at original index $k$ received
  candidate index $cal(R)_(0,N_t-1) (k+q)$, where $N_t$ is the
  presentation’s number of simulation steps. For group jitter, let
  $k_"min"$ and $k_"max"$ be the earliest and latest original
  inhibitory-event indices within a source window, across all inhibitory
  neurons. We reflected the shared displacement into
  $[-k_"min", N_t-1-k_"max"]$, then added it to every event in that
  group. Group membership remained defined by the original timeline.
  Zero jitter and empty spike streams returned the original stream
  unchanged.

  *D2 — Collision resolution.* Collision-resolved replay supplied both
  #manuscript-figure-ref(<fig:replay-perturbations>) perturbation series.

  After reflection, we resolved duplicate destinations separately for
  each presentation and inhibitory neuron. Original chronological order
  determined event priority. At each pass, the earliest original event
  among those targeting the same timestep retained that destination;
  the remaining events advanced through offsets $+1, -1, +2, -2, dots$,
  measured from their respective reflected candidate indices.
  Out-of-range proposals were skipped. All destinations were checked
  again after each pass, allowing newly introduced collisions to be
  resolved under the same priority rule.

  Resolution ended when every destination was in range and unique within
  its presentation and neuron. The procedure aborted if an event
  required more than $2N_t$ search attempts. We reconstructed a binary
  replay stream and verified each inhibitory neuron’s spike count
  against its original count in every presentation, rejecting any
  mismatch. Collision adjustments acted on individual events and could
  therefore change relative timing within a shifted group.

  #reference-list((
    (text: [N. Brunel. “Dynamics of Sparsely Connected Networks of
      Excitatory and Inhibitory Spiking Neurons.”
      _Journal of Computational Neuroscience_ 8(3), 183–208 (2000).],
      doi: "10.1023/A:1008925309027"),
  ))
  #metadata("exp110-end")
]

#let body = if inputs.all(key => data-file(key) != none) {
  render-report(data-file)
} else [
  A required run is unavailable, so there is no content to display yet.
]
