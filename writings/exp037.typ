#import "templates/article-layout.typ": journal-article
#import "templates/result-card.typ": result-figure-ref
#import "/.demolab/lib.typ": data-json, data-image
#import "templates/dataset.typ": data-file, inputs-ready, pending-report, run-view, input-assets
#import "templates/abstract.typ": journal-abstract
#import "templates/methods.typ": methods-heading
#let data-file = data-file.with(article: "exp037")

#let meta = (
  tags: ("data", "v36.0.0"),
  title: "Dropped Spikes vs Added Noise",
  created_at: "2026-05-30T00:00:00Z",
  updated_at: "2026-09-07",
  description: "Both trained networks tolerated substantial spike deletion, but PING accuracy fell more sharply under added spikes. The perturbations changed both recurrent feedback and readout input, so they do not isolate gamma gating.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp037",)
#let preview-figures = (
  (path: "exp037/perturbation_curves.svg", label: "perturbation curves"),
  (path: "exp037/perturbation_rasters.png", label: "spike perturbation rasters"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let run = data-json(data-file("exp037/numbers.json"))
#let relative = run.at("addition_axis", default: "reference_image_percent") == "per_seed_test_baseline_percent"
#let cfg = run.config
#let rounded(value) = calc.round(value, digits: 1)
#let mean(values) = values.sum() / values.len()
#let pert = run.perturbation_summary
#let point(model, mode, level) = pert.filter(r => r.model == model and r.mode == mode and calc.abs(r.level - level) < 0.001).first()
#let acc(model, mode, level) = rounded(point(model, mode, level).acc)
#let reference-rate(model) = mean(run.baseline_results.filter(r => r.model == model and r.rate_target_hz == none).map(r => r.rate_e))
#let eval_n = cfg.evaluation_samples_per_seed.first()
#let add_max = calc.max(..pert.filter(r => r.mode == "add").map(r => r.level))
#let knee-points = pert.filter(r => r.model == "ping" and r.mode == "add" and r.acc < 80).sorted(key: r => r.level)
#let knee = if knee-points.len() > 0 { knee-points.first().level } else { none }
#let labels = run.at("illustrative_labels", default: ())
#let trial-description = if labels.len() > 0 and labels.all(label => label == labels.first()) {
  [the same test image of digit #labels.first()]
} else { [the same test-image index in each condition] }
// HTML has no paged layout context; keep its images native and responsive.
#let report-image(path, alt, ratio: 0.58) = context {
  if target() == "html" {
    data-image(data-file(path), width: 100%, alt: alt)
  } else {
    // Fix the page frame so a short remainder cannot crop the image.
    layout(size => {
      let width = size.width
      box(width: width, height: width * ratio,
        data-image(data-file(path), width: width, height: width * ratio, fit: "contain", alt: alt))
    })
  }
}

  #journal-abstract(body: [
  We asked whether trained COBA and PING classifiers fail differently when hidden
  spikes are removed or spurious spikes are inserted during inference. We replayed validation-selected networks under matched deletion and insertion
  perturbations without retraining them.

  #if relative [
  We compared insertion at equal fractions of each network’s unperturbed
  excitatory firing rate. The accuracy curves measure tolerance to deletion and
  relative insertion; they do not isolate recurrent timing from firing-rate
  and readout effects.
  ] else [Both architectures tolerated substantial deletion, while added spikes damaged
  PING accuracy much more sharply than COBA accuracy. This reveals an asymmetric
  robustness profile, but does not separate recurrent timing from firing-rate
  and readout effects.]
  ])

  == Results

  + *Both networks tolerated substantial spike deletion.* At 80% deletion,
    COBA/PING retained #acc("coba", "drop", 0.8)%/#acc("ping", "drop", 0.8)%
    accuracy. #if pert.any(r => r.mode == "drop" and calc.abs(r.level - 0.9) < 0.001) [
    At 90%, accuracy fell to #acc("coba", "drop", 0.9)% and
    #acc("ping", "drop", 0.9)%, respectively.] Complete deletion reduced both
    to #acc("ping", "drop", 1)%
    (#result-figure-ref(<fig:exp037-result-1>, panel: "A")).

  + *Added spikes exposed a marked difference in robustness.*
    #if relative [With addition matched to each network’s baseline excitatory
    rate, COBA declined gradually while PING fell sharply around 80–110%
    addition. At 100%, accuracy was #acc("coba", "add", 100)% for COBA versus
    #acc("ping", "add", 100)% for PING; at #add_max%, it was
    #acc("coba", "add", add_max)% versus #acc("ping", "add", add_max)%.
    ] else [PING accuracy fell more sharply than COBA accuracy, although
    the shared nominal-Hz sweep did not match relative perturbation doses.]
    (#result-figure-ref(<fig:exp037-result-1>, panel: "B")).

  #figure(
    report-image("exp037/perturbation_curves.svg",
      "Mean test accuracy under deletion and insertion, with sample-standard-deviation bands across three seeds.", ratio: 0.55),
    caption: [
      *(A)* Random hidden-spike deletion. *(B)* Independent Bernoulli spike
      insertion. Lines and markers show means across seeds 42–44,
      #eval_n test images per seed; translucent bands show ±1 sample SD.
      The PING uncertainty band is grey and is most visible in *(B)*.
      The dashed line is nominal 10% chance.
      #if relative [The insertion axis is the prescribed percentage of each
      model–seed network’s own unperturbed test-set E rate. Lines average
      accuracy at matched percentages; nominal Hz can differ across seeds.
      ] else [The added-rate axis divides by final-epoch reference-image E rates
      (#rounded(reference-rate("ping"))/#rounded(reference-rate("coba")) Hz),
      not test-set baseline rates.]
    ],
  ) <fig:exp037-result-1>

  #set enum(start: 3)
  + *Rasters illustrate how transmitted activity changed.* Partial deletion
    preserved visible PING banding, while addition introduced spikes throughout
    the intervals between bands
    (#result-figure-ref(<fig:exp037-result-2>, panel: "A, B")).
    COBA activity thinned under deletion and became denser under addition
    (#result-figure-ref(<fig:exp037-result-2>, panel: "C, D")).
    These single-image examples do not quantify gamma coherence.

  #figure(
    report-image("exp037/perturbation_rasters.png",
      "Four panels of transmitted spikes: PING deletion and addition, then COBA deletion and addition, each at three perturbation levels.", ratio: 0.74),
    caption: [
      Seed-42 trials of #trial-description. *(A)* PING deletion;
      *(B)* PING addition; *(C)* COBA deletion; *(D)* COBA addition.
      Each deletion panel shows probabilities 0, 0.5 and 1; each addition
      panel shows nominal rates 0, #if relative [100] else [20] and
      #add_max#if relative [% of that network’s unperturbed test-set E rate] else [ Hz per neuron].
      Insertions were applied independently to E and I. Black marks denote
      E spikes and red marks I spikes, including inserted events.
      Each raster samples the same 200 E and 64 I neurons; annotated E rates
      use the full excitatory population over the displayed trial.
    ],
  ) <fig:exp037-result-2>

  #set enum(start: 4)
  + *Total firing concealed suppression of naturally generated spikes.*
    #if relative {
      let baseline = run.test_baselines.filter(r => r.model == "ping" and r.seed == 42).first()
      let row = run.perturbation.filter(r => r.model == "ping" and r.seed == 42 and r.mode == "add" and r.level == 100).first()
      let counts = row.perturbation.populations.e1
      let natural = counts.raw_spikes / counts.slots * 1000 / row.perturbation.dt_ms
      [For PING seed 42 at 100% addition, natural E firing fell from
      #rounded(baseline.e_rate_hz) to #rounded(natural) Hz, while successful
      insertions contributed #rounded(row.successful_insertion_hz.e1) Hz.
      This separately measured decomposition is not distinguished in
      #result-figure-ref(<fig:exp037-result-2>). It supports disruption of
      recurrent dynamics, but does not establish gamma disruption as the cause
      of classification failure.]
    } else [The displayed transmitted spikes include injected events; these
      rasters do not separate naturally generated activity from insertion.]

  #methods-heading()
  #set enum(start: 1)

  + *Evaluate trained classifiers.* Validation-selected COBA and PING
    checkpoints from seeds 42–44 were tested on the same #eval_n MNIST images
    without retraining. The reused training pool contained 6,300
    optimization and 700 validation images. We selected the minimum-validation-loss
    epoch from #cfg.epochs epochs in the unregularized conditions. Networks had
    1,024 E and 256 I neurons, with E→I→E coupling enabled for PING and disabled
    for COBA; recurrent weights were fixed, while input and readout weights were
    learned. Training voltage-increment gradients were divided by 1,000 for
    PING and 1 for COBA, so these are different trained recipes, not an isolated
    loop control. Trials lasted #cfg.t_ms ms at #cfg.dt ms resolution, with no
    warm-up. Pixel intensity set Poisson input rates up to 25 Hz; an independent
    perturbation generator preserved the input-encoding stream across conditions.
    Prediction selected the largest time-averaged output membrane potential.
    The wider activity-penalty comparison is described in
    #link("/exp025/")[exp025] — #link("/exp025/")[_Accuracy and Firing Rate With and Without Inhibition._]

  + *Delete spikes after neuronal spike generation.* At each timestep,
    conductances were updated using the preceding timestep’s transmitted spikes.
    Membrane integration and threshold/reset operations then generated natural
    spikes. Immediately afterward, before recording or readout input, each
    emitted E/I spike was independently removed with probability 0–100%, in
    10-percentage-point steps. Surviving spikes entered the current readout
    update and the next timestep’s recurrent feedback. Deletion did not undo
    the membrane reset associated with a removed spike
    (#result-figure-ref(<fig:exp037-result-1>, panel: "A");
    #result-figure-ref(<fig:exp037-result-2>, panel: "A, C")).

  + *Insert spikes at the same point in the flow.* After membrane integration
    and natural spike generation, but before recording or readout input,
    independent Bernoulli events were added to E/I outputs, capped at one spike
    per neuron per timestep. Collisions with existing spikes added nothing.
    Inserted spikes entered the current readout update and next timestep’s
    recurrent feedback, but did not themselves trigger a membrane reset or
    refractory period. #if relative [Nominal addition spanned 0–#add_max% of
    each model–seed’s unperturbed test E rate in 10-percentage-point steps.
    We calibrated this fixed baseline over the same #eval_n test images and
    multiplied it by the requested percentage divided by 100 to obtain Hz.
    ] else [Nominal addition spanned 0–#add_max Hz in 2 Hz steps.]
    The same nominal per-neuron rate was applied independently to E and I
    (#result-figure-ref(<fig:exp037-result-1>, panel: "B");
    #result-figure-ref(<fig:exp037-result-2>, panel: "B, D")).

  + *Separate natural and transmitted activity.* We counted natural spikes
    immediately before modification, successful insertions or deletions during
    modification, and transmitted spikes afterward. Natural activity therefore
    refers to neuron-generated spikes within the already perturbed network.
    Accuracy curves show means ± sample SD across three seeds, not confidence
    intervals. Illustrative rasters show transmitted activity from 200 E and
    64 I neurons for seed 42 and test-image index 0; rate annotations use the
    full E population. Insertion percentages use the test-set baseline rather
    than the illustrative image’s rate.

  == Discussion

  Spike deletion did not undo the membrane-voltage reset, and insertion did not
  trigger a reset or refractory period. These interventions modified transmitted
  events, affecting both recurrent feedback and readout input. A follow-up could
  perturb recurrent synaptic inputs while leaving the readout driven by natural
  E spikes, alongside a readout-only control, to distinguish circuit disruption
  from direct readout contamination.

  #run-view("exp037", inputs)

]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(data-file, inputs,
    [How do trained COBA and PING networks respond to deletion and insertion of hidden spikes? Compare recorded inference trials from validation-selected classifiers.],
    preview-figures, json-inputs: ("exp037",))
}

#let meta = meta + (assets: input-assets("exp037", inputs))
#let body = journal-article("exp037", inputs, report-body, dataset-placed: inputs-ready(data-file, inputs))
