#import "templates/article-layout.typ": journal-article
#import "templates/result-card.typ": result-figure-ref, result-card, with-result-sections
#import "templates/references.typ": journal-references
#import "/.demolab/lib.typ": data-json, data-image, cite
#import "templates/dataset.typ": data-file, inputs-ready, pending-report, run-view, input-assets
#import "templates/abstract.typ": journal-abstract
#import "templates/methods.typ": journal-methods
#let data-file = data-file.with(article: "exp038")

#let meta = (
  tags: ("data", "v35.4.0"),
  title: "Switching On the Inhibitory Loop",
  created_at: "2026-05-30T00:00:00Z",
  updated_at: "2026-08-31T00:00:00Z",
  description: "Enabling an inhibitory loop after feedforward training reduced excitatory firing but lowered classification accuracy; the experiment does not isolate a benefit of gamma timing.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp038",)
#let preview-figures = (
  (path: "exp038/loop_transfer_compound.png", label: "loop transfer compound"),
  (path: "exp038/ei_rasters.png", label: "ei rasters"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let run = data-json(data-file("exp038/numbers.json"))
#let cfg = run.config
#let eval_n = cfg.evaluation_samples_per_seed.first()
#let eval_pool = cfg.evaluation_pool_samples
#let ei = run.ei_sweep_summary
#let at(strength) = ei.filter(r => calc.abs(r.ei_strength - strength) < 0.001).first()
#let loop_off = at(0.0)
#let loop_on = at(1.0)
#let rate_off = calc.round(loop_off.hid_rate_hz)
#let rate_on = calc.round(loop_on.hid_rate_hz)
#let inhibition_on = calc.round(loop_on.inh_rate_hz)
#let rate_ratio = calc.round(loop_off.hid_rate_hz / loop_on.hid_rate_hz, digits: 1)
#let acc_off = calc.round(loop_off.acc)
#let acc_on = calc.round(loop_on.acc)
#let acc_cost = calc.round(loop_off.acc - loop_on.acc)
#let labels = run.at("illustrative_labels", default: none)
#let image-description = if labels == none { [the same test image] } else { [the same digit-#labels.ei_rasters.first() test image] }

#let body = [
  #journal-abstract(body: [
  We asked what happens when reciprocal inhibition is added after a feedforward
  classifier has already been trained. We kept the learned input and readout
  weights fixed while progressively enabling bidirectional excitatory–inhibitory
  coupling during inference.

  Stronger coupling grouped activity into bursts and sharply suppressed
  excitatory firing, but also reduced classification accuracy. This demonstrates post-training rate suppression, not a benefit of gamma timing or evidence that
  retraining would recover the lost accuracy.
  ])

  == Results

  #with-result-sections[

  #result-card[
  === Loop-strength rates and accuracy

  At full loop strength, E rate fell from approximately #rate_off to #rate_on Hz,
  I rate reached #inhibition_on Hz, and accuracy fell by #acc_cost percentage
  points. Because both coupling directions varied, lower activity alone does not
  identify a causal benefit of rhythm (#result-figure-ref(<fig:exp038-result-1>)).

  #figure(
    data-image(data-file("exp038/loop_transfer_compound.png"), width: 100%,
      alt: "Rasters with the loop off and enabled after training, followed by population firing rates and test accuracy across bidirectional loop strengths."),
    caption: [
      Reanalysed inference observations; no retraining. *(A)* Seed-42 raster
      with the loop off and *(B)* with the loop fully enabled, for
      #image-description; each shows 200 E neurons (black) and 64 I neurons
      (red). *(C)* Population rates and *(D)* accuracy on #eval_n test images
      per seed; curves show means ± sample SD across seeds 42–44.
    ],
  ) <fig:exp038-result-1>

  ]

  #result-card[
  === Loop-strength PING rasters

  Burst grouping increased across the sampled loop strengths. These illustrative
  panels do not estimate gamma frequency or establish a continuous transition (#result-figure-ref(<fig:exp038-result-2>)).

  #figure(
    data-image(data-file("exp038/ei_rasters.png"), width: 100%,
      alt: "Six E/I spike rasters from the same test image, at loop strengths zero through one, showing increasingly grouped bursts."),
    caption: [
      Seed 42, #image-description, at bidirectional loop strengths
      *(A–F)* $s = 0, 0.2, 0.4, 0.6, 0.8, 1$, respectively. Learned input and readout weights were fixed;
      recurrent E↔I weights were initialized at each strength without training.
      Rows show the same sampled 200 E and 64 I neurons over 200 ms.
    ],
  ) <fig:exp038-result-2>

  ]
  ]

  #journal-methods(
    orientation: [
  We reused networks from the #link("/exp022/")[exp022] — #link("/exp022/")[_Training Runs_] and
  reanalysed recorded inference observations. No new training or simulation
  was performed for this account.
    ],
    compute: [
  + *Reuse trained classifiers.* MNIST handwritten digits #cite(1) supplied
    6,300 training and 700 validation images from the official training partition.
    Conductance-based leaky-integrate-and-fire networks had 784 Poisson input
    channels, 1,024 excitatory (E), 256 inhibitory (I), and 10 output neurons;
    pixels set rates up to 25 Hz. Input and readout weights trained for 50 epochs;
    class scores were mean pre-reset output voltages. We selected the minimum
    mean validation cross-entropy over three fixed encoding draws, breaking
    ties by accuracy and then earliest epoch, rather than using final-epoch weights.

  + *Enable the loop after training.* Three feedforward controls, seeds 42–44,
    had no activity penalty or recurrent coupling during training.
    Dimensionless strength $s$ took eleven values from 0 to 1 in steps of 0.1;
    it set E→I and I→E initializer means to $s$ and $2s$, respectively, with
    standard deviations one tenth of those means and normalization by source
    population size. These lower-clamped normal weights replaced the zero
    recurrent matrices; learned input and readout weights stayed fixed, and
    E→E and I→I coupling stayed zero. The same network seed was reused across
    strengths; no optimization followed the intervention.
    ],
    analyse: [
  #set enum(start: 3)

  + *Evaluate responses.* Each strength used the same #eval_n images from the
    official #eval_pool\-image test partition, with 200 ms presentations and
    0.1 ms steps. Accuracy counted correct classifications; rates included all
    neurons and all evaluated presentations:

    #math.equation(block: true, $ r_P = 1 / (N_"eval" N_P T_"present") sum_(b=1)^(N_"eval") sum_(n in P) n_"spike"(b,n). $)

    Here $P$ denotes E or I, $N_P$ its neuron count, $N_"eval"$ the number of presentations,
    $T_"present"$ their duration in seconds, and $n_"spike"(b,n)$ neuron $n$'s spike count during
    presentation $b$; $r_P$ is in hertz. Curves show means and sample standard
    deviations across the three networks; single-image rasters are illustrative.
    ],
    present: [
  #set enum(start: 4)

  + *Probe input drive.* Auxiliary probes reused seed-42 classifiers trained
    with and without the loop. Uniform independent Poisson inputs covered
    26 rates between 0 and 100 Hz, with 32 trials per rate; a separate trained-loop
    probe used one test image at ten maximum pixel rates from 0 to approximately
    23.08 Hz. These recorded firing curves complement the
    #link("/exp023/")[exp023] — #link("/exp023/")[_Turning the PING Loop On_]; they are not
    additional loop-transfer accuracy evaluations.
    ],
  )
  #run-view("exp038", inputs)

  == Appendix: Training and probe settings

  #table(
    columns: 2,
    [Parameter], [Value],
    [Integration timestep], [0.1 ms],
    [Trial duration], [200 ms],
    [Evaluation corpus], [Official MNIST test partition (#eval_pool images)],
    [Evaluated images per seed], [#eval_n],
    [Training epochs], [50],
    [Optimizer], [AdamW; learning rate $4 times 10^(-4)$; zero weight decay],
    [Batch size; gradient-norm clip], [256; 1],
    [Excitatory / inhibitory synaptic decay], [2 / 6 ms],
  )

  The broader activity frontier contains 36 classifiers: two network
  configurations, three seeds, and six activity conditions (penalty off or
  ceilings of 25, 10, 5, 2.5 and 1 Hz). Its summaries preserve selected and
  final-epoch validation accuracy, final-epoch training E rate, and across-seed
  means and SEM; these are distinct from the inference measurements above.
  The loop-transfer comparison used only the three unpenalised feedforward
  classifiers. The #link("/exp025/")[exp025] — #link("/exp025/")[_Accuracy and Firing Rate With and Without Inhibition_] describes the
  broader training design.

  Feedforward and loop-enabled training used voltage-gradient damping of 1 and
  1,000 respectively; this difference affects the auxiliary between-model
  comparisons, not the within-classifier inference intervention. Dale's law
  was enforced, adaptive thresholds were disabled, and membrane time constants
  were not trained.

  Illustrative snapshots use test-image index 0, not selection by digit class.
  A fixed pseudorandom sample selects 200 E and 64 I neurons for display; reported
  firing rates use the full populations. Uniform-input E+I overlays add the
  two population means without weighting by neuron count, and are not a
  whole-network mean. Neither the transfer rasters nor these firing curves
  provide a spectral gamma-frequency estimate. The accuracy decline is
  consistent with changing the trained network's dynamics, but does not
  identify the readout as its sole cause or test recovery by retraining.

  #journal-references((
    (text: [Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner.
      “Gradient-based learning applied to document recognition.”
      _Proceedings of the IEEE_ 86(11), 2278–2324 (1998).],
      doi: "10.1109/5.726791"),
  ))
]
#body
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [What changes when an inhibitory loop is enabled after feedforward training? Sweep loop strength at inference while holding learned input and readout weights and the test-image subset fixed.],
    preview-figures, json-inputs: ("exp038",),
  )
}

#let meta = meta + (assets: input-assets("exp038", inputs))
#let body = journal-article("exp038", inputs, report-body, dataset-placed: inputs-ready(data-file, inputs))
