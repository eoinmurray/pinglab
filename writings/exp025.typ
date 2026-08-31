#import "contents.typ": with-contents, with-result-sections
#import "/.demolab/lib.typ": data-json, data-image, cite, reference-list
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp025")

#let meta = (
  status: "[▦ DATA]",
  title: "Accuracy and Firing Rate With and Without Inhibition",
  date: "2026-05-30",
  updated_at: "2026-08-31",
  description: "Reused MNIST networks compare accuracy and excitatory firing rates with and without an inhibitory loop. Different gradient damping limits causal attribution to gamma timing.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp025",)
#let preview-figures = (
  (path: "exp025/results_compound.png", label: "results compound"),
  (path: "exp025/theta_p_fgamma.svg", label: "theta p fgamma"),
  (path: "exp025/low_w_in_sweep.svg", label: "low w in sweep"),
  (path: "exp025/w_in_scale_sweep.svg", label: "w in scale sweep"),
  (path: "exp025/w_in_scale_sweep_vs_rate.svg", label: "w in scale sweep vs rate"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
// Provenance (HOUSESTYLE H9/H19): every run number below is interpolated from the
// run's numbers.json, never hand-typed, so a re-run updates the prose automatically.
#let run = data-json(data-file("exp025/numbers.json"))
#let mean(a) = a.sum() / a.len()
#let rate-target(r) = if r.keys().contains("rate_target_hz") {
  r.rate_target_hz
} else if r.theta_u == none {
  none
} else {
  r.theta_u * 5
}
#let pfg-rows = if run.keys().contains("rate_target_p_fgamma") {
  run.rate_target_p_fgamma
} else {
  run.theta_p_fgamma
}

// Frontier points are averaged over the three independent training seeds.
#let res = run.results
#let baseline(model, key) = mean(res.filter(r => r.model == model and rate-target(r) == none).map(r => r.at(key)))
#let coba_off_rate = calc.round(baseline("coba", "rate_e"), digits: 1)
#let ping_off_rate = calc.round(baseline("ping", "rate_e"), digits: 1)
#let coba_off_acc = calc.round(baseline("coba", "final_acc"), digits: 1)
#let ping_off_acc = calc.round(baseline("ping", "final_acc"), digits: 1)
#let baseline_ratio = calc.round(baseline("coba", "rate_e") / baseline("ping", "rate_e"), digits: 2)

// Participation and frequency comparison (Figure 2): the five rate target sweep cells per model.
#let ping_pfg = pfg-rows.filter(r => r.model == "ping" and rate-target(r) != none)
#let coba_pfg = (
  pfg-rows.filter(r => r.model == "coba" and rate-target(r) != none).sorted(key: rate-target)
)
#let p_lo = calc.round(calc.min(..ping_pfg.map(r => r.p)), digits: 2)
#let p_hi = calc.round(calc.max(..ping_pfg.map(r => r.p)), digits: 2)
#let fg_hi = calc.round(calc.max(..ping_pfg.map(r => r.f_gamma)))
#let fg_lo = calc.round(calc.min(..ping_pfg.map(r => r.f_gamma)))
#let ping_acc_lo = calc.round(calc.min(..ping_pfg.map(r => r.acc)))
#let ping_acc_hi = calc.round(calc.max(..ping_pfg.map(r => r.acc)))
#let coba_acc_loose = calc.round(coba_pfg.last().acc)
#let coba_acc_tight = calc.round(coba_pfg.first().acc)
#let pfg_err_max = calc.round(calc.max(..ping_pfg.map(r => calc.abs((r.p * r.f_gamma - r.e_rate) / r.e_rate) * 100)), digits: 1)

// Low-W_in recruitment sweep (Figure 3), columns ordered 0.05 / 0.1 / 0.3 / 0.9.
#let low_accs = run.low_w_in_sweep.map(r => calc.round(r.final_acc, digits: 1))
#let low_is = run.low_w_in_sweep.map(r => calc.round(r.rate_i, digits: 1))

// Inference-time W_in scale sweep (Figures 4-5), trained point at s = 1.
#let ping_ws = run.w_in_scale_sweep.filter(r => r.cell == "ping@rt1hz" or r.cell == "ping@tu0.2")
#let coba_ws = run.w_in_scale_sweep.filter(r => r.cell == "coba@rt1hz" or r.cell == "coba@tu0.2")
#let coba_pen_s3 = calc.round(coba_ws.filter(r => r.scale == 3.0).first().penalty)
#let coba_acc_s3 = calc.round(coba_ws.filter(r => r.scale == 3.0).first().acc)
#let coba_e_s3 = calc.round(coba_ws.filter(r => r.scale == 3.0).first().rate_e)
#let ping_star_e = calc.round(ping_ws.filter(r => r.scale == 1.0).first().rate_e, digits: 1)
#let coba_star_e = calc.round(coba_ws.filter(r => r.scale == 1.0).first().rate_e, digits: 1)
#let ping_plateau = calc.round(calc.max(..ping_ws.map(r => r.acc)))

#let body = [
  == Abstract

  Asked how recurrent inhibition and activity constraints shape the trade-off
  between MNIST accuracy and excitatory firing. Compared trained COBA and PING
  families across activity ceilings, cycle participation, oscillation frequency
  and input coupling.

  PING had lower excitatory rates and retained more accuracy under strict
  ceilings, but no structural rate floor appeared. The comparison is confounded
  by gradient damping and therefore neither isolates a benefit of gamma timing
  nor measures energy use.

  == Results

  #with-result-sections[

  === Test accuracy against excitatory rate across activity ceilings

  At the unpenalised operating points, PING reached #ping_off_acc% at
  #ping_off_rate Hz and COBA reached #coba_off_acc% at #coba_off_rate Hz. The
  comparison includes different gradient damping, so it does not establish a
  structural lower firing-rate limit.

  #figure(
    data-image(data-file("exp025/results_compound.png"),
      width: 100%,
      alt: "Two-by-two panel: COBA and PING single-trial rasters, per-epoch learning curves, and the accuracy–rate frontier across hidden-E rate ceilings.",
    ),
    caption: [
      *Top:* illustrative 400 ms rasters for the same digit-0 example, seed 42;
      E spikes black, I spikes red. *Bottom left:* baseline validation accuracy
      over training. *Bottom right:* test accuracy versus mean E firing rate
      across activity ceilings; means ± SEM over three seeds, unpenalised points
      starred. These rates are test-set averages, not raster estimates. PING
      black, COBA red in the lower panels.
    ],
  )

  === PING participation and frequency across activity ceilings

  PING participation varied from #p_lo to #p_hi and oscillation frequency from
  approximately #fg_hi to #fg_lo Hz. The $p_"part" f_gamma$ approximation
  differed from measured E rate by up to #pfg_err_max%, so participation was not
  constant. PING accuracy spanned #ping_acc_lo–#ping_acc_hi%; COBA fell from
  #coba_acc_loose% to #coba_acc_tight% as the ceiling tightened.

  #figure(
    data-image(data-file("exp025/theta_p_fgamma.svg"),
      width: 100%,
      alt: "PING participation fraction p and oscillation frequency f_gamma across the activity-ceiling sweep, with the p·f_gamma product overlaid on the measured E rate.",
    ),
    caption: [
      Five penalised conditions per model, seed 42, #run.recipe.pfg_samples
      test images each. The dashed curve is the $p_"part" f_gamma$
      approximation. These are individual-seed measurements, not across-seed
      estimates.
    ],
  )

  === PING learning curves across initial input coupling

  Final validation accuracies were #low_accs.at(0)% / #low_accs.at(1)% /
  #low_accs.at(2)% / #low_accs.at(3)%, while final I rates were
  #low_is.at(0) / #low_is.at(1) / #low_is.at(2) / #low_is.at(3) Hz. The similar
  endpoints across these initializations do not prove basin attractivity.

  #figure(
    data-image(data-file("exp025/low_w_in_sweep.svg"),
      width: 100%,
      alt: "Across-seed mean per-epoch validation accuracy and E/I firing rates for four PING input summed-coupling parent means, one column per condition.",
    ),
    caption: [
      PING learning curves for initial input-coupling means 0.05, 0.1, 0.3,
      and 0.9 (columns); rate ceiling 1 Hz throughout. Lines and shading show
      means ± SEM across seeds 42–44. *Top:* validation accuracy, despite the
      panel label “Test accuracy”. *Bottom:* validation E (black) and I (red)
      rates.
    ],
  )

  === Accuracy and population rates across inference input scaling

  COBA's penalty reached approximately #coba_pen_s3 at $s = 3$. The empirical
  inhibitory-rate crossing marks a sampled transition, not a fitted bifurcation.

  #figure(
    data-image(data-file("exp025/w_in_scale_sweep.svg"),
      width: 100%,
      alt: "Inference-time W_in scale sweep: CE loss, activity penalty, total objective, test accuracy, and E/I rates versus scalar s for PING and COBA.",
    ),
    caption: [
      Seed-42 networks trained with a 1 Hz ceiling; input weights scaled
      at inference over #run.recipe.scales.len() values, all other weights fixed.
      *Top:* cross-entropy, rate penalty, and their sum. *Bottom:* test accuracy
      (dotted chance line), E rate, and I rate; #run.recipe.evaluation_samples
      images per condition. PING black, COBA red. Dashed $s = 1$ marks training;
      the dotted marker labelled $f^*$ denotes the empirical input scale
      #run.plot_data.scale_crossing where I rate crosses 0.05 Hz, not a fitted
      bifurcation. Penalty and total-objective axes stop at 4.
    ],
  )

  === Accuracy against excitatory rate across inference input scaling

  PING's highest sampled accuracy was approximately #ping_plateau%. At input
  scale $s = 3$, COBA reached approximately #coba_acc_s3% at #coba_e_s3 Hz.
  This single direction of weight scaling does not map the full loss landscape.

  #figure(
    data-image(data-file("exp025/w_in_scale_sweep_vs_rate.svg"),
      width: 100%,
      alt: "The W_in scale sweep re-projected with hidden E rate on the x-axis, trained operating points starred for PING and COBA.",
    ),
    caption: [
      Figure 4 replotted against mean E rate. Stars mark the trained
      points: PING #ping_star_e Hz and COBA #coba_star_e Hz.
    ],
  )

  ]

  == Methods

  We reused networks and learning histories from the
  #link("/exp022/")[shared training study] and reanalysed recorded inference
  measurements; no new training or simulation was performed.

  + *Prepare digit inputs.* Training used 6,300 MNIST images and 700 validation
    images from the official training partition. Pixels drove 784 Poisson channels
    at a 25 Hz maximum, for 200 ms with 0.1 ms steps.

  + *Compare network configurations.* Both conductance-based networks had 1,024
    excitatory (E), 256 inhibitory (I), and 10 output leaky-integrate-and-fire neurons.
    Pyramidal-interneuron gamma (PING) enabled fixed E↔I coupling; COBA disabled it;
    E→E and I→I coupling were zero throughout. Only input and readout weights
    trained; class scores were mean pre-reset output membrane voltages
    (#link("/exp006/")[readout specification]). Voltage-gradient damping differed:
    1 for COBA, 1,000 for PING.

  + *Train with activity ceilings.* Networks trained for 50 epochs with AdamW
    (zero weight decay), learning rate $4 times 10^(-4)$, batch size 256, and
    gradient-norm clipping at 1. Cross-entropy was supplemented by:

    #math.equation(block: true, numbering: "(1)", $ r_b = 1 / (N_E T_"present") sum_(n in E) n_"spike"(b,n), quad
      L_"rate" = lambda_"rate" / B sum_b max(r_b - r_(E,"ceil"), 0)^2. $)

    Here $n_"spike"(b,n)$ counts excitatory neuron $n$'s spikes in presentation $b$, $E$ is the
    excitatory population, $N_E$ its size, $T_"present"$ its duration in seconds, and $B$ minibatch
    size. Rates $r_b$ and ceilings $r_(E,"ceil")$ are in hertz; $lambda_"rate" = 0.041$
    $"Hz"^(-2)$ weights the dimensionless penalty $L_"rate"$. Six ceiling conditions
    (#link(<sec-training-settings>)[Training settings]) and three seeds yielded 36 networks.

  + *Evaluate training endpoints.* Final-epoch weights, rather than
    validation-selected weights, supplied the endpoint comparisons. Each network
    was evaluated on the same 1,000 official-test images; frontier points show
    means and standard errors across seeds 42–44.

  + *Measure cycle participation.* For seed 42, oscillation frequency $f_gamma$
    was the 5–150 Hz peak of trial-averaged Welch spectra of E activity #cite(1).
    Participation $p_"part"$ was the fraction of E-neuron/cycle pairs containing at least
    one spike, with cycles delimited by I-burst midpoints (#link(<sec-measurement-details>)[Measurement details]).

    #math.equation(block: true, numbering: "(1)", $ r_E approx p_"part" f_gamma. $)

    This diagnostic approximation relates mean E rate $r_E$ (Hz), dimensionless
    $p_"part"$, and $f_gamma$ (Hz); repeated spikes and differing cycle/frequency
    aggregation prevent treating it as an identity.

  + *Vary input coupling.* Twelve PING networks used four initial input-coupling
    means and three seeds, with a 1 Hz ceiling throughout; validation histories
    measured recruitment during training. Separately, seed-42 PING and COBA
    networks trained at 1 Hz were evaluated after multiplying all input weights
    by dimensionless $s in [0.05, 3]$, holding other weights fixed, on the same
    1,000 test images at 24 scales.

  #run-view("exp025", inputs)

  == Appendix: Training settings <sec-training-settings>

  #table(
    columns: 2,
    [Parameter], [Value],
    [Integration timestep $Delta t_"sim"$], [0.1 ms],
    [Presentation duration $T_"present"$], [200 ms; illustrative rasters use 400 ms],
    [MNIST training pool], [7,000 official-training images: 6,300 optimizer-training / 700 validation],
    [Evaluation], [Fixed 1,000-image official-test subset],
    [Epochs], [50],
    [Rate ceilings], [Penalty off, 25, 10, 5, 2.5, and 1 Hz],
    [Independent seeds], [42, 43, 44],
    [Trainable weights], [$W_"in"$: $784 times 1024$; $W_"out"$: $1024 times 10$; 813,056 parameters],
    [Stored parameters], [2,451,456, including fixed and zero weights],
    [Fixed synaptic decay], [$tau_"AMPA" = 2$ ms; $tau_"GABA" = 6$ ms],
  )

  Input weights $W_"in"$ used a lower-clamped normal initializer with parent
  mean 0.9 and standard deviation 0.09, with 95% Bernoulli zeroing, sparsity
  compensation, and fan-in normalization. The mean parameter describes
  expected summed input coupling, not a per-connection mean. Initial zeros
  remained trainable; this was not a permanent connectivity mask. The four
  recruitment conditions replaced the mean by 0.05, 0.1, 0.3, or 0.9, with
  standard deviation one tenth of the mean.

  Readout weights $W_"out"$ used a directly specified lower-clamped normal
  initializer (mean 1.12060546875, standard deviation 0.8349609375).
  PING's fixed E→I and I→E initializers used summed-coupling means 1 and 2,
  respectively, with standard deviations 0.1 and 0.2 and normalization by
  source-population size. COBA set both connections to zero. Dale's law was
  enforced; membrane time constants were not trained and adaptive thresholds
  were disabled.

  Validation used three fixed Poisson draws per image. The training study also
  retained checkpoints selected by minimum validation cross-entropy, with ties
  resolved by accuracy then earliest epoch; the comparisons here used epoch 50.

  == Appendix: Measurement details <sec-measurement-details>

  Welch spectra used each full 200 ms demeaned E-population trace with a Hann
  window; constant traces were excluded. Spectra were averaged before selecting
  the 5–150 Hz peak, refined by three-bin parabolic interpolation capped at half
  a frequency bin. The symbol $f_gamma$ denotes this estimator even when its
  peak falls below the usual gamma range.

  For participation, I-population activity was smoothed with a 1 ms Gaussian.
  Peaks required 5% of the trial maximum and at least half a cycle of separation,
  using that trial's E spectral peak. Cycle boundaries lay midway between I
  peaks, with the trial endpoints closing the first and last cycles. Trials with
  no usable frequency or I peak were omitted from participation; active
  neuron/cycle pairs were pooled across accepted trials. This extends the
  #link("/exp041/")[frequency–rate comparison] using the
  #link("/exp046/")[cycle-participation measurement].

  Input-scale values were 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
  0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1, 1.15, 1.3, 1.5, 1.75, 2, 2.5,
  and 3. The crossing marker is the midpoint of the first adjacent pair whose
  mean I rates cross 0.05 Hz. The inference objective adds cross-entropy to
  Equation 1's sample-wise penalty; its quadratic excess-rate dependence does
  not imply a quadratic dependence on input scale.

  #reference-list((
    (text: [P. D. Welch. “The use of the fast Fourier transform for the estimation
      of power spectra: A method based on time averaging over short, modified
      periodograms.” _IEEE Transactions on Audio and Electroacoustics_ 15(2),
      70–73 (1967).], doi: "10.1109/TAU.1967.1161901"),
  ))

]
#body
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [How does inhibition change the trade-off between accuracy and firing rate? Compare COBA and PING across activity ceilings, then examine sensitivity to input coupling.],
    preview-figures, json-inputs: ("exp025",),
  )
}

#let meta = meta + (assets: input-assets("exp025", inputs))
#let body = with-datasets("exp025", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-contents(body)
