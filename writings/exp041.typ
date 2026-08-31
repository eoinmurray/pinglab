#import "contents.typ": with-contents, with-result-sections
#import "/.demolab/lib.typ": data-json, data-image, cite, reference-list
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp041")

#let meta = (
  status: "[▦ DATA]",
  title: "Firing Rate Tracks Gamma Frequency",
  date: "2026-06-02",
  updated_at: "2026-08-31",
  description: "Across PING networks trained at different inhibitory decay times, compare excitatory firing rate with gamma frequency and test accuracy.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp041",)
#let preview-figures = (
  (path: "exp041/training_curves.svg", label: "training curves"),
  (path: "exp041/psds.svg", label: "psds"),
  (path: "exp041/raster_strip.png", label: "raster strip"),
  (path: "exp041/rate_vs_fgamma.svg", label: "rate vs fgamma"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#set math.equation(numbering: "(1)")
#let run = data-json(data-file("exp041/numbers.json"))
#let cfg = run.config
#let fit = run.fit
#let fa = calc.round(fit.a_affine, digits: 2)
#let fp = calc.round(fit.p_affine, digits: 3)
#let fr = calc.round(fit.r2_affine, digits: 3)
#let po = calc.round(fit.p_origin, digits: 3)
#let pr = calc.round(fit.r2_origin, digits: 3)
#let fgs = run.results.map(x => x.f_gamma_hz)
#let ers = run.results.map(x => x.e_rate_hz)
#let fg_lo = calc.round(calc.min(..fgs))
#let fg_hi = calc.round(calc.max(..fgs))
#let er_lo = calc.round(calc.min(..ers), digits: 1)
#let er_hi = calc.round(calc.max(..ers), digits: 1)
#let accs = run.results.map(x => x.acc)
#let acc_lo = calc.round(calc.min(..accs))
#let acc_hi = calc.round(calc.max(..accs))

#let body = [
  == Abstract

  - Asked whether the excitatory firing rate of trained PING classifiers changes
    systematically with their gamma frequency.
  - Reused networks trained with different inhibitory decay times and measured
    their final population rhythms, activity and MNIST performance.
  - Slower rhythms accompanied lower excitatory firing while useful classification
    persisted across the sweep.
  - The association supports a cycle-participation account, but does not prove
    constant participation or identify a physical non-rhythmic baseline.

  == Results

  #with-result-sections[

  === Validation accuracy and excitatory rate across inhibitory decay times

  #figure(
    data-image(data-file("exp041/training_curves.svg"),
      width: 100%,
      alt: "Validation accuracy and excitatory firing rate over training across the inhibitory-decay sweep.",
    ),
    caption: [
      Reused validation accuracy (top) and excitatory rate (bottom), one line per
      trained network across #cfg.epochs epochs. Subsequent test measurements
      used final-epoch weights; accuracy convergence did not select the epoch.
    ],
  )

  === Population spectra across inhibitory decay times

  Per-network interpolated frequencies spanned approximately #fg_lo–#fg_hi Hz
  across the inhibitory-decay sweep.

  #figure(
    data-image(data-file("exp041/psds.svg"),
      width: 100%,
      alt: "Population-E power spectra by τ_GABA, gamma peak shifting with the inhibitory time constant.",
    ),
    caption: [
      Population-E power spectral densities (PSDs), averaged across trials and
      then seeds at each inhibitory decay time. Dots mark binned peaks of these
      displayed means.
    ],
  )

  === Spike rasters across inhibitory decay times

  #figure(
    data-image(data-file("exp041/raster_strip.png"),
      width: 100%,
      alt: "One MNIST trial through each τ_GABA network; the gamma cycle period lengthens with τ_GABA.",
    ),
    caption: [
      The same illustrative MNIST image at each decay time, using seed 42.
      Each panel shows 200 excitatory and 64 inhibitory neurons during the first
      100 ms; displayed rates use the full populations and 200 ms trial.
      These probes illustrate timing, not the population frequency estimate.
    ],
  )

  === Post-training rate and accuracy against gamma frequency

  Individual network rates spanned #er_lo–#er_hi Hz and accuracies
  #acc_lo–#acc_hi% across the six inhibitory-decay conditions.

  #figure(
    data-image(data-file("exp041/rate_vs_fgamma.svg"),
      width: 100%,
      alt: "Post-training E rate against gamma frequency; points lie on the affine fit line.",
    ),
    caption: [
      Final-epoch excitatory rate (top) and test accuracy (bottom) against gamma
      frequency. Each point is a mean over three seeds; error bars show ±1
      standard error. The affine line fits six condition means.
    ],
  )

  === Affine and origin-constrained rate–frequency fits

  #figure(
    table(
      columns: 4,
      [fit], [intercept (Hz)], [slope (Hz/Hz)], [$R_"fit"^2$],
      [affine], [#fa], [#fp], [#fr],
      [through origin], [0], [#po], [#pr],
    ),
    caption: [Both least-squares fits used the same six condition means and
      centred total sum of squares. The origin-constrained fit tests how much
      the association depends on a free intercept.],
  )

  ]

  == Methods

  I compared final-epoch dynamics across matched networks trained at different
  inhibitory decay times, keeping the evaluation data fixed.

  + *Reuse matched trained networks.* Six inhibitory GABA decay constants,
    $tau_"GABA" in {4.5, 6, 9, 12, 18, 27}$ ms, and seeds 42–44 defined 18 PING
    networks. Training used 6,300 MNIST training images and 700 held-out
    validation images, 50 epochs of AdamW with zero weight decay, learning rate
    $4 times 10^(-4)$, batches of 256, a time-averaged membrane-potential readout
    and no spike budget. Only inhibitory decay varied between conditions;
    simulation used 0.1 ms timesteps and 200 ms trials. I reused final-epoch
    weights to measure endpoint dynamics, without retraining or selecting
    weights by test performance.

  + *Measure fixed-trial responses.* Each network received the same fixed
    subset of #cfg.evaluation_samples images from the official MNIST test partition.
    I measured classification accuracy, mean excitatory spikes per neuron per
    second, and each trial's population-E trace over the full 200 ms. The
    illustrative raster used image index 0 and seed 42; a fixed random seed of
    0 selected displayed neurons without replacement.

  + *Estimate rhythm frequency.* I demeaned each trial's trace and used a
    Welch density estimate with one full-trial Hann window #cite(1), then
    averaged PSDs across trials. The largest peak between 5 and 150 Hz defined
    the candidate gamma frequency; its neighbouring linear-power values gave

    $ f_"peak" = f_"peak,bin" + 1/2 (y_0 - y_2)/(y_0 - 2 y_1 + y_2) dot Delta f_"bin". $

    Here $f_"peak"$ is the interpolated spectral-peak frequency, $f_"peak,bin"$ the peak-bin frequency,
    and $Delta f_"bin" = 5$ Hz the bin spacing; $y_0$, $y_1$ and $y_2$ are PSD values
    immediately below, at and above that bin. Because the peak search is restricted to the defined gamma band, the reported estimator is $f_gamma eq.triple f_"peak"$. I clamped the correction to
    half a bin, using zero offset for zero curvature or a spectrum endpoint.
    Interpolation reduces bin quantisation but can remain biased #cite(2).
    Per-trial peak distributions were diagnostics; their medians did not enter
    the fit.

  + *Fit the rate–frequency relation.* I averaged each network's frequency
    and excitatory rate over the three seeds, then fitted the six condition
    points with equal weight by least squares:

    $ r_E = a + beta_(r f) dot f_gamma, quad r_E = beta_(r f,0) dot f_gamma. $

    Here $r_E$ is mean excitatory firing rate in hertz, $a$ is the affine
    intercept in hertz, and $beta_(r f)$ and $beta_(r f,0)$ are dimensionless fitted slopes. Both fits
    report $R_"fit"^2$, the coefficient of determination using centred total sum of
    squares; error bars are sample standard deviations divided by $sqrt(3)$.

  #run-view("exp041", inputs)

  == Appendix: Cycle-participation model

  If a participation fraction $p_"part"$ of excitatory neurons emits exactly one spike during each
  cycle of duration $1 / f_gamma$, its cyclic per-neuron rate is $p_"part" dot f_gamma$.
  Adding a frequency-independent contribution $a$ gives

  $ r_E = underbrace(a, "non-rhythmic contribution") +
    underbrace(p_"part" dot f_gamma, "cyclic contribution"). $

  This is a proposed interpretation of the affine form, conditional on stable
  participation. Neurons nearest threshold when inhibition drops could contribute
  one spike while others recover; long inhibitory decay could instead sustain
  tonic inhibition and leave a feedforward contribution. Neither mechanism is
  established by the fit alone. In particular, an extrapolated intercept need
  not be a physical baseline, and a negative intercept cannot represent a
  nonnegative background firing rate.

  #reference-list((
    (text: [P. D. Welch. “The use of the fast Fourier transform for the estimation
      of power spectra: A method based on time averaging over short, modified
      periodograms.” _IEEE Transactions on Audio and Electroacoustics_ 15(2),
      70–73 (1967).], doi: "10.1109/TAU.1967.1161901"),
    (text: [J. O. Smith III. _Spectral Audio Signal Processing._ W3K Publishing
      (2011), #link("https://www.dsprelated.com/freebooks/sasp/Quadratic_Interpolation_Spectral_Peaks.html")[“Quadratic Interpolation of Spectral Peaks”]
      and #link("https://www.dsprelated.com/freebooks/sasp/Bias_Parabolic_Peak_Interpolation.html")[“Bias of Parabolic Peak Interpolation.”]],),
  ))
]
#body
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [How does inhibitory decay affect gamma frequency, excitatory firing rate, and accuracy? Compare matched trained networks across inhibitory timescales.],
    preview-figures, json-inputs: ("exp041",),
  )
}

#let meta = meta + (assets: input-assets("exp041", inputs))
#let body = with-datasets("exp041", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-contents(body)
