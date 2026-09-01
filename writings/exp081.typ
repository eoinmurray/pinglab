#import "contents.typ": with-contents, with-numbered-equations, with-result-sections
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#import "/.demolab/lib.typ": cite, reference-list
#let data-file = data-file.with(article: "exp081")

#let meta = (
  status: "[▦ DATA | v31.2.0]",
  title: "How Pixel Features Respond to Input Rate",
  created_at: "2026-08-10T00:00:00Z",
  updated_at: "2026-08-31T15:55:29Z",
  description: "Standalone empirical and analytical study of synaptic, membrane, and finite-window filtering under sparse Poisson drive.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp081",)
#let preview-figures = (
  (path: "exp081/empirical_moments.svg", label: "empirical moments"),
  (path: "exp081/response_distributions.svg", label: "response distributions"),
  (path: "exp081/frequency_response.svg", label: "frequency response"),
  (path: "exp081/analytical_empirical.svg", label: "analytical empirical"),
)

#let result-card-style = context {
  if target() == "html" {
    html.elem("style",
      ".pinglab-result-card { margin: 1.25rem 0; padding: 1.2rem 1.35rem 1.3rem; border: 1px solid var(--rule-strong); border-radius: 3px; background: var(--paper); } "
      + ".pinglab-result-card > h4:first-child { margin-top: 0; } "
      + ".pinglab-result-card > :last-child { margin-bottom: 0; } "
      + "math[display='block'] { display: block; max-width: 100%; overflow-x: auto; overflow-y: hidden; } "
      + "@media (max-width: 520px) { .pinglab-result-card { margin: 1rem 0; padding: .95rem 1rem 1.05rem; } }",
    )
  }
}

#let result-card(body) = context {
  if target() == "html" {
    html.elem("article", attrs: (class: "pinglab-result-card"), body)
  } else { body }
}

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let r = data-json(data-file("exp081/numbers.json"))
#let p = r.parameters
#let rounded(x, digits: 3) = if x == none { "not defined" } else { str(calc.round(x, digits: digits)) }
#let body = [
  == Abstract

  We asked how a fully active pixel's input spike rate becomes the finite-window
  voltage feature used by downstream classifiers, and whether stationary filter
  theory predicts that transformation. We compared direct conductance-driven
  membrane simulations with a local stationary linearization of the same system.

  The analytical model captured the filtering structure and broad mean-response
  shape, but overestimated response magnitude and did not reproduce the empirical
  variability. Sparse finite-window features remained governed by discrete event
  counts and timing, so quantitative predictions still require direct simulation.

  == Results

  #with-result-sections[

  #result-card-style

  #result-card[
  === Finite-window feature mean and variability across input rates

  Each presentation began from rest and converted one fully active pixel into a
  finite-window mean depolarization while input rate and event strength varied.

  Higher rate should increase mean depolarization, while discrete event counts,
  event timing and nonlinear conductance can make the sample SD non-monotonic.

  Mean response rose overall with input rate and event strength. The sample SD
  rose rapidly at sparse rates, then flattened or declined for the two smaller
  event increments while remaining high for the largest. Post hoc, this pattern
  is consistent with reduced relative count fluctuations, shunting and voltage
  saturation; those mechanisms were not manipulated separately.

  #figure(
    data-image(data-file("exp081/empirical_moments.svg"), width: 100%,
      alt: "Two panels show empirical mean feature and feature standard deviation against input rate for three conductance increments."),
    caption: [Empirical finite-window mean (A) and sample SD (B), in mV, across
      #p.input_rate_grid_hz.len() input rates. Each condition summarizes
      #p.moment_draws independent presentations. Black, red and cyan denote
      #p.probes_uS.map(str).join(", ") μS event increments; curves are summaries,
      not individual trials or confidence intervals.],
  )
  ]

  #result-card[
  === Feature distributions across sparse and dense input rates

  These probes held presentation duration and event strength fixed while input
  rate changed from sparse to dense drive.

  Input rate fixes only the expected event count. Individual presentations can
  contain no events, one event or several events, and equal counts can produce
  different finite-window averages when their event times differ.

  The empirical distribution was dominated by near-zero responses at the
  sparsest rate, became a mixed sparse distribution at the intermediate rate,
  and formed a broad, nearly continuous distribution at the densest rate.

  #figure(
    data-image(data-file("exp081/response_distributions.svg"), width: 100%,
      alt: "Three logarithmic histograms show feature distributions becoming smoother as input rate increases."),
    caption: [Feature distributions at #p.distribution_rates_hz.map(str).join(", ") spikes/s
      and #p.nominal_probe_uS μS, with #p.distribution_draws independent
      presentations per condition. Bars show probability per common fixed-width
      bin on a logarithmic scale; feature values are in mV.],
  )
  ]

  #result-card[
  === Predicted filtering before and after finite-window averaging

  This local linearization describes a theoretical response around three
  stationary operating rates; it is not a measured modulation experiment.

  The model predicts a flat low-frequency response followed by attenuation when
  modulation outpaces the synaptic and membrane timescales. Greater drive lowers
  the low-frequency plateau through shunting and reduced excitatory driving
  force, while finite-window averaging adds regularly spaced zeros and lobes.

  #figure(
    data-image(data-file("exp081/frequency_response.svg"), width: 100%,
      alt: "Two panels show the analytical synapse and membrane frequency response before and after finite-window averaging."),
    caption: [Analytical frequency responses, not measured modulation responses, at
      #p.nominal_probe_uS μS. Black, red and cyan denote operating rates
      #p.frequency_response_rates_hz.map(str).join(", ") spikes/s. Panel A shows
      synapse-plus-membrane gain; B adds #p.presentation_ms ms averaging. Gain is
      in dB relative to the lowest-drive DC response; frequency is in Hz.],
  )
  ]

  #result-card[
  === Analytical and empirical feature moments across input rates

  The stationary calculation replaces random conductance paths with one mean
  operating point and treats the remaining input as a small linear fluctuation.

  If that approximation were quantitatively adequate, analytical and empirical
  moments should retain both their rate dependence and magnitude.

  The mean prediction retained the broad curvature and conductance ordering
  (Pearson correlation #rounded(r.comparison.mean.pearson_r)) but exceeded
  empirical magnitude by a median factor of
  #rounded(r.comparison.mean.median_predicted_empirical_ratio). SD correspondence
  was weak overall (Pearson correlation
  #rounded(r.comparison.standard_deviation.pearson_r)), especially for the largest
  event increment. The approximation therefore describes qualitative filtering
  structure, not finite-window variability.

  #figure(
    data-image(data-file("exp081/analytical_empirical.svg"), width: 100%,
      alt: "Two panels compare analytical curves with empirical points for feature mean and standard deviation over input rate."),
    caption: [Stationary predictions (solid curves) and independently simulated estimates
      (points) of mean feature (A) and sample SD (B), in mV. Each estimate uses
      #p.moment_draws presentations per rate. Black, red and cyan denote
      #p.probes_uS.map(str).join(", ") μS event increments.],
  )
  ]

  ]

  == Methods

  === Compute

  + *Conditions and repetitions.* We fixed pixel intensity at one and varied
    input rate from #p.input_rate_grid_hz.first() to
    #p.input_rate_grid_hz.last() spikes/s across
    #p.input_rate_grid_hz.len() conditions and event strength across
    #p.probes_uS.map(str).join(", ") μS. Each condition used
    #p.moment_draws independent encoding draws of a #p.presentation_ms ms
    presentation; separate distribution probes used #p.distribution_draws draws
    at #p.distribution_rates_hz.map(str).join(", ") spikes/s and
    #p.nominal_probe_uS μS.

  + *Generate independent events.* At simulation step $k$, a fully active pixel
    generated a Bernoulli event with probability
    $p_"event"=r_"input" Delta t_"sim"/1000$, where $r_"input"$ is input rate in
    spikes/s and $Delta t_"sim"=#p.dt_ms$ ms. Physical time was
    $t_k=k Delta t_"sim"$ and $N_t=T_"present"/Delta t_"sim"$ steps formed one
    presentation. Separate deterministic random streams supplied the moment and
    distribution draws; every draw began with zero conductance and resting
    voltage.

  + *Integrate conductance and voltage.* AMPA conductance decayed with time
    constant $tau_"AMPA"=#p.membrane.tau_ampa_ms$ ms before each event added its
    conductance increment. Membrane voltage obeyed
    $ C_m (d V_m)/(d t)=g_L (E_L-V_m)+g(t)(E_e-V_m). $
    Here $t$ is physical time in ms, $V_m$ is membrane voltage in mV, $g$ is
    excitatory conductance in μS, $C_m=#p.membrane.C_m_nF$ nF is capacitance,
    $g_L=#p.membrane.g_L_uS$ μS is leak conductance, and
    $E_L=#p.membrane.E_L_mV$ mV and $E_e=#p.membrane.E_e_mV$ mV are reversal
    potentials. Each single-precision voltage step used the exact exponential
    solution with updated conductance held fixed for that step.

  + *Record the finite-window feature.* For each encoding draw we approximated
    $ z_"feature"=1/T_"present" integral_0^(T_"present") (V_m(t)-E_L) dif t.
 $
    Here $z_"feature"$ is baseline-subtracted mean voltage in mV and
    $T_"present"=#p.presentation_ms$ ms is presentation duration. The recorded
    discrete estimate averaged all $N_t$ post-update voltages after subtracting
    rest.

  === Analyse

  #set enum(start: 5)

  + *Estimate empirical moments and distributions.* We calculated the arithmetic
    mean and sample SD across encoding draws for every rate–strength condition,
    using one fewer than the number of draws as the variance denominator. For
    the distribution probes, #p.histogram_bins common linear bins spanned zero
    to the maximum response rounded upward to 5 mV, and counts were divided by
    the number of draws.

  + *Calculate the stationary prediction.* We evaluated mean conductance and
    equilibrium voltage at each condition, then linearized the synaptic and
    membrane responses around that operating point. Rectangular-window averaging
    gave $H_r(omega)$, the input-to-feature transfer function at angular
    frequency $omega$ in rad/ms, with predicted variance
    $ "Var"(z_"feature")=1/(2 pi) integral_(-oo)^oo abs(H_r(omega))^2
      (r_"input"/1000) dif omega. $
    #link(<sec-appendix-model-specification-and-calculations>)[Appendix: model specification and calculations] specifies this approximation, and
    #link(<sec-appendix-derivation-of-the-analytical-filter>)[Appendix: derivation of the analytical filter] derives it.

  + *Integrate and compare.* We integrated Equation 3 by the trapezoidal rule on
    a logarithmic frequency grid, added the low-frequency DC tail and checked a
    coarser grid. Mean and SD comparisons used Pearson correlation, mean absolute
    error and median predicted-to-empirical ratio; joint zeros were excluded,
    ratios required positive empirical values, and an undefined correlation was
    retained as undefined. Analytical frequency-response magnitudes were
    normalized to the lowest-drive DC gain.

  === Present

  #set enum(start: 8)

  + *Expose direct-simulation evidence.* We displayed the recorded empirical
    means and sample SDs across all rate–strength conditions, plus probability
    per common bin for the three distribution probes. These figures reused the
    retained analysis without generating new encoding draws or confidence
    intervals.

  + *Expose analytical evidence.* We displayed the calculated local frequency
    responses before and after window averaging and overlaid analytical and
    empirical moments for the same conditions. The frequency-response figure is
    a theoretical mechanism, not a measured modulation response; presentation
    did not recompute the estimators or rerun simulation.

  #run-view("exp081", inputs)

  == Appendix: model specification and calculations <sec-appendix-model-specification-and-calculations>

  === Simulate the finite-window feature

  We fixed the normalized pixel intensity at $x=1$ and varied only its input
  rate $r_"input"$ from #p.input_rate_grid_hz.first() to
  #p.input_rate_grid_hz.last() spikes/s. At each timestep, the pixel generated
  an event with probability

  $ p_"event" = (r_"input" Delta t_"sim") / 1000. quad "(A1)" $

  Here $r_"input"$ is the input rate in spikes/s and $Delta t_"sim"=#p.dt_ms$ ms. During a
  presentation of duration $T_"present"=#p.presentation_ms$ ms, the expected number of
  events is $r_"input" T_"present" / 1000$. Expected event count is therefore a consequence of the
  chosen rate, not a separate experimental variable.

  Conductance and membrane voltage followed

  $ g[k] = beta_"AMPA" g[k-1] + w_"event" s[k], quad "(A2)" $

  $ beta_"AMPA" = exp(-Delta t_"sim"/tau_"AMPA"), quad "(A3)" $

  $
    C_m (d V_m)/(d t) = g_L (E_L-V_m) + g(t)(E_e-V_m). quad "(A4)"
  $

  Here $s[k]$ is one when an event occurs and zero otherwise, $g[k]$ is AMPA
  conductance, $w_"event"$ is the conductance increment per event, $beta_"AMPA"$ is the
  per-timestep decay factor, and $tau_"AMPA"$ is the AMPA decay time constant.
  The membrane voltage is $V_m(t)$, $C_m$ is membrane capacitance, $g_L$ is leak
  conductance, $E_L$ is the leak reversal potential, and $E_e$ is the
  excitatory reversal potential.

  We used $C_m=1$ nF, $g_L=0.05$ μS, $E_L=-65$ mV, $E_e=0$ mV,
  $tau_"AMPA"=2$ ms, and independent probe conductances
  $w_"event" in {#p.probes_uS.map(str).join(", ")}$ μS. Every presentation began from
  $g(0)=0$ and $V_m(0)=E_L$. Its continuous-time scalar feature was

  $ z_"feature" = 1/T_"present" integral_0^(T_"present") (V_m(t)-E_L) dif t. quad "(A5)" $

  Here $z_"feature"$ is the baseline-subtracted mean voltage and $T_"present"$ is presentation
  duration. Simulation used the arithmetic mean of post-update voltages as the
  discrete approximation to this integral.

  For the mean and SD estimates in Figure 1, we generated #p.moment_draws new
  presentations at each of #p.input_rate_grid_hz.len() input rates and each
  conductance. For the full response distributions in Figure 2, we generated
  #p.distribution_draws new presentations at each selected rate. Direct
  simulation retains the nonstationary finite-window behaviour of filtered
  conductance shot noise#cite(1).


  === Calculate the stationary operating point

  For stationary Poisson rate $r=r_"input"$, filtered-shot-noise theory#cite(2) gives the
  mean AMPA conductance

  $ macron(g)_r = r w_"event" tau_"AMPA" / 1000. quad "(A6)" $

  Replacing the fluctuating conductance by this mean gives the operating-point
  voltage

  $
    macron(v)_r =
    (g_L E_L + macron(g)_r E_e) /
    (g_L + macron(g)_r). quad "(A7)"
  $

  Because this operating point is constant across the window, its predicted
  mean feature is

  $ mu_"linear" (z) = macron(v)_r-E_L. quad "(A8)" $

  An overbar denotes an operating-point quantity, $macron(g)_r$ is mean conductance at
  rate $r$, $macron(v)_r$ is the voltage evaluated at mean conductance, and
  $mu_"linear" (z)$ is the resulting linear prediction for mean feature value.
  #link(<sec-stationary-conductance-and-voltage>)[Stationary conductance and voltage] derives Equations A6–A8 from the continuous conductance and
  membrane equations.


  === Derive the local frequency response

  Linearizing Equations A2–A4 in their continuous-time limit around the operating point gives the response from
  input-rate fluctuation to voltage fluctuation,

  $
    G_r (omega) =
    w_"event"/(i omega + 1/tau_"AMPA") dot
    (E_e-macron(v)_r)/
    (i omega C_m + g_L + macron(g)_r). quad "(A9)"
  $

  Here $omega$ is angular frequency in rad/ms and $i$ is the imaginary unit.
  The transfer function $G_r (omega)$ maps a small input-rate fluctuation to its
  voltage response around the operating point. #link(<sec-synapse-plus-membrane-linearization>)[Synapse-plus-membrane linearization] derives Equation A9
  by linearizing the synaptic and membrane equations.

  Averaging voltage over the finite presentation contributes

  $ A_(T_"present") (omega) =
    (1-exp(-i omega T_"present"))/(i omega T_"present"). quad "(A10)" $

  so the complete input-to-feature response is

  $ H_r (omega) = A_(T_"present") (omega) G_r (omega). quad "(A11)" $

  Here $A_(T_"present") (omega)$ is the transfer function of averaging over
  presentation duration $T_"present"$, and $H_r (omega)$ is the complete
  input-to-feature transfer function. #link(<sec-finite-window-average>)[Finite-window average]
  derives Equations A10 and A11.

  To generate Figure 3, we evaluated Equations A9 and A11 from
  #p.frequency_plot_bounds_hz.first() to #p.frequency_plot_bounds_hz.last() Hz at
  #p.frequency_response_rates_hz.map(str).join(", ") spikes/s for the nominal
  #p.nominal_probe_uS μS probe. Frequency in Hz was converted using
  $omega=2 pi f/1000$. The Bode magnitude was reported relative to the
  low-drive DC response,

  $
    M_X (f) = 20 log_10 (abs(X_r (2 pi f/1000))/
    abs(X_(r_"low") (0))). quad "(A12)"
  $

  Here $M_X (f)$ is Bode magnitude in dB, $X_r=G_r$ for Figure 3A,
  $X_r=H_r$ for Figure 3B, $f$ is modulation frequency in Hz, and $r_"low"$ is
  the lowest operating rate used as the DC reference.


  === Calculate the linearized feature variance

  The centred ideal Poisson input has a white two-sided spectrum on the
  millisecond time base,

  $ S_"in" (omega) = r_"input"/1000. quad "(A13)" $

  The feature spectrum and variance are therefore

  $ S_z (omega)=abs(H_r (omega))^2 S_"in" (omega), quad "(A14)" $

  $
    "Var"_"linear" (z) = 1/(2 pi) integral_(-oo)^oo
    abs(H_r (omega))^2 S_"in" (omega) dif omega. quad "(A15)"
  $

  Here $S_"in" (omega)$ is the two-sided input power spectrum, $S_z (omega)$ is
  the feature power spectrum, and $"Var"_"linear"(z)$ is the predicted feature
  variance. #link(<sec-poisson-spectrum-and-feature-variance>)[Poisson spectrum and feature variance] derives Equations A13–A15 from the Poisson spectrum and
  the linear-filter variance identity.

  We integrated Equation A15 numerically on a logarithmic frequency grid. A
  second calculation with half as many points measured quadrature refinement.


  === Explain the stationary approximation's failure <sec-stationary-approximation-failure>

  The stationary calculation evaluates the response at mean conductance,
  $z(E[g])$, whereas simulation estimates the response over random conductance
  paths, $E[z(g)]$. Saturation makes these unequal, approximately

  $ E[z(g)] < z(E[g]). $

  The stationary input also acts throughout the window; a real late event
  contributes only briefly to Equation A5.

  The empirical distribution is a spike-count and spike-time mixture,

  $
    p_Z (z) = sum_(n=0)^oo P(N=n) p_(Z|N) (z|n). quad "(D2)"
  $

  Here $Z$ is the random feature value, $N$ is the event count, $P(N=n)$ is the
  probability of observing $n$ events, and $p_(Z|N) (z|n)$ is the feature
  distribution conditional on that count.

  Equation A15 instead treats input as a small stationary fluctuation around one
  operating point. At low rates, zero-event trials remain exactly at rest, one
  large event produces a timing-dependent response, and multiple events interact
  through shunting and saturation. That non-Gaussian mixture explains why the
  analytical SD rises too sharply and peaks too early, especially for the
  largest conductance increment.


  == Appendix: derivation of the analytical filter <sec-appendix-derivation-of-the-analytical-filter>

  === Stationary conductance and voltage <sec-stationary-conductance-and-voltage>

  The discrete indicator $s[k]$ records whether a simulation step contains an
  event. Its continuous-time counterpart is the impulse train
  $s(t)=sum_j delta(t-t_j)$, where $j$ identifies events rather than simulation
  steps. The continuous AMPA equation is

  $ (d g)/(d t) = -g/tau_"AMPA" + w_"event" s(t). quad "(B1)" $

  Here $t_j$ is the time of event $j$, and $delta(t-t_j)$ is a Dirac impulse at
  that time.

  For stationary input rate $r$ spikes/s, the rate on the millisecond time base
  is $r/1000$. Taking expectations of Equation B1 and setting the stationary
  derivative to zero gives

  $
    0=-macron(g)_r/tau_"AMPA"+w_"event" r/1000,
  $

  hence Equation A6. The stationary membrane equation is

  $
    0=g_L (E_L-macron(v)_r)
      +macron(g)_r (E_e-macron(v)_r), quad "(B2)"
  $

  and collecting the terms multiplying $macron(v)_r$ gives Equation A7.

  === Synapse-plus-membrane linearization <sec-synapse-plus-membrane-linearization>

  Decompose each quantity into its operating point and a small fluctuation,

  $
    g=macron(g)_r+delta g,
    quad v=macron(v)_r+delta v,
    quad s=r/1000+delta s.
  $

  Substitution into Equation B1 and cancellation of the stationary terms gives

  $ (d delta g)/(d t) = -delta g/tau_"AMPA" + w_"event" delta s. quad "(B3)" $

  With Fourier convention $(d)/(d t) -> i omega$,

  $
    delta g(omega)=w_"event"/(i omega+1/tau_"AMPA") delta s(omega). quad "(B4)"
  $

  Substitute the decompositions into Equation A4. After cancelling Equation B2
  and discarding the second-order product $delta g delta v$,

  $
    C_m (d delta v)/(d t)
    =-(g_L+macron(g)_r)delta v
      +(E_e-macron(v)_r)delta g. quad "(B5)"
  $

  Fourier transformation and substitution of Equation B4 yields Equation A9.
  Its two denominator factors are the AMPA and membrane low-pass terms; the
  numerator contains the conductance increment and local excitatory driving
  force.

  === Finite-window average <sec-finite-window-average>

  Linearizing Equation A5 leaves the average of the voltage fluctuation,

  $ delta z_"feature" = 1/T_"present" integral_0^(T_"present") delta v(t) dif t.
    quad "(B6)" $

  The averaging kernel is $a_(T_"present") (t)=1/T_"present"$ for
  $0 <= t <= T_"present"$ and zero otherwise. Its Fourier transform is

  $
    A_(T_"present") (omega)=1/T_"present" integral_0^(T_"present") exp(-i omega t) dif t
    =(1-exp(-i omega T_"present"))/(i omega T_"present"),
  $

  proving Equation A10. Because averaging follows the membrane response, the
  transfer functions multiply, proving Equation A11. Using
  $1-exp(-i x)=2i exp(-i x/2)sin(x/2)$ gives

  $
    abs(A_(T_"present") (omega))=
    abs(sin(omega T_"present"/2)/(omega T_"present"/2)),
  $

  With $T_s=T_"present"/1000$ s and $omega=2 pi f/1000$,

  $
    abs(A_(T_"present") (2 pi f/1000)) =
    abs(sin(pi f T_s)/(pi f T_s)). quad "(D1)"
  $

  Its zeros at $f=n/T_s$ cancel modulation containing an integer number of
  cycles within the presentation. Here $n$ is any nonzero integer. The absolute
  sinc response creates the lobes, while the synapse-plus-membrane low-pass
  response supplies their falling envelope.

  === Poisson spectrum and feature variance <sec-poisson-spectrum-and-feature-variance>

  For ideal Poisson events with rate $r_"input"/1000$ per ms, disjoint intervals have
  independent counts and count variance equals count mean. The centred impulse
  train therefore has autocovariance

  $ C_(delta s) (ell)=(r_"input"/1000)delta(ell). quad "(B7)" $

  Here $C_(delta s) (ell)$ is the input autocovariance at lag $ell$, and
  $delta(ell)$ is the Dirac delta function.

  Fourier transformation of the delta function gives the constant spectrum in
  Equation A13. A linear time-invariant filter multiplies the input spectrum by
  squared transfer magnitude, proving Equation A14. Finally, inverse Fourier
  transformation at zero lag gives

  $
    "Var"(z)=R_z (0)=1/(2 pi) integral_(-oo)^oo S_z (omega) dif omega,
  $

  which, after substituting Equation A14, proves Equation A15. The result is exact
  for the stated stationary linearized model but only approximate for the
  nonlinear, start-from-rest, finite-event simulation.

  #reference-list((
    (
      text: [Marco Brigham and Alain Destexhe: _Nonstationary Filtered Shot-Noise Processes and Applications to Neuronal Membranes_. Physical Review E, 2015.],
      doi: "10.1103/PhysRevE.91.062102",
    ),
    (
      text: [Lars Wolff and Benjamin Lindner: _Mean, Variance, and Autocorrelation of Subthreshold Potential Fluctuations Driven by Filtered Conductance Shot Noise_. Neural Computation, 2010.],
      doi: "10.1162/neco.2009.02-09-958",
    ),
  ))
]
#body
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [How do pixel-derived features respond to input rate? Compare predicted response statistics with measured moments, distributions, and frequency responses.],
    preview-figures, json-inputs: ("exp081",),
  )
}

#let meta = meta + (assets: input-assets("exp081", inputs))
#let body = with-datasets("exp081", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-numbered-equations(body)
#let body = with-contents(body)
