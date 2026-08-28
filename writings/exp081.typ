#import "contents.typ": with-contents
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#import "/.demolab/lib.typ": cite, reference-list
#let data-file = data-file.with(article: "exp081")

#let meta = (
  status: "[▦ DATA]",
  title: "How Pixel Features Respond to Input Rate",
  date: "2026-08-10",
  updated_at: "2026-08-27",
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

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let r = data-json(data-file("exp081/numbers.json"))
#let p = r.parameters
#let rounded(x, digits: 3) = if x == none { "not defined" } else { str(calc.round(x, digits: digits)) }
#let body = [
  == Abstract

  Stationary linear-filter theory explains the filtering of a sparse
  conductance-driven pixel feature, but does not accurately predict its
  finite-window magnitude. We simulated an AMPA synapse, a conductance-based
  membrane and #p.presentation_ms ms averaging at
  #p.input_rate_grid_hz.len() rates from #p.input_rate_grid_hz.first() to
  #p.input_rate_grid_hz.last() Hz, with #p.probes_uS.len() event strengths and
  #p.moment_draws presentations per condition. The median predicted-to-empirical
  mean ratio was #rounded(r.comparison.mean.median_predicted_empirical_ratio).
  The theory explains slow-response plateaus, high-frequency attenuation and
  averaging lobes, but its standard deviation peaks too early. Sparse responses
  depend on discrete spike counts and timing, outside the small stationary
  fluctuation approximation; direct simulation remains necessary for quantitative
  predictions in this regime.

  #run-view("exp081", inputs)

  == Results

  === Empirical finite-window response

  #figure(
    data-image(data-file("exp081/empirical_moments.svg"), width: 100%,
      alt: "Two panels show empirical mean feature and feature standard deviation against input rate for three conductance increments."),
    caption: [Empirical finite-window mean (A) and sample SD (B), in mV, across
      #p.input_rate_grid_hz.len() input rates. Each condition summarizes
      #p.moment_draws independent presentations. Black, red and cyan denote
      #p.probes_uS.map(str).join(", ") μS event increments; curves are summaries,
      not individual trials or confidence intervals.],
  )

  #figure(
    data-image(data-file("exp081/response_distributions.svg"), width: 100%,
      alt: "Three logarithmic histograms show feature distributions becoming smoother as input rate increases."),
    caption: [Feature distributions at #p.distribution_rates_hz.map(str).join(", ") spikes/s
      and #p.nominal_probe_uS μS, with #p.distribution_draws independent
      presentations per condition. Bars show probability per common fixed-width
      bin on a logarithmic scale; feature values are in mV.],
  )

  === Analytical frequency response

  #figure(
    data-image(data-file("exp081/frequency_response.svg"), width: 100%,
      alt: "Two panels show the analytical synapse and membrane frequency response before and after finite-window averaging."),
    caption: [Analytical frequency responses, not measured modulation responses, at
      #p.nominal_probe_uS μS. Black, red and cyan denote operating rates
      #p.frequency_response_rates_hz.map(str).join(", ") spikes/s. Panel A shows
      synapse-plus-membrane gain; B adds #p.presentation_ms ms averaging. Gain is
      in dB relative to the lowest-drive DC response; frequency is in Hz.],
  )

  === Analytical and empirical moments

  #figure(
    data-image(data-file("exp081/analytical_empirical.svg"), width: 100%,
      alt: "Two panels compare analytical curves with empirical points for feature mean and standard deviation over input rate."),
    caption: [Stationary predictions (solid curves) and independently simulated estimates
      (points) of mean feature (A) and sample SD (B), in mV. Each estimate uses
      #p.moment_draws presentations per rate. Black, red and cyan denote
      #p.probes_uS.map(str).join(", ") μS event increments.],
  )

  == Methods

  We compared independent finite-window simulations with a stationary local
  linearization of the same conductance-driven passive membrane.

  #enum(
    [*Generate independent events.* At each timestep, a fully active pixel
    generated a Bernoulli event with probability $r Delta t/1000$, where $r$ is
    input rate in spikes/s and $Delta t=#p.dt_ms$ ms. Separate deterministic random
    streams, derived from seed #p.seed, supplied moment and distribution probes;
    each presentation started with zero conductance and resting voltage.],

    [*Update synapse and membrane.* Conductance decayed exponentially with a
    #p.membrane.tau_ampa_ms ms AMPA time constant before adding the event increment.
    Voltage followed
    $ C (d v)/(d t)=g_L (E_L-v)+g(t)(E_e-v). quad "(1)" $
    Here $t$ is time in ms, $v$ is membrane voltage in mV, $g$ is excitatory conductance in μS,
    $C=#p.membrane.C_m_nF$ nF is capacitance, $g_L=#p.membrane.g_L_uS$ μS is leak
    conductance, and $E_L=#p.membrane.E_L_mV$ mV and $E_e=#p.membrane.E_e_mV$ mV
    are reversal potentials. Each voltage step used the exact exponential solution
    with the updated conductance held fixed; arithmetic used single precision.],

    [*Measure the finite-window feature.* We approximated
    $ z=1/T integral_0^T (v(t)-E_L) dif t. quad "(2)" $
    Here $z$ is mean depolarization in mV and $T=#p.presentation_ms$ ms is the
    presentation duration. The discrete estimate averaged every post-update voltage
    after subtracting rest; individual estimates were retained for reuse.

    At each rate and conductance we
    calculated the arithmetic mean and sample SD, using the number of presentations
    minus one as the variance denominator. Distribution probes used
    #p.histogram_bins common linear bins, from zero to the maximum response rounded
    upward to a multiple of 5 mV; counts were divided by presentation count.],

    [*Calculate the stationary prediction.* Mean conductance was rate times event
    increment times synaptic decay time, with rates converted to events/ms. We
    evaluated equilibrium voltage at that mean and linearized the synaptic and
    membrane responses about it. Multiplication by the rectangular averaging
    response gave $H_r (omega)$, the input-to-feature transfer function at angular
    frequency $omega$ in rad/ms. Its predicted variance was
    $ "Var"(z)=1/(2 pi) integral_(-oo)^oo abs(H_r (omega))^2 (r/1000) dif omega.
      quad "(3)" $
    #link(<sec-appendix-model-specification-and-calculations>)[Appendix: model specification and calculations] specifies the transfer and #link(<sec-appendix-derivation-of-the-analytical-filter>)[Appendix: derivation of the analytical filter] derives it.],

    [*Integrate and compare.* We used trapezoidal integration over a logarithmic
    frequency grid, added the low-frequency DC tail, and checked a coarser grid.
    Mean and SD comparisons used Pearson correlation, mean absolute error and
    median predicted-to-empirical ratio, excluding joint zeros and requiring
    positive empirical values for ratios. Frequency-response magnitudes were
    expressed relative to the lowest-drive DC gain.],
  )

  == Discussion

  === Interpreting the empirical finite-window response

  Mean response increased with input rate and event strength. Variability first
  increased as spike count and timing diversified, then flattened or declined
  as relative count fluctuations, shunting, and voltage saturation became more
  important.



  Input rate fixes only the average number of events. Individual presentations
  can still contain no events, one event, or several events. Even when two
  presentations contain the same number, different event times change the
  finite-window average.

  === Analytical Bode-magnitude response

  The synapse and membrane passed slow modulation but attenuated fast
  modulation. Finite-window averaging superimposed regularly spaced zeros and
  lobes on that falling response.


  In Panel A, each curve is flat at low frequency because the AMPA conductance
  and membrane voltage can follow a slowly varying input quasi-statically.
  Greater drive lowers that plateau by shunting the membrane and reducing the
  excitatory driving force. The curves bend downward once modulation becomes
  too fast for the synaptic and membrane timescales to follow.

  Panel B is the same synapse-plus-membrane response multiplied by the
  rectangular-window response. With $T_s=T/1000$ s,

  $
    abs(A_T (2 pi f/1000)) =
    abs(sin(pi f T_s)/(pi f T_s)). quad "(D1)"
  $

  Its zeros at $f=n/T_s$ cancel modulation containing an integer number of
  cycles within the presentation. Here $n$ is any nonzero integer. The absolute
  sinc response creates the lobes, while the Panel A low-pass response supplies
  their falling envelope. #link(<sec-finite-window-average>)[Finite-window average] derives Equation D1 from the
  finite-window averaging kernel.

  === Interpreting analytical and empirical moments

  The stationary model reproduced the mean curve's broad shape but not its
  magnitude, and it failed to reproduce the shape of the empirical standard
  deviation.


  The analytical mean preserved the broad curvature and conductance ordering but
  exceeded the empirical magnitude. Its median predicted-to-empirical ratio was
  #rounded(r.comparison.mean.median_predicted_empirical_ratio). The stationary
  calculation evaluates the response at mean conductance, $z(E[g])$, whereas
  simulation estimates the response over random conductance paths, $E[z(g)]$.
  Saturation makes these unequal, approximately

  $ E[z(g)] < z(E[g]). $

  The stationary input also acts throughout the window; a real late event
  contributes only briefly to Equation A5.

  The SD mismatch is more fundamental. The empirical distribution is a
  spike-count and spike-time mixture,

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

  == Appendix: model specification and calculations <sec-appendix-model-specification-and-calculations>

  === Simulate the finite-window feature

  We fixed the normalized pixel intensity at $x=1$ and varied only its input
  rate $r$ from #p.input_rate_grid_hz.first() to
  #p.input_rate_grid_hz.last() spikes/s. At each timestep, the pixel generated
  an event with probability

  $ p_"event" = (r Delta t) / 1000. quad "(A1)" $

  Here $r$ is the input rate in spikes/s and $Delta t=#p.dt_ms$ ms. During a
  presentation of duration $T=#p.presentation_ms$ ms, the expected number of
  events is $r T / 1000$. Expected event count is therefore a consequence of the
  chosen rate, not a separate experimental variable.

  Conductance and membrane voltage followed

  $ g(t) = beta g(t-Delta t) + w S(t), quad "(A2)" $

  $ beta = exp(-Delta t/tau_"AMPA"), quad "(A3)" $

  $
    C (d v)/(d t) = g_L (E_L-v) + g(t)(E_e-v). quad "(A4)"
  $

  Here $S(t)$ is one when an event occurs and zero otherwise, $g(t)$ is AMPA
  conductance, $w$ is the conductance increment per event, $beta$ is the
  per-timestep decay factor, and $tau_"AMPA"$ is the AMPA decay time constant.
  The membrane voltage is $v(t)$, $C$ is membrane capacitance, $g_L$ is leak
  conductance, $E_L$ is the leak reversal potential, and $E_e$ is the
  excitatory reversal potential.

  We used $C=1$ nF, $g_L=0.05$ μS, $E_L=-65$ mV, $E_e=0$ mV,
  $tau_"AMPA"=2$ ms, and independent probe conductances
  $w in {#p.probes_uS.map(str).join(", ")}$ μS. Every presentation began from
  $g(0)=0$ and $v(0)=E_L$. Its continuous-time scalar feature was

  $ z = 1/T integral_0^T (v(t)-E_L) dif t. quad "(A5)" $

  Here $z$ is the baseline-subtracted mean voltage and $T$ is presentation
  duration. Simulation used the arithmetic mean of post-update voltages as the
  discrete approximation to this integral.

  For the mean and SD estimates in Figure 1, we generated #p.moment_draws new
  presentations at each of #p.input_rate_grid_hz.len() input rates and each
  conductance. For the full response distributions in Figure 2, we generated
  #p.distribution_draws new presentations at each selected rate. Direct
  simulation retains the nonstationary finite-window behaviour of filtered
  conductance shot noise#cite(1).


  === Calculate the stationary operating point

  For stationary Poisson rate $r$, filtered-shot-noise theory#cite(2) gives the
  mean AMPA conductance

  $ macron(g)_r = r w tau_"AMPA" / 1000. quad "(A6)" $

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
    w/(i omega + 1/tau_"AMPA") dot
    (E_e-macron(v)_r)/
    (i omega C + g_L + macron(g)_r). quad "(A9)"
  $

  Here $omega$ is angular frequency in rad/ms and $i$ is the imaginary unit.
  The transfer function $G_r (omega)$ maps a small input-rate fluctuation to its
  voltage response around the operating point. #link(<sec-synapse-plus-membrane-linearization>)[Synapse-plus-membrane linearization] derives Equation A9
  by linearizing the synaptic and membrane equations.

  Averaging voltage over the finite presentation contributes

  $ A_T (omega) = (1-exp(-i omega T))/(i omega T), quad "(A10)" $

  so the complete input-to-feature response is

  $ H_r (omega) = A_T (omega) G_r (omega). quad "(A11)" $

  Here $A_T (omega)$ is the transfer function of averaging over duration $T$, and
  $H_r (omega)$ is the complete input-to-feature transfer function. #link(<sec-finite-window-average>)[Finite-window average]
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

  $ S_"in" (omega) = r/1000. quad "(A13)" $

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


  == Appendix: derivation of the analytical filter <sec-appendix-derivation-of-the-analytical-filter>

  === Stationary conductance and voltage <sec-stationary-conductance-and-voltage>

  Write the continuous AMPA equation driven by an event train
  $s(t)=sum_k delta(t-t_k)$ as

  $ (d g)/(d t) = -g/tau_"AMPA" + w s(t). quad "(B1)" $

  Here $s(t)$ is the impulse train, $t_k$ is the time of event $k$, and
  $delta(t-t_k)$ is a Dirac impulse at that time.

  For stationary input rate $r$ spikes/s, the rate on the millisecond time base
  is $r/1000$. Taking expectations of Equation B1 and setting the stationary
  derivative to zero gives

  $
    0=-macron(g)_r/tau_"AMPA"+w r/1000,
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

  $ (d delta g)/(d t) = -delta g/tau_"AMPA" + w delta s. quad "(B3)" $

  With Fourier convention $(d)/(d t) -> i omega$,

  $
    delta g(omega)=w/(i omega+1/tau_"AMPA") delta s(omega). quad "(B4)"
  $

  Substitute the decompositions into Equation A4. After cancelling Equation B2
  and discarding the second-order product $delta g delta v$,

  $
    C (d delta v)/(d t)
    =-(g_L+macron(g)_r)delta v
      +(E_e-macron(v)_r)delta g. quad "(B5)"
  $

  Fourier transformation and substitution of Equation B4 yields Equation A9.
  Its two denominator factors are the AMPA and membrane low-pass terms; the
  numerator contains the conductance increment and local excitatory driving
  force.

  === Finite-window average <sec-finite-window-average>

  Linearizing Equation A5 leaves the average of the voltage fluctuation,

  $ delta z = 1/T integral_0^T delta v(t) dif t. quad "(B6)" $

  The averaging kernel is $a_T (t)=1/T$ for $0 <= t <= T$ and zero otherwise.
  Its Fourier transform is

  $
    A_T (omega)=1/T integral_0^T exp(-i omega t) dif t
    =(1-exp(-i omega T))/(i omega T),
  $

  proving Equation A10. Because averaging follows the membrane response, the
  transfer functions multiply, proving Equation A11. Using
  $1-exp(-i x)=2i exp(-i x/2)sin(x/2)$ gives

  $
    abs(A_T (omega))=abs(sin(omega T/2)/(omega T/2)),
  $

  which becomes Equation D1 after substituting $omega=2 pi f/1000$ and
  $T_s=T/1000$.

  === Poisson spectrum and feature variance <sec-poisson-spectrum-and-feature-variance>

  For ideal Poisson events with rate $r/1000$ per ms, disjoint intervals have
  independent counts and count variance equals count mean. The centred impulse
  train therefore has autocovariance

  $ R_(delta s) (tau)=(r/1000)delta(tau). quad "(B7)" $

  Here $R_(delta s) (tau)$ is the input autocovariance at lag $tau$, and
  $delta(tau)$ is the Dirac delta function.

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
#let body = with-contents(body)
