#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "/.demolab/lib.typ": cite, reference-list
#let data-file = data-file.with(article: "exp081")

#let meta = (
  title: "How Pixel Features Respond to Input Rate",
  date: "2026-08-10",
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
#let rounded(x, digits: 3) = str(calc.round(x, digits: digits))
#let body = [
  == Abstract

  We tested what stationary linear-filter theory explains about a sparse
  conductance-driven pixel feature, and where the approximation fails. We
  simulated how an AMPA synapse, a conductance-based membrane, and finite-time
  averaging transform a fully active pixel driven from
  #p.input_rate_grid_hz.first() to #p.input_rate_grid_hz.last() Hz. The model
  explains the flat response to slow input fluctuations, attenuation of fast
  fluctuations, and lobes caused by temporal averaging.

  The model does not accurately predict feature magnitude: it overpredicts the
  mean, and its standard deviation peaks too early and has the wrong shape.
  Sparse responses depend strongly on the discrete number and timing of spikes,
  violating the model's assumption of small, stationary fluctuations. The
  theory explains the filtering, but direct simulation remains necessary for
  quantitative predictions.

  == Methods

  #enum(
    [*Simulate the finite-window feature.*

  We fixed the normalized pixel intensity at $x=1$ and varied only its input
  rate $r$ from #p.input_rate_grid_hz.first() to
  #p.input_rate_grid_hz.last() spikes/s. At each timestep, the pixel generated
  an event with probability

  $ p_"event" = (r Delta t) / 1000. quad "(1)" $

  Here $r$ is the input rate in spikes/s and $Delta t=#p.dt_ms$ ms. During a
  presentation of duration $T=#p.presentation_ms$ ms, the expected number of
  events is $r T / 1000$. Expected event count is therefore a consequence of the
  chosen rate, not a separate experimental variable.

  Conductance and membrane voltage followed

  $ g(t) = beta g(t-Delta t) + w S(t), quad "(2)" $

  $ beta = exp(-Delta t/tau_"AMPA"), quad "(3)" $

  $
    C (d v)/(d t) = g_L (E_L-v) + g(t)(E_e-v). quad "(4)"
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
  $g(0)=0$ and $v(0)=E_L$. Its scalar feature was

  $ z = 1/T integral_0^T (v(t)-E_L) dif t. quad "(5)" $

  Here $z$ is the baseline-subtracted mean voltage and $T$ is presentation
  duration.

  For the mean and SD estimates in Figure 1, we generated #p.moment_draws new
  presentations at each of #p.input_rate_grid_hz.len() input rates and each
  conductance. For the full response distributions in Figure 2, we generated
  #p.distribution_draws new presentations at each selected rate. Direct
  simulation retains the nonstationary finite-window behaviour of filtered
  conductance shot noise#cite(2).

    ],

    [*Calculate the stationary operating point.*

  For stationary Poisson rate $r$, filtered-shot-noise theory#cite(1) gives the
  mean AMPA conductance

  $ macron(g)_r = r w tau_"AMPA" / 1000. quad "(6)" $

  Replacing the fluctuating conductance by this mean gives the operating-point
  voltage

  $
    macron(v)_r =
    (g_L E_L + macron(g)_r E_e) /
    (g_L + macron(g)_r). quad "(7)"
  $

  Because this operating point is constant across the window, its predicted
  mean feature is

  $ mu_"linear" (z) = macron(v)_r-E_L. quad "(8)" $

  An overbar denotes a stationary mean, $macron(g)_r$ is mean conductance at
  rate $r$, $macron(v)_r$ is the corresponding mean voltage, and
  $mu_"linear"(z)$ is the resulting linear prediction for mean feature value.
  Appendix A.1 derives Equations 6--8 from the continuous conductance and
  membrane equations.

    ],

    [*Derive the local frequency response.*

  Linearizing Equations 2--4 around the operating point gives the response from
  input-rate fluctuation to voltage fluctuation,

  $
    G_r (omega) =
    w/(i omega + 1/tau_"AMPA") dot
    (E_e-macron(v)_r)/
    (i omega C + g_L + macron(g)_r). quad "(9)"
  $

  Here $omega$ is angular frequency in rad/ms and $i$ is the imaginary unit.
  The transfer function $G_r(omega)$ maps a small input-rate fluctuation to its
  voltage response around the operating point. Appendix A.2 derives Equation 9
  by linearizing the synaptic and membrane equations.

  Averaging voltage over the finite presentation contributes

  $ A_T (omega) = (1-exp(-i omega T))/(i omega T), quad "(10)" $

  so the complete input-to-feature response is

  $ H_r (omega) = A_T (omega) G_r (omega). quad "(11)" $

  Here $A_T(omega)$ is the transfer function of averaging over duration $T$, and
  $H_r(omega)$ is the complete input-to-feature transfer function. Appendix A.3
  derives Equations 10 and 11.

  To generate Figure 3, we evaluated Equations 9 and 11 from
  #p.frequency_plot_bounds_hz.first() to #p.frequency_plot_bounds_hz.last() Hz at
  #p.frequency_response_rates_hz.map(str).join(", ") spikes/s for the nominal
  #p.nominal_probe_uS μS probe. Frequency in Hz was converted using
  $omega=2 pi f/1000$. The Bode magnitude was reported relative to the
  low-drive DC response,

  $
    M_X (f) = 20 log_10 (abs(X_r (2 pi f/1000))/
    abs(X_(r_"low") (0))). quad "(12)"
  $

  Here $M_X(f)$ is Bode magnitude in dB, $X_r=G_r$ for Figure 3A,
  $X_r=H_r$ for Figure 3B, $f$ is modulation frequency in Hz, and $r_"low"$ is
  the lowest operating rate used as the DC reference.

    ],

    [*Calculate the linearized feature variance.*

  The centred ideal Poisson input has a white two-sided spectrum on the
  millisecond time base,

  $ S_"in" (omega) = r/1000. quad "(13)" $

  The feature spectrum and variance are therefore

  $ S_z (omega)=abs(H_r (omega))^2 S_"in" (omega), quad "(14)" $

  $
    "Var"_"linear" (z) = 1/(2 pi) integral_(-oo)^oo
    abs(H_r (omega))^2 S_"in" (omega) dif omega. quad "(15)"
  $

  Here $S_"in"(omega)$ is the two-sided input power spectrum, $S_z(omega)$ is
  the feature power spectrum, and $"Var"_"linear"(z)$ is the predicted feature
  variance. Appendix A.4 derives Equations 13--15 from the Poisson spectrum and
  the linear-filter variance identity.

  We integrated Equation 15 numerically on a logarithmic frequency grid. A
  second calculation with half as many points measured quadrature refinement.

    ],
  )

  == Results

  === Empirical finite-window response

  Mean response increased with input rate and event strength. Variability first
  increased as spike count and timing diversified, then flattened or declined
  as relative count fluctuations, shunting, and voltage saturation became more
  important.

  #figure(
    data-image(data-file("exp081/empirical_moments.svg"), width: 100%,
      alt: "Two panels show empirical mean feature and feature standard deviation against input rate for three conductance increments."),
    caption: [Empirical finite-window response of a fully active pixel over the
      input-rate grid. Both horizontal axes show input rate in spikes/s. Panel A
      shows mean feature in mV; Panel B shows feature SD in mV. Each coloured
      curve contains #p.input_rate_grid_hz.len() rate conditions. At every rate,
      the plotted mean or SD summarizes #p.moment_draws independently simulated
      presentations; the figure does not display those individual presentations
      as points. Black, red, and cyan denote
      #p.probes_uS.map(str).join(", ") μS conductance increments. Mean response
      rises with rate, while variability reaches a broad maximum or plateau.],
  )

  #figure(
    data-image(data-file("exp081/response_distributions.svg"), width: 100%,
      alt: "Three logarithmic histograms show feature distributions becoming smoother as input rate increases."),
    caption: [Empirical feature distributions at
      #p.distribution_rates_hz.map(str).join(", ") spikes/s for the nominal
      #p.nominal_probe_uS μS probe. Each panel contains #p.distribution_draws new
      presentations. The horizontal axes show feature value in mV. Bars report
      probability per common fixed-width bin on a shared logarithmic vertical
      axis. Sparse input produces an atom at rest and separated timing-dependent
      responses; increasing rate produces a smoother distribution.],
  )

  Input rate fixes only the average number of events. Individual presentations
  can still contain no events, one event, or several events. Even when two
  presentations contain the same number, different event times change the
  finite-window average.

  === Analytical Bode-magnitude response

  The synapse and membrane passed slow modulation but attenuated fast
  modulation. Finite-window averaging superimposed regularly spaced zeros and
  lobes on that falling response.

  #figure(
    data-image(data-file("exp081/frequency_response.svg"), width: 100%,
      alt: "Two panels show the analytical synapse and membrane frequency response before and after finite-window averaging."),
    caption: [Analytical Bode-magnitude plots at the nominal
      #p.nominal_probe_uS μS probe. Black, red, and cyan denote operating rates
      #p.frequency_response_rates_hz.map(str).join(", ") spikes/s. The horizontal
      axis is modulation frequency in Hz and the vertical axis is gain relative
      to the low-drive direct-current response in decibels. Panel A shows
      Equation 9; Panel B shows the complete #p.presentation_ms ms
      window-averaged response from Equation 11.],
  )

  In Panel A, each curve is flat at low frequency because the AMPA conductance
  and membrane voltage can follow a slowly varying input quasi-statically.
  Greater drive lowers that plateau by shunting the membrane and reducing the
  excitatory driving force. The curves bend downward once modulation becomes
  too fast for the synaptic and membrane timescales to follow.

  Panel B is the same synapse-plus-membrane response multiplied by the
  rectangular-window response. With $T_s=T/1000$ s,

  $
    abs(A_T (2 pi f/1000)) =
    abs(sin(pi f T_s)/(pi f T_s)). quad "(16)"
  $

  Its zeros at $f=n/T_s$ cancel modulation containing an integer number of
  cycles within the presentation. Here $n$ is any nonzero integer. The absolute
  sinc response creates the lobes, while the Panel A low-pass response supplies
  their falling envelope. Appendix A.3 derives Equation 16 from the
  finite-window averaging kernel.

  === Analytical and empirical moments

  The stationary model reproduced the mean curve's broad shape but not its
  magnitude, and it failed to reproduce the shape of the empirical standard
  deviation.

  #figure(
    data-image(data-file("exp081/analytical_empirical.svg"), width: 100%,
      alt: "Two panels compare analytical curves with empirical points for feature mean and standard deviation over input rate."),
    caption: [Analytical and empirical moments over input rate for a fully active
      pixel. The horizontal axis is input rate in spikes/s. Solid curves are
      stationary predictions from Equations 8 and 15; faint points are
      independently simulated finite-window estimates. Panel A shows mean
      feature in mV and Panel B feature SD in mV. The stationary theory
      overpredicts the mean and places the SD maximum too early.],
  )

  The analytical mean preserved the broad curvature and conductance ordering but
  exceeded the empirical magnitude. Its median predicted-to-empirical ratio was
  #rounded(r.comparison.mean.median_predicted_empirical_ratio). The stationary
  calculation evaluates the response at mean conductance, $z(E[g])$, whereas
  simulation estimates the response over random conductance paths, $E[z(g)]$.
  Saturation makes these unequal, approximately

  $ E[z(g)] < z(E[g]). $

  The stationary input also acts throughout the window; a real late event
  contributes only briefly to Equation 5.

  The SD mismatch is more fundamental. The empirical distribution is a
  spike-count and spike-time mixture,

  $
    p_Z (z) = sum_(n=0)^oo P(N=n) p_(Z|N) (z|n). quad "(17)"
  $

  Here $Z$ is the random feature value, $N$ is the event count, $P(N=n)$ is the
  probability of observing $n$ events, and $p_(Z|N)(z|n)$ is the feature
  distribution conditional on that count.

  Equation 15 instead treats input as a small stationary fluctuation around one
  operating point. At low rates, zero-event trials remain exactly at rest, one
  large event produces a timing-dependent response, and multiple events interact
  through shunting and saturation. That non-Gaussian mixture explains why the
  analytical SD rises too sharply and peaks too early, especially for the
  largest conductance increment.

  == Conclusion

  Stationary linear-filter theory correctly explains the low-frequency plateau,
  high-frequency roll-off, averaging-window zeros, and qualitative operating-
  point dependence. It does not quantitatively predict finite-window moments in
  the sparse, large-jump regime. Direct simulation is therefore required for
  empirical rate selection, while the analytical model remains useful for
  explaining the filter's structure.

  == Limitations

  The empirical moment grid used #p.moment_draws presentations per condition;
  it was designed to resolve the qualitative theoretical comparison, not to
  estimate extreme distribution tails. The analytical variance assumes ideal
  continuous-time Poisson drive and a local stationary linearization, whereas
  simulation uses discrete Bernoulli events and begins from rest.

  == Reproducibility

  `uv run python experiments/exp081.py` regenerates every sample, summary, and
  figure. The runner contains a complete independent physical specification and
  consumes no artifacts or code from another experiment.

  == Appendix A: derivation of the analytical filter

  === A.1 Stationary conductance and voltage

  Write the continuous AMPA equation driven by an event train
  $s(t)=sum_k delta(t-t_k)$ as

  $ (d g)/(d t) = -g/tau_"AMPA" + w s(t). quad "(A1)" $

  Here $s(t)$ is the impulse train, $t_k$ is the time of event $k$, and
  $delta(t-t_k)$ is a Dirac impulse at that time.

  For stationary input rate $r$ spikes/s, the rate on the millisecond time base
  is $r/1000$. Taking expectations of Equation A1 and setting the stationary
  derivative to zero gives

  $
    0=-macron(g)_r/tau_"AMPA"+w r/1000,
  $

  hence Equation 6. The stationary membrane equation is

  $
    0=g_L (E_L-macron(v)_r)
      +macron(g)_r (E_e-macron(v)_r), quad "(A2)"
  $

  and collecting the terms multiplying $macron(v)_r$ gives Equation 7.

  === A.2 Synapse-plus-membrane linearization

  Decompose each quantity into its operating point and a small fluctuation,

  $
    g=macron(g)_r+delta g,
    quad v=macron(v)_r+delta v,
    quad s=r/1000+delta s.
  $

  Substitution into Equation A1 and cancellation of the stationary terms gives

  $ (d delta g)/(d t) = -delta g/tau_"AMPA" + w delta s. quad "(A3)" $

  With Fourier convention $(d)/(d t) -> i omega$,

  $
    delta g(omega)=w/(i omega+1/tau_"AMPA") delta s(omega). quad "(A4)"
  $

  Substitute the decompositions into Equation 4. After cancelling Equation A2
  and discarding the second-order product $delta g delta v$,

  $
    C (d delta v)/(d t)
    =-(g_L+macron(g)_r)delta v
      +(E_e-macron(v)_r)delta g. quad "(A5)"
  $

  Fourier transformation and substitution of Equation A4 yields Equation 9.
  Its two denominator factors are the AMPA and membrane low-pass terms; the
  numerator contains the conductance increment and local excitatory driving
  force.

  === A.3 Finite-window average

  Linearizing Equation 5 leaves the average of the voltage fluctuation,

  $ delta z = 1/T integral_0^T delta v(t) dif t. quad "(A6)" $

  The averaging kernel is $a_T (t)=1/T$ for $0 <= t <= T$ and zero otherwise.
  Its Fourier transform is

  $
    A_T (omega)=1/T integral_0^T exp(-i omega t) dif t
    =(1-exp(-i omega T))/(i omega T),
  $

  proving Equation 10. Because averaging follows the membrane response, the
  transfer functions multiply, proving Equation 11. Using
  $1-exp(-i x)=2i exp(-i x/2)sin(x/2)$ gives

  $
    abs(A_T (omega))=abs(sin(omega T/2)/(omega T/2)),
  $

  which becomes Equation 16 after substituting $omega=2 pi f/1000$ and
  $T_s=T/1000$.

  === A.4 Poisson spectrum and feature variance

  For ideal Poisson events with rate $r/1000$ per ms, disjoint intervals have
  independent counts and count variance equals count mean. The centred impulse
  train therefore has autocovariance

  $ R_(delta s) (tau)=(r/1000)delta(tau). quad "(A7)" $

  Here $R_(delta s)(tau)$ is the input autocovariance at lag $tau$, and
  $delta(tau)$ is the Dirac delta function.

  Fourier transformation of the delta function gives the constant spectrum in
  Equation 13. A linear time-invariant filter multiplies the input spectrum by
  squared transfer magnitude, proving Equation 14. Finally, inverse Fourier
  transformation at zero lag gives

  $
    "Var"(z)=R_z (0)=1/(2 pi) integral_(-oo)^oo S_z (omega) dif omega,
  $

  which, after substituting Equation 14, proves Equation 15. The result is exact
  for the stated stationary linearized model but only approximate for the
  nonlinear, start-from-rest, finite-event simulation.

  #reference-list((
    (
      text: [Wolff & Lindner: _Mean, Variance, and Autocorrelation of Subthreshold Potential Fluctuations Driven by Filtered Conductance Shot Noise_. Neural Computation, 2010.],
      doi: "10.1162/neco.2009.02-09-958",
    ),
    (
      text: [Brigham & Destexhe: _Nonstationary Filtered Shot-Noise Processes and Applications to Neuronal Membranes_. Physical Review E, 2015.],
      doi: "10.1103/PhysRevE.91.062102",
    ),
  ))
]
#body
]

#let body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [How do pixel-derived features respond to input rate? Compare predicted response statistics with measured moments, distributions, and frequency responses.],
    preview-figures, json-inputs: ("exp081",),
  )
}
