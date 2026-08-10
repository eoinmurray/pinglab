#import "/.demolab/lib.typ": cite, reference-list

#let meta = (
  title: "Linear-filter analysis of sparse conductance-driven pixel features",
  date: "2026-08-10",
  description: "Standalone empirical and analytical study of synaptic, membrane, and finite-window filtering under sparse Poisson drive.",
  collection: "gamma-gated-sparsity",
  status: "draft",
)

#let r = json("/artifacts/data/exp081/numbers.json")
#let p = r.parameters
#let rounded(x, digits: 3) = str(calc.round(x, digits: digits))
#let body = [
  == Abstract

  We independently simulated and analyzed the conductance-filtered single-pixel
  feature used in rate-coded MNIST experiments. A stationary linearization
  explains the flat low-frequency gain, the high-frequency roll-off, and the
  lobes introduced by finite-window averaging. It preserves the ordering and
  broad curvature of empirical response means but overpredicts their magnitude;
  its predicted feature SD also peaks too early. The discrepancy arises because
  0--5 expected events form discrete spike-count and spike-time mixtures rather
  than small Gaussian fluctuations around a stationary operating point.

  == Purpose

  This standalone experiment asks:

  #quote(block: true)[What does stationary linear-filter theory explain about
    the finite-window conductance-driven pixel feature, and where does that
    approximation fail under sparse input?]

  The experiment specifies and simulates its own physical model. It shares no
  artifacts or experiment code with EXP080; the two entries are related only by
  studying the same parameterized feature family.

  == Methods

  === Independent finite-window simulation

  Each condition was defined directly by its expected event count
  $Lambda in [0,5]$ during a $T=#p.presentation_ms$ ms presentation. The
  per-timestep event probability was

  $ p_"event" = Lambda / N = lambda Delta t / 1000, quad "(1)" $

  where $N=T/Delta t$, $Delta t=#p.dt_ms$ ms, and
  $lambda=1000 Lambda/T$ spikes/s. This direct expected-count grid avoids
  duplicating conditions through different rate--intensity pairs.

  Conductance and membrane voltage followed

  $ g(t) = beta g(t-Delta t) + w S(t), quad "(2)" $

  $ beta = exp(-Delta t/tau_"AMPA"), quad "(3)" $

  $
    C (d v)/(d t) = g_L (E_L-v) + g(t)(E_e-v). quad "(4)"
  $

  We used $C=1$ nF, $g_L=0.05$ μS, $E_L=-65$ mV, $E_e=0$ mV,
  $tau_"AMPA"=2$ ms, and independent probe conductances
  $w in {0.6,1.2,2.4}$ μS. Every presentation began from $g(0)=0$ and
  $v(0)=E_L$. Its scalar feature was

  $ z = 1/T integral_0^T (v(t)-E_L) dif t. quad "(5)" $

  At each of #p.expected_spike_grid.len() unique expected-count conditions and
  each conductance, we generated #p.moment_draws new presentations. Separate
  Separate simulations with #p.distribution_draws draws characterized representative full
  response distributions.

  === Stationary operating point

  For stationary Poisson rate $lambda$, the mean AMPA conductance is

  $ overline(g)_lambda = lambda w tau_"AMPA" / 1000. quad "(6)" $

  Replacing the fluctuating conductance by this mean gives the operating-point
  voltage

  $
    overline(v)_lambda =
    (g_L E_L + overline(g)_lambda E_e) /
    (g_L + overline(g)_lambda). quad "(7)"
  $

  Because this operating point is constant across the window, its predicted
  mean feature is

  $ mu_"linear" (z) = overline(v)_lambda-E_L. quad "(8)" $

  === Local frequency response

  Linearizing Equations 2--4 around the operating point gives the response from
  input-rate fluctuation to voltage fluctuation,

  $
    G_lambda (omega) =
    w/(i omega + 1/tau_"AMPA") dot
    (E_e-overline(v)_lambda)/
    (i omega C + g_L + overline(g)_lambda). quad "(9)"
  $

  Here $ω$ is angular frequency in rad/ms. Averaging voltage over the finite
  presentation contributes

  $ A_T (omega) = (1-exp(-i omega T))/(i omega T), quad "(10)" $

  so the complete input-to-feature response is

  $ H_lambda (omega) = A_T (omega) G_lambda (omega). quad "(11)" $

  We evaluated Equations 9 and 11 from 0.1--200 Hz at 0.25, 3, and 25
  spikes/s for the nominal 1.2 μS probe. Frequency in Hz was converted using
  $omega=2 pi f/1000$. Gain was reported relative to the low-drive DC response,

  $
    20 log_10 (abs(G_lambda (2 pi f/1000))/
    abs(G_(lambda_"low") (0))). quad "(12)"
  $

  === Linearized feature variance

  The centred ideal Poisson input has a white two-sided spectrum on the
  millisecond time base,

  $ S_"in" (omega) = lambda/1000. quad "(13)" $

  The feature spectrum and variance are therefore

  $ S_z (omega)=abs(H_lambda (omega))^2 S_"in" (omega), quad "(14)" $

  $
    "Var"_"linear" (z) = 1/(2 pi) integral_(-oo)^oo
    abs(H_lambda (omega))^2 S_"in" (omega) dif omega. quad "(15)"
  $

  We integrated Equation 15 numerically on a logarithmic frequency grid. A
  second calculation with half as many points measured quadrature refinement.

  == Results

  === Empirical finite-window response

  #figure(
    image("/artifacts/data/exp081/empirical_moments.svg", width: 100%),
    caption: [Empirical finite-window response over a unique expected-event-count
      grid. Panel A shows mean feature; Panel B shows feature SD. Every point
      summarizes #p.moment_draws independently simulated presentations. Black,
      red, and cyan denote 0.6, 1.2, and 2.4 μS conductance increments.],
  )

  Mean response increased with expected event count and event strength.
  Variability initially increased as spike count and timing diversified, then
  flattened or declined as relative count fluctuations, shunting, and voltage
  saturation became more important.

  #figure(
    image("/artifacts/data/exp081/response_distributions.svg", width: 100%),
    caption: [Empirical feature distributions at 0.05, 0.6, and 5 expected
      events for the nominal 1.2 μS probe. Each panel contains
      #p.distribution_draws new presentations. Bars report probability per
      common fixed-width bin on a shared logarithmic axis. Sparse input produces
      an atom at rest and separated timing-dependent responses; increasing event
      count produces a smoother distribution.],
  )

  An expected count is not an observed count. At $Lambda=1$, for example,
  presentations can contain zero, one, two, or more events. Even at fixed count,
  their event times alter the finite-window average.

  === Analytical frequency response

  #figure(
    image("/artifacts/data/exp081/frequency_response.svg", width: 100%),
    caption: [Analytical linearized frequency response at the nominal 1.2 μS
      probe. Black, red, and cyan denote operating rates 0.25, 3, and 25
      spikes/s. Panel A shows Equation 9; Panel B shows the complete
      #p.presentation_ms ms window-averaged response from Equation 11.],
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

  Its zeros at $f=n/T_s$ cancel complete cycles at 5, 10, 15 Hz, and so on.
  The absolute sinc response creates the lobes, while the Panel A low-pass
  response supplies their falling envelope.

  === Analytical and empirical moments

  #figure(
    image("/artifacts/data/exp081/analytical_empirical.svg", width: 100%),
    caption: [Analytical and empirical moments over expected input events.
      Solid curves are stationary predictions from Equations 8 and 15; faint
      points are independently simulated finite-window estimates. Panel A shows
      mean feature and Panel B feature SD.],
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

  Equation 15 instead treats input as a small stationary fluctuation around one
  operating point. At low expected counts, zero-event trials remain exactly at
  rest, one large event produces a timing-dependent response, and multiple
  events interact through shunting and saturation. That non-Gaussian mixture
  explains why the analytical SD rises too sharply and peaks too early,
  especially for the largest conductance increment.

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

  For stationary rate $λ$ spikes/s, the rate on the millisecond time base is
  $lambda/1000$. Taking expectations of Equation A1 and setting the stationary
  derivative to zero gives

  $
    0=-overline(g)_lambda/tau_"AMPA"+w lambda/1000,
  $

  hence Equation 6. The stationary membrane equation is

  $
    0=g_L (E_L-overline(v)_lambda)
      +overline(g)_lambda (E_e-overline(v)_lambda), quad "(A2)"
  $

  and collecting the terms multiplying $overline(v)_lambda$ gives Equation 7.

  === A.2 Synapse-plus-membrane linearization

  Decompose each quantity into its operating point and a small fluctuation,

  $
    g=overline(g)_lambda+delta g,
    quad v=overline(v)_lambda+delta v,
    quad s=lambda/1000+delta s.
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
    =-(g_L+overline(g)_lambda)delta v
      +(E_e-overline(v)_lambda)delta g. quad "(A5)"
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

  For ideal Poisson events with rate $λ/1000$ per ms, disjoint intervals have
  independent counts and count variance equals count mean. The centred impulse
  train therefore has autocovariance

  $ R_(delta s) (tau)=(lambda/1000)delta(tau). quad "(A7)" $

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
