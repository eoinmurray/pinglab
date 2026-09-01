#import "contents.typ": with-contents, with-numbered-equations, with-result-sections
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#import "/.demolab/lib.typ": cite, reference-list
#let data-file = data-file.with(article: "exp081")

#let meta = (
  status: "◉ REVIEWED",
  writing_guide: "31.2.0",
  title: "How Pixel Features Respond to Input Rate",
  created_at: "2026-08-10T00:00:00Z",
  updated_at: "2026-09-01",
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

  We asked how the spike rate produced by a fully active pixel determines the
  average voltage feature passed to downstream classifiers. We compared direct
  membrane simulations with a simpler theory that treats the system as steady
  and approximately linear.

  The theory reproduced the general filtering pattern and mean-response shape,
  but predicted responses that were too large and failed to capture variation
  between presentations. Accurate finite-window predictions therefore still
  require direct simulation of individual event counts and timings in this
  system.

  == Results

  #with-result-sections[

  #result-card-style

  #result-card[
  === Average voltage and its variation across input rates

  Each presentation began from rest. We varied the input rate and the
  conductance added by each event, then averaged the resulting voltage rise over
  the presentation.

  A higher input rate should raise the average voltage. Variation need not rise
  steadily, however, because presentations differ in both the number and timing
  of their events, and membrane responses are nonlinear.

  The mean voltage feature rose with both input rate and event strength. Its SD
  rose quickly at low rates, then levelled off or fell for the two weaker event
  strengths, but remained high for the strongest. One possible explanation is
  that event counts become relatively steadier while stronger conductance limits
  further voltage change through shunting and saturation. We did not test these
  mechanisms separately.

  #figure(
    data-image(data-file("exp081/empirical_moments.svg"), width: 100%,
      alt: "Two panels show empirical mean feature and feature standard deviation against input rate for three conductance increments."),
    caption: [Simulated mean voltage feature (A) and sample SD (B), in mV, across
      #p.input_rate_grid_hz.len() input rates. Each condition summarizes
      #p.moment_draws independent presentations. Black, red and cyan denote
      #p.probes_uS.map(str).join(", ") μS added per event; curves summarize the
      presentations and are not confidence intervals.],
  )
  ]

  #result-card[
  === Response distributions from sparse to dense input

  Here we kept presentation duration and event strength fixed, and changed only
  the input rate from sparse to dense drive.

  Input rate determines the expected number of events, not the exact number in
  any one presentation. A presentation may contain no events, one event or many;
  even equal event counts can give different average voltages when their timing
  differs.

  At the lowest rate, most responses were near zero. The middle rate produced a
  mixture of sparse responses, while the highest rate produced a broad, almost
  continuous spread of responses.

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
  === Predicted filtering before and after time averaging

  This figure shows the theory's response around three steady input rates. It is
  a theoretical prediction, not a measurement from a modulation experiment.

  The model responds similarly to slow changes, but increasingly suppresses
  changes that are faster than the synapse and membrane can follow. Stronger
  input also reduces the response to slow changes because it increases shunting
  and leaves less voltage difference to drive excitation. Averaging over a fixed
  window adds the regularly spaced dips and rebounds in the response.

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
  === Theory compared with simulation across input rates

  The simpler theory replaces each random conductance history with its mean
  steady state, then assumes that the remaining fluctuations are small.

  If this approximation were quantitatively accurate, it would reproduce both
  the shape and size of the simulated mean and SD curves.

  The predicted means had the same broad curvature and ordering by event
  strength (Pearson correlation #rounded(r.comparison.mean.pearson_r)), but were
  too large by a median factor of
  #rounded(r.comparison.mean.median_predicted_empirical_ratio). SD correspondence
  was weak (Pearson correlation
  #rounded(r.comparison.standard_deviation.pearson_r)), especially for the largest
  event strength. The theory therefore explains the general filtering pattern,
  but not the amount of variation across finite presentations.

  #figure(
    data-image(data-file("exp081/analytical_empirical.svg"), width: 100%,
      alt: "Two panels compare analytical curves with empirical points for feature mean and standard deviation over input rate."),
    caption: [Steady-state predictions (solid curves) and simulated estimates
      (points) of mean feature (A) and sample SD (B), in mV. Each estimate uses
      #p.moment_draws presentations per rate. Black, red and cyan denote
      #p.probes_uS.map(str).join(", ") μS added per event.],
  )
  ]

  ]

  == Methods

  === Compute

  + *Set the conditions.* We used a fully active pixel and varied its input rate
    from #p.input_rate_grid_hz.first() to
    #p.input_rate_grid_hz.last() spikes/s across
    #p.input_rate_grid_hz.len() conditions. We also varied the conductance added
    by each event across
    #p.probes_uS.map(str).join(", ") μS. Each condition used
    #p.moment_draws independent encoding draws of a #p.presentation_ms ms
    presentation. Separate distribution measurements used
    #p.distribution_draws draws at
    #p.distribution_rates_hz.map(str).join(", ") spikes/s and
    #p.nominal_probe_uS μS.

  + *Generate input events.* At each simulation step $k$, we made an independent
    yes-or-no event draw with probability
    $p_"event"=r_"input" Delta t_"sim"/1000$, where $r_"input"$ is input rate in
    spikes/s and $Delta t_"sim"=#p.dt_ms$ ms. Physical time was
    $t_k=k Delta t_"sim"$ and $N_t=T_"present"/Delta t_"sim"$ steps formed one
    presentation. We used separate reproducible random streams for the moment
    and distribution measurements. Every presentation began with zero
    conductance and the membrane at its resting voltage.

  + *Integrate conductance and voltage.* AMPA conductance decayed with time
    constant $tau_"AMPA"=#p.membrane.tau_ampa_ms$ ms, and each event then added
    its assigned conductance. Membrane voltage obeyed
    $ C_m (d V_m)/(d t)=g_L (E_L-V_m)+g(t)(E_e-V_m). $
    Here $t$ is physical time in ms, $V_m$ is membrane voltage in mV, $g$ is
    excitatory conductance in μS, $C_m=#p.membrane.C_m_nF$ nF is capacitance,
    $g_L=#p.membrane.g_L_uS$ μS is leak conductance, and
    $E_L=#p.membrane.E_L_mV$ mV and $E_e=#p.membrane.E_e_mV$ mV are reversal
    potentials. Within each step, we held the updated conductance fixed and used
    the exact exponential voltage solution in single precision.

  + *Measure the voltage feature.* For each encoding draw, we approximated
    $ z_"feature"=1/T_"present" integral_0^(T_"present") (V_m(t)-E_L) dif t.
 $
    Here $z_"feature"$ is the mean voltage rise above rest, in mV, and
    $T_"present"=#p.presentation_ms$ ms is presentation duration. The recorded
    estimate subtracted resting voltage and averaged the $N_t$ voltages measured
    after each simulation update.

  === Analyse

  #set enum(start: 5)

  + *Summarize the simulated responses.* For every rate–strength condition, we
    calculated the mean and sample SD across encoding draws; the SD used the
    usual $n-1$ variance denominator. For the distribution measurements, we used
    #p.histogram_bins equal-width bins from zero to the largest response rounded
    up to the next 5 mV, then divided each bin count by the number of draws.

  + *Calculate the steady-state prediction.* For each condition, we calculated
    the mean conductance and its equilibrium voltage. We then assumed that small
    fluctuations around this steady state followed a linear synaptic and
    membrane response. Averaging across the presentation gave $H_r(omega)$, the
    predicted mapping from input fluctuations to the voltage feature at angular
    frequency $omega$ in rad/ms, with predicted variance
    $ "Var"(z_"feature")=1/(2 pi) integral_(-oo)^oo abs(H_r(omega))^2
      (r_"input"/1000) dif omega. $
    #link(<sec-appendix-model-specification-and-calculations>)[Appendix: model specification and calculations] specifies this approximation, and
    #link(<sec-appendix-derivation-of-the-analytical-filter>)[Appendix: derivation of the analytical filter] derives it.

  + *Integrate and compare.* We numerically integrated Equation 3 on a
    logarithmically spaced frequency grid, included the unrepresented
    near-zero-frequency tail and repeated the calculation on a coarser grid as a
    check. We compared predicted and simulated means and SDs using Pearson
    correlation, mean absolute error and the median predicted-to-simulated ratio.
    We excluded pairs in which both values were zero, calculated ratios only for
    positive simulated values and left undefined correlations undefined. We
    expressed frequency-response magnitude relative to the zero-frequency
    response at the lowest input rate.

  === Present

  #set enum(start: 8)

  + *Show the simulated evidence.* We displayed the recorded means and sample
    SDs for every rate–strength condition, together with the probability in each
    shared bin for the three distribution measurements. We reused the retained
    analysis without generating new draws or confidence intervals.

  + *Show the theoretical evidence.* We displayed the calculated local
    frequency responses before and after window averaging, then overlaid the
    predicted and simulated moments for the same conditions. The
    frequency-response figure shows a theoretical mechanism, not a measured
    modulation response. We did not recalculate the statistics or rerun the
    simulation while preparing the figures.

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
