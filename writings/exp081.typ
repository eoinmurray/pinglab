#import "contents.typ": contents-here, with-contents, with-numbered-equations, with-result-sections
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#import "/.demolab/lib.typ": cite, reference-list
#let data-file = data-file.with(article: "exp081")

#let meta = (
  tags: ("data", "reviewed", "v35.0.0"),
  title: "How Pixel Features Respond to Input Rate",
  created_at: "2026-08-10T00:00:00Z",
  updated_at: "2026-09-01",
  description: "Direct simulations and a simpler theory test how sparse input events become average voltage features.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp081",)
#let preview-figures = (
  (path: "exp081/empirical_moments.svg", label: "simulated mean and variation"),
  (path: "exp081/response_distributions.svg", label: "response distributions"),
  (path: "exp081/frequency_response.svg", label: "frequency response"),
  (path: "exp081/analytical_empirical.svg", label: "theory compared with simulation"),
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
  between presentations. Accurately predicting voltage averaged over one
  presentation therefore still requires direct simulation of individual event
  counts and timings in this system.

  #contents-here()

  == Results

  #with-result-sections[

  #result-card-style

  #result-card[
  === Average voltage and its variation across input rates

  #figure(
    data-image(data-file("exp081/empirical_moments.svg"), width: 100%,
      alt: "Two panels show how the simulated mean voltage and its standard deviation change with input rate for three event strengths."),
    caption: [*(A)* Simulated mean voltage feature and *(B)* sample SD, in mV, across
      #p.input_rate_grid_hz.len() input rates. Each condition summarizes
      #p.moment_draws independent presentations. Black, red and cyan denote
      #p.probes_uS.map(str).join(", ") μS added per event; curves summarize the
      presentations and are not confidence intervals.],
  )

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
  ]

  #result-card[
  === Response distributions from sparse to dense input

  #figure(
    data-image(data-file("exp081/response_distributions.svg"), width: 100%,
      alt: "Three histograms show voltage responses changing from mostly near zero to a broad continuous spread as input rate increases."),
    caption: [Distributions of the voltage feature at *(A–C)*
      #p.distribution_rates_hz.map(str).join(", ") spikes/s, respectively, and
      #p.nominal_probe_uS μS, using #p.distribution_draws independent
      presentations per condition. Each bar gives the probability within one
      shared fixed-width bin. The vertical axis is logarithmic and voltage is in
      mV.],
  )

  Here we kept presentation duration and event strength fixed, and changed only
  the input rate from sparse to dense drive.

  Input rate determines the expected number of events, not the exact number in
  any one presentation. A presentation may contain no events, one event or many;
  even equal event counts can give different average voltages when their timing
  differs.

  At the lowest rate, most responses were near zero. The middle rate produced a
  mixture of sparse responses, while the highest rate produced a broad, almost
  continuous spread of responses.
  ]

  #result-card[
  === Predicted filtering before and after time averaging

  #figure(
    data-image(data-file("exp081/frequency_response.svg"), width: 100%,
      alt: "Two panels show the predicted response to slow and fast input changes before and after averaging voltage over the presentation."),
    caption: [Theoretical responses at #p.nominal_probe_uS μS. The simulated
      input rate was not deliberately varied over time to measure these curves.
      Black, red and cyan denote steady input rates of
      #p.frequency_response_rates_hz.map(str).join(", ") spikes/s. *(A)* shows
      the synapse and membrane response. *(B)* also includes
      #p.presentation_ms ms time averaging. Magnitude is in dB relative to the
      steady response at the lowest input rate; frequency is in Hz.],
  )

  This figure shows the theory's response around three steady input rates. It is
  a theoretical prediction. We did not measure these curves by deliberately
  varying the input rate over time.

  The model responds similarly to slow changes, but increasingly suppresses
  changes that are faster than the synapse and membrane can follow. Stronger
  input also reduces the response to slow changes because it increases shunting
  and leaves less voltage difference to drive excitation. Averaging over a fixed
  window adds the regularly spaced dips and rebounds in the response.
  ]

  #result-card[
  === Theory compared with simulation across input rates

  #figure(
    data-image(data-file("exp081/analytical_empirical.svg"), width: 100%,
      alt: "Two panels compare theoretical curves with simulated points for mean voltage and standard deviation across input rates."),
    caption: [Steady-state predictions (solid curves) and simulated estimates
      (points) of *(A)* mean feature and *(B)* sample SD, in mV. Each estimate uses
      #p.moment_draws presentations per rate. Black, red and cyan denote
      #p.probes_uS.map(str).join(", ") μS added per event.],
  )

  The simpler theory replaces each random conductance history with its mean
  steady state, then assumes that the remaining fluctuations are small.

  If this approximation were quantitatively accurate, it would reproduce both
  the shape and size of the simulated mean and SD curves.

  The predicted means had the same broad curvature and ordering by event
  strength (Pearson correlation #rounded(r.comparison.mean.pearson_r)), but were
  too large by a median factor of
  #rounded(r.comparison.mean.median_predicted_empirical_ratio). The predicted and
  simulated SDs agreed poorly (Pearson correlation
  #rounded(r.comparison.standard_deviation.pearson_r)), especially for the largest
  event strength. The theory therefore explains the general filtering pattern,
  but not the amount of variation across finite presentations.
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
    #link(<sec-appendix-model-specification-and-calculations>)[Appendix: simulation and theory details] specifies this approximation, and
    #link(<sec-appendix-derivation-of-the-analytical-filter>)[Appendix: how the
    simpler theory is derived] shows where it comes from.

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

  + *Show the theoretical evidence.* We displayed the predicted response to
    different frequencies before and after time averaging, then overlaid the
    predicted and simulated means and SDs for the same conditions. The
    frequency-response figure shows a theoretical mechanism, not a measured
    response to an input rate deliberately varied over time. We did not
    recalculate the statistics or rerun the simulation while preparing the
    figures.

  #run-view("exp081", inputs)

  == Appendix: simulation and theory details <sec-appendix-model-specification-and-calculations>

  === Simulate one pixel's voltage feature

  We set the normalized pixel intensity to its maximum, $x=1$, and varied only
  its input rate $r_"input"$, from #p.input_rate_grid_hz.first() to
  #p.input_rate_grid_hz.last() spikes/s. At each simulation step, the pixel
  either generated an event or did not. The event probability was

  $ p_"event" = (r_"input" Delta t_"sim") / 1000. quad "(A1)" $

  Here $r_"input"$ is the input rate in spikes/s and
  $Delta t_"sim"=#p.dt_ms$ ms is the length of one simulation step. Over a
  presentation lasting $T_"present"=#p.presentation_ms$ ms, the expected event
  count is $r_"input" T_"present" / 1000$. We therefore controlled the rate;
  the number of events in an individual presentation remained random.

  Each event increased AMPA conductance, which then decayed over time:

  $ g[k] = beta_"AMPA" g[k-1] + w_"event" s[k], quad "(A2)" $

  $ beta_"AMPA" = exp(-Delta t_"sim"/tau_"AMPA"), quad "(A3)" $

  $
    C_m (d V_m)/(d t) = g_L (E_L-V_m) + g(t)(E_e-V_m). quad "(A4)"
  $

  The indicator $s[k]$ equals one when step $k$ contains an event and zero
  otherwise. The AMPA conductance at that step is $g[k]$; each event adds
  $w_"event"$. Between events, conductance retains the fraction $beta_"AMPA"$
  from the previous step, as set by the AMPA decay time
  $tau_"AMPA"$. Membrane voltage $V_m(t)$ then follows Equation A4, where $C_m$
  is membrane capacitance, $g_L$ is leak conductance, $E_L$ is the resting leak
  potential and $E_e$ is the excitatory reversal potential.

  We used $C_m=1$ nF, $g_L=0.05$ μS, $E_L=-65$ mV, $E_e=0$ mV,
  $tau_"AMPA"=2$ ms, and event strengths
  $w_"event" in {#p.probes_uS.map(str).join(", ")}$ μS. Every presentation
  began with no AMPA conductance, $g(0)=0$, and the membrane at rest,
  $V_m(0)=E_L$. We defined its voltage feature as

  $ z_"feature" = 1/T_"present" integral_0^(T_"present") (V_m(t)-E_L) dif t. quad "(A5)" $

  The feature $z_"feature"$ is the average voltage above rest during the
  presentation. In the discrete simulation, we approximated this integral by
  averaging the voltage after each update and subtracting $E_L$.

  For the mean and SD estimates in Figure 1, we generated #p.moment_draws new
  presentations at each of #p.input_rate_grid_hz.len() input rates and each
  conductance. For the full response distributions in Figure 2, we generated
  #p.distribution_draws new presentations at each selected rate. This direct
  simulation preserves the random event counts and timings, the start from
  rest, and their combined effect within the finite presentation#cite(1).


  === Replace random input with its steady average

  The simpler theory begins by replacing the changing AMPA conductance with its
  long-run average. For an ongoing Poisson input at rate $r=r_"input"$,
  filtered-shot-noise theory#cite(2) gives this mean as

  $ macron(g)_r = r w_"event" tau_"AMPA" / 1000. quad "(A6)" $

  Holding conductance at this mean gives the corresponding steady membrane
  voltage:

  $
    macron(v)_r =
    (g_L E_L + macron(g)_r E_e) /
    (g_L + macron(g)_r). quad "(A7)"
  $

  Because this voltage does not change during the presentation, the predicted
  mean feature is simply its rise above rest:

  $ mu_"linear" (z) = macron(v)_r-E_L. quad "(A8)" $

  An overbar marks a steady average. Thus $macron(g)_r$ is the mean conductance
  at input rate $r$, $macron(v)_r$ is the voltage produced by that mean
  conductance, and $mu_"linear" (z)$ is the theory's predicted mean voltage
  feature.
  #link(<sec-stationary-conductance-and-voltage>)[From events to steady
  conductance and voltage] derives Equations A6–A8 from the continuous
  conductance and membrane equations.


  === Predict responses to slow and fast changes

  Next, we asked how a small change in input rate would change voltage around
  the steady state. Treating these changes as small makes the equations linear
  and gives

  $
    G_r (omega) =
    w_"event"/(i omega + 1/tau_"AMPA") dot
    (E_e-macron(v)_r)/
    (i omega C_m + g_L + macron(g)_r). quad "(A9)"
  $

  Here $omega$ is angular frequency in rad/ms and $i$ is the imaginary unit. The
  transfer function $G_r(omega)$ tells us how strongly voltage responds to an
  input change at each frequency. It describes small changes around the steady
  state, not the full start-from-rest simulation.
  #link(<sec-synapse-plus-membrane-linearization>)[Synapse and membrane response]
  derives Equation A9.

  The experiment does not use instantaneous voltage: it averages voltage over
  the presentation. That averaging has its own frequency response,

  $ A_(T_"present") (omega) =
    (1-exp(-i omega T_"present"))/(i omega T_"present"). quad "(A10)" $

  which multiplies the synapse-and-membrane response to give the complete
  response from input rate to voltage feature:

  $ H_r (omega) = A_(T_"present") (omega) G_r (omega). quad "(A11)" $

  Here $A_(T_"present")(omega)$ describes averaging over presentation duration
  $T_"present"$, and $H_r(omega)$ describes the complete path from input-rate
  change to the averaged voltage feature.
  #link(<sec-finite-window-average>)[Averaging over the presentation] derives
  Equations A10 and A11.

  To generate Figure 3, we evaluated Equations A9 and A11 from
  #p.frequency_plot_bounds_hz.first() to #p.frequency_plot_bounds_hz.last() Hz at
  #p.frequency_response_rates_hz.map(str).join(", ") spikes/s for the nominal
  #p.nominal_probe_uS μS event strength. We converted frequency $f$ in Hz to
  angular frequency using $omega=2 pi f/1000$. To compare response shapes, we
  expressed every magnitude relative to the steady response at the lowest input
  rate:

  $
    M_X (f) = 20 log_10 (abs(X_r (2 pi f/1000))/
    abs(X_(r_"low") (0))). quad "(A12)"
  $

  Here $M_X(f)$ is response magnitude in decibels (dB). We used $X_r=G_r$ for
  Figure 3A and $X_r=H_r$ for Figure 3B. The variable $f$ is the frequency of
  the input change in Hz, and $r_"low"$ is the lowest steady input rate used as
  the reference.


  === Predict variation in the voltage feature

  An ideal Poisson event train has equal fluctuation power at every frequency.
  After subtracting its mean, its power across positive and negative frequencies
  on the millisecond time base is

  $ S_"in" (omega) = r_"input"/1000. quad "(A13)" $

  The filter scales that input power by $abs(H_r(omega))^2$. The predicted power
  spectrum of the voltage feature and its total variance are therefore

  $ S_z (omega)=abs(H_r (omega))^2 S_"in" (omega), quad "(A14)" $

  $
    "Var"_"linear" (z) = 1/(2 pi) integral_(-oo)^oo
    abs(H_r (omega))^2 S_"in" (omega) dif omega. quad "(A15)"
  $

  Here $S_"in"(omega)$ is the input power at each frequency,
  $S_z(omega)$ is the resulting feature power, and
  $"Var"_"linear"(z)$ is the predicted variance of the voltage feature.
  #link(<sec-poisson-spectrum-and-feature-variance>)[From Poisson events to
  feature variance] derives Equations A13–A15.

  We evaluated the integral in Equation A15 numerically on a logarithmically
  spaced frequency grid. We repeated the calculation with half as many points
  to check that the numerical result was stable.


  === Why the simpler theory misses the variation <sec-stationary-approximation-failure>

  The simpler theory first averages conductance and then calculates the voltage
  response, written $z(E[g])$. The simulation instead calculates a response for
  every random conductance history and then averages those responses, written
  $E[z(g)]$. Because voltage saturates, these two operations are not
  interchangeable. Here the approximate relationship is

  $ E[z(g)] < z(E[g]). $

  The steady theory also treats input as if it acted throughout the whole
  presentation. In the simulation, an event arriving near the end contributes
  to the average in Equation A5 for only a short time.

  More generally, the simulated feature distribution combines presentations
  with different event counts and, within each count, different event timings:

  $
    p_Z (z) = sum_(n=0)^oo P(N=n) p_(Z|N) (z|n). quad "(D2)"
  $

  Here $Z$ is the random voltage feature, $N$ is the event count, $P(N=n)$ is
  the probability of observing $n$ events, and $p_(Z|N)(z|n)$ is the
  distribution of feature values among presentations containing exactly $n$
  events.

  Equation A15 instead describes small fluctuations around one steady state. At
  low rates, a presentation with no events stays exactly at rest, a single large
  event produces a response that depends strongly on its arrival time, and
  multiple events interact through shunting and voltage saturation. This
  irregular mixture explains why the theoretical SD rose too sharply and
  peaked too early, especially for the strongest events.


  == Appendix: how the simpler theory is derived <sec-appendix-derivation-of-the-analytical-filter>

  === From events to steady conductance and voltage <sec-stationary-conductance-and-voltage>

  In the simulation, $s[k]$ records whether step $k$ contains an event. For the
  continuous-time calculation, we represent the same event sequence as
  $s(t)=sum_j delta(t-t_j)$: a train of instantaneous impulses at event times
  $t_j$. AMPA conductance then follows

  $ (d g)/(d t) = -g/tau_"AMPA" + w_"event" s(t). quad "(B1)" $

  Here $delta(t-t_j)$ is a Dirac impulse marking event $j$ at time $t_j$. Each
  impulse adds $w_"event"$ conductance, while existing conductance decays with
  time constant $tau_"AMPA"$.

  A steady input rate of $r$ spikes/s is $r/1000$ events per millisecond. To find
  the long-run mean conductance, we average Equation B1 and set its mean rate of
  change to zero:

  $
    0=-macron(g)_r/tau_"AMPA"+w_"event" r/1000,
  $

  Solving this equation gives Equation A6. At the corresponding steady membrane
  voltage, inward and outward current balance:

  $
    0=g_L (E_L-macron(v)_r)
      +macron(g)_r (E_e-macron(v)_r), quad "(B2)"
  $

  Rearranging the terms containing $macron(v)_r$ gives Equation A7.

  === Small changes through the synapse and membrane <sec-synapse-plus-membrane-linearization>

  We write each changing quantity as its steady value plus a small fluctuation:

  $
    g=macron(g)_r+delta g,
    quad v=macron(v)_r+delta v,
    quad s=r/1000+delta s.
  $

  Substituting these expressions into Equation B1 makes the steady terms cancel,
  leaving an equation for the conductance fluctuation:

  $ (d delta g)/(d t) = -delta g/tau_"AMPA" + w_"event" delta s. quad "(B3)" $

  In the frequency domain, differentiation with respect to time becomes
  multiplication by $i omega$. The conductance response is therefore

  $
    delta g(omega)=w_"event"/(i omega+1/tau_"AMPA") delta s(omega). quad "(B4)"
  $

  We apply the same decomposition to the membrane equation, Equation A4. The
  steady terms cancel through Equation B2. Because the theory assumes small
  fluctuations, it drops their product $delta g delta v$, which is second order.
  This leaves

  $
    C_m (d delta v)/(d t)
    =-(g_L+macron(g)_r)delta v
      +(E_e-macron(v)_r)delta g. quad "(B5)"
  $

  Transforming Equation B5 into the frequency domain and substituting Equation
  B4 gives Equation A9. Its denominator contains two filters: one from AMPA
  conductance and one from the membrane. The numerator contains the conductance
  added per event and the local voltage difference driving excitation.

  === Averaging over the presentation <sec-finite-window-average>

  After assuming that fluctuations are small, the feature in Equation A5 is
  simply their average voltage during the presentation:

  $ delta z_"feature" = 1/T_"present" integral_0^(T_"present") delta v(t) dif t.
    quad "(B6)" $

  Averaging gives equal weight, $1/T_"present"$, to every time from zero to
  $T_"present"$ and zero weight outside that interval. The frequency response of
  this rectangular averaging window is

  $
    A_(T_"present") (omega)=1/T_"present" integral_0^(T_"present") exp(-i omega t) dif t
    =(1-exp(-i omega T_"present"))/(i omega T_"present"),
  $

  which is Equation A10. Because the averaging happens after the membrane
  response, their two frequency responses multiply, giving Equation A11. The
  identity $1-exp(-i x)=2i exp(-i x/2)sin(x/2)$ shows that the magnitude is

  $
    abs(A_(T_"present") (omega))=
    abs(sin(omega T_"present"/2)/(omega T_"present"/2)),
  $

  With $T_s=T_"present"/1000$ s and $omega=2 pi f/1000$,

  $
    abs(A_(T_"present") (2 pi f/1000)) =
    abs(sin(pi f T_s)/(pi f T_s)). quad "(D1)"
  $

  This response becomes zero at $f=n/T_s$, where $n$ is any nonzero integer.
  At those frequencies, the presentation contains a whole number of cycles, so
  positive and negative parts cancel in the average. The repeated dips and
  rebounds come from this sinc-shaped averaging response; the synapse and
  membrane make the overall response fall at higher frequencies.

  === From Poisson events to feature variance <sec-poisson-spectrum-and-feature-variance>

  For ideal Poisson input at $r_"input"/1000$ events per millisecond, counts in
  separate time intervals are independent and the variance of each count equals
  its mean. After subtracting the mean event rate, the input autocovariance is

  $ C_(delta s) (ell)=(r_"input"/1000)delta(ell). quad "(B7)" $

  Here $C_(delta s)(ell)$ measures how the centred input relates to itself after
  a time lag $ell$. The Dirac delta $delta(ell)$ means that this ideal input is
  correlated with itself only at zero lag.

  The frequency transform of a Dirac delta is constant, which gives the flat
  input spectrum in Equation A13. A linear filter multiplies that spectrum by
  the squared magnitude of its frequency response, giving Equation A14. Adding
  the feature power across all frequencies then gives its variance:

  $
    "Var"(z)=R_z (0)=1/(2 pi) integral_(-oo)^oo S_z (omega) dif omega,
  $

  Substituting Equation A14 gives Equation A15. This result is exact for the
  simplified model with steady input and small linear fluctuations. It is only
  an approximation to the simulated system, which is nonlinear, begins each
  presentation from rest and contains a finite random number of events.

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
    [How does a pixel's input rate shape its average voltage? Compare direct simulation with the simpler theory across mean responses, variation, distributions and response speed.],
    preview-figures, json-inputs: ("exp081",),
  )
}

#let meta = meta + (assets: input-assets("exp081", inputs))
#let body = with-datasets("exp081", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-numbered-equations(body)
#let body = with-contents(body)
