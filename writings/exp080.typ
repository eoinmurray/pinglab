#import "/.demolab/lib.typ": cite, reference-list

#let meta = (
  title: "Filter-matched rate calibration for variable-rate PING training",
  date: "2026-08-05",
  description: "Empirical and analytical calibration of Poisson-driven, filter-matched MNIST features for conductance-based PING inputs.",
  collection: "gamma-gated-sparsity",
  status: "draft",
)

#let presentation-ms = 200
#let dt-ms = 0.1
#let calibration-rates-hz = (0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 10, 25)
#let decoder-rates-hz = (0.01, 0.05, 0.1, ..calibration-rates-hz)
#let probe-us = (0.6, 1.2, 2.4)
#let seeds = (42, 43, 44)
#let r = json("/artifacts/data/exp080/numbers.json")
#let m = json("/artifacts/data/exp080/step2_manifest.json")
#let diagnostic-k = m.diagnostic_draws_per_condition_per_seed
#let s3 = r.step3
#let response-distributions = json("/artifacts/data/exp080/response_distributions.json")
#let s4 = r.step4
#let s5 = json("/artifacts/data/exp080/step5_outcome.json")
#let s6 = json("/artifacts/data/exp080/step6_outcome.json")
#let decision = json("/artifacts/data/exp080/decision.json")
#let expanded = json("/artifacts/data/exp080/expanded_rate_outcome.json")
#let s3-nominal-low = s3.agreement_summaries.at(3)
#let s3-nominal-transition = s3.agreement_summaries.at(4)
#let s3-nominal-high = s3.agreement_summaries.at(5)
#let s4-nominal-low = s4.condition_records.at(3).comparison
#let s4-nominal-transition = s4.condition_records.at(4).comparison
#let s4-nominal-high = s4.condition_records.at(5).comparison
#let nominal-hundredth = expanded.held_out_evaluation.nominal_added_rate_rows.at(0)
#let nominal-twentieth = expanded.held_out_evaluation.nominal_added_rate_rows.at(1)
#let nominal-tenth = expanded.held_out_evaluation.nominal_added_rate_rows.at(2)
#let nominal-quarter = s6.nonlinear.at(18)
#let nominal-half = s6.nonlinear.at(19)
#let rounded(x, digits: 3) = str(calc.round(x, digits: digits))
#let pct(x) = rounded(100 * x, digits: 2) + "%"
#let calibration-rate-text = calibration-rates-hz.map(str).join(", ")
#let decoder-rate-text = decoder-rates-hz.map(str).join(", ")
#let probe-text = probe-us.map(str).join(", ")
#let seed-text = seeds.map(str).join(", ")
#let body = [
  == Abstract

  Poisson-encoded images are transformed by synaptic filtering, membrane
  integration, and finite observation windows before reaching a neural
  decoder. We characterized this transformation by passing MNIST pixel spike
  trains through the target AMPA and non-spiking conductance-based membrane.
  Sparse inputs produced zero-dominated, strongly skewed shot-noise features;
  with increasing spike count, overlapping responses formed smoother,
  approximately Gaussian distributions. A linear operating-point model
  reproduced the ordering of empirical means but overpredicted both feature
  mean and variability, showing that stationary linearization did not replace
  direct finite-window simulation. We therefore trained matched nonlinear and
  linear decoders on freshly simulated features across 0.01--25 Hz. On the
  official MNIST test set, nonlinear accuracy was reliably above chance from
  0.01 Hz and reliably exceeded 50% from 0.5 Hz. The 0.5 Hz practical floor was
  unchanged across 0.6, 1.2, and 2.4 μS probes, supporting 0.5--25 Hz for later
  variable-rate PING training.

  == Purpose and scope

  The question is:

  #quote(block: true)[At a given encoding rate, does the temporally filtered
    image still contain enough information for an ANN to recognize the digit?]

  The target PING architecture has 784 pixel channels feeding conductance-based
  excitatory cells. Its fixed synaptic and membrane equations define the
  features, without trained input, recurrent, or output weights. Matched linear
  and nonlinear decoders therefore measure decoder-relative accessibility, not
  an absolute information boundary or PING accuracy.

  == Methods

  1. *Prepared the dataset partitions.*
    MNIST provides an official 60,000-image training set and a separate
    10,000-image test set. We split the official training set into a 55,000-image
    training subset, used to fit decoder parameters, and a 5,000-image validation
    subset, evaluated after every epoch to select the best checkpoint.
    Validation also selected among the linear decoder's three L2 weight-decay
    values. Only after these choices and the evaluation protocol were frozen did
    we load the official test set for the psychometric curves and exploratory
    rate-range decision.

  2. *Generated and validated filter-matched pixel features.*

    Figure 1 illustrates the single-pixel feature dynamics.

    Every image used a #presentation-ms ms presentation and #dt-ms ms timestep. Decoder training
    and evaluation treated #decoder-rate-text Hz as a uniform categorical grid.
    For separate characterization, we constructed an *empirical response table*:
    a multidimensional lookup array of simulated single-pixel features z,
    indexed by random seed, probe conductance, encoding rate, uint8 pixel
    intensity, and stochastic draw. The table was used only to estimate and plot
    the response mean and variance at each condition, compare table sampling with fresh direct
    simulation, and assess the analytical approximation. It never generated
    decoder inputs. It covered the #calibration-rate-text Hz characterization
    grid, while decoder training and evaluation used fresh direct simulations
    at all 15 rates.
    Seeds #seed-text defined independent feature and table-generation runs.

    Each pixel drove an independent AMPA synapse and uncoupled non-spiking
    excitatory membrane. This captured synaptic decay, conductance-dependent
    integration, and within-window timing, but included no threshold, reset,
    recurrence, inhibition, shared noise, or trained synaptic weights.

    The complete feature path is

    $ S_i (t) -> g_i^"pix" (t) -> v_i (t) -> z_i -> "ANN". quad "(1)" $

    Here S#sub[i], g#super[pix]#sub[i], v#sub[i], and z#sub[i] are pixel i's spike
    train, AMPA conductance, subthreshold voltage, and time-averaged feature;
    t is time and ANN is the classifier.

    All dynamical equations used milliseconds. We approximated independent
    Poisson input in discrete time. At each timestep, pixel i could emit at most
    one event, drawn according to

    $ S_i (t) tilde "Bernoulli"(r x_i Delta t / 1000). quad "(2)" $

    Here r is the reported maximum-pixel encoding rate in spikes/s, x#sub[i] is
    normalized pixel intensity, division by 1000 reconciles seconds with the
    millisecond timestep Δt, and Bernoulli is an independent binary draw. Pixel i
    therefore had reported rate $r x_i$ Hz.

    We updated the AMPA conductance using

    $ g_i^"pix" (t) = beta_"AMPA" g_i^"pix" (t-1) + w_"probe" S_i (t). quad "(3)" $

    $ beta_"AMPA" = exp(-(Delta t) / tau_"AMPA"). quad "(4)" $

    Here β#sub[AMPA] is one-timestep conductance retention,
    τ#sub[AMPA] = 2 ms is the AMPA decay time, w#sub[probe] is the AMPA conductance increment produced by one
    pixel spike. Other symbols follow
    Equation 2. The primary 1.2 μS
    probe is the target PING network architecture's nominal mean initial pixel-to-excitatory
    conductance; 0.6 and 2.4 μS test half and double scale for comparison.

    We integrated the membrane without thresholding or resetting:

    $ C_E (d v_i) / (d t) = g_"L,E" (E_L - v_i) + g_i^"pix" (t) (E_e - v_i). quad "(5)" $

    $ tau_"eff",i (t) = C_E / (g_"L,E" + g_i^"pix" (t)). quad "(6)" $

    Here C#sub[E] is excitatory capacitance; g#sub[L,E] is leak conductance;
    E#sub[L] and E#sub[e] are leak and AMPA reversal potentials; and
    τ#sub[eff,i] is the instantaneous conductance-dependent time constant. Other
    symbols follow Equation 1.

    Voltage began at E#sub[L] and conductance at zero. After one presentation,

    $ z_i = 1 / T integral_0^T (v_i (t) - E_L) dif t. quad "(7)" $

    where T is presentation duration and z#sub[i] is mean baseline-subtracted
    voltage over the complete window, not terminal voltage. Threshold and reset
    were disabled, so “subthreshold” refers to the non-spiking membrane
    equations rather than a constraint on attained voltage. Neither the feature
    nor the decoder input included rate, so sparse inputs retained their weaker,
    noisier signal.

    Figure 1 shows the spike train, conductance response, membrane response, and
    persistence of timing information after window averaging.

  3. *Characterized the empirical response moments.*

    Figures 2 and 3 show the empirical response moments and how the full
    response distribution changes with input rate.

    The stored table contained three seeds, three conductances, the twelve
    #calibration-rate-text Hz rates, all 256 uint8 intensities, and
    #m.selected_K simulations per condition and seed. Figure 2 used the first
    #diagnostic-k draws from each seed as a practical plotting subset. This
    subset size was neither a convergence gate nor an ANN-training requirement.
    It estimated the mean response

    $ hat(mu)_z (x, r, w_"probe") = 1 / K sum_(k=1)^K z^(k). quad "(8)" $

    and response variance

    $ hat(sigma)_z^2 (x, r, w_"probe") = 1 / (K - 1) sum_(k=1)^K (z^(k) - hat(mu)_z)^2. quad "(9)" $

    Here x, r, and w#sub[probe] are pixel-intensity, rate, and probe conductance;
    z#super[(k)] is draw k; K is the draw count.

    Figure 2 plots these estimated response means and SDs across the calibrated
    rate--intensity conditions.

    Figure 3 plots all #m.selected_K simulations from each of the three seeds at
    maximum pixel intensity, the nominal 1.2 μS probe, and representative low,
    intermediate, and high rates. A Gaussian with the same empirical mean and
    SD was overlaid at each rate to assess distribution shape.

    In contrast to this diagnostic lookup array, ANN decoder presentation n used a
    newly generated spike train and direct synapse--membrane simulation:

    $ S_i^(n)(t) -> g_i^"pix,n"(t) -> v_i^(n)(t) -> z_i^(n) -> "ANN". quad "(10)" $

    Repetition sampled the full response distribution at each condition without
    an intermediate noise model.

  4. *Derived the analytical linear-filter response.*

    Figures 4 and 5 show the analytical transfer functions and the analytical
    moments over input drive, respectively.

    We calculated a local linear approximation to the synapse, membrane, and finite averaging window.
    This diagnostic was not used to select a training range. Appendix A derives the equations in this section.

    For reported encoding rate r and normalized intensity x, the analytical
    rate was $lambda = r x$ spikes/s. Its stationary
    mean conductance is

    $ overline(g)_lambda = lambda w_"probe" tau_"AMPA" / 1000. quad "(11)" $

    and the deterministic operating-point voltage is

    $
      overline(v)_lambda = (g_"L,E" E_L + overline(g)_lambda E_e) /
      (g_"L,E" + overline(g)_lambda). quad "(12)"
    $

    At this operating point, voltage is constant across the presentation, so
    applying the baseline subtraction and time average in Equation 7 gives the
    analytical mean feature

    $
      mu_"linear"(z) = overline(v)_lambda - E_L. quad "(12a)"
    $

    Here $overline(g)_lambda$ is stationary mean conductance,
    $overline(v)_lambda$ is the voltage obtained by replacing the fluctuating
    conductance with $overline(g)_lambda$, and
    μ#sub[linear] (z) is the predicted mean of the final feature z. Other symbols
    follow Equations 3--7.

    Linearizing the synapse and membrane around that operating point gives

    $
      G_lambda(omega) = w_"probe" / (i omega + 1 / tau_"AMPA") dot
      (E_e - overline(v)_lambda) /
      (i omega C_E + g_"L,E" + overline(g)_lambda). quad "(13)"
    $

    Here G#sub[λ] (ω) is the local synapse-plus-membrane transfer function; ω is
    angular frequency in rad/ms.

    The finite averaging window contributes

    $ A_T(omega) = (1 - exp(-i omega T)) / (i omega T). quad "(14)" $

    $ H_lambda(omega) = A_T(omega) G_lambda(omega). quad "(15)" $

    Here A#sub[T] (ω) is the duration-T averaging response; H#sub[λ] (ω) is the
    complete response from input fluctuation to averaged feature and T is
    presentation duration in ms.

    Figure 4, Panel A plots $G_lambda$ from Equation 13, and Panel B plots the
    complete window-averaged $H_lambda$ from Equation 15.

    Appendix A.4 derives the input spectrum from the discrete Bernoulli
    encoder, its continuous-time Poisson limit, and the resulting
    autocovariance. The centred ideal Poisson input has the constant, or white,
    two-sided spectrum

    $ S_"in"(omega) = lambda / 1000. quad "(16)" $

    $ S_z(omega) = abs(H_lambda(omega))^2 S_"in"(omega). quad "(17)" $

    $
      "Var"_"linear"(z) = 1 / (2 pi) integral_(-oo)^oo
      abs(H_lambda(omega))^2 S_"in"(omega) dif omega. quad "(18)"
    $

    Here λ is in spikes/s and division by 1000 accounts for the millisecond time base;
    S#sub[in] (ω) and S#sub[z] (ω) are input and predicted output power
    spectral densities; |H#sub[λ] (ω)|#super[2] is transmitted noise power,
    and Var#sub[linear] (z) is predicted feature variance.

    We also calculated finite-presentation moments from the same initial state as
    the empirical simulations: $g_0 = 0$, $v_0 = E_L$, and a zero accumulated
    feature. The deterministic conductance and voltage means were advanced for
    all #presentation-ms ms with the same discrete decay-then-add synapse and
    exponential membrane updates as Equations 3 and 4. Their time average gives
    the transient mean $mu_"trans"(z)$.

    For variability, we linearized each discrete update along that
    time-dependent deterministic trajectory. With state
    $bold(x)_k = (g_k, v_k, y_k)$, where $y_k$ accumulates $v_k - E_L$, its
    start-from-rest covariance obeys

    $
      P_(k+1) = J_k P_k J_k^T + p(1-p) bold(b)_k bold(b)_k^T,
      quad P_0 = bold(0). quad "(18a)"
    $

    Here $J_k$ is the Jacobian of the decay-then-add synapse, exponential
    membrane, and accumulator update; $bold(b)_k$ is that update's sensitivity
    to one input event; and $p = lambda Delta t / 1000$ is the exact Bernoulli
    event probability per timestep. The finite-window moments are

    $
      mu_"trans"(z) = 1/N sum_(k=1)^N (overline(v)_k - E_L),
      quad sigma_"trans"(z) = sqrt((P_N)_(y y)) / N. quad "(18b)"
    $

    Equation 18a therefore includes the quiet initial condition, synaptic and
    membrane settling, event-count variance, event timing, and the finite
    accumulator. It remains a local covariance approximation because the
    conductance-dependent membrane update is linearized around the transient
    mean trajectory.

    At every rate, intensity, and conductance, we set $lambda = r x$, selected
    the corresponding transient and stationary models, and compared their
    predictions with the empirical response table without new stochastic
    simulation. Figure 5 plots the transient mean and SD from Equations 18a and
    18b as solid curves, and retains the stationary mean from Equation 12a and
    stationary SD from Equation 18 as dashed references.

  5. *Constructed and compared complete MNIST feature images.*

    Figure 6 compares the resulting complete feature images.

    For each uint8 pixel, the diagnostic sampler used its exact 0--255 intensity index and an
    empirical draw with independent deterministic pixel and image
    streams. A fresh direct simulation provided the comparison. This used
    training images 0--15, never validation or test images, and covered all
    #s4.dataset.image_shape.at(1) × #s4.dataset.image_shape.at(2) pixels, eight
    replicates, three conductances, and representative sparse, intermediate,
    and upper-grid conditions at 0.25, 3, and 25 Hz.

    Figure 6 places the original MNIST image, an empirical-table sample, and a
    fresh direct simulation side by side.

  6. *Trained mixed-rate decoders.*

    Figure 7 shows the validation histories used for epoch selection.

    Each simulated image produced a feature vector $bold(z) in RR^784$, containing one time-averaged membrane-voltage
    displacement per pixel. The primary ANN transformed this vector through a
    fully connected 1,024-unit hidden layer,

    $
      bold(h) = "ReLU"(W_h bold(z) + bold(b)_h), quad
      bold(s) = W_o bold(h) + bold(b)_o,
    $

    where $bold(h)$ is hidden activation, $bold(s) in RR^10$ contains one logit
    per digit, and elementwise $"ReLU"(x) = max(0, x)$ retains positive inputs
    and zeros negative ones. Both weight matrices and bias vectors were
    trainable; prediction used $arg max_c s_c$. The 784--1,024--10 network had
    no convolution, recurrence, normalization, dropout, or pretrained weights.

    The matched linear decoder omitted the hidden layer and computed

    $ bold(s) = W_"lin" bold(z) + bold(b)_"lin". $

    Its 784 × 10 weights and ten biases were trainable. “Linear” described the
    feature-to-logit map, not fitting: both models used backpropagation. Linear
    performance measured directly accessible digit evidence; the ANN advantage
    measured the benefit of its nonlinear representation. Neither received rate.

    Both models were fitted with ten-class cross-entropy. For logits
    $bold(s)$ and true digit label $y$, the loss for one image was

    $
      cal(L)_"CE"(bold(s), y)
      = -log(exp(s_y) / sum_(c=0)^9 exp(s_c)).
    $

    Backpropagation minimized this penalty for low correct-class probability.

    Each image presentation independently sampled one of the 15 registered
    rates with probability 1/15. That maximum rate applied to the complete
    image, while each pixel was scaled by its intensity. Each of the 55,000
    training images appeared once per epoch for 15 epochs, with a newly sampled
    rate and fresh spike trains on every presentation. The same 5,000 validation
    images were evaluated after every epoch using new reproducibly seeded rate
    assignments and spike trains.

    Within a training run, the ANN and linear models received the same directly
    simulated feature batches. Adam used learning rate 0.001 and batch size 256.
    The ANN used no weight decay, and validation accuracy selected its best
    epoch. We also trained three otherwise identical linear decoders with L2
    weight decays 10#super[-5], 10#super[-4], and 10#super[-3]. The candidate with the best
    validation accuracy was selected. The nonlinear architecture and absence of
    weight decay were fixed rather than tuned. Test accuracy played no role in
    these choices.

    Seeds #seed-text controlled parameter initialization and stochastic training
    and validation features, so they represented complete training replicates.
    Every conductance and seed had a separate decoder; runs were not pooled
    during training.

    Figure 7 plots the validation histories from which the best epochs were
    selected.

  7. *Evaluated frozen official-test psychometric curves and selected the rate
    range.*

    Figure 8 shows the official-test psychometric curves and derived thresholds.

    After validation had fixed the best epochs and linear weight decay, we loaded the same official 10,000 MNIST test images for every condition. At
    each rate and conductance, we encoded every image three times with newly
    simulated spike trains. Within one rate, conductance, and encoding draw, all
    three trained decoder seeds received exactly the same feature vectors. At
    the nominal 1.2 μS conductance, the selected ANN and linear decoder also
    shared those features. Different conductances used different spike-train
    realizations, and discarded linear weight-decay candidates were not tested.
    Accuracy at each rate was the mean across test images, the three encodings,
    and the three trained decoders. The primary curve was

    $ A_r (r) = P("correct" | r, "mixed-rate nonlinear decoder"). quad "(19)" $

    Here A#sub[r] is official-test nonlinear-ANN accuracy, P is
    correct-classification probability, and r is rate. The linear decoder was
    diagnostic.

    We estimated uncertainty separately at every tested rate and conductance;
    rates were never mixed in this calculation. For each condition, we created
    2,000 alternative versions of the recorded evaluation. One version was
    formed by selecting 10,000 entries from the list of test images, three from
    the list of encoding draws, and three from the list of trained decoders.
    After each selection the chosen item remained available, so an image, draw,
    or decoder could appear several times while another could be absent. This is
    sampling “with replacement.” We calculated accuracy for each alternative
    version. L#sub[r] was the fifth percentile of those 2,000 accuracies: after
    sorting them, 95% were at least L#sub[r]. The decoder-relative edge was

    $ r_"decode" = "lowest " r in cal(R) " satisfying " L_r (r) > 1 / N_"class". quad "(20)" $

    The practical training floor was

    $ r_"train" = "lowest " r in cal(R) " satisfying " L_r (r) >= a_"use". quad "(21)" $

    Here $cal(R)$ is the tested grid, N#sub[class] the class count,
    r#sub[decode] the lowest rate reliably above chance, a#sub[use] the 50%
    useful-accuracy target, and r#sub[train] its first reliable crossing. The
    10% chance criterion corresponds to uniform guessing across ten classes. Primary
    thresholds used the 1.2 μS ANN; the linear decoder and other conductances
    measured sensitivity. The 784-to-10 decoder without a hidden layer was
    evaluated on the test set only at the nominal 1.2 μS conductance, not at 0.6
    or 2.4 μS. Accuracies were measured only at the 15 listed rates; no values
    between adjacent rates were estimated or treated as data. We used the
    resulting r#sub[train] as the lower bound of the later PING training range
    and the highest tested rate, 25 Hz, as its upper bound. The 50% criterion was
    a pragmatic screen for substantial digit information, not a theoretical
    information boundary.

    Figure 8 plots the official-test psychometric curves and the thresholds
    obtained from this procedure.

  #block(breakable: false)[
    == Results

    === Filter-matched feature generation

    We first asked whether the feature retained the timing imposed by the AMPA
    synapse and finite presentation window. A single spike was placed at
    different times within a #presentation-ms ms presentation, passed through
    the uncoupled conductance and membrane model, and reduced to the
    time-averaged feature z.

    #figure(
      image(
        "/artifacts/data/exp080/probe_dynamics.svg",
        width: 100%,
        alt: "Input spikes, AMPA conductance, membrane voltage, and presentation-averaged feature across spike times.",
      ),
      caption: [Finite-window timing sensitivity. Panel A places one input spike
        at 20 or 180 ms. Panels B and C pass those inputs through the registered
        AMPA synapse and subthreshold membrane. Panel D repeats the simulation
        across 101 single-spike times and plots the presentation-averaged feature
        z. Conductance and voltage relax after each spike; z falls for later
        spikes because the fixed 200 ms window truncates more of the response.],
    )
  ]

  === Empirical response moments

  We next characterized the magnitude and variability of z under stochastic
  encoding across all 256 pixel intensities, 12 calibration rates, and three
  probe conductances. The stored table contained #m.selected_K simulations for
  each condition and random seed. For Figure 2, we chose the first
  #diagnostic-k simulations per seed as a smaller practical plotting subset.
  Combining the three independent seeds gave
  #(diagnostic-k * seeds.len()) feature values from which we calculated one mean
  and sample SD for each plotted condition.

  #figure(
    image(
      "/artifacts/data/exp080/response_library.png",
      width: 100%,
      alt: "Empirical feature mean and standard deviation against expected input spikes for three probe conductances.",
    ),
    caption: [Mean and variability of the simulated single-pixel feature. We
      evaluated 9,216 conditions: 256 pixel intensities × 12 encoding rates ×
      three probe conductances. For each condition, we selected #diagnostic-k
      simulations from each of three independent random seeds, giving
      #(diagnostic-k * seeds.len()) values of feature z. Each point in Panel A is
      the mean of those #(diagnostic-k * seeds.len()) values; the corresponding
      point in Panel B is their standard deviation. Thus, each panel contains
      9,216 points. The #diagnostic-k simulations per seed were chosen as a
      practical descriptive subset of the available simulations, not through a
      statistical convergence test. The horizontal axis is the expected number
      of input spikes during the #presentation-ms ms presentation: encoding
      rate × normalized pixel intensity × 0.2 s. Colour
      identifies probe conductance. Mean response increases with expected spike
      count. Variability initially rises as spike counts and times diversify,
      then can plateau or fall when averaging, shunting, and voltage saturation
      limit additional excursions.],
  )

  ==== Distribution shape across rates

  To distinguish sparse shot noise from a high-count approximation, we examined
  the complete recorded z distribution for a maximum-intensity pixel at the
  nominal 1.2 μS probe. The three rates give 0.05, 0.6, and 5 expected spikes
  during one #presentation-ms ms presentation, where
  $E[N] = lambda T / 1000 = r x T / 1000$.

  #figure(
    image(
      "/artifacts/data/exp080/response_distributions.svg",
      width: 100%,
      alt: "Feature distributions at low, intermediate, and high input rates compared with Gaussians having the same means and standard deviations.",
    ),
    caption: [Transition from sparse shot noise toward a Gaussian-like feature
      distribution. Each panel contains #response-distributions.records.at(0).sample_count
      recorded z values: #m.selected_K simulations × three seeds for pixel
      intensity 255 and the nominal 1.2 μS probe. Bars show empirical
      probability per bin on a logarithmic axis; the red dashed curve is a
      Gaussian with the same empirical mean and SD. Panel A, at 0.25 Hz, is
      dominated by no-spike presentations and separated responses to rare spike
      counts and times. Panel B, at 3 Hz, remains strongly non-Gaussian because
      more than half of presentations contain no spike. Panel C, at 25 Hz,
      contains about five expected spikes per presentation; overlapping filtered
      responses form a much smoother distribution that approaches the matched
      Gaussian, while conductance nonlinearity and finite-window timing leave
      visible deviations.],
  )

  === Analytical linear-filter response

  ==== Analytical frequency response

  We then evaluated the linearized synapse-plus-membrane response and its
  #presentation-ms ms window-averaged form from 0.1--200 Hz at low,
  intermediate, and high operating rates. These curves were calculated from
  Equation 13 ($G_lambda$) and Equation 15 ($H_lambda$), rather than from
  simulated traces.

  #figure(
    image(
      "/artifacts/data/exp080/linear_filter.svg",
      width: 100%,
      alt: "Analytical Bode magnitudes for the synapse-plus-membrane response and the response after 200 millisecond averaging at three input rates.",
    ),
    caption: [Analytically calculated linearized frequency response. All curves
      use maximum pixel intensity $x = 1$ and the nominal probe conductance
      $w_"probe" = 1.2$ μS. The three operating rates are therefore
      $lambda = 0.25$, 3, and 25 spikes/s; each curve is the corresponding local
      member of the Equation 13 family. The functions contain no simulated
      traces. Frequency f is a small sinusoidal modulation around each operating
      rate. Panel A
      evaluates Equation 13 ($G_lambda$) from 0.1--200 Hz, with
      $omega = 2 pi f / 1000$ rad/ms.
      Gain is

      $
        20 log_10 (abs(G_lambda(2 pi f / 1000)) / abs(G_(lambda_"low")(0))),
      $

      relative to the 0.25 Hz DC response. Thus the black curve begins near 0
      dB; higher mean conductance lowers the red and cyan DC gains. Panel B plots
      Equation 15 ($H_lambda = A_T G_lambda$) after the analytical
      #presentation-ms ms rectangular-window average.

      Panel A combines an AMPA low-pass pole at $1 / tau_"AMPA"$ with a membrane
      pole set by effective conductance and capacitance, suppressing fast
      modulation. Greater drive shunts and depolarizes the membrane, reducing
      excitatory driving force and separating the curves.

      Panel B adds the averaging magnitude. Writing the presentation duration
      in seconds as $T_s = #presentation-ms / 1000$ s,
      $abs(A_T(2 pi f / 1000)) = abs(sin(pi f T_s) / (pi f T_s))$. It is one at zero
      frequency and zero at $f = n / T_s$ for nonzero integer n. Integer cycles
      therefore cancel at 5, 10, 15 Hz, and so
      on. Magnitude folds alternating sinc signs into positive lobes; the two
      low-pass terms attenuate successive lobes into the falling envelope.],
  )

  ==== Analytical moments over input drive

  We next plotted the analytical moments over the same expected-input-spike
  coordinate as the empirical response in Figure 2. This exposes the predicted
  scale and curvature before collapsing each condition into a predicted-versus-
  empirical point.

  #figure(
    image(
      "/artifacts/data/exp080/linear_filter_drive_response.svg",
      width: 100%,
      alt: "Analytical and empirical feature means and standard deviations versus expected input spikes for three probe conductances.",
    ),
    caption: [Transient and stationary analytical response moments over input drive. The horizontal axis
      matches Figure 2: expected input spikes are $lambda T / 1000 = r x T /
      1000$. Panel A shows mean feature; Panel B shows feature SD. Solid curves
      are the start-from-rest transient predictions from Equations 18a and 18b;
      dashed curves are the stationary references from Equations 12a and 18;
      faint points are the corresponding empirical values from Figure 2.
      Black, red, and cyan denote probe conductances 0.6, 1.2, and 2.4 μS.],
  )

  In Panel A, starting the deterministic trajectory from rest lowers the solid
  transient prediction relative to the dashed stationary operating point. The
  remaining overprediction is expected: replacing random conductance by its
  mean ignores the concavity caused by voltage saturation, so approximately
  $E[v(g)] < v(E[g])$. The gap grows with event size, which is why equal
  expected spike counts do not collapse the three probe curves.

  In Panel B, the start-from-rest covariance modestly changes the curve but does
  not recover the empirical shape. Both analytical variants propagate only
  local, small conductance fluctuations around a deterministic trajectory.
  Sparse large synaptic jumps instead produce a strongly non-Gaussian mixture
  of no-event, early-event, and late-event responses; shunting and saturation
  then act separately on those paths. A first-order covariance cannot represent
  that mixture, so its variance maximum remains displaced even after the finite
  initial condition is handled correctly.

  === Complete feature images

  We then tested whether independently generated single-pixel responses formed
  recognizable complete images. For the same MNIST training images and nominal
  probe, we compared the original pixels with features sampled from the
  empirical response table and with fresh direct simulations at representative
  low, intermediate, and high rates.

  #figure(
    image(
      "/artifacts/data/exp080/feature_images.png",
      width: 100%,
      alt: "At three rates, rows compare an MNIST training image, an empirical response-table sample, and a fresh direct simulation.",
    ),
    caption: [Training-image feature comparison. Rows show an original MNIST
      training image, a response-table sample, and a fresh direct
      simulation at 0.25, 3, and 25 Hz for the nominal 1.2 μS probe, using common
      0--65 mV limits. Increasing rate reveals the intensity pattern as spatial
      signal grows relative to independent sampling noise.],
  )

  At 0.25 Hz, median image-level mean differences between response-table and
  direct features were
  #pct(s4.condition_records.at(0).comparison.metrics.image_mean_relative_difference_median),
  #pct(s4.condition_records.at(3).comparison.metrics.image_mean_relative_difference_median),
  and #pct(s4.condition_records.at(6).comparison.metrics.image_mean_relative_difference_median)
  at 0.6, 1.2, and 2.4 μS. Median variance differences were
  #pct(s4.condition_records.at(0).comparison.metrics.image_variance_relative_difference_median),
  #pct(s4.condition_records.at(3).comparison.metrics.image_variance_relative_difference_median),
  and #pct(s4.condition_records.at(6).comparison.metrics.image_variance_relative_difference_median),
  and spatial correlations were
  #rounded(s4.condition_records.at(0).comparison.metrics.spatial_mean_correlation),
  #rounded(s4.condition_records.at(3).comparison.metrics.spatial_mean_correlation),
  and #rounded(s4.condition_records.at(6).comparison.metrics.spatial_mean_correlation).

  At 3 Hz, the corresponding median mean differences were
  #pct(s4.condition_records.at(1).comparison.metrics.image_mean_relative_difference_median),
  #pct(s4.condition_records.at(4).comparison.metrics.image_mean_relative_difference_median),
  and #pct(s4.condition_records.at(7).comparison.metrics.image_mean_relative_difference_median);
  median variance differences were
  #pct(s4.condition_records.at(1).comparison.metrics.image_variance_relative_difference_median),
  #pct(s4.condition_records.at(4).comparison.metrics.image_variance_relative_difference_median),
  and #pct(s4.condition_records.at(7).comparison.metrics.image_variance_relative_difference_median);
  and spatial correlations were
  #rounded(s4.condition_records.at(1).comparison.metrics.spatial_mean_correlation),
  #rounded(s4.condition_records.at(4).comparison.metrics.spatial_mean_correlation),
  and #rounded(s4.condition_records.at(7).comparison.metrics.spatial_mean_correlation).

  At 25 Hz, median mean differences were
  #pct(s4.condition_records.at(2).comparison.metrics.image_mean_relative_difference_median),
  #pct(s4.condition_records.at(5).comparison.metrics.image_mean_relative_difference_median),
  and #pct(s4.condition_records.at(8).comparison.metrics.image_mean_relative_difference_median);
  median variance differences were
  #pct(s4.condition_records.at(2).comparison.metrics.image_variance_relative_difference_median),
  #pct(s4.condition_records.at(5).comparison.metrics.image_variance_relative_difference_median),
  and #pct(s4.condition_records.at(8).comparison.metrics.image_variance_relative_difference_median);
  and spatial correlations were
  #rounded(s4.condition_records.at(2).comparison.metrics.spatial_mean_correlation),
  #rounded(s4.condition_records.at(5).comparison.metrics.spatial_mean_correlation),
  and #rounded(s4.condition_records.at(8).comparison.metrics.spatial_mean_correlation).

  === Mixed-rate decoder training

  We trained the nonlinear ANN and matched linear decoder on fresh directly
  simulated features while sampling all 15 rates equally. After every epoch,
  the fixed 5,000-image validation subset was freshly encoded and evaluated;
  these histories selected the best epoch for each trained run.

  #figure(
    image(
      "/artifacts/data/exp080/step5_training_history.svg",
      width: 100%,
      alt: "Validation accuracy across fifteen epochs for nonlinear and linear decoders at three probe conductances.",
    ),
    caption: [Mixed-rate validation histories. Each panel plots mean validation
      accuracy across decoder seeds, with their range shaded, under equal
      sampling of all 15 rates from 0.01--25 Hz and fresh direct simulations.
      Curves rise then flatten as class structure is acquired; conductances
      overlap because response scaling preserves similar spatial evidence.],
  )

  At the nominal probe, seeds 42, 43, and 44 respectively selected nonlinear epochs
  #s5.records.at(3).selected_nonlinear_epoch,
  #s5.records.at(4).selected_nonlinear_epoch, and
  #s5.records.at(5).selected_nonlinear_epoch, with accuracies
  #pct(s5.records.at(3).selected_nonlinear_validation_accuracy),
  #pct(s5.records.at(4).selected_nonlinear_validation_accuracy), and
  #pct(s5.records.at(5).selected_nonlinear_validation_accuracy). The matched
  linear decoders reached
  #pct(s5.records.at(3).selected_linear_validation_accuracy),
  #pct(s5.records.at(4).selected_linear_validation_accuracy), and
  #pct(s5.records.at(5).selected_linear_validation_accuracy).

  === Official-test psychometric curves and rate range

  After model selection was complete, we evaluated the frozen decoders on the
  official 10,000-image MNIST test set at each rate. Every image was encoded
  three times, and accuracy was averaged across those encodings and three
  independently trained decoder runs before applying the above-chance and 50%
  practical criteria.

  #figure(
    image(
      "/artifacts/data/exp080/psychometric.svg",
      width: 100%,
      alt: "Official-test accuracy against encoding rate for linear and nonlinear decoders and three probe conductances.",
    ),
    caption: [Official-test psychometric curves. Each point pools the official
      10,000-image test partition across three fresh simulation draws and three
      independently trained runs. Within each rate, conductance, and draw, runs
      share feature vectors; different conductances use different spike draws.
      Accuracies, not predictions, are averaged. Bands bootstrap images, draws,
      and runs; horizontal rules mark 10% uniform guessing and the 50% pragmatic
      criterion. Accuracy rises as more pixels spike within 200 ms. Because each
      conductance has separately retrained decoders, overlap shows recoverability
      after retraining, not robustness of one fixed decoder.],
  )

  At the nominal probe, accuracy was #pct(nominal-hundredth.accuracy) at 0.01
  Hz, #pct(nominal-twentieth.accuracy) at 0.05 Hz, and
  #pct(nominal-tenth.accuracy) at 0.1 Hz. Their one-sided lower bounds were
  #pct(nominal-hundredth.lower_95_one_sided),
  #pct(nominal-twentieth.lower_95_one_sided), and
  #pct(nominal-tenth.lower_95_one_sided), respectively. Thus the lowest tested
  rate exceeded 10% chance, but the ultra-sparse range remained below the 50%
  practical criterion. At 0.01
  Hz, a maximum-intensity pixel emitted only
  #rounded(decoder-rates-hz.at(0) * presentation-ms / 1000, digits: 3) expected
  spikes per presentation, so this result reflects rare events and decoder
  priors rather than a visually complete digit.

  At 0.25 Hz, nominal nonlinear accuracy was #pct(nominal-quarter.accuracy),
  with a #pct(nominal-quarter.lower_95_one_sided) one-sided lower bound. At 0.5
  Hz it was #pct(nominal-half.accuracy), with a
  #pct(nominal-half.lower_95_one_sided) lower bound. The nominal linear decoder
  also first crossed the 50% lower-bound criterion at
  #s6.thresholds.at("linear_1.2").r_train_hz Hz. The reported
  #decision.r_train_hz Hz floor is
  the lowest tested rate satisfying the criterion, not an estimate that the
  continuous crossing occurs exactly at 0.5 Hz.

  === Training-range decision

  The nonlinear decoder was reliably above chance from
  #decision.r_decode_hz Hz, and its practical floor was
  #decision.r_train_hz Hz. The practical floor remained
  #decision.conductance_floors_hz.at("0.6") Hz at 0.6 μS,
  #decision.conductance_floors_hz.at("1.2") Hz at 1.2 μS, and
  #decision.conductance_floors_hz.at("2.4") Hz at 2.4 μS. We therefore
  recommend rates from #decision.recommendation.floor_hz Hz to
  #decision.recommendation.ceiling_hz Hz for later variable-rate PING training.
  This decoder-relative range is neither an absolute information limit nor a
  PING accuracy result. Expanded retraining and evaluation cost
  #rounded(expanded.compute.expansion_exact_cost_usd) USD on Modal, bringing
  cumulative exp080 compute to
  #rounded(expanded.compute.cumulative_exact_cost_usd) USD.

  == Limitations

  This measured decoder-relative accessibility, not PING performance, and each
  conductance used separately retrained decoders. Because the official test
  partition informed the exploratory rate decision, confirmation requires new
  held-out data. The response table omitted the three lowest rates, although
  direct-simulation training and test evaluation included them equally. Three
  training runs and three noise draws provide limited between-run and
  between-draw sampling.

  == Relation to prior work

  Filtered conductance shot noise can violate fixed-time-constant models#cite(1),
  and Gaussian approximations can miss skewed responses#cite(2), supporting the
  empirical response table. Decoder performance is not absolute neural
  information#cite(3), and linear and nonlinear decoders can differ#cite(4),
  motivating a decoder-relative edge and both decoder diagnostics.

  #reference-list((
    (
      text: [Wolff & Lindner: _Mean, Variance, and Autocorrelation of Subthreshold Potential Fluctuations Driven by Filtered Conductance Shot Noise_. Neural Computation, 2010.],
      doi: "10.1162/neco.2009.02-09-958",
    ),
    (
      text: [Brigham & Destexhe: _Nonstationary Filtered Shot-Noise Processes and Applications to Neuronal Membranes_. Physical Review E, 2015.],
      doi: "10.1103/PhysRevE.91.062102",
    ),
    (
      text: [Quian Quiroga & Panzeri: _Extracting Information from Neuronal Populations: Information Theory and Decoding Approaches_. Nature Reviews Neuroscience, 2009.],
      doi: "10.1038/nrn2578",
    ),
    (
      text: [Warland, Reinagel & Meister: _Decoding Visual Information From a Population of Retinal Ganglion Cells_. Journal of Neurophysiology, 1997.],
      doi: "10.1152/jn.1997.78.5.2336",
    ),
  ))

  == Appendix A: linear-filter derivation

  This appendix derives Equations 11--18 from the same AMPA synapse and
  subthreshold membrane used by the empirical probe. It replaces the finite
  Bernoulli drive with a continuous-time Poisson point process and linearizes
  the membrane around the mean drive at each calibration condition.

  === A.1 Stationary operating point

  Represent pixel i's idealized spike train and synaptic conductance by

  $
    s_i(t) = sum_k delta(t - t_k), quad
    (d g_i(t)) / (d t) = -g_i(t) / tau_"AMPA" + w_"probe" s_i(t). quad "(A1)"
  $

  Here time is measured in ms; s#sub[i] (t) is a sum of Dirac impulses;
  t#sub[k] is spike k's time; δ is
  the Dirac delta; g#sub[i] (t) is AMPA conductance;
  τ#sub[AMPA] = 2 ms is its decay time;
  w#sub[probe] is conductance added per spike; and t is time. Equation A1 is the
  continuous-time counterpart of the decay-then-add update in Equation 3.

  For stationary Poisson rate λ in spikes/s, the spike train contributes an
  expected λ/1000 events per millisecond and the mean conductance satisfies

  $
    0 = -overline(g)_lambda / tau_"AMPA" + w_"probe" lambda / 1000, quad
    overline(g)_lambda = lambda w_"probe" tau_"AMPA" / 1000. quad "(A2)"
  $

  Here λ is encoding rate times pixel-intensity, and
  $overline(g)_lambda$ is stationary mean conductance. The zero on the left states that
  the mean no longer changes with time.

  To define the deterministic operating point, replace the fluctuating
  conductance by $overline(g)_lambda$ and set the membrane derivative in Equation 5 to
  zero:

  $
    0 = g_"L,E" (E_L - overline(v)_lambda) + overline(g)_lambda (E_e - overline(v)_lambda), quad
    overline(v)_lambda = (g_"L,E" E_L + overline(g)_lambda E_e) / (g_"L,E" + overline(g)_lambda). quad "(A3)"
  $

  Here $overline(v)_lambda$ is the deterministic mean-conductance operating-point
  voltage, not the exact expectation of the stochastic voltage;
  g#sub[L,E] is leak conductance; and E#sub[L] and E#sub[e] are leak and AMPA
  reversal potentials. Other symbols follow Equation A2.

  === A.2 Local synapse-plus-membrane response

  Write each signal as its operating point plus a small fluctuation:

  $
    g_i = overline(g)_lambda + delta g_i, quad
    v_i = overline(v)_lambda + delta v_i, quad
    s_i = lambda / 1000 + delta s_i. quad "(A4)"
  $

  Here δg#sub[i], δv#sub[i], and δs#sub[i] are complete conductance, voltage, and
  input perturbation variables around the means defined in Equations A2 and A3.

  To derive Equation A5, substitute the conductance and input decompositions
  from Equation A4 into the synapse equation, Equation A1:

  $
    (d (overline(g)_lambda + delta g_i)) / (d t)
    = -(overline(g)_lambda + delta g_i) / tau_"AMPA"
    + w_"probe" (lambda / 1000 + delta s_i).
  $

  The operating point is stationary, so its time derivative is zero. Expanding
  the right-hand side then gives

  $
    (d delta g_i) / (d t)
    = (-overline(g)_lambda / tau_"AMPA" + w_"probe" lambda / 1000)
    - frac(delta g_i, tau_"AMPA") + w_"probe" delta s_i.
  $

  The parenthesized stationary terms cancel by Equation A2. The remaining
  perturbation dynamics are

  $
    (d delta g_i) / (d t) = -frac(delta g_i, tau_"AMPA")
    + w_"probe" delta s_i. quad "(A5)"
  $

  To derive Equation A6, substitute the conductance and voltage decompositions
  from Equation A4 into the membrane equation, Equation 5:

  $
    C_E (d (overline(v)_lambda + delta v_i)) / (d t)
    = g_"L,E" (E_L - overline(v)_lambda - delta v_i)
    + (overline(g)_lambda + delta g_i)
    (E_e - overline(v)_lambda - delta v_i).
  $

  As above, the deterministic operating point has zero time derivative. Expanding and grouping
  the right-hand side gives

  $
    C_E (d delta v_i) / (d t)
    = (g_"L,E" (E_L - overline(v)_lambda)
      + overline(g)_lambda (E_e - overline(v)_lambda))
    - (g_"L,E" + overline(g)_lambda) delta v_i
    + (E_e - overline(v)_lambda) delta g_i
    - delta g_i delta v_i.
  $

  The first parenthesized group is zero by Equation A3. The final product is
  second order in the small perturbations and is omitted by the local linear
  approximation. This leaves

  $
    C_E (d delta v_i) / (d t) = -(g_"L,E" + overline(g)_lambda) delta v_i
    + (E_e - overline(v)_lambda) delta g_i. quad "(A6)"
  $

  Here C#sub[E] is excitatory-cell capacitance. All other symbols follow
  Equations A1--A4.

  To derive Equation A7, take the Fourier transform of Equation A5. The
  derivative becomes iω times the transformed signal:

  $
    i omega delta g_i(omega)
    = -frac(delta g_i(omega), tau_"AMPA")
    + w_"probe" delta s_i(omega).
  $

  Move both conductance terms to the left, then divide by their common factor:

  $
    (i omega + 1 / tau_"AMPA") delta g_i(omega)
    = w_"probe" delta s_i(omega),
  $

  $
    delta g_i(omega) = w_"probe" / (i omega + 1 / tau_"AMPA") delta s_i(omega). quad "(A7)"
  $

  Equation A8 follows in the same way from Equation A6. Fourier transformation
  first gives

  $
    i omega C_E delta v_i(omega)
    = -(g_"L,E" + overline(g)_lambda) delta v_i(omega)
    + (E_e - overline(v)_lambda) delta g_i(omega).
  $

  Collect the voltage terms on the left:

  $
    (i omega C_E + g_"L,E" + overline(g)_lambda) delta v_i(omega)
    = (E_e - overline(v)_lambda) delta g_i(omega).
  $

  Dividing by the coefficient of δv#sub[i] (ω) gives

  $
    delta v_i(omega) = (E_e - overline(v)_lambda) /
    (i omega C_E + g_"L,E" + overline(g)_lambda) delta g_i(omega). quad "(A8)"
  $

  Here ω is angular frequency, i is the imaginary unit, and each argument ω
  denotes a Fourier-domain signal.

  Equations A5--A8 form a *linear time-invariant (LTI)* system after
  linearization about the operating point λ. Because the system has been
  linearized about the operating point, it is locally linear and
  time-invariant, so the standard transfer-function formalism applies. The
  canonical frequency-domain input--output relationship is

  $
    delta v_i(omega) = G_lambda(omega) delta s_i(omega).
  $

  Here G#sub[λ] (ω) is the local transfer function relating input spike-train
  perturbations to membrane-voltage perturbations. For non-zero input, the
  input--output relationship can be rearranged as

  $
    G_lambda(omega) = (delta v_i(omega)) / (delta s_i(omega)).
  $

  Thus the ratio follows from the standard input--output relationship rather
  than serving as an independent definition. To derive G#sub[λ] (ω), start from
  Equation A7:

  $
    delta g_i(omega)
    = w_"probe" / (i omega + 1 / tau_"AMPA") delta s_i(omega).
  $

  Substitute this expression for δg#sub[i] (ω) into Equation A8:

  $
    delta v_i(omega)
    = (E_e - overline(v)_lambda) /
    (i omega C_E + g_"L,E" + overline(g)_lambda)
    (w_"probe" / (i omega + 1 / tau_"AMPA") delta s_i(omega)).
  $

  Reorder the scalar factors and factor out δs#sub[i] (ω):

  $
    delta v_i(omega)
    = (w_"probe" / (i omega + 1 / tau_"AMPA") dot
      (E_e - overline(v)_lambda) /
      (i omega C_E + g_"L,E" + overline(g)_lambda))
    delta s_i(omega).
  $

  Comparing this result with the canonical linear time-invariant relationship identifies the
  coefficient multiplying δs#sub[i] (ω) as the transfer function:

  $
    G_lambda(omega)
    = w_"probe" / (i omega + 1 / tau_"AMPA") dot
    (E_e - overline(v)_lambda) /
    (i omega C_E + g_"L,E" + overline(g)_lambda). quad "(A9)"
  $

  Equation A9 is therefore the synaptic filter multiplied by the membrane
  filter, not an arbitrarily chosen ratio. Its λ-dependence captures the change
  in membrane gain and effective time constant with mean conductance.

  === A.3 Finite-window averaging

  Start from the baseline-subtracted feature in Equation 7:

  $
    z_i = 1 / T integral_0^T (v_i(t) - E_L) dif t.
  $

  Within the stationary linearized model used in this appendix, substitute the
  voltage decomposition from Equation A4:

  $
    z_i = 1 / T integral_0^T
    (overline(v)_lambda + delta v_i(t) - E_L) dif t.
  $

  The operating-point voltage and leak reversal potential are constant in time,
  so their integral is their difference multiplied by T. Dividing by T gives

  $
    z_i = overline(v)_lambda - E_L
    + 1 / T integral_0^T delta v_i(t) dif t.
  $

  Define the operating-point feature and its perturbation by

  $
    overline(z)_lambda = overline(v)_lambda - E_L, quad
    delta z_i = z_i - overline(z)_lambda.
  $

  Subtracting the operating-point feature from the expanded expression for
  z#sub[i] cancels the constant term and leaves

  $
    delta z_i = 1 / T integral_0^T delta v_i(t) dif t. quad "(A10)"
  $

  Thus δz#sub[i] is the fluctuation around the operating-point feature, obtained
  by averaging the voltage perturbation over the 200 ms presentation duration
  T. To derive the frequency response, represent the averaging operation as a
  causal rectangular kernel a#sub[T] (u), where

  $
    a_T(u) = 1 / T, quad 0 <= u <= T,
  $

  and a#sub[T] (u) is zero outside that interval. Applying this kernel as a
  moving-average filter gives

  $
    delta z_i(t) = integral_(-infinity)^infinity
    a_T(u) delta v_i(t-u) dif u
    = 1 / T integral_0^T delta v_i(t-u) dif u.
  $

  At the end of the presentation, set t equal to T. To put the resulting
  integral into the same form as Equation A10, introduce the new variable

  $
    t' = T - u, quad dif t' = -dif u.
  $

  When u is zero, t' is T; when u is T, t' is zero. The substitution therefore
  reverses the integration limits. Its minus sign reverses them back:

  $
    delta z_i(T) = 1 / T integral_0^T delta v_i(T-u) dif u
    = -1 / T integral_T^0 delta v_i(t') dif t'
    = 1 / T integral_0^T delta v_i(t') dif t'.
  $

  The name of an integration variable has no effect on the value, so replacing
  t' with t recovers Equation A10 exactly.

  For any time-domain function f(u), use the Fourier-transform convention

  $
    hat(f)(omega) = integral_(-infinity)^infinity
    f(u) exp(-i omega u) dif u.
  $

  Here hat(f) (ω) is the Fourier transform of f(u). Applying this definition
  with f(u) equal to the rectangular kernel a#sub[T] (u), and naming the result
  A#sub[T] (ω), gives

  $
    A_T(omega) = integral_(-infinity)^infinity
    a_T(u) exp(-i omega u) dif u
    = 1 / T integral_0^T exp(-i omega u) dif u.
  $

  Evaluating the integral and rearranging gives

  $
    A_T(omega) = (exp(-i omega T) - 1) / (-i omega T)
    = (1 - exp(-i omega T)) / (i omega T).
  $

  Thus a#sub[T] (u) and A#sub[T] (ω) are not alternative definitions:
  a#sub[T] is the time-domain rectangular averaging kernel, whose height 1/T
  makes its area one, whereas A#sub[T] is that kernel's Fourier transform and
  is the frequency response stated in Equation 14.

  The identity
  $1 - exp(-i x) = 2 i exp(-i x/2) sin(x/2)$, with $x = omega T$, gives the
  equivalent form

  $
    A_T(omega) = exp(-i omega T / 2)
    sin(omega T / 2) / (omega T / 2).
  $

  The exponential has unit magnitude and contributes only the phase delay of a
  window centred at T/2. The magnitude is therefore the sinc-shaped response

  $
    abs(A_T(omega)) = abs(sin(omega T / 2) / (omega T / 2)).
  $

  In this experiment T is measured in ms and physical frequency f is measured
  in Hz, so $omega = 2 pi f / 1000$ rad/ms. Consequently,

  $
    abs(A_T(2 pi f / 1000))
    = abs(sin(pi f T / 1000) / (pi f T / 1000)).
  $

  At zero frequency this expression appears as 0/0, but its continuous limit
  is one, as expected: averaging does not change a constant signal. Its zeros
  occur at $f = 1000 m / T$ Hz for nonzero integers m; for T = 200 ms, the
  first zero is therefore 5 Hz and subsequent zeros occur every 5 Hz. These
  zeros and intervening sidelobes produce the comb-like shape of Figure 4,
  Panel B.

  By the convolution theorem, the averaging filter multiplies the voltage
  spectrum:

  $
    delta z_i(omega) = A_T(omega) delta v_i(omega).
  $

  Substitute the input--voltage relation from Equation A9:

  $
    delta z_i(omega) = A_T(omega) G_lambda(omega)
    delta s_i(omega).
  $

  Define the complete local input--feature response by the canonical linear time-invariant
  relationship

  $
    delta z_i(omega) = H_lambda(omega) delta s_i(omega).
  $

  Comparing the coefficients multiplying δs#sub[i] (ω) identifies
  H#sub[λ] (ω). The two components of Equation A11 are therefore

  $
    A_T(omega) = (1 - exp(-i omega T)) / (i omega T), quad
    H_lambda(omega) = A_T(omega) G_lambda(omega). quad "(A11)"
  $

  Here A#sub[T] (ω) is the averaging-window response, exp is the exponential
  function, and H#sub[λ] (ω) is the complete response from input spike-train
  perturbation to averaged-voltage perturbation.

  For the Bode comparison, define the common reference
  $G_"ref" = abs(G_(lambda_"low")(0))$, where
  $lambda_"low" = 0.25$ spikes/s. The two plotted magnitudes are

  $
    B_G(omega) = 20 log_10 (abs(G_lambda(omega)) / G_"ref"), quad
    B_H(omega) = 20 log_10 (abs(H_lambda(omega)) / G_"ref"). quad "(A12)"
  $

  Here B#sub[G] and B#sub[H] are magnitudes in decibels; |·| is complex
  magnitude; and log#sub[10] is the base-ten logarithm. Figure 4 plots
  B#sub[G] in Panel A and B#sub[H] in Panel B. Because A#sub[T] (0) = 1, the
  low-drive DC values of G#sub[λ] and H#sub[λ] share this reference.

  === A.4 Poisson-noise variance

  In timestep n, let
  $S_n in {0, 1}$ indicate whether a spike occurred. For rate λ and timestep
  Δt, its spike probability and centred value are

  $
    p = lambda Delta t / 1000, quad delta S_n = S_n - p.
  $

  A no-spike bin therefore has $delta S_n = -p$, while a spike bin has
  $delta S_n = 1 - p$; centring does not leave the no-spike bins at zero. For
  example, at λ = 200 spikes/s with 1 ms bins, p = 0.2 and the centred values
  are −0.2 and 0.8. With exp080's 0.1 ms timestep, p = 0.02 and they are −0.02
  and 0.98. In both cases their probability-weighted mean is zero.

  Autocovariance measures whether these deviations from the mean tend to
  occur together at two different times. For discrete lag k it is
  $E[delta S_n delta S_(n+k)]$. Independent Poisson bins give zero covariance
  for $k != 0$; at $k = 0$, a bin is compared with itself and the covariance
  is its variance $p(1-p)$.

  The analytical model takes the continuous-time limit. It represents the
  event train from Equation A4 as
  $s_i(t) = sum_j delta(t - t_j)$, a sum of Dirac impulses rather than a series
  of finite ones and zeros, and defines the centred fluctuation
  $delta s_i(t) = s_i(t) - lambda / 1000$. As Δt tends to zero, the discrete
  zero-lag variance becomes a Dirac impulse. At continuous lag u, the
  autocovariance is

  $
    R_(delta s)(u) = E[delta s_i(t) delta s_i(t + u)]
    = lambda / 1000 delta(u).
  $

  Here E denotes an average over repeated Poisson realizations. A positive
  autocovariance would mean that above-mean deviations at one time tend to
  accompany above-mean deviations at the displaced time; a negative value
  would indicate opposing deviations. For a Poisson process, counts in
  disjoint intervals are independent, so this covariance is zero whenever
  $u != 0$. At zero lag, an event coincides with itself, producing the Dirac
  impulse δ(u). Its coefficient λ/1000 is the mean event count per ms and is
  the area of that impulse, not its finite height.

  By the Wiener--Khinchin relation, the input power spectral density is the
  Fourier transform of the autocovariance. The Fourier transform of δ(u) is
  one at every angular frequency, so the Poisson input has the constant, or
  white, two-sided spectrum

  $
    S_"in"(omega) = lambda / 1000. quad "(A13)"
  $

  Here S#sub[in] (ω) is input power per unit angular frequency and λ is mean
  spike rate in spikes/s; division by 1000 expresses the point-process rate per
  millisecond. Passing this noise through Equation A11 gives

  $
    S_z(omega) = abs(H_lambda(omega))^2 S_"in"(omega). quad "(A14)"
  $

  Here S#sub[z] (ω) is predicted feature-noise power and
  |H#sub[λ] (ω)|#super[2] is the transmitted fraction of input-noise power. Under
  the two-sided angular-frequency convention, total predicted variance is

  $
    "Var"_"linear"(z) = 1 / (2 pi) integral_(-oo)^oo
    abs(H_lambda(omega))^2 S_"in"(omega) dif omega. quad "(A15)"
  $

  Here Var#sub[linear] (z) is predicted feature variance and π is the circle
  constant; the remaining symbols follow Equations A11--A14. This result is
  exact for the stated linear stationary model, but only approximate for the
  finite, discrete, conductance-dependent probe. It is retained as an analytical
  diagnostic rather than as the generator of ANN inputs.

]
