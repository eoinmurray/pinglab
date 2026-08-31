#import "contents.typ": with-contents, with-result-sections
#import "/.demolab/lib.typ": data-json, data-image, cite, reference-list
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp023")

#let meta = (
  status: "[▦ DATA | v28.0.0]",
  title: "Turning the PING Loop On",
  created_at: "2026-05-13T00:00:00Z",
  updated_at: "2026-08-31T00:00:00Z",
  description: "PING stripped to its biophysical fundamentals: free-running activity, population spectra and rate responses with the excitatory–inhibitory loop off and on.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp023",)
#let preview-figures = (
  (path: "exp023/overview_compound.png", label: "overview compound"),
  (path: "exp023/traces__coba__v_e.svg", label: "COBA membrane voltage"),
  (path: "exp023/traces__coba__g_e.svg", label: "COBA conductances"),
  (path: "exp023/traces__coba__i_e.svg", label: "COBA currents"),
  (path: "exp023/traces__ping__v_e.svg", label: "PING excitatory membrane voltage"),
  (path: "exp023/traces__ping__g_e.svg", label: "PING excitatory conductances"),
  (path: "exp023/traces__ping__i_e.svg", label: "PING excitatory currents"),
  (path: "exp023/traces__ping__v_i.svg", label: "PING inhibitory membrane voltage"),
  (path: "exp023/traces__ping__g_i.svg", label: "PING inhibitory conductances"),
  (path: "exp023/traces__ping__i_i.svg", label: "PING inhibitory currents"),
)

// Values and procedural settings come only from the explicitly selected evidence.
#let render-report(data-file) = [
  #let r = data-json(data-file("exp023/numbers.json"))
  #assert(r.schema == "exp023.analysis/v1", message: "Unsupported exp023 presentation")
  #let c = r.config
  #let b = c.biophysics
  #let points = c.drive.raster_operating_points
  #let sweep = c.drive.fi_sweep
  #let fmt(x) = str(calc.round(x, digits: 1))
  #let peak = r.f_gamma_hz.ping
  #let e-coba = r.fi_curves.coba.e.last()
  #let e-ping = r.fi_curves.ping.e.last()
  #set math.equation(numbering: "(1)")

  == Abstract

  Asked what the recurrent PING loop does before introducing training,
  classification or a learned readout. Compared the same Poisson-driven
  excitatory–inhibitory architecture with reciprocal feedback disabled and
  enabled.

  Enabling the loop produced rhythmic population activity and strongly
  suppressed excitatory firing across the matched-drive sweep. Establishes the
  circuit's basic operating behaviour, but the limited simulations do not show
  that gamma timing itself caused the suppression.

  == Results

  #with-result-sections[

  === COBA and PING architecture, activity spectra and rate response

  #figure(
    data-image(data-file("exp023/overview_compound.png"), width: 100%,
      alt: "Loop-off and loop-on schematics above population rasters, spectra and firing-rate curves."),
    caption: [
      COBA (A) and PING (B): E spikes black, I spikes red.
      Rasters and spectra show #points.coba.t_ms ms at
      #points.coba.input_rate_hz and #points.ping.input_rate_hz Hz input,
      respectively. Rate curves show one-trial population means over the
      matched-drive sweep, without error bars; vertical scales differ.
      The dashed marker is the analysis-selected spectral peak, not a
      rhythmicity significance test. Schematics describe the model, not a measurement.
    ],
  )

  === COBA excitatory-neuron membrane voltage

  #figure(
    data-image(data-file("exp023/traces__coba__v_e.svg"), width: 100%),
    caption: [Membrane voltage of the highest-spike-count E neuron with the loop
      off, driven at #points.coba.input_rate_hz Hz for #points.coba.t_ms ms.
      The dashed line marks the #b.threshold_mV mV threshold; spikes reset
      voltage to #b.reset_mV mV. This is an illustrative neuron, not a population mean.],
  )

  === COBA excitatory-neuron conductances

  #figure(
    data-image(data-file("exp023/traces__coba__g_e.svg"), width: 100%),
    caption: [Excitatory conductance (black) and fixed leak (dotted) for the same
      COBA neuron. Input spikes increase excitation, which decays between events;
      the disconnected inhibitory feedback contributes no conductance.],
  )

  === COBA excitatory-neuron currents

  #figure(
    data-image(data-file("exp023/traces__coba__i_e.svg"), width: 100%),
    caption: [Signed excitatory and leak currents for the same COBA neuron.
      Positive current is depolarising; current was reconstructed from voltage
      and conductance using the driving-force relation in Methods.],
  )

  === PING excitatory-neuron membrane voltage

  #figure(
    data-image(data-file("exp023/traces__ping__v_e.svg"), width: 100%),
    caption: [Membrane voltage of the highest-spike-count PING E neuron at
      #points.ping.input_rate_hz Hz input for #points.ping.t_ms ms.
      This is a different input rate from COBA, not a matched-input trace comparison.],
  )

  === PING excitatory-neuron conductances

  #figure(
    data-image(data-file("exp023/traces__ping__g_e.svg"), width: 100%),
    caption: [Excitatory (black), inhibitory (red when nonzero) and leak (dotted)
      conductances on the same PING E neuron. All conductances are non-negative;
      inhibition enters through its reversal potential, not a negative conductance.],
  )

  === PING excitatory-neuron currents

  #figure(
    data-image(data-file("exp023/traces__ping__i_e.svg"), width: 100%),
    caption: [Signed currents on the same PING E neuron. Inhibitory current is
      hyperpolarising when voltage exceeds the #b.E_I_mV mV inhibitory reversal
      potential, even though inhibitory conductance is positive.],
  )

  #if r.raster.ping.i_index != none [
    === PING inhibitory-neuron membrane voltage

    These cellular traces illustrate reciprocal E→I→E feedback; they do not
    measure population-wide phase locking or spikes per cycle.

    #figure(
      data-image(data-file("exp023/traces__ping__v_i.svg"), width: 100%),
      caption: [Membrane voltage of the highest-spike-count PING I neuron,
        from the same trial as the PING E traces. The dashed line marks threshold.],
    )

    === PING inhibitory-neuron conductances

    #figure(
      data-image(data-file("exp023/traces__ping__g_i.svg"), width: 100%),
      caption: [Excitatory conductance arriving from the E population (black)
        and fixed leak (dotted) on the selected I neuron. The model has no I→I synapse.],
    )

    === PING inhibitory-neuron currents

    #figure(
      data-image(data-file("exp023/traces__ping__i_i.svg"), width: 100%),
      caption: [Signed excitatory and leak currents on the same I neuron.
        Positive current is depolarising.],
    )
  ]

  ]

  == Methods

  We compared untrained loop-off and loop-on networks using the following
  simulation and measurement procedure.

  + *Construct the two loop conditions.* Both networks contained #c.n_e
    excitatory (E) and #c.n_i inhibitory (I) neurons. An input layer drove E neurons through
    feedforward weights; reciprocal E→I and I→E weights formed the PING loop,
    without E→E or I→I recurrence. Loop coupling was
    #points.coba.ei_strength or #points.ping.ei_strength; the I→E parent mean
    was #c.ei_ratio times E→I, and both parent standard deviations were 10% of
    their means. Input parent weights had mean #c.input_weight_parent_mean
    and standard deviation #c.input_weight_parent_sd; lower-clamped Gaussian
    draws were normalised by fan-in, with #calc.round(100 * c.input_initial_zero_fraction)%
    initially zero and survivors rescaled.

  + *Generate the drive and simulate.* Uniform Poisson input drove every channel
    at the condition's rate, with seed #c.seed for input and network initialization.
    Membranes began at #c.initial_voltage_mV mV and conductances at
    #c.initial_conductance_uS µS. Exponential-Euler membrane integration used
    #points.coba.dt_ms ms steps; AMPA and GABA decay constants were
    #b.tau_ampa_ms and #b.tau_gaba_ms ms, with E/I refractory periods
    #b.refractory_E_ms/#b.refractory_I_ms ms.
    Full trial recordings included spikes, voltages and conductances.

  + *Measure the drive response.* Both loop conditions used
    #c.trials_per_condition trial at each of #sweep.input_rates_hz.map(str).join([, ]) Hz
    through #sweep.n_in input channels for #sweep.t_ms ms, without a discarded transient.
    Mean per-neuron firing rate was
    $ r_P = n_("spike",P) / (N_P T_"present"), $
    where $P$ identifies E or I, $n_("spike",P)$ is the population's total
    spike count, $N_P$ its neuron count, $T_"present"$ the full presentation
    duration in seconds and $r_P$ the rate in hertz. No averaging across
    independent seeds was performed.

  + *Estimate the population spectrum.* The mean E spike trace was demeaned
    and passed to Welch's density estimator with one full-trial window.#cite(1)
    This is a single-window estimate, not an average over independent segments.
    The largest peak between #r.measurement.frequency_band_hz.first() and
    #r.measurement.frequency_band_hz.last() Hz was refined by three-bin parabolic
    interpolation, clamped to half a bin; it was reported only when I spikes
    were present. This reporting rule is not a test of significant rhythmicity.

  + *Reconstruct signed currents.* Each displayed neuron was the first neuron
    attaining the highest total spike count in its population during that
    raster trial. A silent E population used neuron zero; a silent I population
    had no selected-neuron panels. Raster trials used #points.coba.n_in input
    channels at #points.coba.input_rate_hz Hz for COBA and
    #points.ping.input_rate_hz Hz for PING, unlike the matched-drive sweep.

    #block(breakable: false)[For the selected neurons, recorded conductances
    and voltages gave
    $ I_X^"in" = -g_X (V_m - E_X), $
    where $X$ identifies excitation, inhibition or leak; $g_X$ is conductance in
    µS, $V_m$ membrane voltage and $E_X$ reversal potential in mV, and
    $I_X^"in"$ inward current in nA. Reversals were #b.E_E_mV,
    #b.E_I_mV and #b.E_L_mV mV, respectively; positive current is depolarising.
    The sign lives in the driving force, never in the conductance.]

  #run-view("exp023", inputs)

  #reference-list((
    (text: [P. D. Welch. “The Use of Fast Fourier Transform for the Estimation of
      Power Spectra: A Method Based on Time Averaging Over Short, Modified
      Periodograms.” _IEEE Transactions on Audio and Electroacoustics_ 15(2),
      70–73 (1967).], doi: "10.1109/TAU.1967.1161901"),
  ))
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [How does enabling the excitatory–inhibitory feedback loop change free-running activity?
      Compare the COBA control with PING across input drive, using rasters,
      population spectra and cellular traces.],
    preview-figures,
  )
}

#let meta = meta + (assets: input-assets("exp023", inputs))
#let body = with-datasets("exp023", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-contents(body)
