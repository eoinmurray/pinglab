#import "contents.typ": contents-here, with-contents, result-card, with-numbered-equations, with-result-sections
#import "/.demolab/lib.typ": data-image, data-json, cite, reference-list
#import "run-inputs.typ": data-file, input-assets, inputs-ready, pending-report
#import "run-view.typ": run-view, with-datasets
#let data-file = data-file.with(article: "exp111")

#let meta = (
  status: "[▦ DATA | v34.0.1]",
  title: "Backend Distance Across Gamma-Gated-Sparsity Mechanisms",
  created_at: "2026-09-02T00:00:00Z",
  updated_at: "2026-09-02",
  description: "Twenty fast comparisons test where snnsim and Brian2 agree across the mechanisms and trained networks used by the gamma-gated-sparsity manuscript.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp111",)
#let preview-figures = (
  (path: "exp111/lif-spiking.svg", label: "single-neuron dynamics"),
  (path: "exp111/gaba-frequency.svg", label: "reduced-circuit frequency sweep"),
  (path: "exp111/coba-ping-endpoints.svg", label: "full-network endpoint replay"),
  (path: "exp111/loop-transfer.svg", label: "loop-strength transfer"),
)

#let render-report(data-file) = [
  #let r = data-json(data-file("exp111/numbers.json"))
  #let test(id) = r.tests.filter(row => row.id == id).first()
  #let point(item, label) = item.series.filter(row => row.label == label).first()
  #let rounded(value, digits: 2) = calc.round(value, digits: digits)

  == Abstract

  We tested whether the snnsim mechanisms supporting the gamma-gated-sparsity
  manuscript survive reimplementation in Brian2. The checks moved from isolated
  neuron and synapse dynamics through reduced PING circuits to retained trained
  networks.

  Primitive and reduced-circuit behaviour matched numerically, whereas
  full-network replays showed backend-dependent firing rates and readout
  evidence. The comparison therefore characterises where the implementations
  coincide and how far their checkpoint-level outputs diverge.

  #contents-here()

  == Results

  #with-result-sections[

  #result-card[
  === Passive conductance-based LIF dynamics

  #let t = test("lif-passive")
  Across eight sampled cases, snnsim and Brian2 produced the same final voltages
  to displayed precision, spanning approximately −55.36 to −66.65 mV; their
  maximum difference was #rounded(t.maximum_absolute_difference, digits: 14) mV.
  This is adequate evidence for the tested passive update because the residual
  is at floating-point scale, but it does not validate unsampled parameters.

  #figure(
    data-image(data-file("exp111/lif-passive.svg"), width: 92%, alt: "Passive excitatory and inhibitory LIF samples compared between snnsim and Brian2, with backend residuals."),
    caption: [*(A)* Final membrane voltage for four excitatory-like and four
      inhibitory-like conductance samples in snnsim and Brian2. *(B)* Brian2
      minus snnsim for each sample.],
  )
  ]

  #result-card[
  === Threshold, reset and refractory dynamics

  #let t = test("lif-spiking")
  snnsim and Brian2 each produced six spikes at 0.5, 4.0, 7.5, 11.0, 14.5 and
  18.0 ms, while their voltage traces differed by at most
  #rounded(t.maximum_absolute_difference, digits: 14) mV. This is adequate for
  the tested threshold, reset and refractory sequence because both continuous
  state and discrete event times coincide; it does not cover other drive regimes.

  #figure(
    data-image(data-file("exp111/lif-spiking.svg"), width: 92%, alt: "Repeated threshold crossings, resets and refractory intervals in snnsim and Brian2, with voltage residuals."),
    caption: [*(A)* Membrane voltage over a 20 ms driven-neuron simulation at a
      0.1 ms timestep. *(B)* Brian2 minus snnsim voltage at each recorded step,
      alongside the coincident spike times.],
  )
  ]

  #result-card[
  === AMPA and GABA impulse decay

  #let t = test("synapse-impulses")
  Across the seven decay times, snnsim endpoints ranged from 0.0985 to 1.3682 µS
  and Brian2 produced the same values to displayed precision; the largest trace
  difference was #rounded(t.maximum_absolute_difference, digits: 15) µS. This is
  adequate for the isolated exponential-decay rule because the residual is
  negligible, making that rule an unlikely source of later network discrepancies.

  #figure(
    data-image(data-file("exp111/synapse-impulses.svg"), width: 92%, alt: "Exponential synaptic impulse responses across decay times in snnsim and Brian2, with residuals."),
    caption: [*(A)* Final conductance after the same six-event train across seven
      decay times. *(B)* Brian2 minus snnsim conductance.],
  )
  ]

  #result-card[
  === Event scheduling through the E–I loop

  #let t = test("event-causality")
  snnsim and Brian2 both produced an 18.00 Hz excitatory rate, an 86.67 Hz
  inhibitory rate and a 0.5 ms first E-to-I event lag. This exact comparison is
  adequate for the intended causal ordering in the reduced circuit, but not for
  claiming that event schedules remain equivalent after long recurrent trajectories.

  #figure(
    data-image(data-file("exp111/event-causality.svg"), width: 92%, alt: "Excitatory and inhibitory rates from a reduced E-I loop in snnsim and Brian2, with residuals."),
    caption: [*(A)* Excitatory and inhibitory population-mean firing rates from
      the same 200 ms reduced circuit. *(B)* Brian2 minus snnsim rate, with the
      rate residual.],
  )
  ]

  #result-card[
  === Projection and weight scaling

  #let t = test("projection-scaling")
  At reciprocal scales 0, 0.5 and 1, snnsim measured 209.33, 40.00 and 18.00 Hz,
  and Brian2 measured the same three rates. This is adequate evidence for the
  tested projection orientation and scaling because the exact match spans a
  191.33 Hz intervention effect rather than only a near-static operating point.

  #figure(
    data-image(data-file("exp111/projection-scaling.svg"), width: 92%, alt: "Excitatory firing rate across reciprocal loop scales in snnsim and Brian2, with residuals."),
    caption: [*(A)* Excitatory population-mean firing rate as both reciprocal
      fan-in-normalised projections were scaled. *(B)* Brian2 minus snnsim rate,
      showing zero rate distance at each scale.],
  )
  ]

  #result-card[
  === Matched-drive loop-off and loop-on activity

  #let t = test("matched-loop")
  snnsim measured loop-off and loop-on contrasts of 1.0678 and 0.8831, and Brian2
  measured the same values. This is adequate for the reduced matched-drive
  comparison because the 0.1847 loop effect is reproduced with zero observed
  backend distance; it does not establish agreement in the trained networks.

  #figure(
    data-image(data-file("exp111/matched-loop.svg"), width: 92%, alt: "Autocorrelation contrast with the reduced reciprocal loop disabled and enabled in both simulators."),
    caption: [*(A)* Excitatory autocorrelation lobe–trough contrast under matched
      tonic drive with the loop disabled or enabled. *(B)* Brian2 minus snnsim
      contrast.],
  )
  ]

  #result-card[
  === Input-response curves

  #let t = test("input-response")
  At relative drives 0.75, 1.0, 1.25 and 1.5, snnsim measured 18.00, 18.00,
  21.50 and 23.75 Hz, and Brian2 measured the same rates. This is adequate for
  the sampled input-response curve because both its plateau and 5.75 Hz rise
  coincide; four points remain insufficient to validate arbitrary drive levels.

  #figure(
    data-image(data-file("exp111/input-response.svg"), width: 92%, alt: "Reduced-circuit excitatory rates across four drive levels in snnsim and Brian2, with residuals."),
    caption: [*(A)* Excitatory population-mean firing rate across relative tonic
      drive. *(B)* Brian2 minus snnsim rate.],
  )
  ]

  #result-card[
  === Coupling-plane onset proxy

  #let t = test("coupling-onset")
  At coupling scales 0, 0.25, 0.5, 0.75 and 1, both snnsim and Brian2 measured
  contrasts of 1.076, 1.110, 0.982, 0.954 and 0.765. This is adequate for the
  displayed one-dimensional onset proxy because every sampled value coincides,
  but not for the full two-dimensional coupling plane omitted by this probe.

  #figure(
    data-image(data-file("exp111/coupling-onset.svg"), width: 92%, alt: "Reduced-circuit autocorrelation contrast across reciprocal coupling scales in snnsim and Brian2."),
    caption: [*(A)* Excitatory autocorrelation lobe–trough contrast across five
      reciprocal-coupling scales. *(B)* Brian2 minus snnsim contrast, with the
      backend residual.],
  )
  ]

  #result-card[
  === Uncoupled private-like and shared-like controls

  #let t = test("uncoupled-nulls")
  Without reciprocal coupling, snnsim measured contrasts of 0.9861 and 1.0439
  for the private-like and shared-like drives, and Brian2 measured the same two
  values. This is adequate for these null controls because their 0.0578 contrast
  difference is preserved with zero backend distance; it does not test coupled nulls.

  #figure(
    data-image(data-file("exp111/uncoupled-nulls.svg"), width: 92%, alt: "Two uncoupled-drive controls compared between snnsim and Brian2 using autocorrelation contrast."),
    caption: [*(A)* Excitatory autocorrelation lobe–trough contrast for lower
      private-like and higher shared-like tonic drive without reciprocal coupling.
      *(B)* Brian2 minus snnsim contrast.],
  )
  ]

  #result-card[
  === GABA timescale and spectral-peak frequency

  #let t = test("gaba-frequency")
  At GABA decay times 4.5, 6, 9, 12, 18 and 27 ms, snnsim selected 53.33, 42.22,
  60.00, 68.89, 20.00 and 68.89 Hz, and Brian2 selected the same frequencies.
  This is adequate for comparing the raw spectral-bin estimator, but not for
  claiming identical spectra or sub-bin frequencies because the readout is coarse.

  #figure(
    data-image(data-file("exp111/gaba-frequency.svg"), width: 92%, alt: "Raw excitatory spectral-peak frequency across GABA decay times in snnsim and Brian2."),
    caption: [*(A)* Raw excitatory spectral-peak frequency in the 5–150 Hz search
      interval across inhibitory decay times. *(B)* Brian2 minus snnsim frequency,
      showing the frequency residual.],
  )
  ]

  #result-card[
  === Excitatory spikes per inhibitory cycle

  #let t = test("cycle-participation")
  At every tested decay time, snnsim and Brian2 both measured a one-spike
  conditional fraction of 1.0 and a total-variation distance of zero. The values
  agree exactly, but this is not adequate for a sensitive backend validation:
  saturation at 1.0 leaves no variation with which to expose subtler differences.

  #figure(
    data-image(data-file("exp111/cycle-participation.svg"), width: 92%, alt: "Conditional one-spike fraction per active excitatory neuron and inferred inhibitory cycle in both simulators."),
    caption: [*(A)* Fraction of active excitatory neuron–cycle pairs containing
      one spike, with cycles inferred from inhibitory population peaks. *(B)*
      Brian2 minus snnsim fraction.],
  )
  ]

  #result-card[
  === Selected and final checkpoint replay - NOTABLE

  #let t = test("checkpoint-replay")
  At the selected checkpoint, snnsim measured 24.46 Hz and Brian2 27.08 Hz; at
  the final checkpoint they measured 24.30 and 27.36 Hz. The 2.62–3.07 Hz rate
  gaps are not adequate for claiming backend-independent absolute rates, but
  evidence MAE remained #rounded(point(t, "selected").evidence_mae, digits: 5)
  and #rounded(point(t, "final").evidence_mae, digits: 5), and both predicted
  class 4, making this sample's decision adequately reproduced.

  #figure(
    data-image(data-file("exp111/checkpoint-replay.svg"), width: 92%, alt: "Excitatory firing rates for selected and final checkpoints replayed in snnsim and Brian2."),
    caption: [*(A)* Excitatory population-mean firing rate for the retained
      selected and final canonical-PING parameter files under one fixed held-out
      input. *(B)* Brian2 minus snnsim rate.],
  )
  ]

  #result-card[
  === COBA and PING endpoint replay - NOTABLE

  #let t = test("coba-ping-endpoints")
  For COBA, snnsim measured 237.27 Hz and Brian2 148.24 Hz; for PING they measured
  22.34 and 25.51 Hz. The large 89.02 Hz COBA gap and moderate 3.16 Hz PING gap
  are not adequate for backend-independent absolute-rate claims. Both nevertheless
  predict class 5 and a large COBA-to-PING reduction, so the qualitative inhibitory
  contrast is adequately reproduced for this input, despite evidence MAEs of
  #rounded(point(t, "COBA").evidence_mae, digits: 3) and
  #rounded(point(t, "PING").evidence_mae, digits: 3).

  #figure(
    data-image(data-file("exp111/coba-ping-endpoints.svg"), width: 92%, alt: "Full-network excitatory firing rates at the retained COBA and PING endpoints in snnsim and Brian2."),
    caption: [*(A)* Excitatory population-mean firing rate for retained COBA and
      PING checkpoints under the same fixed held-out input. *(B)* Brian2 minus
      snnsim rate.],
  )
  ]

  #result-card[
  === Post-training loop-strength transfer - NOTABLE

  #let t = test("loop-transfer")
  At loop scales 0, 0.5 and 1, snnsim measured 207.33, 28.98 and 13.57 Hz, while
  Brian2 measured 132.82, 26.61 and 14.46 Hz. The shared downward trend is adequate
  for the qualitative inhibitory effect, but the 74.51 Hz open-loop gap is not
  adequate for its quantitative magnitude. Predictions also differ at scales
  0.5 and 1, so downstream classification is not reproduced for those samples.

  #figure(
    data-image(data-file("exp111/loop-transfer.svg"), width: 92%, alt: "Excitatory firing rate as retained PING recurrent weights are applied to a COBA feedforward checkpoint at three loop scales."),
    caption: [*(A)* Excitatory population-mean firing rate after combining the
      retained COBA feedforward checkpoint with canonical-PING recurrent matrices
      at three scales. *(B)* Brian2 minus snnsim rate.],
  )
  ]

  #result-card[
  === Shared hidden-spike perturbation inputs

  #let t = test("spike-perturbations")
  For half deletion, full deletion, 20 Hz addition and 40 Hz addition, the
  snnsim inputs contained 102, 0, 416 and 593 events, and the Brian2 inputs
  contained the same counts. This is adequate for random-intervention parity,
  but not for simulator validation because no downstream responses are compared.

  #figure(
    data-image(data-file("exp111/spike-perturbations.svg"), width: 92%, alt: "Realised hidden-event counts for shared spike deletion and addition inputs supplied to both backend protocols."),
    caption: [*(A)* Realised excitatory-event counts after two deletion and two
      Poisson-addition settings. *(B)* Brian2-input minus snnsim-input count, with
      a zero event-count residual.],
  )
  ]

  #result-card[
  === Shared inhibitory-jitter inputs

  #let t = test("inhibitory-jitter")
  Under both fixed-window and cellwise jitter, the snnsim input contained 220
  inhibitory events and the Brian2 input also contained 220. This is adequate
  for count preservation, but not for dynamical equivalence because equal counts
  do not establish equal event times or downstream trajectories.

  #figure(
    data-image(data-file("exp111/inhibitory-jitter.svg"), width: 92%, alt: "Retained inhibitory-event counts for two shared jitter-input constructions."),
    caption: [*(A)* Retained inhibitory-event counts for fixed-window and
      cellwise jitter inputs. *(B)* Brian2-input minus snnsim-input count, with
      a zero event-count residual.],
  )
  ]

  #result-card[
  === Integration-timestep robustness - NOTABLE

  #let t = test("timestep-robustness")
  At 0.05, 0.1 and 1 ms, snnsim measured 9.98, 14.84 and 10.83 Hz, while Brian2
  measured 12.25, 14.28 and 9.32 Hz. The 0.57 Hz nominal-timestep gap is adequate
  for a close quantitative comparison, but the 1.51–2.27 Hz extreme-timestep gaps
  are not adequate for timestep-invariant rate claims. Every condition predicted
  class 1, so the tested decisions remain adequately reproduced.

  #figure(
    data-image(data-file("exp111/timestep-robustness.svg"), width: 92%, alt: "Excitatory firing rates from retained networks trained and replayed at three integration timesteps in snnsim and Brian2."),
    caption: [*(A)* Excitatory population-mean firing rate for retained networks
      matched to 0.05, 0.1 and 1 ms integration timesteps. *(B)* Brian2 minus
      snnsim rate.],
  )
  ]

  #result-card[
  === Frozen and trained recurrent weights - NOTABLE

  #let t = test("recurrent-training")
  With frozen recurrence, snnsim measured 18.32 Hz and Brian2 20.08 Hz; with
  trained recurrence they measured 57.70 and 56.21 Hz. These 1.48–1.76 Hz gaps
  are adequate for the broad training comparison because both backends show an
  approximately 36–39 Hz increase and predict class 8, but not for asserting
  identical pointwise rates.

  #figure(
    data-image(data-file("exp111/recurrent-training.svg"), width: 92%, alt: "Excitatory firing rates for frozen and trainable recurrent-weight checkpoints in snnsim and Brian2."),
    caption: [*(A)* Excitatory population-mean firing rate for retained
      frozen-loop and trainable-loop checkpoints under fixed held-out inputs.
      *(B)* Brian2 minus snnsim rate.],
  )
  ]

  #result-card[
  === Continuous hidden-state stream

  #let t = test("stream-resets")
  Across segments 1–3, snnsim measured 7.79, 8.54 and 13.75 Hz, while Brian2
  measured 7.50, 8.22 and 12.73 Hz. The 0.29–1.02 Hz gaps are adequate for a
  broadly similar stream-rate profile, but not for claiming exact state continuity;
  the growing separation could reflect accumulated trajectory divergence, and
  one stream cannot establish whether that growth is systematic.

  #figure(
    data-image(data-file("exp111/stream-resets.svg"), width: 92%, alt: "Segment-wise excitatory firing rates during one uninterrupted three-input stream in snnsim and Brian2."),
    caption: [*(A)* Excitatory population-mean firing rate within three 50 ms
      segments of one uninterrupted hidden-state simulation. *(B)* Brian2 minus
      snnsim rate.],
  )
  ]

  #result-card[
  === Continuous-stream duration and input rate

  #let t = test("duration-rate")
  Across the four duration–rate corners, snnsim measured 0.00, 16.37, 5.41 and
  6.16 Hz, while Brian2 measured 0.00, 16.70, 5.41 and 6.04 Hz. The maximum gap
  of #rounded(t.maximum_absolute_difference) Hz is adequate for reproducing these
  sampled corner rates, but the paired design is not adequate for separating
  duration effects from input-rate effects between conditions.

  #figure(
    data-image(data-file("exp111/duration-rate.svg"), width: 92%, alt: "Excitatory firing rates for four presentation-duration and input-rate combinations in snnsim and Brian2."),
    caption: [*(A)* Excitatory population-mean firing rate for four paired
      presentation-duration and maximum-pixel input-rate conditions. *(B)*
      Brian2 minus snnsim rate.],
  )
  ]

  ]

  == Methods

  === Compute

  We ran each protocol in snnsim and an independently expressed Brian2 model.#cite(1)
  Random inputs were realised once and shared between backends. Reduced circuits
  contained 20 excitatory and 5 inhibitory neurons; full-network probes reused
  fixed seed-42 checkpoints and held-out MNIST inputs from exp022 without retraining.

  + *Passive conductance LIF.* We sampled eight passive excitatory-like
    and inhibitory-like conductance settings and compared complete membrane-voltage
    traces. This checks the subthreshold update underlying
    #link("/exp110/")[\#exp110 Manuscript].

  + *Threshold, reset and refractory.* We drove one neuron through repeated
    spikes and compared voltage, spike count and spike times. This checks the event
    mechanism shared by the trained-network experiments synthesised in
    #link("/exp110/")[\#exp110 Manuscript].

  + *AMPA and GABA impulse responses.* We applied the same six-event train
    across seven conductance-decay times and compared complete conductance traces.
    This checks the synaptic dynamics producing excitation, inhibition and PING
    rhythms throughout #link("/exp110/")[\#exp110 Manuscript].

  + *E–I event scheduling.* We compared population rates and the first
    excitatory-to-inhibitory event lag in a reduced reciprocal circuit. This checks
    the causal E→I→E sequence interpreted in
    #link("/exp023/")[\#exp023 Turning the PING Loop On],
    #link("/exp033/")[\#exp033 Gamma Emerges at a Hopf Bifurcation] and
    #link("/exp054/")[\#exp054 Pinglab Rythmicity Metric].

  + *Projection and weight scaling.* We scaled both reciprocal projections
    from zero to full strength and compared excitatory rates. This checks matrix
    orientation and loop scaling used in
    #link("/exp023/")[\#exp023 Turning the PING Loop On],
    #link("/exp033/")[\#exp033 Gamma Emerges at a Hopf Bifurcation],
    #link("/exp038/")[\#exp038 Switching On the Inhibitory Loop] and
    #link("/exp054/")[\#exp054 Pinglab Rythmicity Metric].

  + *Matched-drive loop comparison.* We reproduced reduced loop-off and
    loop-on conditions under matched tonic drive and measured autocorrelation
    lobe–trough contrast. This targets the recurrent-organisation comparison in
    #link("/exp023/")[\#exp023 Turning the PING Loop On].

  + *Input-response curve.* We varied relative tonic drive across four
    levels and measured excitatory population rates. This targets the input-sweep
    evidence used to distinguish COBA and PING operating regimes in
    #link("/exp023/")[\#exp023 Turning the PING Loop On].

  + *Coupling-onset proxy.* We varied reciprocal coupling over
    five points and measured autocorrelation contrast. This samples the oscillatory
    onset described in #link("/exp033/")[\#exp033 Gamma Emerges at a Hopf Bifurcation]
    and the empirical rhythmicity map defined in
    #link("/exp054/")[\#exp054 Pinglab Rythmicity Metric].

  + *Uncoupled nulls.* We disabled reciprocal coupling and compared
    private-like and shared-like drive controls using autocorrelation contrast.
    This checks the null behaviour of the measure used by
    #link("/exp054/")[\#exp054 Pinglab Rythmicity Metric].

  + *GABA timescale and frequency.* We varied inhibitory conductance
    decay across six values and selected the largest raw spectral-power bin from
    5–150 Hz. This targets the rate–gamma-frequency relationship in
    #link("/exp041/")[\#exp041 Firing Rate Tracks Gamma Frequency].

  + *Cycle participation.* We inferred inhibitory cycles from peaks in
    a 1 ms Gaussian-smoothed inhibitory trace and counted excitatory spikes per
    active neuron–cycle pair. This targets the account in
    #link("/exp046/")[\#exp046 One Spike per Gamma Cycle].

  + *Checkpoint replay.* We replayed the selected and final canonical-PING
    checkpoints under one fixed held-out input and compared excitatory rate,
    normalised readout evidence and class. This targets the convergence claim in
    #link("/exp024/")[\#exp024 Accuracy Plateaus While Firing Rate Rises].

  + *COBA and PING endpoints.* We replayed retained COBA and PING
    checkpoints under the same held-out input and compared rate, evidence and class.
    This targets the accuracy–firing-rate comparison in
    #link("/exp025/")[\#exp025 Accuracy and Firing Rate With and Without Inhibition].

  + *Loop-strength transfer.* We combined a retained COBA feedforward
    checkpoint with canonical-PING recurrent matrices at three inference-time scales.
    This targets post-training activation of inhibition in
    #link("/exp038/")[\#exp038 Switching On the Inhibitory Loop].

  + *Hidden-spike perturbation inputs.* We froze realised half-deletion,
    full-deletion and two Poisson-addition event sets before backend dispatch. This
    checks intervention parity for #link("/exp037/")[\#exp037 Dropped Spikes vs Added Noise]
    without replaying its downstream responses.

  + *Inhibitory-jitter inputs.* We supplied fixed-window and cellwise
    jitter constructions with preserved inhibitory-event counts. This checks input
    parity for #link("/exp042/")[\#exp042 Inhibitory Replay Perturbations Change Excitatory Firing]
    without reproducing its downstream excitatory response.

  + *Integration timestep.* We replayed networks matched to 0.05, 0.1
    and 1 ms training-and-inference timesteps and compared rate, evidence and class.
    This targets the operating range reported in
    #link("/exp044/")[\#exp044 Firing Rate Across the Timestep Sweep].

  + *Recurrent-weight training.* We replayed frozen-loop and trainable-loop
    checkpoints and compared excitatory rate, evidence and class. This targets
    #link("/exp049/")[\#exp049 Training Recurrent Weights Weakens PING Rhythmicity].

  + *Continuous hidden state.* We ran three consecutive 50 ms inputs
    without resetting hidden neuronal state and measured each segment's excitatory
    rate. This targets the state protocol in
    #link("/exp082/")[\#exp082 Spike-Count Classification in a Continuous Stream].

  + *Duration and input-rate corners.* We sampled four paired presentation-
    duration and maximum-pixel-rate conditions and compared excitatory rates. This
    targets the operating range mapped in
    #link("/exp082/")[\#exp082 Spike-Count Classification in a Continuous Stream].

  For reduced-circuit rates, we excluded the first 50 ms and averaged across
  neurons and remaining time; full-network rates used the complete presentation.
  We recorded Brian2 minus snnsim, absolute and relative distance, evidence mean
  absolute error and class agreement where applicable. Each figure displays the
  backend measurements in panel (A) and their signed difference in panel (B).

  #run-view("exp111", inputs)

  Full-network probes used deterministic Poisson encodings of individual images
  sampled from the held-out MNIST test split. This experiment performed no model
  selection; the upstream exp022 training and validation procedure owned its
  selection split and retained model bank. Primitive and reduced-circuit checks
  used no image dataset.

  #reference-list((
    (text: [Stimberg, M., Brette, R. & Goodman, D. F. M. — _Brian 2, an intuitive and efficient neural simulator_. eLife 8:e47314, 2019.], doi: "10.7554/eLife.47314"),
  ))
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file,
    inputs,
    [Where do snnsim and Brian2 agree across the mechanisms used by the gamma-gated-sparsity manuscript?],
    preview-figures,
  )
}

#let meta = meta + (assets: input-assets("exp111", inputs))
#let body = with-datasets("exp111", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-numbered-equations(body)
#let body = with-contents(body)
