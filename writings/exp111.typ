#import "contents.typ": contents-here, with-contents, result-card, with-numbered-equations, with-result-sections
#import "/.demolab/lib.typ": data-image, data-json, cite, reference-list
#import "run-inputs.typ": data-file, input-assets, inputs-ready, pending-report
#import "run-view.typ": run-view, with-datasets
#let data-file = data-file.with(article: "exp111")

#let meta = (
  status: "[▦ DATA | v34.0.1]",
  title: "Brian2 Comparison",
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
  The passive LIF update determines how leak and synaptic conductances move a
  neuron's membrane voltage before any spike threshold is applied. We compared
  eight excitatory-like and inhibitory-like conductance settings to test this
  numerical foundation independently of network activity.

  Across eight sampled cases, snnsim and Brian2 produced the same final voltages
  to displayed precision, spanning approximately −55.36 to −66.65 mV; their
  maximum difference was #rounded(t.maximum_absolute_difference, digits: 14) mV.
  The tested passive update therefore agrees at floating-point scale, although
  the comparison does not cover unsampled parameter settings.

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
  A spiking LIF neuron must cross threshold, emit an event, reset its voltage and
  remain refractory on the same simulation steps in both backends. We repeatedly
  drove one neuron through this complete event sequence and compared its voltage
  trajectory and spike times.

  snnsim and Brian2 each produced six spikes at 0.5, 4.0, 7.5, 11.0, 14.5 and
  18.0 ms, while their voltage traces differed by at most
  #rounded(t.maximum_absolute_difference, digits: 14) mV. Continuous voltage and
  discrete event timing therefore coincide for this drive regime; other regimes
  remain untested.

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
  AMPA and GABA conductances are incremented by presynaptic events and then decay
  between events, shaping excitation and inhibition throughout the network. We
  applied one six-event train across seven decay times to isolate this synaptic
  rule from neuronal and recurrent dynamics.

  Across the seven decay times, snnsim endpoints ranged from 0.0985 to 1.3682 µS
  and Brian2 produced the same values to displayed precision; the largest trace
  difference was #rounded(t.maximum_absolute_difference, digits: 15) µS. The
  isolated decay rule therefore agrees numerically and is unlikely to explain
  the larger discrepancies seen in full-network replays.

  #figure(
    data-image(data-file("exp111/synapse-impulses.svg"), width: 92%, alt: "Exponential synaptic impulse responses across decay times in snnsim and Brian2, with residuals."),
    caption: [*(A)* Final conductance after the same six-event train across seven
      decay times. *(B)* Brian2 minus snnsim conductance.],
  )
  ]

  #result-card[
  === Event scheduling through the E–I loop

  #let t = test("event-causality")
  In the reduced excitatory–inhibitory loop, excitatory spikes must reach the
  inhibitory population before inhibition feeds back to excitation. We compared
  population rates and the first excitatory-to-inhibitory lag to test whether
  both simulators preserve this short-timescale causal order.

  snnsim and Brian2 both produced an 18.00 Hz excitatory rate, an 86.67 Hz
  inhibitory rate and a 0.5 ms first E-to-I event lag. The intended event order
  is therefore reproduced in this reduced circuit, but the test does not show
  that long recurrent trajectories remain identical.

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
  Reciprocal E-to-I and I-to-E projections depend on the same matrix orientation,
  fan-in normalization and weight scaling in both implementations. We varied the
  two projection strengths together from a disabled loop to full coupling and
  used excitatory firing rate as the circuit-level check.

  At reciprocal scales 0, 0.5 and 1, snnsim measured 209.33, 40.00 and 18.00 Hz,
  and Brian2 measured the same three rates. The exact match spans a 191.33 Hz
  intervention effect, supporting the tested projection orientation and scaling
  across substantially different operating points.

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
  This comparison asks whether enabling reciprocal inhibition changes temporal
  organization when the external drive is held fixed. We disabled and enabled
  the reduced E–I loop and measured excitatory autocorrelation lobe–trough
  contrast, a summary of oscillatory structure rather than mean firing rate.

  snnsim measured loop-off and loop-on contrasts of 1.0678 and 0.8831, and Brian2
  measured the same values. Both backends therefore reproduce the 0.1847 change
  caused by enabling the reduced loop, but this result does not establish
  agreement in trained full networks.

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
  A backend comparison should preserve the circuit's response to changing input,
  not merely one operating point. We scaled the reduced circuit's tonic drive
  across four levels and compared its excitatory population-mean firing rate.

  At relative drives 0.75, 1.0, 1.25 and 1.5, snnsim measured 18.00, 18.00,
  21.50 and 23.75 Hz, and Brian2 measured the same rates. Both the initial
  plateau and subsequent 5.75 Hz rise coincide, establishing agreement at these
  sampled drive levels but not across the unsampled response curve.

  #figure(
    data-image(data-file("exp111/input-response.svg"), width: 92%, alt: "Reduced-circuit excitatory rates across four drive levels in snnsim and Brian2, with residuals."),
    caption: [*(A)* Excitatory population-mean firing rate across relative tonic
      drive. *(B)* Brian2 minus snnsim rate.],
  )
  ]

  #result-card[
  === Coupling-plane onset proxy

  #let t = test("coupling-onset")
  Increasing reciprocal coupling can move the reduced circuit toward or away
  from an oscillatory regime. As a one-dimensional proxy for that transition, we
  scaled both recurrent pathways together and tracked excitatory autocorrelation
  lobe–trough contrast.

  At coupling scales 0, 0.25, 0.5, 0.75 and 1, both snnsim and Brian2 measured
  contrasts of 1.076, 1.110, 0.982, 0.954 and 0.765. Every sampled value
  coincides, so the backends reproduce this coupling trajectory; independently
  varying the two pathways across a full coupling plane remains untested.

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
  These null controls ask what the autocorrelation measure reports when no
  reciprocal E–I loop can organize the activity. We removed recurrent coupling
  and compared lower private-like with higher shared-like tonic drive, isolating
  drive structure from loop-generated dynamics.

  Without reciprocal coupling, snnsim measured contrasts of 0.9861 and 1.0439
  for the private-like and shared-like drives, and Brian2 measured the same two
  values. The 0.0578 difference is preserved with zero observed backend distance,
  establishing agreement for these uncoupled controls only.

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
  The inhibitory decay time changes how long GABA conductance persists and can
  alter the circuit's dominant rhythm. We swept six GABA decay times in the
  reduced circuit and compared the raw spectral-bin frequency selected from the
  excitatory population trace.

  At GABA decay times 4.5, 6, 9, 12, 18 and 27 ms, snnsim selected 53.33, 42.22,
  60.00, 68.89, 20.00 and 68.89 Hz, and Brian2 selected the same frequencies.
  The raw frequency estimate therefore agrees at every sampled decay time, but
  the coarse bin selection does not establish identical spectra or sub-bin peaks.

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
  The cycle-participation measure asks whether an active excitatory neuron fires
  once, rather than repeatedly, within each inferred inhibitory cycle. We
  evaluated this conditional fraction across the GABA-decay sweep to test whether
  both backends preserve the same within-cycle spike organization.

  At every tested decay time, snnsim and Brian2 both measured a one-spike
  conditional fraction of 1.0 and a total-variation distance of zero. The values
  agree exactly, but saturation at 1.0 makes this an insensitive backend test:
  there is no variation with which to expose subtler differences.

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
  The selected checkpoint was retained by validation, whereas the final
  checkpoint records the end of training. We replayed both canonical-PING
  checkpoints on one fixed held-out image to ask whether firing rate, readout
  evidence and the resulting class survive a change of simulator.

  At the selected checkpoint, snnsim measured 24.46 Hz and Brian2 27.08 Hz; at
  the final checkpoint they measured 24.30 and 27.36 Hz. The 2.62–3.07 Hz rate
  gaps show that absolute rates differ across backends for this replay, but
  evidence MAE remained #rounded(point(t, "selected").evidence_mae, digits: 5)
  and #rounded(point(t, "final").evidence_mae, digits: 5), and both predicted
  class 4. The decision was therefore reproduced for this sample despite the
  rate differences.

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
  The retained endpoints contrast a feedforward COBA network with a PING network
  containing reciprocal excitatory–inhibitory coupling. We replayed both on the
  same held-out image to test whether Brian2 preserves the change in firing
  regime, readout evidence and predicted class.

  For COBA, snnsim measured 237.27 Hz and Brian2 148.24 Hz; for PING they measured
  22.34 and 25.51 Hz. The large 89.02 Hz COBA gap and moderate 3.16 Hz PING gap
  show substantial backend dependence for this input. Both nevertheless
  predicted class 5 and showed a large COBA-to-PING reduction, so the qualitative
  endpoint contrast survived for this input despite evidence MAEs of
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
  This intervention combines a retained COBA feedforward checkpoint with the
  retained recurrent matrices of a canonical-PING network. Scaling those matrices
  from zero to full strength tests whether both backends preserve the inhibitory
  loop's post-training effect on firing and classification.

  At loop scales 0, 0.5 and 1, snnsim measured 207.33, 28.98 and 13.57 Hz, while
  Brian2 measured 132.82, 26.61 and 14.46 Hz. Both show strong rate suppression,
  but the 74.51 Hz open-loop gap prevents a backend-independent quantitative
  effect estimate. Predictions also differ at scales 0.5 and 1, so classification
  was not reproduced there.

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
  Spike deletion and Poisson addition are stochastic interventions, so unequal
  random draws could masquerade as simulator differences. We constructed each
  perturbed hidden-event set once and supplied that same realized input to both
  backends before comparing event counts.

  For half deletion, full deletion, 20 Hz addition and 40 Hz addition, the
  snnsim inputs contained 102, 0, 416 and 593 events, and the Brian2 inputs
  contained the same counts. This confirms count parity for the four supplied
  interventions, but does not validate simulator dynamics because their
  downstream responses were not compared.

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
  Inhibitory jitter perturbs when recorded inhibitory events are replayed while
  intending to preserve how many events enter each backend. We tested fixed-window
  and neuron-specific jitter constructions to separate input construction from
  any later dynamical response.

  Under both fixed-window and cellwise jitter, the snnsim input contained 220
  inhibitory events and the Brian2 input also contained 220. Event counts were
  therefore preserved, but equal counts alone do not establish identical event
  times or downstream trajectories.

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
  The integration timestep controls how finely neuronal and synaptic dynamics are
  discretized. We replayed networks trained and evaluated at 0.05, 0.1 and 1 ms
  to test whether backend agreement is confined to the nominal timestep or
  persists across the project's tested range.

  At 0.05, 0.1 and 1 ms, snnsim measured 9.98, 14.84 and 10.83 Hz, while Brian2
  measured 12.25, 14.28 and 9.32 Hz. The 0.57 Hz nominal-timestep gap is adequate
  for a close comparison, while the 1.51–2.27 Hz extreme-timestep gaps rule out
  identical rates across the sweep. Every condition predicted class 1, so the
  tested decisions were reproduced.

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
  Training recurrent weights can move the network into a different firing regime
  from keeping the inhibitory loop fixed. We replayed one frozen-loop and one
  trainable-loop checkpoint to ask whether the direction and size of that change,
  as well as the predicted class, survive reimplementation.

  With frozen recurrence, snnsim measured 18.32 Hz and Brian2 20.08 Hz; with
  trained recurrence they measured 57.70 and 56.21 Hz. These 1.48–1.76 Hz gaps
  prevent an exact pointwise match, but both backends show an approximately
  36–39 Hz increase and predict class 8. The broad effect of recurrent training
  was therefore reproduced for these checkpoints.

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
  Continuous-stream inference carries hidden neuronal state from one input into
  the next instead of resetting at presentation boundaries. We ran three
  consecutive 50 ms inputs without hidden-state resets and compared the
  excitatory rate within each segment.

  Across segments 1–3, snnsim measured 7.79, 8.54 and 13.75 Hz, while Brian2
  measured 7.50, 8.22 and 12.73 Hz. Both reproduce the rising segment-wise
  profile, but the gap grows from 0.29 to 1.02 Hz. One stream cannot determine
  whether that separation reflects systematic accumulated trajectory divergence.

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
  Presentation duration and maximum input rate jointly determine how much spike
  drive reaches a continuous-stream network. We replayed four paired corners of
  those settings to test whether both simulators preserve the resulting range of
  excitatory firing rates.

  Across the four duration–rate corners, snnsim measured 0.00, 16.37, 5.41 and
  6.16 Hz, while Brian2 measured 0.00, 16.70, 5.41 and 6.04 Hz. The maximum gap
  was #rounded(t.maximum_absolute_difference) Hz, so the sampled corner rates
  were closely reproduced. Because duration and input rate changed together,
  this design cannot attribute differences to either factor alone.

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
    #link("/exp110/")[_Manuscript (exp110)_].

  + *Threshold, reset and refractory.* We drove one neuron through repeated
    spikes and compared voltage, spike count and spike times. This checks the event
    mechanism shared by the trained-network experiments synthesised in
    #link("/exp110/")[_Manuscript (exp110)_].

  + *AMPA and GABA impulse responses.* We applied the same six-event train
    across seven conductance-decay times and compared complete conductance traces.
    This checks the synaptic dynamics producing excitation, inhibition and PING
    rhythms throughout #link("/exp110/")[_Manuscript (exp110)_].

  + *E–I event scheduling.* We compared population rates and the first
    excitatory-to-inhibitory event lag in a reduced reciprocal circuit. This checks
    the causal E→I→E sequence interpreted in
    #link("/exp023/")[_Turning the PING Loop On (exp023)_],
    #link("/exp033/")[_Gamma Emerges at a Hopf Bifurcation (exp033)_] and
    #link("/exp054/")[_Pinglab Rythmicity Metric (exp054)_].

  + *Projection and weight scaling.* We scaled both reciprocal projections
    from zero to full strength and compared excitatory rates. This checks matrix
    orientation and loop scaling used in
    #link("/exp023/")[_Turning the PING Loop On (exp023)_],
    #link("/exp033/")[_Gamma Emerges at a Hopf Bifurcation (exp033)_],
    #link("/exp038/")[_Switching On the Inhibitory Loop (exp038)_] and
    #link("/exp054/")[_Pinglab Rythmicity Metric (exp054)_].

  + *Matched-drive loop comparison.* We reproduced reduced loop-off and
    loop-on conditions under matched tonic drive and measured autocorrelation
    lobe–trough contrast. This targets the recurrent-organisation comparison in
    #link("/exp023/")[_Turning the PING Loop On (exp023)_].

  + *Input-response curve.* We varied relative tonic drive across four
    levels and measured excitatory population rates. This targets the input-sweep
    evidence used to distinguish COBA and PING operating regimes in
    #link("/exp023/")[_Turning the PING Loop On (exp023)_].

  + *Coupling-onset proxy.* We varied reciprocal coupling over
    five points and measured autocorrelation contrast. This samples the oscillatory
    onset described in #link("/exp033/")[_Gamma Emerges at a Hopf Bifurcation (exp033)_]
    and the empirical rhythmicity map defined in
    #link("/exp054/")[_Pinglab Rythmicity Metric (exp054)_].

  + *Uncoupled nulls.* We disabled reciprocal coupling and compared
    private-like and shared-like drive controls using autocorrelation contrast.
    This checks the null behaviour of the measure used by
    #link("/exp054/")[_Pinglab Rythmicity Metric (exp054)_].

  + *GABA timescale and frequency.* We varied inhibitory conductance
    decay across six values and selected the largest raw spectral-power bin from
    5–150 Hz. This targets the rate–gamma-frequency relationship in
    #link("/exp041/")[_Firing Rate Tracks Gamma Frequency (exp041)_].

  + *Cycle participation.* We inferred inhibitory cycles from peaks in
    a 1 ms Gaussian-smoothed inhibitory trace and counted excitatory spikes per
    active neuron–cycle pair. This targets the account in
    #link("/exp046/")[_One Spike per Gamma Cycle (exp046)_].

  + *Checkpoint replay.* We replayed the selected and final canonical-PING
    checkpoints under one fixed held-out input and compared excitatory rate,
    normalised readout evidence and class. This targets the convergence claim in
    #link("/exp024/")[_Accuracy Plateaus While Firing Rate Rises (exp024)_].

  + *COBA and PING endpoints.* We replayed retained COBA and PING
    checkpoints under the same held-out input and compared rate, evidence and class.
    This targets the accuracy–firing-rate comparison in
    #link("/exp025/")[_Accuracy and Firing Rate With and Without Inhibition (exp025)_].

  + *Loop-strength transfer.* We combined a retained COBA feedforward
    checkpoint with canonical-PING recurrent matrices at three inference-time scales.
    This targets post-training activation of inhibition in
    #link("/exp038/")[_Switching On the Inhibitory Loop (exp038)_].

  + *Hidden-spike perturbation inputs.* We froze realised half-deletion,
    full-deletion and two Poisson-addition event sets before backend dispatch. This
    checks intervention parity for #link("/exp037/")[_Dropped Spikes vs Added Noise (exp037)_]
    without replaying its downstream responses.

  + *Inhibitory-jitter inputs.* We supplied fixed-window and cellwise
    jitter constructions with preserved inhibitory-event counts. This checks input
    parity for #link("/exp042/")[_Inhibitory Replay Perturbations Change Excitatory Firing (exp042)_]
    without reproducing its downstream excitatory response.

  + *Integration timestep.* We replayed networks matched to 0.05, 0.1
    and 1 ms training-and-inference timesteps and compared rate, evidence and class.
    This targets the operating range reported in
    #link("/exp044/")[_Firing Rate Across the Timestep Sweep (exp044)_].

  + *Recurrent-weight training.* We replayed frozen-loop and trainable-loop
    checkpoints and compared excitatory rate, evidence and class. This targets
    #link("/exp049/")[_Training Recurrent Weights Weakens PING Rhythmicity (exp049)_].

  + *Continuous hidden state.* We ran three consecutive 50 ms inputs
    without resetting hidden neuronal state and measured each segment's excitatory
    rate. This targets the state protocol in
    #link("/exp082/")[_Spike-Count Classification in a Continuous Stream (exp082)_].

  + *Duration and input-rate corners.* We sampled four paired presentation-
    duration and maximum-pixel-rate conditions and compared excitatory rates. This
    targets the operating range mapped in
    #link("/exp082/")[_Spike-Count Classification in a Continuous Stream (exp082)_].

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
