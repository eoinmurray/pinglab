#import "contents.typ": contents-here, with-contents, result-card, with-numbered-equations, with-result-sections
#import "/.demolab/lib.typ": data-image, data-json, cite, reference-list
#import "dataset-template.typ": data-file, input-assets, inputs-ready, pending-report, run-view, with-datasets
#let data-file = data-file.with(article: "exp111")

#let meta = (
  tags: ("data", "v35.0.0"),
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
  manuscript survive reimplementation in Brian2.#cite(1) The checks moved from isolated
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
  We first checked the basic voltage calculation on its own, before adding
  spikes or a network. Eight combinations covered both excitatory-like and
  inhibitory-like conductances.

  snnsim and Brian2 produced the same voltage traces to displayed precision; the
  largest difference was #rounded(t.maximum_absolute_difference, digits: 14) mV.
  The basic passive update matches for the tested settings.

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
  We drove one neuron hard enough to spike repeatedly. This checks whether both
  simulators cross threshold, emit a spike, reset the voltage and enforce the
  refractory pause at the same moments.

  Both produced the same six spikes at the same times, and their voltages differed
  by at most #rounded(t.maximum_absolute_difference, digits: 14) mV. The complete
  spike-and-reset sequence matches in this test.

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
  Synaptic events briefly raise AMPA or GABA conductance, which then fades over
  time. We sent the same six events through seven decay settings to check this
  rule without any network feedback.

  The traces matched to displayed precision, with a largest difference of
  #rounded(t.maximum_absolute_difference, digits: 15) µS. Synaptic decay itself
  therefore does not explain the later full-network differences.

  #figure(
    data-image(data-file("exp111/synapse-impulses.svg"), width: 92%, alt: "Exponential synaptic impulse responses across decay times in snnsim and Brian2, with residuals."),
    caption: [*(A)* Final conductance after the same six-event train across seven
      decay times. *(B)* Brian2 minus snnsim conductance.],
  )
  ]

  #result-card[
  === Event scheduling through the E–I loop

  #let t = test("event-causality")
  We used a small loop of 20 excitatory and 5 inhibitory neurons. Excitatory
  spikes should activate inhibition before that inhibition feeds back, so we
  checked both firing rates and the first E-to-I delay.

  snnsim and Brian2 both produced an 18.00 Hz excitatory rate, an 86.67 Hz
  inhibitory rate and a 0.5 ms delay. The short E→I→E event order matches; this
  brief test does not claim that long network histories remain identical.

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
  We varied the two connections in the small E–I loop together, from switched
  off to full strength. This catches reversed connection matrices or incorrectly
  scaled weights by looking at the resulting excitatory firing rate.

  The rate fell from 209.33 to 18.00 Hz as the loop strengthened, and both
  simulators matched at every tested strength. Their connection direction and
  scaling therefore agree in this circuit.

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
  We held the external drive fixed and compared the small circuit with its
  inhibitory loop off and on. A rhythm score measured how strongly the
  excitatory spikes repeated in time.

  Both simulators gave the same score in both conditions and the same change when
  the loop was enabled. The reduced loop matches, although this does not yet test
  a trained full network.

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
  We changed the constant input to the small E–I circuit across four levels.
  This tests whether the simulators preserve a response curve rather than merely
  agreeing at one input strength.

  Both showed the same initial plateau and the same later rise from 18.00 to
  23.75 Hz. The input–response curve matches at the four sampled points.

  #figure(
    data-image(data-file("exp111/input-response.svg"), width: 92%, alt: "Reduced-circuit excitatory rates across four drive levels in snnsim and Brian2, with residuals."),
    caption: [*(A)* Excitatory population-mean firing rate across relative tonic
      drive. *(B)* Brian2 minus snnsim rate.],
  )
  ]

  #result-card[
  === Coupling-plane onset proxy

  #let t = test("coupling-onset")
  We gradually strengthened both sides of the small recurrent loop and tracked
  the rhythm score. This is a simple slice through the wider question of when
  rhythmic activity appears as coupling changes.

  Both simulators returned the same score at all five strengths. This
  one-dimensional coupling sweep matches, but it does not cover every possible
  combination of the two connection strengths.

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
  We removed the recurrent loop entirely, then supplied lower private-like and
  higher shared-like input. These controls show what the rhythm score reports
  when recurrent inhibition cannot organize the spikes.

  Both simulators returned the same score for each input pattern. The uncoupled
  controls match exactly.

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
  GABA decay controls how long inhibition lasts and can change the circuit's
  dominant rhythm. We tested six decay times and selected the strongest frequency
  in the excitatory spike pattern.

  Both simulators selected the same frequency at every decay time. The coarse
  frequency estimate matches, though the complete spectra may still differ.

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
  We asked how often an active excitatory neuron fired exactly once during each
  inhibitory cycle. The cycles were inferred from peaks in the inhibitory
  population, then checked across the GABA-decay sweep.

  Both simulators returned 100% at every decay time. They agree exactly, but a
  score stuck at its maximum cannot reveal smaller differences.

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
  We replayed two saved versions of the same trained PING network: the version
  chosen by validation and the version at the end of training. Both saw the same
  held-out MNIST image, without retraining.

  Brian2 fired 2.62–3.07 Hz faster, so the absolute rates do not match. However,
  evidence MAE remained #rounded(point(t, "selected").evidence_mae, digits: 5)
  and #rounded(point(t, "final").evidence_mae, digits: 5), and both predicted
  class 4. The decision survived the simulator change for this one image.

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
  We compared a saved feedforward COBA network with a saved PING network that
  includes the recurrent inhibitory loop. Both saw the same held-out MNIST image,
  without retraining.

  The COBA rates differed by 89.02 Hz and the PING rates by 3.16 Hz, so absolute
  firing was backend-dependent. Both still predicted class 5 and showed a large
  rate reduction from COBA to PING, despite evidence MAEs of
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
  We added the saved PING recurrent loop to a saved COBA feedforward network,
  then tested the loop at zero, half and full strength. This isolates what the
  inhibitory loop does after training.

  Both simulators showed strong firing-rate suppression as the loop strengthened,
  but they differed by 74.51 Hz with the loop off. They also predicted different
  classes at half and full strength, so only the broad suppression effect matched.

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
  Randomly deleting or adding spikes could create a false simulator difference
  if each backend received a different random draw. We therefore made each
  perturbed spike train once and gave the same train to both.

  The event counts matched under half deletion, full deletion and both added-noise
  conditions. This confirms equal inputs, not equal network responses; those
  responses were not tested here.

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
  Inhibitory jitter moves recorded spike times while keeping the number of spikes
  fixed. We made fixed-window and neuron-specific jitter inputs once, then passed
  the same inputs to both simulators.

  Both received 220 inhibitory events in each condition. The input counts match;
  this card does not compare the network responses.

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
  The integration timestep controls how finely the simulation advances. We
  replayed networks trained for 0.05, 0.1 and 1 ms steps on fixed held-out MNIST
  inputs, without retraining.

  Firing rates differed by 0.57–2.27 Hz, but every condition predicted class 1
  in both simulators. The decisions matched across the tested timesteps even
  though the rates did not match exactly.

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
  We compared one saved network whose recurrent loop stayed fixed during training
  with one whose recurrent weights were trained. Both were replayed on fixed
  held-out MNIST inputs, without further training.

  Both simulators showed a large firing-rate increase after recurrent training
  and predicted class 8. Their rates differed by 1.48–1.76 Hz, so the broad
  training effect matched but the exact values did not.

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
  We presented three 50 ms inputs back to back without resetting the hidden
  neurons between them. This checks whether both simulators carry state through
  a continuous stream in a similar way.

  Both showed the same rising rate pattern, but their gap grew from 0.29 to
  1.02 Hz across the stream. One short stream cannot show whether that drift is
  systematic.

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
  We tested four combinations of how long an input was shown and how strongly
  pixels drove spikes. This samples the operating range of the continuous-stream
  network.

  The largest rate difference was #rounded(t.maximum_absolute_difference) Hz, so
  the four tested conditions matched closely. Because duration and input strength
  changed together, this test cannot separate their individual effects.

  #figure(
    data-image(data-file("exp111/duration-rate.svg"), width: 92%, alt: "Excitatory firing rates for four presentation-duration and input-rate combinations in snnsim and Brian2."),
    caption: [*(A)* Excitatory population-mean firing rate for four paired
      presentation-duration and maximum-pixel input-rate conditions. *(B)*
      Brian2 minus snnsim rate.],
  )
  ]

  ]

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
