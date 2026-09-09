#import "templates/article-layout.typ": journal-article
#import "templates/result-card.typ": journal-result-card, with-result-sections, result-figure-ref
#import "templates/methods.typ": journal-methods, method-card
#import "templates/parameters-table.typ": parameters-table
#import "/.demolab/lib.typ": data-image
#import "templates/dataset.typ": video, data-file, inputs-ready, pending-report, run-view, input-assets
#import "templates/abstract.typ": journal-abstract
#let data-file = data-file.with(article: "exp099")

#let meta = (
  tags: ("data", "v36.0.0"),
  // Author-locked title: do not change.
  title: "Susin and Destexhe (2021)",
  created_at: "2026-08-26T00:00:00Z",
  updated_at: "2026-09-09",
  description: "Independent excitatory afferent drive strengthens population bursting in a calibrated conductance-based E/I network.",
  collection: "demo",
  order: 13,
)

#let inputs = ("exp099",)

#let render-report(data-file) = [
  #journal-abstract(
    question: [We conducted a modified replication of Susin and Destexhe’s (2021) PING-network experiment to test whether increased nonrhythmic afferent excitation recruits collective oscillations.],
    approach: [Using a smaller conductance-based LIF network with independent excitatory inputs, we increased drive onto excitatory neurons alone.],
    finding: [In a single simulation, low-rate irregular activity developed stronger population bursting.],
    scope: [The response links increased independent excitation to stronger collective activity in this calibrated network.],
  )

  == Results

  #with-result-sections[

    #journal-result-card(
      title: "Implemented circuit architecture",
      orientation: [Our circuit comprises 1,600 excitatory and 400 inhibitory neurons connected through recurrent AMPA and GABA synapses (#result-figure-ref(<fig:exp099-network>)). Each neuron receives independent excitatory Poisson input, and recurrent I neurons supply inhibition.],
      visual: [
        #figure(
          data-image(
            data-file("exp099/network.svg"),
            width: 100%,
            alt: "Structural diagram of 1,600 excitatory and 400 inhibitory neurons with independent private excitatory afferents and recurrent AMPA and GABA projections.",
          ),
          caption: [Structural schematic of the implemented populations, recurrent AMPA and GABA projections, and private excitatory afferents. Rates are per afferent source; weights are conductance increments per event.],
          kind: image,
          supplement: [Figure],
        ) <fig:exp099-network>
      ],

    )

    #let private-clip = data-file("exp099/private-e-drive.mp4")
    #if private-clip != none { let _ = read(private-clip, encoding: none) }
    #journal-result-card(
      title: "Private drive strengthens bursting",
      observation: [A 50% increase in E-targeted afferent rate, with I-targeted input held constant, strengthened population bursting (#result-figure-ref(<fig:exp099-private>)). In the single simulation, baseline firing averaged 3.09 Hz in E and 3.22 Hz in I, rising to 23.56 Hz in E and 30.01 Hz in I during sustained stimulation. Recovery firing averaged 2.53 Hz in both populations; the video magnifies the transition onset.

      Baseline activity combined irregular individual spiking with intermittent population bursts. Increased drive strengthened this collective bursting.],
      visual: [
        #figure(
          video(private-clip),
          caption: [Single-seed transition detail over 3,500–5,000 ms, shown over 25 seconds of playback. E-targeted afferent rates rose from 0.6 to 0.9 Hz during 4,000–4,500 ms, then held at 0.9 Hz throughout the remainder of the displayed interval; I-targeted rates remained at 0.6 Hz. A shows sampled neurons and connections, with private input dots representing target-specific aggregate streams. B shows population mean voltages and conductances onto E; C shows their 40 ms conductance trail. D shows per-neuron firing rates with a trailing 20 ms average; E shows prescribed per-source input rates. F shows fixed nonzero recurrent weights.],
          kind: image,
          supplement: [Figure],
        ) <fig:exp099-private>
      ],

    )

  ]

  #journal-methods(body: (
    method-card([Adapt the reference model], [We performed a modified replication of #link("https://doi.org/10.1371/journal.pcbi.1009416")[Susin and Destexhe’s (2021)] PING-network experiment. We reduced network size and replaced adapting AdEx neurons with conductance-based leaky integrate-and-fire (LIF) neurons. @tab:exp099-parameters compares the reference and implemented parameters and explains the differences.]),
    method-card([Construct the circuit], [We connected 1,600 E and 400 I neurons independently with 10% probability, using recurrent AMPA and GABA synapses (#result-figure-ref(<fig:exp099-network>)). Synaptic weights, decay times and delays are specified in @tab:exp099-parameters.]),
    method-card([Calibrate baseline input], [We selected low-rate spiking settings through exploratory calibration. Each neuron received 400 equivalent independent excitatory Poisson afferents (#result-figure-ref(<fig:exp099-network>); @tab:exp099-parameters).]),
    method-card([Increase excitatory drive], [During a 10-second simulation, E-targeted afferent rates rose from 0.6 to 0.9 Hz over 4–4.5 seconds, remained elevated until 6.5 seconds and returned by 7 seconds. I-targeted rates remained fixed. #result-figure-ref(<fig:exp099-private>, panel: "E") displays the input schedule around onset.]),
    method-card([Measure network responses], [We used one simulation with random seed 7 and measured firing rates, interspike-interval variability and pairwise spike-count correlations during baseline (1–4 seconds), stimulation plateau (4.5–6.5 seconds) and recovery (7–10 seconds). We summarized variability as the median interspike-interval coefficient of variation across cells with at least five spikes, and synchrony as mean pairwise Pearson correlation of 10 ms spike counts among at most 100 evenly sampled cells with nonzero count variance per population. #result-figure-ref(<fig:exp099-private>, panel: "B–D") displays population means, conductance trajectories and firing rates for this simulation; displayed rates use a trailing 20 ms average.]),
    method-card([Present the transition], [We used SNNLang to specify the network, SNNSim to execute it and SNNViz to support #result-figure-ref(<fig:exp099-network>) and #result-figure-ref(<fig:exp099-private>). #result-figure-ref(<fig:exp099-private>) magnifies 3.5–5 seconds into 25 seconds of playback; #result-figure-ref(<fig:exp099-private>, panel: "F") shows the fixed recurrent weights listed in @tab:exp099-parameters.]),
  ))

  #parameters-table(
    ([Variable], [Their value], [Our value], [Difference and why]),
    (
      ([Neurons E / I], [20,000 / 5,000], [1,600 / 400], [Smaller computational budget; same ratio.]),
      ([Recurrent connection probability], [2%], [10%], [Partly offsets fewer neurons.]),
      ([Mean recurrent inputs E / I], [400 / 100], [160 / 40], [Lower recurrent input counts at the reduced network size.]),
      ([Neuron model], [Adaptive exponential (AdEx); adapting E], [Conductance LIF; no adaptation], [Simpler membrane dynamics.]),
      ([Capacitance], [150 pF], [150 pF], [Matched.]),
      ([Leak conductance], [10 nS], [10 nS], [Matched; membrane time constant 15 ms.]),
      ([Rest / reset voltage], [−65 mV], [−65 mV], [Matched.]),
      ([Threshold E / I], [−40 / −47.5 mV; effective AdEx threshold], [−50 / −50 mV; hard threshold], [Retained LIF hard thresholds; the reference uses AdEx effective thresholds.]),
      ([Refractory period E / I], [5 / 5 ms], [3 / 1.5 ms], [Retained existing LIF settings.]),
      ([AMPA / GABA decay], [1.5 / 7.5 ms], [1.5 / 7.5 ms], [Matched.]),
      ([Synaptic delay], [1.5 ms], [1.5 ms], [Matched and implemented.]),
      ([Timestep], [0.1 ms], [0.1 ms], [Matched.]),
      ([External afferents per neuron], [400 on average; some shared sources], [400 equivalent independent sources], [Independent target-specific input streams.]),
      ([Baseline source rate E / I], [2 / 2 Hz], [0.6 / 0.6 Hz], [Lowered to avoid excessive baseline bursting in our LIF network.]),
      ([Increased drive], [3 / 3 Hz condition], [0.9 / 0.6 Hz], [E-only increase isolates recruitment through E.]),
      ([External excitatory weight], [4 nS], [4 nS], [Matched.]),
      ([Recurrent excitatory weight], [5 nS], [1.25 nS], [Weakened to control excessive recurrent excitation.]),
      ([Recurrent inhibitory weight], [3.34 nS], [3.34 nS], [Matched; stronger relative to recurrent excitation.]),
      ([Separate external GABA], [None], [None], [Inhibition comes from recurrent I neurons.]),
    ),
    columns: (1.1fr, 1fr, 1fr, 1.5fr),
    table-label: <tab:exp099-parameters>,
    caption: [Reference PING-network settings from #link("https://doi.org/10.1371/journal.pcbi.1009416")[Susin and Destexhe (2021)] and our calibrated conductance-based LIF settings. Paired values are ordered E / I; mean recurrent input counts are expectations under the stated connection probabilities. Synaptic weights are conductance increments per presynaptic event; external rates are per afferent source. The comparison describes a modified replication with model-specific threshold definitions: effective thresholds for AdEx and hard thresholds for LIF.],
  )

  #run-view("exp099", inputs)
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(data-file, inputs, [], ())
}

#let meta = meta + (assets: input-assets("exp099", inputs))
#let body = journal-article("exp099", inputs, report-body, dataset-placed: inputs-ready(data-file, inputs))
