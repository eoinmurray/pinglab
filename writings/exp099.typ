#import "templates/article-layout.typ": journal-article
#import "templates/result-card.typ": journal-result-card, with-result-sections, result-figure-ref, result-card
#import "templates/methods.typ": journal-methods, method-card
#import "templates/parameters-table.typ": parameters-table
#import "templates/references.typ": journal-references
#import "/.demolab/lib.typ": data-image, cite
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
    approach: [Using a smaller conductance-based LIF network with independent excitatory inputs, we increased afferent excitation onto both populations.],
    finding: [In a single simulation, low-rate irregular activity developed stronger population bursting.],
    scope: [The response links increased independent excitation to stronger collective activity in this calibrated network.],
  )

  == Introduction

  Susin and Destexhe (2021) examined how collective gamma oscillations emerge in spiking networks and affect their response to external inputs. In their pyramidal–interneuron gamma (PING) network, increasing the rate of excitatory Poisson afferents shifted activity from an asynchronous-irregular regime toward gamma oscillations. The afferents supplied nonrhythmic excitation, while recurrent interactions between excitatory and inhibitory neurons generated the collective rhythm.#cite(1)

  Here, we attempt to recreate this input-driven transition through a modified replication. We use a smaller network of conductance-based leaky integrate-and-fire neurons, retain the reference synaptic decay times and delays, and provide independent excitatory afferents to each neuron. We recalibrate background drive and recurrent weights, then increase afferent excitation onto both populations together. This tests how the recurrent circuit responds to the reference study’s pattern of increased external drive.

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
      observation: [A 50% increase in afferent rates onto both E and I strengthened population bursting (#result-figure-ref(<fig:exp099-private>)). In the single simulation, firing during the visible 200 ms baseline averaged 2.50 Hz in E and 2.58 Hz in I, rising to 10.00 Hz in E and 10.85 Hz in I during the 100 ms plateau. Recovery rates averaged 6.03 Hz in E and 6.81 Hz in I. The video shows 600 ms following a hidden 500 ms baseline.

      Dispersed baseline spiking developed pronounced population bursts during the input pulse.],
      visual: [
        #figure(
          video(private-clip),
          caption: [Single-seed simulation shown over 25 seconds of playback. Displayed time 0–600 ms follows a hidden 500 ms baseline. E- and I-targeted afferent rates rose from 0.6 to 0.9 Hz during 200–250 ms, held until 350 ms, and returned to 0.6 Hz during 350–400 ms. A shows sampled neurons and connections, with private input dots representing target-specific aggregate streams. B shows population mean voltages and conductances onto E; C shows their 40 ms conductance trail. D shows per-neuron firing rates with a trailing 20 ms average; E shows prescribed per-source input rates. F shows fixed nonzero recurrent weights.],
          kind: image,
          supplement: [Figure],
        ) <fig:exp099-private>
      ],

    )

    #journal-result-card(
      title: "E and I bursts overlap in time",
      notes: [
        - The E and I burst profiles overlap strongly in the close-up; direct external excitation of I is a candidate contributor. In a separate diagnostic run with external I excitation removed, I peaks followed E peaks by approximately 1 ms across nine detected plateau bursts.
        - In the earlier 1.5-second protocol, a PING-like population burst occurred around 280 ms, before the input ramp at 500 ms. Such pre-ramp bursting is qualitatively expected from the reference study: Susin and Destexhe report spontaneous gamma bursts driven by fluctuations in recurrent activity, alongside residual oscillations in the AI-like baseline.#cite(1)
      ],
      observation: [Excitatory and inhibitory neurons formed aligned population bursts during increased afferent drive (#result-figure-ref(<fig:exp099-raster>, panel: "B")). In the plateau close-up, the E and I spike-count profiles overlap strongly, with variable burst amplitudes and timing (#result-figure-ref(<fig:exp099-raster>, panel: "C–D")). The narrow vertical bands reflect coordinated firing across neurons, interspersed with more dispersed spikes.],
      visual: [
        #figure(
          data-image(
            data-file("exp099/spike-raster.png"),
            width: 100%,
            alt: "Four-panel plot showing afferent rates, all E and I spikes over 600 ms, a 250–350 ms raster close-up, and unsmoothed population spike counts in 1 ms bins.",
          ),
          caption: [Spike timing in the same seed-7 simulation as #result-figure-ref(<fig:exp099-private>), with external excitation onto both populations. Time starts after the hidden 500 ms baseline. A shows E- and I-targeted afferent rates; the coincident schedules are drawn as black solid and red dashed lines. B shows every spike from all 1,600 E neurons (black, lower rows) and 400 I neurons (red, upper rows). Dotted vertical lines mark pulse onset and completion; the shaded interval identifies the close-up. C shows 250–350 ms, a fixed 100 ms window centred on the plateau midpoint. D shows unsmoothed spike counts in 1 ms bins over the same window. Counts are population totals; the E population contains four times as many neurons as I.],
          kind: image,
          supplement: [Figure],
        ) <fig:exp099-raster>
      ],
    )

    #result-card[
      === Reference PING network

        #figure(
          data-image(
            data-file("exp099/susin-destexhe-2021-ping.png"),
            width: 55%,
            alt: "Published PING column from Susin and Destexhe Figure 4, showing panels B and C: external Poisson drive and the spike raster.",
          ),
          caption: [Cropped from Figure 4 (left PING column, panels B and C) of Susin and Destexhe (2021).#cite(1) © 2021 Susin and Destexhe; reproduced under #link("https://creativecommons.org/licenses/by/4.0/")[CC BY 4.0]. B shows external Poisson drive; C shows the spike raster. Original panel labels and colours are retained.],
          kind: image,
          supplement: [Figure],
        ) <fig:exp099-reference>

        Our modified replication shares the drive-dependent response, with three main points of comparison:

        - Increased excitation onto E and I strengthens collective activity in both networks; ours has sharper burst bands (#result-figure-ref(<fig:exp099-reference>, panel: "B–C"); #result-figure-ref(<fig:exp099-raster>, panel: "A–B")).
        - Their raster samples 1,000 E and 1,000 I neurons from 25,000; ours shows all 1,600 E and 400 I neurons.
        - Both show 600 ms with a 200 ms pulse. Their drive is 2→3→2 Hz during ongoing activity; ours is 0.6→0.9→0.6 Hz after a hidden 500 ms baseline.

        Sharper bursts may reflect our non-adapting LIF neurons, altered thresholds and refractory periods, fewer recurrent inputs, recalibrated weights and lower background drive (@tab:exp099-parameters). Sampling also affects raster density. These remain candidate explanations requiring controlled comparisons.

    ]

  ]

  #journal-methods(body: (
    method-card([Adapt the reference model], [We performed a modified replication of Susin and Destexhe’s (2021) PING-network experiment.#cite(1) We reduced network size and replaced adapting AdEx neurons with conductance-based leaky integrate-and-fire (LIF) neurons. @tab:exp099-parameters compares the reference and implemented parameters and explains the differences.]),
    method-card([Construct the circuit], [We connected 1,600 E and 400 I neurons independently with 10% probability, using recurrent AMPA and GABA synapses (#result-figure-ref(<fig:exp099-network>)). Synaptic weights, decay times and delays are specified in @tab:exp099-parameters.]),
    method-card([Calibrate baseline input], [We selected low-rate spiking settings through exploratory calibration. Each neuron received 400 equivalent independent excitatory Poisson afferents (#result-figure-ref(<fig:exp099-network>); @tab:exp099-parameters).]),
    method-card([Increase excitatory drive], [We initialized voltages at −65 mV and simulated 500 ms of baseline before the displayed 600 ms window, preserving network state throughout. In displayed time, both afferent rates rose from 0.6 to 0.9 Hz over 200–250 ms, held until 350 ms, then returned to 0.6 Hz over 350–400 ms. #result-figure-ref(<fig:exp099-private>, panel: "E") displays the input schedule around onset.]),
    method-card([Measure network responses], [We used one simulation with random seed 7 and measured firing rates, interspike-interval variability and pairwise spike-count correlations during the visible baseline (0–200 ms), stimulation plateau (250–350 ms) and recovery (400–600 ms). We summarized variability as the median interspike-interval coefficient of variation across cells with at least five spikes, and synchrony as mean pairwise Pearson correlation of 10 ms spike counts among at most 100 evenly sampled cells with nonzero count variance per population. #result-figure-ref(<fig:exp099-private>, panel: "B–D") displays population means, conductance trajectories and firing rates for this simulation; displayed rates use a trailing 20 ms average.]),
    method-card([Present the transition], [We used SNNLang to specify the network, SNNSim to execute it and SNNViz to support #result-figure-ref(<fig:exp099-network>) and #result-figure-ref(<fig:exp099-private>). #result-figure-ref(<fig:exp099-private>) magnifies 0–600 ms into 25 seconds of playback; #result-figure-ref(<fig:exp099-private>, panel: "F") shows the fixed recurrent weights listed in @tab:exp099-parameters.]),
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
      ([Increased drive], [3 / 3 Hz condition], [0.9 / 0.9 Hz], [Both populations receive a 50% increase, matching the reference stimulation pattern.]),
      ([External excitatory weight], [4 nS], [4 nS], [Matched.]),
      ([Recurrent excitatory weight], [5 nS], [1.25 nS], [Weakened to control excessive recurrent excitation.]),
      ([Recurrent inhibitory weight], [3.34 nS], [3.34 nS], [Matched; stronger relative to recurrent excitation.]),
      ([Separate external GABA], [None], [None], [Inhibition comes from recurrent I neurons.]),
    ),
    columns: (1.1fr, 1fr, 1fr, 1.5fr),
    table-label: <tab:exp099-parameters>,
    caption: [Reference PING-network settings from Susin and Destexhe (2021)#cite(1) and our calibrated conductance-based LIF settings. Paired values are ordered E / I; mean recurrent input counts are expectations under the stated connection probabilities. Synaptic weights are conductance increments per presynaptic event; external rates are per afferent source. The comparison describes a modified replication with model-specific threshold definitions: effective thresholds for AdEx and hard thresholds for LIF.],
  )

  #run-view("exp099", inputs)

  #journal-references((
    (text: [E. Susin and A. Destexhe. “Integration, coincidence detection and resonance in networks of spiking neurons expressing Gamma oscillations and asynchronous states.” _PLOS Computational Biology_ 17(9), e1009416 (2021).], doi: "10.1371/journal.pcbi.1009416"),
  ))
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(data-file, inputs, [], ())
}

#let meta = meta + (assets: input-assets("exp099", inputs))
#let body = journal-article("exp099", inputs, report-body, dataset-placed: inputs-ready(data-file, inputs))
