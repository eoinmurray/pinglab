#let meta = (
  title: "When does recurrent inhibition turn excitation into PING-like activity?",
  created_at: "2026-08-25",
  updated_at: "2026-08-25",
  description: "A simulated scout comparing intact PING dynamics, E-to-I ablation, and increasing external drive.",
  collection: "demo",
  order: 12,
  status: "ExpScout",
)

#let r = json("/artifacts/data/exp098/numbers.json")
#let result = r.results
#let loop-video(src) = context {
  if target() == "html" {
    html.elem("video", attrs: (src: src, controls: "", loop: "", playsinline: "", style: "max-width:100%;width:100%"))[]
  } else {
    text(size: 9pt, style: "italic", fill: gray)[[Video — view the web edition to play.]]
  }
}

#let body = [
  == 1. Abstract

  This exploratory simulation scout tests whether recurrent recruitment of inhibitory neurons distinguishes PING-like population activity from excitation alone, and whether increasing external drive reveals the transition. Removing E-to-I coupling abolished inhibitory spikes while excitation persisted, and the rising-drive condition produced repeated inhibitory volleys. The short, single-seed run supports the visual mechanism but cannot locate a robust onset threshold.

  It asks what changes when excitation can recruit inhibition.

  == 2. Design and scope

  One fixed simulated network contains 100 input channels, 100 excitatory neurons, and 25 inhibitory neurons. The scout compares an intact baseline, an E-to-I ablation, and a 20--160 Hz input ramp. All conditions use one network seed and 100 ms of simulated time. The evidence is exploratory and cannot establish robustness, biological validity, or a precise PING onset threshold.

  #figure(
    grid(
      columns: (1fr, 1fr), gutter: 12pt,
      image("/artifacts/data/exp098/network-baseline.svg", width: 100%),
      image("/artifacts/data/exp098/network-w-ei-zero.svg", width: 100%),
    ),
    caption: [Design schematic. Compiled SNNLANG circuits for the intact network (left) and the E-to-I ablation (right). The disabled excitatory-to-inhibitory projection remains structurally explicit in the ablated graph.],
  )

  == 3. Results

  === 3.1 Recurrent inhibition control

  If excitatory recruitment of inhibition is necessary for the observed alternating activity, the intact network should show inhibitory volleys while the E-to-I ablation should retain excitatory spikes but eliminate inhibitory spikes. Persistence of inhibitory spikes after ablation would reject the implementation logic; disappearance of all activity would make the control uninformative.

  #figure(
    loop-video("baseline.mp4"),
    caption: [Simulation result. Intact network at 60 Hz input. The complete raster remains visible while the red cursor marks the current 0.25 ms frame.],
  )

  #figure(
    loop-video("w-ei-zero.mp4"),
    caption: [Simulation result. Matched E-to-I ablation. The network retains excitatory activity but inhibitory neurons do not spike.],
  )

  The intact condition produced #result.baseline.e_spikes excitatory and #result.baseline.i_spikes inhibitory spikes. With E-to-I weights set to zero, excitation increased to #result.at("w-ei-zero").e_spikes spikes while inhibition fell to #result.at("w-ei-zero").i_spikes. This supports the predicted recruitment mechanism within this fixed simulation.

  === 3.2 Recruitment under increasing drive

  If drive recruits PING-like dynamics, the rising-input condition should move from sparse excitation toward repeated excitatory and inhibitory volleys. A smooth rate increase without organized inhibitory volleys would favour nonspecific activation over PING-like recruitment.

  #figure(
    loop-video("input-ramp.mp4"),
    caption: [Simulation result. Input rises linearly from 20 to 160 Hz. The rate program, network state, conductance cycle, and complete spike raster share one moving time cursor.],
  )

  The ramp produced #result.at("input-ramp").e_spikes excitatory and #result.at("input-ramp").i_spikes inhibitory spikes. The raster shows repeated inhibitory volleys as drive rises, but this visual scout does not estimate a transition point or distinguish stable PING from other synchronized E-I activity.

  == 4. Methods

  + Compile the network through SNNLANG and execute it with the graph-native `tools/snnsim` simulator from one experiment implementation. Simulate 100 excitatory and 25 inhibitory neurons at a 0.25 ms timestep for 100 ms. Project 100 independent Poisson channels densely onto the excitatory population. Fix the network seed and input seed at 7 and 8.
  + Set recurrent projection delays to one simulation step, 0.25 ms, so the compiled graph has integer-step delays.
  + Run the intact network at 60 Hz per input channel; repeat with all E-to-I weights set to zero; then restore recurrent connectivity and increase input linearly from 20 to 160 Hz.
  + Record every input, excitatory, and inhibitory spike; membrane voltage; mean excitatory and inhibitory conductance; recurrent weight; and frame-level transmission event.
  + Render one matched video per condition. Keep the complete spike raster visible throughout and move a vertical cursor across its 100 ms time axis.
  + Treat line opacity and width as synaptic-conductance encodings, neuron radius as a membrane-voltage encoding, black as excitation, and red as inhibition.
  + Restrict interpretation to this fixed network, one seed, one short run, and simulated evidence.

  == 5. Conclusion

  *Revise.* Recurrent E-to-I coupling is necessary for inhibitory recruitment in this implementation, and increasing drive produces repeated E-I volleys that the animation makes visible. The scout does not yet establish a PING onset threshold, robustness across seeds, oscillation frequency, or biological relevance. A follow-up scout should measure volley timing and frequency across input rates before escalation.
]
