#import "contents.typ": contents-here, with-contents, result-card, with-numbered-equations, with-result-sections
#import "/.demolab/lib.typ": data-image, cite, reference-list
#import "run-inputs.typ": video
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp099")

#let meta = (
  status: "[▦ DATA | v33.0.0]",
  title: "Video of the AI-to-PING transition",
  created_at: "2026-08-26T00:00:00Z",
  updated_at: "2026-09-02",
  description: "A planned simulation scout comparing a Börgers–Kopell-like input regime with a richer conductance-based background.",
  collection: "demo",
  order: 13,
)

#let inputs = ("exp099",)

#let render-report(data-file) = [
  == Abstract

  Will test whether PING survives when stationary drive is replaced by more
  brainlike, heterogeneous and correlated input. The comparison will add input
  complexity in stages while keeping the same sparse excitatory–inhibitory
  network.

  Current implementation has only a richer-input activity probe and cannot show
  how the rhythm changes relative to a simple baseline. Its media demonstrate
  the probe, not the planned controlled comparison or the input features
  responsible for preserving PING.

  #contents-here()

  == Results

  #with-result-sections[

  #result-card[
  === Excitatory–inhibitory network and afferent-input schematic

  #figure(
    data-image(data-file("exp099/network.svg"), width: 100%),
    caption: [Structural schematic of the excitatory and inhibitory populations,
      recurrent projections, and afferent inputs. This is a model diagram, not evidence.],
    kind: image, supplement: [Figure],
  )

  ]

  #result-card[
  === External inputs and their target populations

  #figure(
    data-image(
      data-file("exp099/input-map.svg"),
      width: 100%,
      alt: "Three external input families—excitatory afferent spikes, fast AMPA background and slow GABA background—branch to both excitatory and inhibitory populations. Slow weather modulates every input rate, while a transient wave modulates afferents only.",
    ),
    caption: [Explanatory map of the implemented external-input architecture.
      Near-black arrows show excitatory afferent and AMPA-background
      conductances; red terminal bars show GABA-background inhibition; dashed
      amber arrows show rate modulation. This is a schematic, not experimental
      evidence.],
    kind: image, supplement: [Figure],
  )

  ]

  #result-card[
  === Single-seed richer-input activity probe

  #let clip = data-file("exp099/richer-input-ai-to-intermittent-ping.mp4")
  // Missing files in a selected run remain errors, not empty-run placeholders.
  #if clip != none { let _ = read(clip, encoding: none) }
  #figure(
    video(clip),
    caption: [Simulated single-seed working media, not an established comparison.
      Transmission paths are sampled for readability; the raster retains all
      recorded spikes. The view spans
      300–1,800 ms with nonuniform playback pacing. The figure label R denotes autocorrelation contrast $R_"contrast"$;
      L is the conductance-loop score smoothed over 75 ms and normalized within
      this probe, not a probability of PING.],
    kind: image, supplement: [Figure],
  )

  ]
  ]

  == Methods

  The intended study compares a simplified Börgers–Kopell-like input regime
  with richer background activity, motivated by reciprocal excitatory–inhibitory
  synchronization.#cite(1) The procedure below describes the implemented
  richer-input probe; selecting a stable simplified reference and conducting
  the controlled comparison remain planned.

  === Compute

  + *Construct the recurrent circuit.* We used 400 excitatory and 100 inhibitory
    conductance-based leaky integrate-and-fire neurons with fixed excitatory–excitatory, excitatory–inhibitory,
    inhibitory–excitatory and inhibitory–inhibitory projections. We set 97.5% of
    each recurrent matrix to zero using an exact-count mask; we sampled the
    remaining weights from lower-clamped normal distributions with respective
    means 0.85, 0.6, 3.0 and 0.4 μS and standard deviations 0.255, 0.18, 0.9 and
    0.12 μS. Excitatory and inhibitory conductance decay times are 2 and 9 ms;
    no training or parameter selection was performed in this probe.

  + *Specify stochastic external drive.* For each afferent stream $X$, we
    sampled binary events as
    #math.equation(block: true,
      $S_X(t) tilde "Bernoulli"(P_X(t)), quad
      P_X(t) = min(1, (r_X Delta t) / 1000 M(t) A_X(t)).$
    )
    Here $P_X(t)$ is the event probability and $S_X(t) in {0, 1}$ is the sampled
    event; $r_X$ is the baseline rate in hertz, $Delta t$ is the timestep in
    milliseconds, $M(t)$ is the shared slow weather multiplier and $A_X(t)$ is
    the afferent-only transient multiplier. We OR-combined a common 10 Hz shared
    afferent sample with independent 15 Hz E-private and I-private samples, then
    delivered both resulting streams through AMPA projections with non-negative
    weights drawn from
    $cal(N)(0.08, 0.008^2)$ μS onto E and $cal(N)(0.02, 0.002^2)$ μS onto I.
    External conductances followed
    #math.equation(block: true,
      $g_c^Y[t+1] = exp(-Delta t / tau_c) g_c^Y[t]
      + bb(1)_[c = "AMPA"] S_Y[t] W_("in",Y) + b_c^Y[t].$
    )
    Here $Y in {E, I}$ is the target population, $c$ is AMPA or GABA,
    $tau_c$ is 2 or 9 ms, $W_("in",Y)$ is the afferent-weight matrix and
    $b_c^Y$ is background shot noise. Background events occurred at 500 Hz
    privately and 80 Hz within groups of 25 E or 10 I neurons. Their
    private/shared amplitudes were 0.06/0.02 μS for AMPA onto E and 0.03/0.01 μS
    for the other three channels. Private rates and amplitudes had fixed
    mean-one, lower-clamped normal cell multipliers with standard deviation 0.1.
    The same $M(t)$ multiplied every background rate; it had a 250 ms timescale
    and fractional spread 0.12.

  + *Apply the transient and record activity.* We simulated 2,000 ms at a 0.25 ms
    timestep with random seed 7. We raised the afferent multiplier smoothly from
    baseline at 600 ms to its peak at 850 ms and back by 1,100 ms; peak
    multipliers are 1.2 for private and 6.5 for shared afferents. We retained spikes,
    neuron voltages, conductances and executed input events, keeping recurrent
    weights fixed throughout.

  === Analyse

  #set enum(start: 4)

  + *Measure temporal organization.* We evaluated excitatory spike-autocorrelation
    lobe–trough contrast in 400 ms windows every 10 ms, using 1 ms lag bins out
    to 100 ms; undefined contrasts were recorded as zero.
    #math.equation(block: true, $R_"contrast" = (A_"lobe" - A_"trough") / (A_"lobe" + A_"trough")$)
    Here $R_"contrast"$ is dimensionless contrast, $A_"lobe"$ is the smoothed autocorrelogram's
    lobe height before its first trough, and $A_"trough"$ is that trough's height.
    We measured conductance-loop compactness and directional coherence in 40 ms
    windows every 5 ms over 300–1,800 ms, smoothed over 75 ms, and clipped the
    10th–95th-percentile normalization to zero–one. We counted excitatory and
    inhibitory spikes over the whole probe; the peak-contrast summary uses
    window centres from 200 through 1,790 ms, while the plot also includes
    1,800 ms. These single-seed descriptors do not establish a causal input
    effect or classify a rhythm as PING by themselves.

  === Present

  #set enum(start: 5)

  + *Display the implemented probe.* We mapped the implemented external-input
    architecture in an explanatory schematic and displayed the recorded spikes,
    conductances and temporal-organization summaries for the single-seed
    richer-input probe without presenting the planned control as executed.

  #run-view("exp099", inputs)

  #reference-list((
    (text: [Börgers, C. & Kopell, N. — _Synchronization in Networks of Excitatory
      and Inhibitory Neurons with Sparse, Random Connectivity_. Neural Computation
      15(3), 509–538, 2003.], doi: "10.1162/089976603321192059"),
  ))
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(data-file, inputs, [], ())
}

#let meta = meta + (assets: input-assets("exp099", inputs))
#let body = with-datasets("exp099", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-numbered-equations(body)
#let body = with-contents(body)
