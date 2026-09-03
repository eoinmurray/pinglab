#import "contents.typ": contents-here, with-contents, result-card, with-numbered-equations, with-result-sections
#import "/.demolab/lib.typ": data-image
#import "run-inputs.typ": video
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp099")

#let meta = (
  status: "[▦ DATA | v34.0.1]",
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
      data-file("exp099/input-map-option-3.svg"),
      width: 100%,
      alt: "E-private and shared afferents combine to drive E neurons through AMPA; shared and I-private afferents combine to drive I neurons. Separate AMPA and GABA backgrounds drive both populations. Weather modulates all input rates, while the transient modulates afferents only.",
    ),
    caption: [Explanatory map of the implemented external-input architecture.
      Shared afferent events enter both destination streams; E-private and
      I-private events remain destination-specific. Both background families
      feed both populations. This is a schematic, not experimental evidence.],
    kind: image, supplement: [Figure],
  )

  ]

  #result-card[
  === Single-seed richer-input activity probe <result-3-richer-input-probe>

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

  #result-card[
  === Shared afferent drive produces an AI-to-PING transition <result-4-shared-drive-transition>

  #let clip = data-file("exp099/shared-drive-ai-to-ping.mp4")
  // Missing files in a selected run remain errors, not empty-run placeholders.
  #if clip != none { let _ = read(clip, encoding: none) }
  #figure(
    video(clip),
    caption: [Single-seed simulation with fixed recurrent weights and fixed
      private and background inputs. The video displays 500--1,500 ms; the
      preceding 500 ms burn-in is omitted. Shared afferent drive begins at the
      displayed midpoint, rises from its baseline to 40 times baseline over
      50 ms, and then remains at that plateau. Transmission paths are sampled
      for readability, whereas the raster retains all recorded spikes.
      $R_"contrast"$ is autocorrelation lobe--trough contrast measured in a
      160 ms window; $L$ is the conductance-loop score smoothed over 75 ms and
      normalized within this simulation.],
    kind: image, supplement: [Figure],
  )

  ]
  ]

  == Methods

  === Compute

  + *Specify stochastic external drive.* We formed the two destination-specific
    afferent streams by OR-combining one shared component with one private
    component:
    #math.equation(block: true,
      $S_E[k] = S_"shared"[k] or S_"E-private"[k], quad
      S_I[k] = S_"shared"[k] or S_"I-private"[k].$
    )
    Each afferent component $X$ was sampled as
    #math.equation(block: true,
      $S_X[k] tilde "Bernoulli"(p_X[k]), quad
      p_X[k] = min(1, (r_X Delta t_"sim") / 1000 M[k] A_X[k]).$
    )
    The resulting external conductances followed
    #math.equation(block: true,
      $g_c^Y[k+1] = exp(-Delta t_"sim" / tau_c) g_c^Y[k]
      + bb(1)_[c = "AMPA"] S_Y[k] W_("in",Y) + b_c^Y[k].$
    )
    Here $S_X[k] in {0, 1}$ is a sampled event, not its probability $p_X[k]$;
    $r_X$ is its baseline rate in hertz; $Delta t_"sim"$ is the timestep in
    milliseconds; $M[k]$ is the weather multiplier applied to every external
    event rate; and $A_X[k]$ is the transient applied only to afferents.
    $Y in {E, I}$ is the target population, $c in {"AMPA", "GABA"}$ is the
    conductance family, $tau_c$ is its decay time, $W_("in",Y)$ is the afferent
    AMPA-weight matrix and $b_c^Y[k]$ is AMPA or GABA background shot noise
    delivered to population $Y$. The afferent rates were 10 Hz shared and 15 Hz
    private; background events occurred at 500 Hz privately and 80 Hz within
    local neuron groups.

  + *Set the Figure 3 inputs.* We used the following executed input settings for
    #link(<result-3-richer-input-probe>)[Figure 3]:
    #table(
      columns: (1.55fr, 0.7fr, 0.65fr, 1.15fr, 0.55fr),
      align: (left, left, right, left, right),
      table.header([*Input*], [*Target*], [*Rate*], [*Event size*], [*$tau$*]),
      [Shared afferent], [E and I], [10 Hz], [target weight], [2 ms],
      [E-private afferent], [E], [15 Hz], [$cal(N)_+(0.08, 0.008^2)$ nS], [2 ms],
      [I-private afferent], [I], [15 Hz], [$cal(N)_+(0.02, 0.002^2)$ nS], [2 ms],
      [AMPA private background], [E / I], [500 Hz], [0.06 / 0.03 nS], [2 ms],
      [AMPA grouped background], [E / I], [80 Hz], [0.02 / 0.01 nS], [2 ms],
      [GABA private background], [E / I], [500 Hz], [0.03 / 0.03 nS], [9 ms],
      [GABA grouped background], [E / I], [80 Hz], [0.01 / 0.01 nS], [9 ms],
    )
    Here $cal(N)_+$ denotes a normal draw lower-clamped at zero; shared and
    private afferent events use the same destination-specific weight matrix.
    The 2,000 ms simulation used $Delta t_"sim" = 0.25$ ms and seed 7; Figure 3
    displays 300--1,800 ms. Grouped background events were shared within groups
    of 25 E or 10 I neurons. Private background rate and amplitude multipliers
    were lower-clamped normal draws with mean 1 and standard deviation 0.1.
    Weather had a 250 ms timescale and fractional standard deviation 0.12 and
    modulated all input rates. The 600--1,100 ms afferent transient peaked at
    850 ms, scaling private afferents by 1.2 and shared afferents by 6.5.

  #run-view("exp099", inputs)
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
