#let meta = (
  title: "From simplified to brainlike input in a PING network",
  created_at: "2026-08-26",
  updated_at: "2026-08-26",
  description: "A planned simulation scout comparing a Börgers–Kopell-like input regime with a richer conductance-based background.",
  collection: "demo",
  order: 13,
)

#let experiment-video(src, poster) = context {
  if target() == "html" {
    html.elem(
      "video",
      attrs: (
        src: src,
        poster: poster,
        controls: "",
        loop: "",
        playsinline: "",
        style: "max-width:100%;width:100%",
      ),
    )[]
  } else {
    image(poster, width: 100%)
  }
}

#let body = [
  == 1. Abstract

  We will compare two input regimes in the same sparse excitatory–inhibitory spiking network.

  First, we will construct a simplified Börgers–Kopell-like regime using stationary excitatory and inhibitory drive with independent noise. We will identify a condition that produces stable pyramidal–interneuron network gamma (PING).

  We will then replace this input with a richer background containing private and shared conductance fluctuations, cellular heterogeneity, correlated afferent spikes, and slow stationary modulation. We will test whether the richer input preserves the reference PING state or changes its stability, including whether the rhythm becomes intermittent.

  This controlled comparison will show which features of brainlike background activity can be added without losing the established PING mechanism.

  In plain terms, we will build a simple known rhythm and then see what happens when its artificial input is replaced with a more realistic background.

  == 2. Working media

  #figure(
    image("/assets/exp099/network.svg", width: 100%),
    caption: [Structural schematic. The compiled SNNLANG graph exposes the excitatory and inhibitory populations, recurrent projections, and afferent inputs used by the richer-input condition. It is a model diagram, not evidence.],
  )

  #figure(
    experiment-video(
      "/exp099/richer-input-ai-to-intermittent-ping.mp4",
      "/exp099/richer-input-ai-to-intermittent-ping.png",
    ),
    caption: [Simulated working media. The richer-input condition moves from asynchronous activity into intermittent PING and then returns toward asynchronous activity while the recurrent network remains fixed. This animation currently illustrates the developing experiment; it is not yet presented as a result.],
  )
]
