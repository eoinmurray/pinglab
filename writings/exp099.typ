#import "/.demolab/lib.typ": data-image, video
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#let data-file = data-file.with(article: "exp099")

#let meta = (
  status: "Planned",
  title: "From simplified to brainlike input in a PING network",
  created_at: "2026-08-26",
  updated_at: "2026-08-26",
  description: "A planned simulation scout comparing a Börgers–Kopell-like input regime with a richer conductance-based background.",
  collection: "demo",
  order: 13,
)

#let inputs = ("exp099",)

#let render-report(data-file) = [
  == 1. Abstract

  We will compare two input regimes in the same sparse excitatory–inhibitory spiking network.

  First, we will construct a simplified Börgers–Kopell-like regime using stationary excitatory and inhibitory drive with independent noise. We will identify a condition that produces stable pyramidal–interneuron network gamma (PING).

  We will then replace this input with a richer background containing private and shared conductance fluctuations, cellular heterogeneity, correlated afferent spikes, and slow stationary modulation. We will test whether the richer input preserves the reference PING state or changes its stability, including whether the rhythm becomes intermittent.

  This controlled comparison will show which features of brainlike background activity can be added without losing the established PING mechanism.

  In plain terms, we will build a simple known rhythm and then see what happens when its artificial input is replaced with a more realistic background.

  == 2. Working media

  #figure(
    data-image(data-file("exp099/network.svg"), width: 100%),
    caption: [Structural schematic of the excitatory and inhibitory populations,
      recurrent projections, and afferent inputs. This is a model diagram, not evidence.],
    kind: image, supplement: [Figure],
  )

  #figure(
    data-image(data-file("exp099/richer-input-ai-to-intermittent-ping.png"), width: 100%),
    caption: [Working animation poster; an illustration of the developing experiment.],
    kind: image, supplement: [Figure],
  )

  #let clip = data-file("exp099/richer-input-ai-to-intermittent-ping.mp4")
  // Missing files in a selected run remain errors, not empty-run placeholders.
  #if clip != none { let _ = read(clip, encoding: none) }
  #video(
    clip,
    caption: [Simulated working media, not an established result.],
  )
]

#let body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(data-file, inputs, [], ())
}
