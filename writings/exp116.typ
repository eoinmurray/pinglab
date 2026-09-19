#import "templates/article-layout.typ": journal-article
#import "templates/abstract.typ": journal-abstract
#import "templates/dataset.typ": data-file, input-assets, inputs-ready, pending-report
#import "templates/methods.typ": journal-methods, method-card
#import "templates/result-card.typ": journal-result-card, result-figure-ref, with-result-sections
#import "/.demolab/lib.typ": data-image, data-json

#let data-file = data-file.with(article: "exp116")

#let meta = (
  tags: ("txt", "v36.0.0"),
  title: "Minimal PING Onset Test",
  created_at: "2026-09-19T00:00:00Z",
  updated_at: "2026-09-19",
  description: "A minimal test of oscillatory onset, sampled criticality and inhibitory-timescale dependence in a four-variable PING closure.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp116",)
#let preview-figures = (
  (path: "exp116/minimal-hopf-evidence.svg", label: "minimal mean-field onset evidence"),
)

#let render-report(data-file) = [
  #let numbers = data-json(data-file("exp116/numbers.json"))
  #let reference = numbers.reference
  #let primary = numbers.conditions.filter(row => row.purpose == "gaba_sweep")

  #journal-abstract(
    question: [We asked whether a four-variable PING closure develops a continuous oscillatory onset and whether slower inhibition lowers its onset frequency.],
    approach: [We combined equilibrium continuation, one matched amplitude-ramp test, a six-value inhibitory-decay sweep and four endpoint robustness checks.],
    finding: [The closure passed the prespecified onset, sampled-criticality, timescale and robustness criteria.],
    scope: [The calculation supports a qualitative comparison with a separate spiking network, not a shared bifurcation or quantitative frequency match.],
  )

  == Results

  #with-result-sections[
    #journal-result-card(
      title: "Minimal onset evidence",
      observation: [The reference equilibrium lost stability at #calc.round(reference.onset.drive_nA, digits: 6) nA with onset frequency #calc.round(reference.onset.frequency_Hz, digits: 2) Hz (#result-figure-ref(<fig:exp116-evidence>, panel: "A")). The sampled amplitude branches met the predefined criteria for consistency with a supercritical transition (#result-figure-ref(<fig:exp116-evidence>, panel: "B")). Across the six inhibitory decay times, onset frequency fell from #calc.round(primary.first().onset.frequency_Hz, digits: 2) to #calc.round(primary.last().onset.frequency_Hz, digits: 2) Hz, and all four endpoint closure checks retained that direction (#result-figure-ref(<fig:exp116-evidence>, panel: "C")). Finite ramps cannot exclude a narrower bistable interval or unstable periodic orbit.],
      visual: [
        #figure(
          data-image(
            data-file("exp116/minimal-hopf-evidence.svg"),
            width: 100%,
            alt: "Three panels showing a mean-field eigenvalue crossing, upward and downward amplitude ramps, and onset frequency across inhibitory decay with endpoint robustness checks.",
          ),
          caption: [*A:* Largest real part of the equilibrium Jacobian eigenvalues versus tonic drive at the reference closure; the marker identifies the accepted crossing. *B:* Upward and downward peak-to-peak excitatory-rate amplitudes over the final 500 ms of each 2-s drive step. *C:* Onset frequency at the six reference-closure inhibitory decay times; light endpoint segments show the four low/high effective-noise and rate-relaxation corner checks. All curves are deterministic.],
          kind: image,
          supplement: [Figure],
        ) <fig:exp116-evidence>
      ],
    )
  ]

  #journal-methods(body: (
    method-card([Define the closure], [We used excitatory and inhibitory population rates plus the recurrent excitatory and inhibitory conductances. Each rate relaxed toward a stationary noisy-LIF gain; conductances followed exponential AMPA and GABA filters. Fixed driving forces and prescribed effective-noise and rate-relaxation scales make this a deterministic closure rather than an exact reduction of the spiking network.]),
    method-card([Locate oscillatory onset], [We continued equilibria across 401 tonic-drive values from 0 to 4 nA and refined the first stable-to-unstable complex-eigenvalue crossing. We repeated every continuation on an 801-point grid and required onset drive and frequency to agree within $10^(-7)$ nA and $10^(-4)$ Hz.]),
    method-card([Test sampled criticality], [At the reference closure, we integrated 25 ascending and descending drive steps from 0.10 nA below to 0.55 nA above onset. Each step lasted 2 s and carried its endpoint forward. We measured excitatory peak-to-peak amplitude over the final 500 ms. Consistency with supercritical onset required a branch gap below $10^(-4)$ $"ms"^(-1)$, positive amplitude-squared slope and $R^2_"fit">0.9$.]),
    method-card([Test timescale and robustness], [We repeated onset detection at inhibitory decay times 4.5, 6, 9, 12, 18 and 27 ms with the reference effective-noise scale and rate relaxation. We then tested only the two decay endpoints at the four low/high corners of those closure choices. The robustness criterion required resolved onset at both endpoints and lower frequency under slower inhibition in every corner.]),
  ))
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(data-file, inputs, [], preview-figures)
}

#let meta = meta + (assets: input-assets("exp116", inputs))
#let body = journal-article("exp116", inputs, report-body)
