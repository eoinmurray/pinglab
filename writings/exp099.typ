#import "templates/article-layout.typ": journal-article
#import "templates/result-card.typ": journal-result-card, result-figure-ref, with-result-sections
#import "/.demolab/lib.typ": data-image
#import "templates/dataset.typ": video, data-file, inputs-ready, pending-report, run-view, input-assets
#import "templates/abstract.typ": journal-abstract
#import "templates/methods.typ": journal-methods, method-card
#let data-file = data-file.with(article: "exp099")

#let meta = (
  tags: ("data", "v35.4.0"),
  // Author-locked title: do not change.
  title: "Video AI-PING transition",
  created_at: "2026-08-26T00:00:00Z",
  updated_at: "2026-09-03",
  description: "Two single-seed conductance-based network simulations show intermittent activity under richer input and sustained alternating volleys during strong shared afferent drive.",
  collection: "demo",
  order: 13,
)

#let inputs = ("exp099",)

#let render-report(data-file) = [
  #journal-abstract(
    question: [We tested how shared and heterogeneous afferent drive organize a sparse conductance-based excitatory–inhibitory network.],
    approach: [We visualized two retained single-seed simulations: a richer-input condition and a condition with a sustained shared-input ramp.],
    finding: [Strong shared drive recruited regular alternating excitatory and inhibitory volleys, whereas the richer-input condition remained irregular with intermittent structure.],
    scope: [These examples establish input-dependent behaviour in the configured simulations, not robustness across seeds or a one-factor comparison between conditions.],
  )

  == Results

  #with-result-sections[

    #journal-result-card(
      title: "Implemented circuit architecture",
      visual: [
        #figure(
          data-image(
            data-file("exp099/network.svg"),
            width: 100%,
            alt: "Structural diagram of the excitatory and inhibitory populations, their recurrent projections, afferent inputs and readout.",
          ),
          caption: [Structural schematic of the implemented excitatory and
            inhibitory populations, recurrent AMPA and GABA projections,
            destination-specific afferent projections and downstream readout.
            The diagram specifies the model and is not experimental evidence.],
          kind: image,
          supplement: [Figure],
        ) <fig:exp099-network>
      ],
      expectation: [The candidate PING mechanism is recurrent excitation
        followed by inhibitory feedback; the schematic identifies the pathways
        capable of producing that sequence
        (#result-figure-ref(<fig:exp099-network>)).],
    )

    #journal-result-card(
      title: "External input architecture",
      visual: [
        #figure(
          data-image(
            data-file("exp099/input-map-option-3.svg"),
            width: 100%,
            alt: "Shared and destination-specific afferents feed the E and I populations, while AMPA and GABA backgrounds act on both populations.",
          ),
          caption: [Shared afferent events enter both destination streams;
            E-private and I-private events remain destination-specific. Separate
            private and grouped AMPA and GABA backgrounds act on both
            populations. This is an explanatory schematic rather than measured
            evidence.],
          kind: image,
          supplement: [Figure],
        ) <fig:exp099-inputs>
      ],
      orientation: [The afferent streams and conductance backgrounds provide
        distinct routes by which input rate, correlation and polarity can alter
        excitatory–inhibitory timing
        (#result-figure-ref(<fig:exp099-inputs>)).],
    )

    #let richer-clip = data-file("exp099/richer-input-ai-to-intermittent-ping.mp4")
    #if richer-clip != none { let _ = read(richer-clip, encoding: none) }
    #journal-result-card(
      title: "Richer input remains intermittent",
      visual: [
        #figure(
          video(richer-clip),
          caption: [Single-seed richer-input simulation over 300–1,800 ms.
            Panel A maps recorded spike and conductance inputs to the E and I
            populations; B shows mean conductances and voltages; C traces the
            excitatory–inhibitory conductance plane; D shows per-neuron E and I
            firing rates in a 20 ms window; E shows shared and private afferent
            multipliers against time; and F shows recurrent-weight
            distributions. Black denotes excitatory or E-targeted quantities,
            red inhibitory or I-targeted quantities, and grey the shared input.
            Transmission paths are sampled for legibility.],
          kind: image,
          supplement: [Figure],
        ) <fig:exp099-richer>
      ],
      observation: [Population rates fluctuated irregularly through the
        afferent transient, with only short structured episodes
        (#result-figure-ref(<fig:exp099-richer>, panel: "D")). This single
        realization does not establish whether richer input generally
        suppresses or preserves PING.],
    )

    #let shared-clip = data-file("exp099/shared-drive-ai-to-ping.mp4")
    #if shared-clip != none { let _ = read(shared-clip, encoding: none) }
    #journal-result-card(
      title: "Shared drive recruits rhythmic volleys",
      visual: [
        #figure(
          video(shared-clip),
          caption: [Single-seed shared-drive simulation over 0–1,000 ms, with
            the same panel and colour mappings as
            #result-figure-ref(<fig:exp099-richer>). The shared afferent
            multiplier rose smoothly from 1 to 25 between 250 and 450 ms and
            remained at 25 thereafter; private afferent multipliers remained at
            1. Recurrent weights and background-input settings were fixed
            throughout this simulation.],
          kind: image,
          supplement: [Figure],
        ) <fig:exp099-shared>
      ],
      observation: [As shared drive increased, low irregular firing gave way to
        sustained alternating E and I volleys and a repeated conductance cycle
        (#result-figure-ref(<fig:exp099-shared>, panel: "C–E")). Because the two
        simulations also differ in fixed weights and background settings, their
        contrast is illustrative rather than a one-factor between-condition
        test.],
    )

  ]

  #journal-methods(
    compute: (
      method-card([Construct the recurrent circuit], [We simulated 400
        excitatory and 100 inhibitory conductance-based leaky integrate-and-fire
        neurons with a 0.25 ms timestep. Excitatory and inhibitory membrane time
        constants were 20 and 5 ms; both populations used a −65 mV reset and
        −50 mV threshold. Each recurrent projection contained an exact 2.5%
        nonzero connection fraction. AMPA and GABA conductances decayed with 2
        and 9 ms time constants.]),
      method-card([Generate external events], [We formed each
        destination-specific afferent stream by combining a shared Bernoulli
        spike component with an independent private component:
        #math.equation(block: true,
          $s_E[k] = s_"shared"[k] or s_"E-private"[k], quad
          s_I[k] = s_"shared"[k] or s_"I-private"[k].$
        )
        Here $s_X[k] in {0, 1}$ is the event indicator for source $X$ at
        timestep $k$. Its probability was
        #math.equation(block: true,
          $p_X[k] = min(1, (r_X Delta t_"sim") / 1000
          m_"weather"[k] m_X[k]),$
        )
        where $r_X$ is the baseline rate in hertz, $Delta t_"sim"$ is the
        integration timestep in milliseconds, $m_X[k]$ is the afferent
        multiplier, and $m_"weather"[k]$ is a slow global rate multiplier when
        present. Independent and locally grouped AMPA and GABA shot noise also
        drove both populations.]),
      method-card([Configure the two simulations], [Both simulations used seed
        7. In the 2,000 ms richer-input condition, the shared multiplier rose
        from 1 to 6.5 and the private multipliers from 1 to 1.2 between 600 and
        850 ms, returning to 1 by 1,100 ms; a stationary lognormal rate process
        with a 250 ms timescale modulated all external events. In the 1,000 ms
        shared-drive condition, the shared multiplier rose from 1 to 25 between
        250 and 450 ms and remained there, while private multipliers and all
        other settings remained fixed through time. The two conditions used
        different fixed feedforward, recurrent-excitatory and background-drive
        scales and therefore were not a one-factor comparison.]),
    ),
    analyse: (
      method-card([Measure population state], [We averaged excitatory and
        inhibitory membrane voltage over neurons and averaged excitatory and
        inhibitory conductance over excitatory neurons at every timestep.
        External AMPA and GABA event trains were transformed with their 2 and 9
        ms exponential kernels before averaging.]),
      method-card([Measure temporal organization], [We computed
        $R_"contrast"$, the autocorrelation lobe–trough contrast of excitatory
        spikes. The richer-input condition used 400 ms windows at 10 ms strides
        with a 100 ms maximum lag; the shorter shared-drive condition used 160
        ms windows at 5 ms strides with a 60 ms maximum lag. Spikes were binned
        at 1 ms, and undefined contrasts were recorded as zero.]),
    ),
    present: (
      method-card([Map retained evidence], [We displayed source-to-target
        transmission from recorded spikes and fixed realized weights, sampling
        paths only to avoid overplotting. Per-neuron E and I firing rates were
        calculated in a centred 20 ms display window. The richer-input view
        covered 300–1,800 ms; the shared-drive view covered 0–1,000 ms. Both
        videos used the same panel grammar, 600 nonuniformly paced frames and a
        representative still selected near maximal measured temporal
        organization.]),
    ),
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
