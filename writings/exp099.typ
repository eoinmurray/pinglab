#import "templates/article-layout.typ": journal-article
#import "templates/result-card.typ": journal-result-card, result-figure-ref, with-result-sections
#import "/.demolab/lib.typ": data-image
#import "templates/dataset.typ": video, data-file, inputs-ready, pending-report, run-view, input-assets
#import "templates/abstract.typ": journal-abstract
#import "templates/methods.typ": journal-methods, method-card
#import "templates/parameters-table.typ": parameters-table
#let data-file = data-file.with(article: "exp099")

#let meta = (
  tags: ("data", "v36.0.0"),
  // Author-locked title: do not change.
  title: "Video AI-PING transition",
  created_at: "2026-08-26T00:00:00Z",
  updated_at: "2026-09-08",
  description: "Two single-seed conductance-based network simulations show intermittent activity under richer input and sustained alternating volleys during strong shared afferent drive.",
  collection: "demo",
  order: 13,
)

#let inputs = ("exp099",)

#let render-report(data-file) = [
  #journal-abstract(
    question: [We tested how shared and heterogeneous afferent drive organize a sparse conductance-based excitatory–inhibitory network.],
    approach: [We visualized two single-seed simulations: a richer-input condition and a condition with a sustained shared-input ramp.],
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

    #let shared-clip = data-file("exp099/shared-drive-ai-to-ping.mp4")
    #if shared-clip != none { let _ = read(shared-clip, encoding: none) }
    #journal-result-card(
      title: "Shared drive recruits rhythmic volleys",
      visual: [
        #figure(
          video(shared-clip),
          caption: [Single-seed shared-drive simulation over 50–1,250 ms,
            omitting the initialization burst, with
            the same panel and colour mappings as
            #result-figure-ref(<fig:exp099-richer>). The shared afferent
            multiplier rose smoothly from 1 to 25 between 500 and 700 ms and
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

    #let richer-clip = data-file("exp099/richer-input-ai-to-intermittent-ping.mp4")
    #if richer-clip != none { let _ = read(richer-clip, encoding: none) }
    #journal-result-card(
      title: "Richer input remains intermittent",
      visual: [
        #figure(
          video(richer-clip),
          caption: [Single-seed richer-input simulation over 0–1,800 ms.
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
        7. The richer-input video began at initialization. The shared-drive video
        omitted the first 50 ms, with neuronal and conductance states carried
        continuously into the displayed interval.
        In the 2,000 ms richer-input condition, the shared multiplier rose
        from 1 to 6.5 and the private multipliers from 1 to 1.2 between 600 and
        850 ms, returning to 1 by 1,100 ms; a stationary lognormal rate process
        with a 250 ms timescale modulated all external events. In the 1,250 ms
        shared-drive condition, the shared multiplier rose from 1 to 25 between
        500 and 700 ms and remained there, while private multipliers and all
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
      method-card([Map recorded activity], [We displayed source-to-target
        transmission from recorded spikes and fixed realized weights, sampling
        paths only to avoid overplotting. Per-neuron E and I firing rates were
        calculated in a centred 20 ms display window. The richer-input view
        covered 0–1,800 ms; the shared-drive view covered 50–1,250 ms. The
        videos used the same panel grammar, with 600 nonuniformly paced frames
        for richer input and 712 for shared drive, and a representative still
        selected near maximal measured temporal organization.]),
    ),
  )

  == Parameter Table

  #parameters-table(
    (
      [Parameter],
      [Richer input],
      [Shared drive],
      [Cortical reference],
    ),
    (
      ([Simulation / view], [2,000 / 0–1,800 ms], [1,250 / 50–1,250 ms], [No canonical duration.]),
      ([Timestep; seed], [0.25 ms; 7], [0.25 ms; 7], [Numerical only.]),
      ([Population], [$N_E=400$; $N_I=100$], [$N_E=400$; $N_I=100$], [Realistic 4:1 ratio; strongly reduced circuit (#link("https://pmc.ncbi.nlm.nih.gov/articles/PMC3839692/")[Meyer et al., 2013]).]),
      ([Shared baseline], [10 Hz], [10 Hz], [Plausible rate; shared/private separation is abstract.]),
      ([Private E/I baseline], [15 / 15 Hz], [14.25 / 14.25 Hz], [Plausible active-input rate; cortical firing is often sparser (#link("https://pmc.ncbi.nlm.nih.gov/articles/PMC4108079/")[Zhou et al., 2014]).]),
      ([Shared multiplier], [$1 arrow 6.5 arrow 1$], [$1 arrow 25$, then held], [No standard; 25-fold represents strong synchrony.]),
      ([Private afferent multiplier], [$1 arrow 1.2 arrow 1$], [Constant at 1], [No direct biological standard.]),
      ([Input timing], [600–850–1,100 ms], [500–700 ms; then held], [Plausible timescale; imposed waveform.]),
      ([Global variation], [Lognormal; $tau=250$ ms; SD 12%], [None], [Qualitative cortical-state model.]),
      ([$w_("in" arrow E)$], [$0.080 plus.minus 0.008$ µS], [$0.160 plus.minus 0.016$ µS], [Low end of unitary excitation (#link("https://pmc.ncbi.nlm.nih.gov/articles/PMC10016070/")[Hunt et al., 2023]).]),
      ([$w_("in" arrow I)$], [$0.020 plus.minus 0.002$ µS], [$0.0040 plus.minus 0.0004$ µS], [Very weak.]),
      ([$w_(E arrow E)$], [$0.85 plus.minus 0.255$ µS], [$1.02 plus.minus 0.306$ µS], [Plausible unitary scale (#link("https://pmc.ncbi.nlm.nih.gov/articles/PMC10016070/")[Hunt et al., 2023]).]),
      ([$w_(E arrow I)$], [$0.60 plus.minus 0.18$ µS], [Same], [Plausible; PV input is often stronger than E→E (#link("https://pubmed.ncbi.nlm.nih.gov/22402650/")[Avermann et al., 2012]).]),
      ([$w_(I arrow E)$], [$3.00 plus.minus 0.90$ µS], [Same], [Plausible strong inhibition (#link("https://pmc.ncbi.nlm.nih.gov/articles/PMC4816789/")[conductance estimates]).]),
      ([$w_(I arrow I)$], [$0.40 plus.minus 0.12$ µS], [Same], [Broadly plausible; subtype dependent.]),
      ([AMPA background], [500 / 80 Hz], [450 / 72 Hz], [Aggregate event stream, not neuron rate.]),
      ([GABA background], [500 / 80 Hz], [1,000 / 160 Hz], [Aggregate stream; shared-drive rate doubled.]),
      ([Connectivity], [2.5% nonzero], [Same], [Low and uniform; nearby L2/3 pathways span roughly 17–60% (#link("https://pmc.ncbi.nlm.nih.gov/articles/PMC4305188/")[Pala and Petersen, 2015]).]),
      ([$tau_("AMPA")$ / $tau_("GABA")$], [2 / 9 ms], [Same], [Plausible: AMPA a few ms; GABA_A about 4–20 ms (#link("https://pubmed.ncbi.nlm.nih.gov/1384578/")[Hestrin, 1992]; #link("https://pmc.ncbi.nlm.nih.gov/articles/PMC2230760/")[Xiang et al., 1998]).]),
      ([$tau_(m,E)$ / $tau_(m,I)$], [20 / 5 ms], [Same], [Plausible; fast-spiking interneurons about 4–9 ms (#link("https://pmc.ncbi.nlm.nih.gov/articles/PMC2730466/")[Goldberg et al., 2008]).]),
    ),
    columns: (1.15fr, 1.25fr, 1.25fr, 2.25fr),
  )

  Timescales are broadly cortical. The main limitation is structural: the
  shared-drive condition changes several inputs and weights simultaneously.

  #run-view("exp099", inputs)
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(data-file, inputs, [], ())
}

#let meta = meta + (assets: input-assets("exp099", inputs))
#let body = journal-article("exp099", inputs, report-body, dataset-placed: inputs-ready(data-file, inputs))
