#import "templates/article-layout.typ": journal-article
#import "templates/abstract.typ": journal-abstract
#import "templates/result-card.typ": journal-result-card, with-result-sections, result-figure-ref
#import "templates/methods.typ": journal-methods, method-card
#import "templates/references.typ": journal-references
#import "templates/dataset.typ": data-file, inputs-ready, pending-report, run-view, input-assets
#import "/.demolab/lib.typ": data-image, cite

#let data-file = data-file.with(article: "exp114")
#let meta = (
  tags: ("data", "v36.0.0"),
  title: "Coupling Locks Detuned PING Modules",
  created_at: "2026-09-15T00:00:00Z",
  updated_at: "2026-09-15",
  description: "Reciprocal excitation locked two compact PING modules and preserved signed phase offsets, but the sampled boundary did not depend on detuning.",
  collection: "gamma-gated-sparsity",
)
#let inputs = ("exp114",)

#let render-report(data-file) = [
  #journal-abstract(
    question: [We asked whether coupling between two compact PING modules can overcome input-induced gamma-frequency detuning and preserve the signed input difference as relative phase.],
    approach: [We crossed symmetric drive differences with reciprocal excitatory coupling in a complete three-seed grid.],
    finding: [Moderate coupling produced common spectral peaks and strong phase concentration across every tested detuning.],
    scope: [Signed phase offsets followed input direction, but the sampled locking threshold did not vary with detuning, leaving the full question incomplete.],
  )

  == Introduction

  Gamma frequency varies with cortical input, so nearby oscillating populations need not share a natural frequency. Lowet et al. (2015) proposed that synchronization is governed by competition between this detuning and coupling: sufficiently coupled oscillators adopt a common emergent frequency, while their stable phase difference can encode relative input.#cite(1)

  We test that central organizing principle in two finite conductance-based pyramidal–interneuron gamma (PING) modules. The question complements prior single-module demonstrations of PING and input-dependent gamma frequency in #link("/exp023/")[exp023] — #link("/exp023/")[_Turning the PING Loop On._]

  == Results

  #with-result-sections[
    #journal-result-card(
      title: "Two PING modules share excitation",
      expectation: [The structural schematic specifies the tested mechanism but is not empirical evidence: private afferents drive each E population, local E/I feedback generates PING, and E spikes weakly excite both populations in the other module.],
      visual: [#figure(
        data-image(data-file("exp114/figure1-network.svg"), width: 100%, alt: "Schematic of two conductance-based PING modules, each containing excitatory and inhibitory populations, with private afferent drive and reciprocal cross-module excitatory projections."),
        caption: [SNNLang-derived circuit rendered by SNNViz. Each module contains 100 E and 25 I cells. Black arrows are AMPA projections and red arrows are GABA projections; cross-module E→E and E→I event weights are varied jointly.],
        kind: image, supplement: [Figure],
      ) <fig:exp114-network>],
    )
    #journal-result-card(
      title: "Coupling collapses frequency gaps",
      observation: [Without cross coupling, both extreme signed drive differences produced median absolute spectral-peak separations of 7.78 Hz. At 6.25 nS, every detuning met the locking definition, with median peak gaps of 0–1.11 Hz and phase concentration of 0.71–0.85 (#result-figure-ref(<fig:exp114-locking>)). The common threshold demonstrates locking but not the required detuning-dependent boundary.],
      visual: [#figure(
        data-image(data-file("exp114/figure2-locking.png"), width: 100%, alt: "Two heatmaps over signed drive difference and cross-module coupling, showing phase concentration rising and peak-frequency difference falling as coupling increases."),
        caption: [Medians across three fixed connectivity and input seeds. A: concentration of the analytic gamma-phase difference. B: absolute difference between raw 20–80 Hz Welch peak bins. White outlines mark cells meeting all predeclared thresholds: concentration at least 0.70, peak difference at most 1.5 Hz, and gamma power fraction at least 0.20 in both modules.],
        kind: image, supplement: [Figure],
      ) <fig:exp114-locking>],
    )
    #journal-result-card(
      title: "Locked phase follows input direction",
      observation: [At both 6.25 and 10 nS, reversing detuning reversed the aggregate median phase-offset sign in both symmetric pairs. At 6.25 nS, offsets ranged from −0.042 to +0.101 cycles, although some individual seeds did not meet the lock thresholds. At 10 nS, every nonzero-detuning seed was individually locked and all phase signs followed input direction (#result-figure-ref(<fig:exp114-phase>, panel: "A")). Phase is omitted for aggregate unlocked cells because their circular mean has no stable interpretation.],
      visual: [#figure(
        data-image(data-file("exp114/figure3-phase.png"), width: 100%, alt: "Relative phase versus drive difference and example excitatory population traces without and with strong cross coupling."),
        caption: [A: median phase offset across three seeds for locked cells; unlocked cells are masked. B–C: seed-17 excitatory population counts for the +0.80 Hz-per-afferent condition, smoothed only for display with a 9 ms moving window, without and with 10 nS cross coupling.],
        kind: image, supplement: [Figure],
      ) <fig:exp114-phase>],
    )
  ]

  #journal-methods(body: (
    method-card([Construct two compact PING modules], [Each module contained 100 excitatory and 25 inhibitory conductance-based LIF cells with 25% independent local recurrent connectivity, including possible self-connections. AMPA and GABA decay constants were 1.5 and 7.5 ms; the timestep was 0.2 ms. Local event weights were 1 nS E→E, 10 nS E→I, 10 nS I→E, and 3.34 nS I→I. Positive E-to-I population lags supported, but did not uniquely prove, a PING sequence.]),
    method-card([Cross drive and coupling], [Each E cell received 200 equivalent independent Poisson afferents with 4 nS AMPA events. The two source rates were centred on 3.0 Hz and differed by −0.80, −0.30, 0, +0.30, or +0.80 Hz per afferent. Reciprocal E→E and E→I cross-module projections had 5% connectivity and event weights of 0, 2.5, 6.25, or 10 nS. We simulated every combination for 1.2 s using seeds 17, 29, and 43, and excluded the first 300 ms as burn-in.]),
    method-card([Measure gamma and phase], [We counted population spikes in 1 ms bins. For each module, the raw spectral estimate was a full-epoch Hann-window Welch density; $f_"peak"$ was the maximum raw bin from 20–80 Hz. Gamma power fraction divided power in that band by power from 5–150 Hz. We band-passed counts from 20–80 Hz with a fourth-order zero-phase Butterworth filter. Phase concentration $R_"phase"$ was the magnitude of the mean unit phasor formed from $phi_2-phi_1$, after discarding 50 ms at each filter edge; $phi_1$ and $phi_2$ are the modules’ analytic phases in radians. Phase offset was their circular mean difference in cycles.]),
    method-card([Apply fixed completion gates], [We aggregated each cell by the median across three seeds. A cell was locked only if median $R_"phase" >= 0.70$, median absolute peak difference was at most 1.5 Hz, and both median gamma power fractions were at least 0.20. Post-review completion additionally required both extreme detunings to exceed zero-detuning separation, a detuning-dependent first-lock boundary, opposite median phase signs with majority-seed support, the complete grid, and no median E or I rate increase above twofold at first lock.]),
  ))

  == Discussion

  Reciprocal excitation transforms different uncoupled spectral peaks into a shared peak with concentrated relative phase, and signed phase follows input direction. The complete fixed grid prevents selection of only favourable conditions. However, the first locking point is 6.25 nS for every sampled detuning. This rectangular boundary does not demonstrate the coupling–detuning trade-off expected for an Arnold tongue, so the full question remains incomplete.

  The result is qualitative rather than a replication of Lowet et al. The modules are small, use conductance-based LIF rather than Hodgkin–Huxley cells, receive stationary independent Poisson drive, and run for only 900 ms after burn-in. Welch resolution is 1.11 Hz and only three seeds quantify finite-size variation. Cross projections supply five expected E inputs per target versus 25 local E inputs; at first lock, no population’s median rate rose by more than 1.66-fold, but this does not prove amplitude-preserving weak coupling. The coarse four-point coupling grid may also conceal a narrow tongue boundary.

  == Conclusion

  Reciprocal excitation reproducibly locked two detuned compact PING modules and preserved the direction of relative input in their phase offsets. Yet locking began at the same sampled coupling for every detuning, failing the fixed Arnold-tongue criterion. After ten compute attempts, the compact experiment therefore provides clear subordinate observations but does not complete its full preregistered question.

  #run-view("exp114", inputs)
  #journal-references((
    (text: [E. Lowet, M. Roberts, A. Hadjipapas, A. Peter, J. van der Eerden, and P. De Weerd. “Input-Dependent Frequency Modulation of Cortical Gamma Oscillations Shapes Spatial Synchronization and Enables Phase Coding.” _PLOS Computational Biology_ 11(2), e1004072 (2015).], doi: "10.1371/journal.pcbi.1004072"),
  ))
]

#let report-body = if inputs-ready(data-file, inputs) { render-report(data-file) } else {
  pending-report(data-file, inputs, [Does coupling overcome detuning in two compact PING modules?], ([Circuit schematic], [locking map], [relative phase and example traces]))
}
#let meta = meta + (assets: input-assets("exp114", inputs))
#let body = journal-article("exp114", inputs, report-body, dataset-placed: inputs-ready(data-file, inputs))
