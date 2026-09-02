#import "contents.typ": contents-here, with-contents, result-card, with-numbered-equations, with-result-sections
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp054")

#let meta = (
  status: "[▦ DATA | v33.0.0]",
  title: "Pinglab Rythmicity Metric",
  created_at: "2026-06-15T00:00:00Z",
  updated_at: "2026-08-31T00:00:00Z",
  description: "Lobe–trough contrast across untrained PING coupling strengths, with private- and shared-input null controls and a separate mean-field onset comparison.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp054",)
#let preview-figures = (
  (path: "exp054/turnon_maps_compound.png", label: "turnon compound"),
  (path: "exp054/grid_rasters.png", label: "grid rasters"),
  (path: "exp054/grid_autocorr.png", label: "grid autocorr"),
  (path: "exp054/rate_invariance.png", label: "rate invariance"),
  (path: "exp054/null_autocorr.png", label: "null autocorr"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let body = [
  == Abstract

  Asked where rhythmic activity appears as reciprocal PING coupling activates
  across the coupling map. Swept coupling directions in a network and compared
  autocorrelation structure with private- and shared-input controls.

  Rhythmic contrast was weak on uncoupled edges and stronger inside the coupled
  map; shared input imitated contrast without a recurrent loop. Supports the
  private-input control and an onset mechanism, but does not establish rate
  invariance or the spiking transition's bifurcation type.

  #contents-here()

  == Results

  #with-result-sections[

  On either zero-coupling edge the loop was broken: E fired at 94.31 Hz and
  contrast was 0.00169, while I was silent when E-to-I coupling was zero. The
  diagonal examples i/ii/iii had contrasts 0.00169/0.270/0.984. Coupled conditions
  generally showed stronger temporal structure, although the map was not
  strictly monotonic.

  #figure(
    data-image(data-file("exp054/turnon_maps_compound.png"), width: 100%,
      alt: "Maps of E rate, I rate and lobe–trough contrast over the coupling grid, above three example E/I rasters showing asynchronous firing and increasingly separated volleys."),
    caption: [*(A–C)* Per-neuron E rate, I rate and lobe–trough contrast,
      respectively, across the 11×11 coupling grid; $W_(E I)$ and $W_(I E)$
      denote E-to-I and I-to-E coupling strengths. The high I rate on one zero
      edge is colour-clipped. *(D–F)* Rasters at diagonal conditions i–iii,
      respectively; E spikes are black and I spikes red.
      Each raster shows 200 ms from the first 160 E and 48 I neurons. One seed
      per condition; no uncertainty estimate.],
  )

  Many coupled-interior conditions showed separated volleys, whereas the
  zero-E-to-I column had silent I neurons and the zero-I-to-E row lacked
  inhibitory feedback to E.

  #figure(
    data-image(data-file("exp054/grid_rasters.png"), width: 100%,
      alt: "E/I rasters at every other coupling-grid coordinate: dense edge activity and separated volleys in much of the coupled interior."),
    caption: [*(A–AJ)* E/I rasters at a 6×6 subset of coupling coordinates,
      ordered row-major from high to low $W_(I E)$ and low to high $W_(E I)$;
      the maps above use all 121 conditions. Display windows and neuron subsets match Figure 1;
      measurements use all neurons over the full post-burn recording.],
  )

  Edge autocorrelograms remained near chance. In the coupled interior,
  short-lag clustering, suppression between volleys and later recurrence peaks
  supplied complementary temporal evidence.

  #figure(
    data-image(data-file("exp054/grid_autocorr.png"), width: 100%,
      alt: "E-population autocorrelograms at a 6×6 subset of coupling coordinates, with lobe and trough markers and a chance reference."),
    caption: [*(A–AJ)* E-population autocorrelograms at the same 6×6 coordinates
      and in the same row-major coupling order, shown over 0–50 ms. Dotted lines
      mark the chance reference $A_"corr" = 1$;
      markers locate the selected lobe (▲) and trough (▼) of the smoothed curve.
      $R_"contrast"$ scores the first lobe and trough, not the later peak's
      frequency.],
  )

  Without an inhibitory loop, the private-input null stayed at or below 0.0671
  over 0.97–94.31 Hz, while the shared-input null reached 0.50 at its sparsest
  firing. Shared afferents can therefore create short-lag coincidence detected
  by this score. These finite, single-seed controls do not prove rate invariance
  or establish rhythmicity from contrast alone.

  #figure(
    data-image(data-file("exp054/rate_invariance.png"), width: 100%,
      alt: "Contrast versus measured E firing rate: private-input null values remain small, while the sparsest shared-input nulls show elevated contrast."),
    caption: [Contrast against measured E firing rate. Black and grey points are
      the private- and shared-input nulls; red points are the PING coupling grid,
      including its near-zero edges.],
  )

  Shared-input examples showed central coincidence without inhibitory feedback;
  private-input examples showed smaller contrast and finite-sample fluctuations
  around chance. Their approximate rate matching did not make rates or spike
  counts equal.

  #figure(
    data-image(data-file("exp054/null_autocorr.png"), width: 100%,
      alt: "Low-rate shared- and private-input null autocorrelograms selected by approximate rate matching, with each actual rate labelled."),
    caption: [Null autocorrelograms nearest to target E rates of 1, 2.5 and 5 Hz.
      Shared-input examples *(A–C)* fired at 0.40/1.72/4.20 Hz; private-input
      examples *(D–F)* at 0.97/1.88/4.90 Hz.],
  )

  ]

  == Methods

  Untrained PING populations tested coupling-dependent temporal structure; uncoupled controls tested the influence of input sharing. The mean-field comparison reused numerical observations from a separate conductance model.

  === Compute

  + *Sweep coupling.* We simulated 256 E and 256 I neurons at all 11×11 combinations of $W_(E I) = 0$–3 µS and $W_(I E) = 0$–6 µS. Each E neuron received a private 100 Hz Poisson channel with identity weight 0.5. We used seed 42, one trial, 0.25 ms steps and 1,000 ms recordings; we discarded the first 100 ms.
  + *Construct uncoupled controls.* We set both coupling strengths to zero. We scanned private input at 1/2/5/10/20/40/70/100 Hz and shared input at 8/12/16/20/28/40/60/100 Hz. Shared input used 200 channels, weight 0.2 and 95% initial zero connections. The 100 Hz private origin was shared with the coupling grid, giving 136 unique probes.
  === Analyse

  #set enum(start: 3)

  + *Measure rates and autocorrelation.* We divided each population's post-burn spike count by neuron count and 0.9 s. We binned E spikes at $Delta t_"bin" = 1$ ms, obtaining counts $n_E[k]$ in $N_"bin" = 900$ bins. For integer lag $ell = 1, dots, 100$, we calculated

    $ A_"corr"(ell) = 1 / (⟨ n_E ⟩^2 (N_"bin" - ell)) sum_(k=0)^(N_"bin"-ell-1) n_E[k] n_E[k+ell]. $ <exp054-autocorrelation>

    Here $k$ indexes bins and $⟨ n_E ⟩$ is their mean count; physical lag is $ell Delta t_"bin"$. Zero-padded FFT correlation, divided by available overlap and mean count squared, gives chance reference 1. We excluded zero lag.
  + *Locate the lobe and trough.* We smoothed with weights $(0.25, 0.5, 0.25)$, filling the excluded zero-lag entry from the first lag. We found the first local minimum from lag 2, and the preceding maximum from lag 1. Their smoothed heights define

    $ R_"contrast" = (A_"lobe" - A_"trough") / (A_"lobe" + A_"trough"). $ <exp054-contrast>

    For $0 <= "trough" <= "lobe"$ and a positive denominator, this lies in $[0,1]$. A missing trough or invalid denominator leaves the score undefined; no trough floor is imposed on contrast.
  + *Compare mean-field onset.* We used the #link("/exp033/")[four-variable conductance model] at 4 mV effective noise. We continued fixed points over 401 drives from 0–4 nA and refined the leading-eigenvalue crossing with Brent's method. We swept 25 drives from 0.1 nA below to 0.55 nA above the crossing in both directions, carrying endpoint states. We integrated 2 s per drive with LSODA and measured peak-to-peak E-rate amplitude ($"ms"^(-1)$) over the final 500 ms. Recorded amplitudes were reused; missing trajectories were not reconstructed.
  === Present

  #set enum(start: 6)

  + *Compare frequencies.* We repeated the theoretical crossing search at inhibitory decays 4.5/6/9/12/18/27 ms. We overlaid the median frequency across three seeded #link("/exp041/")[spiking-network measurements] at each decay; we did not refit the mean-field noise scale.

  #run-view("exp054", inputs)

  == Appendix: reading the autocorrelogram

  The raw lag product $sum_k n_E[k] n_E[k+ell]$ counts spike pairs separated by $ell$ bins. Only $N_"bin"-ell$ bin pairs exist at that lag: the last $ell$ bins have no partner. Dividing by this overlap converts the raw sum into an average product per available pair, removing the taper caused by finite recording length. Dividing again by $⟨ n_E ⟩^2$ places independent firing near $A_"corr" = 1$; this is a reference level, not a lower bound.

  The Wiener–Khinchin route computes the lag products by zero-padding the count trace, taking its fast Fourier transform (FFT), multiplying by the complex conjugate and inverse-transforming. This costs $O(n log n)$ for all lags rather than a full $O(n^2)$ direct correlation, where $O$ denotes asymptotic computational scaling.

  A central lobe above chance describes short-lag spike clustering. A later trough can reflect suppression between volleys; a subsequent peak near the period is separate evidence of recurrence. Contrast is zero when its lobe and trough are equal, and reaches one if a positive lobe is paired with a zero trough. It summarizes their relative heights without measuring a stable oscillation frequency. Finite recordings, missing extrema and shared-input coincidence therefore matter when interpreting this scalar alongside rasters and longer-lag structure.

]
#body
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [Where does gamma appear in the excitatory–inhibitory coupling plane? Compare activity, rhythm diagnostics, null controls, and the mean-field onset prediction.],
    preview-figures, json-inputs: (),
  )
}

#let meta = meta + (assets: input-assets("exp054", inputs))
#let body = with-datasets("exp054", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-numbered-equations(body)
#let body = with-contents(body)
