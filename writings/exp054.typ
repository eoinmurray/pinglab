#import "contents.typ": with-contents
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp054")

#let meta = (
  status: "[▦ DATA]",
  title: "Gamma Turns On Across the Coupling Map",
  date: "2026-06-15",
  updated_at: "2026-08-28",
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
  (path: "exp054/onset_super_compound.png", label: "onset super compound"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let body = [
  == Abstract

  Across 121 untrained PING coupling conditions, the *lobe–trough contrast* of the excitatory spike-time autocorrelation was near zero on the uncoupled edges and generally larger in the coupled interior. Excitatory firing spanned 2.77–94.31 Hz; contrast reached 0.984 at the strong-coupling corner and 0.997 elsewhere. Private-input nulls remained below 0.068 over the tested firing range, whereas shared-input coincidence can produce substantial contrast without an inhibitory loop. These single-seed controls support private input for this comparison, not universal rate invariance. A separate conductance mean-field comparison suggests a possible onset mechanism without establishing the spiking transition's bifurcation type.

  #run-view("exp054", inputs)

  == Results

  #figure(
    data-image(data-file("exp054/turnon_maps_compound.png"), width: 100%,
      alt: "Maps of E rate, I rate and lobe–trough contrast over the coupling grid, above three example E/I rasters showing asynchronous firing and increasingly separated volleys."),
    caption: [*Top*: per-neuron E and I rates and lobe–trough contrast across the 11×11 coupling grid; $W_(E I)$ and $W_(I E)$ denote E-to-I and I-to-E coupling strengths. On either zero edge the loop is broken: E fired at 94.31 Hz and contrast was 0.00169; I was silent when E-to-I coupling is zero and its high rate on the other edge is colour-clipped. *Bottom*: diagonal examples A/B/C had contrasts 0.00169/0.270/0.984. E spikes are black, I spikes red; each raster shows 200 ms from the first 160 E and 48 I neurons. Coupled conditions generally show stronger temporal structure, but the map is not strictly monotonic. One seed per condition; no uncertainty estimate.],
  )

  #figure(
    data-image(data-file("exp054/grid_rasters.png"), width: 100%,
      alt: "E/I rasters at every other coupling-grid coordinate: dense edge activity and separated volleys in much of the coupled interior."),
    caption: [E/I rasters at a 6×6 subset of coupling coordinates; the maps above use all 121 conditions. The left column has no E-to-I coupling and silent I neurons; the bottom row has no inhibitory feedback to E. Many interior conditions show separated volleys. Display windows and neuron subsets match Figure 1; measurements use all neurons over the full post-burn recording.],
  )

  #figure(
    data-image(data-file("exp054/grid_autocorr.png"), width: 100%,
      alt: "E-population autocorrelograms at a 6×6 subset of coupling coordinates, with lobe and trough markers and a chance reference."),
    caption: [E-population autocorrelograms at the same 6×6 coordinates, shown over 0–50 ms; dotted lines mark the chance reference $A = 1$. Markers locate the selected lobe (▲) and trough (▼) of the smoothed curve. Edge curves are near chance. In the coupled interior, short-lag clustering, suppression between volleys and later recurrence peaks provide complementary evidence. The contrast scores the first lobe and trough, not the later peak's frequency.],
  )

  #figure(
    data-image(data-file("exp054/rate_invariance.png"), width: 100%,
      alt: "Contrast versus measured E firing rate: private-input null values remain small, while the sparsest shared-input nulls show elevated contrast."),
    caption: [Contrast against measured E firing rate. Without an inhibitory loop, the private-input null (black) stayed at or below 0.0671 over 0.97–94.31 Hz; the shared-input null (grey) reached 0.50 at its sparsest firing. Shared afferents can create short-lag coincidence that this score detects. Red points show the PING coupling grid, including its near-zero edges. These finite, single-seed controls do not prove rate invariance or establish rhythmicity from contrast alone.],
  )

  #figure(
    data-image(data-file("exp054/null_autocorr.png"), width: 100%,
      alt: "Low-rate shared- and private-input null autocorrelograms selected by approximate rate matching, with each actual rate labelled."),
    caption: [Null autocorrelograms nearest to target E rates of 1, 2.5 and 5 Hz. Shared-input examples (top) actually fired at 0.40/1.72/4.20 Hz; private-input examples (bottom) at 0.97/1.88/4.90 Hz. The rates and spike counts are not equal. Shared-input examples illustrate central coincidence without inhibitory feedback; private-input examples show smaller contrast and finite-sample fluctuations around chance.],
  )

  === Comparison with a mean-field onset

  #figure(
    data-image(data-file("exp054/onset_super_compound.png"), width: 100%,
      alt: "Nine panels compare coupling maps and example rasters with a separate mean-field eigenvalue crossing, amplitude sweep and frequency-versus-inhibitory-decay comparison."),
    caption: [*A–F*: the same coupling maps and diagonal rasters as Figure 1. *G–I*: the #link("/exp033/")[conductance mean-field reference] with a 4 mV effective-noise scale. A complex eigenvalue pair crossed the imaginary axis near external drive $I^* = 0.596$ nA (G); the finite up/down branches of peak-to-peak E-rate amplitude nearly coincided (H). This is compatible with a soft onset, but does not establish supercriticality of the spiking map: theory varies drive, whereas that map varies coupling. Mean-field onset frequency and the three-seed median #link("/exp041/")[spiking frequency] both decreased with inhibitory decay $tau_"GABA"$ (I), with substantial quantitative differences. The phenomenological comparison is a possible mechanism, not a fitted explanation of the same transition.],
  )

  #context if target() != "html" { pagebreak(weak: true) }
  == Methods

  Untrained PING populations tested coupling-dependent temporal structure; uncoupled controls tested the influence of input sharing. The mean-field comparison reused numerical observations from a separate conductance model.

  #set math.equation(numbering: "(1)")
  #counter(math.equation).update(0)
  #show math.equation.where(block: true): equation => context {
    if target() == "html" {
      html.elem("div", attrs: (class: "exp054-equation", style: "display:flex;align-items:center;gap:1em"), {
        html.elem("div", attrs: (style: "flex:1;min-width:0;overflow-x:auto"), equation)
        html.elem("span", numbering("(1)", ..counter(math.equation).at(equation.location())))
      })
    } else { equation }
  }

  + *Sweep coupling.* I simulated 256 E and 256 I neurons at all 11×11 combinations of $W_(E I) = 0$–3 µS and $W_(I E) = 0$–6 µS. Each E neuron received a private 100 Hz Poisson channel with identity weight 0.5. I used seed 42, one trial, 0.25 ms steps and 1,000 ms recordings; I discarded the first 100 ms.
  + *Construct uncoupled controls.* I set both coupling strengths to zero. I scanned private input at 1/2/5/10/20/40/70/100 Hz and shared input at 8/12/16/20/28/40/60/100 Hz. Shared input used 200 channels, weight 0.2 and 95% initial zero connections. The 100 Hz private origin was shared with the coupling grid, giving 136 unique probes.
  + *Measure rates and autocorrelation.* I divided each population's post-burn spike count by neuron count and 0.9 s. I binned E spikes at $Delta t = 1$ ms, obtaining counts $r(t)$ in $n = 900$ bins. For integer lag $ell = 1, dots, 100$, I calculated

    $ A(ell) = 1 / (⟨ r ⟩^2 (n - ell)) sum_(t=0)^(n-ell-1) r(t) r(t+ell). $ <exp054-autocorrelation>

    Here $t$ indexes bins and $⟨ r ⟩$ is their mean count; physical lag is $ell Delta t$. Zero-padded FFT correlation, divided by available overlap and mean count squared, gives chance reference 1. I excluded zero lag.
  + *Locate the lobe and trough.* I smoothed with weights $(0.25, 0.5, 0.25)$, filling the excluded zero-lag entry from the first lag. I found the first local minimum from lag 2, and the preceding maximum from lag 1. Their smoothed heights define

    $ "contrast" = ("lobe" - "trough") / ("lobe" + "trough"). $ <exp054-contrast>

    For $0 <= "trough" <= "lobe"$ and a positive denominator, this lies in $[0,1]$. A missing trough or invalid denominator leaves the score undefined; no trough floor is imposed on contrast.
  + *Compare mean-field onset.* I used the #link("/exp033/")[four-variable conductance model] at 4 mV effective noise. I continued fixed points over 401 drives from 0–4 nA and refined the leading-eigenvalue crossing with Brent's method. I swept 25 drives from 0.1 nA below to 0.55 nA above the crossing in both directions, carrying endpoint states. I integrated 2 s per drive with LSODA and measured peak-to-peak E-rate amplitude ($"ms"^(-1)$) over the final 500 ms. Retained amplitudes were reused; missing trajectories were not reconstructed.
  + *Compare frequencies.* I repeated the theoretical crossing search at inhibitory decays 4.5/6/9/12/18/27 ms. I overlaid the median frequency across three seeded #link("/exp041/")[spiking-network measurements] at each decay; I did not refit the mean-field noise scale.

  == Appendix: reading the autocorrelogram

  The raw lag product $sum_t r(t) r(t+ell)$ counts spike pairs separated by $ell$ bins. Only $n-ell$ bin pairs exist at that lag: the last $ell$ bins have no partner. Dividing by this overlap converts the raw sum into an average product per available pair, removing the taper caused by finite recording length. Dividing again by $⟨ r ⟩^2$ places independent firing near $A = 1$; this is a reference level, not a lower bound.

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
#let body = with-contents(body)
