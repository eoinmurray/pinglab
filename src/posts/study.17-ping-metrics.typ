// title: study.17-ping-metrics
// date: 2026-03-05T11:07:20Z
// description: Study: study.17-ping-metrics

#let config = json("_artifacts/study.17-ping-metrics/config.json")
#let results = json("_artifacts/study.17-ping-metrics/results.json")






= Topology


#figure(
  image("_artifacts/study.17-ping-metrics/graph_dark.png"),
  caption: [Network topology.],
)




= Raster


#figure(
  image("_artifacts/study.17-ping-metrics/raster_main_main_00_dark.png"),
  caption: [Raster plot.],
)




= Population Firing Rates


#figure(
  image("_artifacts/study.17-ping-metrics/pop_rates_main_main_00_dark.png"),
  caption: [Population firing rate for E and I neurons.],
)




== Non-differentiable Autocorrelation Metrics


#figure(
  image("_artifacts/study.17-ping-metrics/autocorr_main_main_00_dark.png"),
  caption: [Autocorrelation of population firing rate with annotated metrics.],
)



We extract five metrics from the population rate autocorrelation to characterize
the oscillatory dynamics of the E-I network.

*Oscillation strength* — height of the first non-zero-lag peak in the
normalized autocorrelation. Values near 1.0 indicate a highly periodic signal;
values near 0 indicate noise. This is the primary measure of whether the network
is oscillating.

*Period $T$* — lag of the first non-zero-lag peak in milliseconds. For a PING
network this corresponds to the gamma cycle period. Here $T$ =
#results.E.oscillation_period_ms ms gives a frequency of {(1000 /
results.E.oscillation_period_ms).toFixed(1)} Hz, squarely in the gamma band.

*Decay time constant $tau$* — time constant of an exponential envelope $A e^(-t/tau)$ fitted to the autocorrelation peak heights. Measures how long the
oscillation persists. Large $tau$ means sustained rhythmicity; small $tau$
means the oscillation damps quickly. E cells show $tau$ =
#results.E.decay_tau_ms ms and I cells $tau$ = #results.I.decay_tau_ms ms.

*SNR* — ratio of the first peak height to the mean absolute value of the noise
floor (lags $> 600$ ms) in the unnormalized autocorrelation. High SNR means the
oscillation stands well above background fluctuations. Both populations show SNR
$> 40$, indicating strong oscillations.

*Zero-lag FWHM* — full width at half maximum of the central autocorrelation
peak. A narrow FWHM indicates sparse, well-timed spiking; a broad FWHM indicates
bursting. Pathological bursting produces a wide central peak with weak side
peaks. Both populations show FWHM = #results.E.zero_lag_fwhm_ms ms,
indicating sharp synchrony without bursting.

#table(
  columns: 6,
  [Population], [Strength], [$T$ (ms)], [$tau$ (ms)], [SNR], [FWHM (ms)],
  [E], [#results.E.oscillation_strength], [#results.E.oscillation_period_ms], [#results.E.decay_tau_ms], [#results.E.snr], [#results.E.zero_lag_fwhm_ms],
  [I], [#results.I.oscillation_strength], [#results.I.oscillation_period_ms], [#results.I.decay_tau_ms], [#results.I.snr], [#results.I.zero_lag_fwhm_ms],
)



== Differentiable PSD Metrics


The five autocorrelation metrics above are computed from hard spikes via binning, peak finding, and curve fitting — none of which are differentiable. To use PING health as a regularization term during BPTT training, we need a differentiable proxy.

By the Wiener-Khinchin theorem the autocorrelation and power spectral density (PSD) are Fourier transform pairs, so every autocorrelation metric has a spectral equivalent. The FFT is a linear operation and fully differentiable in PyTorch (`torch.fft.rfft`), so we can build a differentiable loss entirely in the frequency domain.


=== Power spectrum


The PSD of the binned population rate reveals the harmonic structure of the oscillation. The fundamental $f_0 = #results.E.f0_hz$ Hz (period $= #results.E.oscillation_period_ms$ ms) and its harmonics at $2f_0, 3f_0, dots$ are clearly visible.

#figure(
  image("_artifacts/study.17-ping-metrics/psd_main_main_00_dark.png"),
  caption: [Power spectral density of population firing rate. Vertical lines mark the fundamental and its harmonics.],
)




=== Soft fundamental estimate


Rather than finding the spectral peak (a non-differentiable argmax), compute the center-of-mass frequency over a plausible range (e.g. 5–80 Hz):

$
  f_0 = frac(sum_(f) f dot S(f), sum_(f) S(f))
$


where $S(f) = |"FFT"(r(t))|^2$ is the power spectrum of the population rate. This is a weighted average — differentiable everywhere.

A subtlety: the center-of-mass gives the power-weighted *average* frequency, not the fundamental. When harmonics are present, it gets pulled toward the middle of the spectrum. In the analysis code we use argmax to find the true fundamental; for the differentiable loss a soft-argmax (temperature-scaled softmax over frequency bins) would be needed, or the search band can be restricted to only the fundamental range.


=== Gaussian comb mask


A clean PING oscillation has power at the fundamental and its harmonics ($f_0, 2f_0, 3f_0, dots$). To separate signal from noise in a differentiable way, build a soft comb mask:

$
  W(f) = sum_(k=1)^(K) exp(-frac((f - k f_0)^2, 2 sigma^2))
$


with $sigma approx 1.75$ Hz and $K = 5$ harmonics. This produces a smooth weight $in [0, 1]$ at every frequency bin. Because $f_0$ is a differentiable function of $S(f)$, and $W$ is a smooth function of $f_0$, gradients flow through the entire computation.

#figure(
  image("_artifacts/study.17-ping-metrics/psd_comb_main_main_00_dark.png"),
  caption: [PSD with Gaussian comb mask W(f) overlaid (red fill). Each comb tooth is a Gaussian centered on a harmonic of f0.],
)



The comb teeth align with the spectral peaks. Power captured under the comb is classified as signal; everything between the teeth is noise.


=== Signal / noise decomposition


$
  P_"signal" = sum_f W(f) dot S(f)    P_"noise" = sum_f (1 - W(f)) dot S(f)
$


#figure(
  image("_artifacts/study.17-ping-metrics/psd_decomp_main_main_00_dark.png"),
  caption: [Signal (blue) vs noise (red) decomposition of the PSD using the Gaussian comb mask.],
)



The comb correctly attributes harmonic power to the signal rather than noise. A plain bandpass at $[25, 45]$ Hz would miss power at $2f_0, 3f_0$ and incorrectly treat it as noise. The spectral SNR is #results.E.snr_psd for E and #results.I.snr_psd for I — the I population has cleaner oscillations with less inter-harmonic noise.


=== Why harmonics must be handled


#table(
  columns: 3,
  [Network state], [Power spectrum], [Comb response],
  [Clean oscillation], [Sharp peaks at $f_0, 2f_0, 3f_0$], [$P_"signal"$ high, $P_"noise"$ low],
  [No oscillation], [Flat spectrum], [$P_"signal" approx P_"noise"$],
  [Pathological bursting], [Broad peaks, power between harmonics], [$P_"noise"$ elevated],
)



=== Adaptive comb tracking


As training shifts the oscillation frequency, $f_0$ moves smoothly via the soft estimate, the comb teeth slide correspondingly, and the loss landscape remains smooth. No discrete jumps or mode switching.


=== Composite loss


$
  L_"ping" = -alpha log frac(P_"signal", P_"total") + beta(f_0 - f_"target")^2 + gamma frac(P_"noise", P_"signal")
$


#table(
  columns: 3,
  [Term], [Controls], [ACF equivalent],
  [$-alpha log(P_"signal" / P_"total")$], [Concentrate power into harmonic structure], [Oscillation strength + SNR],
  [$beta(f_0 - f_"target")^2$], [Lock fundamental to desired frequency], [Period $T$],
  [$gamma dot P_"noise" / P_"signal"$], [Suppress inter-harmonic noise], [Clean oscillation, anti-burst],
)


Decay $tau$ and FWHM do not need explicit terms. A long $tau$ corresponds to a narrow spectral peak, which is already rewarded by term 1. A low FWHM means no excess broadband power, which is penalized by term 3.


=== Loss components


#table(
  columns: 6,
  [Population], [$f_0$ (Hz)], [SNR], [$L_"con"$], [$L_"frq"$], [$L_"noi"$],
  [E], [#results.E.f0_hz], [#results.E.snr_psd], [#results.E.l_concentration], [#results.E.l_frequency], [#results.E.l_noise],
  [I], [#results.I.f0_hz], [#results.I.snr_psd], [#results.I.l_concentration], [#results.I.l_frequency], [#results.I.l_noise],
)


$L_"frq" = 0$ because $f_"target"$ defaults to the estimated $f_0$ (no target frequency was specified). In training, setting $f_"target"$ to a desired gamma frequency would produce a non-zero frequency penalty.


=== Hyperparameters


Starting values: $alpha = 1$, $beta = 0.01$, $gamma = 0.1$, $sigma = 1.75$ Hz, $K = 5$.