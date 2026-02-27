---
title: docs.3-recipes
description: Recipes for metrics and algorithms we are developing
---

# Recipes (Mathematical)

Below are the exact recipes used in the codebase. All math is written to match the implementations in `src/pinglab/` and the experiment codefiles.

## Notation

Let spikes be $(t_i, n_i)$ with times in ms and neuron ids. For a population of size $N_{\text{pop}}$ and bin width $\Delta t$ (ms), we compute binned counts $c_b$ and the population rate

$$
r_b = \frac{c_b}{N_{\text{pop}} \, \Delta t / 1000}.
$$

When smoothing with a Gaussian kernel $g_\sigma$, the smoothed rate is $r^\ast = r \ast g_\sigma$.

## Autocorrelation Peak (Population Rate)

**Goal:** quantify rhythmicity by the first significant autocorrelation peak of the population rate.

1) Compute the population rate $r_b$ (optionally smoothed).  
2) Center the signal: $x_b = r_b - \bar{r}$.  
3) Compute the normalized autocorrelation using an overlap-normalized correlation and variance normalization:

$$
C(\tau) = \frac{1}{n-|\tau|}\sum_{b} x_b x_{b+\tau}, \quad
\rho(\tau) = \frac{C(\tau)}{\mathrm{Var}(x)}.
$$

4) Restrict to lags $\tau \in [\tau_{\min}, \tau_{\max}]$ (ms), and find the first significant peak using minimum height and prominence thresholds. If none exist, take the maximum in that window. If the rate is constant or variance is zero, the curve is zeroed.

**Output:** peak value $\rho(\tau^\star)$ plus the rate and autocorrelation curves.

## Mean Pairwise Cross-Correlation Peak (E–E Spikes)

**Goal:** measure synchrony by the first significant peak of the mean pairwise cross-correlation between excitatory spike count series.

1) For each excitatory neuron $i$, bin spike counts $s_i[b]$ and mean-center: $\tilde{s}_i[b] = s_i[b] - \bar{s}_i$.  
2) Form the mean pairwise cross-correlation via

$$
R_{\text{pair}} = \frac{R_{\text{sum}} - R_{\text{auto}}}{N_E (N_E-1)},
$$

where $R_{\text{sum}}$ is the autocorrelation of $\sum_i \tilde{s}_i$ and $R_{\text{auto}}$ is the sum of individual autocorrelations.  
3) Normalize by the number of overlapping bins (divide by $n-|\tau|$) and by the average pairwise variance term used in code:

$$
\text{denom} = \frac{\left(\sum_i \sigma_i\right)^2 - \sum_i \sigma_i^2}{N_E (N_E-1)},
$$

where $\sigma_i$ is the standard deviation of the centered count series for neuron $i$. If $\text{denom} \le 0$, the variance normalization is skipped.  
4) Restrict to $\tau \in [\tau_{\min}, \tau_{\max}]$ and take the first significant peak; if none, take the maximum in the window.

**Output:** peak value plus the lag and correlation curve.

## Coherence (Smoothed Rates)

**Goal:** measure oscillatory coherence across excitatory neurons by mean pairwise cross-correlation of smoothed rate series.

1) For each neuron, bin spikes, smooth with a Gaussian ($\sigma$ ms), and mean-center.  
2) Compute mean pairwise cross-correlation as above, normalize by overlap, and then (if available) by the same average pairwise variance term used in the xcorr peak metric.  
3) Take the absolute value and clip to $[0,1]$ (this matches the `abs_value` flag in code).

The coherence curve is $C_{\text{coh}}(\tau)$; the scalar coherence is

$$
\text{coherence\_peak} = \max_{\tau} C_{\text{coh}}(\tau).
$$

**Output:** peak value and the coherence curve over lags.

## Lagged Coherence (Rhythmicity)

**Goal:** quantify rhythmicity at frequency $f$ by the consistency of windowed Fourier coefficients over time.

1) Compute the population rate $r(t)$ and (optionally) subtract the mean in each window.  
2) Choose window length $T_w = \frac{\text{window\_cycles}}{f}$ and lag $T_\ell = \frac{\text{lag\_cycles}}{f}$.  
3) For each window $k$, compute the complex coefficient

$$
c_k = \frac{1}{|W_k|}\sum_{t \in W_k} r(t)\, e^{-i2\pi f t},
$$

which is the window-averaged Fourier coefficient at frequency $f$. The magnitude $|c_k|$ encodes how strongly the window oscillates at $f$, and the phase $\arg(c_k)$ encodes the local phase of that oscillation. Comparing consecutive windows therefore measures how stable the oscillation is over time.

**Implementation details:** in code, windows slide in steps of $T_\ell$ from $t=\text{start}$ to $t=\text{end}-T_w$. For each window $[t, t+T_w)$ we select the binned rate samples that fall inside, convert times to seconds via $t_s = (t - t_0)\cdot 10^{-3}$, optionally remove the window mean, optionally apply the Hann taper, and compute the average complex coefficient exactly as

$$
c_k = \frac{1}{n_k}\sum_{j=1}^{n_k} x_{k,j}\, e^{-i 2\pi f t_{k,j}},
$$

where $x_{k,j}$ are the windowed (and optionally demeaned/tapered) rate samples and $n_k$ is the number of samples in the window. This matches the code, which computes `coeffs.append(np.mean(window_x * expo))` with `expo = exp(-i 2π f t)` and stores the window center time.

An optional Hann taper is applied to each window in code when `taper="hann"` to reduce edge effects and spectral leakage. The Hann window is

$$
w_j = \tfrac{1}{2}\left(1 - \cos\left(\frac{2\pi j}{n_k - 1}\right)\right), \quad j=0,\dots,n_k-1,
$$

and the windowed samples are $x_{k,j} \leftarrow w_j x_{k,j}$ before computing $c_k$.

4) Compute lagged coherence

$$
\lambda = \frac{\left|\mathbb{E}\left[c_k c_{k+1}^\ast\right]\right|}
{\sqrt{\mathbb{E}|c_k|^2 \, \mathbb{E}|c_{k+1}|^2}},
\quad \lambda \in [0,1].
$$

For the spectrum, evaluate $\lambda(f)$ over a grid of frequencies.

**Output:** $\lambda$ (or $\lambda(f)$) plus rate and window metadata.

## Firing Rate Matching (exp.2)

**Goal:** for each parameter scan cell, find an external drive $I_E$ that matches a target excitatory rate $r_E^\star$.

1) Use a short simulation time $T_{\text{sim}}$ and a rate-estimation window $[t_0, t_1]$.  
2) Evaluate a grid of $I_E$ values and compute

$$
\text{score}(I_E) = \left(r_E(I_E) - r_E^\star\right)^2,
$$

where $r_E$ is the mean excitatory firing rate computed from the spikes in $[t_0, t_1]$.  
3) Pick the best $I_E$ and refine with two narrower searches around it.  
4) Use the matched $I_E$ for the full simulation of that scan cell.

**Output:** the matched drive $I_E$ and the resulting $r_E$ per scan cell.
