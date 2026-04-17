---
layout: ../layouts/MarkdownLayout.astro
title: "Metrics"
---

# Metrics

Every run computes a fixed set of metrics so sweeps, calibrations, and ablations are comparable. They divide into four groups: dynamical observables from the spike rasters, spectral observables from the population rate, spectral predictions from the frozen recurrent matrix, and training-side metrics from the optimisation loop.

## Dynamical observables

Given binary spike arrays $s_E \in \{0,1\}^{T \times N_E}$ and $s_I \in \{0,1\}^{T \times N_I}$ recorded after a burn-in window, let $T_{\text{sec}} = T \Delta t / 1000$.

- **Population firing rates.** $r_E = (\sum s_E)/(N_E T_{\text{sec}})$ and $r_I = (\sum s_I)/(N_I T_{\text{sec}})$, in Hz.
- **E/I ratio.** $r_I / r_E$. Balanced-network theory predicts $r_I / r_E \approx 2\text{-}4$ for PING-like operation.
- **Active fraction.** Share of E neurons that spiked at least once. Flags dead populations (act $< 5\%$) and runaway saturation (act $> 95\%$).
- **Population-count CV.** Spikes are binned at $2$ ms; let $n_b$ be the total $E$-spike count in bin $b$. Then $\text{CV} = \sigma(n_b)/\mu(n_b)$. Async networks sit near $\text{CV} \approx 1$; gamma-locked PING packs spikes into cycle peaks and empty troughs, pushing $\text{CV} \gg 1$.

## Spectral observables

The population-rate time series $r_E(t)$ is the primary signal for oscillation analysis.

- **Power spectral density.** Population rate is computed in $2$ ms bins over the stimulus window, mean-centred, and FFT'd.
- **Fundamental frequency $f_0$.** Found by a subharmonic-aware peak search in the $[5, 80]$ Hz band. If the PSD at $f^*/2$ exceeds $30\%$ of the peak, $f_0 = f^*/2$, else $f_0 = f^*$. The subharmonic rule catches the case where pyramidal neurons fire every other gamma cycle.

## Spectral predictions from $J$

After training, the frozen recurrent matrix is reassembled with E columns positive and I columns negative (Dale's law encoded in the signs). Linear analysis of $J$ gives two frequency predictions that bracket the measured $f_0$.

- **Spectral radius.** $\rho(J) = \max_i |\lambda_i|$ — linear stability margin. $\rho < 1$ is the quiescent regime; the edge of chaos sits near $\rho \approx 1$.
- **Linear $f_0$ prediction.** The eigenvalue $\lambda^* = a + bi$ with largest $|b|$ is the dominant complex mode. Its imaginary part gives a continuous-rate frequency prediction $f_0^{\text{lin}} = |b|/(2\pi) \cdot 1000$ Hz.
- **Refractory-floor prediction.** A spiking network cannot oscillate faster than one cycle per $(\tau_{\text{ref}}^E + \tau_{\text{GABA}})$, giving the lower-bound frequency $f_0^{\text{cor}} = 1000/(\tau_{\text{ref}}^E + \tau_{\text{GABA}})$.

Measured $f_0$ typically lies between $f_0^{\text{cor}}$ and $f_0^{\text{lin}}$: GABA decay and the spiking nonlinearity stretch each cycle beyond what the linear eigenvalue predicts.

## Training metrics

Each epoch writes a structured record to metrics.jsonl and accumulates into metrics.json:

- **Optimisation.** loss (train cross-entropy), test_loss, lr, grad_norm (mean post-clip), grad_ratios (per-layer $\|\text{grad}\|/\|W\|$).
- **Performance.** acc (test-set accuracy, %), new_best flag, plus run-level best_acc and best_epoch. The checkpoint saved to weights.pth is the **best-accuracy** state, not the final-epoch state.
- **Dynamics snapshot.** All observables from the preceding sections, computed on a reference digit presented through the current network — so $f_0$, $\text{CV}$, and rates can be traced epoch-by-epoch alongside accuracy.

For inference runs, metrics.json holds best_acc, n_correct, and a per-sample test_predictions.json with {"idx", "true", "pred", "correct", "logits"} for downstream confusion-matrix or calibration analysis.

## Pathology tags

A WarningTracker scans the per-epoch stream for four patterns; each fires only under a joint condition to suppress false positives:

| Tag | Condition | Typical cause |
| --- | --------- | ------------- |
| ⚠ dead | act $< 1\%$ for $\geq 3$ epochs **and** no acc improvement for $\geq 5$ | drive too weak, weights too sparse |
| ⚠ saturated | act $> 95\%$ for $\geq 3$ epochs **and** no acc improvement for $\geq 5$ | drive too strong, inhibition collapsed |
| ⚠ NaN | loss is NaN on any epoch | gradient explosion, div-by-zero |
| ⚠ grad-clip | fraction of clipped gradients $> 50\%$ | cm-back-scale too low, lr too high |
| ⚠ stuck | no_progress_since $\geq 10$ | past the elbow; early-stop candidate |

The "AND not improving" gate is load-bearing — sparse-coding snntorch sits at $1\text{-}3\%$ activity while learning fine, and gamma-locked PING sits at $> 80\%$; neither should trip a warning on activity alone.

## Firing regimes

- **Sparse-coding snntorch**: 1–3% activity at $\geq 88\%$ accuracy is healthy, not dead.
- **Gamma-locked PING**: $\geq 80\%$ activity with CV $\sim 0.4$ is the intended regime, not saturation.
- **PING $\Delta t$ ceiling.** E→I→E requires $\Delta t \ll \tau_{\text{GABA}} \approx 5$ ms. At $\Delta t \geq 1.5$ ms inhibition can't complete a cycle within one step; network saturates and accuracy collapses. Do not infer with PING at $\Delta t > 1$ ms.

