#import "contents.typ": with-contents
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets
#import "run-inputs.typ": input-assets
#import "manuscript-figures.typ": figure-description
#let data-file = data-file.with(article: "exp092")

#let meta = (
  status: "[▦ DATA]",
  title: "Gamma-Gated Sparsity Figure Gallery",
  date: "2026-08-20",
  description: "The figures from the gamma-gated sparsity manuscript, presented in sequence with their full captions.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp022", "exp023", "exp025", "exp033", "exp037", "exp038", "exp041", "exp042", "exp044", "exp046", "exp048", "exp049", "exp054",)
#let preview-figures = (
  (path: "exp023/overview_compound.png", label: "overview compound"),
  (path: "exp054/onset_super_compound.png", label: "onset super compound"),
  (path: "exp025/results_compound.png", label: "results compound"),
  (path: "exp038/loop_transfer_compound.png", label: "loop transfer compound"),
  (path: "exp049/training_curves.svg", label: "training curves"),
  (path: "exp041/rate_vs_fgamma.svg", label: "rate vs fgamma"),
  (path: "exp046/spikes_per_cycle_distribution.svg", label: "spikes per cycle distribution"),
  (path: "exp037/perturbation_curves.svg", label: "perturbation curves"),
  (path: "exp042/rhythm_compound.png", label: "rhythm compound"),
  (path: "exp044/dt_sweep.svg", label: "dt sweep"),
  (path: "exp048/varying_headline_stream.png", label: "varying headline stream"),
  (path: "exp048/acc_grid_tau_rate.png", label: "acc grid tau rate"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
// Provenance (HOUSESTYLE H9/H19): every run number in the prose and captions below is
// interpolated from the source experiment's numbers.json, never hand-typed, so a re-run
// of the collection's experiments updates this manuscript automatically. The figures are
// imported directly from those same experiments.
#let mean(a) = a.sum() / a.len()
#let r023 = data-json(data-file("exp023/numbers.json"))
#let r025 = data-json(data-file("exp025/numbers.json"))
#let r038 = data-json(data-file("exp038/numbers.json"))
#let r041 = data-json(data-file("exp041/numbers.json"))
#let r048 = data-json(data-file("exp048/numbers.json"))

// exp023 (Figure 1): free-running gamma peak and COBA f-I ceiling.
#let fgamma023 = calc.round(r023.f_gamma_hz.ping)
#let coba_fi_max = calc.round(calc.max(..r023.fi_curves.coba.e))

// exp025 (Figure 3): unpenalised operating points, averaged over seeds 42-44.
#let r25-target(r) = if r.keys().contains("rate_target_hz") { r.rate_target_hz } else { r.theta_u }
#let r25-pfg = if r025.keys().contains("rate_target_p_fgamma") { r025.rate_target_p_fgamma } else { r025.theta_p_fgamma }
#let r25off(m) = r025.results.filter(r => r.model == m and r25-target(r) == none)
#let ping25_rate = calc.round(mean(r25off("ping").map(r => r.rate_e)), digits: 1)
#let ping25_acc = calc.round(mean(r25off("ping").map(r => r.final_acc)))
#let coba25_rate = calc.round(mean(r25off("coba").map(r => r.rate_e)))
#let coba25_acc = calc.round(mean(r25off("coba").map(r => r.final_acc)))
#let rate_ratio25 = calc.round(mean(r25off("coba").map(r => r.rate_e)) / mean(r25off("ping").map(r => r.rate_e)))
// Total-population spike-count reduction (Discussion, Conclusion and Future Directions): E-only rates from `results`, PING inhibitory
// rate from `rate_target_p_fgamma` (the only op-point I measurement). COBA I is silent (loop off).
#let ping25_i = r25-pfg.filter(r => r.model == "ping" and r25-target(r) == none).first().i_rate
#let spike_ratio = calc.round((1024 * mean(r25off("coba").map(r => r.rate_e))) / (1024 * mean(r25off("ping").map(r => r.rate_e)) + 256 * ping25_i))

// exp038 (Figure 4): inference-time loop-transfer endpoints (ei = 0 -> ei = 1).
#let ei038 = r038.at("ei_sweep_summary", default: r038.ei_sweep)
#let ei0 = ei038.filter(r => r.ei_strength == 0.0).first()
#let ei1 = ei038.filter(r => r.ei_strength == 1.0).first()
#let tr_e0 = calc.round(ei0.hid_rate_hz)
#let tr_e1 = calc.round(ei1.hid_rate_hz)
#let tr_i1 = calc.round(ei1.inh_rate_hz)
#let tr_acc0 = calc.round(ei0.acc)
#let tr_acc1 = calc.round(ei1.acc)
#let tr_drop = calc.round(ei0.acc - ei1.acc)
#let tr_ratio = calc.round(ei0.hid_rate_hz / ei1.hid_rate_hz)

// exp041 (Figure 6): affine fit r_E = a + p f_gamma.
#let fit_a = calc.round(r041.fit.a_affine, digits: 2)
#let fit_p = calc.round(r041.fit.p_affine, digits: 3)
#let fit_r2 = calc.round(r041.fit.r2_affine, digits: 3)
#let acc41_fast = calc.round(mean(r041.results.filter(r => r.tau_gaba_ms == 4.5).map(r => r.acc)), digits: 1)
#let acc41_slow = calc.round(mean(r041.results.filter(r => r.tau_gaba_ms == 27.0).map(r => r.acc)), digits: 1)
#let acc41_drop = calc.round(acc41_fast - acc41_slow, digits: 1)
// Canonical trained gamma at tau_GABA = 9 ms (exp041, 3 seeds); sets the streaming cycle bound (Streaming classification, Discussion).
#let fg_canon_raw = mean(r041.results.filter(r => r.tau_gaba_ms == 9.0).map(r => r.f_gamma_hz))
#let fg_canon = calc.round(fg_canon_raw)
#let Tg_canon = calc.round(1000 / fg_canon_raw)
#let tau_floor_cyc = calc.round(15 * fg_canon_raw / 1000, digits: 1)
#let sat_lo_cyc = calc.round(40 * fg_canon_raw / 1000, digits: 1)
#let sat_hi_cyc = calc.round(50 * fg_canon_raw / 1000, digits: 1)

// exp022 (training hub): canonical training length, folded into Training methods.
#let r022 = data-json(data-file("exp022/numbers.json"))
#let hub_epochs = r022.standard.epochs
// exp049 (Figure 5): released-loop training length.
#let r049 = data-json(data-file("exp049/numbers.json"))
#let ep049 = r049.config.epochs
// exp049 rhythmicity (Loop-weight interventions): lobe-trough contrast R at epoch 1 vs canonical, and the
// final trainable-init range, all read from the cached per-epoch logs (numbers.json).
#let r49_can = calc.round(r049.rhythmicity.canonical_contrast, digits: 2)
#let r49_ep1 = calc.round(r049.rhythmicity.epoch1_contrast_trainable, digits: 2)
#let r49_fin_lo = calc.round(r049.rhythmicity.final_contrast_trainable_min, digits: 2)
#let r49_fin_hi = calc.round(r049.rhythmicity.final_contrast_trainable_max, digits: 2)
// Frozen-PING control trained-state operating point (Loop-weight interventions), from the summary rows.
#let r49_fz = r049.summary.filter(r => r.condition == "frozen_ping")
#let r49_fz_acc = calc.round(mean(r49_fz.map(r => r.acc)))
#let r49_fz_e = calc.round(mean(r49_fz.map(r => r.e_rate_hz)))

// exp037 (Figure 8): PING robustness to spike deletion.
#let r037 = data-json(data-file("exp037/numbers.json"))
#let pert037 = r037.at("perturbation_summary", default: r037.perturbation)
#let ping_base37 = calc.round(pert037.filter(r => r.model == "ping" and r.mode == "drop" and r.level == 0.0).first().acc)
#let ping_drop80 = calc.round(pert037.filter(r => r.model == "ping" and r.mode == "drop" and r.level == 0.8).first().acc)

// exp048 (Figure 12): streaming operating point (tau = 200 ms, input 25 Hz).
#let op48_acc = calc.round(r048.grid_sweep_agg.filter(r => r.tau_ms == 200.0 and r.input_rate_hz == 25.0).first().acc)
#let rate48-at(rate) = r048.encoding_rate_psychometric.curve.filter(r => r.input_rate_hz == rate).first()
#let rate48-p05 = rate48-at(0.5)
#let rate48-p2 = rate48-at(2.0)
#let rate48-p5 = rate48-at(5.0)
#let rate48-p10 = rate48-at(10.0)
#let stream48-correct = r048.varying_headline.seg_correct.filter(x => x == 1).len()
#let stream48-total = r048.varying_headline.seg_correct.len()

// exp046 (Figure 7): per-(cell, cycle) spike-count distribution, pooled over the tau_GABA sweep.
#let r046 = data-json(data-file("exp046/numbers.json"))
#let p0_046 = calc.round(r046.global_fracs.zero * 100)
#let p1_046 = calc.round(r046.global_fracs.one * 100)
#let pleq1_046 = calc.round((r046.global_fracs.zero + r046.global_fracs.one) * 100, digits: 2)
#let pmulti_046 = calc.round((r046.global_fracs.two + r046.global_fracs.three_plus) * 100)

// exp044 (Figure 10): integration-timestep invariance (E rate + accuracy bands over the dt sweep).
#let r044 = data-json(data-file("exp044/numbers.json"))
#let er044_lo = calc.round(calc.min(..r044.results.map(r => r.e_rate_hz)), digits: 1)
#let er044_hi = calc.round(calc.max(..r044.results.map(r => r.e_rate_hz)), digits: 1)
#let acc044_lo = calc.round(calc.min(..r044.results.map(r => r.acc)), digits: 1)
#let acc044_hi = calc.round(calc.max(..r044.results.map(r => r.acc)), digits: 1)
#let acc044_pp = calc.round(calc.max(..r044.results.map(r => r.acc)) - calc.min(..r044.results.map(r => r.acc)), digits: 1)

// exp042 (Figure 9): inhibitory-jitter operating points. Both arms are read at the
// SAME jitter magnitude, sigma = 14 ms — only the KIND of jitter differs (per-cell vs
// cycle-coherent). sigma = 14 ms is a measured grid point on both sweeps where the
// realised I rate is still within a few percent of baseline (genuinely rate-matched on
// both arms), whereas at sigma = 100 ms the finite trial window truncates the displaced
// bursts and realised I drops ~24%. Means over seeds 42-44.
#let r042 = data-json(data-file("exp042/numbers.json"))
#let jit_e_base = calc.round(mean(r042.results.filter(r => r.condition == "baseline").map(r => r.e_rate_hz)))
#let jit_e_cyc = calc.round(mean(r042.jitter_sweep.filter(r => r.sigma_ms == 14.0).map(r => r.e_rate_hz)))
#let jit_i_cyc = calc.round(mean(r042.jitter_sweep.filter(r => r.sigma_ms == 14.0).map(r => r.i_rate_hz)))
#let jit_i_cell = calc.round(mean(r042.cell_jitter_sweep.filter(r => r.sigma_ms == 14.0).map(r => r.i_rate_hz)))

// exp049 (Figure 5): additional released-loop aggregates (frozen-control I rate, trainable
// E-rate spread across the three initialisations, and the all-condition accuracy band).
#let r49_fz_i = calc.round(mean(r49_fz.map(r => r.i_rate_hz)))
#let r49_tr_e_means = ("trainable_ping_init", "trainable_zero_init", "trainable_small_init").map(c => mean(r049.summary.filter(r => r.condition == c).map(r => r.e_rate_hz)))
#let r49_tr_e_lo = calc.round(calc.min(..r49_tr_e_means))
#let r49_tr_e_hi = calc.round(calc.max(..r49_tr_e_means))
#let r49_acc_lo = calc.round(calc.min(..r049.summary.map(r => r.acc)))
#let r49_acc_hi = calc.round(calc.max(..r049.summary.map(r => r.acc)))

// exp023 (Figure 1): PING inhibitory f-I ceiling under the strongest drive.
#let ping_i_max = calc.round(calc.max(..r023.fi_curves.ping.i))

// exp033 (Figure 2): mean-field Hopf-onset constants (drive threshold, crossing frequency,
// supercritical amplitude-scaling fit).
#let r033 = data-json(data-file("exp033/numbers.json"))
#let hopf_iext = calc.round(r033.results.hopf.I_ext_star, digits: 2)
#let hopf_fstar = calc.round(r033.results.hopf.freq_star_Hz, digits: 1)
#let crit_r2 = calc.round(r033.results.criticality.A2_r2, digits: 3)

#let body = [
  #figure(
    data-image(data-file("exp023/overview_compound.png"), width: 100%, alt: "Two-column comparison of the free-running network, each column headed by a wiring schematic. Left (COBA, loop off): schematic of input to a lone excitatory population with no inhibitory population, an asynchronous excitatory raster, a power spectrum with no gamma peak, and an f-I curve rising to about " + str(coba_fi_max) + " Hz. Right (PING, loop on): schematic of the excitatory-inhibitory loop (E to I via W_ei, I to E via W_ie), synchronous inhibitory bursts, gamma-banded excitatory raster, a sharp spectral peak near 42 Hz, and an excitatory rate held roughly an order of magnitude lower across two decades of input drive."),
    caption: [*A single recurrent E→I→E loop simultaneously generates a gamma rhythm and clamps the excitatory firing rate.* Free-running activity of the two-population conductance-based network (excitatory pool $N_E = 1024$, inhibitory pool $N_I = 256$; canonical biophysical parameters, #link("/exp109/#single-neuron-and-synapse-dynamics")[Neuron and synapse dynamics]) under matched Poisson drive, in two configurations. Each column shows, from top: a wiring schematic, a single-trial spike raster, the excitatory power spectral density (PSD), and the excitatory $f$–$I$ curve. *(A)* COBA baseline with the recurrent loop disabled ($W^(E I) = W^(I E) = 0$): input projects to the excitatory (E) population only, with no inhibitory (I) population (schematic). The E spike raster is asynchronous, the Welch PSD of the summed E population shows no gamma-band peak, and the excitatory $f$–$I$ curve rises to $approx #coba_fi_max$ Hz under the strongest drive tested. *(B)* PING configuration with the loop engaged (schematic: E→I via $W^(E I)$, I→E via $W^(I E)$; no I→I or E→E synapse): the inhibitory (I) population fires synchronous bursts, the E raster forms gamma bands, the PSD shows a discrete peak at $f_gamma approx #fgamma023$ Hz, and on axes shared with (A) the E rate is held approximately an order of magnitude lower across two decades of input drive while the I rate rises to $approx #ping_i_max$ Hz. Source: #link("/exp023/")[exp023].],
  )
  #figure(
    data-image(data-file("exp054/onset_super_compound.png"), width: 100%, alt: "Nine panels. Top row: heatmaps of excitatory rate, inhibitory rate, and lobe-trough rhythmicity across the W_EI by W_IE coupling plane, with rhythmicity near zero on the COBA edges and rising to about 0.98 at strong coupling. Middle row: single-trial rasters at three points along the coupling diagonal, from asynchronous to sharp gamma volleys. Bottom row: the mean-field reduction, showing a complex eigenvalue pair crossing into the right half-plane at 0.60 nA, a continuous amplitude onset with coincident up and down ramps, and gamma frequency falling with the GABA time constant in agreement with the spiking measurement."),
    caption: [*Gamma emerges through a smooth, reversible onset across the coupling plane, consistent with a supercritical Hopf bifurcation of the mean-field reduction.* Nine panels (A–I). *(A–C)* Steady-state measurements across the $11 times 11$ $W^(E I) times W^(I E)$ coupling plane: mean excitatory firing rate (A), mean inhibitory firing rate (B), and the lobe–trough rhythmicity contrast $R$ (C, #link("/exp109/#measurement-and-analysis")[Measurement and analysis]), which is $approx 0$ along the two COBA edges and rises to $approx 0.98$ toward strong coupling. *(D–F)* Single-trial E (black) and I (red) rasters at three points sampled along the $W^(I E) = 2 W^(E I)$ diagonal (circled in C): the loop-disabled origin (D, asynchronous), weak coupling ($R < 0.5$, E), and strong coupling (F, sharp gamma volleys). *(G–I)* The four-dimensional conductance mean-field reduction (#link("/exp109/#mean-field-reduction")[Mean-field reduction]): a complex-conjugate eigenvalue pair crosses into the right half-plane at external drive $I^star = #hopf_iext$ nA (G), locating a Hopf bifurcation at $f^star approx #hopf_fstar$ Hz; the steady-state oscillation amplitude grows continuously across the onset with coincident up- and down-ramp branches (H), the signature of a supercritical, reversible transition; and the predicted gamma frequency falls with $tau_"GABA"$ in qualitative agreement with the spiking measurement (I). Source: #link("/exp054/")[exp054] (coupling-plane maps and mean-field, incorporating #link("/exp033/")[exp033]); spiking $f_gamma$ from #link("/exp041/")[exp041].],
  )
  #figure(
    data-image(data-file("exp025/results_compound.png"), width: 100%, alt: "Trained-network comparison. Top: single-trial rasters, with COBA firing densely and asynchronously and PING firing in gamma bands with synchronous inhibitory bursts. Bottom left: test accuracy per epoch, both configurations converging to about 91 percent. Bottom right: the accuracy-rate frontier, with PING lying above and to the left of COBA at every spike-budget setting; at the unpenalised operating point PING reaches about 91 percent near 12 Hz against COBA's 91 percent near 181 Hz."),
    caption: [*Trained PING matches COBA classification accuracy while operating at an order-of-magnitude lower excitatory firing rate.* Both configurations were trained on MNIST by surrogate-gradient descent (#link("/exp109/#training")[Training methods]). Top: representative single-trial rasters of the trained networks: COBA fires densely and asynchronously with the inhibitory population silent, whereas PING fires in gamma bands with synchronous inhibitory bursts (red) above excitatory spikes (black). For visualisation, each raster is an extended 400 ms replay of one digit, twice the 200 ms presentation used for training and quantitative evaluation. Bottom left: test accuracy per epoch, both configurations converging to $approx #ping25_acc%$. Bottom right: accuracy–rate frontier traced by sweeping the sample-wise hidden-E rate ceiling (#link("/exp109/#training")[Training methods]); each marker shows mean hidden-E firing rate (abscissa) and test accuracy (ordinate) across three independent seeds, with standard-error bars. PING lies above and to the left of COBA across the sweep. At the unpenalised operating point (starred), PING reached $approx #ping25_acc%$ at $approx #ping25_rate$ Hz, against COBA's $approx #coba25_acc%$ at $approx #coba25_rate$ Hz. Source: #link("/exp025/")[exp025]; rate-attractor analysis in #link("/exp024/")[exp024].],
  )
  #figure(
    data-image(data-file("exp038/loop_transfer_compound.png"), width: 100%, alt: "Inference-time loop activation on COBA networks trained with three seeds. Top: illustrative seed-42 rasters at loop-off and full loop strength. Bottom: across-seed mean rates and accuracy with standard-deviation bands."),
    caption: [*Engaging the recurrent loop at inference reproduces the PING firing-rate reduction with no weight update.* COBA networks trained with seeds 42–44 were evaluated with recurrent E→I coupling scaled from $e i = 0$ to $e i = 1$, without retraining. Top: illustrative seed-42 rasters at loop-off and full strength. Bottom left: across-seed mean excitatory (black) and inhibitory (red) rates; excitation fell approximately $#tr_ratio$-fold (from $approx #tr_e0$ to $approx #tr_e1$ Hz) as inhibition rose to $approx #tr_i1$ Hz. Bottom right: mean accuracy fell from $approx #tr_acc0%$ to $approx #tr_acc1%$ (a $approx #tr_drop$ pp cost). Shaded bands show sample SD across seeds. The rate gating appears when the loop is wired in, whereas the accuracy loss reflects the absence of a feedforward compensation learned in the presence of the loop. Source: #link("/exp038/")[exp038].],
  )
  #figure(
    data-image(data-file("exp049/training_curves.svg"), width: 100%, alt: "Four per-epoch training-metric panels with the loop weights released for training, comparing three initialisations (canonical, zero, and 0.1x canonical) against a frozen-PING control. (A) Test accuracy: all conditions overlap at roughly 90 percent. (B) Mean excitatory rate: the trainable conditions rise well above the frozen control, which stays gated at a low rate. (C) Mean inhibitory rate: the trainable conditions collapse to near zero within a few epochs while the frozen control's inhibitory rate stays high. (D) Rhythmicity: the frozen control holds near its canonical value while every trainable initialisation drains toward zero."),
    caption: [*Releasing the loop weights to gradient descent prunes the rhythm within a single epoch, from every initialisation.* Per-epoch training metrics with the recurrent weights $W^(E I), W^(I E)$ made trainable under the Dale's-law clamp, over #ep049 epochs on MNIST. Lines are the mean of three seeds (42–44); shading is the across-seed range. Conditions differ only in initialisation of the loop weights: canonical PING values (black), zero (red), and $0.1 times$ canonical (amber), against a frozen-PING control (grey, dashed). *(A)* Test accuracy: all conditions overlap at $approx #r49_acc_lo$–$#r49_acc_hi%$. *(B)* Mean excitatory firing rate: the trainable conditions rise to $approx #r49_tr_e_lo$–$#r49_tr_e_hi$ Hz as the loop is dismantled, while the frozen control remains gated near $#r49_fz_e$ Hz. *(C)* Mean inhibitory firing rate: the trainable conditions collapse to $approx 0$ Hz within a few epochs, while the frozen control's inhibitory rate rises to $approx #r49_fz_i$ Hz as its readout trains. *(D)* Lobe–trough rhythmicity contrast: the frozen control holds at $approx #r49_can$ while every trainable initialisation drains to $approx #r49_fin_lo$–$#r49_fin_hi$. Source: #link("/exp049/")[exp049].],
  )
  #figure(
    data-image(data-file("exp041/rate_vs_fgamma.svg"), width: 100%, alt: "Top: mean post-training excitatory firing rate against measured gamma frequency, with per-condition means over three seeds and error bars on both axes; a linear fit passes through every error bar. Bottom: test accuracy over the same sweep, declining systematically toward the low-frequency conditions."),
    caption: [*Post-training excitatory rate covaries affinely with gamma frequency across a sweep of inhibitory decay kinetics.* Networks were trained from scratch at each of six values of the GABA decay constant $tau_"GABA" in {4.5, 6, 9, 12, 18, 27}$ ms. Changing $tau_"GABA"$ alters both the realised gamma frequency and the duration and integrated influence of inhibition (#link("/exp109/#training")[Training methods]). Top: mean post-training excitatory firing rate against measured $f_gamma$; markers are per-condition means over three seeds, with error bars ($plus.minus$ SD) on both axes. The linear fit $r_E = #fit_a + #fit_p f_gamma$ passes through every error bar ($R^2 = #fit_r2$). Bottom: mean test accuracy declined from $#acc41_fast%$ at $tau_"GABA" = 4.5$ ms to $#acc41_slow%$ at $27$ ms, a $#acc41_drop$ percentage-point tradeoff. The sweep establishes covariance with realised frequency, not an independent causal effect of frequency alone. Source: #link("/exp041/")[exp041].],
  )
  #figure(
    data-image(data-file("exp046/spikes_per_cycle_distribution.svg"), width: 100%, alt: "A row of bar charts, one per value of the GABA time constant, each showing the distribution of spikes per (cell, cycle) pair over the categories 0, 1, 2, and 3 or more. Across the sweep the zero-spike category dominates and the one-spike category is next, with the two-and-more categories nearly empty, so almost every pair contains at most one spike."),
    caption: [*The affine rate law follows from a near-binary per-cycle firing statistic: each excitatory cell contributes at most one spike per gamma cycle.* Excitatory spikes were resolved into (cell, cycle) pairs by assigning each spike to the gamma cycle inferred from peaks of the population inhibitory-burst rate (#link("/exp109/#measurement-and-analysis")[Measurement and analysis]). Each panel shows the distribution of spikes per pair (0, 1, 2, or $>= 3$) at one value of $tau_"GABA"$. Across the sweep, $P(0) approx #p0_046%$, $P(1) approx #p1_046%$, and the multi-spike fraction ($>= 2$) stayed near $approx #pmulti_046%$; pooled over all conditions, $#pleq1_046%$ of pairs contained at most one spike. Source: #link("/exp046/")[exp046].],
  )
  #figure(
    data-image(data-file("exp037/perturbation_curves.svg"), width: 100%, alt: "Across-seed mean test accuracy with standard-deviation bands against spike deletion and Poisson addition for PING and COBA."),
    caption: [*The gating is robust to spike deletion but fragile to spike addition, the expected signature of a phase-based code.* PING and COBA networks trained with seeds 42–44 were perturbed at inference without retraining. Left: deletion of a random fraction of emitted spikes. PING retained $approx #ping_drop80%$ mean accuracy through $80%$ deletion, close to its unperturbed value. Right: addition of Poisson spikes, expressed relative to each population's baseline rate. PING fell toward chance as off-phase spikes disrupted the rhythm; COBA, having no rhythm to protect, was more tolerant to addition. Lines show across-seed means and shaded bands show sample SD. Source: #link("/exp037/")[exp037].],
  )
  #figure(
    data-image(data-file("exp042/rhythm_compound.png"), width: 100%, alt: "Two inhibitory-jitter manipulations at the same jitter magnitude, sigma 14 ms, that both hold the mean inhibitory rate fixed. Left column (per-cell jitter): rasters show bursts smeared into a continuous shunt, and the excitatory rate falls to near zero while accuracy falls to chance. Right column (cycle-coherent jitter): whole bursts are displaced but within-burst synchrony is preserved, the excitatory rate rises from about " + str(jit_e_base) + " to about " + str(jit_e_cyc) + " Hz, and accuracy remains high. The bottom sweep panels overlay the realised mean inhibitory rate as a grey line, flat where the two arms are compared. Identical mean inhibition, opposite excitatory outcome."),
    caption: [*Two inhibitory-jitter manipulations at the same magnitude that hold the mean inhibitory rate fixed drive the excitatory rate in opposite directions, isolating timing from level.* The trained PING inhibitory spike train was perturbed at inference while the mean per-cell inhibitory rate was held constant. Both arms used the same jitter magnitude, $sigma = 14$ ms — only the _kind_ of jitter differs. Top: single-trial rasters (E black, I red). Bottom: mean excitatory rate (black) and accuracy (red) versus jitter magnitude $sigma$, with the realised mean inhibitory rate overlaid (grey). Left: per-cell jitter smears each burst into a continuous shunt; the excitatory rate fell to $approx 0$ Hz (inhibitory rate $approx #jit_i_cell$ Hz) and accuracy fell to chance. Right: cycle-coherent jitter displaces whole bursts while preserving within-burst synchrony; the excitatory rate rose from $approx #jit_e_base$ to $approx #jit_e_cyc$ Hz (inhibitory rate $approx #jit_i_cyc$ Hz, matched to the left arm) and accuracy remained high. Both arms are read where the realised inhibitory rate is matched to within a few percent; at larger $sigma$ the cycle-coherent excitatory rate climbs further, but the finite trial window truncates the most-displaced bursts and realised inhibition falls, so the strict comparison is anchored here. Identical mean inhibition, opposite excitatory outcome: the gate is the phase structure of inhibition, not its level. Source: #link("/exp042/")[exp042].],
  )
  #figure(
    data-image(data-file("exp044/dt_sweep.svg"), width: 100%, alt: "Post-training mean excitatory rate (black diamonds, left axis) and test accuracy (red squares, right axis) against integration timestep on a logarithmic abscissa spanning 0.05 to 1.0 ms. The excitatory rate stays within a narrow low band and is non-monotonic in the timestep, so finer stepping does not buy a lower rate; accuracy stays essentially flat across the range. Both training and inference use the same timestep at each point."),
    caption: [*The firing-rate reduction is a physical-time property, invariant to the integration timestep over a twentyfold range.* The network was trained and evaluated at matched integration timestep $Delta t$ for each value across $[0.05, 1.0]$ ms (logarithmic abscissa). Left ordinate (black diamonds): post-training mean excitatory rate, confined to a $#er044_lo$–$#er044_hi$ Hz band and non-monotonic in $Delta t$, so finer stepping does not buy a lower rate. Right ordinate (red squares): test accuracy, flat within $#acc044_pp$ pp ($#acc044_lo$–$#acc044_hi%$). Because both training and inference used the same $Delta t$ at each point, the figure tests invariance of the training-plus-inference pipeline, not the generalisation of one trained network to a varied inference step. Source: #link("/exp044/")[exp044].],
  )
  #figure(
    data-image(data-file("exp048/varying_headline_stream.png"), width: 100%, alt: "A single concatenated stream of five MNIST digits, each with its own presentation duration and Poisson input rate. The two weakest-drive segments are misclassified while the other three are correct; hidden excitatory and inhibitory rasters maintain gamma cycles throughout."),
    caption: [*A PING network trained on isolated digits classifies a continuous, unsegmented stream whose presentation timing varies from segment to segment.* A single input stream concatenates five MNIST digits, each with its own presentation duration ($25$–$200$ ms) and Poisson input rate ($10$–$200$ Hz); the network was given no segmentation cue and was not retrained. Top: the five digit thumbnails with their per-segment (duration, rate) and predicted labels; thumbnail opacity indicates input rate. Below: hidden excitatory and inhibitory rasters, showing gamma cycles maintained throughout (sparser under weak drive, denser under strong drive) and the sliding leaky-integrator readout traces. #stream48-correct of #stream48-total digits were classified correctly; the two errors occurred in the weakest-drive segments and are interpreted against the population curve in Figure 12B. Source: #link("/exp048/")[exp048].],
  )
  #figure(
    data-image(data-file("exp048/acc_grid_tau_rate.png"), width: 100%, alt: "Two panels: a heatmap of streaming accuracy across segment duration and input rate, and a psychometric curve measured at fixed 200 millisecond presentation and readout windows showing chance performance below 0.5 hertz and a steep transition between 1 and 5 hertz."),
    caption: [*Streaming accuracy has distinct integration-time and encoding-rate evidence floors.* *(A)* Per-segment accuracy across presentation duration $tau$ and input rate, averaged over three seeds and 1,200 segments per grid cell. For $tau <= 15$ ms, accuracy did not exceed $approx 80%$ at any input rate; above that floor, diagonal iso-accuracy contours show an approximate dependence on $tau dot "rate"$. The trained operating point ($tau = 200$ ms, $25$ Hz) reached $#op48_acc%$. *(B)* Probability of a correct classification versus encoding rate with both presentation duration and readout window fixed at $200$ ms. All points belong to the same psychometric curve. Performance remained at chance through #rate48-p05.input_rate_hz Hz, was clearly informative by #rate48-p2.input_rate_hz Hz, and reached #calc.round(100 * rate48-p5.accuracy, digits: 1)% at #rate48-p5.input_rate_hz Hz; the dotted line marks the $25$ Hz training rate. Thus the Figure 11 error at $200$ ms and #rate48-p10.input_rate_hz Hz occurred in a condition with #calc.round(100 * rate48-p10.accuracy, digits: 1)% population accuracy, above the nonviable encoder regime. For scale, $tau = 15$ ms is approximately $#tau_floor_cyc$ times the canonical gamma period $T_gamma approx #Tg_canon$ ms, but gamma frequency is not manipulated independently. Source: #link("/exp048/")[exp048].],
  )
]
#body
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [This gallery brings together the experiments underlying the gamma-gated sparsity manuscript. Each figure follows its own input selection; integrated numerical captions require the full set of inputs.],
    preview-figures.map(item => item + (description: figure-description(item.path.split("/").first()),)),
    json-inputs: ("exp022", "exp023", "exp025", "exp033", "exp037", "exp038", "exp041", "exp042", "exp044", "exp046", "exp048", "exp049",),
  )
}

#let meta = meta + (assets: input-assets("exp092", inputs))
#let body = with-datasets("exp092", inputs, report-body)
#let body = with-contents(body)
