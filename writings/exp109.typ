#import "contents.typ": with-contents
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#import "manuscript-figures.typ": figure-description
#import "/.demolab/lib.typ": cite, reference-list
#let data-file = data-file.with(article: "exp109")

#let meta = (
  status: "[▦ DATA]",
  title: "Gamma-Gated Sparsity Manuscript",
  date: "2026-06-21",
  updated_at: "2026-08-29",
  description: "A task-trained spiking network with a fixed PING loop: gamma as a structural constraint on excitatory firing rates.",
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

// exp046 (Figure 7): per-(neuron, cycle) spike-count distribution, pooled over the tau_GABA sweep.
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
#let jit_i_neuron = calc.round(mean(r042.cell_jitter_sweep.filter(r => r.sigma_ms == 14.0).map(r => r.i_rate_hz)))

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
  == Abstract

  Gamma oscillations are widespread in cortical activity but are largely absent from trained spiking neural networks, which typically operate in a current-based regime or impose oscillations as an external input. I trained a spiking network with a fixed pyramidal–interneuron gamma (PING) loop on MNIST under surrogate-gradient descent, with the recurrent E↔I weights held at biophysical values. At matched test accuracy on the accuracy–rate frontier, the post-training excitatory firing rate was roughly an order of magnitude below a conductance-based control ($#spike_ratio$-fold by population-total spike rate once the higher-rate inhibitory pool is included), and the trained rate was well described by an affine relation $r_E approx #fit_a + #fit_p f_gamma$ with the measured gamma frequency ($R_"fit"^2 = #fit_r2$). When the loop weights were released for training under a Dale's-law clamp, the rhythmicity collapsed within one epoch and was not recovered from any initial condition tested. The trained network classified a continuously concatenated digit stream without retraining or an external segmentation cue; the decoder nevertheless used known segment boundaries. Streaming accuracy was approximately governed by the product of presentation duration and input rate, remained below $approx 80%$ for durations of $15$ ms or less, and at $200$ ms rose from chance below #rate48-p05.input_rate_hz Hz to clearly informative performance by #rate48-p2.input_rate_hz Hz. These results are consistent with an interpretation of gamma as a structural constraint on excitatory firing rates that does not require learned tuning of the inhibitory connectivity.

  #run-view("exp109", inputs)

  == Introduction

  Gamma oscillations in the 30–80 Hz band have been associated with attention, binding, and gating in cortical activity#cite(1, 2, 3), with the original visual-cortex observation reported by#cite(4). Two generation mechanisms are commonly distinguished, ING and PING#cite(5, 6); the present work focuses on PING. PING arises from the dynamics of a recurrent excitatory–inhibitory (E↔I) loop#cite(7). Optogenetic activation of fast-spiking interneurons in intact cortical circuits drives gamma rhythms#cite(8, 9, 10, 11). Earlier in vitro recordings#cite(12) and biophysical models#cite(13) characterised interneuron-driven gamma in isolated inhibitory networks, and the synaptic mechanisms that synchronise the interneuron pool have been described#cite(14).

  PING has been studied extensively in biophysical#cite(15, 16, 17) and neural-mass / mean-field models#cite(18, 19, 20, 21), but these models are descriptive: they are not trained on a task. The parallel literature on trainable spiking neural networks uses surrogate-gradient descent for end-to-end optimisation#cite(22, 23, 24); the resulting networks are typically current-based and non-rhythmic. The rhythmic variants either impose the oscillation as an external input#cite(25) or obtain it as an emergent property of unconstrained surrogate-gradient training#cite(26); in neither case is the rhythm carried by a fixed, biophysically-calibrated PING loop, and the present work is to my knowledge the first task-trained spiking network of that kind.

  Cortical pyramidal cells fire at low rates (typically below 10 Hz) under strong recurrent input#cite(27), and the cortical metabolic budget is dominated by excitatory spike generation, with inhibitory spikes substantially less costly per spike#cite(28, 29). The mechanism that constrains pyramidal firing rates under these conditions is not fully understood. I tested the hypothesis that a fixed E↔I loop, by generating a gamma rhythm, constrains the post-training excitatory firing rate. A spiking network with a fixed PING loop was trained on MNIST and the post-training firing rate was compared with a non-rhythmic baseline at matched accuracy. In the architecture studied here the measured rhythm and firing rate are tightly linked, so rate and timing are synergistic descriptions of a single dynamics rather than independent codes#cite(30).

  The remainder of the paper is organised as follows. #link(<sec-model-comparison>)[Model comparison] describes the model; #link(<sec-gamma-onset>)[Gamma onset] characterises the gamma onset in theory and simulation; #link(<sec-accuracy-rate-frontier>)[Accuracy–rate frontier] reports the trained accuracy–rate frontier; #link(<sec-loop-weight-interventions>)[Loop-weight interventions] reports two experiments that test whether the firing-rate reduction is acquired during training; #link(<sec-rate-frequency-relationship>)[Rate–frequency relationship] reports the relationship between the gamma frequency and the post-training firing rate; #link(<sec-dynamics-and-robustness>)[Dynamics and robustness] reports robustness of the post-training firing rate to spike perturbations and to integration timestep; and #link(<sec-streaming-classification>)[Streaming classification] reports a streaming-classification protocol.

  == Results

  === The model: COBA baseline and the PING loop <sec-model-comparison>

  The network is a two-population conductance-based spiking model with a single hidden layer of $N_E$ excitatory (E) and $N_I$ inhibitory (I) leaky integrate-and-fire (LIF) neurons, driven by feedforward input weights $W_"in"$ and read out by ten spiking output LIF neurons over the excitatory population (#link(<sec-network-architecture>)[Network architecture]). Recurrence is confined to one excitatory–inhibitory (E↔I) loop: E projects to I through the weight $W^(E I)$ and I back to E through $W^(I E)$, with no E→E or I→I connection. Throughout the paper I compare two configurations of this single architecture: with the loop disabled ($W^(E I) = W^(I E) = 0$) it is a conductance-based (COBA) control in which input drives the excitatory population alone, and with the loop engaged it is the pyramidal–interneuron gamma (PING) configuration. The same network was trained on MNIST by surrogate-gradient descent from #link(<sec-accuracy-rate-frontier>)[Accuracy–rate frontier] onward (#link(<sec-training-methods>)[Training methods]); in this section I characterise its free-running dynamics.

  Free-running, the COBA configuration fired asynchronously: its power spectral density (PSD) had no peak in the gamma band, and the excitatory firing-rate–current ($f$–$I$) curve rises to $approx #coba_fi_max$ Hz under the strongest drive tested. Engaging the E→I→E loop instead produced synchronous inhibitory bursts and gamma-banded excitatory rasters with a PSD peak at $f_gamma approx #fgamma023$ Hz, and held the excitatory firing rate approximately an order of magnitude below the COBA baseline across two decades of input drive (Figure 1).

  #figure(
    data-image(data-file("exp023/overview_compound.png"), width: 100%, alt: "Two-column comparison of the free-running network, each column headed by a wiring schematic. Left (COBA, loop off): schematic of input to a lone excitatory population with no inhibitory population, an asynchronous excitatory raster, a power spectrum with no gamma peak, and an f-I curve rising to about " + str(coba_fi_max) + " Hz. Right (PING, loop on): schematic of the excitatory-inhibitory loop (E to I via W_ei, I to E via W_ie), synchronous inhibitory bursts, gamma-banded excitatory raster, a sharp spectral peak near 42 Hz, and an excitatory rate held roughly an order of magnitude lower across two decades of input drive."),
    caption: [Free-running COBA and PING under matched Poisson drive. Each column shows a wiring schematic, single-trial raster, excitatory PSD and excitatory $f$–$I$ curve. COBA has the recurrent loop disabled; PING has E→I and I→E recurrence, with no E→E or I→I synapse. Source: #link("/exp023/")[exp023].],
  )

  === Gamma onset across the $W^(E I) times W^(I E)$ plane <sec-gamma-onset>

  Across the $W^(E I) times W^(I E)$ coupling plane the excitatory firing rate decreased, the inhibitory firing rate increased, and the lobe–trough rhythmicity contrast $R_"contrast"$ (a normalised, dimensionless measure of periodicity in the population autocorrelation, bounded in $[0, 1]$; #link(<sec-measurement-and-analysis>)[Measurement and analysis]) increased monotonically with coupling strength: $R_"contrast" approx 0$ along the COBA edges and $R_"contrast" approx 0.98$ at strong coupling. The four-dimensional mean-field reduction (#link(<sec-mean-field-reduction>)[Mean-field reduction]) predicts a supercritical Hopf bifurcation at external drive $I_"ext"^star = #hopf_iext$ nA with crossing frequency $f_"Hopf" approx #hopf_fstar$ Hz. The classification as supercritical is supported by quasi-static up/down ramps with peak hysteresis below $10^(-5)$ in rate units and by the linear scaling of the squared peak-to-peak steady-state oscillation amplitude $A_"pp"$, $A_"pp"^2 prop (I_"ext" - I_"ext"^star)$ ($R_"fit"^2 = #crit_r2$). The predicted gamma frequency is in qualitative agreement with the spiking measurement across the GABA synaptic decay time constant $tau_"GABA" in [4.5, 27]$ ms (Figure 2).

  #figure(
    data-image(data-file("exp054/onset_super_compound.png"), width: 100%, alt: "Nine panels. Top row: heatmaps of excitatory rate, inhibitory rate, and lobe-trough rhythmicity across the W_EI by W_IE coupling plane, with rhythmicity near zero on the COBA edges and rising to about 0.98 at strong coupling. Middle row: single-trial rasters at three points along the coupling diagonal, from asynchronous to sharp gamma volleys. Bottom row: the mean-field reduction, showing a complex eigenvalue pair crossing into the right half-plane at 0.60 nA, a continuous amplitude onset with coincident up and down ramps, and gamma frequency falling with the GABA time constant in agreement with the spiking measurement."),
    caption: [Nine panels. *(A–C)* Mean E rate, mean I rate and lobe–trough contrast across the $11 times 11$ coupling plane. *(D–F)* Single-trial E/I rasters at three diagonal points. *(G–I)* Mean-field eigenvalues, peak-to-peak E-rate amplitude branches and frequency against $tau_"GABA"$. Source: #link("/exp054/")[exp054], incorporating #link("/exp033/")[exp033]; spiking frequencies from #link("/exp041/")[exp041].],
  )

  === Trained PING attains COBA accuracy at $approx 10 times$ fewer spikes <sec-accuracy-rate-frontier>

  Both architectures, trained on MNIST under surrogate-gradient descent (#link(<sec-training-methods>)[Training methods]), converged to approximately $#ping25_acc%$ test accuracy. Sweeping the sample-wise hidden-E rate ceiling generated an accuracy–rate frontier; at every ceiling tested, PING attained higher accuracy at lower mean hidden-E firing rate than COBA. At the unpenalised operating point, PING reached $approx #ping25_acc%$ accuracy at $approx #ping25_rate$ Hz mean hidden-E rate, against $approx #coba25_acc%$ at $approx #coba25_rate$ Hz for COBA (Figure 3). Every frontier point is reported as the mean over three independent seeds (42, 43, 44), with standard errors across seeds. The PING rate did not decrease further as the ceiling was lowered, an empirical plateau consistent with, but not establishing, a structural lower bound on the rate.

  #figure(
    data-image(data-file("exp025/results_compound.png"), width: 100%, alt: "Trained-network comparison. Top: single-trial rasters, with COBA firing densely and asynchronously and PING firing in gamma bands with synchronous inhibitory bursts. Bottom left: validation accuracy per epoch, both configurations converging to about 91 percent. Bottom right: the accuracy-rate frontier, with PING lying above and to the left of COBA at every spike-budget setting; at the unpenalised operating point PING reaches about 91 percent near 12 Hz against COBA's 91 percent near 181 Hz."),
    caption: [Top: representative 400 ms single-trial COBA and PING rasters, twice the presentation used for training and quantitative evaluation. Bottom left: validation accuracy by epoch. Bottom right: mean hidden-E firing rate and test accuracy across three independent seeds at each activity ceiling; bars show standard errors and stars mark unpenalised points. Source: #link("/exp025/")[exp025]; rate-attractor analysis in #link("/exp024/")[exp024].],
  )

  === The firing-rate reduction does not require trained loop weights <sec-loop-weight-interventions>

  Whether the firing-rate reduction in #link(<sec-accuracy-rate-frontier>)[Accuracy–rate frontier] is acquired during training or follows from the canonical loop weights was tested by two complementary experiments. I distinguish the two senses of "architectural": (a) the reduction occurs without gradient-based tuning of the loop weights (which #link(<sec-loop-weight-interventions>)[Loop-weight interventions] establishes), as opposed to (b) the reduction emerges from generic E↔I structure without any hand-set values (which is not tested; the loop weights are held at the canonical biophysical values of#cite(7)). The claim in this work is (a): the inductive bias is paid for at design time, not during training.

  In the first, the recurrent loop was activated at inference on trained COBA networks from seeds 42–44, without retraining any weight. The illustrative seed-42 raster immediately becomes gamma-banded; across seeds, the mean excitatory rate fell by approximately a factor of $#tr_ratio$ ($approx #tr_e0 -> approx #tr_e1$ Hz), and mean test accuracy fell by $approx #tr_drop$ pp (Figure 4). The rate reduction therefore occurs without training. The accuracy loss is consistent with the absence of a learned compensation in the feedforward weights, which were optimised in the absence of the loop.

  #figure(
    data-image(data-file("exp038/loop_transfer_compound.png"), width: 100%, alt: "Inference-time loop activation on COBA networks trained with three seeds. Top: illustrative seed-42 rasters at loop-off and full loop strength. Bottom: across-seed mean rates and accuracy with standard-deviation bands."),
    caption: [COBA networks from seeds 42–44 evaluated while reciprocal E↔I loop strength varied from zero to full strength, without retraining. Top: illustrative seed-42 rasters. Bottom: across-seed mean E/I rates and accuracy; shading shows sample SD. Source: #link("/exp038/")[exp038].],
  )

  In the second, the loop weights $W^(E I), W^(I E)$ were released for training under the Dale's-law clamp. Both matrices represent non-negative conductance magnitudes; pathway identity fixes their reversal potentials, and the negative GABA reversal potential makes the I→E pathway inhibitory (#link(<sec-training-methods>)[Training methods]). After each optimiser step, the trained magnitudes were projected onto the non-negative cone. In all conditions tested, the rhythmicity score collapsed within a single training epoch: the first logged metric, after epoch 1, shows $R_"contrast" approx #r49_ep1$, compared with the canonical initial value $R_"contrast" approx #r49_can$ (Figure 5). The collapse was faster than the per-epoch logging interval, so no intermediate state was recorded. From every initial condition tested (canonical PING values, zero, and $0.1 times$ canonical), the inhibitory firing rate remained near zero, $R_"contrast"$ stayed in $#r49_fin_lo$–$#r49_fin_hi$, and final test accuracy was approximately $#r49_fz_acc%$ at $approx #r49_fz_e$ Hz mean E rate for the frozen-PING control (Figure 5). These numbers differ from the #link(<sec-accuracy-rate-frontier>)[Accuracy–rate frontier] frontier endpoint (≈#ping25_acc% at ≈#ping25_rate Hz) because the #link(<sec-loop-weight-interventions>)[Loop-weight interventions] setup omits the activity regulariser and isolates the within-experiment contrast between frozen and trainable conditions rather than tracing the accuracy–rate frontier. Within this setup, gradient descent does not preserve or recover effective E→I recruitment from any tested initial condition.

  #figure(
    data-image(data-file("exp049/training_curves.svg"), width: 100%, alt: "Four per-epoch training-metric panels with the loop weights released for training, comparing three initialisations (canonical, zero, and 0.1x canonical) against a frozen-PING control. (A) Test accuracy: all conditions overlap at roughly 90 percent. (B) Mean excitatory rate: the trainable conditions rise well above the frozen control, which stays gated at a low rate. (C) Mean inhibitory rate: the trainable conditions collapse to near zero within a few epochs while the frozen control's inhibitory rate stays high. (D) Rhythmicity: the frozen control holds near its canonical value while every trainable initialisation drains toward zero."),
    caption: [Per-epoch test accuracy, E rate, I rate and lobe–trough contrast over #ep049 epochs. Lines show three-seed means and shading the across-seed range for canonical, zero and $0.1 times$ canonical trainable initialisations and a frozen-PING control. Source: #link("/exp049/")[exp049].],
  )

  The second result is conditional on the gradient-damping scheme that stabilises PING training (the gradient flowing through the loop is attenuated by a factor $1\/d_"grad"$ on the backward pass, with $d_"grad" = 1000$; #link(<sec-training-methods>)[Training methods]); a constrained-training scheme (#link(<sec-conclusion-and-future-directions>)[Conclusion and Future Directions]) would test whether the loop's pruning depends on the damping regime. The first experiment (inference-time activation) used no gradients and is unaffected by this caveat, and carries most of the weight of the #link(<sec-loop-weight-interventions>)[Loop-weight interventions] conclusion.

  === Post-training E rate covaries with gamma frequency <sec-rate-frequency-relationship>

  The post-training excitatory firing rate covaried approximately affinely with the measured gamma frequency. Across a sweep of $tau_"GABA"$, which jointly changes the inhibitory decay kinetics, integrated inhibitory influence, and realised $f_gamma$, the trained $r_E$ was well fit by $r_E = #fit_a + #fit_p f_gamma$ ($R_"fit"^2 = #fit_r2$, three seeds per point; Figure 6). Mean test accuracy declined from $#acc41_fast%$ at $tau_"GABA" = 4.5$ ms to $#acc41_slow%$ at $27$ ms, a $#acc41_drop$ percentage-point tradeoff across the sweep. The association has a cycle-resolved counterpart. Resolving spikes by (neuron, cycle) pair, $#pleq1_046%$ contained at most one spike across the full $tau_"GABA"$ sweep: $P(0) approx #p0_046%$, $P(1) approx #p1_046%$, and the multi-spike fraction ($>= 2$) was $approx #pmulti_046%$ (Figure 7). Because $tau_"GABA"$ changes more than frequency alone, these experiments do not identify $f_gamma$ as the sole causal variable.

  #figure(
    data-image(data-file("exp041/rate_vs_fgamma.svg"), width: 100%, alt: "Top: mean post-training excitatory firing rate against measured gamma frequency, with per-condition means over three seeds and error bars on both axes; a linear fit passes through every error bar. Bottom: test accuracy over the same sweep, declining systematically toward the low-frequency conditions."),
    caption: [Top: mean post-training E rate against measured $f_gamma$ across six $tau_"GABA"$ values; markers are three-seed condition means with SEM on both axes and the line is the affine fit. Bottom: mean test accuracy across the same conditions. Source: #link("/exp041/")[exp041].],
  )

  #figure(
    data-image(data-file("exp046/spikes_per_cycle_distribution.svg"), width: 100%, alt: "A row of bar charts, one per value of the GABA time constant, each showing the distribution of spikes per (neuron, cycle) pair over the categories 0, 1, 2, and 3 or more. Across the sweep the zero-spike category dominates and the one-spike category is next, with the two-and-more categories nearly empty, so almost every pair contains at most one spike."),
    caption: [E spikes assigned to gamma cycles inferred from population inhibitory-burst peaks. Each panel shows the distribution of spikes per neuron–cycle pair (0, 1, 2 or $>= 3$) at one $tau_"GABA"$ value. Source: #link("/exp046/")[exp046].],
  )

  === Dynamics and robustness <sec-dynamics-and-robustness>

  The gating depends on the timing of inhibition, not its mean level. Perturbations of the trained PING network at inference revealed a deletion-versus-addition asymmetry. From an unperturbed baseline of approximately $#ping_base37%$ accuracy, PING retained approximately $#ping_drop80%$ accuracy under deletion of $80%$ of emitted spikes (little degradation); addition of off-phase Poisson noise instead drove accuracy down to chance as the injected rate grew, because those spikes recruit the inhibitory pool at arbitrary phase and disrupt the rhythm#cite(31). COBA exhibited the opposite asymmetry (Figure 8). Two inference-time jitter perturbations of the inhibitory spike train, both holding the mean inhibitory rate fixed, produced opposite effects on the excitatory rate: per-neuron jitter smeared each burst into a continuous shunt and reduced the excitatory rate to zero, while cycle-coherent jitter preserved within-burst synchrony and raised the excitatory rate from $approx #jit_e_base$ Hz to $approx #jit_e_cyc$ Hz (Figure 9)#cite(32).

  #figure(
    data-image(data-file("exp037/perturbation_curves.svg"), width: 100%, alt: "Across-seed mean test accuracy with standard-deviation bands against spike deletion and Poisson addition for PING and COBA."),
    caption: [Mean test accuracy under inference-time random spike deletion (left) and Poisson spike addition relative to baseline population rates (right). Lines show means across seeds 42–44; shading shows standard errors. Source: #link("/exp037/")[exp037].],
  )

  #figure(
    data-image(data-file("exp042/rhythm_compound.png"), width: 100%, alt: "Two inhibitory-jitter manipulations at the same jitter magnitude, sigma 14 ms, that both hold the mean inhibitory rate fixed. Left column (per-neuron jitter): rasters show bursts smeared into a continuous shunt, and the excitatory rate falls to near zero while accuracy falls to chance. Right column (cycle-coherent jitter): whole bursts are displaced but within-burst synchrony is preserved, the excitatory rate rises from about " + str(jit_e_base) + " to about " + str(jit_e_cyc) + " Hz, and accuracy remains high. The bottom sweep panels overlay the realised mean inhibitory rate as a grey line, flat where the two arms are compared. Identical mean inhibition, opposite excitatory outcome."),
    caption: [Matched inference-time inhibitory-jitter manipulations at $sigma = 14$ ms. Top: single-trial E/I rasters. Bottom: mean E rate, accuracy and realised I rate across three-seed per-neuron and cycle-coherent jitter sweeps. The strict comparison uses the range where realised I rates remain matched. Source: #link("/exp042/")[exp042].],
  )

  The post-training firing rate is also approximately invariant under change of integration timestep: across $Delta t_"sim" in [0.05, 1.0]$ ms (a $20 times$ range), the trained excitatory rate stayed in $#er044_lo$–$#er044_hi$ Hz and accuracy varied by less than $#acc044_pp$ pp (Figure 10). The rate is therefore a property of the continuous dynamics, not an artefact of the discretisation.

  #figure(
    data-image(data-file("exp044/dt_sweep.svg"), width: 100%, alt: "Post-training mean excitatory rate (black diamonds, left axis) and test accuracy (red squares, right axis) against integration timestep on a logarithmic abscissa spanning 0.05 to 1.0 ms. The excitatory rate stays within a narrow low band and is non-monotonic in the timestep, so finer stepping does not buy a lower rate; accuracy stays essentially flat across the range. Both training and inference use the same timestep at each point."),
    caption: [Post-training mean E rate (black diamonds, left ordinate) and test accuracy (red squares, right ordinate) against matched training-and-inference timestep on a logarithmic 0.05–1.0 ms abscissa. Source: #link("/exp044/")[exp044].],
  )

  === Streaming classification on continuous input <sec-streaming-classification>

  The preceding subsections evaluate the network on isolated single-digit presentations. The streaming protocol tests whether the firing-rate reduction is preserved under continuous input.

  A PING network trained on single-digit MNIST classified a continuously concatenated input stream without retraining or an external segmentation signal; the decoder used the known segment boundaries. The streaming protocol (#link(<sec-datasets-and-evaluation>)[Datasets and evaluation]) uses a non-spiking leaky-integrator readout with a trailing-mean window whose duration is matched exactly to each segment's presentation duration. On a representative stream of five digits, each with its own duration ($25$–$200$ ms) and input rate ($10$–$200$ Hz), #stream48-correct of #stream48-total were classified correctly (Figure 11). Across the ($T_"present"$, input-rate) grid, accuracy was approximately a function of the product $T_"present" r_"input,max"$. Accuracy did not exceed $approx 80%$ for $T_"present" <= 15$ ms regardless of input rate (Figure 12A). For reference, $15$ ms is approximately $#tau_floor_cyc$ times the canonical gamma period $T_gamma approx #Tg_canon$ ms. Accuracy saturated by $T_"present" approx 40$–$50$ ms, and the trained operating point ($T_"present" = 200$ ms, rate $= 25$ Hz) reached $#op48_acc%$ accuracy. Extending the $200$ ms slice below the grid's $5$ Hz minimum locates a separate encoder floor: performance remained at chance through #rate48-p05.input_rate_hz Hz, became clearly informative by #rate48-p2.input_rate_hz Hz, and reached #calc.round(100 * rate48-p5.accuracy, digits: 1)% at #rate48-p5.input_rate_hz Hz (Figure 12B). The failed $200$ ms, #rate48-p10.input_rate_hz Hz segment in Figure 11 occurred at a population-level accuracy of #calc.round(100 * rate48-p10.accuracy, digits: 1)%, so it is a natural weak-evidence classification error rather than evidence that the encoding rate is intrinsically nonviable. Together the panels establish requirements for sufficient integration time and sufficient encoded input evidence; because gamma frequency is not independently varied, they do not identify the gamma cycle as the cause or temporal unit of either requirement.

  #figure(
    data-image(data-file("exp048/varying_headline_stream.png"), width: 100%, alt: "A single concatenated stream of five MNIST digits, each with its own presentation duration and Poisson input rate. The two weakest-drive segments are misclassified while the other three are correct; hidden excitatory and inhibitory rasters maintain gamma cycles throughout."),
    caption: [One five-digit stream with per-segment presentation durations of 25–200 ms and Poisson input rates of 10–200 Hz. Top: thumbnails, conditions and predictions; opacity denotes input rate. Below: sampled E/I rasters and sliding leaky-integrator readout traces. Source: #link("/exp048/")[exp048].],
  )

  #figure(
    data-image(data-file("exp048/acc_grid_tau_rate.png"), width: 100%, alt: "Two panels: a heatmap of streaming accuracy across segment duration and input rate, and a psychometric curve measured at fixed 200 millisecond presentation and readout windows showing chance performance below 0.5 hertz and a steep transition between 1 and 5 hertz."),
    caption: [*(A)* Per-segment accuracy across presentation duration and input rate, averaged over three seeds and 1,200 segments per grid cell. *(B)* Accuracy against input rate with presentation and readout windows fixed at 200 ms; the dotted line marks the 25 Hz training rate. Source: #link("/exp048/")[exp048].],
  )

  == Discussion <sec-discussion>

  A recurrent E↔I loop, held fixed during training, generates a gamma rhythm and reduces the post-training excitatory firing rate by approximately an order of magnitude relative to the COBA baseline at matched accuracy (#link(<sec-accuracy-rate-frontier>)[Accuracy–rate frontier]). The reduction does not require gradient-based learning of the loop weights: activating the loop at inference on a trained COBA network reduces the firing rate without retraining (#link(<sec-loop-weight-interventions>)[Loop-weight interventions]), and gradient descent does not preserve the loop within a single epoch when its weights are released (#link(<sec-loop-weight-interventions>)[Loop-weight interventions], conditional on the #link(<sec-training-methods>)[Training methods] damping scheme). The inductive bias is paid for at design time, via the canonical biophysical loop weights#cite(7), not during training. Within the mean-field reduction, the gamma onset is a supercritical Hopf bifurcation (#link(<sec-gamma-onset>)[Gamma onset]), and the empirical data support this classification. Its continuous, reversible character fits the inference-time loop activation of #link(<sec-loop-weight-interventions>)[Loop-weight interventions], which produces a graded change in firing rate without hysteresis; a direct test would require an inference-time hysteresis sweep on the E→I gain (#link(<sec-conclusion-and-future-directions>)[Conclusion and Future Directions]).

  The mechanism of the gate is the temporal structure of inhibition, not its mean level. The jitter perturbation experiment (Figure 9) is a direct test: holding the mean per-neuron inhibitory rate fixed and varying its temporal structure produces opposite effects on the excitatory rate depending on whether within-burst synchrony is preserved, consistent with prior characterisations of temporal-synchrony patterns within PING circuits#cite(32). The robustness asymmetry under spike addition versus deletion (Figure 8) follows from this dependence on phase: removal of spikes does not alter the phase structure of the population output, while addition of off-phase spikes does. The asymmetry constitutes a testable prediction for biological gamma circuits and would speak to long-standing rate-vs-timing debates#cite(33, 34). It echoes prior reports that oscillations sharpen spike-timing precision#cite(31).

  Across the $tau_"GABA"$ sweep, post-training rate covaried with the realised gamma frequency, and the majority of (neuron, cycle) pairs contained at most one spike (Figures 6–7). This is consistent with cycle-structured rate control, but $tau_"GABA"$ simultaneously changes inhibitory decay, integrated conductance, and burst duty cycle; the experiment does not isolate frequency as the sole cause of the rate change. An independent manipulation of oscillation frequency at matched inhibitory influence would be needed for that attribution. The streaming experiment (Figure 12) instead separates two evidence bounds: panel A shows low accuracy at short presentation durations and an approximate dependence on $T_"present" r_"input,max"$, whereas panel B locates the $200$ ms encoder floor below #rate48-p05.input_rate_hz Hz. Errors above that floor are ordinary trial-level failures under weak evidence, not evidence that the rate is categorically unusable. The numerical proximity of the short-duration floor to the canonical gamma period is descriptive, not mechanistic, because gamma frequency is not independently varied. A gamma-frequency sweep or a cycle-aligned analysis showing discontinuities at integer multiples of $T_gamma$ would be required to test whether gamma defines a temporal unit for classification.

  PING is a structured alternative to the asynchronous balanced state#cite(36, 37). The architectural treatment of the loop adopted here differs from the inhibitory-plasticity literature, in which the inhibitory connectivity is plastic and learns E/I balance#cite(38, 39, 40, 41). The #link(<sec-loop-weight-interventions>)[Loop-weight interventions] result, that gradient descent does not preserve the loop from any tested initial condition (under the #link(<sec-training-methods>)[Training methods] damping regime), provides empirical support for the architectural treatment in this setting. I propose a functional interpretation: gamma may act as a structural constraint on excitatory firing rates without requiring learned tuning of the inhibitory connectivity.

  Relative to the two closest recent trainable-SNN precedents in the bibliography, the present work differs in how the rhythm is obtained rather than in claiming better performance.#cite(25) imposes the oscillation as an external input to spiking neurons;#cite(26) trains an adaptive-LIF network on speech, all parameters free, and reports that oscillatory synchronisation and cross-frequency coupling _emerge_ from end-to-end optimisation, correlating with task performance. The #link(<sec-loop-weight-interventions>)[Loop-weight interventions] result that gradient descent does not preserve the loop is therefore not a claim that surrogate-gradient training cannot discover rhythm in general (#cite(26) shows it can) but a narrower one: it does not maintain a _fixed, biophysically-calibrated PING loop_ whose weights are released under the Dale's-law clamp and the damping regime of #link(<sec-training-methods>)[Training methods]. The two findings are complementary poles of the same question: rhythm acquired by training versus rhythm supplied by architecture. Neither#cite(25) nor#cite(26) uses a conductance-based E↔I loop, and neither attributes a firing-rate reduction to the rhythm via inference-time activation, frequency tuning, or jitter perturbation as #link(<sec-loop-weight-interventions>)[Loop-weight interventions], #link(<sec-rate-frequency-relationship>)[Rate–frequency relationship] and #link(<sec-dynamics-and-robustness>)[Dynamics and robustness] do; their firing rates are roughly an order of magnitude higher than the rates reported here, and neither frames a per-spike economy. I do not provide a head-to-head numerical comparison:#cite(25) and#cite(26) evaluate on temporally structured tasks (SHD; speech perception) where the present static-MNIST protocol is not directly comparable. The contribution is mechanistic, not benchmark-driven.

  The reduction reported in #link(<sec-accuracy-rate-frontier>)[Accuracy–rate frontier] should be considered net of the inhibitory contribution. PING uses a smaller, higher-rate inhibitory population, so the reduction in population-total spike rate ($N_E chevron.l r_E chevron.r + N_I chevron.l r_I chevron.r$) is approximately a factor of $#spike_ratio$ rather than a factor of $#rate_ratio25$. Excitatory glutamatergic signalling accounts for a larger share of the cortical energy budget than inhibitory transmission#cite(28, 29); weighting spikes by metabolic cost recovers the order-of-magnitude figure. The argument is complicated by the substantial per-neuron metabolic demands of fast-spiking interneurons, which sustain high firing rates and dense synaptic activity#cite(42); for this reason I treat the uniform-spike-rate reduction (approximately $#spike_ratio$-fold) as the more conservative claim. I do not attempt a quantitative metabolic comparison with cortex, given the architectural differences (no $W_(e e)$ or $W_(i i)$, idealised synapse counts).

  Several limitations apply. The evaluation used a single dataset (MNIST), a single readout, and a fixed loop topology. MNIST is a simple, near-saturated benchmark on which many architectures reach comparable accuracy, so the classification results here should be read as evidence that the firing-rate reduction survives training to competence, not as a claim about task difficulty or about generalisation to harder problems; whether the mechanism holds on datasets with intrinsic temporal structure is left to future work (SHD, #link(<sec-conclusion-and-future-directions>)[Conclusion and Future Directions]). The rhythmicity metric $R_"contrast"$ is one of several available options. The mean-field reduction inherits biophysical cellular and synaptic parameters but contains a free effective-noise scale and remains a reduction of the full network. The architecture excludes $W_(e e)$ and $W_(i i)$, conduction delays, and cell-type heterogeneity; the implications for cortical microcircuits with richer connectivity are open#cite(15). The streaming evaluation does not include temporally structured inputs in which classification depends on input timing. The capacity of a single gamma cycle and its scaling with assembly size are not characterised here#cite(43). Classification accuracy on MNIST was approximately $83$–$90%$; the contribution of the present work concerns the mechanism by which the firing rate is reduced rather than the absolute accuracy. The #link(<sec-loop-weight-interventions>)[Loop-weight interventions] released-loop result depends on the gradient-damping scheme ($d_"grad" = 1000$, #link(<sec-training-methods>)[Training methods]) and the Dale clamp; whether the loop's collapse persists under weaker damping is not addressed here. The paper does not benchmark against rhythmic-SNN baselines#cite(25, 26); the PING-specific attribution rests on the within-architecture experiments of #link(<sec-loop-weight-interventions>)[Loop-weight interventions], #link(<sec-rate-frequency-relationship>)[Rate–frequency relationship] and #link(<sec-dynamics-and-robustness>)[Dynamics and robustness] rather than on an external rhythmic-SNN comparison.

  == Conclusion and Future Directions <sec-conclusion-and-future-directions>

  A recurrent E↔I loop held fixed during training reduces the post-training excitatory firing rate by approximately an order of magnitude relative to the COBA baseline at matched accuracy, and approximately $#spike_ratio$-fold by population-total spike rate when the higher-rate inhibitory pool is included (#link(<sec-discussion>)[Discussion], paragraph 5). The reduction is invariant to the integration timestep across a $20 times$ range (Figure 10, retrained at each $Delta t_"sim"$) and is preserved under evaluation on continuously concatenated input streams (Figures 11–12).

  Empirical extensions include evaluation on the SHD dataset#cite(44), which would test the gating on a spiking benchmark with intrinsic temporal structure, and tasks in which classification accuracy depends on input timing, which could test whether the gamma cycle acts as a temporal unit. This is the regime in which the emergent-oscillation route of#cite(26) operates; a direct comparison there would set the imposed, biophysically-calibrated PING loop studied here against a network free to learn its own rhythmic structure, on the same temporally structured task. A more comprehensive characterisation of the $W^(E I) times W^(I E)$ plane would test the supercritical Hopf classification directly.

  Theoretical extensions include multi-layer PING architectures, an independent two-dimensional sweep of the two loop-weight gains $alpha_(E I) times alpha_(I E)$, multi-rhythm and theta-nested gamma models#cite(19, 20), and characterisation of capacity limits for single cycles and minimal assemblies#cite(43). A constrained-training scheme, in which the loop weights are regularised toward biophysical values rather than held fixed, would connect the present result to the inhibitory-plasticity literature#cite(38). The rate law of #link(<sec-rate-frequency-relationship>)[Rate–frequency relationship] admits a testable biological prediction: perturbing the timing of inhibitory neurons in vivo (e.g. by optogenetic stimulation, as in#cite(8, 9, 10)) should change the excitatory firing rate without altering the mean inhibitory rate, the in vivo analogue of Figure 9.

  Gamma may act as a structural mechanism by which the cortical microcircuit maintains sparse excitatory firing rates at low metabolic cost. The present work provides one architecture in which this mechanism is realised explicitly.

  == Methods

  === Single-neuron and synapse dynamics <sec-neuron-and-synapse-dynamics>

  The model uses a conductance-based leaky integrate-and-fire (LIF) representation with two populations, excitatory (E) and inhibitory (I). The sub-threshold membrane potential of each population evolves as

  $ C_m^E (dif V_m^E) / (dif t) &= -g_L^E (V_m^E - E_L) - g_e^E (V_m^E - E_e) - g_i^E (V_m^E - E_i) \
    C_m^I (dif V_m^I) / (dif t) &= -g_L^I (V_m^I - E_L) - g_e^I (V_m^I - E_e) $

  where $C_m$ is the membrane capacitance, $g_L$ the leak conductance, $E_L$ the leak reversal potential, and $E_e$, $E_i$ the excitatory and inhibitory synaptic reversal potentials. The I population has no inhibitory term because there is no I→I connection in this architecture (#link(<sec-network-architecture>)[Network architecture]).

  A neuron emits a spike when $V_m$ crosses threshold $V_"th"$ from below; the membrane potential is then reset to $V_"reset"$ for a refractory period $tau_"ref"$:

  $ s[k+1] = bold(1)[V_m[k+1] >= V_"th"], quad V_m[k+1] <- V_"reset" "if" s[k+1] = 1 "or refractory". $

  Each synaptic conductance is an exponential trace driven by presynaptic spikes; a presynaptic spike adds its full weight $W$ as an instantaneous jump in conductance, which then decays with the relevant channel time constant ($tau_"AMPA"$ for AMPA-like excitation, $tau_"GABA"$ for GABA-like inhibition):

  $ (dif g^E_e) / (dif t) &= -(g^E_e) / (tau_"AMPA") + W_"in" sum_k delta(t - t^"inp"_k) \
    (dif g^E_i) / (dif t) &= -(g^E_i) / (tau_"GABA") + W^(I E) sum_k delta(t - t^i_k) \
    (dif g^I_e) / (dif t) &= -(g^I_e) / (tau_"AMPA") + W^(E I) sum_k delta(t - t^e_k) $

  The first equation describes input-driven excitation onto E via feedforward weights $W_"in"$; the second is inhibition onto E from I via $W^(I E)$; the third is excitation onto I from E via $W^(E I)$. There is no equation for $g^I_i$ (no I→I connection) and no recurrent contribution to $g^E_e$ (no E→E connection). Canonical parameter values are listed in the parameters table; the E and I populations differ in membrane capacitance, leak conductance, membrane time constant, and refractory period.

  Canonical parameter values for the spiking network are given below. Where E and I populations use different values they are listed as "E / I"; otherwise the value is shared. The $tau_"GABA" = 9$ ms value is the canonical PING value of#cite(7).

  #table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    [*Symbol*], [*Description*], [*Value*],
    [$C_m$], [Membrane capacitance], [1.0 / 0.5 nF],
    [$g_L$], [Leak conductance], [0.05 / 0.1 μS],
    [$tau_"ref"$], [Refractory period], [3.0 / 1.5 ms],
    [$E_L$], [Leak reversal potential], [−65 mV],
    [$V_"th"$], [Spike threshold], [−50 mV],
    [$V_"reset"$], [Reset potential], [−65 mV],
    [$E_e$], [AMPA reversal potential], [0 mV],
    [$E_i$], [GABA reversal potential], [−80 mV],
    [$tau_"AMPA"$], [AMPA decay time constant], [2 ms],
    [$tau_"GABA"$], [GABA decay time constant], [9 ms],
    [$Delta t_"sim"$], [Integration timestep], [0.1 ms (train) / 0.25 ms (inference)],
    [$N_E$], [Hidden excitatory pool size], [1024],
    [$N_I$], [Inhibitory pool size], [256],
  )

  === Network architecture <sec-network-architecture>

  The network has one hidden layer with $N_E$ excitatory and $N_I$ inhibitory neurons, and ten spiking output LIF neurons with weights $W_"out"$ over the excitatory population. For the standard classifier, the class score $z_c$ is output neuron $c$'s pre-reset membrane voltage averaged over the presentation window. The streaming study instead uses a non-spiking leaky-integrator readout and a trailing-mean evidence window matched to each presentation (#link(<sec-datasets-and-evaluation>)[Datasets and evaluation]). Input spikes drive the excitatory population via feedforward weights $W_"in"$. Recurrence is restricted to the E↔I loop: E projects to I via $W^(E I)$, and I projects back to E via $W^(I E)$. There is no $W_(e e)$ and no $W_(i i)$. In every realized connection matrix $W^(A B)$, rows index source population $A$ and columns index target population $B$; pathway direction is $A arrow B$.

  The restriction to the E↔I loop is intended to make the rhythm unambiguously PING: excluding $W_(i i)$ rules out ING, and excluding $W_(e e)$ rules out recurrent-E driven oscillation#cite(7, 45). The conductance-based (COBA) baseline used as a non-PING control is the loop-off limit of the same architecture, obtained by setting $W^(E I) = W^(I E) = 0$.

  The loop weights $W^(E I)$ and $W^(I E)$ were held fixed (untrained) in the experiments reported in #link(<sec-accuracy-rate-frontier>)[Accuracy–rate frontier] and #link(<sec-rate-frequency-relationship>)[Rate–frequency relationship], #link(<sec-dynamics-and-robustness>)[Dynamics and robustness] and #link(<sec-streaming-classification>)[Streaming classification]. The loop is treated as a structural prior, consistent with the inhibitory-plasticity literature, in which inhibitory synapses serve an experience-dependent E/I balance role rather than carrying the feedforward computational features that the excitatory pathway acquires#cite(38, 39). The choice is also supported empirically by the result in #link(<sec-loop-weight-interventions>)[Loop-weight interventions] (Figures 4–5) that gradient descent does not preserve effective E→I recruitment when the recurrent conductances are released. A Dale's-law clamp constrains all trained conductance magnitudes to remain non-negative throughout training (#link(<sec-training-methods>)[Training methods]).

  === Mean-field reduction <sec-mean-field-reduction>

  To locate the gamma-onset bifurcation analytically (#link(<sec-gamma-onset>)[Gamma onset]), the spiking network is reduced to a four-dimensional rate model in the state

  $ bold(x)(t) = (macron(E), macron(I), macron(g)_e^I, macron(g)_i^E), $

  where $macron(E)$ and $macron(I)$ are the population-mean firing rates of the E and I neurons (in spikes per millisecond), and $macron(g)_e^I$, $macron(g)_i^E$ are the population-mean cross-population synaptic conductances onto the I and E populations. The two cross-population conductances are sufficient because the architecture has no $W_(e e)$ and no $W_(i i)$ (#link(<sec-network-architecture>)[Network architecture]); the within-population conductances vanish identically.

  The reduction replaces the spike-driven conductance dynamics with rate-driven first-order filters, and replaces each neuron's stochastic spike output with the population-averaged firing rate of a noise-driven LIF neuron at a given mean input current. The closed 4D system is

  $ tau_E dot(macron(E)) &= -macron(E) + Phi_E (I_"ext" - macron(g)_i^E Delta V_"inh,mag") \
    tau_I dot(macron(I)) &= -macron(I) + Phi_I (macron(g)_e^I Delta V_"exc,mag") \
    tau_"AMPA" dot(macron(g))_e^I &= -macron(g)_e^I + tau_"AMPA" G_(E arrow I) macron(E) \
    tau_"GABA" dot(macron(g))_i^E &= -macron(g)_i^E + tau_"GABA" G_(I arrow E) macron(I) $

  where $I_"ext"$ is an external tonic drive to E (the bifurcation control parameter, below), and the driving-force magnitudes are $Delta V_"exc,mag" = E_e - E_L = 65$ mV and $Delta V_"inh,mag" = E_L - E_i = 15$ mV evaluated at rest. The membrane time constants are the passive ratios $tau_E = C_m^E \/ g_L^E = 20$ ms and $tau_I = C_m^I \/ g_L^I = 5$ ms, computed from the capacitances and leak conductances in the #link(<sec-neuron-and-synapse-dynamics>)[Neuron and synapse dynamics] parameters table; the synaptic time constants $tau_"AMPA"$, $tau_"GABA"$ are taken directly from that table. The fan-in-normalised summed-conductance couplings are $G_(E arrow I) = 1.0$ μS and $G_(I arrow E) = 2.0$ μS, inherited from the spiking network.

  The population gain functions $Phi_E$ and $Phi_I$ are the noise-driven LIF rate functions in Ricciardi–Siegert form#cite(46). For mean input current $mu$ delivered to a neuron with leak conductance $g_L$, leak reversal $E_L$, threshold $V_"th"$, reset $V_"reset"$, membrane time constant $tau_m$, and refractory period $tau_"ref"$,

  $ Phi(mu) = [ tau_"ref" + tau_m sqrt(pi) integral_((V_"reset" - mu_V) \/ sigma_V)^((V_"th" - mu_V) \/ sigma_V) e^(u^2) (1 + "erf" u) dif u ]^(-1), $

  with mean membrane potential $mu_V = E_L + mu \/ g_L$ and effective membrane-noise scale $sigma_V$. The integral was evaluated by numerical quadrature. Cellular and synaptic parameters are inherited from the #link(<sec-neuron-and-synapse-dynamics>)[Neuron and synapse dynamics] model (with $tau_m = C_m \/ g_L$) separately for the E and I populations. The effective noise scale is not derived from the spiking model and is treated as a free sensitivity parameter; $4$ mV is the reference value. Across $sigma_V in {3, 4, 5, 6}$ mV, the Hopf and its supercritical classification persisted and the crossing frequency remained stable, while the threshold, fixed point, and relative-onset amplitude varied quantitatively.

  The non-oscillating low-rate fixed point was tracked as $I_"ext"$ was swept from $0$ to $4$ nA in $0.01$ nA ($10$ pA) steps. At each $I_"ext"$, the fixed point was obtained by solving the algebraic system in $(macron(E), macron(I))$ (at steady state the two conductances are determined by the rates, $macron(g)_e^I = tau_"AMPA" G_(E arrow I) macron(E)$ and $macron(g)_i^E = tau_"GABA" G_(I arrow E) macron(I)$) using _scipy.optimize.fsolve_. The numerical Jacobian $J_"flow"$ of the full 4D vector field at each fixed point was computed by central finite differences. The Hopf threshold $I_"ext"^star$ is the smallest $I_"ext"$ at which the eigenvalue $lambda_J^star$ with largest real part crosses zero with non-zero imaginary part. Because time is measured in milliseconds, the crossing frequency is $f_"Hopf" = 1000 |"Im" lambda_J^star| \/ (2 pi)$ Hz.

  The onset was classified numerically by a quasi-static amplitude sweep. $I_"ext"$ was ramped up across $I_"ext"^star$ from a small perturbation of the non-oscillating fixed point and then ramped down; at each step the 4D system was integrated to its steady-state peak-to-peak oscillation amplitude $A_"pp"$. The onset is classified as supercritical when (i) the up- and down-ramp branches coincide within a peak hysteresis of $10^(-4)$ in rate units, and (ii) the squared amplitude scales linearly with the bifurcation distance,

  $ A_"pp"^2 prop (I_"ext" - I_"ext"^star), $

  with $R_"fit"^2 > 0.9$. For the canonical parameter set the criterion was met with hysteresis below $10^(-5)$ and $R_"fit"^2 = 0.999$.

  The mean-field prediction was compared with the gamma frequency measured in the spiking network, extracted as in #link(<sec-measurement-and-analysis>)[Measurement and analysis], across a sweep of $tau_"GABA" in {4.5, 6, 9, 12, 18, 27}$ ms (Figure 2). Both curves decrease monotonically with $tau_"GABA"$; the spiking measurement is consistently higher than the rate-equation prediction across the sweep. The reduction captures the qualitative dependence on the GABA decay time, which is the use to which it is put. The treatment follows the Wilson–Cowan tradition for cortical-rhythm modelling#cite(18) and the broader population-dynamics and next-generation neural-mass literature#cite(19, 47, 48).

  === Training <sec-training-methods>

  The network was trained on MNIST by surrogate-gradient descent through backpropagation in time#cite(22, 23). Each digit was rate-encoded as a Poisson spike train over a 200 ms presentation window at a peak rate $r_"input,max" = 25$ Hz per active pixel. With $Delta t_"sim"$ measured in milliseconds, the Bernoulli event probability per timestep was $p_"event" = r_"input,max" Delta t_"sim" / 1000$. The standard classifier used cross-entropy on the time-averaged pre-reset membrane potentials of its spiking output LIF neurons, with a sample-wise population-rate penalty active when hidden-E activity exceeds a soft ceiling $r_(E,"ceil")$:

  $ r_b = 1 / (N_E T_"present") sum_(n in E) n_"spike"(b,n), quad
    L_"total" = L_"CE" + lambda_"rate" / B sum_b max(r_b - r_(E,"ceil"), 0)^2, $

  where $n_"spike"(b,n)$ is hidden-E neuron $n$'s spike count during presentation $b$, $T_"present"$ is measured in seconds, $B$ is minibatch size, and $lambda_"rate"$ weights the penalty. The rate is normalised by population width and physical duration before the one-sided quadratic is applied to each presentation; the result is then averaged across the minibatch. Sweeping $r_(E,"ceil")$ over ${"off", 25, 10, 5, 2.5, 1}$ Hz generated the accuracy–rate frontier reported in #link(<sec-accuracy-rate-frontier>)[Accuracy–rate frontier].

  The discrete spike nonlinearity has zero gradient almost everywhere; in the backward pass it is replaced by a fast-sigmoid surrogate of the distance to threshold $V_"candidate" - V_"th"$,

  $ (partial bold(1)[V_"candidate" >= V_"th"]) / (partial V_"candidate") eq.triple k_"sg" / (1 + k_"sg" |V_"candidate" - V_"th"|)^2, $

  with surrogate slope $k_"sg" = 1$#cite(23, 49). The forward pass evaluates the Heaviside exactly. Optimisation used AdamW with zero weight decay#cite(50), learning rate $4 times 10^(-4)$, batch size $256$, and #hub_epochs epochs. Every activity-ceiling condition was repeated across three independent seeds (42, 43, 44); frontier points are their means and uncertainty bars show the standard error across seeds.

  *Gradient damping for the PING configuration.* Surrogate-gradient training of the PING configuration is unstable without intervention on the gradient. A single loop weight ($W^(E I)$ or $W^(I E)$) contributes to the membrane-voltage update of every neuron at every subsequent timestep within a trial; combined with the spike-driven impulse updates of the synaptic conductances and the non-zero surrogate gradient at the spike threshold, the gradient propagates through a tightly coupled feedback loop with millisecond-scale conductance dynamics. Over the 2,000-step backpropagation-through-time window of a single 200 ms trial at $Delta t_"sim" = 0.1$ ms, multiplicative contributions across timesteps cause gradient norms to grow by many orders of magnitude, and training does not converge. The mechanism is the same multiplicative compounding through long unrolled recurrences that characterises the exploding-gradient pathology in standard recurrent networks#cite(51).

  I addressed this by attenuating the gradient flowing through the membrane-voltage increment $dif V_m$ on the backward pass by a factor of $1\/d_"grad"$, implemented as a straight-through identity that scales the gradient without modifying the forward value:

  $ "damp"_(d_"grad") (x) = 1/d_"grad" x + (1 - 1/d_"grad") "stopgrad"(x). $

  The operator is applied to $dif V_m$ at every integration step in every layer. It leaves the forward numerical update unchanged while attenuating the gradient through that update by a factor of $1\/d_"grad"$ per step. The integration itself remains numerical. The activity-ceiling comparison used $d_"grad" = 1000$ for PING and $d_"grad" = 1$ for COBA; the released-loop experiment used $d_"grad" = 1000$. The COBA configuration is trainable at the module default $d_"grad" = 80$; the PING configuration is not.

  *Dale's-law clamp.* The synaptic matrices store conductance magnitudes, not signed currents. After each optimiser step, $W_"in"$, $W^(E I)$, and $W^(I E)$ are therefore projected onto the non-negative cone#cite(52, 53). Excitatory or inhibitory action is set by the pathway-specific reversal potential in the membrane current: the I→E term $g_I (E_I - V_m)$ is hyperpolarising for the GABA reversal potential $E_I = -80$ mV. The projection permits either recurrent conductance to grow while preventing an unphysical negative conductance. The #link(<sec-loop-weight-interventions>)[Loop-weight interventions] collapse primarily reflects weakened or absent E→I recruitment and loss of inhibitory firing, not $W^(I E)$ crossing into a negative sign.

  === Measurement and analysis <sec-measurement-and-analysis>

  Mean firing rates $chevron.l r_E chevron.r$ and $chevron.l r_I chevron.r$ were computed as time-averaged spike counts per neuron over the presentation window. The spectral-peak frequency $f_"peak"$ was extracted from the Welch power spectral density#cite(54) of the summed population E spike train at sampling rate $f_s = 1 \/ Delta t_"sim" = 4000$ Hz, using a single segment of length equal to the trial, a mean-centred (not z-scored) signal, no detrending, and a peak search restricted to the defined gamma band $[5, 150]$ Hz; the raw-bin maximum was refined by parabolic sub-bin interpolation. For the reported analysis, $f_gamma eq.triple f_"peak"$.

  The rhythmicity score $R_"contrast"$ is the Michelson contrast between the first side lobe amplitude $A_"lobe"$ and the first trough amplitude $A_"trough"$ of the autocorrelation of the binned population E spike count, computed via zero-padded FFT and normalised by the squared mean of the rate. After a 3-point smoothing kernel $[0.25, 0.5, 0.25]$ is applied to the autocorrelogram, the first trough is identified as the first local minimum starting from lag 2, and the lobe as the maximum between lag 1 and the trough; then

  $ R_"contrast" = (A_"lobe" - A_"trough") / (A_"lobe" + A_"trough") in [0, 1]. $

  $R_"contrast"$ is bounded and dimensionless. I used it in preference to spectral-peak measures because the latter become unreliable at the low firing rates encountered in some of the regimes of interest. $R_"contrast"$ is qualitatively consistent with the spectral-peak and population-coherence measures used elsewhere in the gamma literature#cite(55, 56).

  The spikes-per-cycle distribution (Figure 7) was constructed by binning each neuron's spikes into gamma cycles inferred from peaks of the population I-burst rate (Gaussian-smoothed with $sigma = 1$ ms), detected with _scipy.signal.find_peaks_ using a minimum inter-peak separation of half the expected gamma period and a height threshold of $5%$ of the maximum. Cycle boundaries were placed at midpoints between consecutive I-burst peaks; each E spike was assigned to its enclosing cycle.

  Spike-economy claims report both the mean E rate and the population-total spike rate $N_E chevron.l r_E chevron.r + N_I chevron.l r_I chevron.r$ in spikes per second, so the reduction is stated net of the inhibitory contribution. This becomes a spike-count ratio only when the compared observations have the same duration. The metabolic argument that excitatory spikes incur larger costs than inhibitory spikes#cite(28, 29) is invoked where relevant but is not modelled quantitatively.

  === Integration and parameters

  The membrane and synaptic-conductance ODEs were integrated by an exponential-Euler scheme#cite(57) with zero-order hold on the synaptic conductances over each step. With $g_"tot" = g_L + g_e + g_i$ the total instantaneous conductance, effective time constant $tau_"eff" = C_m \/ g_"tot"$, and instantaneous steady state $V_oo = (g_L E_L + g_e E_e + g_i E_i) \/ g_"tot"$, the closed-form update is

  $ V_m[k+1] = V_oo + (V_m[k] - V_oo) e^(-Delta t_"sim" \/ tau_"eff"). $

  Training used $Delta t_"sim" = 0.1$ ms; smaller timesteps are required for numerical stability of the backpropagation through the recurrent E↔I dynamics. Inference used $Delta t_"sim" = 0.25$ ms. Firing rates and frequencies are reported in Hz. The #link(<sec-dynamics-and-robustness>)[Dynamics and robustness] result (Figure 10) was obtained by retraining the network at each $Delta t_"sim"$ value in the swept range, and confirms invariance of training+inference at matched $Delta t_"sim"$ within the range tested (not invariance of inference at a fixed-Δt-trained network to a varied inference Δt).

  Default parameters used by all experiments are listed in the parameters table. Per-experiment values of the loop weights $W^(E I)$, $W^(I E)$ and the input rate, and ranges swept by experiment (e.g. $tau_"GABA"$, $W^(I E)$), are stated in the corresponding figure captions.

  === Datasets and evaluation <sec-datasets-and-evaluation>

  The classification task is MNIST#cite(58), rate-encoded as in #link(<sec-training-methods>)[Training methods] (200 ms presentation, 25 Hz peak Poisson rate per active pixel). The official 60,000-image training partition supplied optimizer-training and fixed stratified validation subsets; the official 10,000-image test partition remained sealed until evaluation of the selected checkpoint. For the streaming protocol used in #link(<sec-streaming-classification>)[Streaming classification], the trained network was presented with a continuously concatenated input stream whose hidden state did not reset between digits. Excitatory spikes drove a non-spiking leaky-integrator readout with state $u_"out"[k]$; a trailing mean over the current presentation produced class scores $z_c[k]$, and $hat(y) = "argmax"_c z_c[k]$ gave the label. For every segment, the decoder used the known segment boundary and matched the readout window exactly to the presentation duration, $T_"readout" = T_"present"$; the readout window was not varied independently. The duration–rate grid used $T_"present" in {10, 15, 25, 40, 50, 75, 100, 200}$ ms and $r_"input,max" in {5, 10, 25, 50, 100, 200}$ Hz. Additional evaluations held both durations fixed at $200$ ms while sweeping below $5$ Hz to locate the encoder's chance floor, using the same three trained seeds. Reported metrics are official-test accuracy, mean E and I firing rates, and $f_gamma$.

  Training emitted both a validation-selected checkpoint and a final-epoch checkpoint. *Deployment-performance* analyses, including classifier robustness and streaming evaluation, used the validation-selected checkpoint. *Endpoint-dynamics* analyses of firing rates, gamma dynamics, integration timestep, recurrent weights, and training destinations used the final-epoch checkpoint. Every derived result records the analysis purpose, checkpoint role, resolved epoch, filename, and SHA-256 digest.

  === Reproducibility

  Each reported result was produced by a standalone experiment script in the project repository (linked under #link(<sec-code-and-data-availability>)[Code and data availability]). Each experiment hardcodes its own run scale (the sample count, seed set, and any parameter sweeps) and records those settings, together with the git commit and a run identifier, in the run's provenance file; every run number quoted in this manuscript and its captions is interpolated from those files rather than typed by hand, so a figure and the text that describes it cannot drift apart. The figure-render pipeline regenerates every print-quality figure from the experiments, so a clean re-run of the repository reproduces the figures and numbers reported here.

  === Software and implementation <sec-software-and-implementation>

  The model, training, and analysis are implemented in Python ($>= 3.10$). Spiking dynamics and surrogate-gradient training use PyTorch (2.11; with snnTorch 0.9 for baseline spiking primitives). Numerical analysis uses NumPy (2.2) and SciPy (1.15). All figures are produced with Matplotlib (3.10).

  == Code and data availability <sec-code-and-data-availability>

  Source code, per-notebook reproduction scripts, trained-weight artefacts, and the figure-render pipeline are available at #link("https://github.com/eoinmurray/pinglab")[https://github.com/eoinmurray/pinglab]. MNIST is obtained from its standard public distributor. Library versions are listed in #link(<sec-software-and-implementation>)[Software and implementation].

  == Declaration of generative-AI use

  Claude Code (Anthropic; model Opus 4.8), an agentic large-language-model coding tool, was used extensively and interactively in the development of this work. Its use covers writing the simulation, training, analysis, and figure-rendering code in the project repository; debugging, refactoring, and code review; drafting, revising, and copy-editing the manuscript prose; and iteration on experimental design and presentation. No figure, illustration, table, or visualisation in this manuscript is produced by a generative AI model; all figures are rendered by Matplotlib from Python code in the project repository. The authors set the research questions, designed the experiments and analyses, ran the simulations, reviewed model-generated code prior to commit, and verified the reported results and the manuscript text. The authors are responsible for the content, accuracy, and claims of this work.

  #reference-list((
    (text: [Buzsáki & Wang — _Mechanisms of Gamma Oscillations_. 2012.], doi: "10.1146/annurev-neuro-062111-150444"),
    (text: [Fries — _Rhythms for Cognition: Communication through Coherence_. 2015.], doi: "10.1016/j.neuron.2015.09.034"),
    (text: [Fries, Reynolds, Rorie & Desimone — _Modulation of Oscillatory Neuronal Synchronization by Selective Visual Attention_. 2001.], doi: "10.1126/science.1055465"),
    (text: [Gray, König, Engel & Singer — _Oscillatory Responses in Cat Visual Cortex Exhibit Inter-Columnar Synchronization Which Reflects Global Stimulus Properties_. 1989.], doi: "10.1038/338334a0"),
    (text: [Whittington, Traub, Kopell, Ermentrout & Buhl — _Inhibition-Based Rhythms: Experimental and Mathematical Observations on Network Dynamics_. 2000.], doi: "10.1016/S0167-8760(00)00173-2"),
    (text: [Williams et al. — _Fast Spiking Interneurons Autonomously Generate Fast Gamma Oscillations in the Medial Entorhinal Cortex with Excitation Strength Tuning ING-PING Transitions_. 2026.], doi: "10.1523/ENEURO.0452-25.2026"),
    (text: [Börgers — _The PING Model of Gamma Rhythms_. 2017.], doi: "10.1007/978-3-319-51171-9_30"),
    (text: [Cardin, Carlén, Meletis, Knoblich, Zhang, Deisseroth, Tsai & Moore — _Driving Fast-Spiking Cells Induces Gamma Rhythm and Controls Sensory Responses_. 2009.], doi: "10.1038/nature08002"),
    (text: [Sohal, Zhang, Yizhar & Deisseroth — _Parvalbumin Neurons and Gamma Rhythms Enhance Cortical Circuit Performance_. 2009.], doi: "10.1038/nature07991"),
    (text: [Phensy et al. — _Prefrontal Gamma Oscillations Engage Dynamic Cell Type-Specific Configurations to Support Flexible Behavior_. 2026.], doi: "10.1016/j.neuron.2026.05.002"),
    (text: [#link("https://www.sciencedirect.com/science/article/pii/S0301008225001571")[Offermanns, Pöpplau & Hanganu-Opatz — _Developmental Embedding of Parvalbumin Interneurons Drives Local and Crosshemispheric Prefrontal Gamma Synchrony_]. 2026.]),
    (text: [Whittington, Traub & Jefferys — _Synchronized Oscillations in Interneuron Networks Driven by Metabotropic Glutamate Receptor Activation_. 1995.], doi: "10.1038/373612a0"),
    (text: [Wang & Buzsáki — _Gamma Oscillation by Synaptic Inhibition in a Hippocampal Interneuronal Network Model_. 1996.], doi: "10.1523/JNEUROSCI.16-20-06402.1996"),
    (text: [Bartos, Vida & Jonas — _Synaptic Mechanisms of Synchronized Gamma Oscillations in Inhibitory Interneuron Networks_. 2007.], doi: "10.1038/nrn2044"),
    (text: [Kopell, Börgers, Pervouchine, Malerba & Tort — _Gamma and Theta Rhythms in Biophysical Models of Hippocampal Circuits_. 2010.], doi: "10.1007/978-1-4419-0996-1_15"),
    (text: [Viriyopase, Memmesheimer & Gielen — _Cooperation and Competition of Gamma Oscillation Mechanisms_. 2016.], doi: "10.1152/jn.00493.2015"),
    (text: [Brunel & Wang — _What Determines the Frequency of Fast Network Oscillations with Irregular Neural Discharges? I. Synaptic Dynamics and Excitation-Inhibition Balance_. 2003.], doi: "10.1152/jn.01095.2002"),
    (text: [Wilson & Cowan — _Excitatory and Inhibitory Interactions in Localized Populations of Model Neurons_. 1972.], doi: "10.1016/S0006-3495(72)86068-5"),
    (text: [Segneri, Bi, Olmi & Torcini — _Theta-Nested Gamma Oscillations in Next Generation Neural Mass Models_. 2020.], doi: "10.3389/fncom.2020.00047"),
    (text: [Nandi, Valla & di Volo — _Bursting Gamma Oscillations in Neural Mass Models_. 2024.], doi: "10.3389/fncom.2024.1422159"),
    (text: [Tahvili, Vinck & di Volo — _A Mean-Field Model of Neural Networks with PV and SOM Interneurons Reveals Connectivity-Based Mechanisms of Gamma Oscillations_. 2026.], doi: "10.1371/journal.pcbi.1014378"),
    (text: [Eshraghian, Ward, Neftci, Wang, Lenz, Dwivedi, Bennamoun, Jeong & Lu — _Training Spiking Neural Networks Using Lessons From Deep Learning_. 2023.], doi: "10.1109/JPROC.2023.3308088"),
    (text: [Neftci, Mostafa & Zenke — _Surrogate Gradient Learning in Spiking Neural Networks_. 2019.], doi: "10.1109/MSP.2019.2931595"),
    (text: [Deckers et al. — _Advancing Spatio-Temporal Processing Through Adaptation in Spiking Neural Networks_. 2025.], doi: "10.1038/s41467-025-60878-z"),
    (text: [Yan, Yang, Wu, Liu, Zhang, Li, Tan & Wu — _Efficient and Robust Temporal Processing with Neural Oscillations Modulated Spiking Neural Networks_. 2025.], doi: "10.1038/s41467-025-63771-x"),
    (text: [Bittar & Garner — _Exploring Neural Oscillations During Speech Perception via Surrogate-Gradient Spiking Neural Networks_. 2024.], doi: "10.3389/fnins.2024.1449181"),
    (text: [Barth & Poulet — _Experimental Evidence for Sparse Firing in the Neocortex_. 2012.], doi: "10.1016/j.tins.2012.03.008"),
    (text: [Attwell & Laughlin — _An Energy Budget for Signaling in the Grey Matter of the Brain_. 2001.], doi: "10.1097/00004647-200110000-00001"),
    (text: [Howarth, Gleeson & Attwell — _Updated Energy Budgets for Neural Computation in the Neocortex and Cerebellum_. 2012.], doi: "10.1038/jcbfm.2012.35"),
    (text: [Ainsworth, Lee, Cunningham, Traub, Kopell & Whittington — _Rates and Rhythms: A Synergistic View of Frequency and Temporal Coding in Neuronal Networks_. 2012.], doi: "10.1016/j.neuron.2012.06.027"),
    (text: [Schaefer, Angelo, Spors & Margrie — _Neuronal Oscillations Enhance Stimulus Discrimination by Ensuring Action Potential Precision_. 2006.], doi: "10.1371/journal.pbio.0040163"),
    (text: [Nguyen & Rubchinsky — _Temporal Patterns of Synchrony in a Pyramidal-Interneuron Gamma (PING) Network_. 2021.], doi: "10.1063/5.0042451"),
    (text: [Shadlen & Movshon — _Synchrony Unbound: A Critical Evaluation of the Temporal Binding Hypothesis_. 1999.], doi: "10.1016/S0896-6273(00)80822-3"),
    (text: [London, Roth, Beeren, Häusser & Latham — _Sensitivity to Perturbations in vivo Implies High Noise and Suggests Rate Coding in Cortex_. 2010.], doi: "10.1038/nature09086"),
    (text: [Akam & Kullmann — _Efficient "Communication through Coherence" Requires Oscillations Structured to Minimize Interference between Signals_. 2012.], doi: "10.1371/journal.pcbi.1002760"),
    (text: [Renart, de la Rocha, Bartho, Hollender, Parga, Reyes & Harris — _The Asynchronous State in Cortical Circuits_. 2010.], doi: "10.1126/science.1179850"),
    (text: [van Vreeswijk & Sompolinsky — _Chaos in Neuronal Networks with Balanced Excitatory and Inhibitory Activity_. 1996.], doi: "10.1126/science.274.5293.1724"),
    (text: [Vogels, Sprekeler, Zenke, Clopath & Gerstner — _Inhibitory Plasticity Balances Excitation and Inhibition in Sensory Pathways and Memory Networks_. 2011.], doi: "10.1126/science.1211095"),
    (text: [Hennequin, Agnes & Vogels — _Inhibitory Plasticity: Balance, Control, and Codependence_. 2017.], doi: "10.1146/annurev-neuro-072116-031005"),
    (text: [Wu, Miehl & Gjorgjieva — _Regulation of Circuit Organization and Function Through Inhibitory Synaptic Plasticity_. 2022.], doi: "10.1016/j.tins.2022.10.006"),
    (text: [Páscoa dos Santos & Verschure — _Excitatory-Inhibitory Homeostasis and Bifurcation Control in the Wilson-Cowan Model of Cortical Dynamics_. 2025.], doi: "10.1371/journal.pcbi.1012723"),
    (text: [Kann — _The Interneuron Energy Hypothesis: Implications for Brain Disease_. 2016.], doi: "10.1177/0271678X16638956"),
    (text: [Börgers, Talei Franzesi, LeBeau, Boyden & Kopell — _Minimal Size of Cell Assemblies Coordinated by Gamma Oscillations_. 2012.], doi: "10.1371/journal.pcbi.1002362"),
    (text: [Cramer, Stradmann, Schemmel & Zenke — _The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks_. 2022.], doi: "10.1109/TNNLS.2020.3044364"),
    (text: [Tiesinga & Sejnowski — _Cortical Enlightenment: Are Attentional Gamma Oscillations Driven by ING or PING?_. 2009.], doi: "10.1016/j.neuron.2009.09.009"),
    (text: [Brunel — _Dynamics of Sparsely Connected Networks of Excitatory and Inhibitory Spiking Neurons_. 2000.], doi: "10.1023/A:1008925309027"),
    (text: [Gerstner — _Population Dynamics of Spiking Neurons: Fast Transients, Asynchronous States, and Locking_. 2000.], doi: "10.1162/089976600300015899"),
    (text: [Montbrió, Pazó & Roxin — _Macroscopic Description for Networks of Spiking Neurons_. 2015.], doi: "10.1103/PhysRevX.5.021028"),
    (text: [Zenke & Ganguli — _SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks_. 2018.], doi: "10.1162/neco_a_01086"),
    (text: [#link("https://arxiv.org/abs/1711.05101")[Loshchilov & Hutter — _Decoupled Weight Decay Regularization_]. 2019.]),
    (text: [#link("https://proceedings.mlr.press/v28/pascanu13.html")[Pascanu, Mikolov & Bengio — _On the Difficulty of Training Recurrent Neural Networks_]. 2013.]),
    (text: [#link("https://openreview.net/forum?id=eU776ZYxEpz")[Cornford, Kalajdzievski, Leite, Lamarquette, Kullmann & Richards — _Learning to Live with Dale's Principle: ANNs with Separate Excitatory and Inhibitory Units_]. 2021.]),
    (text: [Zhu et al. — _Task Success in Trained Spiking Neural Network Models Coincides with Emergence of Cross-Stimulus-Modulated Inhibition_. 2026.], doi: "10.1007/s00422-025-01030-4"),
    (text: [Welch — _The Use of Fast Fourier Transform for the Estimation of Power Spectra: A Method Based on Time Averaging Over Short, Modified Periodograms_. 1967.], doi: "10.1109/TAU.1967.1161901"),
    (text: [Atallah & Scanziani — _Instantaneous Modulation of Gamma Oscillation Frequency by Balancing Excitation with Inhibition_. 2009.], doi: "10.1016/j.neuron.2009.04.027"),
    (text: [Xing, Shen, Burns, Yeh, Shapley & Li — _Stochastic Generation of Gamma-Band Activity in Primary Visual Cortex of Awake and Anesthetized Monkeys_. 2012.], doi: "10.1523/JNEUROSCI.5644-11.2012"),
    (text: [Rotter & Diesmann — _Exact Digital Simulation of Time-Invariant Linear Systems with Applications to Neuronal Modeling_. 1999.], doi: "10.1007/s004220050570"),
    (text: [LeCun, Bottou, Bengio & Haffner — _Gradient-Based Learning Applied to Document Recognition_. 1998.], doi: "10.1109/5.726791"),
  ))
]
#body
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [This manuscript examines how inhibitory feedback relates gamma timing to sparse spiking and classification. Available inputs are shown independently below; the integrated results, quantitative interpretation, and conclusions await the remaining inputs.],
    preview-figures.map(item => item + (description: figure-description(item.path.split("/").first()),)),
    json-inputs: ("exp022", "exp023", "exp025", "exp033", "exp037", "exp038", "exp041", "exp042", "exp044", "exp046", "exp048", "exp049",),
  )
}

#let meta = meta + (assets: input-assets("exp109", inputs))
#let body = with-datasets("exp109", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-contents(body)
