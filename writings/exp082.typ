#let meta = (
  title: "Variable-rate streaming with a spike-count readout",
  date: "2026-08-10",
  description: "A multi-seed study of matched-window spike-count classification across input rates and presentation durations.",
  collection: "gamma-gated-sparsity",
  status: "complete",
)

#let r = json("/artifacts/data/exp082/numbers.json")
#let pct(x) = str(calc.round(100 * x, digits: 1)) + "%"
#let mean(xs) = xs.sum() / xs.len()
#let minimum(xs) = calc.min(..xs)
#let maximum(xs) = calc.max(..xs)
#let at-duration(duration) = r.grid_per_seed.filter(row => row.duration_ms == duration)
#let at-rate(rate) = r.duration_200ms_psychometric.filter(row => row.rate_hz == rate)
#let body = [
  == 1. Abstract

  This study asks whether PING networks trained across input rates can classify continuous MNIST streams with their native output-spike readout when input rate and presentation duration change. The hypothesis is that matching the spike-count window to each presentation preserves useful class evidence across the trained rate distribution, while short or very sparse presentations fail because they provide too few output events. Three independently trained networks were frozen and evaluated over eleven input rates and four presentation durations. The primary estimand is held-out classification accuracy for each duration--rate condition, averaged across training seeds; output silence is a diagnostic estimand. At 200 ms, mean accuracy rose from #pct(mean(at-rate(0.5).map(row => row.accuracy))) at 0.5 Hz to #pct(mean(at-rate(7.5).map(row => row.accuracy))) at 7.5 Hz and remained #pct(mean(at-rate(25.0).map(row => row.accuracy))) at 25 Hz. Across rates, shortening the presentation reduced mean accuracy from #pct(mean(at-duration(200.0).map(row => row.accuracy))) at 200 ms to #pct(mean(at-duration(25.0).map(row => row.accuracy))) at 25 ms. The results support rate-robust deployment over much of the trained range, but not duration invariance or reliable inference at its sparsest edge.

  == 2. Design and scope

  The study consumes three PING networks trained independently with maximum-pixel Poisson rates sampled from 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, and 25 Hz. Recurrent weights, learned output projections, and checkpoint selection are frozen before evaluation. Every digit is classified from the output-LIF spikes emitted during its own presentation; the output neurons and their counters reset at digit boundaries, while hidden PING state remains continuous within each five-digit stream.

  The leading account predicts that a matched spike-count window will retain useful decisions across the trained input-rate range when enough spikes arrive. Its strongest rivals are a narrow rate optimum despite variable-rate training, apparent robustness caused by longer integration rather than rate tolerance, and a failure mode dominated by silent or near-silent output windows. The factorial intervention separates rate from evidence duration. The fixed-200-ms psychometric isolates rate tolerance, while the duration--rate grid tests how much shorter evidence windows degrade it.

  The primary condition-level estimand is

  $ A_(d,r,s) = 1/N sum_(i=1)^N 1_(hat(y)_(i,d,r,s) = y_i), $

  where $A_(d,r,s)$ is accuracy for duration $d$, maximum-pixel rate $r$, and trained seed $s$; $N=200$ classified digits per seed and condition; $hat(y)$ is the spike-count prediction; and $y$ is the true label. Results are summarized across the three trained seeds without treating the 600 digit presentations as independent network replicates. The planned falsifiers are chance-level performance throughout the trained rate range, no benefit from longer matched windows, or pervasive output silence even at moderate rates.

  #block(inset: 10pt, fill: rgb("eef4f8"), radius: 3pt)[
    No separately frozen prospective study plan predates execution. The hypotheses, estimands, rivals, and falsifiers above are reconstructed from the preserved experimental design and code. They remain distinct from the observations below, but this retrospective reconstruction is weaker than prospective registration.
  ]

  All 132 seed-level factorial cells completed: four durations by eleven rates by three trained seeds, with 200 classified digits in each cell. The study evaluates deployment performance of validation-selected classifiers on repeated held-out streams. It does not isolate training-distribution effects from readout effects, compare against a fixed-rate-trained control, or estimate generalization across architectures, datasets, or a larger population of training seeds.

  == 3. Investigations

  === 3.1 How output spikes become a decision

  The first investigation establishes that the deployed readout is the trained spike-count mechanism rather than a reconstructed voltage decoder. At timestep $t$, excitatory spike vector $bold(s)^E(t)$ drives ten learned output-LIF units. For a presentation beginning at $a$ and ending before $b$, cumulative evidence for class $c$ is

  $ z_c(u) = sum_(t=a)^u s_c^"out"(t), quad a <= u < b, $

  where $s_c^"out"(t)$ is the spike emitted by class unit $c$. The prediction is $arg max_c z_c(b-1)$. If rhythmic population bursts deliver useful evidence packets, competing class counts may exchange the lead early before the correct class accumulates a stable margin. A sparse failure should instead appear as few or no output spikes.

  #figure(
    image("/artifacts/data/exp082/single_trial.png", width: 100%, alt: "One MNIST digit above excitatory and inhibitory spike rasters and ten cumulative spike-count evidence traces."),
    caption: [A correctly classified 200 ms presentation at 5 Hz, selected as the first success in a pre-existing matched stream. The result explains the readout trajectory but is not an accuracy estimate. The true and winning class is red.],
  )

  In the selected digit-4 presentation, discrete output spikes increment class counts around successive population bursts. The plotted display applies $p_c(u) = "softmax"_c(bold(z)(u))$ to those counts. This is not a calibrated posterior: for classes $c$ and $j$, $p_c(u)/p_j(u)=exp(z_c(u)-z_j(u))$, so a small integer margin can look nearly decisive. The winner is determined by the underlying counts, not by the display mapping.

  #figure(
    image("/artifacts/data/exp082/single_trial_transition.png", width: 100%, alt: "A zoom showing output spikes, cumulative class counts, and their softmax display."),
    caption: [A post-hoc explanatory enlargement of 91.5--94.5 ms in the same trial. Each output spike increments one cumulative count; normalization can lower another class's displayed share even when its own count is unchanged.],
  )

  The trajectory is consistent with successive PING bursts delivering packets of class evidence, but one outcome-selected example cannot establish that each cycle independently improves the decision or that rhythmic packaging causes the aggregate accuracy pattern.

  === 3.2 Classification in a changing stream

  The second investigation changes both input rate and presentation duration at digit boundaries while enforcing $T_"readout"=T_"presentation"$. If boundary resets isolate decisions correctly, class counts should restart without evidence leaking across labels, although hidden network state may influence early activity. The five registered conditions deliberately span sparse, dense, short, and matched presentations and serve as a qualitative stress test rather than an accuracy estimator.

  #figure(
    image("/artifacts/data/exp082/variable_stream.png", width: 100%, alt: "Excitatory and inhibitory rasters with online spike-count evidence as input rate and digit duration vary."),
    caption: [A five-digit variable stream using a variable-rate-trained PING network and its native output-LIF spike-count readout. Counts reset at each boundary while hidden PING state remains continuous. Thumbnail opacity increases with encoding rate; badges show true label and prediction.],
  )

  Three of the five registered presentations were classified correctly. The 0.5 Hz presentation emitted no output spikes and failed; the 2 Hz presentation also failed, while the 25, 10, and 5 Hz presentations were correct despite durations ranging from 25 to 200 ms. This single stream demonstrates correct matched-window operation and exposes the sparse silent-window failure, but its five outcomes do not estimate condition-level reliability.

  === 3.3 Rate and duration robustness

  The decisive investigation crosses four matched presentation/readout durations with all eleven training rates. Each of the 44 conditions contains 200 classified digits for each of three independently trained networks. Under the leading account, accuracy should improve as rate and duration supply more output events, then remain useful across a broad upper-rate region rather than collapsing around one narrow optimum.

  #figure(
    image("/artifacts/data/exp082/duration_rate_summary.png", width: 100%, alt: "Accuracy over presentation duration and input rate beside the fixed-200-millisecond psychometric."),
    caption: [Held-out accuracy across matched presentation/readout durations and trained input rates. The right panel fixes duration at 200 ms to separate rate tolerance from integration time.],
  )

  Averaged across all rates and seeds, accuracy increased monotonically with duration: #pct(mean(at-duration(25.0).map(row => row.accuracy))) at 25 ms, #pct(mean(at-duration(50.0).map(row => row.accuracy))) at 50 ms, #pct(mean(at-duration(100.0).map(row => row.accuracy))) at 100 ms, and #pct(mean(at-duration(200.0).map(row => row.accuracy))) at 200 ms. Mean output-silence fractions fell over the same sequence from #pct(mean(at-duration(25.0).map(row => row.silent_fraction))) to #pct(mean(at-duration(200.0).map(row => row.silent_fraction))). Shortened windows therefore remove both time for hidden dynamics and time to accumulate output evidence; this design does not separate those two consequences.

  #figure(
    image("/artifacts/data/exp082/psychometric_200ms.svg", width: 85%, alt: "Classification accuracy versus maximum-pixel input rate at a fixed 200 milliseconds."),
    caption: [Input-rate psychometric with presentation and spike-count window fixed at 200 ms. Each rate contains 200 held-out digit presentations for each of three trained seeds.],
  )

  At 200 ms, mean accuracy increased from #pct(mean(at-rate(0.5).map(row => row.accuracy))) at 0.5 Hz to #pct(mean(at-rate(3.0).map(row => row.accuracy))) at 3 Hz and #pct(mean(at-rate(7.5).map(row => row.accuracy))) at 7.5 Hz. It remained between #pct(mean(at-rate(10.0).map(row => row.accuracy))) and #pct(mean(at-rate(15.0).map(row => row.accuracy))) at 10--25 Hz, rather than collapsing at the dense end of the trained range. Across-seed accuracies at 15 Hz ranged from #pct(minimum(at-rate(15.0).map(row => row.accuracy))) to #pct(maximum(at-rate(15.0).map(row => row.accuracy))). The sparse edge remains materially weaker: at 0.5 Hz, accuracy ranged from #pct(minimum(at-rate(0.5).map(row => row.accuracy))) to #pct(maximum(at-rate(0.5).map(row => row.accuracy))), and mean silent-window frequency was #pct(mean(at-rate(0.5).map(row => row.silent_fraction))).

  == 4. Executed methods

  === 4.1 Networks, data, and frozen selection

  Three PING networks were trained independently on MNIST with one maximum-pixel Poisson rate sampled per image presentation from the eleven registered values. Each network used a learned projection from 1024 excitatory cells to ten output-LIF class units and spike-count logits during training. Validation accuracy selected one checkpoint per seed before deployment evaluation. Recurrent weights, output projections, and checkpoints remained fixed throughout this study.

  === 4.2 Streaming protocol

  Input pixels generated independent Bernoulli spikes at 0.1 ms resolution. Every evaluation cell comprised 40 independent five-digit streams per trained seed, giving 200 classified digits. Streams were simulated in batches of five with separate neuronal state for each batch member. Hidden state continued only across the five digits within one stream. At every digit boundary the output-LIF state and spike counter reset, ensuring that a decision used only its matched presentation window.

  === 4.3 Factorial evaluation and diagnostics

  The factorial evaluation crossed 25, 50, 100, and 200 ms presentations with 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, and 25 Hz maximum-pixel rates. The fixed-duration psychometric is the 200 ms slice of this grid. For each seed and condition, the study retained correct and total classifications, output spikes per presentation, silent-window fraction, and excitatory and inhibitory population rates. Aggregate plots summarize seed-level accuracies rather than pooling seeds into one nominal replicate.

  === 4.4 Illustrative trajectories

  The variable stream used five fixed duration--rate pairs and was interpreted qualitatively. The single-trial figure selected the first correctly classified presentation from a pre-existing matched 200 ms, 5 Hz stream specifically to explain a successful readout. The 91.5--94.5 ms enlargement was chosen post hoc around a conspicuous transition. Neither selection contributes to the aggregate accuracy estimand.

  === 4.5 Deviations, uncertainty, and robustness limits

  Execution completed at the registered production scale with no recorded missing factorial cells. The artifact does not contain a separately frozen prospective plan, confidence intervals over a defined population of images and network trainings, a larger training-seed sample, an independently repeated stream bank, or multiplicity-adjusted condition comparisons. Robustness is therefore supported by replication across three trained seeds and the complete factorial surface, not by a population-level variance model. The validation-selected checkpoint policy also makes this a deployment estimate, not a study of final-epoch training dynamics.

  == 5. Conclusion

  The study supports the bounded claim that variable-rate-trained PING networks can use their native matched-window spike-count readout across a broad portion of the trained input-rate range. At 200 ms, accuracy remained high from moderate through dense input rates, so the result rejects a narrow single-rate optimum within the tested range. Longer windows consistently improved accuracy and reduced silence, while the 0.5 Hz edge remained weak; the data therefore reject duration invariance and expose insufficient output evidence as a practical sparse-input failure mode.

  The result does not isolate why variable-rate training works, prove that PING cycles are causal evidence packets, or show that spike-count readout is superior to the exp048 mean-voltage decoder. That comparison changes both training distribution and readout and is confounded. A stronger successor should prospectively freeze its plan, compare fixed-rate and variable-rate training under matched readouts, use more independent training seeds, predefine uncertainty intervals and decision thresholds, and manipulate integration time separately from presentation duration.
]
