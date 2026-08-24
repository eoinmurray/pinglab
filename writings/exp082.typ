#let meta = (
  title: "Spike-count streaming across input rates",
  date: "2026-08-10",
  description: "A multi-seed study of spike-count classification across input rates and presentation durations.",
  collection: "gamma-gated-sparsity",
  status: "ExpScout",
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

  We asked whether a spiking network could classify a continuous digit stream when each digit arrived at a different strength or remained visible for a different time. We froze networks trained on changing input rates and used their output spikes as class evidence. The classifier worked across a broad range of input strengths once enough spikes arrived. Weak or brief inputs were harder, and some ended before the output produced any evidence. Longer presentations helped, but the experiment cannot separate extra viewing time from extra decision time. It also does not show that the network's rhythmic activity caused the result.

  The main lesson is that the classifier handled changing inputs when each digit produced enough evidence before the decision ended.

  == 2. Prospective design and scope

  Three PING networks were trained independently across changing input rates, then frozen before evaluation. Within each five-digit stream, the PING network runs continuously: its hidden neuronal state does not reset when the digit changes. Only the output-LIF state and class counts reset, so each digit starts a new decision without restarting the network. The shared design below shows this streaming arrangement.

  #figure(
    image("/artifacts/data/exp082/shared_design_schematic.svg", width: 100%, alt: "Conceptual streaming design. Changing digit inputs pass through a PING network to output spikes, class counts, and a prediction. Hidden state continues across digit boundaries while output state and counts reset."),
    caption: [Planned streaming evaluation. Hidden network state continues across digits, while output evidence resets at each boundary. This blue-grey schematic defines the prospective mechanism and is not measured evidence.],
  )

  We test whether this readout remains useful across a broad range of input strengths once enough spikes arrive. A narrow preferred rate is the main rival. The fixed-duration evaluation tests rate tolerance, while the duration--rate grid tests shorter matched windows. Chance-level performance would reject useful transfer. Persistent silence at moderate rates would instead identify missing output activity as the limiting failure. Accuracy is summarized across trained networks. Because presentation and decision time change together, the design cannot separate their effects.

  #block(inset: 10pt, fill: rgb("eef4f8"), radius: 3pt)[
    The experiment and publication scaffold were committed before execution. The networks, readout, evaluation grid, checkpoint policy, sampling scale, and planned outputs were fixed. Missing directional thresholds and a formal uncertainty model limit confirmatory interpretation.
  ]

  == 3. Investigations

  === 3.1 How output spikes become a decision

  The first investigation shows how the trained readout forms a decision. At timestep $t$, the excitatory spike vector $bold(s)^E(t)$ drives ten learned output-LIF units. For a presentation beginning at $a$ and ending before $b$, class $c$ accumulates

  $ z_c(u) = sum_(t=a)^u s_c^"out"(t), quad a <= u < b, $

  where $s_c^"out"(t)$ is the spike emitted by output unit $c$. The final prediction is $arg max_c z_c(b-1)$. If rhythmic bursts deliver useful evidence packets, classes can exchange the lead before one builds a stable margin. Sparse failure should instead produce few or no output spikes.

  #figure(
    image("/artifacts/data/exp082/single_trial.png", width: 85%, alt: "One MNIST digit above excitatory and inhibitory spike rasters and ten cumulative spike-count evidence traces."),
    caption: [A correctly classified 200 ms presentation at 5 Hz, selected as the first success in a pre-existing matched stream. The figure explains the readout; it does not estimate accuracy. Red marks the true and winning class.],
  )

  In this digit-4 presentation, output spikes increment class counts around successive population bursts. Softmax converts those counts into the displayed shares, but it is not a calibrated posterior and does not determine the winner. The trajectory explains the readout and is consistent with bursts delivering packets of class evidence. One selected example cannot show that each cycle improves the decision or that rhythmic packaging causes the aggregate pattern. The next investigation therefore tests the same readout in a changing stream.

  === 3.2 Classification in a changing stream

  The second investigation changes input rate and presentation duration at digit boundaries while enforcing $T_"readout"=T_"presentation"$. Boundary resets should prevent class evidence from leaking across labels, although continuing hidden state can shape early activity. Five registered conditions span sparse, dense, short, and matched presentations. This stream is a qualitative stress test, not an accuracy estimate.

  #figure(
    image("/artifacts/data/exp082/variable_stream.png", width: 100%, alt: "Excitatory and inhibitory rasters with online spike-count evidence as input rate and digit duration vary."),
    caption: [A five-digit stream classified by the native output-LIF spike-count readout. Counts reset at each boundary; hidden PING state continues. Thumbnail opacity increases with input rate, and badges show the true and predicted labels.],
  )

  Three of five presentations were correct. The 0.5 Hz presentation produced no output spikes and failed; the 2 Hz presentation also failed. Presentations at 5, 10, and 25 Hz were correct despite lasting from 25 to 200 ms. The stream verifies matched-window operation and exposes a silent sparse-input failure. Five outcomes cannot estimate reliability.

  === 3.3 Rate and duration robustness

  The factorial evaluation crosses four matched presentation/readout durations with all eleven training rates. Each condition contains 200 digits for each network. The leading account predicts that accuracy will rise as rate and duration supply more output events, then remain useful across a broad upper-rate region.

  #figure(
    image("/artifacts/data/exp082/duration_rate_summary.png", width: 100%, alt: "Accuracy over presentation duration and input rate beside the fixed-200-millisecond psychometric."),
    caption: [Held-out accuracy across matched presentation/readout durations and trained input rates. The right panel fixes duration at 200 ms to separate rate tolerance from integration time.],
  )

  Mean accuracy increased monotonically with duration: #pct(mean(at-duration(25.0).map(row => row.accuracy))) at 25 ms, #pct(mean(at-duration(50.0).map(row => row.accuracy))) at 50 ms, #pct(mean(at-duration(100.0).map(row => row.accuracy))) at 100 ms, and #pct(mean(at-duration(200.0).map(row => row.accuracy))) at 200 ms. Mean output silence fell from #pct(mean(at-duration(25.0).map(row => row.silent_fraction))) to #pct(mean(at-duration(200.0).map(row => row.silent_fraction))). Shorter presentations leave less time for both hidden dynamics and evidence accumulation. This design cannot separate those effects.

  #figure(
    image("/artifacts/data/exp082/psychometric_200ms.svg", width: 85%, alt: "Classification accuracy versus maximum-pixel input rate at a fixed 200 milliseconds."),
    caption: [Input-rate psychometric with presentation and spike-count window fixed at 200 ms. Each rate contains 200 held-out digit presentations for each of three trained seeds.],
  )

  At 200 ms, mean accuracy rose from #pct(mean(at-rate(0.5).map(row => row.accuracy))) at 0.5 Hz to #pct(mean(at-rate(3.0).map(row => row.accuracy))) at 3 Hz and #pct(mean(at-rate(7.5).map(row => row.accuracy))) at 7.5 Hz. Across 10--25 Hz, it remained between #pct(mean(at-rate(10.0).map(row => row.accuracy))) and #pct(mean(at-rate(15.0).map(row => row.accuracy))). The dense end did not collapse. At 15 Hz, seed-level accuracy ranged from #pct(minimum(at-rate(15.0).map(row => row.accuracy))) to #pct(maximum(at-rate(15.0).map(row => row.accuracy))). The sparse edge was weaker: at 0.5 Hz, accuracy ranged from #pct(minimum(at-rate(0.5).map(row => row.accuracy))) to #pct(maximum(at-rate(0.5).map(row => row.accuracy))), with #pct(mean(at-rate(0.5).map(row => row.silent_fraction))) silent windows on average.

  == 4. Executed methods

  === 4.1 Networks and frozen selection

  Three PING networks were trained independently on MNIST. Each image presentation sampled one of the eleven registered input rates. A learned projection connected 1024 excitatory cells to ten output-LIF class units, whose spike counts served as logits. Validation accuracy selected one checkpoint per seed. All weights and checkpoints remained fixed during the study.

  === 4.2 Streaming evaluation and diagnostics

  Input pixels generated independent Bernoulli spikes at 0.1 ms resolution. Each evaluation cell comprised 40 independent five-digit streams per seed, giving 200 decisions. We simulated five streams per batch, with separate neuronal state for each stream. Hidden state continued only within a stream. At every digit boundary, the output-LIF state and counter reset. Each decision therefore used its matched presentation window alone. The factorial evaluation crossed presentations of 25, 50, 100, and 200 ms with input rates from 0.5 to 25 Hz. The fixed-duration psychometric is the 200 ms slice. For each seed and condition, we retained accuracy, output spikes per presentation, silent-window fraction, and excitatory and inhibitory population rates. The plots summarize seed-level accuracy; they do not pool seeds into one nominal replicate.

  The primary estimand was condition-level accuracy,

  $ A_(d,r,s) = 1/N sum_(i=1)^N 1_(hat(y)_(i,d,r,s) = y_i), $

  where $A_(d,r,s)$ is accuracy for duration $d$, rate $r$, and trained seed $s$. Each cell contains $N=200$ digits; $hat(y)$ is the predicted label and $y$ is the true label. Output silence is a companion diagnostic. We summarize the three seed-level accuracies rather than treating 600 presentations as independent network replicates.

  === 4.3 Illustrations, deviations, and limits

  The variable stream used five fixed duration--rate pairs and supports only qualitative interpretation. The single-trial figure shows the first correct presentation in a pre-existing 200 ms, 5 Hz stream, selected to explain a successful readout rather than estimate reliability. All factorial cells completed without a known scientific deviation, retraining, or parameter search. Evaluation used held-out streams and validation-selected classifiers. The study lacks a fixed-rate control, population-level intervals, additional training seeds, an independent stream-bank repeat, and adjusted condition comparisons. Agreement across three networks supports robustness within the tested protocol, not across a broader network population. This is a deployment evaluation rather than an account of final-epoch dynamics.

  #pagebreak()

  == 5. Conclusion

  Across these three networks, the native matched-window readout worked over a broad part of the trained rate range. At 200 ms, accuracy remained high from moderate through dense rates, weakening the narrow-optimum rival. Longer windows improved accuracy and reduced silence. The weak 0.5 Hz edge points to insufficient output evidence as the sparse-input failure mode.

  The study does not explain why variable-rate training works, establish a causal role for PING cycles, or show superiority over the exp048 mean-voltage decoder. That comparison changed both training distribution and readout. A stronger successor should compare fixed-rate and variable-rate training under matched readouts. It should add training seeds, predefine uncertainty and decision thresholds, and manipulate presentation time separately from integration time.
]
