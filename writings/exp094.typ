#let meta = (
  title: "Decoder transfer in frozen COBA and PING networks",
  date: "2026-08-22",
  description: "An exploratory decoder-transfer scout using matched conceptual diagrams, measured trajectories, and a balanced 100-image screen.",
  collection: "gamma-gated-sparsity",
  status: "complete",
)

#let r = json("/artifacts/data/exp094/numbers.json")
#let accuracy(model, decoder) = calc.round(100 * r.screen.models.at(model).at(decoder).accuracy)
#let transitions(model, decoder) = r.screen.models.at(model).at(decoder).transitions

#let body = [
  == 1. Abstract

  This exploratory scout asks how much of a frozen MNIST decision survives when its temporal evidence rule changes, and what a later display mapping changes without changing the underlying evidence. The reconstructed hypothesis is that rhythmic PING activity packages class competition into output events that transfer more faithfully than COBA activity from the native mean-voltage decoder to spike-based evidence. The intervention freezes each network and its learned projection while changing only the temporal evidence rule or its later display mapping. The primary estimand is retention of native correct decisions on an outcome-blind example and a balanced 100-image screen. PING output spikes retain substantially more of the native decision than COBA output spikes under the tested decoders, while output mappings alter apparent confidence but not a shared cumulative-count winner. The result clears the escalation gate for a prospective multi-seed study; it is not durable evidence of an architectural effect.

  == 2. Design and scope

  We compare decoder transfer in canonical COBA and PING networks trained on MNIST. When the learned network and output projection are frozen, how much of the native decision survives changes to temporal evidence $z$, and what does a later output mapping $p$ change without changing the winner? The leading mechanism is that PING rhythms package useful class competition into transferable output events. Simpler rivals include output-rate mismatch, dependence on one trained network of each kind, fixed decoder timescales, and the deterministic convenience sample.

  Both networks, their hidden activity, learned output weights, and encoded inputs remain frozen. Native mean-voltage evidence uses the trained non-spiking output. Every spike-based alternative reads one counterfactual output-LIF trajectory driven by the same hidden spikes and weights. All output mappings read the same cumulative-count tensor. PING voting uses detected cycles; COBA uses duration-matched bins. The illustrative MNIST example was chosen without inspecting its outcome, and the screen uses the first ten official-test examples from each digit class.

  The scout stops if decoder transfer is uninformative, requires revision if the intervention is mismatched, and escalates if a COBA–PING contrast survives the balanced screen. Its budget is one illustrative replay plus #r.screen.n balanced screening images, with no retraining, hyperparameter search, network-seed sweep, or full-test evaluation.

  #figure(
    image("/artifacts/data/exp094/decoder_pipeline.svg", width: 100%, alt: "Frozen COBA and PING activity passing through temporal evidence, output mapping, and decision stages."),
    caption: [Shared decoder decomposition. Frozen activity enters a temporal evidence rule $z$, then an optional display mapping $p$, before the final argmax decision.],
  )

  #block(inset: 10pt, fill: rgb("eef4f8"), radius: 3pt)[
    No separately frozen prospective plan predates this run. The expectations and gates are reconstructed from the preserved experimental record and outcome-blind selections. They are kept distinct from observations but cannot substitute for prospective registration.
  ]

  Every planned investigation completed, with no known scientific deviation from the reconstructed design. The evidence remains exploratory.

  == 3. Investigations

  === 3.1 Running mean voltage

  Running mean voltage is the native decoder used to train both frozen networks, so it establishes the reference decision before any output-spike intervention. The true class should finish above its competitors in both networks, although their transient competition may differ before the temporal mean settles. The schematic defines the running mean; its measured mirror follows the true class and strongest competitor.

  #figure([
    #image("/artifacts/data/exp094/z_mean.svg", width: 100%, alt: "Illustrative output voltages becoming mean-voltage evidence.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_mean.svg", width: 100%, alt: "Measured running mean-voltage evidence for frozen COBA and PING trajectories.")
  ], caption: [Native mean-voltage evidence retains subthreshold activity while removing temporal order.])

  Both native decoders select the true class. COBA settles rapidly, while PING retains more early competition before the true-class mean remains larger. This is the baseline against which later evidence rules are judged.

  === 3.2 Cumulative spike count

  Cumulative count asks whether frozen output spikes preserve the trained decision when every spike remains evidence until reset. Transferable output spikes should allow the true class to accumulate a durable lead; dense, poorly matched output activity should instead obscure class separation. The schematic defines count accumulation, and its measured mirror applies that rule to the shared counterfactual output trajectory.

  #figure([
    #image("/artifacts/data/exp094/z_cumulative.svg", width: 100%, alt: "Illustrative output spikes becoming cumulative staircase evidence.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_cumulative.svg", width: 100%, alt: "Measured cumulative output-spike evidence for frozen COBA and PING trajectories.")
  ], caption: [Cumulative evidence preserves every output spike until reset.])

  PING eventually gives the true class the largest cumulative count, whereas COBA does not. The contrast indicates different compatibility with the spike-count intervention, not the accuracy either network would achieve if trained with this decoder.

  === 3.3 Leaky spike count

  Leaky evidence tests whether forgetting old spikes reveals recent class competition better than an irreversible cumulative count. Recent PING evidence may remain separable as older spikes decay, whereas leakage cannot help if the counterfactual output rate keeps all classes near a common ceiling. The measured mirror uses the same fixed leak for both networks.

  #figure([
    #image("/artifacts/data/exp094/z_leaky.svg", width: 100%, alt: "Illustrative output spikes becoming decaying leaky evidence.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_leaky.svg", width: 100%, alt: "Measured leaky spike evidence for frozen COBA and PING trajectories.")
  ], caption: [Leaky evidence weights recent output spikes more strongly.])

  PING retains changing local separation under leakage. COBA remains close to a steady saturated level, so forgetting old evidence does not recover a useful decision.

  === 3.4 Sliding-window spike count

  A finite window makes the decoder's memory boundary explicit and tests whether local rather than accumulated evidence carries the decision. A compatible event representation should show class differences within the fixed window, while saturated windows should erase useful competition. The measured mirror shows the resulting local counts.

  #figure([
    #image("/artifacts/data/exp094/z_window.svg", width: 100%, alt: "Illustrative time window selecting which output spikes remain evidence.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_window.svg", width: 100%, alt: "Measured sliding-window spike evidence for frozen COBA and PING trajectories.")
  ], caption: [Windowed evidence retains spikes for #r.decoder.window_ms ms, then removes them.])

  PING's local competition remains visible as the window moves. COBA windows are nearly full, showing that a memory rule cannot rescue an output event rate poorly matched to it.

  === 3.5 Cycle or matched-bin voting

  Voting tests whether treating rhythmic intervals as discrete evidence packets preserves class information beyond raw burst size. If PING cycles carry structured competition, the true class should win more detected cycles; duration-matched COBA bins provide a non-rhythmic control. The measured mirror accumulates one winner per PING cycle or matched COBA interval.

  #figure([
    #image("/artifacts/data/exp094/z_vote.svg", width: 100%, alt: "Illustrative interval winners becoming cumulative class votes.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_vote.svg", width: 100%, alt: "Measured cumulative cycle or matched-bin votes for frozen COBA and PING trajectories.")
  ], caption: [Cycle or matched-bin evidence gives each interval one class vote.])

  The PING true class wins more detected cycles and pulls away in cumulative votes. Matched COBA bins favour another class. This is consistent with meaningful rhythmic packets in the selected PING trajectory, but voting alone does not establish their causal importance.

  === 3.6 Ordinary softmax

  Ordinary softmax separates evidence accumulation from the visual sharpening produced by a normalized output mapping. It should preserve the cumulative-count winner while converting margins into competing class shares, using the same cumulative-count vector as the other output mappings.

  #figure([
    #image("/artifacts/data/exp094/p_softmax.svg", width: 100%, alt: "Illustrative cumulative counts becoming ordinary-softmax shares.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_p_softmax.svg", width: 100%, alt: "Measured ordinary softmax of cumulative counts for frozen COBA and PING trajectories.")
  ], caption: [Ordinary softmax converts count margins into competing class shares.])

  Softmax makes PING leader changes appear abrupt and decisive but leaves the cumulative-count winner unchanged. COBA's different winner is likewise preserved. The plot changes confidence display, not evidence quality.

  === 3.7 Temperature-softened softmax

  Temperature tests whether apparent decisiveness can be reduced without changing accumulated evidence or the winning class. Dividing the same cumulative-count vector by a positive fixed temperature before softmax should preserve every argmax while making competitors easier to see.

  #figure([
    #image("/artifacts/data/exp094/p_softened.svg", width: 100%, alt: "Illustrative cumulative counts becoming temperature-softened shares.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_p_softened.svg", width: 100%, alt: "Measured temperature-softened softmax of the same cumulative counts.")
  ], caption: [Temperature softening exposes competitors while preserving the winner.])

  Softening exposes more of PING's competing classes while preserving its leader changes and final winner. The difference from ordinary softmax is entirely presentational.

  === 3.8 Independent sigmoid

  Independent sigmoid removes mutual class competition and tests whether a bounded per-class display is meaningful for non-negative accumulating counts. Because each count is transformed independently rather than normalized across classes, several classes should approach one together and make the final decision tie-dominated.

  #figure([
    #image("/artifacts/data/exp094/p_sigmoid.svg", width: 100%, alt: "Illustrative cumulative counts becoming independent sigmoid scores that rise together.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_p_sigmoid.svg", width: 100%, alt: "Measured independent sigmoid scores from the same cumulative counts.")
  ], caption: [Independent sigmoid bounds each class separately and removes mutual competition.])

  All classes saturate near one, so the displayed winner is governed by numerical tie handling rather than useful separation. The failure is specific to applying independent sigmoid directly to cumulative non-negative counts.

  === 3.9 Balanced-screen accuracy

  Aggregate accuracy tests whether the illustrative decoder-transfer contrast immediately collapses across a balanced set of other inputs. A useful signal should survive across digits, although the screen remains descriptive because it contains one trained network of each kind and is not a population estimate. The matrix applies all eight decoder rules to the same first ten official-test examples of every digit.

  #figure(
    image("/artifacts/data/exp094/screen_accuracy.svg", width: 100%, alt: "Accuracy matrix for eight decoder options across a balanced 100-image COBA and PING screening sample."),
    caption: [Decoder-transfer screen. The divider separates temporal evidence rules from mappings of shared cumulative evidence.],
  )

  COBA's spike-based rules retain only #accuracy("coba", "vote")--#accuracy("coba", "cumulative")% accuracy from a #accuracy("coba", "mean")% native baseline, whereas PING retains #accuracy("ping", "window")--#accuracy("ping", "cumulative")% from a #accuracy("ping", "mean")% native baseline. The contrast clears the scout's cheap aggregate gate without establishing a general architectural effect.

  === 3.10 Native-decision transitions

  Aggregate accuracy alone cannot distinguish preserved native decisions from compensating repairs and new failures. A compatible alternative should preserve most native correct decisions rather than merely exchange one set of errors for another, so the transition plot cross-tabulates every alternative decision against its network's native decision.

  #figure(
    image("/artifacts/data/exp094/screen_transitions.svg", width: 100%, alt: "Stacked transition counts showing which native decisions each alternative decoder preserves, breaks, or repairs."),
    caption: [Transitions show whether each alternative preserves, breaks, repairs, or retains a native error.],
  )

  COBA alternatives break most decisions that its native decoder gets right. PING alternatives preserve substantially more native correctness, with only limited repairs. The contrast is therefore a difference in decision retention, not an artefact of aggregate averaging.

  === 3.11 Class-specific changes

  Per-class changes test whether aggregate decoder loss is confined to the selected digit or one unusually difficult class. A broad incompatibility should affect several digits, whereas a selected-example artefact should remain narrowly concentrated. The heatmap subtracts native class accuracy from each alternative's class accuracy on the balanced screen.

  #figure(
    image("/artifacts/data/exp094/screen_classes.svg", width: 100%, alt: "Per-digit accuracy changes from native decoding for COBA and PING across all decoder options."),
    caption: [Class-specific changes show whether aggregate loss is confined to one digit.],
  )

  COBA's loss spans nearly every digit and disproportionately collapses to class zero. PING's smaller losses are distributed rather than confined to the illustrative class. This weakens the selected-example explanation but does not replace a multi-seed full-test study.

  == 4. Executed methods

  These methods give the complete scientific account of the executed scout and replace the retrospectively reconstructed planned protocol in this rendering.

  === 4.1 Shared frozen replay and output intervention

  For hidden spike $s_i^H(t)$ and learned output weight $W_(i c)^"out"$, the shared class drive is $u_c(t) = sum_i W_(i c)^"out" s_i^H(t)$. The native non-spiking output yields pre-reset voltage $v_c^"pre"(t)$. The counterfactual output leaky integrate-and-fire unit receives the same $u_c(t)$ and yields spike $s_c(t) in {0, 1}$. The resulting shared input and output trajectories support Investigations 1--8; no network or projection weight is retrained.

  === 4.2 Running mean voltage

  For class $c$ at timestep $t$,
  $ z_c^"mean"(t) = 1/(t + 1) sum_(tau=0)^t v_c^"pre"(tau). $
  The final class is $arg max_c z_c^"mean"(t_"final")$. This is the native baseline used in Investigation 1.

  === 4.3 Cumulative spike count

  Cumulative evidence retains every counterfactual output spike:
  $ z_c^"cum"(t) = sum_(tau=0)^t s_c(tau). $
  It supplies the evidence used in Investigation 2, the shared vector transformed in Investigations 6--8, and the cumulative-count decisions summarized in Investigations 9--11.

  === 4.4 Leaky spike count

  Leaky evidence retains a fraction $lambda$ of earlier evidence and adds the current spike:
  $ z_c^"leak"(t) = lambda z_c^"leak"(t - 1) + s_c(t), quad 0 < lambda < 1. $
  The fixed leak timescale is #r.decoder.leak_tau_ms ms. The same definition is used for the trajectory in Investigation 3 and the screen diagnostics.

  === 4.5 Sliding-window spike count

  For a window of $W$ timesteps,
  $ z_c^"win"(t) = sum_(tau=max(0, t - W + 1))^t s_c(tau). $
  Here $W$ corresponds to #r.decoder.window_ms ms. The same window is used for the trajectory in Investigation 4 and the screen diagnostics.

  === 4.6 Cycle or matched-bin voting

  Class spikes $n_(j,k)$ are counted inside interval $k$; each interval is assigned to its largest count, and votes accumulate as
  $ z_c^"vote"(K) = sum_(k=1)^K 1_(c = arg max_j n_(j,k)). $
  PING intervals use detected cycles, while COBA intervals use the same durations. The same vote definition is used in Investigation 5 and the screen diagnostics.

  === 4.7 Ordinary softmax

  Ordinary softmax applies one normalized exponential to shared cumulative evidence:
  $ p_c(t) = frac(exp(z_c^"cum"(t)), sum_j exp(z_j^"cum"(t))). $
  Its argmax equals the cumulative-count argmax.

  === 4.8 Temperature-softened softmax

  Temperature-softened softmax divides the same evidence by fixed temperature $T=#r.decoder.softmax_temperature$ before normalization:
  $ p_c(t; T) = frac(exp(z_c^"cum"(t) / T), sum_j exp(z_j^"cum"(t) / T)). $
  Positive temperature preserves the argmax.

  === 4.9 Independent sigmoid

  Independent sigmoid transforms each cumulative count separately:
  $ p_c^"sig"(t) = 1 / (1 + exp(-z_c^"cum"(t))). $
  The scores need not sum to one; numerical ties use the implementation's default argmax.

  === 4.10 Balanced screen and transition accounting

  For $N=#r.screen.n$ labelled images with predictions $hat(y)_i$, accuracy is
  $ A = 1/N sum_(i=1)^N 1_(hat(y)_i = y_i). $
  Each alternative is cross-tabulated against the native decoder as correct-to-correct, correct-to-wrong, wrong-to-correct, or wrong-to-wrong. Per-class change subtracts native class accuracy from alternative class accuracy. The first ten official-test images per class are fixed without outcome inspection.

  == 5. Interpretive framework

  The investigations were intended to separate three questions: whether spike-based evidence preserves the native decision, whether a display mapping merely changes apparent confidence, and whether the illustrative contrast survives a cheap balanced screen. A PING advantage across cumulative, leaky, windowed, and cycle-aware evidence, accompanied by native-decision retention across classes, would favour decoder compatibility over a selected-example artefact. Similar COBA and PING transfer would stop the branch; common saturation or silence would require revision of the output intervention. A contrast confined to cycle voting would motivate a rhythm-specific test, whereas a contrast shared by all spike rules would leave output-rate mismatch as the strongest rival. Completion required one shared replay, all planned visual mirrors, and the balanced-screen diagnostics.

  == 6. Conclusion

  The scout supports escalation with revision. Under this frozen-weight intervention, PING output spikes preserve substantially more of the native decision than COBA output spikes, and the contrast survives the cheap balanced screen. Output mappings alone change apparent confidence but cannot repair incompatible evidence.

  The strongest unresolved rivals are output-rate mismatch, one-network dependence, fixed decoder timescales, and the deterministic convenience sample. There are no sampling intervals, independent training seeds, or held-out choices of leak, window, and temperature. A prospective multi-seed full-test study should control decoder timescales, measure rate compatibility, and test whether cycle-aware evidence explains transfer beyond cumulative counts. This scout itself remains exploratory.
]
