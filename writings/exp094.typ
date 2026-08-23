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

  == 2. Shared

  *Identity and scope.* This is a complete local `ExpScout` in the `gamma-gated-sparsity` collection. It compares decoder transfer in canonical seed-42 COBA and PING checkpoints without retraining, parameter search, or population inference.

  *Scientific frame.* When the learned network and output projection are frozen, how much of the native decision survives changes to temporal evidence $z$, and what does a later output mapping $p$ change without changing the winner? The leading mechanism is that PING rhythms package useful class competition into transferable output events. The strongest rivals are a simpler output-rate mismatch, dependence on one checkpoint, fixed decoder timescales, and the deterministic convenience sample.

  *Decision gates and budget.* Stop if decoder transfer is uninformative, revise if the intervention is mismatched, or escalate if a COBA–PING contrast survives the balanced screen. The local budget was one illustrative replay plus #r.screen.n balanced screening images, with no retraining, hyperparameter search, seed sweep, or full-test evaluation.

  *Inputs and controls.*
  The scout uses final-epoch `mem-mean` checkpoints from publication run #raw(r.training_source.publication_run_id), upstream campaign #raw(r.training_source.upstream_campaign_id): #raw(r.training_source.cells.coba) and #raw(r.training_source.cells.ping). Official MNIST test index #r.selection.official_test_index, label #r.selection.label, was chosen without inspecting its outcome and encoded with shared Poisson seed #r.shared_input.seed. The screen uses seed #r.screen.design.poisson_seed and batches of #r.screen.design.batch_size.

  Both networks, their hidden activity, learned output weights, and encoded inputs remain frozen. Native mean-voltage evidence uses the trained non-spiking output. Every spike-based alternative reads one counterfactual output-LIF trajectory driven by the same hidden spikes and weights. All output mappings read the same cumulative-count tensor. PING voting uses detected cycles; COBA uses duration-matched bins.

  #figure(
    image("/artifacts/data/exp094/decoder_pipeline.svg", width: 100%, alt: "Frozen COBA and PING activity passing through temporal evidence, output mapping, and decision stages."),
    caption: [Shared decoder decomposition. Frozen activity enters a temporal evidence rule $z$, then an optional display mapping $p$, before the final argmax decision.],
  )

  #block(inset: 10pt, fill: rgb("eef4f8"), radius: 3pt)[
    *Registration limitation.* No separately frozen prospective `ExpScoutPlan` predates this run. The plan scaffold preserved here is a retrospective reconstruction from the runner, manifest, outputs, and outcome-blind selections. It separates reconstructed expectations from observations but cannot manufacture prospective registration after execution.
  ]

  *Shared execution.* Run `r005` completed locally on 2026-08-22 at commit `dac6703b` with the recorded source dirty; `_dirty.patch` preserves the exact code difference. The executed configuration matches the values recorded above. No investigation failed or remained incomplete, and no additional scientific deviation is known. The preserved record includes `numbers.json`, compact arrays, checkpoint hashes, runner, manifest, and dirty-source patch under `artifacts/data/exp094/`. The evidence is exploratory.

  == 3. Investigations

  === 3.1 Investigation 1 — Running mean voltage

  *Relevance.* Running mean voltage is the native decoder used to train both frozen checkpoints. It establishes the reference decision before any output-spike intervention.

  *Reconstructed expected patterns.* The true class should finish above its competitors in both networks; different transient competition may remain visible before the temporal mean settles.

  *ExpVisualSet.* The blue-grey schematic defines the running mean and the red-black mirror shows the measured true-class and strongest-competitor trajectories.

  #figure([
    #image("/artifacts/data/exp094/z_mean.svg", width: 100%, alt: "Illustrative output voltages becoming mean-voltage evidence.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_mean.svg", width: 100%, alt: "Measured running mean-voltage evidence for frozen COBA and PING trajectories.")
  ], caption: [Native mean-voltage evidence retains subthreshold activity while removing temporal order.])

  *Measured result — complete. Observed patterns.* Both native decoders select the true class. COBA settles rapidly, while PING retains more early competition before the true-class mean remains larger. This is the baseline against which later evidence rules are judged.

  === 3.2 Investigation 2 — Cumulative spike count

  *Relevance.* Cumulative count asks whether frozen output spikes preserve the trained decision when every spike remains evidence until reset.

  *Reconstructed expected patterns.* Transferable output spikes should allow the true class to accumulate a durable lead. Dense, poorly matched output activity should instead obscure class separation.

  *ExpVisualSet.* The schematic defines count accumulation and the measured mirror applies it to the single shared counterfactual output trajectory.

  #figure([
    #image("/artifacts/data/exp094/z_cumulative.svg", width: 100%, alt: "Illustrative output spikes becoming cumulative staircase evidence.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_cumulative.svg", width: 100%, alt: "Measured cumulative output-spike evidence for frozen COBA and PING trajectories.")
  ], caption: [Cumulative evidence preserves every output spike until reset.])

  *Measured result — complete. Observed patterns.* PING eventually gives the true class the largest cumulative count, whereas COBA does not. The contrast indicates different compatibility with the spike-count intervention, not the accuracy either network would achieve if trained with this decoder.

  === 3.3 Investigation 3 — Leaky spike count

  *Relevance.* Leaky evidence tests whether forgetting old spikes reveals recent class competition better than an irreversible cumulative count.

  *Reconstructed expected patterns.* Recent PING evidence may remain separable as older spikes decay. Leakage cannot help if the counterfactual output rate keeps all classes near a common ceiling.

  *ExpVisualSet.* The schematic shows spike-triggered increments and decay; the measured mirror uses the same fixed leak for both networks.

  #figure([
    #image("/artifacts/data/exp094/z_leaky.svg", width: 100%, alt: "Illustrative output spikes becoming decaying leaky evidence.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_leaky.svg", width: 100%, alt: "Measured leaky spike evidence for frozen COBA and PING trajectories.")
  ], caption: [Leaky evidence weights recent output spikes more strongly.])

  *Measured result — complete. Observed patterns.* PING retains changing local separation under leakage. COBA remains close to a steady saturated level, so forgetting old evidence does not recover a useful decision.

  === 3.4 Investigation 4 — Sliding-window spike count

  *Relevance.* A finite window makes the decoder's memory boundary explicit and tests whether local rather than accumulated evidence carries the decision.

  *Reconstructed expected patterns.* A compatible event representation should show class differences within the fixed window. Saturated windows should erase useful competition.

  *ExpVisualSet.* The schematic defines membership in the window and the measured mirror shows the resulting local counts.

  #figure([
    #image("/artifacts/data/exp094/z_window.svg", width: 100%, alt: "Illustrative time window selecting which output spikes remain evidence.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_window.svg", width: 100%, alt: "Measured sliding-window spike evidence for frozen COBA and PING trajectories.")
  ], caption: [Windowed evidence retains spikes for #r.decoder.window_ms ms, then removes them.])

  *Measured result — complete. Observed patterns.* PING's local competition remains visible as the window moves. COBA windows are nearly full, showing that a memory rule cannot rescue an output event rate poorly matched to it.

  === 3.5 Investigation 5 — Cycle or matched-bin voting

  *Relevance.* Voting tests whether treating rhythmic intervals as discrete evidence packets preserves class information beyond raw burst size.

  *Reconstructed expected patterns.* If PING cycles carry structured competition, the true class should win more detected cycles. Duration-matched COBA bins provide a non-rhythmic control.

  *ExpVisualSet.* The schematic assigns one winner per interval; the mirror accumulates PING cycle votes and duration-matched COBA votes.

  #figure([
    #image("/artifacts/data/exp094/z_vote.svg", width: 100%, alt: "Illustrative interval winners becoming cumulative class votes.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_vote.svg", width: 100%, alt: "Measured cumulative cycle or matched-bin votes for frozen COBA and PING trajectories.")
  ], caption: [Cycle or matched-bin evidence gives each interval one class vote.])

  *Measured result — complete. Observed patterns.* The PING true class wins more detected cycles and pulls away in cumulative votes. Matched COBA bins favour another class. This is consistent with meaningful rhythmic packets in the selected PING trajectory, but voting alone does not establish their causal importance.

  === 3.6 Investigation 6 — Ordinary softmax

  *Relevance.* Ordinary softmax separates evidence accumulation from the visual sharpening produced by a normalized output mapping.

  *Reconstructed expected patterns.* Softmax should preserve the cumulative-count winner while converting margins into competing class shares.

  *ExpVisualSet.* The mapping receives the same cumulative-count vector used by every investigation in this output-mapping group.

  #figure([
    #image("/artifacts/data/exp094/p_softmax.svg", width: 100%, alt: "Illustrative cumulative counts becoming ordinary-softmax shares.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_p_softmax.svg", width: 100%, alt: "Measured ordinary softmax of cumulative counts for frozen COBA and PING trajectories.")
  ], caption: [Ordinary softmax converts count margins into competing class shares.])

  *Measured result — complete. Observed patterns.* Softmax makes PING leader changes appear abrupt and decisive but leaves the cumulative-count winner unchanged. COBA's different winner is likewise preserved. The plot changes confidence display, not evidence quality.

  === 3.7 Investigation 7 — Temperature-softened softmax

  *Relevance.* Temperature tests whether apparent decisiveness can be reduced without changing accumulated evidence or the winning class.

  *Reconstructed expected patterns.* Positive temperature should preserve every argmax while making competitors easier to see.

  *ExpVisualSet.* The same cumulative-count vector is divided by one fixed temperature before softmax.

  #figure([
    #image("/artifacts/data/exp094/p_softened.svg", width: 100%, alt: "Illustrative cumulative counts becoming temperature-softened shares.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_p_softened.svg", width: 100%, alt: "Measured temperature-softened softmax of the same cumulative counts.")
  ], caption: [Temperature softening exposes competitors while preserving the winner.])

  *Measured result — complete. Observed patterns.* Softening exposes more of PING's competing classes while preserving its leader changes and final winner. The difference from ordinary softmax is entirely presentational.

  === 3.8 Investigation 8 — Independent sigmoid

  *Relevance.* Independent sigmoid removes mutual class competition and tests whether a bounded per-class display is meaningful for non-negative accumulating counts.

  *Reconstructed expected patterns.* Several classes should approach one together, making the final decision tie-dominated and uninformative.

  *ExpVisualSet.* Each cumulative count is transformed independently rather than normalized across classes.

  #figure([
    #image("/artifacts/data/exp094/p_sigmoid.svg", width: 100%, alt: "Illustrative cumulative counts becoming independent sigmoid scores that rise together.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_p_sigmoid.svg", width: 100%, alt: "Measured independent sigmoid scores from the same cumulative counts.")
  ], caption: [Independent sigmoid bounds each class separately and removes mutual competition.])

  *Measured result — complete. Observed patterns.* All classes saturate near one, so the displayed winner is governed by numerical tie handling rather than useful separation. The failure is specific to applying independent sigmoid directly to cumulative non-negative counts.

  === 3.9 Investigation 9 — Balanced-screen accuracy

  *Relevance.* Aggregate accuracy tests whether the illustrative decoder-transfer contrast immediately collapses across a balanced set of other inputs.

  *Reconstructed expected patterns.* A useful signal should survive across digits, while remaining descriptive because the screen contains one checkpoint per architecture and is not a population estimate.

  *ExpVisualSet.* The accuracy matrix applies all eight decoder rules to the same first ten official-test examples of every digit.

  #figure(
    image("/artifacts/data/exp094/screen_accuracy.svg", width: 100%, alt: "Accuracy matrix for eight decoder options across a balanced 100-image COBA and PING screening sample."),
    caption: [Decoder-transfer screen. The divider separates temporal evidence rules from mappings of shared cumulative evidence.],
  )

  *Measured result — complete. Observed patterns.* COBA's spike-based rules retain only #accuracy("coba", "vote")--#accuracy("coba", "cumulative")% accuracy from a #accuracy("coba", "mean")% native baseline, whereas PING retains #accuracy("ping", "window")--#accuracy("ping", "cumulative")% from a #accuracy("ping", "mean")% native baseline. The contrast clears the scout's cheap aggregate gate without establishing a general architectural effect.

  === 3.10 Investigation 10 — Native-decision transitions

  *Relevance.* Aggregate accuracy alone cannot distinguish preserved native decisions from compensating repairs and new failures.

  *Reconstructed expected patterns.* A compatible alternative should preserve most native correct decisions rather than merely exchange one set of errors for another.

  *ExpVisualSet.* The transition plot cross-tabulates every alternative decision against its network's native decision.

  #figure(
    image("/artifacts/data/exp094/screen_transitions.svg", width: 100%, alt: "Stacked transition counts showing which native decisions each alternative decoder preserves, breaks, or repairs."),
    caption: [Transitions show whether each alternative preserves, breaks, repairs, or retains a native error.],
  )

  *Measured result — complete. Observed patterns.* COBA alternatives break most decisions that its native decoder gets right. PING alternatives preserve substantially more native correctness, with only limited repairs. The contrast is therefore a difference in decision retention, not an artefact of aggregate averaging.

  === 3.11 Investigation 11 — Class-specific changes

  *Relevance.* Per-class changes test whether aggregate decoder loss is confined to the selected digit or one unusually difficult class.

  *Reconstructed expected patterns.* A broad incompatibility should affect several digits; a selected-example artefact should remain narrowly concentrated.

  *ExpVisualSet.* The heatmap subtracts native class accuracy from each alternative's class accuracy on the balanced screen.

  #figure(
    image("/artifacts/data/exp094/screen_classes.svg", width: 100%, alt: "Per-digit accuracy changes from native decoding for COBA and PING across all decoder options."),
    caption: [Class-specific changes show whether aggregate loss is confined to one digit.],
  )

  *Measured result — complete. Observed patterns.* COBA's loss spans nearly every digit and disproportionately collapses to class zero. PING's smaller losses are distributed rather than confined to the illustrative class. This weakens the selected-example explanation but does not replace a multi-seed full-test study.

  == 4. Executed methods

  These methods are the complete executed account and replace the retrospectively reconstructed planned protocol in this rendering. `experiments/exp094.py` generated the concrete arrays and plots cited below; `_manifest.json`, `_run.txt`, and `_dirty.patch` preserve run and code provenance.

  === 4.1 Shared frozen replay and output intervention

  *Mathematical steps.* For hidden spike $s_i^H(t)$ and learned output weight $W_(i c)^"out"$, the shared class drive is $u_c(t) = sum_i W_(i c)^"out" s_i^H(t)$. The native non-spiking output yields pre-reset voltage $v_c^"pre"(t)$. The counterfactual output LIF receives the same $u_c(t)$ and yields spike $s_c(t) in {0, 1}$.

  *Output and investigation link.* The replay produces `measurements.npz`, the shared input and output trajectories used by Investigations 1--8. No network or projection weight is retrained.

  === 4.2 Running mean voltage

  *Mathematical steps.* For class $c$ at timestep $t$,
  $ z_c^"mean"(t) = 1/(t + 1) sum_(tau=0)^t v_c^"pre"(tau). $
  The final class is $arg max_c z_c^"mean"(t_"final")$.

  *Output and investigation link.* This produces the native baseline and `measured_z_mean.svg` for Investigation 1.

  === 4.3 Cumulative spike count

  *Mathematical steps.* Accumulate every counterfactual output spike:
  $ z_c^"cum"(t) = sum_(tau=0)^t s_c(tau). $

  *Output and investigation link.* This produces `measured_z_cumulative.svg` for Investigation 2, supplies the shared evidence vector for Investigations 6--8, and enters the diagnostics in Investigations 9--11.

  === 4.4 Leaky spike count

  *Mathematical steps.* Retain a fraction $lambda$ of earlier evidence and add the current spike:
  $ z_c^"leak"(t) = lambda z_c^"leak"(t - 1) + s_c(t), quad 0 < lambda < 1. $
  The fixed leak timescale is #r.decoder.leak_tau_ms ms.

  *Output and investigation link.* This produces `measured_z_leaky.svg` for Investigation 3 and the leaky diagnostic inputs for Investigations 9--11.

  === 4.5 Sliding-window spike count

  *Mathematical steps.* For a window of $W$ timesteps,
  $ z_c^"win"(t) = sum_(tau=max(0, t - W + 1))^t s_c(tau). $
  Here $W$ corresponds to #r.decoder.window_ms ms.

  *Output and investigation link.* This produces `measured_z_window.svg` for Investigation 4 and the window diagnostic inputs for Investigations 9--11.

  === 4.6 Cycle or matched-bin voting

  *Mathematical steps.* Count class spikes $n_(j,k)$ inside bin $k$, assign the bin to its largest count, and accumulate votes:
  $ z_c^"vote"(K) = sum_(k=1)^K 1_(c = arg max_j n_(j,k)). $
  PING bins use detected cycles; COBA bins use the same durations.

  *Output and investigation link.* This produces `measured_z_vote.svg` for Investigation 5 and the vote diagnostic inputs for Investigations 9--11.

  === 4.7 Ordinary softmax

  *Mathematical steps.* Apply one normalized exponential to shared cumulative evidence:
  $ p_c(t) = frac(exp(z_c^"cum"(t)), sum_j exp(z_j^"cum"(t))). $

  *Output and investigation link.* This produces `measured_p_softmax.svg` for Investigation 6 and the softmax diagnostic inputs for Investigations 9--11. Its argmax equals cumulative count.

  === 4.8 Temperature-softened softmax

  *Mathematical steps.* Divide the same evidence by fixed temperature $T=#r.decoder.softmax_temperature$ before normalization:
  $ p_c(t; T) = frac(exp(z_c^"cum"(t) / T), sum_j exp(z_j^"cum"(t) / T)). $

  *Output and investigation link.* This produces `measured_p_softened.svg` for Investigation 7 and the softened diagnostic inputs for Investigations 9--11. Positive temperature preserves the argmax.

  === 4.9 Independent sigmoid

  *Mathematical steps.* Transform each cumulative count independently:
  $ p_c^"sig"(t) = 1 / (1 + exp(-z_c^"cum"(t))). $
  The scores need not sum to one; numerical ties use the implementation's default argmax.

  *Output and investigation link.* This produces `measured_p_sigmoid.svg` for Investigation 8 and the sigmoid diagnostic inputs for Investigations 9--11.

  === 4.10 Balanced screen and transition accounting

  *Mathematical steps.* For $N=#r.screen.n$ labelled images with predictions $hat(y)_i$, accuracy is
  $ A = 1/N sum_(i=1)^N 1_(hat(y)_i = y_i). $
  Each alternative is cross-tabulated against the native decoder as correct-to-correct, correct-to-wrong, wrong-to-correct, or wrong-to-wrong. Per-class change subtracts native class accuracy from alternative class accuracy.

  *Output and investigation link.* The method produces `screen_predictions.npz`; `screen_accuracy.svg` is consumed by Investigation 9, `screen_transitions.svg` by Investigation 10, and `screen_classes.svg` by Investigation 11. The first ten official-test images per class are fixed without outcome inspection.

  == 5. Reconstructed planned synthesis

  The investigations were intended to separate three questions: whether spike-based evidence preserves the native decision, whether a display mapping merely changes apparent confidence, and whether the illustrative contrast survives a cheap balanced screen. A PING advantage across cumulative, leaky, windowed, and cycle-aware evidence, accompanied by native-decision retention across classes, would favour decoder compatibility over a selected-example artefact. Similar COBA and PING transfer would stop the branch; common saturation or silence would require revision of the output intervention. A contrast confined to cycle voting would motivate a rhythm-specific test, whereas a contrast shared by all spike rules would leave output-rate mismatch as the strongest rival. Completion required one shared replay, all planned visual mirrors, and the balanced-screen diagnostics.

  == 6. Conclusion

  *Decision: escalate, with revision.* Under this frozen-weight intervention, PING output spikes preserve substantially more of the native decision than COBA output spikes, and the contrast survives the cheap balanced screen. Output mappings alone change apparent confidence but cannot repair incompatible evidence.

  The strongest unresolved rivals are output-rate mismatch, one-checkpoint dependence, fixed decoder timescales, and the deterministic convenience sample. There are no sampling intervals, independent training seeds, or held-out choices of leak, window, and temperature. The next artifact must be a new prospective `ExpStudyPlan` that freezes a multi-seed full-test estimand, controls decoder timescales, measures rate compatibility, and tests whether cycle-aware evidence explains transfer beyond cumulative counts. This scout itself remains exploratory.
]
