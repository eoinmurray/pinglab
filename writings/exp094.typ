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

  This exploratory scout asks how much of a frozen MNIST decision survives when its temporal evidence rule changes, and what a later display mapping changes without changing the underlying evidence. One outcome-blind example and a balanced 100-image screen compare canonical seed-42 COBA and PING checkpoints. PING output spikes retain substantially more of the native decision than COBA output spikes under the tested decoders, while output mappings alter apparent confidence but not a shared cumulative-count winner. The result motivates a prospective multi-seed study; it is not durable evidence of an architectural effect.

  == 2. Shared

  *Question.* When the learned network and output projection are frozen, how much of the native decision survives changes to temporal evidence $z$, and what does a later output mapping $p$ change without changing the winner?

  *Decision gate.* Stop if decoder transfer is uninformative, revise if the intervention is mismatched, or escalate if a COBA–PING contrast survives the balanced screen.

  The scout uses final-epoch `mem-mean` checkpoints from publication run #raw(r.training_source.publication_run_id), upstream campaign #raw(r.training_source.upstream_campaign_id): #raw(r.training_source.cells.coba) and #raw(r.training_source.cells.ping). Official MNIST test index #r.selection.official_test_index, label #r.selection.label, was chosen without inspecting its outcome and encoded with shared Poisson seed #r.shared_input.seed. The screen uses seed #r.screen.design.poisson_seed and batches of #r.screen.design.batch_size.

  Both networks, their hidden activity, learned output weights, and encoded inputs remain frozen. Native mean-voltage evidence uses the trained non-spiking output. Every spike-based alternative reads one counterfactual output-LIF trajectory driven by the same hidden spikes and weights. All output mappings read the same cumulative-count tensor. PING voting uses detected cycles; COBA uses duration-matched bins.

  #figure(
    image("/artifacts/data/exp094/decoder_pipeline.svg", width: 100%, alt: "Frozen COBA and PING activity passing through temporal evidence, output mapping, and decision stages."),
    caption: [Shared decoder decomposition. Frozen activity enters a temporal evidence rule $z$, then an optional display mapping $p$, before the final argmax decision.],
  )

  #block(inset: 10pt, fill: rgb("eef4f8"), radius: 3pt)[
    *Record limitation.* No separately frozen `ExpScoutPlan` predates this run. The expectations and gates below are reconstructed from the runner, manifest, outputs, and outcome-blind selections. This records the executed scout honestly but cannot substitute for prospective registration.
  ]

  The local budget was one illustrative replay plus #r.screen.n balanced screening images, with no retraining, hyperparameter search, seed sweep, or full-test evaluation. The preserved record includes `numbers.json`, compact arrays, checkpoint hashes, runner, manifest, and dirty-source patch under `artifacts/data/exp094/`. The evidence is exploratory.

  == 3. Investigations

  === 3.1 Investigation 1 — Does temporal evidence transfer?

  *Relevance.* A classifier trained on temporally averaged subthreshold voltage may not remain meaningful when the same frozen projection is interpreted through output spikes. Decoder transfer therefore tests compatibility, not end-to-end performance.

  *What we expect.* If rhythmic PING activity packages class competition into more distinct events, spike-based evidence should preserve its native winner more often than in COBA. A null result would show similar behaviour across architectures; saturation or silence would instead expose an intervention mismatch.

  *ExpVisualSet.* Blue-grey schematics show the prospective mechanism; their red-black mirrors show the measured true-class and strongest-competitor trajectories.

  #figure([
    #image("/artifacts/data/exp094/z_mean.svg", width: 100%, alt: "Illustrative output voltages becoming mean-voltage evidence.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_mean.svg", width: 100%, alt: "Measured running mean-voltage evidence for frozen COBA and PING trajectories.")
  ], caption: [Native mean-voltage evidence retains subthreshold activity while removing temporal order.])

  #figure([
    #image("/artifacts/data/exp094/z_cumulative.svg", width: 100%, alt: "Illustrative output spikes becoming cumulative staircase evidence.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_cumulative.svg", width: 100%, alt: "Measured cumulative output-spike evidence for frozen COBA and PING trajectories.")
  ], caption: [Cumulative evidence preserves every output spike until reset.])

  #figure([
    #image("/artifacts/data/exp094/z_leaky.svg", width: 100%, alt: "Illustrative output spikes becoming decaying leaky evidence.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_leaky.svg", width: 100%, alt: "Measured leaky spike evidence for frozen COBA and PING trajectories.")
  ], caption: [Leaky evidence weights recent output spikes more strongly.])

  #figure([
    #image("/artifacts/data/exp094/z_window.svg", width: 100%, alt: "Illustrative time window selecting which output spikes remain evidence.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_window.svg", width: 100%, alt: "Measured sliding-window spike evidence for frozen COBA and PING trajectories.")
  ], caption: [Windowed evidence retains spikes for #r.decoder.window_ms ms, then removes them.])

  #figure([
    #image("/artifacts/data/exp094/z_vote.svg", width: 100%, alt: "Illustrative interval winners becoming cumulative class votes.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_vote.svg", width: 100%, alt: "Measured cumulative cycle or matched-bin votes for frozen COBA and PING trajectories.")
  ], caption: [Cycle or matched-bin evidence gives each interval one class vote.])

  *Discussion.* The native mean-voltage decoder selects the true class in both networks. Changing to output-spike evidence breaks the illustrative COBA decision but preserves the PING decision across the tested rules. COBA's output activity saturates cumulative, leaky, and windowed evidence; PING retains visible competition and a later true-class separation. This is consistent with better decoder compatibility in this PING trajectory, but it does not isolate rhythmic packetization from output-rate mismatch.

  === 3.2 Investigation 2 — What changes when only the output mapping changes?

  *Relevance.* Evidence accumulation and confidence display are different operations. Conflating them can make a visually decisive trace look like a better decoder even when its winning class is unchanged.

  *What we expect.* Ordinary and temperature-softened softmax should preserve the cumulative-count argmax while changing apparent sharpness. Independent sigmoid should remove class competition and may saturate because cumulative counts are non-negative.

  *ExpVisualSet.* Each mapping receives the identical cumulative-count vector; only $p$ changes.

  #figure([
    #image("/artifacts/data/exp094/p_softmax.svg", width: 100%, alt: "Illustrative cumulative counts becoming ordinary-softmax shares.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_p_softmax.svg", width: 100%, alt: "Measured ordinary softmax of cumulative counts for frozen COBA and PING trajectories.")
  ], caption: [Ordinary softmax converts count margins into competing class shares.])

  #figure([
    #image("/artifacts/data/exp094/p_softened.svg", width: 100%, alt: "Illustrative cumulative counts becoming temperature-softened shares.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_p_softened.svg", width: 100%, alt: "Measured temperature-softened softmax of the same cumulative counts.")
  ], caption: [Temperature softening exposes competitors while preserving the winner.])

  #figure([
    #image("/artifacts/data/exp094/p_sigmoid.svg", width: 100%, alt: "Illustrative cumulative counts becoming independent sigmoid scores that rise together.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_p_sigmoid.svg", width: 100%, alt: "Measured independent sigmoid scores from the same cumulative counts.")
  ], caption: [Independent sigmoid bounds each class separately and removes mutual competition.])

  *Discussion.* Both softmax variants preserve every cumulative-count leader change and final winner; temperature changes only how decisive the same evidence appears. Independent sigmoid drives all classes toward one and yields an uninformative tie-dominated decision. The observed difference belongs to the display mapping, not to a new evidence trajectory.

  === 3.3 Investigation 3 — Does the contrast survive a balanced screen?

  *Relevance.* One illustrative digit can reveal mechanism but cannot show whether the contrast immediately collapses on other inputs. A cheap balanced screen is the minimum escalation gate.

  *What we expect.* A useful signal should survive across digits without depending entirely on the selected example. The screen remains descriptive: it has one checkpoint per architecture and is not a test-set accuracy estimate.

  *ExpVisualSet.* The screen summarizes accuracy, decision transitions relative to native decoding, and class-specific changes for the first ten official-test examples of each digit.

  #figure(
    image("/artifacts/data/exp094/screen_accuracy.svg", width: 100%, alt: "Accuracy matrix for eight decoder options across a balanced 100-image COBA and PING screening sample."),
    caption: [Decoder-transfer screen. The divider separates temporal evidence rules from mappings of shared cumulative evidence.],
  )

  #figure(
    image("/artifacts/data/exp094/screen_transitions.svg", width: 100%, alt: "Stacked transition counts showing which native decisions each alternative decoder preserves, breaks, or repairs."),
    caption: [Transitions show whether each alternative preserves, breaks, repairs, or retains a native error.],
  )

  #figure(
    image("/artifacts/data/exp094/screen_classes.svg", width: 100%, alt: "Per-digit accuracy changes from native decoding for COBA and PING across all decoder options."),
    caption: [Class-specific changes show whether aggregate loss is confined to one digit.],
  )

  *Discussion.* The contrast survives the screen. COBA's spike-based rules retain only #accuracy("coba", "vote")--#accuracy("coba", "cumulative")% accuracy from a #accuracy("coba", "mean")% native baseline, whereas PING retains #accuracy("ping", "window")--#accuracy("ping", "cumulative")% from a #accuracy("ping", "mean")% baseline. COBA's failures span nearly every digit and disproportionately collapse to class zero. PING's losses are smaller and distributed. This clears the scout's escalation gate without establishing a population-level architectural effect.

  == 4. Methods

  === 4.1 Shared frozen replay and output intervention

  *Mathematical steps.* For hidden spike $s_i^H(t)$ and learned output weight $W_(i c)^"out"$, the shared class drive is $u_c(t) = sum_i W_(i c)^"out" s_i^H(t)$. The native non-spiking output yields pre-reset voltage $v_c^"pre"(t)$. The counterfactual output LIF receives the same $u_c(t)$ and yields spike $s_c(t) in {0, 1}$.

  *Output and investigation link.* The replay produces `measurements.npz`, the shared input and output trajectories used by Investigations 1 and 2. No network or projection weight is retrained.

  === 4.2 Running mean voltage

  *Mathematical steps.* For class $c$ at timestep $t$,
  $ z_c^"mean"(t) = 1/(t + 1) sum_(tau=0)^t v_c^"pre"(tau). $
  The final class is $arg max_c z_c^"mean"(t_"final")$.

  *Output and investigation link.* This produces the native baseline and `measured_z_mean.svg` for Investigation 1.

  === 4.3 Cumulative spike count

  *Mathematical steps.* Accumulate every counterfactual output spike:
  $ z_c^"cum"(t) = sum_(tau=0)^t s_c(tau). $

  *Output and investigation link.* This produces `measured_z_cumulative.svg`, supplies the shared evidence vector for Investigation 2, and enters the screen in Investigation 3.

  === 4.4 Leaky spike count

  *Mathematical steps.* Retain a fraction $lambda$ of earlier evidence and add the current spike:
  $ z_c^"leak"(t) = lambda z_c^"leak"(t - 1) + s_c(t), quad 0 < lambda < 1. $
  The fixed leak timescale is #r.decoder.leak_tau_ms ms.

  *Output and investigation link.* This produces `measured_z_leaky.svg` and the leaky screen column for Investigations 1 and 3.

  === 4.5 Sliding-window spike count

  *Mathematical steps.* For a window of $W$ timesteps,
  $ z_c^"win"(t) = sum_(tau=max(0, t - W + 1))^t s_c(tau). $
  Here $W$ corresponds to #r.decoder.window_ms ms.

  *Output and investigation link.* This produces `measured_z_window.svg` and the window screen column for Investigations 1 and 3.

  === 4.6 Cycle or matched-bin voting

  *Mathematical steps.* Count class spikes $n_(j,k)$ inside bin $k$, assign the bin to its largest count, and accumulate votes:
  $ z_c^"vote"(K) = sum_(k=1)^K 1_(c = arg max_j n_(j,k)). $
  PING bins use detected cycles; COBA bins use the same durations.

  *Output and investigation link.* This produces `measured_z_vote.svg` and the vote screen column for Investigations 1 and 3.

  === 4.7 Ordinary softmax

  *Mathematical steps.* Apply one normalized exponential to shared cumulative evidence:
  $ p_c(t) = frac(exp(z_c^"cum"(t)), sum_j exp(z_j^"cum"(t))). $

  *Output and investigation link.* This produces `measured_p_softmax.svg` and the softmax screen column for Investigations 2 and 3. Its argmax equals cumulative count.

  === 4.8 Temperature-softened softmax

  *Mathematical steps.* Divide the same evidence by fixed temperature $T=#r.decoder.softmax_temperature$ before normalization:
  $ p_c(t; T) = frac(exp(z_c^"cum"(t) / T), sum_j exp(z_j^"cum"(t) / T)). $

  *Output and investigation link.* This produces `measured_p_softened.svg` and the softened screen column for Investigations 2 and 3. Positive temperature preserves the argmax.

  === 4.9 Independent sigmoid

  *Mathematical steps.* Transform each cumulative count independently:
  $ p_c^"sig"(t) = 1 / (1 + exp(-z_c^"cum"(t))). $
  The scores need not sum to one; numerical ties use the implementation's default argmax.

  *Output and investigation link.* This produces `measured_p_sigmoid.svg` and the sigmoid screen column for Investigations 2 and 3.

  === 4.10 Balanced screen and transition accounting

  *Mathematical steps.* For $N=#r.screen.n$ labelled images with predictions $hat(y)_i$, accuracy is
  $ A = 1/N sum_(i=1)^N 1_(hat(y)_i = y_i). $
  Each alternative is cross-tabulated against the native decoder as correct-to-correct, correct-to-wrong, wrong-to-correct, or wrong-to-wrong. Per-class change subtracts native class accuracy from alternative class accuracy.

  *Output and investigation link.* The method produces `screen_predictions.npz`, `screen_accuracy.svg`, `screen_transitions.svg`, and `screen_classes.svg` for Investigation 3. The first ten official-test images per class are fixed without outcome inspection.

  == 5. Conclusion

  *Decision: escalate, with revision.* Under this frozen-weight intervention, PING output spikes preserve substantially more of the native decision than COBA output spikes, and the contrast survives the cheap balanced screen. Output mappings alone change apparent confidence but cannot repair incompatible evidence.

  The strongest unresolved rivals are output-rate mismatch, one-checkpoint dependence, fixed decoder timescales, and the deterministic convenience sample. There are no sampling intervals, independent training seeds, or held-out choices of leak, window, and temperature. The next artifact must be a new prospective `ExpStudyPlan` that freezes a multi-seed full-test estimand, controls decoder timescales, measures rate compatibility, and tests whether cycle-aware evidence explains transfer beyond cumulative counts. This scout itself remains exploratory.
]
