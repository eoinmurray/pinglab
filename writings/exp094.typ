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
  == 1. Scout identity

  *Artifact:* `ExpScout` · *execution status:* complete · *evidence class:* exploratory

  *Question.* When the learned network and output projection are frozen, how much of the native MNIST decision survives changes to the temporal evidence function $z$, and what does a later output mapping $p$ change without changing the winner?

  *Decision sought.* Decide whether decoder transfer is immediately uninformative, needs revision, or warrants a new prospective multi-seed full-test study.

  == 2. Abstract

  A decoder converts neural activity into a class decision. Here the learned output projection remains frozen while two later choices change: $z$ decides what temporal activity counts as evidence, and $p$ decides how that evidence is displayed as relative confidence.

  One preselected MNIST digit and one Poisson spike tensor were replayed through the canonical seed-42 COBA and PING final-epoch checkpoints represented in the latest gold-star publication view. These are the full-MNIST reference networks, not exp022's 10%-MNIST activity-frontier (`off`) condition. Their trained decoder is `mem-mean`: a non-spiking output membrane whose temporal mean supplies the class logits. To compare spike-based alternatives without retraining, the same frozen hidden activity and learned output weights also drive a counterfactual output-LIF `spike-count` decoder at inference time. Every spike-based $z$ reads that one shared intervention trajectory, and every $p$ reads the same cumulative counts. The comparison therefore explains decoder choices; it does not estimate the accuracy each alternative would achieve if trained end to end. Each subsection pairs a blue-grey qualitative mechanism diagram with a structurally matched red-black measured mirror.

  #figure(
    image("/artifacts/data/exp094/decoder_pipeline.svg", width: 100%, alt: "Frozen COBA and PING activity passing through temporal evidence, output mapping, and decision stages."),
    caption: [Decoder decomposition. Frozen COBA and PING networks produce output spikes and voltages. A temporal evidence function $z$ compresses that activity, an output mapping $p$ displays the resulting competition, and the decision selects the largest final score.],
  )

  == 3. Frozen scout plan

  #block(inset: 10pt, fill: rgb("eef4f8"), radius: 3pt)[
    *Record status.* No separately frozen `ExpScoutPlan` predates this completed run. The contract below is reconstructed from the preserved runner, manifest, outputs, and outcome-blind selections. It records what was actually constrained, but it must not be mistaken for prospective registration. That missing pre-run freeze is a validity limitation of this scout.
  ]

  === 3.1 Scientific frame

  The mechanism under test is decoder compatibility. A readout trained to use temporally averaged subthreshold voltage may not transfer to output spikes even when hidden activity and learned weights are unchanged. PING could preserve more transferable spike evidence because its rhythmic packets separate class competition in time; alternatively, any apparent advantage could be a selected-trial accident, a rate mismatch in the counterfactual output neurons, or a decoder-timescale artefact.

  === 3.2 Investigation units and gates

  + *IU1 — mechanism mirror.* Replay one outcome-blind official-test image through canonical seed-42 COBA and PING checkpoints. Pair each qualitative blue-grey decoder diagram with a structurally matched red-black measured trajectory. *Gate:* continue only if the shared-input replay and decoder interventions are finite and comparable.

  + *IU2 — temporal evidence transfer.* Compare native running mean voltage with cumulative, leaky, 25 ms windowed, and cycle-or-matched-bin spike evidence. *Gate:* stop if the alternatives merely reproduce the native winner in both networks without exposing a meaningful difference.

  + *IU3 — output mapping control.* Pass one unchanged cumulative-count vector to ordinary softmax, temperature-4 softmax, and independent sigmoid. *Gate:* treat changes in displayed confidence separately from changes in accumulated evidence or argmax.

  + *IU4 — cheap balanced screen.* Run the first ten official-test images from each class, selected without inspecting outcomes, through both frozen networks. *Escalation gate:* a decoder-transfer contrast must survive beyond the illustrative image without being presented as a population estimate.

  === 3.3 Controls, budget, and completion rule

  The networks, hidden trajectories, learned output weights, inputs, and random encodings are paired wherever the comparison permits. COBA receives duration-matched bins where PING receives detected-cycle bins. All three $p$ mappings receive the identical cumulative-count tensor. The local budget is one illustrative replay plus 100 balanced screening images, with no retraining, hyperparameter search, seed sweep, or full-test evaluation. Completion requires recorded provenance, all matched visual pairs, the balanced screen, explicit limitations, and one `stop`, `revise`, or `escalate` decision.

  == 4. Implementation and provenance

  The scout used final-epoch `mem-mean` checkpoints from publication run #raw(r.training_source.publication_run_id), upstream campaign #raw(r.training_source.upstream_campaign_id): #raw(r.training_source.cells.coba) and #raw(r.training_source.cells.ping). The illustrative input is official MNIST test index #r.selection.official_test_index, label #r.selection.label, chosen without inspecting its outcome; its shared Poisson seed is #r.shared_input.seed. The balanced screen uses Poisson seed #r.screen.design.poisson_seed and batches of #r.screen.design.batch_size images.

  The native non-spiking output provides mean-voltage evidence. A single counterfactual output-LIF trajectory, driven by the same frozen hidden spikes and learned output weights, supplies every spike-based alternative. The leak time constant and window width are both 25 ms. The measured assets, compact arrays, `numbers.json`, runner, checkpoint hashes, dirty-source patch, and run manifest are preserved under `artifacts/data/exp094/`.

  *Deviation from an ideal scout.* The plan was not frozen before execution. The illustrative mechanism view uses one seed and one image; the screen adds images but not training seeds. Cycle detection is available only for PING, so COBA uses duration-matched bins rather than a biologically homologous cycle definition.

  == 5. IU1–IU2: temporal evidence functions z

  Each subsection changes only $z$. In the measured plots, red denotes the true class, black its strongest competitor, and time runs horizontally; the diagrams use blue and grey to remain visibly distinct. Mean voltage uses the native non-spiking readout. The four spike-based alternatives use the single counterfactual output-LIF trajectory driven by the same frozen hidden spikes and output weights.

  === 5.1 Mean output voltage

  $ z_c^"mean"(t) = 1/(t + 1) sum_(tau=0)^t v_c^"pre"(tau). $

  Here $c$ identifies a class, $t$ is the current timestep, $tau$ indexes earlier timesteps, and $v_c^"pre"(tau)$ is pre-reset output voltage. Mean voltage retains subthreshold evidence but removes temporal order. It is the decoder used to train the canonical exp022 COBA and PING checkpoints.

  #figure([
    #image("/artifacts/data/exp094/z_mean.svg", width: 100%, alt: "Illustrative output voltages becoming mean-voltage evidence.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_mean.svg", width: 100%, alt: "Measured running mean-voltage evidence for frozen COBA and PING trajectories.")
  ], caption: [Mean-voltage decoder. The diagram shows pre-reset voltages entering a running temporal mean; the measured mirror shows the resulting frozen COBA and PING trajectories.])

  The native decoder selects the true class in both networks. COBA settles rapidly, whereas PING retains a visible early competition before the true-class mean becomes consistently larger.

  === 5.2 Cumulative spike count

  $ z_c^"cum"(t) = sum_(tau=0)^t s_c(tau). $

  Here $s_c(tau)$ is one when output neuron $c$ spikes at timestep $tau$ and zero otherwise. Every spike remains evidence until reset. Timing and order disappear; only accumulated count survives.

  #figure([
    #image("/artifacts/data/exp094/z_cumulative.svg", width: 100%, alt: "Illustrative output spikes becoming cumulative staircase evidence.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_cumulative.svg", width: 100%, alt: "Measured cumulative output-spike evidence for frozen COBA and PING trajectories.")
  ], caption: [Cumulative-count decoder. Each output spike increments one class trace and no evidence decays; the staircase exposes early leads and later reversals.])

  The counterfactual spike decoder changes the outcome. PING eventually gives the true class the largest cumulative count, while COBA does not. Frozen weights therefore do not make a new temporal evidence rule equivalent to the rule used during training.

  === 5.3 Leaky spike count

  $ z_c^"leak"(t) = lambda z_c^"leak"(t - 1) + s_c(t), quad 0 < lambda < 1. $

  Here $lambda$ is the fraction of previous evidence retained for one timestep. Spikes add evidence; between spikes it decays. The decoder can follow changing input, but its meaning depends on the chosen memory timescale.

  #figure([
    #image("/artifacts/data/exp094/z_leaky.svg", width: 100%, alt: "Illustrative output spikes becoming decaying leaky evidence.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_leaky.svg", width: 100%, alt: "Measured leaky spike evidence for frozen COBA and PING trajectories.")
  ], caption: [Leaky-count decoder. Spikes create upward steps and evidence decays smoothly between them, using the same leak factor for COBA and PING.])

  Leakage exposes recent differences in PING rather than preserving its entire history. COBA's much denser counterfactual output keeps both displayed classes near a steady ceiling, so forgetting adds little useful separation.

  === 5.4 Sliding-window spike count

  $ z_c^"win"(t) = sum_(tau=max(0, t - W + 1))^t s_c(tau). $

  Here $W$ is the number of timesteps in the window. A spike contributes while it remains inside the window and disappears at the trailing boundary. Memory is finite and explicit rather than gradual.

  #figure([
    #image("/artifacts/data/exp094/z_window.svg", width: 100%, alt: "Illustrative time window selecting which output spikes remain evidence.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_window.svg", width: 100%, alt: "Measured sliding-window spike evidence for frozen COBA and PING trajectories.")
  ], caption: [Sliding-window decoder. Evidence persists for 25 ms, then disappears abruptly when its spike leaves the window.])

  The finite window makes PING's changing local competition explicit. COBA again remains saturated: nearly every window is full, showing that a memory rule cannot rescue an output representation whose event rate is poorly matched to it.

  === 5.5 Cycle or matched-bin voting

  $ z_c^"vote"(K) = sum_(k=1)^K 1_(c = arg max_j n_(j,k)). $

  Here $K$ is the number of completed bins, $k$ identifies one bin, $j$ indexes classes, and $n_(j,k)$ is class $j$'s spike count inside bin $k$. The subscripted one is one when its condition is true and zero otherwise. PING bins follow measured cycles; COBA uses equal-duration controls. Each bin contributes one vote, discarding within-bin burst size.

  #figure([
    #image("/artifacts/data/exp094/z_vote.svg", width: 100%, alt: "Illustrative interval winners becoming cumulative class votes.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_z_vote.svg", width: 100%, alt: "Measured cumulative cycle or matched-bin votes for frozen COBA and PING trajectories.")
  ], caption: [Cycle or matched-bin voting. Each interval names one winner before adding one vote. PING votes follow detected cycles and COBA votes use duration-matched controls.])

  PING's true class wins more detected cycles and pulls away in cumulative votes. The matched COBA bins consistently favour another class. Cycle voting is therefore meaningful here because PING supplies distinct rhythmic evidence packets, not because voting itself guarantees a better decision.

  == 6. IU3: output mappings p

  #block(inset: 10pt, fill: rgb("f3f0e8"), radius: 3pt)[
    *Controlled comparison.* All three $p$ mappings receive the identical ten-class cumulative-count vector

    $ bold(z)^"cum"(t) = (z_0^"cum"(t), dots, z_9^"cum"(t)). $

    The counterfactual output-LIF trajectory is generated once from the shared frozen hidden drive, and $bold(z)^"cum"(t)$ is calculated once. It is then passed unchanged to ordinary softmax, temperature-softened softmax, and independent sigmoid. We do not select or recompute a different $z$ for any $p$. Therefore, every difference between these three displays is caused by $p$ alone.
  ]

  Here $bold(z)^"cum"(t)$ is the ten-element cumulative-count vector at time $t$, and its entries $z_0^"cum"(t)$ through $z_9^"cum"(t)$ correspond to MNIST classes zero through nine.

  === 6.1 Ordinary softmax

  *Fixed input to $p$:* the shared cumulative-count vector $bold(z)^"cum"(t)$.

  $ p_c(t) = frac(exp(z_c^"cum"(t)), sum_j exp(z_j^"cum"(t))). $

  Here $p_c(t)$ is class $c$'s displayed share, $exp$ is the exponential function, and $j$ ranges over all ten classes. Scores sum to one and preserve the winner. Count differences are exponentially sharpened.

  #figure([
    #image("/artifacts/data/exp094/p_softmax.svg", width: 100%, alt: "Illustrative cumulative counts becoming ordinary-softmax shares.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_p_softmax.svg", width: 100%, alt: "Measured ordinary softmax of cumulative counts for frozen COBA and PING trajectories.")
  ], caption: [Ordinary softmax applied to cumulative spike counts. The measured mirror shows how an accumulating count margin becomes apparent confidence.])

  Softmax hides absolute activity and displays only count margins. PING's early leader changes produce abrupt transfers of displayed share before the true class dominates; COBA's different final winner remains unchanged by the mapping.

  === 6.2 Temperature-softened softmax

  *Fixed input to $p$:* the same $bold(z)^"cum"(t)$ used by ordinary softmax.

  $ p_c(t; T) = frac(exp(z_c^"cum"(t) / T), sum_j exp(z_j^"cum"(t) / T)), quad T > 1. $

  Here $T$ is temperature. Dividing all counts by the same positive $T$ preserves the winner but reduces visual sharpening. This separates accumulated evidence from its apparent decisiveness.

  #figure([
    #image("/artifacts/data/exp094/p_softened.svg", width: 100%, alt: "Illustrative cumulative counts becoming temperature-softened shares.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_p_softened.svg", width: 100%, alt: "Measured temperature-softened softmax of the same cumulative counts.")
  ], caption: [Temperature-softened softmax applied to the same counts. It differs from ordinary softmax only in temperature, isolating the effect of $p$.])

  Temperature preserves every leader change and the final winner while making the competing PING classes easier to see. It changes how decisive the same evidence appears, not which evidence was accumulated.

  === 6.3 Independent sigmoid

  *Fixed input to $p$:* the same $bold(z)^"cum"(t)$ used by both softmax mappings.

  $ p_c^"sig"(t) = 1 / (1 + exp(-z_c^"cum"(t))). $

  Here $p_c^"sig"(t)$ is an independent bounded score. One class does not suppress the others and the ten scores need not sum to one. Since cumulative counts are non-negative and grow, several classes may approach one together. This exposes why independent sigmoid is poorly matched to mutually exclusive MNIST labels when applied directly to raw counts.

  #figure([
    #image("/artifacts/data/exp094/p_sigmoid.svg", width: 100%, alt: "Illustrative cumulative counts becoming independent sigmoid scores that rise together.")
    #v(6pt)
    #image("/artifacts/data/exp094/measured_p_sigmoid.svg", width: 100%, alt: "Measured independent sigmoid scores from the same cumulative counts.")
  ], caption: [Independent sigmoid applied to cumulative spike counts. It retains the same $z$ used by both softmax variants and reveals what is lost when classes no longer compete.])

  Because every cumulative count grows, all ten sigmoid scores saturate near one almost immediately. The mapping preserves neither useful relative separation nor a meaningful class decision; its failure follows from applying an independent bounded transform to non-negative accumulating counts.

  == 7. IU4: screening check across 100 images

  The single digit above suggests that PING exposes class information through spikes more robustly than COBA, but one example cannot establish that pattern. A fast screening run therefore selected the first ten official-test images from each MNIST class without inspecting their outcomes. The same fixed Poisson encoding was presented to both frozen networks. Inputs were processed in batches, and every alternative decoder reused its recorded activity. This 100-image convenience sample completed locally in under one minute. It is a check on whether the observation immediately collapses, not an estimate of test-set accuracy.

  #figure(
    image("/artifacts/data/exp094/screen_accuracy.svg", width: 100%, alt: "Accuracy matrix for eight decoder options across a balanced 100-image COBA and PING screening sample."),
    caption: [Decoder transfer screen. Each cell gives accuracy and, below it, the change from that network's native mean-voltage decoder. The divider separates temporal evidence choices from mappings of the shared cumulative-count evidence.],
  )

  The contrast survives the screen. COBA's native decoder classifies all #r.screen.n images correctly, but its spike-based evidence rules retain only #accuracy("coba", "vote")--#accuracy("coba", "cumulative")%. PING's native decoder reaches #accuracy("ping", "mean")%; cumulative count retains #accuracy("ping", "cumulative")%, leaky count #accuracy("ping", "leaky")%, cycle voting #accuracy("ping", "vote")%, and the #r.decoder.window_ms ms window #accuracy("ping", "window")%. This does not show that PING is decoder-independent. It shows that its spike events preserve substantially more of the trained decision than COBA's counterfactual output spikes do.

  Ordinary and softened softmax exactly match cumulative-count accuracy because both preserve its winning class. Independent sigmoid falls to #accuracy("coba", "sigmoid")% for both networks: the accumulating non-negative counts saturate together, and numerical ties default to class zero. Its apparent winner is therefore not a meaningful decoded decision.

  #figure(
    image("/artifacts/data/exp094/screen_transitions.svg", width: 100%, alt: "Stacked transition counts showing which native decisions each alternative decoder preserves, breaks, or repairs."),
    caption: [Decision transitions relative to native decoding. Green is preserved correctness; red is a native correct decision broken by the alternative. Yellow denotes a repair and grey a prediction that remains wrong.],
  )

  COBA's alternatives break #transitions("coba", "cumulative").correct_to_wrong -- #transitions("coba", "vote").correct_to_wrong decisions that its native decoder gets right. PING loses #transitions("ping", "cumulative").correct_to_wrong decisions under cumulative count, #transitions("ping", "leaky").correct_to_wrong under leaky count, #transitions("ping", "window").correct_to_wrong under the finite window, and #transitions("ping", "vote").correct_to_wrong under cycle voting. Cycle voting also repairs #transitions("ping", "vote").wrong_to_correct of PING's #(r.screen.n - r.screen.models.ping.mean.transitions.correct_to_correct) native errors. The distinction is therefore not merely a shift in aggregate accuracy: it is a large difference in how often changing the decoder destroys an already correct decision.

  #figure(
    image("/artifacts/data/exp094/screen_classes.svg", width: 100%, alt: "Per-digit accuracy changes from native decoding for COBA and PING across all decoder options."),
    caption: [Class-specific change from native accuracy, in percentage points. The screen is balanced at ten images per digit, so each image changes its class cell by ten points.],
  )

  COBA's loss spans almost every digit rather than reproducing only the original class-4 failure. Its spike decoders disproportionately return class zero, explaining why digit zero is the lone robust row. PING's losses are smaller and distributed across classes, with cumulative decoding weakest here for digits five and six. A full test-set, multi-seed evaluation is still required before treating decoder robustness as an architectural property.

  == 8. Provisional interpretation and uncertainty

  The scout supports one narrow observation: under this frozen-weight intervention, PING output spikes preserve substantially more of the native decision than COBA output spikes. It does not establish that PING is generally decoder-robust, that rhythmic packetization causes the difference, or that any alternative decoder would perform similarly if trained end to end. The largest unresolved rivals are output-rate mismatch, one-checkpoint dependence, fixed decoder timescales, and the convenience screen's small deterministic sample.

  There are no sampling intervals, independent training seeds, or held-out choices of leak, window, and temperature. Reported percentages are descriptive values for the fixed 100-image screen. Softmax equivalence is algebraic at the argmax; sigmoid failure here is specific to applying it directly to non-negative cumulative counts.

  == 9. Scout decision

  *Decision: escalate, with revision.* The contrast clears the cheap screening gate: PING retains #accuracy("ping", "window")--#accuracy("ping", "cumulative")% accuracy across the spike-based $z$ rules while COBA retains #accuracy("coba", "vote")--#accuracy("coba", "cumulative")%, relative to native scores of #accuracy("ping", "mean")% and #accuracy("coba", "mean")%. The next artifact must be a new prospective `ExpStudyPlan`, not a relabelled continuation of this scout. It should freeze a multi-seed full-test estimand, preselect decoder timescales or a nested validation rule, measure output-rate compatibility, and test whether cycle-aware evidence explains transfer beyond simpler cumulative counts. The absent prospective plan prevents this scout itself from serving as durable confirmatory evidence.
]
