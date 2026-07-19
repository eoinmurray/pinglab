#let meta = (
  title: "Matched COBA and PING on a full-development SHD split",
  date: "2026-07-19",
  description: "A one-seed exploratory comparison trains matched Dale-constrained COBA and PING networks on a speaker-and-class-stratified SHD development split, freezes checkpoints on validation data, and evaluates the official test once.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
  order: 6,
)

#let r = json("/artifacts/data/exp068/numbers.json")
#let c = r.cells.coba
#let p = r.cells.ping
#let delta = r.primary.delta_accuracy_pp

#let body = [
  == Abstract

  This one-seed exploratory experiment asks whether PING's modest `exp066`
  point advantage becomes clearer with substantially more Spiking Heidelberg
  Digits (SHD) development data and leakage-resistant model selection. COBA and
  PING train on the same #r.split.development_train_count utterances, select
  checkpoints using the same validation split of #r.split.validation_count
  utterances, and are evaluated once on all #r.split.official_test_count official-test
  utterances after both checkpoints freeze. PING minus COBA official-test
  accuracy is #calc.round(delta, digits: 2) percentage points, which meets the
  registered interpretation band *#r.primary.interpretation*. This single seed
  identifies an exploratory direction; it does not estimate an expected
  architecture effect.

  == Methods

  === Data separation prevents test-driven selection

  The official SHD training split of #r.split.official_train_count utterances is
  partitioned at seed #r.seed into #r.split.development_train_count training and
  #r.split.validation_count validation utterances. Stratification is joint by
  speaker and class. Both architectures use the same index hashes. The complete
  official test is unavailable during training and checkpoint selection. Only
  after both checkpoint identities are frozen is each checkpoint evaluated once
  on the complete official test. Speakers 4 and 5, absent from official
  training, define the registered unseen-speaker diagnostic.

  === The two cells differ only in the inhibitory recipe

  Both networks have #r.config.n_hidden excitatory and
  #r.config.n_inhibitory inhibitory cells, fixed Dale-constrained recurrence,
  input-weight mean #r.config.input_weight_mean, 95% input sparsity, a
  membrane-mean readout scaled by #r.config.readout_scale, Adam learning rate
  #r.config.learning_rate, batch size #r.config.batch_size, a
  #r.config.dt_ms ms timestep, and #r.config.t_ms ms utterance duration. COBA
  disables the inhibitory loop and uses no voltage-gradient dampening; PING
  enables the loop and uses dampening 1000. Neither cell uses a firing-rate
  regulariser.

  Each cell trains for #r.config.epochs epochs. The selected checkpoint maximises
  validation accuracy, breaking ties with lower validation cross-entropy and
  then the earlier epoch. The experiment-side selector preserves the required
  epoch state without modifying the SNN tool.

  == Results

  === Validation selects checkpoints without official-test feedback

  COBA selects epoch #c.selected_epoch at
  #calc.round(c.selected_validation_accuracy_pct, digits: 2)% validation
  accuracy; PING selects epoch #p.selected_epoch at
  #calc.round(p.selected_validation_accuracy_pct, digits: 2)%.

  #figure(
    image("/artifacts/data/exp068/validation_curves.svg", width: 100%,
      alt: "Matched COBA and PING training loss, validation loss, and validation accuracy over forty epochs, with selected checkpoints marked."),
    caption: [Training and validation learning curves under the locked split. Red dashed traces and square markers denote COBA; black solid traces and diamond markers denote PING. The marked epoch for each cell is selected solely by validation accuracy, validation cross-entropy, and the registered earlier-epoch tie-break.],
  )

  === The official test determines the exploratory accuracy verdict

  COBA obtains #calc.round(c.official_test.accuracy_pct, digits: 2)% overall
  official-test accuracy and PING obtains
  #calc.round(p.official_test.accuracy_pct, digits: 2)%. Their registered
  difference is #calc.round(delta, digits: 2) percentage points. The macro-class,
  seen-speaker, and unseen-speaker panels are diagnostics rather than additional
  primary outcomes.

  #figure(
    image("/artifacts/data/exp068/official_test_diagnostics.svg", width: 100%,
      alt: "COBA and PING official-test overall, macro-class, seen-speaker, and unseen-speaker accuracy, beside overall accuracy plotted against excitatory firing rate."),
    caption: [Official-test results after both validation checkpoints were frozen. Left: overall accuracy, macro class accuracy, and accuracy for speakers seen versus unseen during development training. Right: the same overall accuracy plotted directly against official-test excitatory firing rate; no post-hoc composite score is used.],
  )

  === PING uses less excitatory activity under the matched recipe

  On the official test, COBA's excitatory population fires at
  #calc.round(c.official_test.e_rate_hz, digits: 2) Hz. PING fires at
  #calc.round(p.official_test.e_rate_hz, digits: 2) Hz excitatory and
  #calc.round(p.official_test.i_rate_hz, digits: 2) Hz inhibitory. Validation
  activity is shown across all epochs so activity drift is visible rather than
  reduced to a single selected point.

  #figure(
    image("/artifacts/data/exp068/activity_curves.svg", width: 100%,
      alt: "COBA and PING validation excitatory and inhibitory firing rates over forty epochs."),
    caption: [Population firing rates on the validation split during training. COBA's disabled inhibitory loop stays at zero. PING sustains active E and I populations; the accuracy comparison remains matched on every non-architectural setting.],
  )

  === Fixed utterances expose matched population dynamics

  #figure(
    image("/artifacts/data/exp068/matched_rasters.png", width: 100%,
      alt: "Input, excitatory, and inhibitory rasters for matched official-test utterances in COBA and PING."),
    caption: [Matched spike rasters for official-test positions #r.config.raster_positions.map(str).join(", "), fixed before predictions were inspected. Each column uses byte-identical binned cochlear input in COBA and PING. Rows show input, excitatory, and inhibitory spikes on the same time axis and fixed population order.],
  )

  == Verdict and limits

  The registered exploratory interpretation is *#r.primary.interpretation*:
  PING minus COBA accuracy is #calc.round(delta, digits: 2) percentage points
  against the predeclared +2-point promising-signal threshold. This conclusion
  applies to one seed and one matched recipe. It cannot support uncertainty
  intervals or a population-level architecture claim. A later confirmatory
  mandate is warranted only if this exploratory result supplies a claim worth
  defending.

  Both cells used identical partition hashes, official-test evaluation occurred
  after the paired checkpoint freeze, and neither cell recorded a skipped update
  or non-finite forward batch. Total RunPod spend was
  #calc.round(r.runpod.total_spend_usd, digits: 3) USD, with
  #r.runpod.active_pods_after_collection active pods after collection.
]
