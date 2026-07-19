#let meta = (
  title: "Extending matched SHD training to eighty epochs",
  date: "2026-07-19",
  description: "A one-seed validation-only test asks whether doubling the unchanged matched COBA and PING training duration raises PING accuracy beyond its forty-epoch baseline.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
  order: 8,
)

#let r = json("/artifacts/data/exp069/numbers.json")
#let c = r.cells.coba
#let p = r.cells.ping
#let gain = r.primary.ping_change_from_exp068_pp
#let gap = r.primary.ping_minus_coba_validation_pp

#let body = [
  == Abstract

  This one-seed exploratory experiment changes only matched training duration
  from forty to eighty epochs under the `exp068` recipe. The official SHD test
  remains sealed: all outcomes use the identical deterministic
  #r.split.development_train_count training / #r.split.validation_count validation
  development split. PING selects #calc.round(p.selected_validation_accuracy_pct,
  digits: 2)% validation accuracy, a #calc.round(gain, digits: 2)-percentage-point
  gain over its registered 43.50% baseline. This *#r.primary.interpretation* and
  meets the predeclared +#r.primary.registered_threshold_pp point threshold.
  The result supports longer training for this recipe, not a population-level
  architecture claim.

  == Methods

  === The intervention is training duration alone

  Both cells use seed #r.seed, #r.config.n_hidden excitatory and
  #r.config.n_inhibitory inhibitory cells, fixed Dale-constrained recurrence,
  input-weight mean #r.config.input_weight_mean, 95% input sparsity, a
  membrane-mean readout scaled by #r.config.readout_scale, Adam learning rate
  #r.config.learning_rate, batch size #r.config.batch_size, a
  #r.config.dt_ms ms timestep, and #r.config.t_ms ms utterance duration. COBA
  disables the inhibitory loop and uses no voltage-gradient dampening; PING
  enables the loop and uses dampening 1000. These are the unchanged exp068
  settings. Each cell trains for #r.config.epochs epochs instead of forty.

  The official SHD training file is split jointly by speaker and class into
  #r.split.development_train_count training and #r.split.validation_count
  validation utterances. Both cells use identical index hashes, which also
  match exp068. The official SHD test is neither staged nor loaded. Checkpoints
  maximise validation accuracy, breaking ties by lower validation
  cross-entropy and then earlier epoch.

  == Results

  === Longer training clears the registered PING threshold

  PING selects epoch #p.selected_epoch at
  #calc.round(p.selected_validation_accuracy_pct, digits: 2)% accuracy and
  #calc.round(p.selected_validation_loss, digits: 3) cross-entropy. Its gain
  from the exp068 baseline is #calc.round(gain, digits: 2) points, exceeding
  the registered +#r.primary.registered_threshold_pp point criterion. COBA
  selects epoch #c.selected_epoch at
  #calc.round(c.selected_validation_accuracy_pct, digits: 2)%, a
  #calc.round(r.primary.coba_change_from_exp068_pp, digits: 2)-point change
  from its registered 40.81% baseline. The contemporaneous PING–COBA gap is
  #calc.round(gap, digits: 2) points.

  #figure(
    image("/artifacts/data/exp069/validation_curves.svg", width: 100%,
      alt: "Matched COBA and PING training loss, validation loss, and validation accuracy over eighty epochs, with selected checkpoints marked."),
    caption: [Matched learning curves for the fixed eighty-epoch protocol. Red dashed traces and squares denote COBA; black solid traces and diamonds denote PING. Markers identify checkpoints selected solely from validation accuracy and the registered tie-breaks.],
  )

  === Both selected networks remain active

  At its selected checkpoint, COBA's validation excitatory rate is
  #calc.round(c.selected_validation_e_rate_hz, digits: 2) Hz. PING's selected
  validation E/I rates are
  #calc.round(p.selected_validation_e_rate_hz, digits: 2) /
  #calc.round(p.selected_validation_i_rate_hz, digits: 2) Hz. Neither cell
  records a skipped update or non-finite forward batch.

  #figure(
    image("/artifacts/data/exp069/activity_curves.svg", width: 100%,
      alt: "COBA and PING validation excitatory and inhibitory firing rates over eighty epochs."),
    caption: [Validation population activity across training. COBA's disabled inhibitory loop remains at zero; PING sustains active excitatory and inhibitory populations.],
  )

  === Matched validation utterances expose different dynamics

  #figure(
    image("/artifacts/data/exp069/matched_rasters.png", width: 100%,
      alt: "Input, excitatory, and inhibitory rasters for matched validation utterances in COBA and PING."),
    caption: [Input, E, and I spike rasters for fixed validation positions #r.config.raster_positions.map(str).join(", "). Each column uses byte-identical binned input in both cells, with the same time axis and population order.],
  )

  == Verdict and limits

  The registered primary criterion is met: extending the unchanged recipe to
  eighty epochs raises PING validation accuracy by #calc.round(gain, digits: 2)
  points, beyond the required +#r.primary.registered_threshold_pp points.
  PING also remains #calc.round(gap, digits: 2) points ahead of contemporaneous
  COBA. Both selected epochs occur late, so the learning curves—not another
  unregistered extension—are the appropriate evidence for judging remaining
  headroom.

  This is one seed on one development split. It establishes a useful
  exploratory training-duration signal but supplies no uncertainty interval
  and does not justify a general architecture claim. The official SHD test was
  not accessed. Exact RunPod spend was
  #calc.round(r.runpod.total_spend_usd, digits: 3) USD, with
  #r.runpod.active_pods_after_collection active pods after collection.
]
