#let meta = (
  title: "A preregistered short-run ladder fails to accelerate matched SHD learning",
  date: "2026-07-19",
  description: "Three one-seed, validation-only interventions test temporal resolution, shared input drive, and learning rate in matched Dale-constrained COBA and PING networks.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
  order: 9,
)

#let base = json("/artifacts/data/exp069/numbers.json")
#let summary = json("/artifacts/data/exp070/numbers.json")
#let temporal = json("/artifacts/data/exp070/raw/temporal_2ms/attempt_decision.json")
#let input = json("/artifacts/data/exp070/raw/input_scale_1p2/attempt_decision.json")
#let rate = json("/artifacts/data/exp070/raw/learning_rate_1e3/attempt_decision.json")

#let result-row(label, result, cell) = {
  let x = result.cells.at(cell)
  ([#label], [#upper(cell)],
   [#calc.round(x.validation_accuracy_pct, digits: 2)%],
   [#calc.round(x.accuracy_gain_pp, digits: 2) pp],
   [#calc.round(x.validation_cross_entropy, digits: 3)],
   [#calc.round(x.cross_entropy_change, digits: 3)])
}

#let body = [
  == Abstract

  This one-seed exploratory experiment asks whether any of three locked,
  ordered interventions makes both matched Dale-constrained COBA and PING
  networks learn SHD faster. Each candidate first passes a local 128/128
  two-epoch smoke, then trains for five epochs on the same 7,340-utterance
  development-training partition and 816-utterance held-out validation set as
  exp069. The official test set remains sealed. No candidate clears the
  preregistered requirement of at least +3 percentage points and non-worse
  cross-entropy in both cells. The ladder therefore terminates without a
  forty-epoch run.

  == Methods

  === Matched comparison and locked ladder

  Every cell uses seed 42, 256 excitatory and 64 inhibitory slots, fixed
  Dale-constrained recurrence, batch size 32, a membrane-mean readout scaled by
  225, and identical input and readout dimensions. COBA disables the
  inhibitory loop and uses voltage-gradient dampening 1—the engine's
  no-dampening value. PING enables the loop and uses dampening 1000. The
  development split, sample order, checkpoint rule, raster positions, and all
  unnamed settings remain matched.

  The candidates were evaluated in this preregistered order, stopping at the
  first promotion: (1) change only the timestep from 1 ms to 2 ms; (2) restore
  1 ms and change only shared input-weight mean from 0.9 to 1.2; (3) restore
  input mean 0.9 and change only matched Adam learning rate from 0.0004 to
  0.001. Candidates were never combined. Promotion required both cells at
  epoch five to gain at least three accuracy points over their own exp069
  epoch-five result, with non-worse validation cross-entropy, finite training,
  no skipped update, and active non-saturated populations.

  == Results

  === Every candidate is killed

  #table(
    columns: (1.25fr, auto, auto, auto, auto, auto),
    table.header([*Candidate*], [*Cell*], [*Accuracy*], [*Accuracy change*], [*CE*], [*CE change*]),
    ..result-row([2 ms], temporal, "coba"),
    ..result-row([2 ms], temporal, "ping"),
    ..result-row([input mean 1.2], input, "coba"),
    ..result-row([input mean 1.2], input, "ping"),
    ..result-row([learning rate 0.001], rate, "coba"),
    ..result-row([learning rate 0.001], rate, "ping"),
  )

  Temporal coarsening harms both cells. Stronger input drive leaves COBA
  accuracy unchanged and improves PING by only 1.10 points, while both losses
  are slightly worse. The higher learning rate improves PING loss and adds
  1.59 points, but COBA loses 1.84 points. Because the primary gate is joint,
  none is promoted.

  #figure(
    grid(
      columns: (1fr, 1fr), gutter: 10pt,
      image("/artifacts/data/exp070/attempts/temporal_2ms/short_validation_curves.svg",
        width: 100%, alt: "Validation curves for the 2 ms candidate."),
      image("/artifacts/data/exp070/attempts/input_scale_1p2/short_validation_curves.svg",
        width: 100%, alt: "Validation curves for the input-mean-1.2 candidate."),
      image("/artifacts/data/exp070/attempts/learning_rate_1e3/short_validation_curves.svg",
        width: 100%, alt: "Validation curves for the learning-rate-0.001 candidate."),
    ),
    caption: [Five-epoch learning curves in registered order: 2 ms, input mean 1.2, and learning rate 0.001. The decision uses epoch five, not the visually best intermediate checkpoint.],
  )

  === Failures are scientific, not numerical

  All six cells finish with finite losses and gradients, zero skipped updates,
  and zero non-finite forward batches. COBA excitatory rates span 14.22–30.64
  Hz across the three epoch-five endpoints. PING remains active at 6.61–10.16
  Hz E and 29.11–45.26 Hz I. These rates are far below the registered
  saturation thresholds. The failures therefore concern learning efficacy,
  not silence, saturation, or broken plumbing.

  #figure(
    grid(
      columns: (1fr, 1fr), gutter: 10pt,
      image("/artifacts/data/exp070/attempts/temporal_2ms/short_activity_curves.svg",
        width: 100%, alt: "Population firing rates for the 2 ms candidate."),
      image("/artifacts/data/exp070/attempts/input_scale_1p2/short_activity_curves.svg",
        width: 100%, alt: "Population firing rates for the input-mean-1.2 candidate."),
      image("/artifacts/data/exp070/attempts/learning_rate_1e3/short_activity_curves.svg",
        width: 100%, alt: "Population firing rates for the learning-rate-0.001 candidate."),
    ),
    caption: [Held-out population firing rates for all three candidates. Stronger input and higher learning rate raise activity without approaching saturation.],
  )

  #figure(
    image("/artifacts/data/exp070/attempts/learning_rate_1e3/matched_rasters.png",
      width: 100%, alt: "Matched input, excitatory, and inhibitory spike rasters for COBA and PING under the learning-rate candidate."),
    caption: [Matched validation rasters for the final candidate. Each COBA/PING column uses byte-identical binned input and fixed validation positions. The corresponding rasters for both earlier candidates are retained alongside their curves.],
  )

  == Verdict and limits

  The preregistered exploratory hypothesis is not supported. None of the three
  isolated interventions improves both matched cells by at least three points
  at epoch five without worsening loss, so the protocol forbids the planned
  forty-epoch extension. The higher learning rate is the most promising PING
  signal, but its opposing COBA result means it is not a shared solution and
  cannot be promoted post hoc.

  This is one seed and one held-out development split, deliberately appropriate
  for exploratory triage rather than a defended population claim. The official
  SHD test was neither staged nor loaded. Exact provider billing was
  #calc.round(summary.runpod.attempts.temporal_2ms.total_spend_usd, digits: 6)
  USD for candidate 1,
  #calc.round(summary.runpod.attempts.input_scale_1p2.total_spend_usd, digits: 6)
  USD for candidate 2, and
  #calc.round(summary.runpod.attempts.learning_rate_1e3.total_spend_usd, digits: 6)
  USD for candidate 3: #calc.round(summary.runpod.total_spend_usd, digits: 6)
  USD total, with #summary.runpod.active_pods_after_collection active pods after
  collection.
]
