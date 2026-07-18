#let meta = (
  title: "Matched Dale-constrained COBA and PING learn a small SHD task",
  date: "2026-07-18",
  description: "A registered one-seed feasibility pilot compares matched COBA and PING training on 1,000 Spiking Heidelberg Digits utterances and records learning curves, firing rates, and shared-input population rasters.",
  collection: "spiking-heidelberg-digits",
  status: "final",
  order: 3,
)

#let r = json("/artifacts/data/exp066/numbers.json")
#let c = r.cells.coba
#let p = r.cells.ping
#let rc = r.raster_evidence.cells.coba
#let rp = r.raster_evidence.cells.ping

#let body = [
  == Abstract

  Both matched Dale-constrained conductance-based networks learn the small
  Spiking Heidelberg Digits (SHD) task above the registered threshold. COBA
  reaches a best held-out accuracy of #c.best_accuracy_pct%, and PING reaches
  #p.best_accuracy_pct%, against the #(r.chance_accuracy_pct)% chance floor and
  the pre-registered #(r.success_criterion_pct)% success criterion. Neither run
  records a skipped update or non-finite forward batch. Fixed test utterances
  replayed through the trained input and recurrent weights produce directly
  comparable input, excitatory, and inhibitory rasters. This one-seed result
  establishes feasibility; it does not compare the architectures' expected
  accuracy or efficiency.

  == Methods

  SHD contains event-based recordings of spoken digits across
  #(r.config.n_input) cochlear channels and #(r.config.n_classes)
  classes. The registered protocol has three ordered stages:

  + *Gate with a local smoke test.* Train seed #(r.seed) on 128 training
    utterances for two epochs. Require finite loss, active hidden populations,
    saved checkpoints, and no skipped updates before allocating cloud compute.
    The original shared input-weight mean passes, so the single allowed shared
    adjustment is not used.
  + *Train the pilot.* Train each cell for #(r.config.epochs) epochs on the same
    #(r.config.max_samples)-utterance subset. Both cells have
    #(r.config.n_hidden) excitatory and #(r.config.n_inhibitory) inhibitory
    cells, Dale's law, fixed recurrence, #(r.config.input_sparsity * 100)% input
    sparsity, input-weight mean #(r.config.input_weight_mean), membrane-mean
    readout scaled by #(r.config.readout_scale), Adam learning rate
    #(r.config.learning_rate), batch size #(r.config.batch_size), integration
    timestep #(r.config.dt_ms) ms, and duration #(r.config.t_ms) ms. COBA disables
    the inhibitory loop and uses voltage-gradient dampening
    #(r.config.recipes.coba.v_grad_dampen); PING enables the fixed loop and uses
    dampening #(r.config.recipes.ping.v_grad_dampen).
  + *Record matched dynamics.* Select official test positions
    #(r.preselected_test_positions.map(str).join(", ")) before inspecting
    predictions, bin their cochlear events on the same integration grid, and
    replay the identical spike tensors through both trained input and recurrent
    weight sets. The replay omits the classifier output tensor because the
    batch-raster interface constructs a ten-output probe head; the omitted
    readout does not feed back into the hidden E/I dynamics plotted here.

  Every published raw file is listed with its SHA-256 digest in
  `raw_sha256.json`. The artifact manifest and reproducer record the publishing
  commit and exact runner invocation.

  == Results

  === Both cells cross the registered accuracy criterion

  COBA peaks at #c.best_accuracy_pct% in epoch #c.best_epoch and ends at
  #c.final_accuracy_pct%. PING peaks at #p.best_accuracy_pct% in epoch
  #p.best_epoch and ends at #p.final_accuracy_pct%. Both exceed the registered
  criterion, so the hypothesis survives this feasibility pilot.

  #figure(
    image("/artifacts/data/exp066/learning_curves.svg", width: 100%,
      alt: "COBA and PING train and test loss curves beside held-out accuracy curves over twenty epochs."),
    caption: [Matched learning curves for the registered SHD pilot. The left panel shows cross-entropy loss against epoch for training and held-out test subsets; the right panel shows held-out classification accuracy in percent. Red dashed traces and square markers identify COBA; black solid traces and diamond markers identify PING, with line style and marker shape retaining the distinction in grayscale. Horizontal lines mark #(r.chance_accuracy_pct)% chance and the #(r.success_criterion_pct)% registered criterion. COBA reaches #c.best_accuracy_pct% and PING #p.best_accuracy_pct%; both train above the criterion.],
  )

  === Activity stays finite and architecture-appropriate

  At the final training epoch, COBA's excitatory population fires at
  #calc.round(c.final_e_rate_hz, digits: 1) Hz and its disabled inhibitory
  population remains at #c.final_i_rate_hz Hz. PING ends at
  #calc.round(p.final_e_rate_hz, digits: 1) Hz excitatory and
  #calc.round(p.final_i_rate_hz, digits: 1) Hz inhibitory activity. Neither
  cell is silent or saturated.

  #figure(
    image("/artifacts/data/exp066/firing_rates.svg", width: 100%,
      alt: "Excitatory and inhibitory firing rates for COBA and PING over twenty training epochs."),
    caption: [Hidden-population firing rates during training. The horizontal axis is epoch; the vertical axes are excitatory rate in hertz (left) and inhibitory rate in hertz (right). Red dashed traces denote COBA and black solid traces PING. COBA's disabled loop gives zero inhibitory activity while its excitatory population remains active. PING sustains both populations without a non-finite batch or skipped update.],
  )

  === Identical utterances expose comparable E/I dynamics

  The matched replay gives COBA #calc.round(rc.e_rate_hz, digits: 1) Hz
  excitatory activity and no inhibitory activity. PING gives
  #calc.round(rp.e_rate_hz, digits: 1) Hz excitatory and
  #calc.round(rp.i_rate_hz, digits: 1) Hz inhibitory activity. The shared input
  rows are identical within each test-position column by construction.

  #figure(
    image("/artifacts/data/exp066/matched_rasters.png", width: 100%,
      alt: "Six rows of input, excitatory, and inhibitory spike rasters for COBA and PING on the same three SHD test utterances."),
    caption: [Matched spike rasters for the preselected official SHD test positions #(r.preselected_test_positions.map(str).join(", ")). Columns are utterances and share a time axis in milliseconds. Within each architecture, rows show the 700-channel cochlear input, #(r.config.n_hidden) excitatory cells (E), and #(r.config.n_inhibitory) inhibitory cells (I), with fixed population order. COBA and PING receive byte-identical binned input spikes. COBA has active E cells and a silent disabled I loop; PING produces active E and I populations on the same utterances.],
  )

  == Verdict and limits

  The registered feasibility criterion is met: both matched networks train above
  #(r.success_criterion_pct)% and yield comparable population recordings. This
  is one pre-registered seed on a small subset. It cannot establish a reliable
  accuracy difference, a PING advantage, or generalisation to full-scale SHD.
  Two cloud evidence-capture attempts failed after valid training because the
  single-sample replay path could not emit SHD snapshots; the final rasters use
  the existing arbitrary-input batch interface locally. Total RunPod spend is
  #calc.round(r.runpod.total_spend_usd, digits: 2) USD, and
  #(r.runpod.active_pods_after_collection) pods remain active.
]
