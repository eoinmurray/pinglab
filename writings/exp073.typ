#let meta = (
  title: "Plastic recurrent excitation fails the local SHD gate",
  date: "2026-07-20",
  description: "A one-seed exploratory SHD experiment tries nonzero Dale-constrained trainable W_EE in matched COBA/PING networks, but the required local smoke gate kills the COBA cell before paid compute.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
  order: 12,
)

#let r = json("/artifacts/data/exp073/numbers.json")
#let maybe-pct(x) = {
  if type(x) == float or type(x) == int {
    [#calc.round(x, digits: 2)%]
  } else {
    [pending]
  }
}
#let maybe-hz(x) = {
  if type(x) == float or type(x) == int {
    [#calc.round(x, digits: 2) Hz]
  } else {
    [pending]
  }
}
#let attempt-figure(suffix) = "/artifacts/data/exp073/" + r.stage + "_" + r.attempt + "_" + suffix

#let body = [
  == Abstract

  This exploratory experiment tested the next biologically plausible path after
  exp071/exp072: keep the cumulative-potential readout, keep the historical
  one-seed validation-only SHD split, and make recurrent E→E excitation both
  nonzero and plastic while preserving Dale signs. The original matched
  COBA/PING design still failed at the required local smoke gate: PING remained
  finite and active, but the matched COBA cell repeatedly produced
  skipped/non-finite updates even after reducing the requested W_EE scale from
  `0.3 0.01` through `0.03 0.01`, `0.003 0.001`, and finally `0.0003 0.0001`.
  After that gate failure, an explicitly authorized PING-only continuation ran
  the surviving cell for the eight-epoch short stage on Modal. It reached
  #maybe-pct(r.cells.ping.selected_validation_accuracy_pct) held-out
  validation accuracy at epoch #r.cells.ping.selected_epoch, with active E/I
  firing and nonnegative trained W_EE. Because this is below, not meaningfully
  above, the historical ~70% PING regime from exp071/exp072, the forty-epoch
  promotion was not run.

  == Goal prompt

  #raw(read("/artifacts/data/exp073/goal.txt"), block: true)

  == Methods

  The intended comparison inherited the exp071/exp072 SHD recipe: seed #r.seed,
  #r.split.development_train_count development-training utterances,
  #r.split.validation_count held-out validation utterances, batch size
  #r.config.batch_size, Adam learning rate #r.config.learning_rate,
  #r.config.dt_ms ms simulation steps, #r.config.t_ms ms windows, matched input
  preprocessing, and the cumulative-potential readout with signed abstract
  readout weights and a trainable bias. The official SHD test set remained
  sealed and inaccessible to the runner.

  The only intended new scientific ingredient was `--trainable-w-ee` with
  nonzero `--w-ee`. Dale's law stayed enabled; `W_EI`, `W_IE`, and `W_II` stayed
  frozen. COBA kept the no-inhibitory-loop recipe and no voltage-gradient
  dampening; PING kept the registered inhibitory loop and dampening 1000.

  During implementation, the local CLI accepted `--w-ee` but training did not
  pass it into model construction. Commit `247e186` fixed that plumbing and
  added a focused regression test before this experiment ran. A later transport
  fix moved exp073's Modal worker function to module scope and aligned its
  Python image with the host, without changing any scientific parameter.

  == Local gate

  The final registered local scout used the same nonzero `W_EE`
  initialization now recorded in the PING-only configuration:
  `--w-ee #r.config.w_ee_init.at(0) #r.config.w_ee_init.at(1)`.
  The gate required both matched cells to be finite, active, non-saturated, and
  free of skipped or non-finite updates before any paid matched run. COBA failed
  that finite-update criterion while PING passed, so the locked matched
  COBA/PING result could not be promoted. The cloud result below is therefore
  the later authorized PING-only continuation, not a matched-network claim.

  #if r.result_status == "done" [
    == Authorized PING-only continuation

    After the matched local gate failed, the user authorized a design pivot:
    continue with the surviving PING cell only, without recasting it as a
    matched COBA/PING result. This continuation uses the same one seed,
    development-training/validation split, input preprocessing, cumulative
    potential readout, nonzero trainable `W_EE`, and PING dampening 1000.

    #table(
      columns: (auto, auto, auto, auto, auto),
      table.header([*Cell*], [*Selected acc.*], [*Epoch*], [*E rate*], [*I rate*]),
      [PING],
      [#maybe-pct(r.cells.ping.selected_validation_accuracy_pct)],
      [#r.cells.ping.selected_epoch],
      [#maybe-hz(r.cells.ping.final_validation_e_rate_hz)],
      [#maybe-hz(r.cells.ping.final_validation_i_rate_hz)],
    )

    #image(attempt-figure("ping_only_validation_curves.svg"), width: 100%)
    #v(4pt)
    #image(attempt-figure("ping_only_activity_curves.svg"), width: 100%)
    #v(4pt)
    #image(attempt-figure("ping_only_matched_rasters.png"), width: 100%)

    The PING-only `W_EE` diagnostics show selected mean
    #calc.round(r.parameters.ping.layers.at("1").selected.mean, digits: 8)
    and mean absolute movement from initialization
    #calc.round(r.parameters.ping.layers.at("1").selected_delta_from_initial.mean_abs, digits: 8).

    The short-stage run was dispatched on #r.runpod.provider with GPU
    #r.runpod.gpu. The timestamped compute ledger records
    #r.runpod.started_at to #r.runpod.finished_at UTC, estimated billable GPU
    time #calc.round(r.runpod.billable_gpu_seconds_estimate, digits: 1) s, and
    estimated spend #calc.round(r.runpod.total_spend_usd, digits: 3) USD. This
    is a timestamp estimate pending provider-billing reconciliation, not an
    exact invoice.

    == Timestamped result checkpoint

    At #r.runpod.finished_at UTC, the short-stage result was finite, active, and
    clean, but not accurate enough to justify promotion: selected validation
    accuracy was #maybe-pct(r.cells.ping.selected_validation_accuracy_pct),
    below the historical ~70% PING regime. The next action recorded by this
    article is therefore _stop before final40_, not _extend training_.
  ]

  == Conclusion

  #if r.result_status == "done" [
    The locked matched experiment still failed at the local gate: COBA could not
    be honestly promoted. The later PING-only run is therefore exploratory
    continuation evidence for the stable PING recipe, not matched COBA/PING
    evidence. Its #maybe-pct(r.cells.ping.selected_validation_accuracy_pct)
    selected validation accuracy does not move beyond the exp071/exp072
    historical ~70% PING regime, so the pre-registered final40 promotion gate
    was not met.

    Paid compute spend: #calc.round(r.spend.total_spend_usd, digits: 3) USD.
    Provider billing exact: #r.spend.exact_provider_billing. Active RunPod pods
    after this checkpoint: #r.runpod.active_pods_after_collection.
  ] else [
    The requested matched experiment cannot honestly proceed to RunPod under the
    locked local-gate rule. Nonzero plastic recurrent excitation is not yet a safe
    plug-in addition to the current matched COBA/PING SHD recipe: COBA's
    no-inhibitory-loop cell develops pathological gradients before the pilot
    stage, while PING remains well behaved. Progress from here requires a new
    design authority decision, such as adding a stabilizer, changing COBA's
    recurrence recipe, or giving recurrent weights their own optimizer controls.

    Paid compute spend: #calc.round(r.runpod.total_spend_usd, digits: 3) USD.
    Active pods after this checkpoint: #r.runpod.active_pods_after_collection.
  ]
]
