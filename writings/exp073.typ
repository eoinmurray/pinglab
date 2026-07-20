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
#let coba-wee = r.smoke_w_ee_diagnostics.coba.layers.at("1")
#let ping-wee = r.smoke_w_ee_diagnostics.ping.layers.at("1")

#let body = [
  == Abstract

  This exploratory experiment tested the next biologically plausible path after
  exp071/exp072: keep the cumulative-potential readout, keep the historical
  one-seed validation-only SHD split, and make recurrent E→E excitation both
  nonzero and plastic while preserving Dale signs. The required local smoke
  stage killed the experiment before RunPod use. PING remained finite and
  active, but the matched COBA cell repeatedly produced skipped/non-finite
  updates even after reducing the requested W_EE scale from `0.3 0.01` through
  `0.03 0.01`, `0.003 0.001`, and finally `0.0003 0.0001`. No paid compute was
  run. The result is therefore a design-gate failure, not evidence about
  eight-epoch or forty-epoch SHD accuracy.

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
  added a focused regression test before this experiment ran.

  == Local gate

  The final registered local scout used `--w-ee #r.config.w_ee_init.at(0)
  #r.config.w_ee_init.at(1)` and two smoke epochs on 128 training / 128
  validation samples. The gate required both cells to be finite, active,
  non-saturated, and free of skipped or non-finite updates before any RunPod
  dispatch.

  #table(
    columns: (auto, auto, auto, auto, auto, 1.4fr),
    table.header([*Cell*], [*Passed*], [*Selected acc.*], [*E rate*], [*I rate*], [*Errors*]),
    [COBA],
    [#r.smoke.cells.coba.passed],
    [#maybe-pct(r.smoke.cells.coba.selected_validation_accuracy_pct)],
    [#maybe-hz(r.smoke.cells.coba.final_e_rate_hz)],
    [#maybe-hz(r.smoke.cells.coba.final_i_rate_hz)],
    [skipped/non-finite updates],
    [PING],
    [#r.smoke.cells.ping.passed],
    [#maybe-pct(r.smoke.cells.ping.selected_validation_accuracy_pct)],
    [#maybe-hz(r.smoke.cells.ping.final_e_rate_hz)],
    [#maybe-hz(r.smoke.cells.ping.final_i_rate_hz)],
    [none],
  )

  COBA was active but failed the finite-update criterion. PING passed the local
  plumbing check. Because the mandate requires both cells to clear the smoke
  stage before cloud execution, no short RunPod stage was launched.

  == W_EE diagnostics

  The saved smoke checkpoints confirm that `W_EE` was nonzero, Dale-constrained,
  and trainable enough to move from its reconstructed initialization:

  #table(
    columns: (auto, auto, auto, auto, auto),
    table.header([*Cell*], [*Initial mean*], [*Selected mean*], [*Mean |Δ|*], [*Nonnegative*]),
    [COBA],
    [#calc.round(coba-wee.initial.mean, digits: 8)],
    [#calc.round(coba-wee.selected.mean, digits: 8)],
    [#calc.round(coba-wee.selected_delta_from_initial.mean_abs, digits: 8)],
    [#coba-wee.selected_nonnegative],
    [PING],
    [#calc.round(ping-wee.initial.mean, digits: 8)],
    [#calc.round(ping-wee.selected.mean, digits: 8)],
    [#calc.round(ping-wee.selected_delta_from_initial.mean_abs, digits: 8)],
    [#ping-wee.selected_nonnegative],
  )

  The killed local scouts are archived under
  `artifacts/data/exp073/raw/killed_scouts/`. Their pattern was consistent:
  COBA failed the skipped-update criterion at every nonzero W_EE scale tried,
  while PING passed.

  #table(
    columns: (auto, auto, auto, auto, auto),
    table.header([*Scout*], [*COBA passed*], [*COBA selected*], [*PING passed*], [*PING selected*]),
    [`0.3 0.01`],
    [#r.killed_scouts.w_ee_0p3.smoke_summary.cells.coba.passed],
    [#maybe-pct(r.killed_scouts.w_ee_0p3.smoke_summary.cells.coba.selected_validation_accuracy_pct)],
    [#r.killed_scouts.w_ee_0p3.smoke_summary.cells.ping.passed],
    [#maybe-pct(r.killed_scouts.w_ee_0p3.smoke_summary.cells.ping.selected_validation_accuracy_pct)],
    [`0.03 0.01`],
    [#r.killed_scouts.w_ee_0p03.smoke_summary.cells.coba.passed],
    [#maybe-pct(r.killed_scouts.w_ee_0p03.smoke_summary.cells.coba.selected_validation_accuracy_pct)],
    [#r.killed_scouts.w_ee_0p03.smoke_summary.cells.ping.passed],
    [#maybe-pct(r.killed_scouts.w_ee_0p03.smoke_summary.cells.ping.selected_validation_accuracy_pct)],
    [`0.003 0.001`],
    [#r.killed_scouts.w_ee_0p003.smoke_summary.cells.coba.passed],
    [#maybe-pct(r.killed_scouts.w_ee_0p003.smoke_summary.cells.coba.selected_validation_accuracy_pct)],
    [#r.killed_scouts.w_ee_0p003.smoke_summary.cells.ping.passed],
    [#maybe-pct(r.killed_scouts.w_ee_0p003.smoke_summary.cells.ping.selected_validation_accuracy_pct)],
    [`0.0003 0.0001`],
    [#r.killed_scouts.w_ee_0p0003.smoke_summary.cells.coba.passed],
    [#maybe-pct(r.killed_scouts.w_ee_0p0003.smoke_summary.cells.coba.selected_validation_accuracy_pct)],
    [#r.killed_scouts.w_ee_0p0003.smoke_summary.cells.ping.passed],
    [#maybe-pct(r.killed_scouts.w_ee_0p0003.smoke_summary.cells.ping.selected_validation_accuracy_pct)],
  )

  == Conclusion

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
