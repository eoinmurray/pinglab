#let meta = (
  title: "Night shift — matched COBA and PING on SHD",
  date: "2026-07-18",
  description: "The first mandate for restarting the Spiking Heidelberg Digits programme: establish that matched Dale-constrained COBA and PING networks both train on a small SHD subset and produce directly comparable learning curves and rasters.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
  order: 1,
  queue: (
    (
      id: "exp066",
      hypothesis: "With matched input and readout settings, both Dale-constrained COBA and PING learn a small SHD classification task above chance and produce stable, directly comparable population rasters.",
      kill: "Stop before scaling if either model remains below 20% test accuracy, develops a non-finite loss, or stays silent or saturated after the single pre-registered shared recipe adjustment.",
      baseline: "The 5% chance floor for SHD's 20 classes; both models must exceed 20% test accuracy in the pilot.",
      seeds: 1,
      budget: "One local smoke stage plus at most two RunPod training attempts per model; 2 h per attempt and $40 total RunPod spend.",
      status: "queued",
      origin: "human",
    ),
  ),
)

#let body = [
  == Digest

  This section is intentionally empty until the shift finishes. It will report
  what completed, what failed its registered criterion, the compute used, and
  the next question worth testing.

  == Mandate

  === Programme goal

  Establish a minimal working Spiking Heidelberg Digits (SHD) baseline for
  matched COBA and PING networks under Dale's law. Both networks must learn
  above chance on a small subset, produce stable learning curves, and expose
  input, excitatory, and inhibitory spike rasters that can be compared on the
  same utterances. This shift establishes feasibility. It does not test whether
  PING is more accurate, more efficient, or better able to use temporal
  structure than COBA.

  === Why this is the first question

  The existing MNIST training hub shows that both modes of the same
  conductance-based network learn a static image task. COBA disables the
  reciprocal excitatory-to-inhibitory-to-excitatory loop, while PING keeps the
  fixed loop and produces gamma-rhythmic population activity. SHD is the next
  test because its input already consists of timed cochlear spikes rather than
  an image converted into Poisson events. Before comparing any computational
  advantage, the two established modes must first train cleanly on this input
  and yield interpretable dynamics.

  === Registered experiment

  #table(
    columns: (auto, 1fr),
    table.header([*Field*], [*exp066*]),
    [Hypothesis], [#meta.queue.first().hypothesis],
    [Baseline], [#meta.queue.first().baseline],
    [Kill criterion], [#meta.queue.first().kill],
    [Seed count], [#str(meta.queue.first().seeds)],
    [Budget], [#meta.queue.first().budget],
    [Status], [#meta.queue.first().status],
  )

  The comparison has two predeclared architectural differences. COBA disables
  the inhibitory loop and uses no voltage-gradient dampening. PING enables the
  fixed inhibitory loop and uses the established voltage-gradient dampening
  factor of 1000. The dampening is part of PING's training recipe and will not
  be tuned after comparing outcomes.

  Everything else is matched: one hidden layer with 256 excitatory cells and
  64 inhibitory cells, Dale's law, fixed recurrent weights, identical input
  weight initialization and sparsity, identical readout initialization and
  mode, Adam learning rate, batch size, integration timestep, simulation
  duration, data subset, epoch count, and seed. The proposed first recipe uses
  a shared input-weight mean of 0.9, 95% input sparsity, a readout scale of 225,
  membrane-mean readout, learning rate 0.0004, batch size 32, integration
  timestep 1 ms, simulation duration 1000 ms, seed 42, and no firing-rate
  regulariser.

  === Staged execution

  + *Smoke stage.* Train both cells on 128 training utterances for 2 epochs on
    the local CPU. This stage checks the SHD loader, tensor shapes, finite loss,
    non-silent hidden activity, checkpointing, and recording. It is plumbing,
    not evidence for the hypothesis.
  + *Pilot stage.* If both smoke cells are finite and active, train both cells
    on the same 1000-utterance subset for 20 epochs on one RunPod GPU. The pilot
    is the registered result.
  + *Shared adjustment.* If either smoke cell is silent or saturated, both
    models receive one identical change to the input-weight scale before the
    smoke stage is repeated. No model-specific input or readout tuning is
    permitted. If the repeated smoke still fails, the experiment is killed and
    the pilot does not run.

  === Evidence packet

  The experiment must report train and test loss by epoch, test accuracy by
  epoch, excitatory and inhibitory firing rates, non-finite losses or skipped
  updates, and the exact configuration. It must also render input, excitatory,
  and inhibitory rasters for the same pre-selected test utterances in both
  models. The utterances are selected by dataset position before predictions
  are inspected, and every panel uses the same time axis and population order.

  === Operating contract

  ```yaml
  budgets:
    wall_clock_per_run: 2h
    wall_clock_per_shift: 8h
    runpod_total_usd: 40
    max_pods_concurrent: 1
    seeds_default: 1
  scope:
    collection: spiking-heidelberg-digits
    may_propose: true
    may_edit_snn_cli: false
  stop_when:
    - queue_empty
    - shift_budget_exhausted
    - runpod_budget_exhausted
    - build_red_twice
  ```

  === Branch and logging contract

  This programme runs on the existing `autoresearch/shd` pull-request branch.
  The mandate is committed there before the experiment runner or any result,
  making that commit the pre-run anchor. The branch is merged into `main` only
  if the scientist accepts the completed evidence packet.

  Implementation changes, debugging decisions, and other engineering work are
  recorded in focused commit messages so the pull request carries a readable
  code history. Scientific events and conclusions are recorded here in the
  Record as the shift proceeds. The Record may link to a commit when the exact
  implementation change is relevant to interpreting a result, but it does not
  duplicate the engineering log.

  == Record

  This section is intentionally empty until the mandate is approved and
  committed. During the shift it will record the scientific outcome of each stage,
  including seeds, aborts, criterion failures, and anomalies. Engineering
  details belong in the experiment commit rather than here.
]
