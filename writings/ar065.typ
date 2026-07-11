#let meta = (
  title: "Night shift",
  date: "2026-07-11",
  description: "The operating contract the unattended overnight agent obeys for the SHD program: what it may run, its budgets and scope, and — explicitly for this program — authorisation to use RunPod within a hard cost ceiling.",
  collection: "spiking-heidelberg-digits",
  status: "draft",
)

#let body = [
  This is the *policy* the overnight agent obeys — separate from the
  #link("/ar063/")[plan] (the science) and the #link("/ar064/")[log] (the record).
  It runs unattended for at most one night, works the plan's queue, and obeys the
  contract below.

  == What the night shift may do

  - *Run only what is queued* by a human (queue entries with _origin: human_).
    Never invent an experiment, never run a _proposed_ entry.
  - *May propose.* It can draft new _proposed_ queue entries for the morning
    review, but never runs them — proposing is a suggestion, not an action.
  - *Publish nothing.* It ends by prepending a digest to the #link("/ar064/")[log]
    and leaving its output for review. Nothing merges without a human — the
    morning gate decides.
  - *Enforce its own budgets*, aborting a run at its wall-clock ceiling and
    ending the shift on any stop condition below.

  == Compute — RunPod is authorised for this program

  Both local (Mac MPS, free) and RunPod are allowed. *This program explicitly
  authorises the night shift to launch RunPod pods and spend real money
  unattended* — a standing, deliberate exception to the default rule that cloud
  runs need per-run permission. It exists because the stability sweeps
  (exp061–063) are faster on a GPU and the scientist has accepted the cost.

  The authorisation is *bounded, not open*: a hard #emph[max cost per night]
  ceiling, a fixed GPU tier, and one pod at a time. When the ceiling is reached
  the shift stops, even mid-queue. Pods are reaped at shift end; a leaked pod is
  an anomaly for the morning gate.

  == The contract

  ```yaml
  budgets:
    epochs_per_run: 40          # ≈ one seed at native dt on MPS; less on RunPod
    wall_clock_per_run: 6h      # hard ceiling — kills a diverging run
    wall_clock_per_night: 8h
    seeds_default: 3            # one seed is an anecdote, not a result
  scope:
    collection: spiking-heidelberg-digits   # the only collection this shift may touch
    may_propose: true                        # draft `proposed` entries; never run them
  compute:
    local: allowed              # free, the default
    runpod: allowed             # EXPLICITLY authorised for this program — real spend
    gpu: "4090"
    max_cost_per_night_usd: 40  # hard $ ceiling; stop when reached
    max_pods_concurrent: 1
    reap_pods_at_end: true
  stop_when:
    - queue_empty
    - night_budget_exhausted
    - cost_ceiling_reached
    - build_red_twice           # bail rather than thrash
  ```

  == The morning gate

  Every night ends with a ruthless digest at the top of the #link("/ar064/")[log]:
  what confirmed, what died by its own kill criterion, what anomaly (a leaked
  pod, an unexplained NaN, a blown budget) needs a human. The review reads that
  first, and decides what — if anything — merges. The shift never merges its own
  work.
]
