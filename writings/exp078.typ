#let meta = (
  title: "Predicting phase acquisition between coupled PING circuits",
  date: "2026-08-06",
  description: "A planned phase-response experiment will predict when reciprocal E-to-I coupling should transform drifting gamma phases into a bounded relationship, then test that prediction in a continuous switch-on trial.",
  collection: "miscellaneous",
  status: "draft",
)

#let body = [
  == Abstract

  This planned experiment asks whether reciprocal excitation from each PING
  circuit's excitatory population to the other circuit's inhibitory population
  can acquire gamma synchrony over several cycles. The protocol will first measure
  how one mature PING rhythm responds to an E-to-I pulse. It will use that response
  to predict the existence, phase lag, and convergence rate of a stable coupled
  state before running two circuits. The decisive trial must visibly separate an
  unsynchronized baseline, progressive acquisition, and a synchronized state.

  == Methods

  1. *Calibrate two independent mature PING oscillators.* Construct two 800 E /
    200 I circuits with independent Poisson drive and zero cross-circuit weight.
    Select fixed intrinsic and drive parameters that produce sustained gamma while
    retaining enough natural-frequency detuning to generate visible phase drift.
    Report E and I firing rates, natural frequency, cycle variability, and
    uncoupled phase drift over a small calibration seed set. Reject silent,
    saturated, intermittent, or already phase-stationary pairs, then freeze the
    parameters and reserve separate mature states and future-input seeds for
    validation.

  2. *Measure the macroscopic E-to-I phase-response curve.* Give the graph a
    dedicated spike input connected to I using the same declared connection rule,
    weight, AMPA kinetics, and axonal delay intended for cross-circuit coupling.
    Replay E-volley spike patterns measured during Step 1, preserving their size
    and temporal dispersion rather than synchronously stimulating every I neuron.
    At evenly spaced phases of a mature E cycle, compare a sender-equivalent volley
    with an identical-state, identical-input silent continuation. Define the
    response as the phase displacement after the rhythm has recovered; retain the
    one-cycle displacement only as a transient diagnostic. Record changes in volley
    size, additional or merged I volleys, and suppressed or skipped E volleys. This
    establishes both the phase response and the amplitude-changing boundary beyond
    which a phase-only description is inadequate.

  3. *Predict the coupled phase dynamics.* Combine the measured phase-response
    curve, sender-volley waveform, and candidate delays to estimate the interaction
    function for two reciprocally coupled rhythms. Restrict prediction to strengths
    that primarily shift phase and preserve the PING cycle. Predeclare one condition
    predicted to lock, one predicted not to lock, the stable lag, the common mean
    frequency, and a convergence-time range. Extra volleys, skipped cycles,
    suppression, or a regime transition are phase-model violations rather than
    failed simulations.

  4. *Test predicted synchronization acquisition.* Compile structurally identical
    zero-weight and coupled graphs containing both reciprocal E-to-I projections.
    Run the zero-weight graph through burn-in and at least 500 ms of recorded
    pre-switch activity, then continue its complete runtime state in the coupled
    graph using the next segment of the same pre-generated inputs. Exact split-run
    parity and first-effect timing are engineering acceptance tests for this
    protocol. The two executor calls must form one causal trajectory; no spike train
    may be shifted or selected using its post-switch outcome.

    Run both predeclared conditions for at least three predicted convergence times
    after the switch, with a minimum post-switch duration of 2000 ms. Use
    E-population volley times as the primary phase diagnostic. Plot sampled rasters
    for circuits A and B, separate A and B E-rate panels on a common grid, and
    cycle-by-cycle A-minus-B phase. Report mean-frequency convergence as a summary
    statistic rather than differentiating the noisy phase trace into another
    primary plot. A successful locking trajectory must show pre-switch drift,
    progressive phase correction, and preregistered residence around the predicted
    lag. Compare observed lag and convergence time with the Step 3 prediction and
    report later slips or amplitude disruptions without cropping them.

  5. *Replicate the prediction test.* Repeat the locking and non-locking conditions
    across the reserved mature states and future-input seeds. Apply the same
    analysis to every run. Report acquisition probability, convergence-time
    distribution, phase-lag error, post-acquisition residence, phase slips, mean
    frequency difference, and population health. A representative raster may
    illustrate the mechanism, but the verdict comes from the complete held-out set.
    Reciprocal E-to-I coupling explains acquisition only if the sender-equivalent
    phase response predicts the observed lag, stability, common frequency, and
    convergence timescale without materially changing cycle amplitude. Treat
    E-to-I as an isolated effective motif; E-to-E and mixed coupling are future
    experiments.

  === Implementation boundary

  The first implementation requires no general time-dependent-weight feature.
  _snnlang_ can already declare independent spike inputs, fixed AMPA and GABA
  projections, reciprocal feedback paths, delays, and the two otherwise identical
  zero-weight and coupled graphs. _tools/snn_ can already continue complete runtime
  state across those graphs because its compatibility signature excludes parameter
  values while retaining structural checks. The runner must own input generation,
  phase-target selection, branching, response-curve estimation, phase-model
  fitting, condition grids, and every plot.

  The present system cannot change a weight or trigger an intervention from an
  online phase estimate within one executor call. It also has no arbitrary analog
  conductance-input primitive. Those capabilities are not prerequisites here: a
  declared spike volley through a fixed AMPA projection supplies the phase probe,
  and exact split-run continuation supplies the coupling switch. Add a schedule or
  event-controller abstraction only if a later experiment genuinely requires
  continuously varying parameters or closed-loop intervention.

  == Results

  This experiment is planned and has no results yet.
]
