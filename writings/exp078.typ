#import "/.demolab/lib.typ": cite, reference-list

#let meta = (
  title: "Arnold tongue of two coupled PING circuits",
  date: "2026-08-07",
  description: "A graph-native reproduction of the detuning-by-coupling Arnold tongue reported for two reciprocally coupled PING circuits.",
  collection: "miscellaneous",
  status: "planned",
)

#let body = [
  == Abstract

  This experiment will attempt to reproduce a subset of Lowet et al.#cite(1):
  the Arnold tongue formed by two reciprocally coupled PING circuits as their
  natural-frequency detuning and coupling strength are varied. An Arnold tongue
  is the region in which coupled oscillators adopt a common frequency despite
  having different uncoupled natural frequencies. The result has two control
  axes—uncoupled natural-frequency detuning $Delta f_0$ and coupling strength
  $K$—and one measured response axis, the fraction of valid trials classified as
  frequency- and phase-locked. The reproduction is restricted to the paper's
  two-circuit synchronization result. It does not test the spatially extended
  network, natural-image reconstruction, or proposed phase and frequency codes.
  Two independently driven PING circuits will be simulated across the two
  control axes. A successful reproduction will recover a contiguous locked
  region centred near zero detuning that widens as coupling increases, with the
  intrinsically faster circuit leading in phase inside that region.

  == Methods

  #enum(
    [*Construct two independently driven PING circuits.* Circuits A and B will
  each contain 80 conductance-based leaky integrate-and-fire excitatory neurons
  and 20 inhibitory neurons. Within each circuit,
  excitatory neurons will project to inhibitory neurons through AMPA synapses,
  and inhibitory neurons will project to excitatory neurons through GABA-A
  synapses. The circuits will use identical neuron, synapse, connectivity, and
  within-circuit weight parameters.

  Independent Poisson spike populations will drive the two excitatory
  populations. Input spikes will be independently sampled between circuits and
  trials; no input spikes will be shared. The network parameter seed will remain
  fixed across the primary sweep so that detuning and coupling are not
  confounded with different weight realizations.

  Reciprocal coupling will match the two-circuit configuration of Lowet et
  al.#cite(1):

  $ E_A -> E_B, quad E_A -> I_B, quad E_B -> E_A, quad E_B -> I_A. $

  All four cross-circuit projections will use AMPA synapses. Cross-circuit
  E-to-E and E-to-I weights will be controlled by the same coupling parameter
  $K$ and varied together. The uncoupled graph will retain all four projections
  with zero-valued weights, preserving graph topology across conditions.

  #figure(
    image(
      "/artifacts/data/exp078/network.svg",
      width: 100%,
      alt: "Two independently driven PING circuits with reciprocal excitatory-to-excitatory and excitatory-to-inhibitory cross-circuit coupling.",
    ),
    caption: [Network graph for the reproduction. Each independent input drives
      one local E↔I PING loop. Dashed projections are the four reciprocal
      cross-circuit AMPA pathways varied together by $K$. Red bar-headed
      projections are local GABA-A inhibition.],
  )],

    [*Calibrate drive against uncoupled gamma frequency.* One uncoupled PING
  circuit will be simulated across a bounded input-rate grid. The operating
  range will be the contiguous valid interval in which gamma peak frequency
  increases monotonically with input rate. The calibration will be completed
  before coupled conditions are inspected.

  Pairs of input rates will be selected to sample signed target detunings of
  approximately

  $ Delta f_0 in { -6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6 } " Hz". $

  For every input pair and trial seed, the two circuits will first run with
  $K=0$. Their measured post-transient frequencies will define natural-
  frequency detuning:

  $ Delta f_0 = f_A^0 - f_B^0. $

  This measurement, rather than input-rate difference or coupled frequency
  difference, will determine the horizontal coordinate of every coupled result.],

    [*Register and execute the detuning-by-coupling sweep.* A bounded pilot at
  zero detuning and at the largest positive and negative target detunings will
  locate a coupling interval containing uncoupled, partially locked, and locked
  behaviour without silencing or saturating either circuit. The pilot will set
  only the limits of $K$ and will not enter the primary analysis.

  The primary grid will contain $K=0$ and ten equally spaced nonzero values over
  the selected interval. Every measured detuning will be crossed with every
  coupling value and at least five independently generated pairs of input spike
  trains. No grid point will be added or removed after inspecting the primary
  locking map.

  Simulations will use a 0.1 ms timestep and run for 3 s. The first 500 ms will
  be excluded from analysis. Population spikes, membrane voltages, and
  projection conductances will be retained with the compiled graph, generated
  inputs, parameter tensors, run configuration, and seed ledger.

  A trial will be invalid if any recorded state is non-finite, either mean
  excitatory or inhibitory firing rate is below 1 Hz, or either excitatory
  population has no spectral peak between 25 and 80 Hz. The same validity rule
  will be applied to every sweep cell.],

    [*Measure frequency and relative phase.* Excitatory population spikes will
  be converted to population rates by Gaussian smoothing with a 5 ms standard
  deviation. Gamma peak frequency will be estimated from the post-transient
  rate spectrum. Instantaneous phase will be calculated from the analytic
  signal after zero-phase band-pass filtering from 25 to 90 Hz. The filter will
  remain fixed across conditions.

  Relative phase will be

  $ phi(t) = "unwrap"(theta_A(t) - theta_B(t)). $

  Each trial will report the two emergent frequencies, their absolute
  difference, the linear slope of $phi(t)$, the number of complete $2 pi$ phase
  slips, phase-locking value, and circular mean phase difference.],

    [*Classify locking without using phase-locking value alone.* Frequency,
  drift, and phase-slip tolerances will be fixed from estimator variability in
  repeated zero-detuning and uncoupled calibration trials before the primary
  sweep. A valid trial will be classified as locked only if its emergent
  frequency difference, absolute relative-phase slope, and phase-slip count all
  fall within their registered tolerances. Phase-locking value will describe
  locking strength but will not independently determine the classification.],

    [*Reconstruct and test the Arnold tongue.* The primary result will be the
  fraction of valid trials classified as locked at each measured $Delta f_0$
  and $K$. Supporting maps will show emergent frequency difference, relative-
  phase slope, phase-slip rate, phase-locking value, and circular mean phase
  difference on the same grid.

  The reproduction will pass if the locking map contains a contiguous region
  centred near zero detuning whose width increases across at least three
  successive nonzero coupling levels. Within locked nonzero-detuning
  conditions, the circuit with the greater uncoupled natural frequency must
  lead in phase. It will fail if locking is absent, confined to zero detuning,
  does not widen with coupling, or occurs only in invalid conditions.

  Representative traces will be selected by fixed grid location: zero
  detuning, one nonzero-detuning condition inside the locking region, and the
  same detuning immediately outside it. Each trace will show both excitatory
  rasters, both excitatory population rates, unwrapped relative phase, and
  sliding-window frequency difference.],
  )

  #reference-list((
    (
      text: [Lowet, Roberts, Hadjipapas, Peter, van der Eerden & De Weerd: _Input-Dependent Frequency Modulation of Cortical Gamma Oscillations Shapes Spatial Synchronization and Enables Phase Coding_. PLOS Computational Biology, 2015.],
      doi: "10.1371/journal.pcbi.1004072",
    ),
  ))
]
