#let meta = (
  title: "Variable-rate streaming with a summed-spiking readout",
  date: "2026-08-10",
  description: "Successor to exp048 using the variable-rate PING training bank and a readout matched to each presentation window.",
  collection: "gamma-gated-sparsity",
  status: "draft",
)

#let body = [
  == Abstract

  This experiment will test whether PING networks trained across input rates can classify continuous MNIST streams without the fixed-rate and output-membrane mismatch inherited by exp048. The trained readout sums excitatory spikes, and inference uses exactly the same operation over a window equal to the current digit's presentation duration. The planned results comprise matched-condition streaming, variable-rate and variable-duration streaming, a 200 ms rate psychometric, and a duration-by-rate accuracy map. Results remain pending until exp022 produces all three variable-rate checkpoints.

  == Methods

  === 1. Training source

  The experiment loads `ping__variable_rate__seed42`, `ping__variable_rate__seed43`, and `ping__variable_rate__seed44` from exp022. Each PING network is trained by sampling one maximum-pixel Poisson rate independently per image presentation from 0.5, 1, 2, 5, 10, and 25 Hz. The recurrent weights and classifier are frozen throughout this experiment.

  *Note.* The exp022 cells are registered and `tools/snn` implements their variable-rate training interface, but the full Cambridge jobs have not run yet. This entry remains their downstream target and fails loudly rather than falling back to fixed-rate weights.

  === 2. Summed-spiking readout

  For a presentation beginning at timestep $a$ and ending before timestep $b$, class evidence is

  $ bold(z) = (sum_(t=a)^(b-1) bold(s)^E(t)) W_"out". quad "(1)" $

  Here $bold(s)^E(t)$ is the 1024-element excitatory spike vector at timestep $t$, $W_"out"$ is the trained 1024-by-10 readout matrix, and $bold(z)$ contains the ten class logits. The predicted digit is the index of the largest logit. This is the trained `rate` readout: no output LIF membrane, leak constant, or post-hoc `mem-mean` reconstruction is introduced.

  === 3. Matched inference

  Every digit is read from only the spikes generated during its own presentation:

  $ T_"readout" = T_"presentation". quad "(2)" $

  Here $T_"readout"$ is the spike-summation window and $T_"presentation"$ is the duration of the corresponding digit. The first protocol holds both at 200 ms and presents a five-digit stream. This checks the trained duration and readout before either input rate or duration is varied.

  === 4. Variable-rate and variable-duration inference

  A second five-digit stream changes both presentation duration and maximum-pixel Poisson rate at digit boundaries. Each digit retains its own matched readout window. A factorial evaluation then crosses presentation durations 25, 50, 100, and 200 ms with the six training rates. Every cell is evaluated independently for seeds 42, 43, and 44.

  === 5. Psychometric protocol

  The rate psychometric fixes presentation and readout duration at 200 ms, varies the maximum-pixel rate across the six training values, and reports held-out classification accuracy across the three trained seeds. This isolates rate robustness from shortened evidence windows.

  == Results

  === 1. Matched-condition stream

  *TODO.* Reproduce the structure of exp048's streaming figure: excitatory and inhibitory rasters, digit boundaries, and the ten online class-evidence traces. Unlike exp048, every trace will be formed by accumulating spikes from the start of the current digit.

  === 2. Variable conditions

  *TODO.* Reproduce exp048's joint temporal and encoding-rate summary using the variable-rate-trained weights. Pair the duration-by-rate accuracy map with the 200 ms psychometric curve, so the effects of rate and available integration time remain distinguishable.

  === 3. Comparison with exp048

  *TODO.* Compare the completed results with exp048. The comparison must be interpretive rather than a silent replacement. Exp048 uses fixed-25-Hz training and reconstructs a sliding `mem-mean` output. This experiment uses mixed-rate training and the trained summed-spiking readout. Any accuracy difference therefore reflects the combined training-distribution and readout correction, not a single isolated intervention.
]
