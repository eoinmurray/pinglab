# snnlang report — coupling_onset

Populations: 4 (200 units)
Projections: 10
Operations: 0
Parameters: 10 tensors / 42,880 scalars
Estimated state: 200 scalars per sample and timestep
Estimated dense projection edges: 42,880
Trainable this recipe: 0 tensors
Outputs: none
Recurrent paths: a_E_to_I, a_E_to_b_E, a_E_to_b_I, a_I_to_E, b_E_to_I, b_E_to_a_E, b_E_to_a_I, b_I_to_E
Diagnostics: 0 errors, 0 warnings

## Populations
- a_E: 80 × coba_lif (spiking)
- a_I: 20 × coba_lif (spiking)
- b_E: 80 × coba_lif (spiking)
- b_I: 20 × coba_lif (spiking)

## Projections
- a_E_to_I: a_E.spikes → a_I.excitatory [recurrent, excitatory]
- a_E_to_b_E: a_E.spikes → b_E.excitatory [feedback, excitatory]
- a_E_to_b_I: a_E.spikes → b_I.excitatory [feedback, excitatory]
- a_I_to_E: a_I.spikes → a_E.inhibitory [recurrent, inhibitory]
- a_input: drive_a.value → a_E.excitatory [feedforward, excitatory]
- b_E_to_I: b_E.spikes → b_I.excitatory [recurrent, excitatory]
- b_E_to_a_E: b_E.spikes → a_E.excitatory [feedback, excitatory]
- b_E_to_a_I: b_E.spikes → a_I.excitatory [feedback, excitatory]
- b_I_to_E: b_I.spikes → b_E.inhibitory [recurrent, inhibitory]
- b_input: drive_b.value → b_E.excitatory [feedforward, excitatory]

## Parameters
- a_E_to_I.weight: [20, 80] nS (frozen/unselected)
- a_E_to_b_E.weight: [80, 80] nS (frozen/unselected)
- a_E_to_b_I.weight: [20, 80] nS (frozen/unselected)
- a_I_to_E.weight: [80, 20] nS (frozen/unselected)
- a_input.weight: [80, 128] nS (frozen/unselected)
- b_E_to_I.weight: [20, 80] nS (frozen/unselected)
- b_E_to_a_E.weight: [80, 80] nS (frozen/unselected)
- b_E_to_a_I.weight: [20, 80] nS (frozen/unselected)
- b_I_to_E.weight: [80, 20] nS (frozen/unselected)
- b_input.weight: [80, 128] nS (frozen/unselected)
