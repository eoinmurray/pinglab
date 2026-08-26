# snnlang report — two_ping_unidirectional

Populations: 4 (35 units)
Projections: 11
Operations: 0
Parameters: 11 tensors / 873 scalars
Estimated state: 35 scalars per sample and timestep
Estimated dense projection edges: 873
Trainable this recipe: 0 tensors
Outputs: none
Recurrent paths: a_E_to_E, a_E_to_I, a_I_to_E, a_I_to_I, a_I_to_b_E, b_E_to_E, b_E_to_I, b_I_to_E, b_I_to_I
Diagnostics: 0 errors, 0 warnings

## Populations
- a_E: 16 × coba_lif (spiking)
- a_I: 4 × coba_lif (spiking)
- b_E: 12 × coba_lif (spiking)
- b_I: 3 × coba_lif (spiking)

## Projections
- a_E_to_E: a_E.spikes → a_E.excitatory [recurrent, excitatory]
- a_E_to_I: a_E.spikes → a_I.excitatory [recurrent, excitatory]
- a_I_to_E: a_I.spikes → a_E.inhibitory [recurrent, inhibitory]
- a_I_to_I: a_I.spikes → a_I.inhibitory [recurrent, inhibitory]
- a_I_to_b_E: a_I.spikes → b_E.inhibitory [feedback, inhibitory]
- a_input: drive_a.value → a_E.excitatory [feedforward, excitatory]
- b_E_to_E: b_E.spikes → b_E.excitatory [recurrent, excitatory]
- b_E_to_I: b_E.spikes → b_I.excitatory [recurrent, excitatory]
- b_I_to_E: b_I.spikes → b_E.inhibitory [recurrent, inhibitory]
- b_I_to_I: b_I.spikes → b_I.inhibitory [recurrent, inhibitory]
- b_input: drive_b.value → b_E.excitatory [feedforward, excitatory]

## Parameters
- a_E_to_E.weight: [16, 16] nS (frozen/unselected)
- a_E_to_I.weight: [4, 16] nS (frozen/unselected)
- a_I_to_E.weight: [16, 4] nS (frozen/unselected)
- a_I_to_I.weight: [4, 4] nS (frozen/unselected)
- a_I_to_b_E.weight: [12, 4] nS (frozen/unselected)
- a_input.weight: [16, 8] nS (frozen/unselected)
- b_E_to_E.weight: [12, 12] nS (frozen/unselected)
- b_E_to_I.weight: [3, 12] nS (frozen/unselected)
- b_I_to_E.weight: [12, 3] nS (frozen/unselected)
- b_I_to_I.weight: [3, 3] nS (frozen/unselected)
- b_input.weight: [12, 6] nS (frozen/unselected)
