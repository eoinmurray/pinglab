# snnlang report — two_ping_gamma_coupling

Populations: 4 (100 units)
Projections: 6
Operations: 0
Parameters: 6 tensors / 2,880 scalars
Estimated state: 100 scalars per sample and timestep
Estimated dense projection edges: 2,880
Trainable this recipe: 0 tensors
Outputs: none
Recurrent paths: a_E_to_I, a_I_to_E, b_E_to_I, b_I_to_E
Diagnostics: 0 errors, 0 warnings

## Populations
- a_E: 40 × coba_lif (spiking)
- a_I: 10 × coba_lif (spiking)
- b_E: 40 × coba_lif (spiking)
- b_I: 10 × coba_lif (spiking)

## Projections
- a_E_to_I: a_E.spikes → a_I.excitatory [recurrent, excitatory]
- a_I_to_E: a_I.spikes → a_E.inhibitory [recurrent, inhibitory]
- a_input: drive_a.value → a_E.excitatory [feedforward, excitatory]
- b_E_to_I: b_E.spikes → b_I.excitatory [recurrent, excitatory]
- b_I_to_E: b_I.spikes → b_E.inhibitory [recurrent, inhibitory]
- b_input: drive_b.value → b_E.excitatory [feedforward, excitatory]

## Parameters
- a_E_to_I.weight: [10, 40] nS (frozen/unselected)
- a_I_to_E.weight: [40, 10] nS (frozen/unselected)
- a_input.weight: [40, 16] nS (frozen/unselected)
- b_E_to_I.weight: [10, 40] nS (frozen/unselected)
- b_I_to_E.weight: [40, 10] nS (frozen/unselected)
- b_input.weight: [40, 16] nS (frozen/unselected)
