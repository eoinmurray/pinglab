# snnlang report — exp097_large_ping

Populations: 2 (1,000 units)
Projections: 3
Operations: 0
Parameters: 3 tensors / 422,400 scalars
Estimated state: 1,000 scalars per sample and timestep
Estimated dense projection edges: 422,400
Trainable this recipe: 0 tensors
Outputs: none
Recurrent paths: ping_E_to_I, ping_I_to_E
Diagnostics: 0 errors, 0 warnings

## Populations
- ping_E: 800 × coba_lif (spiking)
- ping_I: 200 × coba_lif (spiking)

## Projections
- ping_E_to_I: ping_E.spikes → ping_I.excitatory [recurrent, excitatory]
- ping_I_to_E: ping_I.spikes → ping_E.inhibitory [recurrent, inhibitory]
- ping_input: drive.value → ping_E.excitatory [feedforward, excitatory]

## Parameters
- ping_E_to_I.weight: [200, 800] nS (frozen/unselected)
- ping_I_to_E.weight: [800, 200] nS (frozen/unselected)
- ping_input.weight: [800, 128] nS (frozen/unselected)
