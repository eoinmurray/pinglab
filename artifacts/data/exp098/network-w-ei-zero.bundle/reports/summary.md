# snnlang report — exp098_ping_dynamics

Populations: 2 (125 units)
Projections: 3
Operations: 0
Parameters: 3 tensors / 15,000 scalars
Estimated state: 125 scalars per sample and timestep
Estimated dense projection edges: 15,000
Trainable this recipe: 0 tensors
Outputs: none
Recurrent paths: ping_E_to_I, ping_I_to_E
Diagnostics: 0 errors, 0 warnings

## Populations
- ping_E: 100 × coba_lif (spiking)
- ping_I: 25 × coba_lif (spiking)

## Projections
- ping_E_to_I: ping_E.spikes → ping_I.excitatory [recurrent, excitatory]
- ping_I_to_E: ping_I.spikes → ping_E.inhibitory [recurrent, inhibitory]
- ping_input: drive.value → ping_E.excitatory [feedforward, excitatory]

## Parameters
- ping_E_to_I.weight: [25, 100] nS (frozen/unselected)
- ping_I_to_E.weight: [100, 25] nS (frozen/unselected)
- ping_input.weight: [100, 100] nS (frozen/unselected)
