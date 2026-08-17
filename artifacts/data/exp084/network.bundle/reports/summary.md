# snnlang report — ping_inhibitory_timescale

Populations: 2 (100 units)
Projections: 3
Operations: 0
Parameters: 3 tensors / 13,440 scalars
Estimated state: 100 scalars per sample and timestep
Estimated dense projection edges: 13,440
Trainable this recipe: 0 tensors
Outputs: none
Recurrent paths: ping_E_to_I, ping_I_to_E
Diagnostics: 0 errors, 0 warnings

## Populations
- ping_E: 80 × coba_lif (spiking)
- ping_I: 20 × coba_lif (spiking)

## Projections
- ping_E_to_I: ping_E.spikes → ping_I.excitatory [recurrent, excitatory]
- ping_I_to_E: ping_I.spikes → ping_E.inhibitory [recurrent, inhibitory]
- ping_input: drive.value → ping_E.excitatory [feedforward, excitatory]

## Parameters
- ping_E_to_I.weight: [20, 80] nS (frozen/unselected)
- ping_I_to_E.weight: [80, 20] nS (frozen/unselected)
- ping_input.weight: [80, 128] nS (frozen/unselected)
