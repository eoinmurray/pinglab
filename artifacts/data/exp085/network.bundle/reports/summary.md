# snnlang report — ping_pair

Populations: 4 (200 units)
Projections: 10
Operations: 0
Parameters: 10 tensors / 42,880 scalars
Estimated state: 200 scalars per sample and timestep
Estimated dense projection edges: 42,880
Trainable this recipe: 0 tensors
Outputs: none
Recurrent paths: PING_A_E_to_I, PING_A_E_to_PING_B_E, PING_A_E_to_PING_B_I, PING_A_I_to_E, PING_B_E_to_I, PING_B_E_to_PING_A_E, PING_B_E_to_PING_A_I, PING_B_I_to_E
Diagnostics: 0 errors, 0 warnings

## Populations
- PING_A_E: 80 × coba_lif (spiking)
- PING_A_I: 20 × coba_lif (spiking)
- PING_B_E: 80 × coba_lif (spiking)
- PING_B_I: 20 × coba_lif (spiking)

## Projections
- PING_A_E_to_I: PING_A_E.spikes → PING_A_I.excitatory [recurrent, excitatory]
- PING_A_E_to_PING_B_E: PING_A_E.spikes → PING_B_E.excitatory [feedback, excitatory]
- PING_A_E_to_PING_B_I: PING_A_E.spikes → PING_B_I.excitatory [feedback, excitatory]
- PING_A_I_to_E: PING_A_I.spikes → PING_A_E.inhibitory [recurrent, inhibitory]
- PING_A_input: drive_a.value → PING_A_E.excitatory [feedforward, excitatory]
- PING_B_E_to_I: PING_B_E.spikes → PING_B_I.excitatory [recurrent, excitatory]
- PING_B_E_to_PING_A_E: PING_B_E.spikes → PING_A_E.excitatory [feedback, excitatory]
- PING_B_E_to_PING_A_I: PING_B_E.spikes → PING_A_I.excitatory [feedback, excitatory]
- PING_B_I_to_E: PING_B_I.spikes → PING_B_E.inhibitory [recurrent, inhibitory]
- PING_B_input: drive_b.value → PING_B_E.excitatory [feedforward, excitatory]

## Parameters
- PING_A_E_to_I.weight: [20, 80] nS (frozen/unselected)
- PING_A_E_to_PING_B_E.weight: [80, 80] nS (frozen/unselected)
- PING_A_E_to_PING_B_I.weight: [20, 80] nS (frozen/unselected)
- PING_A_I_to_E.weight: [80, 20] nS (frozen/unselected)
- PING_A_input.weight: [80, 128] nS (frozen/unselected)
- PING_B_E_to_I.weight: [20, 80] nS (frozen/unselected)
- PING_B_E_to_PING_A_E.weight: [80, 80] nS (frozen/unselected)
- PING_B_E_to_PING_A_I.weight: [20, 80] nS (frozen/unselected)
- PING_B_I_to_E.weight: [80, 20] nS (frozen/unselected)
- PING_B_input.weight: [80, 128] nS (frozen/unselected)
