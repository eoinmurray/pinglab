# snnlang report — coupled_ping_feedback

Populations: 4 (480 units)
Projections: 8
Operations: 0
Parameters: 8 tensors / 159,744 scalars
Estimated state: 480 scalars per sample and timestep
Estimated dense projection edges: 159,744
Trainable this recipe: 0 tensors
Outputs: none
Recurrent paths: b_feedback_a, object_a_E_to_I, object_a_I_to_E, object_b_E_to_I, object_b_I_to_E
Diagnostics: 0 errors, 0 warnings

## Populations
- object_a_E: 192 × coba_lif (spiking)
- object_a_I: 48 × coba_lif (spiking)
- object_b_E: 192 × coba_lif (spiking)
- object_b_I: 48 × coba_lif (spiking)

## Projections
- a_to_b: object_a_E.spikes → object_b_E.excitatory [feedforward, excitatory]
- b_feedback_a: object_b_E.spikes → object_a_E.modulatory [feedback, modulatory]
- object_a_E_to_I: object_a_E.spikes → object_a_I.excitatory [recurrent, excitatory]
- object_a_I_to_E: object_a_I.spikes → object_a_E.inhibitory [recurrent, inhibitory]
- object_a_input: stimulus.value → object_a_E.excitatory [feedforward, excitatory]
- object_b_E_to_I: object_b_E.spikes → object_b_I.excitatory [recurrent, excitatory]
- object_b_I_to_E: object_b_I.spikes → object_b_E.inhibitory [recurrent, inhibitory]
- object_b_input: stimulus.value → object_b_E.excitatory [feedforward, excitatory]

## Parameters
- a_to_b.weight: [192, 192] nS (frozen/unselected)
- b_feedback_a.weight: [192, 192] nS (frozen/unselected)
- object_a_E_to_I.weight: [48, 192] nS (frozen/unselected)
- object_a_I_to_E.weight: [192, 48] nS (frozen/unselected)
- object_a_input.weight: [192, 128] nS (frozen/unselected)
- object_b_E_to_I.weight: [48, 192] nS (frozen/unselected)
- object_b_I_to_E.weight: [192, 48] nS (frozen/unselected)
- object_b_input.weight: [192, 128] nS (frozen/unselected)
