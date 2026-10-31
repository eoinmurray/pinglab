# snnlang report — CTC attention: A and B converge on C

Populations: 6 (480 units)
Projections: 26
Operations: 0
Parameters: 26 tensors / 201,728 scalars
Estimated state: 480 scalars per sample and timestep
Estimated dense projection edges: 201,728
Trainable this recipe: 0 tensors
Outputs: none
Recurrent paths: A_E_to_A_E, A_E_to_A_I, A_I_to_A_E, A_I_to_A_I, B_E_to_B_E, B_E_to_B_I, B_I_to_B_E, B_I_to_B_I, C_E_to_A_E, C_E_to_A_I, C_E_to_B_E, C_E_to_B_I, C_E_to_C_E, C_E_to_C_I, C_I_to_C_E, C_I_to_C_I
Diagnostics: 0 errors, 0 warnings

## Populations
- A_E: 128 × coba_lif (spiking)
- A_I: 32 × coba_lif (spiking)
- B_E: 128 × coba_lif (spiking)
- B_I: 32 × coba_lif (spiking)
- C_E: 128 × coba_lif (spiking)
- C_I: 32 × coba_lif (spiking)

## Projections
- A_E_to_A_E: A_E.spikes → A_E.excitatory [recurrent, excitatory]
- A_E_to_A_I: A_E.spikes → A_I.excitatory [recurrent, excitatory]
- A_E_to_C_E: A_E.spikes → C_E.excitatory [feedforward, excitatory]
- A_E_to_C_I: A_E.spikes → C_I.excitatory [feedforward, excitatory]
- A_I_to_A_E: A_I.spikes → A_E.inhibitory [recurrent, inhibitory]
- A_I_to_A_I: A_I.spikes → A_I.inhibitory [recurrent, inhibitory]
- B_E_to_B_E: B_E.spikes → B_E.excitatory [recurrent, excitatory]
- B_E_to_B_I: B_E.spikes → B_I.excitatory [recurrent, excitatory]
- B_E_to_C_E: B_E.spikes → C_E.excitatory [feedforward, excitatory]
- B_E_to_C_I: B_E.spikes → C_I.excitatory [feedforward, excitatory]
- B_I_to_B_E: B_I.spikes → B_E.inhibitory [recurrent, inhibitory]
- B_I_to_B_I: B_I.spikes → B_I.inhibitory [recurrent, inhibitory]
- C_E_to_A_E: C_E.spikes → A_E.excitatory [recurrent, excitatory]
- C_E_to_A_I: C_E.spikes → A_I.excitatory [recurrent, excitatory]
- C_E_to_B_E: C_E.spikes → B_E.excitatory [recurrent, excitatory]
- C_E_to_B_I: C_E.spikes → B_I.excitatory [recurrent, excitatory]
- C_E_to_C_E: C_E.spikes → C_E.excitatory [recurrent, excitatory]
- C_E_to_C_I: C_E.spikes → C_I.excitatory [recurrent, excitatory]
- C_I_to_C_E: C_I.spikes → C_E.inhibitory [recurrent, inhibitory]
- C_I_to_C_I: C_I.spikes → C_I.inhibitory [recurrent, inhibitory]
- attention_A_to_A_E: attention_A.value → A_E.excitatory [feedforward, excitatory]
- attention_B_to_B_E: attention_B.value → B_E.excitatory [feedforward, excitatory]
- background_C_to_C_E: background_C.value → C_E.excitatory [feedforward, excitatory]
- background_C_to_C_I: background_C.value → C_I.excitatory [feedforward, excitatory]
- sensory_A_to_A_E: sensory_A.value → A_E.excitatory [feedforward, excitatory]
- sensory_B_to_B_E: sensory_B.value → B_E.excitatory [feedforward, excitatory]

## Parameters
- A_E_to_A_E.weight: [128, 128] uS (frozen/unselected)
- A_E_to_A_I.weight: [32, 128] uS (frozen/unselected)
- A_E_to_C_E.weight: [128, 128] uS (frozen/unselected)
- A_E_to_C_I.weight: [32, 128] uS (frozen/unselected)
- A_I_to_A_E.weight: [128, 32] uS (frozen/unselected)
- A_I_to_A_I.weight: [32, 32] uS (frozen/unselected)
- B_E_to_B_E.weight: [128, 128] uS (frozen/unselected)
- B_E_to_B_I.weight: [32, 128] uS (frozen/unselected)
- B_E_to_C_E.weight: [128, 128] uS (frozen/unselected)
- B_E_to_C_I.weight: [32, 128] uS (frozen/unselected)
- B_I_to_B_E.weight: [128, 32] uS (frozen/unselected)
- B_I_to_B_I.weight: [32, 32] uS (frozen/unselected)
- C_E_to_A_E.weight: [128, 128] uS (frozen/unselected)
- C_E_to_A_I.weight: [32, 128] uS (frozen/unselected)
- C_E_to_B_E.weight: [128, 128] uS (frozen/unselected)
- C_E_to_B_I.weight: [32, 128] uS (frozen/unselected)
- C_E_to_C_E.weight: [128, 128] uS (frozen/unselected)
- C_E_to_C_I.weight: [32, 128] uS (frozen/unselected)
- C_I_to_C_E.weight: [128, 32] uS (frozen/unselected)
- C_I_to_C_I.weight: [32, 32] uS (frozen/unselected)
- attention_A_to_A_E.weight: [128, 64] uS (frozen/unselected)
- attention_B_to_B_E.weight: [128, 64] uS (frozen/unselected)
- background_C_to_C_E.weight: [128, 64] uS (frozen/unselected)
- background_C_to_C_I.weight: [32, 64] uS (frozen/unselected)
- sensory_A_to_A_E.weight: [128, 64] uS (frozen/unselected)
- sensory_B_to_B_E.weight: [128, 64] uS (frozen/unselected)
