# snnlang report — spatial_gaussian_relay

Populations: 6 (1,500 units)
Projections: 21
Operations: 0
Parameters: 21 tensors / 3,198,000 scalars
Estimated state: 1,500 scalars per sample and timestep
Estimated dense projection edges: 3,198,000
Trainable this recipe: 0 tensors
Outputs: none
Recurrent paths: none
Diagnostics: 0 errors, 0 warnings

## Populations
- A_E: 400 × coba_lif (spiking)
- A_I: 100 × coba_lif (spiking)
- B_E: 400 × coba_lif (spiking)
- B_I: 100 × coba_lif (spiking)
- C_E: 400 × coba_lif (spiking)
- C_I: 100 × coba_lif (spiking)

## Projections
- A_E_to_C_E: A_E.spikes → C_E.excitatory [feedforward, excitatory]
- A_E_to_C_I: A_E.spikes → C_I.excitatory [feedforward, excitatory]
- A_E_to_E: A_E.spikes → A_E.excitatory [feedforward, excitatory]
- A_E_to_I: A_E.spikes → A_I.excitatory [feedforward, excitatory]
- A_I_to_E: A_I.spikes → A_E.inhibitory [feedforward, inhibitory]
- A_I_to_I: A_I.spikes → A_I.inhibitory [feedforward, inhibitory]
- B_E_to_C_E: B_E.spikes → C_E.excitatory [feedforward, excitatory]
- B_E_to_C_I: B_E.spikes → C_I.excitatory [feedforward, excitatory]
- B_E_to_E: B_E.spikes → B_E.excitatory [feedforward, excitatory]
- B_E_to_I: B_E.spikes → B_I.excitatory [feedforward, excitatory]
- B_I_to_E: B_I.spikes → B_E.inhibitory [feedforward, inhibitory]
- B_I_to_I: B_I.spikes → B_I.inhibitory [feedforward, inhibitory]
- C_E_to_E: C_E.spikes → C_E.excitatory [feedforward, excitatory]
- C_E_to_I: C_E.spikes → C_I.excitatory [feedforward, excitatory]
- C_I_to_E: C_I.spikes → C_E.inhibitory [feedforward, inhibitory]
- C_I_to_I: C_I.spikes → C_I.inhibitory [feedforward, inhibitory]
- cue_A: U_A.value → A_E.excitatory [feedforward, excitatory]
- cue_B: U_B.value → B_E.excitatory [feedforward, excitatory]
- sensory_A: X_A.value → A_E.excitatory [feedforward, excitatory]
- sensory_B: X_B.value → B_E.excitatory [feedforward, excitatory]
- sensory_C: X_C.value → C_E.excitatory [feedforward, excitatory]

## Parameters
- A_E_to_C_E.weight: [400, 400] uS (frozen/unselected)
- A_E_to_C_I.weight: [100, 400] uS (frozen/unselected)
- A_E_to_E.weight: [400, 400] uS (frozen/unselected)
- A_E_to_I.weight: [100, 400] uS (frozen/unselected)
- A_I_to_E.weight: [400, 100] uS (frozen/unselected)
- A_I_to_I.weight: [100, 100] uS (frozen/unselected)
- B_E_to_C_E.weight: [400, 400] uS (frozen/unselected)
- B_E_to_C_I.weight: [100, 400] uS (frozen/unselected)
- B_E_to_E.weight: [400, 400] uS (frozen/unselected)
- B_E_to_I.weight: [100, 400] uS (frozen/unselected)
- B_I_to_E.weight: [400, 100] uS (frozen/unselected)
- B_I_to_I.weight: [100, 100] uS (frozen/unselected)
- C_E_to_E.weight: [400, 400] uS (frozen/unselected)
- C_E_to_I.weight: [100, 400] uS (frozen/unselected)
- C_I_to_E.weight: [400, 100] uS (frozen/unselected)
- C_I_to_I.weight: [100, 100] uS (frozen/unselected)
- cue_A.weight: [400, 1024] uS (frozen/unselected)
- cue_B.weight: [400, 1024] uS (frozen/unselected)
- sensory_A.weight: [400, 1024] uS (frozen/unselected)
- sensory_B.weight: [400, 1024] uS (frozen/unselected)
- sensory_C.weight: [400, 1024] uS (frozen/unselected)
