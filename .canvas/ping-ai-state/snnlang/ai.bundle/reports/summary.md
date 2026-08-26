# snnlang report — balanced_ai_state

Populations: 3 (510 units)
Projections: 7
Operations: 1
Parameters: 7 tensors / 454,000 scalars
Estimated state: 510 scalars per sample and timestep
Estimated dense projection edges: 454,000
Trainable this recipe: 0 tensors
Outputs: state_logits
Recurrent paths: balanced_circuit_E_to_E, balanced_circuit_E_to_I, balanced_circuit_I_to_E, balanced_circuit_I_to_I
Diagnostics: 0 errors, 0 warnings

## Populations
- balanced_circuit_E: 400 × coba_lif (spiking)
- balanced_circuit_I: 100 × coba_lif (spiking)
- state_readout_integrator: 10 × leaky_integrator (non-spiking)

## Projections
- balanced_circuit_E_to_E: balanced_circuit_E.spikes → balanced_circuit_E.excitatory [recurrent, excitatory]
- balanced_circuit_E_to_I: balanced_circuit_E.spikes → balanced_circuit_I.excitatory [recurrent, excitatory]
- balanced_circuit_I_to_E: balanced_circuit_I.spikes → balanced_circuit_E.inhibitory [recurrent, inhibitory]
- balanced_circuit_I_to_I: balanced_circuit_I.spikes → balanced_circuit_I.inhibitory [recurrent, inhibitory]
- balanced_circuit_input_E: afferent_e.value → balanced_circuit_E.excitatory [feedforward, excitatory]
- balanced_circuit_input_I: afferent_i.value → balanced_circuit_I.excitatory [feedforward, excitatory]
- state_readout_projection: balanced_circuit_E.spikes → state_readout_integrator.excitatory [feedforward, excitatory]

## Parameters
- balanced_circuit_E_to_E.weight: [400, 400] nS (frozen/unselected)
- balanced_circuit_E_to_I.weight: [100, 400] nS (frozen/unselected)
- balanced_circuit_I_to_E.weight: [400, 100] nS (frozen/unselected)
- balanced_circuit_I_to_I.weight: [100, 100] nS (frozen/unselected)
- balanced_circuit_input_E.weight: [400, 400] nS (frozen/unselected)
- balanced_circuit_input_I.weight: [100, 400] nS (frozen/unselected)
- state_readout_projection.weight: [10, 400] nS (frozen/unselected)
