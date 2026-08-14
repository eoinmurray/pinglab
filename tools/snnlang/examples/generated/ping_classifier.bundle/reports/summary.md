# snnlang report — ping_classifier

Populations: 3 (330 units)
Projections: 6
Operations: 1
Parameters: 6 tensors / 305,664 scalars
Estimated state: 330 scalars per sample and timestep
Estimated dense projection edges: 305,664
Trainable this recipe: 2 tensors
Outputs: class_logits
Recurrent paths: sensory_ping_E_to_E, sensory_ping_E_to_I, sensory_ping_I_to_E, sensory_ping_I_to_I
Diagnostics: 0 errors, 0 warnings

## Populations
- classifier_integrator: 10 × leaky_integrator (non-spiking)
- sensory_ping_E: 256 × coba_lif (spiking)
- sensory_ping_I: 64 × coba_lif (spiking)

## Projections
- classifier_projection: sensory_ping_E.spikes → classifier_integrator.excitatory [feedforward, excitatory]
- sensory_ping_E_to_E: sensory_ping_E.spikes → sensory_ping_E.excitatory [recurrent, excitatory]
- sensory_ping_E_to_I: sensory_ping_E.spikes → sensory_ping_I.excitatory [recurrent, excitatory]
- sensory_ping_I_to_E: sensory_ping_I.spikes → sensory_ping_E.inhibitory [recurrent, inhibitory]
- sensory_ping_I_to_I: sensory_ping_I.spikes → sensory_ping_I.inhibitory [recurrent, inhibitory]
- sensory_ping_input: image.value → sensory_ping_E.excitatory [feedforward, excitatory]

## Parameters
- classifier_projection.weight: [10, 256] nS (selected)
- sensory_ping_E_to_E.weight: [256, 256] nS (frozen/unselected)
- sensory_ping_E_to_I.weight: [64, 256] nS (frozen/unselected)
- sensory_ping_I_to_E.weight: [256, 64] nS (frozen/unselected)
- sensory_ping_I_to_I.weight: [64, 64] nS (frozen/unselected)
- sensory_ping_input.weight: [256, 784] nS (selected)
