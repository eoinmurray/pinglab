# snnlang report — mnist_ping_training_demo

Populations: 3 (170 units)
Projections: 4
Operations: 1
Parameters: 4 tensors / 109,824 scalars
Estimated state: 170 scalars per sample and timestep
Estimated dense projection edges: 109,824
Trainable this recipe: 2 tensors
Outputs: class_logits
Recurrent paths: sensory_ping_E_to_I, sensory_ping_I_to_E
Diagnostics: 0 errors, 0 warnings

## Populations
- classifier_integrator: 10 × leaky_integrator (non-spiking)
- sensory_ping_E: 128 × coba_lif (spiking)
- sensory_ping_I: 32 × coba_lif (spiking)

## Projections
- classifier_projection: sensory_ping_E.spikes → classifier_integrator.excitatory [feedforward, excitatory]
- sensory_ping_E_to_I: sensory_ping_E.spikes → sensory_ping_I.excitatory [recurrent, excitatory]
- sensory_ping_I_to_E: sensory_ping_I.spikes → sensory_ping_E.inhibitory [recurrent, inhibitory]
- sensory_ping_input: image_spikes.value → sensory_ping_E.excitatory [feedforward, excitatory]

## Parameters
- classifier_projection.weight: [10, 128] nS (selected)
- sensory_ping_E_to_I.weight: [32, 128] nS (frozen/unselected)
- sensory_ping_I_to_E.weight: [128, 32] nS (frozen/unselected)
- sensory_ping_input.weight: [128, 784] nS (selected)
