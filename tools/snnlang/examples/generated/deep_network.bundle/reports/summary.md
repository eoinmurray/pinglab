# snnlang report — deep_ping_hierarchy

Populations: 6 (960 units)
Projections: 9
Operations: 2
Parameters: 10 tensors / 517,120 scalars
Estimated state: 960 scalars per sample and timestep
Estimated dense projection edges: 514,560
Trainable this recipe: 10 tensors
Outputs: gesture_logits
Recurrent paths: association_E_to_I, association_I_to_E, decision_E_to_I, decision_I_to_E, encoder_E_to_I, encoder_I_to_E
Diagnostics: 0 errors, 0 warnings

## Populations
- association_E: 256 × coba_lif (spiking)
- association_I: 64 × coba_lif (spiking)
- decision_E: 128 × coba_lif (spiking)
- decision_I: 32 × coba_lif (spiking)
- encoder_E: 384 × coba_lif (spiking)
- encoder_I: 96 × coba_lif (spiking)

## Projections
- association_E_to_I: association_E.spikes → association_I.excitatory [recurrent, excitatory]
- association_I_to_E: association_I.spikes → association_E.inhibitory [recurrent, inhibitory]
- association_input: encoder_E.spikes → association_E.excitatory [feedforward, excitatory]
- decision_E_to_I: decision_E.spikes → decision_I.excitatory [recurrent, excitatory]
- decision_I_to_E: decision_I.spikes → decision_E.inhibitory [recurrent, inhibitory]
- decision_input: association_E.spikes → decision_E.excitatory [feedforward, excitatory]
- encoder_E_to_I: encoder_E.spikes → encoder_I.excitatory [recurrent, excitatory]
- encoder_I_to_E: encoder_I.spikes → encoder_E.inhibitory [recurrent, inhibitory]
- encoder_input: events.value → encoder_E.excitatory [feedforward, excitatory]

## Parameters
- association_E_to_I.weight: [64, 256] nS (selected)
- association_I_to_E.weight: [256, 64] nS (selected)
- association_input.weight: [256, 384] nS (selected)
- decision_E_to_I.weight: [32, 128] nS (selected)
- decision_I_to_E.weight: [128, 32] nS (selected)
- decision_input.weight: [128, 256] nS (selected)
- encoder_E_to_I.weight: [96, 384] nS (selected)
- encoder_I_to_E.weight: [384, 96] nS (selected)
- encoder_input.weight: [384, 700] nS (selected)
- gesture_readout_projection.weight: [20, 128] 1 (selected)
