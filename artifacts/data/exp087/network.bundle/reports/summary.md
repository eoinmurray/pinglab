# snnlang report — diesmann_synfire_chain

Populations: 6 (600 units)
Projections: 12
Operations: 0
Parameters: 12 tensors / 420,000 scalars
Estimated state: 600 scalars per sample and timestep
Estimated dense projection edges: 420,000
Trainable this recipe: 0 tensors
Outputs: none
Recurrent paths: none
Diagnostics: 0 errors, 0 warnings

## Populations
- pool_1: 100 × coba_lif (spiking)
- pool_2: 100 × coba_lif (spiking)
- pool_3: 100 × coba_lif (spiking)
- pool_4: 100 × coba_lif (spiking)
- pool_5: 100 × coba_lif (spiking)
- pool_6: 100 × coba_lif (spiking)

## Projections
- background_to_pool_1: independent_background.value → pool_1.excitatory [feedforward, excitatory]
- background_to_pool_2: independent_background.value → pool_2.excitatory [feedforward, excitatory]
- background_to_pool_3: independent_background.value → pool_3.excitatory [feedforward, excitatory]
- background_to_pool_4: independent_background.value → pool_4.excitatory [feedforward, excitatory]
- background_to_pool_5: independent_background.value → pool_5.excitatory [feedforward, excitatory]
- background_to_pool_6: independent_background.value → pool_6.excitatory [feedforward, excitatory]
- packet_to_pool_1: pulse_packet.value → pool_1.excitatory [feedforward, excitatory]
- pool_1_to_pool_2: pool_1.spikes → pool_2.excitatory [feedforward, excitatory]
- pool_2_to_pool_3: pool_2.spikes → pool_3.excitatory [feedforward, excitatory]
- pool_3_to_pool_4: pool_3.spikes → pool_4.excitatory [feedforward, excitatory]
- pool_4_to_pool_5: pool_4.spikes → pool_5.excitatory [feedforward, excitatory]
- pool_5_to_pool_6: pool_5.spikes → pool_6.excitatory [feedforward, excitatory]

## Parameters
- background_to_pool_1.weight: [100, 600] nS (frozen/unselected)
- background_to_pool_2.weight: [100, 600] nS (frozen/unselected)
- background_to_pool_3.weight: [100, 600] nS (frozen/unselected)
- background_to_pool_4.weight: [100, 600] nS (frozen/unselected)
- background_to_pool_5.weight: [100, 600] nS (frozen/unselected)
- background_to_pool_6.weight: [100, 600] nS (frozen/unselected)
- packet_to_pool_1.weight: [100, 100] nS (frozen/unselected)
- pool_1_to_pool_2.weight: [100, 100] nS (frozen/unselected)
- pool_2_to_pool_3.weight: [100, 100] nS (frozen/unselected)
- pool_3_to_pool_4.weight: [100, 100] nS (frozen/unselected)
- pool_4_to_pool_5.weight: [100, 100] nS (frozen/unselected)
- pool_5_to_pool_6.weight: [100, 100] nS (frozen/unselected)
