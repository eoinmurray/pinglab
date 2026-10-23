# Toolbox half-centre oscillator

Implement Task 2 of P23: one pair of sigmoid-gated sodium/potassium neurons
connected by two explicit voltage-driven inhibitory synapses in SNNLab 0.24.
Run the proposal's 200 ms protocol at 0.001 ms in float64 on CPU. No batching
benchmark is included. Independent Euler agreement does not replace missing
Toolbox reference traces. Current schedules are internal to each neuron because
native custom neuron ports share one unit (µS here); this fixed protocol does
not expose an interactive current-input interface. nF/µS/nA preserve the
proposal's numerical values consistently with mV/ms; C/g/I units are assumed.

1. `compute.py [--run-id RESERVED_ID]`: save the native bundle, circuit,
   simulation-start snapshot, full `traces.npz` and execution timing.
2. `analyse.py --source COMPUTE_ID [--run-id RESERVED_ID]`: check traces against
   independent Euler equations and measure spike counts and alternation.
3. `present.py --source ANALYSE_ID [--run-id RESERVED_ID]`: save one four-panel
   figure with both voltages, synaptic conductances, phase portrait and currents.
4. `uv run pytest experiments/exp147/test.py -q`: numerical and native execution
   checks, including current boundaries and continuation.

Run stages independently from the repository root with
`uv run python experiments/exp147/<stage>.py ...`. Trace states use post-step
`time_ms`; currents apply over intervals beginning at `input_time_ms`.
