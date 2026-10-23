# Toolbox single-neuron compatibility

Implement Task 1 of P23 with SNNLab custom dynamics: gate-first forward Euler,
continuous sodium/potassium conductances and unchanged current injection.
The 50 ms, 0.001 ms protocol uses float64 on CPU. Numerical agreement with an
independent scalar Euler implementation does not replace missing Toolbox traces.
The proposal omits capacitance, conductance and current units; nF/µS/nA embed
its numerical values consistently with mV and ms, pending Toolbox confirmation.

1. `compute.py [--run-id RESERVED_ID]`: save native bundle, start snapshot,
   circuit, `traces.npz` and execution timing in a new compute run.
2. `analyse.py --source COMPUTE_ID [--run-id RESERVED_ID]`: validate saved
   traces against the independent scalar Euler equations.
3. `present.py --source ANALYSE_ID [--run-id RESERVED_ID]`: save one four-panel
   voltage/gates/conductances/current figure as SVG and PNG.
4. `uv run pytest experiments/exp146/test.py -q`: run focused numerical and
   native execution integration tests.

Run each stage independently from the repository root with
`uv run python experiments/exp146/<stage>.py ...`. Trace states are post-step
samples at `time_ms`; `current_na` applies over intervals beginning at
`input_time_ms`. All trace arrays have 50,000 samples in the full protocol.
