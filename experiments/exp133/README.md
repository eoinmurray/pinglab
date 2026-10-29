# exp133 — Lowet (2015): Coupled Rhythms

Map finite-time phase locking between reciprocally coupled A/B circuits, then
show coupling onset at one selected map point. The revised network includes
local E→E, E→I, I→E and I→I recurrence in both circuits and independent private
inputs to A and B: 14 projections in total. Both directions of inter-circuit
excitation target E and I. Local self-pathways use illustrative CTC-note totals.
The unused attention input and its projection are absent from the current circuit.

The full runset contains one compute, one analyse and one present run. Analysis
extracts the onset diagnostic directly from the selected saved map trajectory.
Historical v1/v2 evidence remains readable with its original architecture. A standalone
onset preview can also be generated without a map, using freshly initialized weights.

## CLI

1. `compute.py`: `--protocol grid|sender-receiver` (optional; default `grid`),
   `--parameter-source <compute-run-id>` (optional), `--run-id <id>`
   (optional). Without a source, sender–receiver runs only the 1,500 ms onset
   diagnostic with freshly initialized weights. Supplied parameter sources must
   use v3 connectivity without attention.
2. `analyse.py`: `--source <compute-run-id>` (required), `--run-id <id>` (optional).
3. `present.py`: `--source <grid-analyse-run-id>` (required), `--run-id <id>`
   (optional). Historical split analyses also need
   `--sender-receiver <onset-analyse-run-id>`. For a standalone local Fig. 3 preview, use
   `--fig3-only --sender-receiver <onset-analyse-run-id> --output <directory>`;
   writes PNG/SVG and source pins without selecting a new report presentation.
4. `test.py`: `uv run pytest experiments/exp133/test.py`; optional pytest arguments.

Refresh figures and diagrams using an explicit presentation's same analysis
and compute inputs: `uv run python experiments/exp133/present.py
--redraw-from PRESENT_ID [--run-id ID]`.
