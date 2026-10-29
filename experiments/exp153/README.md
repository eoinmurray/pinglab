# exp153 — Mean-field coupling

Define two reciprocal E/I circuits locally and measure a 143-condition locking
map with an unfitted delayed rate-conductance closure. Both circuits include all
four local pathways and private sensory input; no attention input is present.
The model has 14 conductance filters and four population-rate variables. Local
seeded initialization supplies weights without advancing the spiking reference.
The neuronal reference matches exp133’s default network, including projection
order; sensory gain is applied once during mean-field reduction.

## CLI

1. `compute.py`: optional `--difference-hz` (−4), `--coupling-us` (0.09),
   `--parallel` (1), `--onset-only` and `--run-id`. No upstream source is required. Runs the full
   143-condition grid for 5,000 ms; difference and coupling select the onset
   excerpt, without changing sweep coverage. Saves its own native initialization,
   graph, models and trajectories. `--onset-only` runs just the selected
   1,500 ms condition for Figure 2, without computing the map.
2. `analyse.py`: required `--source COMPUTE_ID`; optional `--run-id`.
   Accepts this experiment's standalone compute, validates coverage and native
   trajectories, calibrates detuning from each zero-coupling condition and
   measures locking over 800–5,000 ms. Extracts the selected first 1,500 ms.
3. `present.py`: required `--source ANALYSE_ID`; optional `--run-id` and
   `--map-source PRESENT_ID`. Onset-only analysis requires an explicit map
   presentation to preserve Figure 1 byte-for-byte with its original provenance.
   Exports the map and onset figures in SVG, PNG and PDF, the native model,
   numerical results and circuit diagram with routed self-loops and weight labels.
4. `test.py`: `uv run pytest experiments/exp153/test.py -q`.

Compute has no scientific input runs. Analysis and presentation consume only
explicitly identified runs from this experiment. Stages execute independently;
completed historical evidence retains its original immutable provenance.
