# exp153 — Mean-field coupling

Define two reciprocal E/I circuits locally and measure a 143-condition locking
map with a delayed rate-conductance closure. Both circuits include all
four local pathways and private sensory input; no attention input is present.
The model has 14 conductance filters and four population-rate variables. Local
seeded initialization supplies weights without advancing the spiking reference.
The neuronal reference follows exp133’s network and projection order, with
locally tuned recurrence and rate relaxation; sensory gain is applied once
during mean-field reduction. Parameter tuning is qualitative and uses one seed.

## CLI

1. `compute.py`: optional `--difference-hz` (−4), `--coupling-us` (0.09),
   `--parallel` (1), `--onset-only` and `--run-id`. No upstream source is required. Runs the full
   143-condition grid for 5,000 ms; difference and coupling select the onset
   excerpt, without changing sweep coverage. Saves its own native initialization,
   graph, models and trajectories. `--onset-only` runs just the selected
   1,500 ms condition for Figure 2, without computing the map.
   Optional `--tau-rate-e-ms` (5), `--tau-rate-i-ms` (5) and
   `--sensory-gain` (9.2) tune the closure response times and sensory drive;
   every trial records its resolved settings.
   `--recurrent-ee-us` (0.4), `--recurrent-ii-us` (0.8) set nominal self-pathway
   totals; `--local-feedback-scale` (1.5) scales both E→I and I→E weight
   distributions before native initialization.
   `--cross-delay-ms` (2) sets the delay of all four cross-circuit pathways.
   `--cross-tau-ms` (0.5) sets their synaptic conductance decay time.
2. `analyse.py`: required `--source COMPUTE_ID`; optional `--run-id`.
   Accepts this experiment's standalone compute, validates coverage and native
   trajectories, calibrates detuning from each zero-coupling condition and
   measures locking over 800–5,000 ms. Extracts the selected first 1,500 ms.
3. `present.py`: required `--source ANALYSE_ID`; optional `--run-id` and
   `--map-source PRESENT_ID`. Onset-only analysis requires an explicit map
   presentation to preserve Figure 1 byte-for-byte with its original provenance.
   Exports the map and onset figures in SVG, PNG and PDF, the native model,
   numerical results and circuit diagram with routed self-loops and weight labels.
   Use `--fig2-only --output PATH` with an explicit analysis source to render
   only a Figure 2 preview (PNG/SVG/PDF and source pins), without a map source
   or a new full-report presentation.
4. `test.py`: `uv run pytest experiments/exp153/test.py -q`.

Compute has no scientific input runs. Analysis and presentation consume only
explicitly identified runs from this experiment. Stages execute independently;
completed historical evidence retains its original immutable provenance.
