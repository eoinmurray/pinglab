# exp160 — Spiking one-way entrainment

One matched pair of conductance-LIF simulations, with the neuronal-reference
weights pinned to an explicit exp159 compute. Both trials receive the same full
private input spike trains; A→B is disabled in the control and enabled at 500 ms
in the coupled trial. No B→A feedback, probes, training or competition task.

## CLI

1. `compute.py`: required `--parameter-source COMPUTE_ID` (exp159 native reference);
   optional `--coupling-us` (0.28 µS), `--difference-hz` (6.5 Hz/channel), `--run-id`.
   Runs only the 1,500 ms pair on CPU. Saves actual execution-start snapshots,
   binary inputs/spikes and native per-cell conductance recordings with units and
   sample coordinates. Control/coupled local weights, prefix spikes and all sender
   spikes are checked; cross strengths deliberately differ.
2. `analyse.py`: required `--source COMPUTE_ID`; optional `--run-id`. Validates
   binary arrays, native conductance metadata and scaled native initial weights.
   Rates use 1 ms Gaussian smoothing; phase metrics use 800–1,500 ms.
3. `present.py`: required `--source ANALYSE_ID`; optional `--run-id`. Creates the
   two-column SVG/PNG/PDF and styled native network diagram. `--redraw-from PRESENT_ID`
   redraws only the network while preserving scientific figures/numbers.
4. `test.py`: `uv run pytest experiments/exp160/test.py -q`.

The original 0.06 µS pair failed to lock. A 0.24 µS pair locked but did not
reliably place input peaks before receiver output. Subsequent paired iterations
changed one parameter at a time (or returned to an already tested setting before
changing another). The selected pair uses A/B 28.25/21.75 Hz/channel and 0.28 µS.
It passes the unchanged phase-locking criteria and the explicit illustrative
alignment criterion: at least 80% of input-conductance peaks inside the existing
low-inhibition windows, and at least 80% followed by a raw B E population-spike
peak 0.2–10 ms later. Selected fractions are 19/19 and 16/19 respectively.
This is an exploratory selection rule, not an established CTC test or a causal
transmission assay. Peak timing uses native conductance and unsmoothed spikes.

`analyse.py --selection-source ANALYSE_ID` may be repeated to retain the explicit
pre-selection analyses as scientific provenance. Those references are not used
to run or alter the selected pair. The final analysis pins the complete set of
prior paired conditions. All original runs remain immutable. Seed robustness,
preferential information transmission and attention remain untested.

## Iteration record

| Coupling (µS) | A-minus-B drive (Hz/channel) | Locking | Timing fraction | Analysis |
| --- | --- | --- | --- | --- |
| 0.06 | 2 | No | Not measured initially | r002 |
| 0.24 | 2 | Yes | Not measured initially | r006 |
| 0.16 | 2 | Yes | 4/18 | r009 |
| 0.16 | 4 | No | 10/18 | r011 |
| 0.24 | 4 | Yes | 11/18 | r013 |
| 0.24 | 6 | No | 10/18 | r015 |
| 0.30 | 6 | Yes | 13/18 | r017 |
| 0.28 | 6 | Yes | 14/18 | r019 |
| 0.27 | 6 | Yes | 14/18 | r021 |
| 0.26 | 6 | No | 14/18 | r023 |
| 0.30 | 6.5 | Yes | 15/19 | r025 |
| 0.28 | 6.5 | Yes | 16/19 | r027 |

Analysis suffixes identify immutable `exp160-rNNN-analyse` runs. Timing fraction
counts peaks followed by raw receiver output 0.2–10 ms later, not the separate
low-inhibition fraction; both timing fractions must reach 80%. The first two
analyses predate this timing rule. Failed and near-passing cases were retained
rather than changing the 80% threshold, lag range, shading threshold or windows.
