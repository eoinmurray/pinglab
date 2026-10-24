# exp148 — External-I-drive transition

Test whether reducing only direct external excitation of inhibitory neurons
produces population oscillations in a shared-input COBA network derived from
exp145. The 4,000 E / 1,000 I network has full recurrence and full Input→E/I
strength from initialization. After 50 ms burn-in, measure three 750 ms windows:
AI target, AI-to-PING transition, and PING target. Only Input→I gain falls,
linearly from one to zero during the middle window. Input→E and all recurrent
weights stay at reference strength; source rate stays at 2.5 Hz/source.
One 4,000-source pool feeds fixed 5% input masks; recurrent connectivity is 20%.
State is continuous throughout the 2,300 ms simulation. AI and PING are protocol
targets; state classification and the oscillation mechanism remain unestablished.

## CLI

1. `compute.py`: run the three-window trajectory. Optional `--seed` (42),
   `--device` (cpu), `--compile-mode` (`off` or `default`; default `off`),
   `--n-e` / `--n-i` (4000 / 1000), `--record-state`, and `--run-id`.
   `SNNLAB_DEVICE` overrides `--device`. `--record-state` additionally saves
   native population mean voltage and mean conductance for each pathway.
   Alternatively, `--mean-field-source SPIKING_COMPUTE_ID` runs the four-state
   exp117 extension using that run's recorded protocol and sampled matrices;
   it reuses the immutable spiking evidence without rerunning the network.
2. `analyse.py`: `--source COMPUTE_ID` (required), `--run-id` (optional).
   Add `--mean-field MODEL_COMPUTE_ID` to include a model tied to the same source.
   Validate the trajectory and measure all three windows plus local hold timing.
3. `present.py`: `--source ANALYSE_ID` (required), `--run-id` (optional).
   Render the raster, full-width Input→I weight schedule and three window spectra.
4. `test.py`: `uv run pytest experiments/exp148/test.py -q`.
