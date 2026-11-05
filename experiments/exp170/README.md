# exp170 — COBA with count-matched asynchronous inhibition

Compare one frozen, unpenalised TR02 COBA response with PING and COBA receiving
private GABA inputs. Match exactly 24 events per source over 200 ms, plus each
E neuron's PING inhibitory-conductance mean and approximately its temporal SD.
The report follows exp169: Abstract, Results with three diagnostic figures then the measured
statistics table, five-step Methods, and appendix diagrams/parameters.

1. `uv run python experiments/exp170/compute.py --import-source RETAINED_ID`
   imports authenticated existing evidence without simulation. The migration
   source is `exp169-r014-compute-phase-scratch-count-matched-inhibition`.
2. `uv run python experiments/exp170/compute.py` explicitly calibrates private
   timing and strength and simulates one native frozen CPU trial. Requires
   simulation authorization. No upstream training or simulation is launched.
3. `uv run python experiments/exp170/analyse.py --source COMPUTE_ID` validates
   exact counts, checkpoint/snapshot parameters, GABA filtering and moment errors,
   then measures source rates, correlations and E firing without simulation.
4. `uv run python experiments/exp170/present.py --source ANALYSE_ID --traces TRACE_COMPUTE_ID` renders house
   style raster/voltage/conductance figures, diagrams and the shared evidence payload for the report/table.
5. `uv run pytest experiments/exp170/test.py` runs focused non-simulation checks.

Stage commands accept optional `--run-id ID`. Imports do not execute stages.
Code and rendering have no dependency on `.scratch` or another experiment's code.
The imported payload, original configuration, native snapshots and original
execution timing remain unchanged and explicitly pinned in Pingstore.

There are 1024 private channels, not 256 I neurons. Timing and amplitudes are
calibrated independently per recipient without examining E output. These are
count-constrained inputs, not Poisson events; feedback and higher-order timing
remain different, and COBA/PING have different trained weights. This is one
image and artificial seed, without retraining or accuracy evaluation.

`compute.py --record-traces COMPUTE_ID` replays only the retained artificial
trial, without training or recalibration, and requires simulation authorization.
It retains E neuron 512’s voltage and excitatory/inhibitory conductances in a
new `membrane-traces` compute phase, verifying all original outputs exactly.
Pass that phase ID to `present.py --traces`. The COBA and PING panels use their
original saved traces; each condition gets its own three-panel figure.
