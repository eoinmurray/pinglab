# EXP099: richer-input probe

The implemented scope is one richer-input probe, not the article's planned
simplified-versus-richer controlled comparison. The scientific network and
simulation specification are preserved from the former flat runner.

## Independent stages

```sh
uv run python experiments/exp099/compute.py
uv run python experiments/exp099/analyse.py --source <compute-run-id>
uv run python experiments/exp099/present.py --source <analyse-run-id>
```

Each command completes exactly one source-neutral `pingstore.run/v4` run.
`--run-id` accepts an unused stage identity reserved before dispatch. Analysis
and presentation validate explicit inputs, including payload and manifest pins;
they never launch upstream work. No command materializes, copies web assets,
selects a published run, or migrates historical data.

- Compute retains the data-only authored bundle, simulation snapshot/configuration
  and seeded initialized weights in its export. The simulator command is recorded
  under provenance and in the authoritative execution record.
- Analyse retains numerical measurements and results, referencing compute.
- Present references both analysis and its pinned compute source. It lowers the
  authenticated `snnlang` bundle into a structural diagram and renders it through
  `snnviz`; its flat export contains that network diagram, explanatory input map,
  poster, video, numbers and the shared publication-metadata projection.
  Publication is a separate authorized step.

## Production renderer recovery

The missing source `.canvas/ping-ai-state/render_emergence_style.py` was recovered
from the parent of Git commit `27e9ba5d7e681859c28f9ec589cd6bd1cb37fd95`, which
deleted it. It is now `render.py`; this is not the old `render_approximation`.
Recovery inspected versioned source only, not historical run evidence.

The input/story/inside-band configuration, seven-panel layout, transmission
sampling, ridgelines, frame pacing, poster selection and encoding settings are
retained. The visual styling now follows the `snnviz` guide: an opaque white
background, the shared black/red scientific palette and monospace typography,
with the poster rendered at 240 DPI. The entrypoint is import-safe, uses explicit
validated source data, and receives measurements computed by analyse. Missing
authenticated inputs are errors rather than triggers reconstructing random input
or substituting zeros.

The video and poster use a compact 2:1 canvas shaped around the scientific
content. There is no globally reserved player-control band; the diagnostic
panels occupy the right half of the frame.

The complete frame is now declared through `snnviz.FigureGrid`: a bounded
header and half-width animated network. On the right, panels B and C form the
top row, equal-height panels D and E share the lower-left cell, and panel F
occupies the lower-right cell. The named and nested grids replace duplicated
normalized axis coordinates and are resolved once for the poster and all 600
animation frames.

Panel A stacks E-targeting spikes, shared spikes, AMPA conductance, GABA
conductance and I-targeting spikes down the left. Spike sources are compact
neuron grids; conductance sources are running traces. The E and I populations
share one aligned width on the right, with the E box four times the area of the
I box. Neuron grids fill their frames with only marker-safe edge insets.
Every grid uses the same five-point physical inset on all four sides, independent
of its frame dimensions. The two internal stacks share a 0.25-inch physical
inset from every outer edge of panel A, and their rows are distributed evenly
between those aligned bounds.
Authenticated afferent and
recurrent spikes continue to illuminate their actual source-to-target paths;
aggregate conductance arrows vary with the retained conductance signals.

The frame retains concise A–F panel titles and labels the five inputs and two
populations inside panel A. Equal-height panels D and E respectively show E/I
population firing-rate traces computed over a fixed 20 ms display window and
the retained shared, E-private and I-private afferent multipliers. The multiplier
colours match their source grids in panel A; the two private traces coincide
because this condition gives them the same envelope, so the I-private trace is
dashed. Panel E carries the common time axis, while synchronized animated
cursors connect both panels. It
preserves panel boundaries, data marks, network structure and animation state
while suppressing legends, axis labels, tick labels, values and dynamic textual
readouts.

The former summary excludes the final 1,800 ms rhythmicity-window centre; the
production plot includes it. Both conventions are retained explicitly. The
conductance-loop score and its run-relative normalization are retained, not
recomputed during presentation. Exponential traces in the renderer control
animated transmission intensity; they do not execute the network.

This refactor does not establish new scientific results or authorize execution,
historical migration, selection, materialization or publication. Fixture tests
exercise the stage contract and renderer separately from a production run.
