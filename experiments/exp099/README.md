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

- Compute retains the authored bundle, simulation snapshot/configuration and
  seeded initialized weights in its export. The simulator command is recorded
  under provenance and in the authoritative execution record.
- Analyse retains numerical measurements and results, referencing compute.
- Present references both analysis and its pinned compute source. Its flat
  export contains the network diagram, poster, video, numbers and the shared
  publication-metadata projection. Publication is a separate authorized step.

## Production renderer recovery

The missing source `.canvas/ping-ai-state/render_emergence_style.py` was recovered
from the parent of Git commit `27e9ba5d7e681859c28f9ec589cd6bd1cb37fd95`, which
deleted it. It is now `render.py`; this is not the old `render_approximation`.
Recovery inspected versioned source only, not historical run evidence.

The input/story/inside-band configuration, seven-panel layout, colours, transmission
sampling, ridgelines, frame pacing, poster selection and encoding settings are
retained. The entrypoint is import-safe, uses explicit validated source data,
and receives measurements computed by analyse. Missing authenticated inputs
are errors rather than triggers reconstructing random input or substituting zeros.
Labels no longer assert an established AI-to-PING transition: they name the
baseline, afferent bout and recovery phases instead.

The former summary excludes the final 1,800 ms rhythmicity-window centre; the
production plot includes it. Both conventions are retained explicitly. The
conductance-loop score and its run-relative normalization are retained, not
recomputed during presentation. Exponential traces in the renderer control
animated transmission intensity; they do not execute the network.

This refactor does not establish new scientific results or authorize execution,
historical migration, selection, materialization or publication. Fixture tests
exercise the stage contract and renderer separately from a production run.
