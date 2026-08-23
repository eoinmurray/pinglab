# Exp097 ScientificCollectionState

## Registration

- Writing: `writings/exp097.typ`
- Collection: `snnlang`
- Status: draft `ExpScoutPlan`; not executed
- Prospective schematic: `artifacts/data/exp097/ping_engine_storyboard.svg`

## Established technical dependencies

- SNNLANG graph execution with `recording="full"` already records population
  voltages and per-projection conductances in addition to authored spike
  observables. This is an execution prerequisite, not an investigation.
- Demolab already supports HTML video output. Web delivery is an implementation
  acceptance check, not scientific evidence and not an investigation.
- Keep implementation local. Do not use paid compute.

## Execution handoff

1. Add an exp097 runner that recreates the frozen exp084 active-gamma condition:
   80 E cells, 20 I cells, 128 homogeneous Poisson input channels at 100 Hz,
   `dt=0.1 ms`, `tau_GABA=2 ms`, network seed 83, five predeclared input seeds,
   and 500 ms per trial.
2. Request full recordings and preserve E/I spikes, E/I voltage,
   E-to-I AMPA conductance, and I-to-E GABA conductance before aggregation.
3. Implement the three frozen scientific analyses in `writings/exp097.typ`.
4. Render the measured animation from a compact, versioned state artifact. Keep
   biological time, playback slowdown, units, source keys, cycle boundaries,
   trial selection, and checksums in provenance.
5. Encode a looping H.264 MP4 with a static poster and transcript. Verify the
   built Demolab page at mobile and desktop widths without editing `.demolab/`.

## Publication blockers

- No experiment runner or measured state artifact exists.
- No measured animation, poster, transcript, or observed result exists.
- The current SVG is a design schematic and must remain visibly labelled as
  prospective until execution supplies measured evidence.
