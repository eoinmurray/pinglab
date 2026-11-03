# exp168 — AI-like and PING states without EE or II recurrence

Preserve the selected seed-43 scratch states as a native, independently analysable
experiment. Both have EE and II absent. The AI-like state uses external rate
1.25 Hz/source, input probability 0.005 and constant input-to-I gain 0.3. The
strong PING state uses 2.5 Hz/source, probability 0.05 and gain zero. Input jump
scaling compensates connection probability. This is a three-parameter endpoint
comparison, not a single-parameter transition.

The AI-like state narrowly missed the scratch search's conservative maximum-I-bin
screen (0.033 versus 0.030). The later gain-0.3 reversal of the strongest PING
state remained rhythmic and is not the selected AI-like state. No new simulation
was performed for migration. Original executed graph names and source provenance
are preserved; source-store identities are scoped to their original isolated store.

- `recipe.py`: fresh resolved settings and native network authoring.
- `dataset.py`: reproducible Poisson input encoding.
- `compute.py --import-store PATH`: validate the two explicit source IDs/digests,
  import native bundles, snapshots and raw spikes, and archive original records/code.
  Import creates a new run; it never edits the source store.
- `compute.py --simulate [--seed 43] [--device cpu]`: explicitly simulate both
  conditions using the standalone recipe. Requires simulation authorization.
- `analyse.py --source COMPUTE_ID`: native rhythmicity, native per-cell ISI CV,
  rates, spectra and correlations from raw events.
- `present.py --source ANALYSE_ID`: SVG/PDF/PNG raster comparison and report data.
- `test.py`: focused non-simulation checks of configuration, native graphs,
  reproducibility metadata and source independence.

Run stages with `.venv/bin/python experiments/exp168/<stage>.py`, or the
repository's equivalent `uv run python` commands. Every stage accepts `--run-id`.
Source-run arguments are explicit; there is no latest-input selection. The report
and appendix use the core-selected presentation and its pinned compute snapshots.

The retained evidence includes native start parameters and sparse events for both
states. Plotting and analysis need no access to the original import directory.
The two-second, single-seed measurements describe these selected realizations;
they do not establish a phase boundary or robustness across seeds.
