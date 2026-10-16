# exp134 — Leak and capacitance modulation of PING dynamics

Partner to exp123 in the Neuromodulation collection. Figure 1 varies leak
conductance; Figure 2 varies capacitance. Each intervention scales E and I
together over 19 log-spaced multipliers from 0.5 to 2. The other parameter,
GABA decay (6 ms), weights and all other circuit settings stay fixed.

The current recipe uses all 50 images per digit from exp123's balanced
500-image bank (selection seed 134), with ten paired encodings per image
(identifiers 42–51). Fresh encodings follow the original seed rule. Both sweeps
share the exact baseline and inputs: 37 configurations, 185,000 presentations.
Each presentation lasts 200 ms at 0.1 ms resolution and 25 Hz maximum pixel rate.
The selected article presentation still reports the completed 100-image run
described in the historical execution below.

The experiment owns its recipe, execution, estimators and plots. It imports
maintained shared helpers and public SNNLab APIs, never another experiment's
implementation in production. Tests compare the baseline and estimator definitions
with exp123. Compute uses explicit checkpoint and raw-image data inputs; analyse
measures its single compute result; present uses analysis activity measurements
and retained native diagrams from its validated compute ancestry.

```sh
.venv/bin/python experiments/exp134/compute.py --source exp022-r001-compute --image-source exp123-r068-compute
.venv/bin/python experiments/exp134/analyse.py --source <compute-run-id>
.venv/bin/python experiments/exp134/present.py --source <analyse-run-id>
```

Analysis defaults to four local workers; `--workers 1` selects serial execution.
Network bundles and native parameter snapshots are retained for every setting.
One shared circuit diagram is saved and shown in the article footer before
Dataset. Analysis records participation coverage; no uncertainty estimates
or additional evidence/accuracy figures are included.

## Modal CPU execution

The shared launcher transfers the explicitly named, validated source runs and
their ancestry to checksum-addressed paths in the Modal results volume. It pins
the direct inputs in the durable dispatch receipt and protects them from local
pruning until collection. The worker restores complete source runs, then invokes
the ordinary compute command with its reserved run ID. Collection rejects changed
or mismatched input references.

```sh
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML python -m modal run --detach experiments/helpers/modal_compute.py --experiment exp134 --gpu none --cpu 1 --timeout 21600 --source exp022-r001-compute --image-source exp123-r068-compute
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML python experiments/helpers/modal_jobs.py --run-id <printed-compute-run-id> --collect
```

This requests one physical CPU, 16 GiB RAM, no GPU and a six-hour timeout.
Compute remains sequential with one PyTorch thread and five-image batches.
The source upload includes the complete checkpoint and image runs, approximately
2.8 GB; use only with authorization to transfer those research artifacts to Modal.
Analyse and present remain separate commands after validated collection.

## Validation

On the first selected image (index 140, digit 6), all ten regenerated encoding
tensors matched exp123-r068-compute exactly. A separate model built from
exp123's current recipe also matched every spike under installed SNNLab 0.18.0.
After execution of the shared baseline in five-image batches, all E, I and
output spikes matched the retained exp123 recordings across all 100 selected
images and ten draws (1,000 presentations), with zero differences. This confirms
baseline equivalence, not equivalence of the interventions.

## Completed execution — 2026-10-16

1. `exp134-r001-compute`: all 37,000 presentations, with the pinned exp022
   checkpoint and raw pixels from `exp123-r068-compute`. Compute took 20 min
   31 s. The validated payload retains 74 native start/end snapshots and 37
   sets of circuit/training/expanded diagrams, plus the scientific recordings.
2. `exp134-r002-analyse`: all 37 conditions measured. Every setting has 1,000
   accepted draws across all 100 images; no participation observations are missing.
3. `exp134-r003-present`: initial presentation with two seven-panel dynamics
   figures in PNG/SVG/PDF, scientific numbers, and repeated native diagrams.
4. `exp134-r004-present`: same measurements and dynamics figures, with one
   shared circuit SVG instead of repeated diagram views. The article selects
   this presentation and places the diagram in its footer immediately before
   Dataset. Future compute runs likewise save only the shared diagram while
   retaining all parameter-specific bundles and snapshots. Completed historical
   runs remain immutable.

Across ×0.5 to ×2, total E firing and cycle participation increase in both
sweeps; spectral peaks shift modestly with local reversals. Rhythm contrast
decreases under leak modulation and increases overall under capacitance
modulation. These describe one frozen checkpoint, with no uncertainty estimates.
