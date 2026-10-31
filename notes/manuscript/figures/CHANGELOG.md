
# exp110 manuscript changelog

## 2026-10-18 — Hardware and TR01 training timings

**Import scope for `../gamma-paper-1`:** update the Methods subsection previously
titled **Software** (P45 in exp110) to **Software and hardware**, and use the
following final paragraph. Preserve its existing repository links when adapting
to the destination manuscript's format.

> The experiment code is maintained in the [Pinglab experiments repository](https://github.com/eoinmurray/pinglab), and the spiking neural network (SNN) simulator is maintained in the [SNNLab simulator repository](https://github.com/eoinmurray/snnlab). Spiking simulation, training and evaluation used custom Python/PyTorch code; all 90 classifiers were trained with PyTorch 2.11.0+cu128 on NVIDIA A100-SXM4 GPUs with 80 GB of device memory, using one CUDA device per classifier. Mean-field calculations used Python 3.10.19, NumPy 2.2.6 and SciPy 1.15.3. For TR01, mean training-loop and validation times over 50 epochs were 402 and 158 min for loop-off and 429 and 161 min for loop-on, respectively, averaged across three seeds per architecture. Mean epoch times, including training and validation, were 11.2 min for loop-off and 11.8 min for loop-on over epochs 2–50, excluding first-epoch warm-up, reference-image diagnostics and subsequent logging or checkpoint-file writes.

This replaces the former final sentence, “Experiment records retain
stage-specific parameters, seeds, software/source revisions and original
provenance for reused classifiers,” with the TR01 timing summary, and adds the
GPU model, memory and one-device-per-classifier description.

**Do not import the intermediate timing table.** The accepted revision contains
no Table 2 or associated introduction/caption. Table 1 and P27 are unchanged.
Keep the preceding software-development and verification paragraph unchanged.

### Evidence and validation

- Canonical source: `report/report.mdx`, **Software and hardware**, P45.
- Timings derive from the six TR01 canonical classifiers in
  `exp022-r015-compute`, seeds 42–44 for each architecture. Training and
  validation totals sum their separate recorded timers over all 50 epochs;
  epoch times average epochs 2–50 within each network, then across the three
  networks per architecture. Values are rounded means, not mean ± SD.
- No training, inference or scientific analysis was rerun. This is a manuscript
  revision using retained execution evidence.
- Checked numerical values, unchanged Table 1, removal of Table 2 and its
  references, and the rendered paragraph. Astro build and read-only Pingstore
  discovery passed; Writing Guide 37.0.0 was retained.
- This entry records the change for a later import; `../gamma-paper-1` has not
  been modified.
