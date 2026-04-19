---
layout: ../layouts/MarkdownLayout.astro
title: "The Oscilloscope"
---

# The Oscilloscope

*src/pinglab/oscilloscope.py* is the single CLI that drives every model in the repo. The name is literal: most subcommands end by rendering a multi-panel figure — spike rasters, weight histograms, population rate, PSD, optionally training curves — laid out like an instrument face. One command runs a simulation, probes the network with a parameter sweep, trains it, or evaluates a trained checkpoint; every run writes its own self-contained directory with the config and weights needed to reproduce the result.

## Subcommands

The CLI has five subcommands; flags are shared via a *parent* parser so every mode understands the full network/input/weight/output/execution vocabulary.

| Mode | What it does | Typical output |
| ---- | ------------ | -------------- |
| *sim* | One forward pass, print firing-rate [metrics](/metrics/), no plot. | *metrics.json*, *metrics.jsonl* |
| *image* | One forward pass, render a still oscilloscope figure. | *oscilloscope.png* |
| *video* | Sweep one parameter linearly over *--frames*, render one frame per value. | *oscilloscope.mp4* |
| *train* | Surrogate-gradient BPTT training loop. | *weights.pth*, per-epoch metrics, optional training video |
| *infer* | Evaluate a trained checkpoint, optionally sweep *dt*. | *metrics.json*, *test_predictions.json*, optional dt-sweep video |

*image* and *video* are the diagnostic modes — they run a network forward once (or *frames* times) and show you what it does. *train* and *infer* are the learning and evaluation modes. *sim* is the smoke test: no plotting, just numbers.

## The run directory

Every run writes a directory containing *config.json* (full argv + derived params + git SHA), *run.sh* (the command that produced it), *metrics.json*, *metrics.jsonl*, and whatever figures/videos/weights that mode emits. Two flags govern where and how that directory is used.

- *--out-dir* picks where to write. Defaults to *src/artifacts/&lt;mode&gt;/&lt;model&gt;-&lt;dataset&gt;*.
- *--wipe-dir* clears the target first. Off by default; notebook runners override this — see the *feedback_notebook_wipe_dir* convention.
- *--from-dir* (image / video / infer) inherits every param from an earlier *train* run's *config.json*, then lets explicit CLI flags override. This is how you reload a trained network for probing.

The *--from-dir* inheritance is what makes trained-network probing a one-liner. A training run directory is a complete specification of the network and data pipeline; image/video/infer can re-enter it without the caller having to restate the matching flags.

## Input modes

Three input modes select how the network is driven. See [Training → Input modes](/training/#input-modes-and-tasks) for the dynamics.

- *synthetic-conductance* — Börgers-style step drive injected directly into layer-1 E neurons. For baseline oscillation studies where encoding should not confound the drive.
- *synthetic-spikes* — Poisson spike trains at *--input-rate* with a stimulus window where the rate is multiplied by *--stim-overdrive*.
- *dataset* — real images (scikit-digits, mnist, smnist) Poisson-encoded per-pixel. The CLI auto-flips to *dataset* mode when *--digit*, *--sample*, or *--dataset* is set explicitly (so *--input* usually does not need to be passed by hand).

## Scans

The *video* subcommand sweeps one variable linearly between *--scan-min* and *--scan-max* over *--frames*. Each frame is a fresh simulation at one value of the scan variable; the rendered MP4 plays the parameter axis as time.

| *--scan-var* | Unit | What it controls |
| ------------ | ---- | ---------------- |
| *stim-overdrive* | × | multiplier on the stimulus-window input rate |
| *tau_gaba* | ms | inhibitory synapse decay time constant |
| *tau_ampa* | ms | excitatory synapse decay time constant |
| *w_ei_mean*, *w_ie_mean* | μS | recurrent E→I and I→E weight means |
| *w_in_overdrive* | × | multiplier on the input-layer weight |
| *ei_strength* | — | scales $W^{EI}$ and $W^{IE}$ jointly — the internal PING loop gain |
| *spike_rate* | Hz | baseline input spike rate |
| *bias* | μS | tonic bias current |
| *dt* | ms | simulation timestep (diagnostic — do not train under a dt scan) |
| *digit* | — | iterates dataset class 0–9 — for trained-network digit tours |
| *noise* | Hz | adds Poisson noise to the input layer |

By default a scan reuses the same Poisson seed on every frame so the input pattern is identical and only the scan variable varies. Pass *--resample-input* to draw a fresh seed per frame (useful when the question is about ensemble behaviour rather than one realisation).

## Layouts and panels

The oscilloscope figure is a grid of panels assembled from a catalog — [*header*](/metrics/#header), [*e_raster*](/metrics/#e_raster), [*i_raster*](/metrics/#i_raster), [*weights*](/metrics/#weights), [*drive*](/metrics/#drive), [*output*](/metrics/#output), [*psd*](/metrics/#psd), [*participation*](/metrics/#participation), plus training-only [*acc_curve*](/metrics/#acc_curve), [*grad_flow*](/metrics/#grad_flow), [*rate_curve*](/metrics/#rate_curve), [*digit_image*](/metrics/#digit_image), and sweep-only [*sweep*](/metrics/#sweep), [*sweep_rates*](/metrics/#sweep_rates), [*sweep_f0*](/metrics/#sweep_f0). *--layout* picks a named preset. See [Metrics](/metrics/) for what each panel shows and how to read it.

| *--layout* | Panels | Used for |
| ---------- | ------ | -------- |
| *full* | header, rasters, drive, weights, output, psd, participation | default for *image* and *sim* |
| *video* | *full* + *sweep* | default for *video* |
| *dataset* | *full* + *digit_image* | single-image dataset probe |
| *dataset_video* | *video* + *digit_image* | digit/dataset scan |
| *sweep_video* | rasters + PSD + per-frame *sweep_rates* / *sweep_f0* panels | long parameter sweeps with trend panels |
| *train* | *full* + *acc_curve*, *grad_flow*, *rate_curve* | one frame per epoch during *train --observe video* |

Pass *--panels* with a comma-separated list to override any preset — useful when you want a minimal raster-only snapshot.

## Training and inference

See [Training](/training/) for the surrogate-gradient, loss, encoding, and BPTT details. On the CLI side, *train* adds the optimiser flags (*--lr*, *--epochs*, *--batch-size*, *--adaptive-lr*, *--cm-back-scale*, *--early-stopping*) plus *--observe video* which renders one oscilloscope frame per epoch into a training MP4. *infer* adds *--dt-sweep* for the dt-stability protocol and *--frozen-inputs* to OR-pool a reference spike train (see [Training → dt stability sweep](/training/#dt-stability-sweep)).

## Reproducibility

*--seed* threads through Python, NumPy, and torch RNGs before dataset construction and model init, and is persisted in the run's *config.json* alongside the git SHA. A fixed seed plus a fixed git SHA is the reproducibility contract — same *config.json* plus same SHA should regenerate the same *metrics.json*. When *--seed* is omitted the run draws fresh RNG state and the config records *seed: null*, so unseeded runs are visible as such.

## Where things live in the source

- *oscilloscope.py* — argument parsing, subcommand dispatch, scan drivers, training and inference loops, run-artifact bookkeeping.
- *plot.py* — the panel catalog, layout presets, and the per-panel draw functions.
- *models.py* — the LIF update rules and network classes.
- *inputs.py* — synthetic drive generators and Poisson encoders.
- *config.py* — shared defaults and the *Config* / *patch_dt* plumbing that keeps *dt*-invariant quantities invariant when *--dt* changes.

Notebook runners under *src/pinglab/notebook/* invoke *oscilloscope.py* internally — they are the promotion gate from raw artifacts into *src/docs/public/figures/notebook/&lt;slug&gt;/*. See [Introduction § Notebook](/introduction/#notebook) for the entry/runner pairing.
