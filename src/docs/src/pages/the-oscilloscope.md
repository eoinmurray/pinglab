---
layout: ../layouts/MarkdownLayout.astro
title: "The Oscilloscope"
---

# The Oscilloscope

*src/pinglab/oscilloscope.py* is the single CLI that drives every model in the repo. The name is literal: most subcommands end by rendering a multi-panel figure — spike rasters, weight histograms, population rate, PSD, optionally training curves — laid out like an instrument face. One command runs a simulation, probes the network with a parameter sweep, trains it, or evaluates a trained checkpoint; every run writes its own self-contained directory with the config and weights needed to reproduce the result.

This page is the full flag reference. Conceptual context for individual knobs lives in the linked background pages ([Models](/models/), [Training](/training/), [Metrics](/metrics/)); here every flag is listed with its default, units, and scope.

## Subcommands

The CLI has five subcommands. Shared flags live on a *parent* parser so every mode understands the full network/input/weight/output/execution vocabulary; each subcommand then adds its own mode-specific flags.

| Mode | What it does | Typical output |
| ---- | ------------ | -------------- |
| *sim* | One forward pass, print firing-rate [metrics](/metrics/), no plot. | *metrics.json*, *metrics.jsonl* |
| *image* | One forward pass, render a still oscilloscope figure. | *oscilloscope.png* |
| *video* | Sweep one parameter linearly over *--frames*, render one frame per value. | *oscilloscope.mp4* |
| *train* | Surrogate-gradient BPTT training loop. | *weights.pth*, per-epoch metrics, optional training video |
| *infer* | Evaluate a trained checkpoint, optionally sweep *dt*. | *metrics.json*, *test_predictions.json*, optional dt-sweep video |

*image* and *video* are the diagnostic modes — they run a network forward once (or *frames* times) and show you what it does. *train* and *infer* are the learning and evaluation modes. *sim* is the smoke test: no plotting, just numbers.

## The run directory

Every run writes a directory containing *config.json* (full argv + derived params + git SHA), *run.sh* (the command that produced it), *metrics.json*, *metrics.jsonl*, and whatever figures/videos/weights that mode emits. Three flags govern where and how that directory is used.

| Flag | Default | What it does |
| ---- | ------- | ------------ |
| *--out-dir DIR* | *src/artifacts/&lt;mode&gt;/&lt;model&gt;-&lt;dataset&gt;* | Where to write artifacts. |
| *--wipe-dir* | off | Clear the target directory first. Notebook runners override to on — see the *feedback_notebook_wipe_dir* convention. |
| *--from-dir DIR* | — | (*image* / *video* / *infer*) Inherit every param from an earlier *train* run's *config.json*, then let explicit CLI flags override. This is how you reload a trained network for probing. *infer* auto-picks *weights.pth* from the same dir. |

The *--from-dir* inheritance is what makes trained-network probing a one-liner. A training run directory is a complete specification of the network and data pipeline; image/video/infer can re-enter it without the caller having to restate the matching flags.

## Network flags (shared)

These are on the shared parent parser and apply to every subcommand.

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| *--model* | *ping* | Model family: [*ping*](/models/#ping), [*cuba*](/models/#cuba), [*standard-snn*](/models/#standard-snn), [*snntorch-library*](/models/#snntorch-library). |
| *--n-hidden N ...* | dataset-aware (scikit 64, mnist 1024, smnist 32) | Hidden layer sizes. Single value = one layer, multiple = stacked (e.g. *--n-hidden 128 256*). $N_I = N_E / 4$ per layer. |
| *--n-input N* | *N_E* | Number of input neurons. |
| *--ei-strength S* | 0.5 | E-I coupling: sets $W_{EI} = s$, $W_{IE} = s \cdot \text{ratio}$. |
| *--ei-ratio R* | 2.0 | $W_{IE} / W_{EI}$ ratio. |
| *--ei-sparsity F* | 0.2 | E-I connection sparsity. |
| *--w-in-sparsity F* | 0.95 | Input-weight sparsity. |
| *--bias B* | 0.0002 | Background conductance to E neurons, μS. |
| *--dt DT* | 0.25 | Integration timestep, ms. |
| *--t-ms T* | 200.0 | Total simulation duration, ms. For synthetic-step modes, must exceed *STEP_ON_MS* (default 200) so the stimulus window is reached. |
| *--kaiming-init* | off | Use plain Kaiming-uniform init (signed weights, no fan-in normalisation), matching the canonical snnTorch tutorial. Applies to *standard-snn* / *cuba* only; *--w-in* is ignored when set. |
| *--init-scale-weight X* | 1.0 | Multiply initial weight matrices by X after *build_net* (train mode). For *cuba* at training-dt, pass $\Delta t / (1 - e^{-\Delta t / \tau_\text{mem}})$ to match *standard-snn*'s per-step spike drive. |
| *--init-scale-bias X* | 1.0 | Multiply initial bias vectors by X. For *cuba*, pass $1 / (1 - e^{-\Delta t / \tau_\text{mem}})$ for matched bias drive. |
| *--readout* | *rate* | Output reduction: *rate* (sum spike counts, project at final step) or *li* (non-spiking leaky integrator, max-over-time). See [Readouts](#readouts). |
| *--dales-law* / *--no-dales-law* | on | Clamp weights to non-negative (default) or allow signed weights (*standard-snn* / *cuba* only). |
| *--rec-layers L ...* | all when *--w-rec* set | Which hidden layers (1-indexed) get recurrence. |
| *--ei-layers L ...* | all | Which hidden layers (1-indexed) get E-I structure (PING only). |
| *--surrogate-slope β* | 1.0 | Fast-sigmoid surrogate-gradient slope. Larger = narrower active window. Cramer et al. SHD RSNNs use 40. Applies to *SurrogateSpike* and snnTorch's *fast_sigmoid*. |
| *--device* | auto (cuda > mps > cpu) | Compute device: *cpu* / *mps* / *cuda*. |

## Input flags (shared)

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| *--input* | *synthetic-spikes* | Input mode: *synthetic-conductance*, *synthetic-spikes*, *dataset*. Auto-flips to *dataset* if *--digit* / *--sample* / *--dataset* is passed. |
| *--input-rate HZ* | 25.0 | Baseline Poisson rate for *synthetic-spikes*. |
| *--stim-overdrive X* | 1.0 | Multiplier on the stimulus-window input rate. |
| *--drive D* | — | Baseline tonic conductance for *synthetic-conductance*. |
| *--dataset* | *scikit* | *scikit* / *mnist* / *smnist* / *shd*. |
| *--digit D* | 0 | Digit class (0–9) for dataset input. |
| *--sample I* | 0 | Sample index within the dataset class. |

Three input modes select how the network is driven (see [Training → Input modes](/training/#tasks-and-inputs) for the dynamics): *synthetic-conductance* is Börgers-style step drive injected directly into layer-1 E neurons — used for baseline oscillation studies where encoding should not confound the drive. *synthetic-spikes* is Poisson spike trains at *--input-rate* with a stimulus window where the rate is multiplied by *--stim-overdrive*. *dataset* is real images (scikit-digits, mnist, smnist) or audio (shd) Poisson-encoded per-pixel / per-channel.

## Weight flags (shared, advanced)

Overrides for the init distributions. Leave unset unless you know why you need them.

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| *--w-in MEAN STD* | 0.3 0.06 | Input-weight init. *standard-snn* dense needs roughly 10 2. |
| *--w-ei MEAN STD* | from *--ei-strength* | E→I init (overrides *--ei-strength*). |
| *--w-ie MEAN STD* | from *--ei-strength* | I→E init. |
| *--w-rec MEAN STD* | 0 0.1 | Recurrent-weight init. |
| *--w-in-overdrive X* | 1.0 | Multiplier on input weights applied on top of the init. |

## Output flags (shared)

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| *--out-dir DIR* | auto | Output directory (see [The run directory](#the-run-directory)). |
| *--wipe-dir* | off | Clear target directory before the run. |
| *--raster* | *scatter* | Raster style: *scatter* or *imshow*. |
| *--layout* | *full* | Panel layout preset (see [Layouts and panels](#layouts-and-panels)). |
| *--panels NAMES* | — | Comma-separated panel list — overrides the preset. |

## Execution flags (shared)

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| *--seed N* | unseeded | RNG seed. Seeds Python, NumPy, and torch (CPU + CUDA + MPS) before dataset load and model init. Persisted to *config.json*. Unseeded runs record *seed: null*. |
| *--modal* | off | Run on Modal.com instead of locally; artifacts sync back to *--out-dir*. See [Modal dispatch](#modal-dispatch). |
| *--modal-gpu* | *T4* | GPU type for Modal runs: *none* (CPU), *T4*, *L4*, *A10G*, *A100*, *H100*. |
| *--coba-integrator* | *expeuler* | Membrane ODE integrator for COBA/PING: *expeuler* (dt-invariant) or *fwd* (forward Euler, for parity comparisons). |

## Subcommand: *sim*

No additional flags. Runs one forward pass and prints firing-rate metrics to *metrics.json* / *metrics.jsonl* without rendering a plot.

## Subcommand: *image*

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| *--from-dir DIR* | — | Inherit params + weights from a training run. |
| *--load-weights PATH* | — | Load a *weights.pth* directly (alternative to *--from-dir*). |
| *--fake-progress X* | off | Overlay a progress-bar indicator at level 0–1 (demo / teaching). |

## Subcommand: *video*

Sweeps one parameter linearly between *--scan-min* and *--scan-max* over *--frames*. Each frame is a fresh simulation at one scan value; the rendered MP4 plays the parameter axis as time.

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| *--scan-var* | *stim-overdrive* | Parameter to sweep (see table below). |
| *--scan-min* | 1.0 | Sweep start, in the variable's units. For *digit*: integer class. |
| *--scan-max* | 50.0 | Sweep end. For *digit*: integer class. |
| *--frames N* | 10 | Number of frames. Overridden by scan range for *digit*. |
| *--frame-rate FPS* | 10 | Output video frame rate. |
| *--resample-input* | off | Fresh Poisson seed per frame (default: same seed for all frames). |
| *--from-dir DIR* | — | Inherit from a training run. |
| *--load-weights PATH* | — | Load weights directly. |

Scan variables:

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

By default the scan reuses the same Poisson seed on every frame so the input pattern is identical and only the scan variable varies. Pass *--resample-input* to draw a fresh seed per frame (useful when the question is about ensemble behaviour rather than one realisation).

## Subcommand: *train*

Surrogate-gradient BPTT. See [Training](/training/) for loss, encoding, and BPTT details.

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| *--epochs N* | 0 | Training epochs. 0 = probe only (init snapshot, no training). |
| *--lr RATE* | 0.01 | Optimizer learning rate. Biophysical models typically want $10^{-4}$. |
| *--optimizer* | *adam* | *adam* or *adamax*. Adamax uses the $L_\infty$ norm for the second moment instead of an EMA of squared grads, so a single pathological batch cannot poison the preconditioner. Canonical SNN choice (Cramer et al. 2022, Zenke Spytorch). |
| *--batch-size B* | 64 | DataLoader mini-batch size. |
| *--burn-in MS* | 20.0 | Burn-in period in ms. |
| *--max-samples N* | — | Limit dataset to N samples (smoke-test). |
| *--adaptive-lr* | off | Enable *ReduceLROnPlateau* (factor 0.5, patience 5). |
| *--early-stopping N* | — | Stop after N epochs without improvement. |
| *--grad-clip X* | 1.0 | Global gradient-norm clip passed to *clip_grad_norm_*. Overrides *models.GRAD_CLIP*. |
| *--skip-bad-grad-threshold X* | — | Skip the optimizer step (and zero grads) whenever the clipped gradient norm is NaN, inf, or exceeds X. Band-aid against single exploded batches poisoning Adam's second-moment estimate; prefer *--optimizer adamax* as the principled fix. |
| *--v-grad-dampen S* | 80.0 | Gradient dampening for the COBA membrane. |
| *--observe* | — | Save oscilloscope per epoch: *video* (MP4) or *images* (one PNG per epoch). |
| *--observe-every N* | 1 | Observe every Nth epoch. |
| *--frame-rate FPS* | 10 | Video fps for *--observe video*. |
| *--fr-reg-lower-theta* $\theta_l$ | 0.0 | Firing-rate regulariser lower target (spikes/neuron/trial). Penalty $s_l \sum_i \text{relu}(\theta_l - \langle z_i \rangle)^2$. 0 disables. Cramer et al. SHD RSNN: 0.01. |
| *--fr-reg-lower-strength* $s_l$ | 0.0 | Strength on the lower-bound FR regulariser. Cramer et al.: 1.0. |
| *--fr-reg-upper-theta* $\theta_u$ | 0.0 | Upper target. Penalty $s_u \sum_i \text{relu}(\langle z_i \rangle - \theta_u)^2$. Cramer et al.: 100. |
| *--fr-reg-upper-strength* $s_u$ | 0.0 | Strength on the upper-bound FR regulariser. Cramer et al.: 0.06. |

## Subcommand: *infer*

Evaluate a trained checkpoint. Requires *--from-dir* or *--load-weights*.

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| *--from-dir DIR* | — | Inherit params + weights from a training run. |
| *--load-weights PATH* | — | Load *weights.pth* directly (auto-detected from *--from-dir*). |
| *--max-samples N* | — | Limit dataset to N samples. |
| *--dt-sweep DT ...* | — | Run inference at each dt and produce a sweep summary (e.g. *0.05 0.1 0.25 0.5 1.0*). Overrides *--dt*. |
| *--observe* | — | Save oscilloscope: with *--dt-sweep*, *video* renders one frame per dt; without, *image* is a single snapshot. |
| *--frozen-inputs* | off | Freeze input spike patterns across the dt sweep. Shorthand for *--frozen-inputs-mode upsample*. |
| *--frozen-inputs-mode* | — | How input spikes are transported across dt, anchored at train-dt (Parthasarathy et al. §2.1, §2.3): *upsample* (count-preserving zero-pad to finer eval-dt, requires eval-dt ≤ train-dt), *downsample* (count-preserving sum-pool to coarser eval-dt, requires eval-dt ≥ train-dt), *resample* (fresh Poisson at each eval-dt). |

See [Training → dt stability sweep](/training/#dt-stability-sweep) for the protocol.

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

## Readouts

*--readout* picks how the last-hidden layer is reduced to class logits. The choice is orthogonal to the model family — any of [*cuba*](/models/#cuba), [*standard-snn*](/models/#standard-snn), [*ping*](/models/#ping), [*snntorch-library*](/models/#snntorch-library) can be paired with either readout.

| *--readout* | How it reads out | When to use |
| ----------- | ---------------- | ----------- |
| *rate* (default) | Accumulate last-hidden spike counts across all timesteps, one linear projection $W_\text{out}\cdot\sum_t s_t + b_\text{out}$ at the final step. | Default ladder decoder, matched across models. No temporal structure in the decoder — the hidden dynamics carry all the temporal work. |
| *li* | Non-spiking leaky integrator per class: $v_\text{out} \leftarrow \beta\,v_\text{out} + (1-\beta)(W_\text{out}\cdot s_t + b_\text{out})$, logits are $\max_t v_\text{out}$. | Field-standard SHD readout (Zenke-style). Makes the decoder itself temporal — useful when the classification signal is localised in time rather than cumulative. |

Both use the same $W_\text{out}, b_\text{out}$ shape so the learnable parameter count is identical; only the reduction over time differs. The *li* path currently reuses $\tau_\text{mem}$ as its time constant — see [notebook 004](/notebooks/nb004/) for how this interacts with init scaling.

## Reproducibility

*--seed* threads through Python, NumPy, and torch RNGs before dataset construction and model init, and is persisted in the run's *config.json* alongside the git SHA. A fixed seed plus a fixed git SHA is the reproducibility contract — same *config.json* plus same SHA should regenerate the same *metrics.json*. When *--seed* is omitted the run draws fresh RNG state and the config records *seed: null*, so unseeded runs are visible as such.

## Modal dispatch

Any oscilloscope subcommand can run on [Modal.com](https://modal.com) serverless compute by adding *--modal --modal-gpu {none, T4, L4, A10G, A100, H100}* — *none* runs on Modal CPU, the others pick the listed GPU. Local output paths resolve the same way on either side; the wrapper syncs the Modal volume back to *src/artifacts/* when the job finishes.

Every notebook runner under *src/pinglab/notebooks/* forwards a top-level *--modal-gpu* argument through to the oscilloscope invocations it makes, so an entire notebook entry (e.g. *uv run src/pinglab/notebooks/nb004.py --modal-gpu T4*) dispatches to Modal with one flag. Omit the flag and the run stays local.

## Where things live in the source

- *oscilloscope.py* — argument parsing, subcommand dispatch, scan drivers, training and inference loops, run-artifact bookkeeping.
- *plot.py* — the panel catalog, layout presets, and the per-panel draw functions.
- *models.py* — the LIF update rules and network classes.
- *inputs.py* — synthetic drive generators and Poisson encoders.
- *config.py* — shared defaults and the *Config* / *patch_dt* plumbing that keeps *dt*-invariant quantities invariant when *--dt* changes.

Notebook runners under *src/pinglab/notebooks/* invoke *oscilloscope.py* internally — they are the promotion gate from raw artifacts into *src/docs/public/figures/notebooks/&lt;slug&gt;/*. See [Introduction § Notebook](/introduction/#notebook) for the entry/runner pairing.
