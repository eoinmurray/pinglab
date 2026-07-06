#let meta = (
  title: "The SNN tool",
  date: "2026-07-06",
  description: "Reference for tools/snn/tool.py, the single command-line tool that drives every simulation, training run, and measurement in the project.",
  collection: "documentation",
)

#let body = [
  One command-line tool, _tools/snn/tool.py_, drives every simulation, training run, and measurement in this project. Experiment runners never import the model code; they shell out to this tool, which writes data files (metrics, rasters, population traces, weight dumps) into an output directory, and then draw their own figures from those files. The tool is the engine; the plotting lives in the runners under _experiments/_.

  This page is the reference for that engine. It is not a tutorial. It is the dictionary you reach for when an experiment mentions _--ei-strength 0.5_ and you want to know exactly what that did.

  *At a glance*

  - Run it with _uv run python tools/snn/tool.py_ (never a bare _python_; the repo convention is _uv_).
  - Three subcommands: _sim_ (forward pass plus metrics), _train_ (surrogate-gradient BPTT), _dump-weights_ (emit weight matrices).
  - Every parameter is a flag. There is no global config file; a saved run's _config.json_ can be inherited with _--load-config_.
  - Every run writes _config.json_, _run.sh_, _output.log_, and _run.jsonl_ for reproducibility.
  - The tool emits data, not figures. The runner draws the figure.

  == The tool and the experiment

  demolab draws a hard line between a *tool* and an *experiment*. A tool holds the reusable science and speaks only through files; an experiment (a runner in _experiments/expNNN.py_) chooses which tool commands to run, then reads their data files back and renders the figures. The runner reaches the tool by running its CLI as a subprocess, never by importing it. That firewall is what keeps _tools/snn/_ generic and lets the same engine serve every writeup in the lab.

  So a typical experiment does three things: it invokes _tool.py_ (often many times, sweeping a parameter), it aggregates each run's config plus headline metrics into a single _numbers.json_, and it draws PNG or SVG figures from the run's data. The committed record lands in _artifacts/data/expNNN/_; the tool's own scratch output is disposable (see #link("#artifacts")[Artifacts]).

  == Quick start

  Run a metrics-only forward pass of the canonical PING network:

  ```
  uv run python tools/snn/tool.py sim --model ping --ei-strength 0.5
  ```

  Train on MNIST for 100 epochs:

  ```
  uv run python tools/snn/tool.py train --dataset mnist --epochs 100 --lr 0.0001
  ```

  Evaluate a trained run on the test set, replaying it at a coarser timestep:

  ```
  uv run python tools/snn/tool.py sim --infer \
    --load-config runs/foo/config.json \
    --load-weights runs/foo/weights.pth \
    --dt 0.5
  ```

  #quote(block: true)[
    Each subcommand has its own complete _--help_, and that per-subcommand help is authoritative. The top-level dispatcher's group summary is hand-maintained and can lag the parser. When a flag here disagrees with reality, run _sim --help_ and trust that.
  ]

  == Commands

  === sim

  Run one forward pass and report firing-rate metrics. This is the cheapest mode (no training, no plots), used by the test suite, the dt-stability checks in #link("/ar003/")[ar003], and every experiment that needs a single inference pass over a trained or fresh network.

  ```
  uv run python tools/snn/tool.py sim --model ping --dataset mnist --digit 3
  ```

  On its own it prints metrics. The flags below make it load trained weights, evaluate a test set, emit extra data artifacts, or inject perturbations. There is no _--image_ or _--video_: where those retired flags once produced panels and sweep MP4s, _sim_ now emits raw data (via _--outputs_) that the calling runner plots, and sweep videos are assembled runner-side from many _sim_ calls.

  #table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    [*flag*], [*default*], [*description*],
    [_--infer_], [off], [Load trained weights and evaluate test-set accuracy; writes _results.json_ (and _metrics.json_).],
    [_--load-config PATH_], [—], [Load a saved _config.json_ and inherit its model, dataset, and parameters. Explicit CLI flags override loaded values.],
    [_--load-weights PATH_], [—], [Path to a _weights.pth_ file for inference.],
    [_--max-samples INT_], [all], [[_--infer_] Cap the evaluation set to N samples.],
    [_--outputs OUTPUT [dots.c]_], [—], [[_--infer_] Extra artifacts from the single forward pass (_metrics.json_ always written): _per_cell_rates_ (per-cell E/I Hz to _per_cell_rates.npz_), _pop_traces_ (per-trial population activity to _pop_traces.npz_, base signal for PSD / f_γ), _rasters_ (sparse spike indices to _rasters.npz_, for cycle-level analysis).],
    [_--tau-gaba FLOAT_], [inherited / 9.0], [[_--infer_] Override τ_GABA (ms) to replay a trained cell under specified inhibitory dynamics. Normally unset: _--load-config_ inherits the trained value.],
    [_--skip-load PREFIX [dots.c]_], [—], [[_--infer_] Drop _state_dict_ keys with these prefixes before loading (e.g. _W_ei. W_ie._) so a fresh sub-block survives. Transfer-load probes (#link("/exp038/")[exp038]).],
    [_--perturb-mode {drop, add}_], [—], [[_--infer_] Hidden-spike perturbation inside the forward loop: _drop_ (Bernoulli mask), _add_ (Poisson Hz). The #link("/exp037/")[exp037] drop/add asymmetry.],
    [_--perturb-level LEVEL [dots.c]_], [—], [[_--perturb-mode_] One value: probability for _drop_, Hz for _add_.],
    [_--i-override-file PATH_], [—], [[_--infer_] NPZ with a sparse per-trial I-spike stream to substitute for the inhibitory spikes each timestep. Injection dual of _--outputs rasters_ (#link("/exp042/")[exp042]).],
    [_--input-file PATH_], [—], [NPZ with _input_spikes_ (T, B, N_IN) to forward instead of Poisson input. Arbitrary stimulus (#link("/exp048/")[exp048] digit streams).],
    [_--scale-w-in / --scale-w-ei / --scale-w-ie FLOAT_], [1.0], [[_--infer_] Multiply loaded input / E→I / I→E weights before the forward pass. Inference-time coupling sweeps without retraining (#link("/exp038/")[exp038]).],
    [_--sample-index INT_], [—], [Raw test-set index for a snapshot, overriding _--digit_ / _--sample_.],
    [_--n-in / --n-inh / --n-batch INT_], [784 / — / 64], [[synthetic-spikes] Input channels, inhibitory pool size, Poisson trials averaged.],
    [_--w-ei-mean / --w-ie-mean FLOAT_], [from _--ei-strength_], [[synthetic-spikes] Explicit W_EI / W_IE mean (std = 0.1·mean).],
    [_--private-w-in_], [off], [[synthetic-spikes] Identity W_in: one input channel per E cell.],
  )

  The block from _--skip-load_ down is the *generic-primitive family*: small, experiment-agnostic hooks (perturb hidden spikes, inject an inhibitory stream, forward an arbitrary input file, scale a weight block at inference). Runners compose these instead of importing model code, and that is what keeps the tool/experiment boundary clean.

  === train

  Surrogate-gradient BPTT training loop. Writes _weights.pth_, _metrics.json_, a per-step _metrics.jsonl_, and _test_predictions.json_.

  ```
  uv run python tools/snn/tool.py train --model ping --dataset mnist \
    --epochs 100 --lr 0.0001 --v-grad-dampen 1000
  ```

  _--epochs 0_ runs the init snapshot only, useful as a probe.

  #table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    [*flag*], [*default*], [*description*],
    [_--lr FLOAT_], [0.01], [Adam learning rate. Biophysical models (ping) typically need 0.0001; current-based models 0.01.],
    [_--epochs INT_], [0], [Number of epochs. 0 = init-snapshot probe only.],
    [_--batch-size INT_], [64], [DataLoader batch size.],
    [_--max-samples INT_], [all], [Cap dataset to N samples for smoke tests.],
    [_--v-grad-dampen FLOAT_], [80.0], [Dampening factor _d_ on the COBA membrane-voltage gradient. PING needs a much larger value (_d = 1000_ in the paper) to stabilise BPTT through the E↔I loop; COBA trains at the default. Theory in #link("/ar006/")[ar006].],
    [_--fr-reg-upper-theta FLOAT_], [0 (off)], [Upper-bound target spike count per neuron per trial (θ_u). Adds _s_u · Σ relu(⟨z_i⟩ − θ_u)²_ to the loss. Cramer et al. SHD RSNN: 100.],
    [_--fr-reg-upper-strength FLOAT_], [0], [Coefficient s_u on the upper regulariser. Cramer et al.: 0.06.],
    [_--tau-gaba FLOAT_], [9.0 ms], [Override τ_GABA. Default = _models.py_'s value (Börgers / Buzsáki-Wang range). #link("/exp041/")[exp041] sweeps this across {4.5 … 27} ms while training PING from scratch; the realised gamma frequency f_γ tracks 1/τ_GABA.],
  )

  === dump-weights

  Build the network from a config and emit its weight matrices to _weights_dump.npz_: the init state, plus (with _--load-weights_) the trained state. It runs no forward pass.

  ```
  uv run python tools/snn/tool.py dump-weights \
    --load-config runs/foo/config.json \
    --load-weights runs/foo/weights.pth \
    --out-dir runs/foo/dump
  ```

  Keys follow _W_ff_N_init_ / _W_ff_N_trained_ (feedforward, per layer N) plus the E-I blocks _W_ei_ / _W_ie_. This is how a runner recovers the trained readout matrix (W_out = the last _W_ff_) or compares init-vs-trained loop weights (the #link("/exp049/")[exp049] pruning analysis) without loading the model in-process. It takes the shared option groups plus _--load-config_ / _--load-weights_.

  == Shared options

  These groups are attached to every subcommand. The grouping matches what each subcommand's _--help_ prints.

  === Network

  #table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    [*flag*], [*default*], [*description*],
    [_--model {ping}_], [_ping_], [Architecture. _ping_ is the COBANet with E↔I coupling; with _--ei-strength 0_ the inhibitory loop is silenced for a no-rhythm control.],
    [_--n-hidden INT [INT dots.c]_], [dataset-dependent], [Hidden layer sizes. One integer = single layer; multiple stacks layers. Default for mnist: 1024.],
    [_--readout {rate, mem-mean}_], [_rate_], [Output stage. _rate_ sums last-hidden spikes and projects linearly at the final timestep. _mem-mean_ averages a per-class output-LIF membrane over time (the trained classification readout).],
    [_--dales-law_ / _--no-dales-law_], [on], [Enforce Dale's law (non-negative weights) or allow signed weights. _--no-dales-law_ is used for balanced-network experiments.],
    [_--ei-strength FLOAT_], [0.5], [E-I coupling strength s. Sets W_EI = s and W_IE = s·ratio.],
    [_--ei-ratio FLOAT_], [2.0], [W_IE / W_EI.],
    [_--w-in-sparsity FLOAT_], [0.95], [Fraction of input weights zeroed at init.],
    [_--ei-sparsity FLOAT_], [0.0], [Sparsity of the recurrent E↔I matrices: fraction of entries zeroed, survivors rescaled by 1/(1−s) to preserve expected drive. Use ≈ 1 − K/N for Brunel/Vreeswijk sparse random connectivity.],
    [_--exact-k_], [off], [Fixed-fan-in (exact-K) recurrent connectivity: every post cell draws exactly K = round((1−ei_sparsity)·N_pre) inputs. No effect unless _--ei-sparsity_ > 0.],
    [_--dt FLOAT_], [0.25], [Integration timestep (ms).],
    [_--t-ms FLOAT_], [200], [Total trial duration (ms). Metrics are measured over the full trace; runners strip any startup transient in post.],
    [_--readout-w-out-scale FLOAT_], [1.0], [Scalar applied to the readout matrix after _build_net_, compensating for low hidden firing rate under _mem-mean_. Train-mode only.],
    [_--surrogate-slope FLOAT_], [1.0], [Fast-sigmoid surrogate-gradient slope β. Larger = narrower active window. Cramer et al. use 40 for SHD RSNNs.],
  )

  The drive family below also lives in the Network group. It exists for the balanced-network (Brunel / van Vreeswijk-Sompolinsky) experiments and the Lyapunov chaos probe; canonical PING runs leave all of it off.

  #table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    [*flag*], [*default*], [*description*],
    [_--independent-drive RATE G_PER_SPIKE_], [off], [Per-E-cell independent Poisson drive (bypasses W_in): N_E uncorrelated streams at RATE Hz, each spike adding G_PER_SPIKE μS of g_E. Zero cross-cell correlation.],
    [_--independent-drive-i RATE G_PER_SPIKE_], [off], [As above, targeting the I population directly. Needed for the full V&S asynchronous-irregular state.],
    [_--quenched-drive MEAN STD_], [off], [Per-E-cell DC conductance drawn once from N(MEAN, STD) μS and frozen for the trial. V&S quenched input: no fluctuation, so it cannot pin spike times; the Lyapunov probe then measures autonomous chaos.],
    [_--quenched-drive-i MEAN STD_], [off], [Per-I-cell frozen DC conductance.],
    [_--lyapunov-eps FLOAT_], [0 (off)], [If > 0 (synthetic-spikes mode), rerun with all membranes ε-perturbed at t=0 and save the divergence ‖ΔV(t)‖ to _snapshot.npz_. Its growth rate is the max Lyapunov exponent: positive for the chaotic V&S state, ≈ 0 for cycle-locked PING.],
  )

  === Input

  #table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    [*flag*], [*default*], [*description*],
    [_--input {synthetic-spikes, dataset}_], [_synthetic-spikes_], [Stimulus regime. _synthetic-spikes_ is Poisson at _--input-rate_. _dataset_ draws from _--dataset_.],
    [_--input-rate FLOAT_], [25], [Baseline input rate (Hz).],
    [_--digit INT_], [0], [Dataset class (0–9).],
    [_--sample INT_], [0], [Sample index within the class.],
    [_--sample-index INT_], [—], [Raw test-set index, overriding _--digit_ / _--sample_.],
    [_--dataset {mnist}_], [_mnist_], [Dataset. _mnist_ is the full 28×28 image encoded to spikes.],
  )

  === Weights (advanced)

  #table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    [*flag*], [*default*], [*description*],
    [_--w-in MEAN [STD]_], [_0.3 0.06_], [Input fan-in init. Single value sets STD = MEAN × 0.1.],
    [_--w-ei MEAN STD_], [from _--ei-strength_], [Override the W_EI init.],
    [_--w-ie MEAN STD_], [from _--ei-strength_ / _--ei-ratio_], [Override the W_IE init.],
    [_--w-ii MEAN STD_], [_0 0_], [W_II (I→I) init. Off by default (canonical PING has no I→I). Enable for balanced-network experiments.],
    [_--w-ee MEAN STD_], [_0 0_], [W_EE (E→E) init. Off by default. Enable for the full four-coupling balanced network, where recurrent excitation pins the E rate.],
    [_--trainable-w-ei_], [frozen], [Promote E→I to gradient-carrying. Asks whether the optimiser will discover the PING-loop weights from scratch.],
    [_--trainable-w-ie_], [frozen], [Promote I→E. The #link("/exp049/")[exp049] result _gradient descent dismantles PING_ comes from flipping _--trainable-w-ei_ and _--trainable-w-ie_ on simultaneously.],
  )

  === Output

  #table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    [*flag*], [*default*], [*description*],
    [_--out-dir DIR_], [_temp/pinglab-cli/_], [Output directory. The default is scratch (gitignored); runners always pass an explicit path.],
    [_--wipe-dir_], [off], [Clear the output directory before the run.],
  )

  === Execution

  #table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    [*flag*], [*default*], [*description*],
    [_--seed INT_], [—], [RNG seed. Seeds Python, NumPy, torch (CPU + CUDA + MPS) before dataset load and model init. Persisted to _config.json_.],
    [_--modal_], [off], [Re-dispatch to Modal.com. Artifacts sync back to _--out-dir_ after completion.],
    [_--modal-gpu {none, T4, L4, A10G, A100, H100}_], [_T4_], [GPU type for Modal runs. _none_ runs CPU-only.],
  )

  #quote(block: true)[
    _--modal_ costs money. The project default is local; only pass it when explicitly instructed.
  ]

  == Config inheritance

  The trick that makes the experiment chain work is _--load-config_. Every _train_ run writes a _config.json_ alongside its _weights.pth_; a later _sim_ or _dump-weights_ run inherits from it:

  ```
  uv run python tools/snn/tool.py sim --infer \
    --load-config runs/foo/config.json \
    --load-weights runs/foo/weights.pth \
    --dt 0.5
  ```

  This inherits the model, hidden sizes, dataset, E-I parameters, input rate, τ_GABA, and seed, while the explicitly-passed _--dt 0.5_ overrides the trained value, replaying the network at a new timestep. Precedence is: explicit CLI flag, then loaded config, then default. The parser builds the set of CLI-explicit flags from _sys.argv_ before applying inheritance.

  Backwards compatibility: old configs that stored _n_hidden_ as a scalar are remapped to the _hidden_sizes_ list, legacy model names are aliased with a one-line stderr note, and configs missing _dales_law_ trigger a warning to pass it explicitly or retrain.

  == Artifacts <artifacts>

  Every subcommand calls _save_run_artifacts_ on entry, writing four provenance files into _--out-dir_:

  #table(
    columns: (auto, 1fr),
    align: (left, left),
    [*file*], [*contents*],
    [_config.json_], [The parsed argparse namespace plus a provenance block (_git_sha_ with a _dirty_ suffix, _torch_version_, _device_, _python_env_hash_, _run_id_, _started_at_) and the _mode_. Consumed by _--load-config_ and the runner's metadata extractors.],
    [_run.sh_], [The literal _sys.argv_ joined with spaces and prefixed with a shebang. Re-running reproduces the run (modulo seeded RNG state, also in _config.json_).],
    [_output.log_], [The human run log. ANSI escapes are stripped from the file but preserved on stdout, so terminals see colour while the log stays grep-friendly.],
    [_run.jsonl_], [The machine-readable event spine: one typed JSON object per event (epoch rows, warnings, summary). This is what a runner parses when it wants structured progress, not scraped log text.],
  )

  Each command adds its own outputs: _train_ writes _weights.pth_, _metrics.json_, _metrics.jsonl_, _test_predictions.json_; _sim --infer_ writes _metrics.json_ (and _results.json_) plus whatever _--outputs_ requested; _dump-weights_ writes _weights_dump.npz_.

  These files are scratch. By default they land under _temp/pinglab-cli/_, which is gitignored and overwritten every run. The committed record is produced by the runner: it aggregates each command's config plus headline metrics into one _artifacts/data/expNNN/numbers.json_ and renders its figures alongside. That folder, not the ephemeral tool output, is what the publisher reads and what reaches the site.

  == Recipes

  *Train, then measure the trained network.* Train writes a run directory; _sim --infer_ reads it back and emits the population traces a runner needs for a PSD:

  ```
  uv run python tools/snn/tool.py train --dataset mnist --epochs 100 \
    --lr 0.0001 --v-grad-dampen 1000 --out-dir runs/ping

  uv run python tools/snn/tool.py sim --infer \
    --load-config runs/ping/config.json \
    --load-weights runs/ping/weights.pth \
    --outputs pop_traces per_cell_rates
  ```

  *Loop-transfer at inference (#link("/exp038/")[exp038]).* Take a network trained as COBA and scale its E→I coupling up at inference, with no retraining:

  ```
  uv run python tools/snn/tool.py sim --infer \
    --load-config runs/coba/config.json \
    --load-weights runs/coba/weights.pth \
    --scale-w-ei 1.0 --scale-w-ie 1.0 --outputs rasters
  ```

  *Perturbation sweep (#link("/exp037/")[exp037]).* Drop a fraction of emitted spikes, or add off-phase Poisson noise, inside the forward loop:

  ```
  uv run python tools/snn/tool.py sim --infer \
    --load-config runs/ping/config.json --load-weights runs/ping/weights.pth \
    --perturb-mode drop --perturb-level 0.8
  ```

  *Recover the trained readout matrix.* Dump weights and read _W_ff_N_trained_ (the last layer is W_out):

  ```
  uv run python tools/snn/tool.py dump-weights \
    --load-config runs/ping/config.json \
    --load-weights runs/ping/weights.pth --out-dir runs/ping/dump
  ```
]
