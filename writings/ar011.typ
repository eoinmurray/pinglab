#let meta = (
  title: "The CLI",
  date: "2026-07-02",
  description: "Reference for src/cli/cli.py, the single command-line tool that drives every simulation, training run, and measurement in the project.",
  collection: "documentation",
)

#let body = [
  One command-line tool — _src/cli/cli.py_ — drives every simulation, training run, and measurement in this project. Notebooks never import the model code; they shell out to this tool, which writes data files (metrics, rasters, population traces, weight dumps) into an output directory, and then plot those files themselves. The CLI is the engine; the plotting lives in the notebooks.

  This page is the reference for that engine. It is not a tutorial — it is the dictionary you reach for when a notebook mentions _--ei-strength 0.5_ and you want to know exactly what that did.

  *At a glance*

  - Three subcommands: _sim_ (forward pass + metrics), _train_ (surrogate-gradient BPTT), _dump-weights_ (emit weight matrices).
  - Every parameter is a flag. There is no global config file; a saved run's _config.json_ can be inherited with _--load-config_.
  - Every run writes _config.json_, _run.sh_, and _output.log_ for reproducibility.
  - The CLI emits data, not figures.

  == Quick start

  Run a metrics-only forward pass of the canonical PING network:

  ```
  uv run python src/cli/cli.py sim --model ping --ei-strength 0.5
  ```

  Train on MNIST for 100 epochs:

  ```
  uv run python src/cli/cli.py train --dataset mnist --epochs 100 --lr 0.0001
  ```

  Evaluate a trained run on the test set, replaying it at a coarser timestep:

  ```
  uv run python src/cli/cli.py sim --infer \
    --load-config runs/foo/config.json \
    --load-weights runs/foo/weights.pth \
    --dt 0.5
  ```

  #quote(block: true)[
    Each subcommand has its own complete _--help_, and that per-subcommand help is authoritative. The top-level dispatcher's group summary is hand-maintained and can lag the parser — when a flag here disagrees with reality, run _sim --help_ and trust that.
  ]

  == Commands

  === sim

  Run one forward pass and report firing-rate metrics. This is the cheapest mode — no training, no plots — used by the test suite, the dt-stability checks in #link("/ar003/")[ar003], and every notebook that needs a single inference pass over a trained or fresh network.

  ```
  uv run python src/cli/cli.py sim --model ping --dataset mnist --digit 3
  ```

  On its own it prints metrics. The flags below make it load trained weights, evaluate a test set, emit extra data artifacts, or inject perturbations. There is no _--image_ or _--video_: where those retired flags once produced panels and sweep MP4s, _sim_ now emits raw data (via _--outputs_) that the calling notebook plots, and sweep videos are assembled notebook-side from many _sim_ calls.

  #table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    [*flag*], [*default*], [*description*],
    [_--infer_], [off], [Load trained weights and evaluate test-set accuracy; writes _metrics.json_ (and _results.json_).],
    [_--load-config PATH_], [—], [Load a saved _config.json_ and inherit its model, dataset, and parameters. Explicit CLI flags override loaded values.],
    [_--load-weights PATH_], [—], [Path to a _weights.pth_ file for inference.],
    [_--max-samples INT_], [all], [[_--infer_] Cap the evaluation set to N samples.],
    [_--outputs OUTPUT [dots.c]_], [—], [[_--infer_] Extra artifacts from the single forward pass (_metrics.json_ always written): _per_cell_rates_ (per-cell E/I Hz → _per_cell_rates.npz_), _pop_traces_ (per-trial population activity → _pop_traces.npz_, base signal for PSD / f_γ), _rasters_ (sparse spike indices → _rasters.npz_, for cycle-level analysis).],
    [_--tau-gaba FLOAT_], [inherited / 9.0], [[_--infer_] Override τ_GABA (ms) to replay a trained cell under specified inhibitory dynamics. Normally unset — _--load-config_ inherits the trained value.],
    [_--skip-load PREFIX [dots.c]_], [—], [[_--infer_] Drop _state_dict_ keys with these prefixes before loading (e.g. _W_ei. W_ie._) so a fresh sub-block survives — transfer-load probes (nb038).],
    [_--perturb-mode {drop, add, add_split}_], [—], [[_--infer_] Hidden-spike perturbation inside the forward loop: _drop_ (Bernoulli mask), _add_ (Poisson Hz), _add_split_ (separate E/I Poisson). The nb037 drop/add asymmetry.],
    [_--perturb-level LEVEL [dots.c]_], [—], [[_--perturb-mode_] One value for _drop_ (probability) or _add_ (Hz); two values (E Hz, I Hz) for _add_split_.],
    [_--i-override-file PATH_], [—], [[_--infer_] NPZ with a sparse per-trial I-spike stream to substitute for the inhibitory spikes each timestep — injection dual of _--outputs rasters_ (nb042).],
    [_--input-file PATH_], [—], [NPZ with _input_spikes_ (T, B, N_IN) to forward instead of Poisson input — arbitrary stimulus (nb048 digit streams).],
    [_--scale-w-in / --scale-w-ei / --scale-w-ie FLOAT_], [1.0], [[_--infer_] Multiply loaded input / E→I / I→E weights before the forward pass — inference-time coupling sweeps without retraining (nb038).],
    [_--sample-index INT_], [—], [Raw test-set index for a snapshot, overriding _--digit_ / _--sample_.],
    [_--n-in / --n-inh / --n-batch INT_], [784 / — / 64], [[synthetic-spikes] Input channels, inhibitory pool size, Poisson trials averaged.],
    [_--w-ei-mean / --w-ie-mean FLOAT_], [from _--ei-strength_], [[synthetic-spikes] Explicit W_EI / W_IE mean (std = 0.1·mean).],
    [_--private-w-in_], [off], [[synthetic-spikes] Identity W_in: one input channel per E cell.],
  )

  The block from _--skip-load_ down is the *generic-primitive family*: small, experiment-agnostic hooks (perturb hidden spikes, inject an inhibitory stream, forward an arbitrary input file, scale a weight block at inference). Notebooks compose these instead of importing model code — that is what keeps the notebook/CLI boundary clean.

  === train

  Surrogate-gradient BPTT training loop. Writes _weights.pth_, _metrics.json_, a per-step _metrics.jsonl_, and _test_predictions.json_.

  ```
  uv run python src/cli/cli.py train --model ping --dataset mnist \
    --epochs 100 --lr 0.0001 --v-grad-dampen 1000
  ```

  _--epochs 0_ runs the init snapshot only — useful as a probe.

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
    [_--fr-reg-mode {per-neuron, population}_], [_per-neuron_], [Pooling axis for the regulariser. _per-neuron_ concentrates pressure on the highest-firing cells (Cramer recipe); _population_ uses one scalar over the grand mean, distributing pressure uniformly.],
    [_--tau-gaba FLOAT_], [9.0 ms], [Override τ_GABA. Default = _models.py_'s value (Börgers / Buzsáki-Wang range). nb041 sweeps this across {4.5 … 27} ms while training PING from scratch; the realised gamma frequency f_γ tracks 1/τ_GABA.],
    [_--frame-rate INT_], [10], [fps for the per-epoch observation video, where one is emitted.],
  )

  === dump-weights

  Build the network from a config and emit its weight matrices to _weights_dump.npz_ — the init state, plus (with _--load-weights_) the trained state. It runs no forward pass.

  ```
  uv run python src/cli/cli.py dump-weights \
    --load-config runs/foo/config.json \
    --load-weights runs/foo/weights.pth \
    --out-dir runs/foo/dump
  ```

  Keys follow _W_ff_N_init_ / _W_ff_N_trained_ (feedforward, per layer N) plus the E-I blocks _W_ei_ / _W_ie_. This is how notebooks recover the trained readout matrix (W_out = the last _W_ff_) or compare init-vs-trained loop weights (the nb049 pruning analysis) without loading the model in-process. It takes the shared option groups plus _--load-config_ / _--load-weights_.

  == Shared options

  These groups are attached to every subcommand. The grouping matches what each subcommand's _--help_ prints.

  === Network

  #table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    [*flag*], [*default*], [*description*],
    [_--model {ping}_], [_ping_], [Architecture. _ping_ is the COBANet with E↔I coupling; with _--ei-strength 0_ the inhibitory loop is silenced for a no-rhythm control.],
    [_--n-hidden INT [INT dots.c]_], [dataset-dependent], [Hidden layer sizes. One integer = single layer; multiple stacks layers. Defaults: mnist 1024, smnist 32, shd 256.],
    [_--readout {rate, li, spike-count, mem-mean}_], [_rate_], [Output stage. _rate_ sums last-hidden spikes and projects linearly at the final timestep. _li_ is a non-spiking leaky integrator per class with max-over-time — the SHD-standard readout. _spike-count_ and _mem-mean_ are alternative pooling rules.],
    [_--dales-law_ / _--no-dales-law_], [on], [Enforce Dale's law (non-negative weights) or allow signed weights. _--no-dales-law_ is used for balanced-network experiments.],
    [_--ei-layers INT [INT dots.c]_], [all layers], [1-indexed list of hidden layers that get E-I structure. PING only.],
    [_--ei-strength FLOAT_], [0.5], [E-I coupling strength s. Sets W_EI = s and W_IE = s·ratio.],
    [_--ei-ratio FLOAT_], [2.0], [W_IE / W_EI.],
    [_--w-in-sparsity FLOAT_], [0.95], [Fraction of input weights zeroed at init.],
    [_--n-e INT_], [1024], [Excitatory pool size. With _--n-i_ and _--exact-k_, sweep N at fixed fan-in K to test Vreeswijk-Sompolinsky N-invariance.],
    [_--n-i INT_], [256], [Inhibitory pool size.],
    [_--ei-sparsity FLOAT_], [0.0], [Sparsity of the recurrent E↔I matrices: fraction of entries zeroed, survivors rescaled by 1/(1−s) to preserve expected drive. Use ≈ 1 − K/N for Brunel/Vreeswijk sparse random connectivity.],
    [_--exact-k_], [off], [Fixed-fan-in (exact-K) recurrent connectivity: every post cell draws exactly K = round((1−ei_sparsity)·N_pre) inputs. No effect unless _--ei-sparsity_ > 0.],
    [_--dt FLOAT_], [0.25], [Integration timestep (ms).],
    [_--t-ms FLOAT_], [200], [Total trial duration (ms). For synthetic-step inputs, must exceed _STEP_ON_MS_ (200 ms) or the stimulus window never opens.],
    [_--tau-mem FLOAT_], [10 ms], [Membrane time constant τ_mem. Cramer et al. SHD: 20 ms.],
    [_--tau-syn FLOAT_], [2 ms], [Synaptic (AMPA) time constant τ_syn. Cramer et al. SHD: 10 ms. Only affects COBA / PING.],
    [_--readout-tau-out FLOAT_], [5 ms], [Output-LIF τ_out for the _spike-count_ readout. Smaller values prevent saturation under high-rate drive at coarse dt. No-op for _rate_ / _li_.],
    [_--readout-w-out-scale FLOAT_], [1.0], [Scalar applied to the readout matrix after _build_net_, compensating for low hidden firing rate under _mem-mean_ / _spike-count_. Train-mode only.],
    [_--surrogate-slope FLOAT_], [1.0], [Fast-sigmoid surrogate-gradient slope β. Larger = narrower active window. Cramer et al. use 40 for SHD RSNNs.],
    [_--device {cpu, mps, cuda}_], [auto], [Compute device. Auto-detected: cuda > mps > cpu.],
  )

  The drive family below also lives in the Network group. It exists for the balanced-network (Brunel / van Vreeswijk-Sompolinsky) experiments and the Lyapunov chaos probe; canonical PING runs leave all of it off.

  #table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    [*flag*], [*default*], [*description*],
    [_--independent-drive RATE G_PER_SPIKE_], [off], [Per-E-cell independent Poisson drive (bypasses W_in): N_E uncorrelated streams at RATE Hz, each spike adding G_PER_SPIKE μS of g_E. Zero cross-cell correlation.],
    [_--independent-drive-i RATE G_PER_SPIKE_], [off], [As above, targeting the I population directly. Needed for the full V&S asynchronous-irregular state.],
    [_--shared-drive RATE G_PER_SPIKE_], [off], [One Poisson stream at RATE Hz broadcast to every E cell — fully correlated. Combine with _--independent-drive_ to tune cross-cell input correlation.],
    [_--shared-drive-i RATE G_PER_SPIKE_], [off], [As _--shared-drive_ but for I cells.],
    [_--noise-std FLOAT_], [0], [Additive white-noise into each E cell's excitatory conductance every timestep — vary input SNR independently of the mean.],
    [_--quenched-drive MEAN STD_], [off], [Per-E-cell DC conductance drawn once from N(MEAN, STD) μS and frozen for the trial — V&S quenched input. No fluctuation, so it cannot pin spike times; the Lyapunov probe then measures autonomous chaos.],
    [_--quenched-drive-i MEAN STD_], [off], [Per-I-cell frozen DC conductance.],
    [_--lyapunov-eps FLOAT_], [0 (off)], [If > 0 (synthetic-spikes mode), rerun with all membranes ε-perturbed at t=0 and save the divergence ‖ΔV(t)‖ to _snapshot.npz_. Its growth rate is the max Lyapunov exponent: positive for the chaotic V&S state, ≈ 0 for cycle-locked PING.],
  )

  === Input

  #table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    [*flag*], [*default*], [*description*],
    [_--input {synthetic-conductance, synthetic-spikes, dataset}_], [_synthetic-spikes_], [Stimulus regime. _synthetic-spikes_ is Poisson at _--input-rate_. _synthetic-conductance_ is a step current with _--stim-overdrive_. _dataset_ draws from _--dataset_.],
    [_--input-rate FLOAT_], [25], [Baseline input rate (Hz).],
    [_--stim-overdrive FLOAT_], [1.0], [Stimulus multiplier.],
    [_--digit INT_], [0], [Dataset class (0–9).],
    [_--sample INT_], [0], [Sample index within the class.],
    [_--sample-index INT_], [—], [Raw test-set index, overriding _--digit_ / _--sample_.],
    [_--dataset {mnist, smnist, shd}_], [_mnist_], [_mnist_ = full 28×28; _smnist_ = MNIST row-by-row over time; _shd_ = Spiking Heidelberg Digits audio (700 channels). SHD is cached under _\$PINGLAB_SHD_DIR_.],
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
    [_--trainable-w-ie_], [frozen], [Promote I→E. The nb049 result _gradient descent dismantles PING_ comes from flipping _--trainable-w-ei_ and _--trainable-w-ie_ on simultaneously.],
  )

  === Output

  #table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    [*flag*], [*default*], [*description*],
    [_--out-dir DIR_], [_src/artifacts/oscilloscope/_], [Output directory.],
    [_--wipe-dir_], [off], [Clear the output directory before the run.],
    [_--raster {scatter, imshow}_], [_scatter_], [Raster-plot style for the data emitted to raster artifacts.],
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

  The trick that makes the notebook chain work is _--load-config_ (successor to the old _--from-dir_). Every _train_ run writes a _config.json_ alongside its _weights.pth_; a later _sim_ or _dump-weights_ run inherits from it:

  ```
  uv run python src/cli/cli.py sim --infer \
    --load-config runs/foo/config.json \
    --load-weights runs/foo/weights.pth \
    --dt 0.5
  ```

  This inherits the model, hidden sizes, dataset, E-I parameters, input rate, τ_GABA, and seed — while the explicitly-passed _--dt 0.5_ overrides the trained value, replaying the network at a new timestep. Precedence is: explicit CLI flag > loaded config > default. The parser builds the set of CLI-explicit flags from _sys.argv_ before applying inheritance.

  Backwards compatibility: old configs that stored _n_hidden_ as a scalar are remapped to the _hidden_sizes_ list, legacy model names are aliased with a one-line stderr note, and configs missing _dales_law_ trigger a warning to pass it explicitly or retrain.

  == Artifacts

  Every subcommand calls _save_run_artifacts_ on entry, producing three files in _--out-dir_:

  #table(
    columns: (auto, 1fr),
    align: (left, left),
    [*file*], [*contents*],
    [_config.json_], [The parsed argparse namespace plus a provenance block (_git_sha_, _torch_version_, _device_, _python_env_hash_, _run_id_, _started_at_). Consumed by _--load-config_ and the notebook metadata extractors.],
    [_run.sh_], [The literal _sys.argv_ joined with spaces and prefixed with a shebang. Re-running reproduces the run (modulo seeded RNG state, also in _config.json_).],
    [_output.log_], [The run log. ANSI escapes are stripped from the file but preserved on stdout, so terminals see colour while the log stays grep-friendly.],
  )

  Each command adds its own outputs: _train_ writes _weights.pth_, _metrics.json_, _metrics.jsonl_, _test_predictions.json_; _sim --infer_ writes _metrics.json_ plus whatever _--outputs_ requested; _dump-weights_ writes _weights_dump.npz_.

  == Recipes

  *Train, then measure the trained network.* Train writes a run directory; _sim --infer_ reads it back and emits the population traces a notebook needs for a PSD:

  ```
  uv run python src/cli/cli.py train --dataset mnist --epochs 100 \
    --lr 0.0001 --v-grad-dampen 1000 --out-dir runs/ping

  uv run python src/cli/cli.py sim --infer \
    --load-config runs/ping/config.json \
    --load-weights runs/ping/weights.pth \
    --outputs pop_traces per_cell_rates
  ```

  *Loop-transfer at inference (nb038).* Take a network trained as COBA and scale its E→I coupling up at inference, with no retraining:

  ```
  uv run python src/cli/cli.py sim --infer \
    --load-config runs/coba/config.json \
    --load-weights runs/coba/weights.pth \
    --scale-w-ei 1.0 --scale-w-ie 1.0 --outputs rasters
  ```

  *Perturbation sweep (nb037).* Drop a fraction of emitted spikes, or add off-phase Poisson noise, inside the forward loop:

  ```
  uv run python src/cli/cli.py sim --infer \
    --load-config runs/ping/config.json --load-weights runs/ping/weights.pth \
    --perturb-mode drop --perturb-level 0.8
  ```

  *Recover the trained readout matrix.* Dump weights and read _W_ff_N_trained_ (the last layer is W_out):

  ```
  uv run python src/cli/cli.py dump-weights \
    --load-config runs/ping/config.json \
    --load-weights runs/ping/weights.pth --out-dir runs/ping/dump
  ```
]
