#import "contents.typ": with-contents, with-numbered-equations
#import "run-view.typ": with-datasets
#let meta = (
  status: "[≡ TXT | v33.0.0]",
  title: "SNNSIM command-line guide",
  updated_at: "2026-08-28T00:00:00Z",
  created_at: "2026-07-06T00:00:00Z",
  description: "Run a small simulation, choose the legacy or graph executor, and interpret the CLI options and output files.",
  collection: "snnsim-docs",
  order: 1,
)

#let body = [
  == Start here

  `tools/snnsim/tool.py` is the command-line entry point for simulation, training, and weight inspection. It emits data; analysis and presentation stages turn those data into measurements and figures. This guide focuses on the legacy CLI and points to the graph API where its contract differs.

  Read the SNNSIM documentation in this order:

  + *This guide:* commands, outputs, and execution boundaries.
  + #link("/exp004/")[Parameters & Units]: defaults, overrides, and dimensional checks.
  + #link("/exp100/")[COBANet]: the implemented neuron and synapse updates.
  + #link("/exp006/")[Training]: readouts, checkpoints, initialization, and regularisation.
  + #link("/exp015/")[Gradient Stabilisation]: backward-pass controls and diagnostics.

  === Choose an executor

  #table(
    columns: (auto, 1fr),
    [*Interface*], [*Use it for*],
    [`--executor legacy`], [The built-in COBANet model and existing CLI recipes. This remains the default.],
    [`--executor graph --bundle PATH`], [An authored SNNLANG bundle with named inputs, outputs, and graph execution contracts. See #link("/exp107/")[Compiling and executing bundles].],
    [`tools.snnsim.execution`], [Python callers using `ExecutionSpec`, `build`, `simulate`, `train`, or `infer`. See #link("/exp102/")[SNNLANG developer documentation].],
  )

  A bundle argument alone does not select the graph executor. Do not assume legacy flags, weight files, or artifact names apply unchanged to graph execution.

  == Quick start

  From the repository root, first inspect the available commands:

  ```sh
  uv run python tools/snnsim/tool.py sim --help
  uv run python tools/snnsim/tool.py train --help
  uv run python tools/snnsim/tool.py dump-weights --help
  ```

  Run a small synthetic-input forward pass into a fresh scratch directory:

  ```sh
  uv run python tools/snnsim/tool.py sim --executor legacy \
    --model ping --input synthetic-spikes --n-hidden 32 \
    --n-in 32 --n-batch 1 --t-ms 20 --dt 0.25 --seed 42 \
    --out-dir temp/docs-simulation
  ```

  This needs no image dataset. Inspect the emitted `metrics.json` and provenance files in the output directory; success means the command completed with finite measurements, not that this tiny example established gamma dynamics. Use #link("/exp006/#start-a-small-training-run")[the training example] for a short train-and-evaluate workflow.

  Per-subcommand help defines accepted flags, but prose descriptions of defaults can lag the parser or model. For reproducibility, set important parameters explicitly and inspect the resolved configuration. Use a fresh `--out-dir` for every invocation: reused directories can mix old payloads with new metadata.

  == The tool and the experiment

  Keep reusable model operations in SNNSIM. Experiment stages choose scientific conditions and retain their own outputs. Existing CLI runners call the tool as a subprocess; Python execution is also a supported interface, so “all access is through files” is not an API-wide rule.

  Compute performs training or simulation. Analyse reads explicit completed evidence and calculates measurements. Present reads analysis outputs and renders figures. These stages complete independently: downstream work does not launch upstream computation, and no stage automatically publishes. The canonical runner contract is #link("https://github.com/eoinmurray/pinglab/blob/main/experiments/README.md")[the Experiment Runner Guide].

  A tool scratch directory is not a completed Pingstore run. A stage retains its outputs under a validated v3 run's `export/`, with authoritative provenance in `run.json`; a writing consumes an explicitly selected present run. See #link("/exp103/")[Compute options] for the operational workflow.

  == Commands

  === sim

  Run one forward pass and report firing-rate metrics. This performs no training; its cost depends on the network, duration, batch, and requested outputs.

  ```
  uv run python tools/snnsim/tool.py sim --model ping --input dataset \
    --dataset mnist --digit 3 --out-dir temp/docs-digit
  ```

  On its own it prints metrics. The flags below make it load trained weights, evaluate a test set, emit extra data artifacts, or inject perturbations. There is no `--image` or `--video`: where those retired flags once produced panels and sweep MP4s, `sim` now emits raw data (via `--outputs`) that the calling runner plots, and sweep videos are assembled runner-side from many `sim` calls.

  #table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    [*flag*], [*default*], [*description*],
    [`--infer`], [off], [Load trained weights and evaluate test-set accuracy; writes `results.json` (and `metrics.json`).],
    [`--load-config PATH`], [—], [Load a saved `config.json` and inherit its model, dataset, and parameters. Explicit CLI flags override loaded values.],
    [`--load-weights PATH`], [—], [Path to a `weights.pth` file for inference.],
    [`--max-samples INT`], [all], [[`--infer`] Cap the evaluation set to N samples.],
    [`--outputs OUTPUT [...]`], [—], [[`--infer`] Extra artifacts from the single forward pass (`metrics.json` always written): `per_cell_rates` (per-cell E/I Hz to `per_cell_rates.npz`), `pop_traces` (per-trial population activity to `pop_traces.npz`, base signal for PSD / f_γ), `rasters` (sparse spike indices to `rasters.npz`, for cycle-level analysis).],
    [`--tau-gaba FLOAT`], [inherited / 9.0], [[`--infer`] Override τ_GABA (ms) to replay a trained cell under specified inhibitory dynamics. Normally unset: `--load-config` inherits the trained value.],
    [`--skip-load PREFIX [...]`], [—], [[`--infer`] Drop `state_dict` keys with these prefixes before loading (e.g. `W_ei. W_ie.`) so a fresh sub-block survives. Transfer-load probes (#link("/exp038/")[exp038]).],
    [`--perturb-mode {drop, add}`], [—], [[`--infer`] Hidden-spike perturbation inside the forward loop: `drop` (Bernoulli mask), `add` (Poisson Hz). The #link("/exp037/")[exp037] drop/add asymmetry.],
    [`--perturb-level LEVEL [...]`], [—], [[`--perturb-mode`] One value: probability for `drop`, Hz for `add`.],
    [`--i-override-file PATH`], [—], [[`--infer`] NPZ with a sparse per-trial I-spike stream to substitute for the inhibitory spikes each timestep. Injection dual of `--outputs rasters` (#link("/exp042/")[exp042]).],
    [`--input-file PATH`], [—], [NPZ with `input_spikes` (T, B, N_IN) to forward instead of Poisson input. Arbitrary stimulus (#link("/exp048/")[exp048] digit streams).],
    [`--scale-w-in / --scale-w-ei / --scale-w-ie FLOAT`], [1.0], [[`--infer`] Multiply loaded input / E→I / I→E weights before the forward pass. Inference-time coupling sweeps without retraining (#link("/exp038/")[exp038]).],
    [`--sample-index INT`], [—], [Raw test-set index for a snapshot, overriding `--digit` / `--sample`.],
    [`--n-in / --n-inh / --n-batch INT`], [path-dependent / — / 64], [[synthetic-spikes] Input channels, inhibitory pool size, Poisson trials averaged.],
    [`--w-ei-mean / --w-ie-mean FLOAT`], [from `--ei-strength`], [[synthetic-spikes] Explicit W_EI / W_IE mean (std = 0.1·mean).],
    [`--private-w-in`], [off], [[synthetic-spikes] Identity W_in: one input channel per E cell.],
  )

  The block from `--skip-load` down is the *generic-primitive family*: small, experiment-agnostic hooks (perturb hidden spikes, inject an inhibitory stream, forward an arbitrary input file, scale a weight block at inference). CLI runners compose these operations without implementing their own neuron dynamics.

  === train

  Surrogate-gradient BPTT training loop. Writes selected and final parameter files, training metrics, and validation predictions. See #link("/exp006/#checkpoints-and-evaluation")[checkpoint roles] before choosing a weight file.

  ```
  uv run python tools/snnsim/tool.py train --model ping --dataset mnist \
    --epochs 50 --lr 0.0001 --v-grad-dampen 1000 \
    --readout mem-mean --seed 42 --out-dir temp/docs-production-example
  ```

  `--epochs 0` runs the init snapshot only, useful as a probe.

  #table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    [*flag*], [*default*], [*description*],
    [`--lr FLOAT`], [0.01], [AdamW learning rate. Choose the value from the intended recipe; the examples explicitly use 0.0001.],
    [`--epochs INT`], [0], [Number of epochs. 0 = init-snapshot probe only.],
    [`--batch-size INT`], [64], [DataLoader batch size.],
    [`--max-samples INT`], [all], [Cap dataset to N samples for smoke tests.],
    [`--v-grad-dampen FLOAT`], [80.0], [Divisor on the backward derivative through the biophysical membrane increment. The training example uses 1000; stability and learning must be checked for the chosen recipe. Mechanism and limitations in #link("/exp015/")[Gradient Stabilisation].],
    [`--fr-reg-upper-target-hz FLOAT`], [0], [Population-mean hidden-E firing-rate ceiling per presentation, in Hz. The one-sided squared overshoot is averaged over presentations and hidden layers; active only when the strength is positive.],
    [`--fr-reg-upper-strength FLOAT`], [0], [Coefficient on the upper regulariser; choose it with the target and rate units.],
    [`--tau-gaba FLOAT`], [9.0 ms], [Override the inhibitory conductance decay time. Recipe values can differ from the model default; retain the value used for training when replaying a checkpoint.],
  )

  === dump-weights

  Build the network from a config and emit its weight matrices to `weights_dump.npz`: the init state, plus (with `--load-weights`) the trained state. It runs no forward pass.

  ```
  uv run python tools/snnsim/tool.py dump-weights \
    --load-config temp/example-source/config.json \
    --load-weights temp/example-source/weights.pth \
    --out-dir temp/example-source/dump
  ```

  Keys follow `W_ff_N_init` / `W_ff_N_trained` (feedforward, per layer N) plus the E-I blocks `W_ei` / `W_ie`. This is how a runner recovers the trained readout matrix (W_out = the last `W_ff`) or compares init-vs-trained loop weights (the #link("/exp049/")[exp049] pruning analysis) without loading the model in-process. It takes the shared option groups plus `--load-config` / `--load-weights`.

  == Shared options

  These groups are attached to every subcommand. The grouping matches what each subcommand's `--help` prints.

  === Network

  #table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    [*flag*], [*default*], [*description*],
    [`--model {ping}`], [`ping`], [Architecture. `ping` is the COBANet with E↔I coupling; with `--ei-strength 0` the inhibitory loop is silenced to remove that feedback loop (other enabled pathways still matter).],
    [`--n-hidden INT [INT ...]`], [dataset-dependent], [Hidden layer sizes. One integer = single layer; multiple stacks layers. Default for mnist: 1024.],
    [`--readout MODE`], [`rate`], [Modes: `rate`, `mem-mean`, `spike-count`, `spike-rate`, `cumulative-potential`. See #link("/exp006/#readout")[the readout table] for their different score definitions.],
    [`--dales-law` / `--no-dales-law`], [on], [Enforce Dale's law (non-negative weights) or allow signed weights. `--no-dales-law` is used for balanced-network experiments.],
    [`--ei-strength FLOAT`], [0.5], [E-I coupling strength s. Sets the parent initialization means to s and s·ratio; stored edges are fan-in normalised.],
    [`--ei-ratio FLOAT`], [2.0], [W_IE / W_EI.],
    [`--w-in-initial-zero-fraction FLOAT`], [0.95], [Fraction of input parameters set to zero at initialization. All remain trainable and may regrow.],
    [`--recurrent-initial-zero-fraction FLOAT`], [0.0], [Fraction of recurrent parameters set to zero at initialization; survivors are rescaled by 1/(1−s). Trainable matrices may become dense.],
    [`--exact-k-initialization`], [off], [Choose exactly K = round((1−s)·N_pre) initially non-zero recurrent entries per post cell instead of Bernoulli zeroing. This does not impose a persistent connectivity mask.],
    [`--dt FLOAT`], [0.25], [Integration timestep (ms).],
    [`--t-ms FLOAT`], [200], [Total trial duration (ms). Metrics are measured over the full trace; runners strip any startup transient in post.],
    [`--readout-w-out-scale FLOAT`], [1.0], [Scalar applied to the readout matrix after `build_net`, compensating for low hidden firing rate under `mem-mean`. Train-mode only.],
    [`--surrogate-slope FLOAT`], [5.0 (model)], [If omitted, inherits the configured model value. Larger values narrow the window and increase its peak in Pinglab's normalization. See #link("/exp006/#surrogate-gradients")[Surrogate gradients].],
  )

  The drive family below also lives in the Network group. It exists for the balanced-network (Brunel / van Vreeswijk-Sompolinsky) experiments and the Lyapunov chaos probe; canonical PING runs leave all of it off.

  #table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    [*flag*], [*default*], [*description*],
    [`--independent-drive RATE G_PER_SPIKE`], [off], [Per-E-cell independent Poisson drive (bypasses W_in): N_E uncorrelated streams at RATE Hz, each spike adding G_PER_SPIKE μS of g_E. Zero cross-cell correlation.],
    [`--independent-drive-i RATE G_PER_SPIKE`], [off], [As above, targeting the I population directly. Needed for the full V&S asynchronous-irregular state.],
    [`--quenched-drive MEAN STD`], [off], [Per-E-cell DC conductance drawn once from N(MEAN, STD) μS and frozen for the trial. V&S quenched input: no fluctuation, so it cannot pin spike times; the Lyapunov probe then measures autonomous chaos.],
    [`--quenched-drive-i MEAN STD`], [off], [Per-I-cell frozen DC conductance.],
    [`--lyapunov-eps FLOAT`], [0 (off)], [If > 0 (synthetic-spikes mode), rerun with all membranes ε-perturbed at t=0 and save the divergence ‖ΔV(t)‖ to `snapshot.npz`. Use a defined estimator and fitting window before interpreting divergence as a Lyapunov exponent; the raw curve alone does not establish chaos.],
  )

  === Input

  #table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    [*flag*], [*default*], [*description*],
    [`--input {synthetic-spikes, dataset}`], [`synthetic-spikes`], [Stimulus regime. `synthetic-spikes` is Poisson at `--input-rate`. `dataset` draws from `--dataset`.],
    [`--input-rate FLOAT`], [25], [Baseline input rate (Hz).],
    [`--digit INT`], [0], [Dataset class (0–9).],
    [`--sample INT`], [0], [Sample index within the class.],
    [`--sample-index INT`], [—], [Raw test-set index, overriding `--digit` / `--sample`.],
    [`--dataset {mnist}`], [`mnist`], [Dataset. `mnist` is the full 28×28 image encoded to spikes.],
  )

  === Weights (advanced)

  #table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    [*flag*], [*default*], [*description*],
    [`--w-in MEAN [STD]`], [`0.3 0.06`], [Input fan-in init. Single value sets STD = MEAN × 0.1.],
    [`--w-ei MEAN STD`], [from `--ei-strength`], [Override the W_EI init.],
    [`--w-ie MEAN STD`], [from `--ei-strength` / `--ei-ratio`], [Override the W_IE init.],
    [`--w-ii MEAN STD`], [`0 0`], [W_II (I→I) init. Off by default (canonical PING has no I→I). Enable for balanced-network experiments.],
    [`--w-ee MEAN STD`], [`0 0`], [W_EE (E→E) init. Off by default. Enable for the full four-coupling balanced network, where recurrent excitation pins the E rate.],
    [`--trainable-w-ei`], [frozen], [Promote E→I to gradient-carrying. Asks whether the optimiser will discover the PING-loop weights from scratch.],
    [`--trainable-w-ie`], [frozen], [Promote I→E. The #link("/exp049/")[exp049] result _gradient descent dismantles PING_ comes from flipping `--trainable-w-ei` and `--trainable-w-ie` on simultaneously.],
  )

  === Output

  #table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    [*flag*], [*default*], [*description*],
    [`--out-dir DIR`], [`temp/pinglab-cli/`], [Output directory. The default is scratch (gitignored); runners always pass an explicit path.],
    [`--wipe-dir`], [off], [Clear the output directory before the run.],
  )

  === Execution

  #table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    [*flag*], [*default*], [*description*],
    [`--seed INT`], [—], [RNG seed. Seeds Python, NumPy, torch (CPU + CUDA + MPS) before dataset load and model init. Persisted to `config.json`.],
    [`--modal`], [off], [Re-dispatch to Modal.com. Artifacts sync back to `--out-dir` after completion.],
    [`--modal-gpu {none, T4, L4, A10G, A100, H100}`], [`T4`], [GPU type for Modal runs. `none` runs CPU-only.],
  )

  #quote(block: true)[
    `--modal` costs money. The project default is local; only pass it when explicitly instructed.
  ]

  == Config inheritance

  `--load-config` carries the saved configuration into a later legacy `sim` or `dump-weights` invocation:

  ```
  uv run python tools/snnsim/tool.py sim --infer \
    --load-config temp/example-source/config.json \
    --load-weights temp/example-source/weights.pth \
    --dt 0.5 --out-dir temp/example-timestep
  ```

  This inherits the model, hidden sizes, dataset, E-I parameters, input rate, τ_GABA, and seed, while the explicitly-passed `--dt 0.5` overrides the trained value, replaying the network at a new timestep. Precedence is: explicit CLI flag, then loaded config, then default. The parser builds the set of CLI-explicit flags from `sys.argv` before applying inheritance.

  Backwards compatibility: old configs that stored `n_hidden` as a scalar are remapped to the `hidden_sizes` list, legacy model names are aliased with a one-line stderr note, and configs missing `dales_law` trigger a warning to pass it explicitly or retrain.

  == Artifacts <artifacts>

  The legacy command path uses `save_run_artifacts` to write provenance into `--out-dir`:

  #table(
    columns: (auto, 1fr),
    align: (left, left),
    [*file*], [*contents*],
    [`config.json`], [The parsed argparse namespace plus a provenance block (`git_sha` with a `dirty` suffix, `torch_version`, `device`, `python_env_hash`, `run_id`, `started_at`) and the `mode`. Consumed by `--load-config` and the runner's metadata extractors.],
    [`run.sh`], [The literal `sys.argv` joined with spaces and prefixed with a shebang. This is a command record, not a portable replay guarantee: the current writer joins arguments without shell quoting or an explicit Python launcher. Inspect it and reconstruct the command with `uv run python`, the matching code, inputs, and configuration.],
    [`output.log`], [The human run log. ANSI escapes are stripped from the file but preserved on stdout, so terminals see colour while the log stays grep-friendly.],
    [`run.jsonl`], [The machine-readable event spine: one typed JSON object per event (epoch rows, warnings, summary). This is what a runner parses when it wants structured progress, not scraped log text.],
  )

  Each command adds its own outputs: `train` writes `weights.pth` when a selected epoch exists, `weights_final.pth`, `metrics.json`, `metrics.jsonl`, and the legacy-named `test_predictions.json` containing validation predictions; `sim --infer` writes `metrics.json` (and `results.json`) plus whatever `--outputs` requested; `dump-weights` writes `weights_dump.npz`.

  These files are scratch. By default they land under `temp/pinglab-cli/`, which is gitignored; a new invocation can overwrite metadata without removing old payloads. The retained record is produced by the experiment stages: analysis derives numerical results, and presentation exports report-ready `numbers.json` and figures into an immutable Pingstore run. The writing reads the selected presentation input through `run-inputs.typ`, never the ephemeral tool output. Without a selected input, the report shows an unavailable-data notice.

  == Recipes

  *Train, then measure the trained network.* Train writes a run directory; `sim --infer` reads it back and emits the population traces a runner needs for a PSD:

  ```
  uv run python tools/snnsim/tool.py train --dataset mnist --epochs 50 \
    --lr 0.0001 --v-grad-dampen 1000 --readout mem-mean \
    --seed 42 --out-dir temp/example-training

  uv run python tools/snnsim/tool.py sim --infer \
    --load-config temp/example-training/config.json \
    --load-weights temp/example-training/weights.pth \
    --outputs pop_traces per_cell_rates --out-dir temp/example-measurement
  ```

  *Scale existing recurrent weights at inference.* The scale flags multiply loaded matrices; they cannot turn zero weights into nonzero coupling. To introduce a previously absent block, use an explicit compatible initialization/transfer procedure rather than assuming a multiplier creates it.

  ```sh
  uv run python tools/snnsim/tool.py sim --infer \
    --load-config temp/example-training/config.json \
    --load-weights temp/example-training/weights.pth \
    --scale-w-ei 0.5 --scale-w-ie 1.0 --outputs rasters \
    --out-dir temp/example-scaled-inference
  ```

  *Perturbation sweep (#link("/exp037/")[exp037]).* Drop a fraction of emitted spikes, or add off-phase Poisson noise, inside the forward loop:

  ```
  uv run python tools/snnsim/tool.py sim --infer \
    --load-config temp/example-training/config.json --load-weights temp/example-training/weights.pth \
    --perturb-mode drop --perturb-level 0.8 --out-dir temp/example-perturbation
  ```

  *Recover the trained readout matrix.* Dump weights and read `W_ff_N_trained` (the last layer is W_out):

  ```
  uv run python tools/snnsim/tool.py dump-weights \
    --load-config temp/example-training/config.json \
    --load-weights temp/example-training/weights.pth --out-dir temp/example-training/dump
  ```
  == Troubleshooting

  + *Unexpected input.* Set `--input dataset` or `--input synthetic-spikes` explicitly. The parser can infer dataset mode from explicit dataset/digit/sample flags or a loaded MNIST configuration when `--input` is omitted.
  + *Unexpected inherited value.* Explicit CLI flags take precedence over `--load-config`. Check the resolved output config, not just the source file.
  + *Unexpected files.* Do not reuse scratch directories or infer completion from file presence alone. Failed runs can leave partial output.
  + *Graph request rejected.* Check `--executor graph`, the bundle, named input bindings, and supported options in #link("/exp090/")[Compatibility, status, and extension].
  + *Training is finite but ineffective.* Inspect skipped steps, readout choice, and gradient diagnostics; see #link("/exp015/")[Gradient Stabilisation].

  The option tables cover common controls, not every parser option. For implementation details, follow `parse_args` and `configure_models` in `tools/snnsim/tool.py`, then the selected executor. Tool-level Modal execution and experiment-runner dispatch are different interfaces: do not assume the tool's `--modal` waits for a separate `--live` flag.

  #link("/exp004/")[Next: Parameters & Units]
]

#let body = with-datasets("exp011", (), body)
#let body = with-numbered-equations(body)
#let body = with-contents(body)
