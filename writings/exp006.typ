#import "contents.typ": with-contents
#import "run-view.typ": with-datasets
#let meta = (
  status: "[≡ TXT]",
  title: "Training",
  updated_at: "2026-08-28",
  date: "2026-05-14",
  description: "How to configure legacy SNNSIM training, interpret checkpoints and readouts, and understand the gradients, regularisation, and initialization used by the implementation.",
  collection: "snnsim-docs",
  order: 4,
)

#let body = [
  This page describes the legacy `--executor legacy` training path in `tools/snnsim/train.py`. It connects the training controls to their implementation. For declarative graph training, use #link("/exp088/")[Training recipes and graph-native learning]; graph checkpoints and input bindings have a different contract.

  == Start a small training run

  From the repository root, choose a fresh scratch directory and run a short MNIST plumbing check:

  ```sh
  uv run python tools/snnsim/tool.py train --executor legacy \
    --model ping --dataset mnist --n-hidden 32 --epochs 1 \
    --max-samples 128 --batch-size 16 --t-ms 20 --dt 0.25 \
    --readout mem-mean --lr 0.0001 --v-grad-dampen 1000 \
    --seed 42 --out-dir temp/docs-training
  ```

  This example exercises the training path; its short duration and small dataset are not a scientific baseline. The first run may download MNIST. Inspect finite loss, gradient diagnostics, skipped updates, and output files before scaling up. `--epochs 0` is an initialization probe, not a trained checkpoint. Use the committed experiment recipe for a production run.

  == Checkpoints and evaluation

  For MNIST, training uses a held-out validation partition from the official training data. Checkpoint selection minimises validation cross-entropy averaged over encoder draws; validation accuracy breaks exact loss ties. The official test partition is reserved for separate inference.

  #table(
    columns: (auto, 1fr),
    [*Output*], [*Meaning*],
    [`weights.pth`], [Validation-selected parameters, when a selected epoch exists.],
    [`weights_final.pth`], [Parameters at the end of training. Use when measuring final-epoch dynamics.],
    [`config.json`], [Resolved training configuration, split and encoder settings.],
    [`metrics.json` / `metrics.jsonl`], [Summary and training history; checkpoint metadata records roles and hashes.],
    [`test_predictions.json`], [Legacy filename for validation predictions; the filename does not make these held-out test results.],
  )

  A weight file is not a complete optimizer-resume checkpoint. Keep it with its configuration and recorded role. To evaluate the selected parameters separately:

  ```sh
  uv run python tools/snnsim/tool.py sim --infer --executor legacy \
    --load-config temp/docs-training/config.json \
    --load-weights temp/docs-training/weights.pth \
    --max-samples 128 --outputs per_cell_rates \
    --out-dir temp/docs-evaluation
  ```

  This capped evaluation is a plumbing check. Neither command creates a completed Pingstore run; experiment stages retain scientific evidence separately.

  == Backpropagation through time

  Every model here is a recurrent system run forward in time, so gradients come from Backpropagation Through Time (BPTT): unroll the recurrence into a deep feedforward graph — one layer per timestep, all sharing the same weights — and backpropagate through it.

  Take a hidden state $h^t$ that evolves as

  $ h^t = f(h^(t-1), x^t; theta), quad y^t = g(h^t; theta) $

  with input $x^t$, output $y^t$, and parameters $theta$ shared across time. Running $T$ steps gives a chain $h^0 -> h^1 -> dots.c -> h^T$, which for gradients is treated as a depth-$T$ feedforward network with tied weights.

  For a scalar loss $cal(L)$ depending on the final state, define $lambda^t = partial cal(L) / partial h^t$ as the total sensitivity to state $h^t$. Then the contribution through state updates is

  $ (partial cal(L)) / (partial theta_k) = sum_(t=1)^T (lambda^t)^top (partial f(h^(t-1), x^t; theta)) / (partial theta_k). $

  Here $theta_k$ is one shared parameter and the derivative of $f$ holds its state and input arguments fixed. Direct parameter use in the readout contributes an additional term. Losses or readouts accumulated over time also inject sensitivities at intermediate steps.

  The backward pass contains products of per-step Jacobians $partial h^(t+1)\/partial h^t$, the matrices of state derivatives. Repeated contraction can suppress gradients and amplification can enlarge them; individual norms above 1 do not by themselves prove that the product grows.

  One simulation step is one step of the recurrence: the state includes membrane voltages, synaptic conductances, and refractory counters. A 200 ms trial at $Delta t = 0.1$ ms unrolls to $T = 2000$ steps, where $Delta t$ is the timestep and $T$ the number of steps. Gradient behaviour depends on the trajectory, surrogate, weights, and reset gates; recurrent coupling alone does not prove divergence. See #link("/exp015/")[Gradient Stabilisation] for the implemented intervention.

  == Surrogate gradients

  The spike function $S = bold(1)[U >= theta]$ has zero gradient almost everywhere, so the backward pass substitutes a smooth surrogate. The legacy spike helper uses a fast-sigmoid surrogate. Forward is the hard step; backward is

  $ (partial tilde(S)) / (partial U) = (k) / ((1 + k |U - theta|)^2) $

  Here $U$ is the pre-reset membrane value, $theta$ the spike threshold, and $k$ the slope in inverse units of $U$; $tilde(S)$ denotes the backward surrogate, not the forward spike. This is Pinglab's normalization. snnTorch's #link("https://snntorch.readthedocs.io/en/latest/_modules/snntorch/surrogate.html#FastSigmoid")[FastSigmoid] uses numerator 1 instead of $k$: equal slopes do not generally give equal gradients.

  It takes its slope from `SURROGATE_SLOPE = 5.0`, overridable per-run with `--surrogate-slope`.

  == Gradient stabilisation

  `--v-grad-dampen` scales the backward derivative through the biophysical membrane increment. It does not rescale every gradient or guarantee stable training. Start from the chosen recipe and inspect the diagnostics before changing it; #link("/exp015/")[Gradient Stabilisation] explains the local derivatives and trade-offs.

  == The training loop

  Logits from the readout go into cross-entropy loss:

  $ L_"CE" = -(1) / (B) sum_(b=1)^B log (exp(hat(y)_(b, c_b))) / (sum_k exp(hat(y)_(b, k))) $

  Here $L_"CE"$ is cross-entropy loss, $B$ is batch size, $b$ indexes samples, $hat(y)_b$ is the score vector, $c_b$ the true class, and $k$ indexes classes. Uniform predictions on ten classes give loss $ln 10 approx 2.30$. The implementation uses AdamW with `--weight-decay 0` by default. Gradients are clipped to unit norm (`GRAD_CLIP = 1.0`); an update with non-finite gradient norm is skipped. Checkpoint selection uses validation loss, not this training loss.

  == Readout

  `--readout` selects the legacy class-score calculation. Set it explicitly when comparing recipes; the CLI default is `rate`, not `mem-mean`.

  #table(
    columns: (auto, 1fr),
    [*Mode*], [*Implemented reduction*],
    [`rate`], [Sum last-hidden spikes, then multiply by the readout matrix. Despite its name, this path does not divide by duration or apply softmax.],
    [`mem-mean`], [Average the output LIF's pre-reset membrane over time. Its subtractive reset changes later voltages.],
    [`spike-count`], [Count spikes of each output LIF neuron, not the hidden population.],
    [`spike-rate`], [Divide output-LIF counts by presentation duration in seconds.],
    [`cumulative-potential`], [Accumulate per-step softmax values from a non-spiking leaky decoder.],
  )

  `li` is not an accepted legacy CLI mode. Changing a readout changes the score scale and gradient path; it is not merely a display choice. Output membrane parameters are separate from the hidden biophysical constants.

  == Firing-rate regularisation

  Hidden activity can be limited with `--fr-reg-upper-target-hz` and `--fr-reg-upper-strength`:

  $ r_b = 1 / (N_E D) sum_(n in E) z_(b n), quad
    cal(L)_"fr" = s_u / B sum_b "ReLU"(r_b - r_"max")^2 $

  Here $z_(b n)$ is the spike count of hidden excitatory neuron $n$ in sample $b$, $N_E$ is the number of those neurons, $D$ is presentation duration in seconds, $B$ is batch size, $r_b$ and $r_"max"$ are rates in Hz, and $s_u$ is the configured penalty coefficient.

  The ceiling is applied separately to each presentation's population-mean hidden-E rate before averaging across the minibatch. The loss is normalised over neurons, presentation duration, samples, and hidden layers. This is the mechanism behind the activity sweep in #link("/exp025/")[exp025] and the rate-floor framing in #link("/exp109/")[exp109].

  == Weight init

  Dale-constrained magnitudes use a lower-clamped Gaussian, not a half-normal or truncated normal:

  $ X_(i j) tilde cal(N)(mu, sigma^2), quad U_(i j) = max(0, X_(i j)). $

  Here $X_(i j)$ is a Gaussian draw for input index $i$ and output index $j$, and $U_(i j)$ is its non-negative clamp. The configured $mu$ and $sigma$ are parent-Gaussian parameters on the summed-coupling scale, not moments of one stored edge. With initial-zero fraction $s in [0, 1)$ and Bernoulli indicator $B_(i j)$, the stored initialization is

  $ W_(i j)^(0) = B_(i j) U_(i j) \/ ((1-s) N_"pre"). $

  Here $N_"pre"$ is fan-in and $B_(i j)$ is 1 with probability $1-s$, otherwise 0. Direct readout initialization (`--readout-w-init-mean` and `--readout-w-init-std`) bypasses this fan-in scaling; do not interpret its parameters as summed coupling.

  The compensation keeps the expected column sum independent of $s$; lower clamping means that expected sum is $cal(E)[max(0, X)]$, which is recorded alongside the configured parent parameters. Both lower-clamp zeros and explicitly zeroed entries remain trainable and may become positive. This is sparse initialization of a dense trainable matrix, not structural sparsity.

  == Dale's law during optimization

  When Dale's law is on, the feedforward matrices $W_"ff"$ are clamped to $W >= 0$ when they are read by the forward pass and every trainable constrained matrix is projected back into the non-negative cone by `project_dales()` after each optimiser step. The recurrent conductance matrices $W_(e e)$, $W_(e i)$, $W_(i e)$, and $W_(i i)$ are not forward-clamped: they are initialised non-negative and, when trainable, kept non-negative by the post-step projection. Their entries are conductance magnitudes; pathway-specific reversal potentials, rather than a negative stored $W_(i e)$, determine whether a synapse is excitatory or inhibitory.

  == Troubleshooting

  + *Nothing trained.* Check `--epochs`, the skipped-update count, and whether `weights.pth` exists. An initialization snapshot is not a successful training run.
  + *Non-finite gradients.* Inspect the first failing batch, forward values, and gradient norms. Damping and clipping act at different points; neither repairs invalid inputs or an unstable forward model.
  + *Unexpected scores or rates.* Check readout mode, duration, input encoding, and whether the selected or final checkpoint was loaded.
  + *Unexpected memory use.* BPTT retains a time-unrolled graph. Reduce the plumbing example's batch size, duration, or network width before attempting the production recipe.

  Implementation reference: `tools/snnsim/train.py`, `tools/snnsim/models.py`, and `tools/snnsim/tool.py`.

  #link("/exp100/")[Previous: COBANet] · #link("/exp015/")[Next: Gradient Stabilisation]
]

#let body = with-datasets("exp006", (), body)
#let body = with-contents(body)
