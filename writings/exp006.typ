#import "contents.typ": with-contents, with-numbered-equations
#import "run-view.typ": with-datasets
#let meta = (
  tags: ("txt", "v35.0.0"),
  title: "Training",
  updated_at: "2026-08-29T00:00:00Z",
  created_at: "2026-05-14T00:00:00Z",
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

  Take a hidden state $h^k$ that evolves as

  $ h^k = f(h^(k-1), x^k; theta_"param"), quad z^k = g(h^k; theta_"param") $

  with input $x^k$, score output $z^k$, and parameters $theta_"param"$ shared across time. Running $N_t$ steps gives a chain $h^0 -> h^1 -> dots.c -> h^(N_t)$, which for gradients is treated as a depth-$N_t$ feedforward network with tied weights.

  For a scalar loss $L_"total"$ depending on the final state, define $a^k = partial L_"total" / partial h^k$ as the total sensitivity to state $h^k$. Then the contribution through state updates is

  $ (partial L_"total") / (partial theta_j) = sum_(k=1)^(N_t) (a^k)^top (partial f(h^(k-1), x^k; theta_"param")) / (partial theta_j). $

  Here $theta_j$ is one shared parameter and the derivative of $f$ holds its state and input arguments fixed. Direct parameter use in the readout contributes an additional term. Losses or readouts accumulated over time also inject sensitivities at intermediate steps.

  The backward pass contains products of per-step Jacobians $partial h^(k+1)\/partial h^k$, the matrices of state derivatives. Repeated contraction can suppress gradients and amplification can enlarge them; individual norms above 1 do not by themselves prove that the product grows.

  One simulation step is one step of the recurrence: the state includes membrane voltages, synaptic conductances, and refractory counters. A 200 ms trial at $Delta t_"sim" = 0.1$ ms unrolls to $N_t = 2000$ steps. Gradient behaviour depends on the trajectory, surrogate, weights, and reset gates; recurrent coupling alone does not prove divergence. See #link("/exp015/")[Gradient Stabilisation] for the implemented intervention.

  == Surrogate gradients

  The spike function $s[k] = bold(1)[V_"candidate" >= V_"th"]$ has zero gradient almost everywhere, so the backward pass substitutes a smooth surrogate. The legacy spike helper uses a fast-sigmoid surrogate. Forward is the hard step; backward is

  $ (partial tilde(s)) / (partial V_"candidate") = k_"sg" / (1 + k_"sg" |V_"candidate" - V_"th"|)^2 $

  Here $V_"candidate"$ is the pre-reset membrane value, $V_"th"$ the spike threshold, and $k_"sg"$ the surrogate slope in inverse voltage units; $tilde(s)$ denotes the backward surrogate, not the forward spike. This is Pinglab's normalization. snnTorch's #link("https://snntorch.readthedocs.io/en/latest/_modules/snntorch/surrogate.html#FastSigmoid")[FastSigmoid] uses numerator 1 instead of $k_"sg"$: equal slopes do not generally give equal gradients.

  It takes its slope from `SURROGATE_SLOPE = 5.0`, overridable per-run with `--surrogate-slope`.

  == Gradient stabilisation

  `--v-grad-dampen` scales the backward derivative through the biophysical membrane increment. It does not rescale every gradient or guarantee stable training. Start from the chosen recipe and inspect the diagnostics before changing it; #link("/exp015/")[Gradient Stabilisation] explains the local derivatives and trade-offs.

  == The training loop

  Logits from the readout go into cross-entropy loss:

  $ L_"CE" = -1 / B sum_(b=1)^B log (exp(z_(b, c_b))) / (sum_c exp(z_(b, c))) $

  Here $L_"CE"$ is cross-entropy loss, $B$ is minibatch size, $b$ indexes presentations, $z_b$ is the score vector, $c_b$ the true class, and $c$ indexes classes. Uniform predictions on ten classes give loss $ln 10 approx 2.30$. The implementation uses AdamW with `--weight-decay 0` by default. Gradients are clipped to unit norm (`GRAD_CLIP = 1.0`); an update with non-finite gradient norm is skipped. Checkpoint selection uses validation loss, not this training loss.

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

  $ r_b = 1 / (N_E T_"present") sum_(n in E) n_"spike"(b,n), quad
    L_"rate" = lambda_"rate" / B sum_b "ReLU"(r_b - r_(E,"ceil"))^2 $

  Here $n_"spike"(b,n)$ is the spike count of hidden excitatory neuron $n$ in presentation $b$, $N_E$ is the number of those neurons, $T_"present"$ is presentation duration in seconds, $B$ is minibatch size, $r_b$ and $r_(E,"ceil")$ are rates in Hz, and $lambda_"rate"$ is the configured rate-penalty coefficient.

  The ceiling is applied separately to each presentation's population-mean hidden-E rate before averaging across the minibatch. The loss is normalised over neurons, presentation duration, samples, and hidden layers. This is the mechanism behind #link("/exp025/")[the activity-ceiling comparison]; #link("/exp024/")[the training-convergence study] tests the associated rate-plateau interpretation.

  == Weight init

  Dale-constrained magnitudes use a lower-clamped Gaussian, not a half-normal or truncated normal:

  $ X_(i j) tilde cal(N)(mu_"init", sigma_"init"^2), quad X_(i j)^+ = max(0, X_(i j)). $

  Here $X_(i j)$ is a Gaussian draw for input index $i$ and output index $j$, and $X_(i j)^+$ is its non-negative clamp. The configured $mu_"init"$ and $sigma_"init"$ are parent-Gaussian parameters on the summed-coupling scale, not moments of one stored edge. With initial-zero fraction $q_"zero" in [0, 1)$ and Bernoulli indicator $M_(i j)$, the stored initialization is

  $ W_(i j)^(0) = M_(i j) X_(i j)^+ \/ ((1-q_"zero") N_"pre"). $

  Here $N_"pre"$ is fan-in and $M_(i j)$ is 1 with probability $1-q_"zero"$, otherwise 0. Direct readout initialization (`--readout-w-init-mean` and `--readout-w-init-std`) bypasses this fan-in scaling; do not interpret its parameters as summed coupling.

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
#let body = with-numbered-equations(body)
#let body = with-contents(body)
