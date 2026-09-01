#import "contents.typ": with-contents, with-numbered-equations
#import "run-view.typ": with-datasets
#let meta = (
  status: "[≡ TXT | v31.1.0]",
  title: "Gradient Stabilisation",
  created_at: "2026-06-12T00:00:00Z",
  updated_at: "2026-08-29T00:00:00Z",
  description: "What voltage-gradient damping changes in SNNSIM, how to check it, and how to diagnose unstable training without confusing local derivatives with a global stability guarantee.",
  collection: "snnsim-docs",
  order: 5,
)

#let body = [
  == Using the control

  `--v-grad-dampen` changes the backward pass through the legacy biophysical neuron's membrane increment. It is intended to reduce gradient amplification without intentionally changing the forward model. It does not divide the entire voltage gradient by a constant, repair a non-finite forward pass, or guarantee convergence.

  In `tools/snnsim/models.py`, both `lif_step` and `lif_step_expeuler` apply `_scale_grad(dv, 1.0 / v_grad_dampen)` to the voltage increment `dv`. The legacy training CLI defaults to 80; the #link("/exp006/#start-a-small-training-run")[small training example] explicitly uses 1000. Treat either value as a configuration choice, not a universal threshold. A value of 1 disables the scaling. Use positive values, and values at least 1 when the intention is damping rather than amplification.

  Damping changes the optimization problem's supplied gradients. Keep it fixed when reproducing a recipe; record it when comparing training runs. Graph-native training has its own recipe contract: see #link("/exp088/")[Training recipes and graph-native learning].

  == Check the primitive

  This small check exercises the actual helper without training a network or downloading data. Run it from the repository root:

  ```python
  import torch
  from tools.snnsim.models import _scale_grad

  x = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
  y = _scale_grad(x, 0.01)
  y.sum().backward()
  torch.testing.assert_close(y, x)
  torch.testing.assert_close(x.grad, torch.tensor([0.01], dtype=x.dtype))
  ```

  Expected result: both assertions pass. The forward value is approximately 2 and its derivative is approximately 0.01. The helper is private; this is an implementation check, not a new public API.

  The helper computes

  $ F_c(x) = c x + (1-c) "detach"(x). $

  Here $x$ is the input tensor, $c$ is a dimensionless gradient scale, and `detach` retains the value while removing its autograd dependency. In exact arithmetic $F_c(x)=x$, but autograd returns derivative $c$. Floating-point multiplication and addition can introduce rounding: this expression does not establish bitwise identity of full trajectories. PyTorch documents the dependency boundary in #link("https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html")[Tensor.detach].

  == The implemented update

  The following equations describe the local exponential-Euler membrane update with noise and active voltage clamps excluded. They are an implementation derivation, not a proof of global network stability. The #link("/exp100/")[COBANet] page covers the surrounding dynamics.

  The synapse helper decays the previous conductance and then adds the spike kick:

  $ g^(k+1) = beta_"syn" g^k + s[k] W, quad beta_"syn" = e^(-Delta t_"sim" / tau_"syn"). $

  Here $g^k$ is a row of conductances in μS at step $k$, $s[k]$ the supplied dimensionless presynaptic spike row, $W$ the stored weight matrix in μS, $Delta t_"sim"$ the integration timestep in ms, and $tau_"syn"$ the pathway's synaptic decay time in ms. In particular, the new kick is not multiplied by $beta_"syn"$. Network scheduling determines which spike row reaches each pathway.

  Using the updated excitatory and inhibitory conductances $g_e$ and $g_i$, define

  $ g_"tot" = g_L + g_e + g_i, quad
    V_oo = (g_L E_L + g_e E_e + g_i E_i) / g_"tot", quad
    alpha_"mem" = e^(-Delta t_"sim" g_"tot" / C_m). $

  Here $g_L$ is leak conductance, $C_m$ capacitance in nF, and $E_L$, $E_e$, $E_i$ the leak, excitatory, and inhibitory reversal potentials in mV. $g_"tot"$ is total conductance, $V_oo$ the frozen-conductance equilibrium voltage, and $alpha_"mem"$ the dimensionless membrane decay.

  For current voltage $V_m$, the increment and candidate voltage are

  $ dif V_m = (V_oo - V_m)(1-alpha_"mem"), quad V_"candidate" = V_m + F_(1/d_"grad")(dif V_m). $

  Here $d_"grad"$ is the dimensionless damping divisor configured by `v_grad_dampen`, and $V_"candidate"$ is the candidate voltage before noise, clamps, thresholding, and reset. The implementation advances the voltage before testing its spike threshold. On a spiking or refractory neuron, `torch.where` replaces the retained voltage with the reset value; the emitted spike remains a separate output with its surrogate derivative.

  == What the derivatives say

  Holding conductances fixed, the backward derivative of the candidate voltage is

  $ (partial V_"candidate") / (partial V_m) = 1 - (1-alpha_"mem")/d_"grad". $

  Thus damping the increment preserves the direct $V_m$ pathway; it does not replace the full derivative by $alpha_"mem"/d_"grad"$. For $d_"grad" >= 1$ and positive conductances this local derivative lies between $alpha_"mem"$ and 1.

  Define the undamped conductance sensitivity for channel $q in {e,i}$ as

  $ kappa_q = (1-alpha_"mem")(E_q - V_oo)/g_"tot"
    - (Delta t_"sim" / C_m) alpha_"mem" (V_m - V_oo). $

  $E_q$ is that channel's reversal potential. The damped candidate-voltage derivative is $partial V_"candidate" / partial g_q = kappa_q / d_"grad"$. At small timesteps, $kappa_q approx (Delta t_"sim" / C_m)(E_q-V_m)$. This is where the control reduces sensitivity to both recurrent and feedforward conductance inputs.

  These are local derivatives before reset and active clamps. Reset gates every derivative of the retained reset voltage, not just its membrane self-term. Gradients through the emitted spike can still propagate along other paths. The full backward pass combines these paths across cells and time; time-accumulated readouts also inject gradients at multiple steps.

  A scalar loop-gain estimate can be a heuristic, but does not prove that gradients grow once per gamma cycle or that dividing an estimated gain by $d_"grad"^2$ makes the entire network contractive. Such claims require the actual trajectory, stored fan-in-scaled weights, surrogate normalization, gates, and full coupled Jacobians. Neither successful forward simulation nor removal of the inhibitory loop guarantees stable learning.

  == Diagnosing a training failure

  + *Locate the first non-finite value.* Check inputs, forward states, loss, and then gradients. If the forward pass is invalid, changing the backward derivative is not the repair.
  + *Inspect the optimizer diagnostics.* The legacy trainer records gradient norms and skipped updates. It clips the assembled gradient to norm 1 and skips an update if that norm is non-finite. Clipping cannot make an already invalid gradient informative.
  + *Change one control at a time.* Compare positive damping values while keeping the seed, input, duration, timestep, initialization, and optimizer settings fixed. Lower learning rate and stronger damping are different interventions.
  + *Check learning as well as finiteness.* Stronger damping also reduces useful conductance-input sensitivities. A finite loss with negligible learning is not sufficient evidence of a good setting.
  + *Keep the scope of the check explicit.* The helper assertion verifies its local derivative. Establishing training stability or accuracy requires a separately authorized experiment, with recorded diagnostics and validation results.

  #link("/exp006/")[Previous: Training] · #link("/exp011/")[Back to SNNSIM command-line guide]
]

#let body = with-datasets("exp015", (), body)
#let body = with-numbered-equations(body)
#let body = with-contents(body)
