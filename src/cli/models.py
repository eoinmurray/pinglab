"""SNN model definitions and layer primitives.

All constants are hardcoded defaults. Override via module-level assignment
or by passing arguments to model constructors.

Models: COBANet.
Layer primitives: exp_synapse, lif_step.
"""

from __future__ import annotations


import torch
import torch.nn as nn
import numpy as np

# ── Simulation ────────────────────────────────────────────────────────────
dt: float = 0.25  # ms — integration timestep
T_ms: float = 1000.0  # ms — total simulation time per sample
T_steps: int = int(T_ms / dt)

# ── Biophysics ────────────────────────────────────────────────────────────
tau_m_E = 20.0  # ms — excitatory membrane time constant
tau_m_ratio = 4.0  # tau_m_E / tau_m_I (Börgers: 20ms / 5ms)
C_m_E = 1.0  # nF — excitatory capacitance (fixed reference)
_CM_RATIO = 2.0  # C_m_E / C_m_I (fixed)
g_L_E = C_m_E / tau_m_E  # 0.05 uS
tau_m_I = tau_m_E / tau_m_ratio  # 5.0 ms
C_m_I = C_m_E / _CM_RATIO  # 0.5 nF
g_L_I = C_m_I / tau_m_I  # 0.1 uS
E_L = -65.0  # mV — leak / resting potential
E_e = 0.0  # mV — excitatory (AMPA) reversal
E_i = -80.0  # mV — inhibitory (GABA) reversal
V_th = -50.0  # mV — spike threshold
V_reset = -65.0  # mV — post-spike reset voltage
V_floor = -200.0  # mV — hard lower clamp
ref_ms_E = 3.0  # ms — excitatory refractory period
_REF_RATIO = 2.0  # ref_ms_E / ref_ms_I (Börgers)
ref_ms_I = ref_ms_E / _REF_RATIO  # 1.5 ms
tau_ampa = 2.0  # ms — AMPA decay
tau_gaba = 9.0  # ms — GABA decay (Börgers: 9 ms; Buzsaki & Wang: 8-12 ms)

# ── Input encoding ────────────────────────────────────────────────────────
max_rate_hz = (
    25.0  # Hz — max Poisson rate for fully-on pixel (sensory-input scale, LGN-ish)
)

# ── snnTorch ──────────────────────────────────────────────────────────────
tau_snn = 10.0  # ms — membrane time constant
# Output-LIF time constant for the spike-count readout. Smaller than
# tau_snn means faster leak: the output membrane decays before saturating
# under high-rate hidden drive, which is what was breaking the spike-count
# readout for snnTorch-family models at coarse dt where hidden rates run
# 60–1000 Hz. Override per-run via --readout-tau-out.
tau_out_ms = 2.0  # ms
thr_snn = 1.0  # spike threshold

# ── Architecture ──────────────────────────────────────────────────────────
N_IN: int = 64  # input neurons (8×8 scikit-digits)
N_HID: int = 64  # hidden excitatory neurons (last layer size for compat)
N_INH: int = 16  # inhibitory neurons (PING only, per E-I layer)
N_OUT: int = 10  # output neurons (one per digit class)
HIDDEN_SIZES: list[int] = [64]  # hidden layer sizes (N_HID is always last entry)

# ── Weight init ───────────────────────────────────────────────────────────
# Weight init — p1/p2 are pre-fan-in values (init_weight divides by N_pre)
W_FF_MEAN = 5.1  # uS — feedforward init mean (pre-fan-in)
W_FF_STD = 3.8  # uS — feedforward init std (pre-fan-in)
W_IN_MEAN = W_FF_MEAN  # alias
W_IN_STD = W_FF_STD  # alias
W_HID_MEAN = W_FF_MEAN  # alias
W_HID_STD = W_FF_STD  # alias
W_EE_MEAN = 0.0  # uS — E→E recurrent init mean (0 per Börgers: PING needs no E→E)
W_EE_STD = 0.0  # uS — E→E recurrent init std (0 per Börgers: PING needs no E→E)
W_EI_MEAN = 1.0  # uS — E→I init mean (just suprathreshold, Börgers)
W_EI_STD = 0.5  # uS — E→I init std (pre-fan-in)
W_IE_MEAN = 3.0  # uS — I→E init mean (2-3× E→I, Viriyopase et al.)
W_IE_STD = 1.5  # uS — I→E init std (pre-fan-in)
W_II_MEAN = 0.0  # uS — I→I recurrent init mean (0 by default: PING needs no I→I,
W_II_STD = 0.0  #  enable explicitly for Brunel/Vreeswijk balanced-network experiments)

# ── Training ──────────────────────────────────────────────────────────────
BATCH_SIZE = 64
# Surrogate-gradient steepness (fast-sigmoid slope). 5 is the stable
# end-to-end value at pinglab's current BPTT depth / clip / optimizer
# config — slope=10 (Neftci canonical) and slope=40 (Cramer) both blow
# up to gn 1e10+ even with the global-norm gradient clip at 100.
SURROGATE_SLOPE = 5.0
V_GRAD_DAMPEN = 80.0

# Derived
decay_ampa = np.exp(-dt / tau_ampa)
decay_gaba = np.exp(-dt / tau_gaba)
ref_steps_E = max(1, int(round(ref_ms_E / dt)))
ref_steps_I = max(1, int(round(ref_ms_I / dt)))
p_scale = max_rate_hz * dt / 1000.0
beta_snn = np.exp(-dt / tau_snn)
beta_out = np.exp(-dt / tau_out_ms)


def _env_no_compile() -> bool:
    """PINGLAB_NO_COMPILE=1 disables torch.compile in the model forward path.
    Set this for ablation runs that want the eager baseline (e.g. comparing
    compile vs eager wall time on the same hardware)."""
    import os

    return os.environ.get("PINGLAB_NO_COMPILE", "") == "1"


# Dynamo compile-cache size. Default 8 is exhausted by:
#   - train vs eval (requires_grad differs on state tensors per forward call)
#   - last partial batch in an epoch (smaller batch dim than 256)
#   - any incidental shape variation in compiled functions
# Bump to 32 so the cache holds a generous superset of distinct trace
# variants without falling back to eager. Unlike force_parameter_static_shapes
# (which we tried in 85cd304 and reverted: dynamic-shape codegen produced
# slower kernels on CUDA), this knob only affects how many specialised
# variants we keep around — each one is still a fully-static, shape-
# specialised compiled graph.
import torch._dynamo  # noqa: E402

torch._dynamo.config.recompile_limit = 32  # ty: ignore[invalid-assignment]


# ── Surrogate gradient ───────────────────────────────────────────────────


def fast_sigmoid_spike(u, slope):
    """Fast-sigmoid surrogate spike.

    Forward: Heaviside(u). Backward gradient: slope / (1 + slope·|u|)^2 —
    equivalent to the snntorch FastSigmoid surrogate that the library path
    uses, so slope=1 is a pure update-rule comparison against snntorch-library,
    not a surrogate comparison.

    Implementation is a detach-style straight-through estimator: the forward
    value is the hard step, but the gradient flows through the smooth proxy
    p(u) = slope·u / (1 + slope·|u|), whose derivative is exactly the
    fast-sigmoid kernel. Pure tensor ops — torch.compile-friendly with no
    custom autograd.Function graph break.
    """
    hard = (u >= 0).float()
    proxy = slope * u / (1.0 + slope * u.abs())
    # Parenthesise so (proxy - proxy.detach()) collapses to bitwise zero
    # in the forward pass; otherwise (hard + proxy) then - proxy.detach()
    # loses fp32 precision when |proxy| is close to 1 (value drifts to
    # 0.9999994), which poisons downstream int(s.item()) spike counts.
    return hard.detach() + (proxy - proxy.detach())


def spike_biophysical(v, threshold_offset=0.0):
    # mV-scale membrane: slope=1 keeps gradient support at the ~mV width of
    # typical threshold crossings. `threshold_offset` shifts the effective
    # threshold up — used by ALIF where each neuron's threshold rises with
    # its own recent firing.
    return fast_sigmoid_spike(v - V_th - threshold_offset, SURROGATE_SLOPE)


def _scale_grad(x, scale):
    """Return x unchanged in forward, but multiply gradient by scale in backward."""
    return x * scale + x.detach() * (1.0 - scale)


# ── Layer primitives ─────────────────────────────────────────────────────


def exp_synapse(g, spikes, W, decay):
    """Exponential synapse: decay, then add the undecayed spike kick.

    Canonical exponential synapse — a presynaptic spike makes g jump by its
    full weight W (its peak conductance), then decays as exp(-dt/tau). So W is
    the per-spike conductance increment, independent of dt. (Decay-then-add;
    cf. the older add-then-decay form, which scaled every kick by one extra
    factor of `decay`.)
    """
    return g * decay + spikes @ W


def lif_step(
    v,
    I_total,
    ref,
    C_m,
    g_L,
    ref_steps,
    spike_fn,
    V_floor=V_floor,
    V_max=None,
    v_grad_dampen=1.0,
):
    """One LIF timestep: voltage update, spike decision, then reset.
    Returns (v, s, ref)."""
    dv = (dt / C_m) * (-g_L * (v - E_L) + I_total)
    if v_grad_dampen != 1.0:
        dv = _scale_grad(dv, 1.0 / v_grad_dampen)
    v = v + dv
    v = v.clamp(min=V_floor) if V_max is None else v.clamp(min=V_floor, max=V_max)
    ref = (ref - 1).clamp(min=0)
    can_spike = ref == 0
    s = spike_fn(v) * can_spike.float()
    spiked_or_ref = s.bool() | (~can_spike)
    v = torch.where(spiked_or_ref, torch.full_like(v, V_reset), v)
    ref = torch.where(s.bool(), torch.full_like(ref, ref_steps), ref)
    return v, s, ref


def lif_step_expeuler(
    v,
    ref,
    g_e,
    g_i,
    C_m,
    g_L,
    ref_steps,
    spike_fn,
    v_grad_dampen=1.0,
    dt_override=None,
    V_floor=V_floor,
    V_max=None,
    threshold_offset=None,
    v_noise_std=0.0,
):
    """COBA LIF step under exponential Euler with a zero-order hold on g_e, g_i.

    Closed-form integration of
        C_m dv/dt = -g_L (v - E_L) - g_e (v - E_e) - g_i (v - E_i)
    over one step of length `dt`, holding g_e and g_i constant. Yields
        g_tot   = g_L + g_e + g_i
        tau_eff = C_m / g_tot
        v_inf   = (g_L*E_L + g_e*E_e + g_i*E_i) / g_tot
        v_{t+1} = v_inf + (v_t - v_inf) * exp(-dt / tau_eff)
    which is dt-invariant under N-vs-1 step in the passive case, unlike the
    forward-Euler `lif_step` above. Returns (v, s, ref).

    The kwarg is `dt_override` (not `dt`) so the module-level `dt` is
    accessible without `globals()['dt']`, which is a Dynamo graph-break.
    The graph-break was forcing a per-call recompile cascade that defeated
    torch.compile on COBANet's CUDA path (recompile_limit hit silently).
    """
    dt_step = dt if dt_override is None else dt_override
    if g_i is None:
        g_sum = g_e
        g_E_drive = g_e * E_e
    else:
        g_sum = g_e + g_i
        g_E_drive = g_e * E_e + g_i * E_i
    g_tot = g_L + g_sum
    v_inf = (g_L * E_L + g_E_drive) / g_tot
    decay = torch.exp(-dt_step / (C_m / g_tot))
    dv = (v_inf - v) * (1.0 - decay)
    if v_grad_dampen != 1.0:
        dv = _scale_grad(dv, 1.0 / v_grad_dampen)
    v = v + dv
    if v_noise_std > 0.0:
        # Diffusive membrane noise: zero-mean Wiener increment on v, injected
        # before the spike decision so it jitters threshold-crossing *timing*
        # (the quantity a gamma clock can resynchronise). Scaled by
        # sqrt(2*dt/tau_leak) so the stationary subthreshold std ≈ v_noise_std
        # (mV) in the passive limit, independent of dt — unlike the old
        # per-step g_E noise whose power scaled with the step count.
        tau_leak = C_m / g_L
        v = v + v_noise_std * (2.0 * dt_step / tau_leak) ** 0.5 * torch.randn_like(v)
    v = v.clamp(min=V_floor) if V_max is None else v.clamp(min=V_floor, max=V_max)
    ref = (ref - 1).clamp(min=0)
    can_spike = ref == 0
    if threshold_offset is None:
        s = spike_fn(v) * can_spike.float()
    else:
        s = spike_fn(v, threshold_offset) * can_spike.float()
    spiked_or_ref = s.bool() | (~can_spike)
    v = torch.where(spiked_or_ref, torch.full_like(v, V_reset), v)
    ref = torch.where(s.bool(), torch.full_like(ref, ref_steps), ref)
    return v, s, ref


def coba_current(g_e, v, g_i=None):
    """COBA synaptic current: g_e*(E_e - v) [+ g_i*(E_i - v)]."""
    I = g_e * (E_e - v)
    if g_i is not None:
        I = I + g_i * (E_i - v)
    return I


def init_lif_state(B, N, device, randomize=False, ref_mean=0.0, ref_std=0.0):
    """Initialise (v, ref) for a LIF population.
    If randomize=True, scatter initial voltages uniformly between E_L and V_th
    so neurons start at different phases (Börgers-style asynchronous init).
    ref_mean/ref_std: if nonzero, sample initial refractory from N(mean,std)
    clamped to [0, inf) so neurons come out of refractory at staggered times.
    """
    if randomize:
        v = E_L + (V_th - E_L) * torch.rand(B, N, device=device)
    else:
        v = torch.full((B, N), E_L, device=device)
    if ref_std > 0:
        ref = (
            (torch.randn(B, N, device=device) * ref_std + ref_mean).clamp(min=0).long()
        )
    else:
        ref = torch.zeros(B, N, device=device, dtype=torch.long)
    return v, ref


def init_conductance(B, N, device):
    """Initialise a conductance variable to zero."""
    return torch.zeros(B, N, device=device)


def _parse_weight_spec(w, default_dist, default_sparsity):
    """Parse a weight spec tuple: (p1, p2), (p1, p2, dist), or (p1, p2, dist, sparsity)."""
    if len(w) >= 4:
        return w[0], w[1], w[2], w[3]
    elif len(w) == 3:
        return w[0], w[1], w[2], default_sparsity
    return w[0], w[1], default_dist, default_sparsity


# Connectivity mode for the sparsifier in init_weight:
#   False (default) — per-entry Bernoulli: each entry zeroed independently
#     with probability `sparsity`. Fan-in per post cell is binomial, so it
#     varies cell to cell.
#   True            — fixed fan-in (exact-K): every post cell (column) keeps
#     exactly K = round((1-sparsity)·N_pre) random presynaptic inputs.
#     Removes the binomial fan-in variance — the Brunel/Vreeswijk convention.
# Set via the --exact-k CLI flag (M.EXACT_K_CONNECTIVITY = True).
EXACT_K_CONNECTIVITY = False


def init_weight(shape, dist="normal", p1=0.0, p2=0.1, sparsity=0.0):
    """Initialise a weight tensor with fan-in normalization."""
    n_pre = shape[0]
    if dist == "signed_normal":
        w = torch.randn(*shape).mul_(p2).add_(p1)
    elif dist == "normal":
        w = torch.randn(*shape).mul_(p2).add_(p1).clamp_(min=0)
    elif dist == "uniform":
        w = torch.rand(*shape).mul_(p2 - p1).add_(p1)
    elif dist == "constant":
        w = torch.full(shape, p1)
    elif dist == "zeros":
        w = torch.zeros(*shape)
    else:
        raise ValueError(f"Unknown dist: {dist!r}")
    if sparsity > 0:
        if EXACT_K_CONNECTIVITY and len(shape) == 2:
            # Fixed fan-in: each column keeps exactly K random rows.
            n_post = shape[1]
            k = max(1, int(round((1.0 - sparsity) * n_pre)))
            mask = torch.zeros(n_pre, n_post)
            for j in range(n_post):
                idx = torch.randperm(n_pre)[:k]
                mask[idx, j] = 1.0
            w = w * mask
            # Rescale by exact fan-in so per-column expected drive is
            # preserved (matches the Bernoulli path's 1/(1-sparsity)).
            w = w * (n_pre / k)
        else:
            w = w * (torch.rand(*shape) > sparsity).float()
            w = w / (1.0 - sparsity)
    w = w / n_pre
    return w


# ── E-step and I-step composites ─────────────────────────────────────────

COBA_INTEGRATOR = "expeuler"  # "expeuler" | "fwd"  — parity toggle for COBA integration


def e_step_coba(v, ref, g_e, g_i=None, ref_steps=None, threshold_offset=None,
                v_noise_std=0.0):
    """One E-neuron LIF step with COBA driving force."""
    if ref_steps is None:
        ref_steps = ref_steps_E
    if COBA_INTEGRATOR == "expeuler":
        return lif_step_expeuler(
            v,
            ref,
            g_e,
            g_i,
            C_m_E,
            g_L_E,
            ref_steps,
            spike_biophysical,
            v_grad_dampen=V_GRAD_DAMPEN,
            threshold_offset=threshold_offset,
            v_noise_std=v_noise_std,
        )
    return lif_step(
        v,
        coba_current(g_e, v, g_i),
        ref,
        C_m_E,
        g_L_E,
        ref_steps,
        spike_biophysical,
        v_grad_dampen=V_GRAD_DAMPEN,
    )


def i_step_coba(v, ref, g_e, g_i=None, threshold_offset=None, v_noise_std=0.0):
    """One I-neuron LIF step with COBA driving force.

    ``g_i`` is the I→I inhibitory conductance on the I cell, used for
    Brunel/Vreeswijk-style balanced-network experiments where I-cells have
    recurrent self-inhibition. Default ``None`` preserves the canonical PING
    architecture (no I→I)."""
    if COBA_INTEGRATOR == "expeuler":
        return lif_step_expeuler(
            v,
            ref,
            g_e,
            g_i,
            C_m_I,
            g_L_I,
            ref_steps_I,
            spike_biophysical,
            v_grad_dampen=V_GRAD_DAMPEN,
            threshold_offset=threshold_offset,
            v_noise_std=v_noise_std,
        )
    return lif_step(
        v,
        coba_current(g_e, v, g_i),
        ref,
        C_m_I,
        g_L_I,
        ref_steps_I,
        spike_biophysical,
        v_grad_dampen=V_GRAD_DAMPEN,
    )


# ── Base class ───────────────────────────────────────────────────────────


class SNNBase(nn.Module):
    recording = False

    def _set_meta(self, B, n_spk, rec, sizes):
        t_sec = T_ms / 1000.0
        self.rates = {k: v / (B * sizes[k] * t_sec) for k, v in n_spk.items()}
        if rec is not None:
            # Accept either pre-stacked tensors or lists of per-timestep tensors
            self.spike_record = {
                k: (v if isinstance(v, torch.Tensor) else torch.stack(v))
                for k, v in rec.items()
            }

    def project_dales(self) -> None:
        """Project trainable weights back into the Dale's-law cone.

        Override in subclasses that enforce signed weights at forward
        time via clamp(min=0). After every optimiser step, calling
        this method makes the *stored* parameter values match what the
        network actually uses — so a downstream consumer reading
        weights.pth sees the constrained weights, not the raw
        pre-clamp ones with negative entries that the optimiser drove
        toward but the forward pass discarded. No-op by default.
        """
        pass


# ── Model classes ────────────────────────────────────────────────────────



class COBANet(SNNBase):
    signed_weights = False

    def __init__(
        self,
        w_in=(W_IN_MEAN, W_IN_STD),
        w_hid=(W_HID_MEAN, W_HID_STD),
        w_ee=(W_EE_MEAN, W_EE_STD),
        w_ei=(W_EI_MEAN, W_EI_STD),
        w_ie=(W_IE_MEAN, W_IE_STD),
        w_ii=(W_II_MEAN, W_II_STD),
        dist="normal",
        sparsity=0.0,
        dales_law=True,
        hidden_sizes=None,
        ei_layers=None,
        readout_mode="rate",
        trainable_w_ei=False,
        trainable_w_ie=False,
        trainable_w_ii=False,
        n_inh_per_layer=None,
    ):
        super().__init__()
        if readout_mode not in ("rate", "li", "spike-count", "mem-mean"):
            raise ValueError(
                f"readout_mode must be 'rate', 'li', 'spike-count', or "
                f"'mem-mean', got {readout_mode!r}"
            )
        self.readout_mode = readout_mode
        self.signed_weights = not dales_law
        # Lazy-compile cache for the per-timestep body (compiled on first call).
        self._compiled_cache: dict = {}
        # Optional per-step hook fired right after every layer's spikes are
        # emitted and before they propagate (readout + next-step recurrence +
        # I-loop). Signature: (s_e, s_i_or_None, layer_idx) -> (s_e', s_i').
        # Set on the live instance to do hidden-spike perturbation experiments
        # without modifying the forward graph. Forces the eager step body.
        self._hidden_perturb_fn = None

        sizes = hidden_sizes if hidden_sizes is not None else HIDDEN_SIZES
        self.hidden_sizes = list(sizes)
        self.n_layers = len(sizes)
        all_sizes = [N_IN] + list(sizes) + [N_OUT]
        self.all_sizes = all_sizes

        # Which layers get E-I structure (1-indexed; default: all)
        if ei_layers is not None:
            self.ei_layers = set(ei_layers)
        else:
            self.ei_layers = set(range(1, self.n_layers + 1))

        # Per-layer N_I override (1-indexed). When None, falls back to
        # n_e // 4. Used by the I-pool sweep in nb047 to vary the E:I
        # ratio without retraining the rest of the architecture.
        self.n_inh_per_layer = dict(n_inh_per_layer or {})

        # Feedforward weights: input→H1, H1→H2, ..., HN→output
        self.W_ff = nn.ParameterList()
        for idx, (n_pre, n_post) in enumerate(zip(all_sizes[:-1], all_sizes[1:])):
            spec = w_in if idx == 0 else w_hid
            p1, p2, d, s = _parse_weight_spec(spec, dist, sparsity)
            self.W_ff.append(nn.Parameter(init_weight((n_pre, n_post), d, p1, p2, s)))

        # E-I weights per E-I layer. W_ee / W_ei / W_ie are fixed anatomical
        # connectivity (the recurrent inhibitory circuit is a substrate the
        # readout learns to read, not a trainable thing).
        self.W_ee = nn.ParameterDict()
        self.W_ei = nn.ParameterDict()
        self.W_ie = nn.ParameterDict()
        # W_ii is the I→I recurrent matrix (added for Brunel/Vreeswijk balanced-
        # network experiments). Zero-mean / zero-std by default so the canonical
        # PING architecture is unchanged; set w_ii to non-zero to enable.
        self.W_ii = nn.ParameterDict()
        for i in self.ei_layers:
            n_e = sizes[i - 1]
            n_i = self.n_inh_per_layer.get(i, n_e // 4)
            k = str(i)
            p1, p2, d, s = _parse_weight_spec(w_ee, dist, sparsity)
            w_ee_t = nn.Parameter(
                init_weight((n_e, n_e), d, p1, p2, s),
                requires_grad=False,
            )
            p1, p2, d, s = _parse_weight_spec(w_ei, dist, sparsity)
            w_ei_t = nn.Parameter(
                init_weight((n_e, n_i), d, p1, p2, s),
                requires_grad=trainable_w_ei,
            )
            p1, p2, d, s = _parse_weight_spec(w_ie, dist, sparsity)
            w_ie_t = nn.Parameter(
                init_weight((n_i, n_e), d, p1, p2, s),
                requires_grad=trainable_w_ie,
            )
            p1, p2, d, s = _parse_weight_spec(w_ii, dist, sparsity)
            w_ii_t = nn.Parameter(
                init_weight((n_i, n_i), d, p1, p2, s),
                requires_grad=trainable_w_ii,
            )
            self.W_ee[k] = w_ee_t
            self.W_ei[k] = w_ei_t
            self.W_ie[k] = w_ie_t
            self.W_ii[k] = w_ii_t

    def _hid_key(self, layer_idx):
        if self.n_layers == 1:
            return "hid"
        return f"hid_{layer_idx}"

    def _inh_key(self, layer_idx):
        if self.n_layers == 1:
            return "inh"
        return f"inh_{layer_idx}"

    def forward(
        self,
        noise_std=0.0,
        randomize_init=False,
        ref_mean=0.0,
        ref_std=0.0,
        ext_g=None,
        ext_g_i=None,
        drive_sigma=0.0,
        input_spikes=None,
        v_perturb_eps=0.0,
        v_perturb_seed=0,
        noise_on_inh=True,
    ):
        has_ext_g = ext_g is not None
        has_ext_g_i = ext_g_i is not None
        has_input_spikes = input_spikes is not None

        if has_ext_g and ext_g.dim() == 3:
            B, device = ext_g.shape[1], ext_g.device
        elif has_input_spikes and input_spikes.dim() == 3:
            B, device = input_spikes.shape[1], input_spikes.device
        else:
            B, device = (
                1,
                (
                    ext_g.device
                    if has_ext_g
                    else input_spikes.device
                    if has_input_spikes
                    else torch.device("cpu")
                ),
            )

        if self.signed_weights:
            W_ff = list(self.W_ff)
        else:
            W_ff = [W.clamp(min=0) for W in self.W_ff]
        # `noise_std` is the diffusive membrane-noise amplitude (mV, on the
        # voltage), applied per-cell-independent to every E and I cell inside
        # the LIF step (see lif_step_expeuler). Zero-mean and dt-invariant —
        # replaces the old rectified, first-layer-E-only g_E conductance noise.
        v_noise_std = float(noise_std) if noise_std > 0 else 0.0

        # Per-step matmul on every device — single code path (no CUDA-only
        # fast-path).

        # Per-layer state
        v_e, ref_e, ge_e, gi_e, s_e = {}, {}, {}, {}, {}
        v_i, ref_i, ge_i, gi_i, s_i = {}, {}, {}, {}, {}
        drive_gains = {}
        for i in range(1, self.n_layers + 1):
            n_e = self.hidden_sizes[i - 1]
            k = str(i)
            v_e[k], ref_e[k] = init_lif_state(
                B,
                n_e,
                device,
                randomize=randomize_init,
                ref_mean=ref_mean,
                ref_std=ref_std,
            )
            ge_e[k] = init_conductance(B, n_e, device)
            s_e[k] = torch.zeros(B, n_e, device=device)
            if i in self.ei_layers:
                n_i = self.n_inh_per_layer.get(i, n_e // 4)
                gi_e[k] = init_conductance(B, n_e, device)
                v_i[k], ref_i[k] = init_lif_state(
                    B, n_i, device, randomize=randomize_init
                )
                ge_i[k] = init_conductance(B, n_i, device)
                gi_i[k] = init_conductance(B, n_i, device)
                s_i[k] = torch.zeros(B, n_i, device=device)
            if drive_sigma > 0 and i == 1:
                drive_gains[k] = (
                    1.0 + drive_sigma * torch.randn(B, n_e, device=device)
                ).clamp(min=0)

        # Lyapunov perturbation: add a fixed-norm random offset to every
        # membrane voltage at t=0 so a second forward pass on identical
        # input diverges only by the chaos of the dynamics. Seeded so the
        # perturbation is reproducible across the clean/perturbed pair.
        if v_perturb_eps > 0:
            pgen = torch.Generator(device="cpu").manual_seed(int(v_perturb_seed))
            for i in range(1, self.n_layers + 1):
                k = str(i)
                dv = (
                    torch.randn(
                        v_e[k].shape, generator=pgen
                    ).to(device) * v_perturb_eps
                )
                v_e[k] = v_e[k] + dv
                if k in v_i:
                    dvi = (
                        torch.randn(
                            v_i[k].shape, generator=pgen
                        ).to(device) * v_perturb_eps
                    )
                    v_i[k] = v_i[k] + dvi

        # Output: cumulative last-hidden-layer spikes → linear decoder
        # (no output spiking neurons — a clean linear decode at the output).
        hidden_accum = init_conductance(B, self.hidden_sizes[-1], device)
        v_out = torch.zeros(B, N_OUT, device=device)
        logits_max = torch.full((B, N_OUT), float("-inf"), device=device)
        s_count = torch.zeros(B, N_OUT, device=device)
        mem_sum = torch.zeros(B, N_OUT, device=device)

        # Pre-allocate recording buffers on GPU
        rec_buf = None
        if self.recording:
            rec_buf = {"out": torch.zeros(T_steps, B, N_OUT, device=device)}
            if has_input_spikes:
                rec_buf["input"] = torch.zeros(T_steps, B, N_IN, device=device)
            for i in range(1, self.n_layers + 1):
                n_e = self.hidden_sizes[i - 1]
                rec_buf[self._hid_key(i)] = torch.zeros(T_steps, B, n_e, device=device)
                # Extra trace buffers: membrane voltage and conductances for
                # E (and I, where present). Lets downstream image-mode dump
                # per-neuron v/g traces alongside spikes.
                rec_buf[f"v_e_{i}"] = torch.zeros(T_steps, B, n_e, device=device)
                rec_buf[f"ge_e_{i}"] = torch.zeros(T_steps, B, n_e, device=device)
                if i in self.ei_layers:
                    n_inh = self.n_inh_per_layer.get(i, n_e // 4)
                    rec_buf[self._inh_key(i)] = torch.zeros(
                        T_steps, B, n_inh, device=device
                    )
                    rec_buf[f"gi_e_{i}"] = torch.zeros(T_steps, B, n_e, device=device)
                    rec_buf[f"v_i_{i}"] = torch.zeros(T_steps, B, n_inh, device=device)
                    rec_buf[f"ge_i_{i}"] = torch.zeros(T_steps, B, n_inh, device=device)
                    rec_buf[f"gi_i_{i}"] = torch.zeros(T_steps, B, n_inh, device=device)
        # GPU-side spike accumulators
        n_spk_tensors = {}
        for i in range(1, self.n_layers + 1):
            n_spk_tensors[self._hid_key(i)] = torch.zeros(1, device=device)
            if i in self.ei_layers:
                n_spk_tensors[self._inh_key(i)] = torch.zeros(1, device=device)
        n_spk_tensors["out"] = torch.zeros(1, device=device)
        # Per-layer (B, n_e) spike-count accumulator for the firing-rate
        # regulariser — must keep gradient attached, so it sums state["s_e"]
        # post-step.
        rate_counts = [
            torch.zeros(B, n, device=device) for n in self.hidden_sizes
        ]

        # Bundle mutating state and per-call config so _step_body can be
        # compiled per-timestep. The Python
        # int `t` and rec_buf writes stay in _forward_loop so the compiled
        # graph never has to re-trace per-t.
        state = {
            "v_e": v_e,
            "ref_e": ref_e,
            "ge_e": ge_e,
            "gi_e": gi_e,
            "s_e": s_e,
            "v_i": v_i,
            "ref_i": ref_i,
            "ge_i": ge_i,
            "gi_i": gi_i,
            "s_i": s_i,
            "hidden_accum": hidden_accum,
            "v_out": v_out,
            "logits_max": logits_max,
            "s_count": s_count,
            "mem_sum": mem_sum,
        }
        cfg = {
            "B": B,
            "device": device,
            "W_ff": W_ff,
            "drive_gains": drive_gains,
            "ei_layers": self.ei_layers,
            "has_input_spikes": has_input_spikes,
            "has_ext_g": has_ext_g,
            "has_ext_g_i": has_ext_g_i,
            "readout_mode": self.readout_mode,
            "v_noise_std": v_noise_std,
            "noise_on_inh": bool(noise_on_inh),
            "n_e0": self.hidden_sizes[0],
            "n_spk_tensors": n_spk_tensors,
        }

        # Lazy-init torch.compile on the per-timestep body, with a CPU-skip
        # and a PINGLAB_NO_COMPILE escape hatch. CPU is skipped because
        # Inductor's cpp build fails on
        # some hosts and the error escapes the try/except (surfaces only
        # at first compiled call, not at torch.compile() construction).
        if (
            "step" not in self._compiled_cache
            and not _env_no_compile()
            and device.type != "cpu"
            and self._hidden_perturb_fn is None
        ):
            try:
                self._compiled_cache["step"] = torch.compile(
                    self._step_body, dynamic=False
                )
            except Exception as exc:  # noqa: BLE001
                self._compiled_cache["step"] = self._step_body
                self._compiled_cache["compile_error"] = str(exc)
        # When the perturb hook is set, always use the eager body so the
        # Python callable doesn't break (or trigger recompile on) the graph.
        step = (
            self._step_body
            if self._hidden_perturb_fn is not None
            else self._compiled_cache.get("step", self._step_body)
        )

        for t in range(T_steps):
            slc = {
                "in_t": (
                    input_spikes[t].unsqueeze(0)
                    if has_input_spikes and input_spikes.dim() == 2
                    else (input_spikes[t] if has_input_spikes else None)
                ),
                "ext_t": (
                    ext_g[t].unsqueeze(0)
                    if has_ext_g and ext_g.dim() == 2
                    else (ext_g[t] if has_ext_g else None)
                ),
                "ext_t_i": (
                    ext_g_i[t].unsqueeze(0)
                    if has_ext_g_i and ext_g_i.dim() == 2
                    else (ext_g_i[t] if has_ext_g_i else None)
                ),
            }
            logits_t = step(slc, cfg, state)
            # Accumulate per-neuron E spike counts for fr-reg (grad-attached).
            for i in range(1, self.n_layers + 1):
                rate_counts[i - 1] = rate_counts[i - 1] + state["s_e"][str(i)]
            if rec_buf is not None:
                if slc["in_t"] is not None and "input" in rec_buf:
                    rec_buf["input"][t] = slc["in_t"]
                for i in range(1, self.n_layers + 1):
                    k = str(i)
                    rec_buf[self._hid_key(i)][t] = state["s_e"][k]
                    rec_buf[f"v_e_{i}"][t] = state["v_e"][k]
                    rec_buf[f"ge_e_{i}"][t] = state["ge_e"][k]
                    if i in self.ei_layers:
                        rec_buf[self._inh_key(i)][t] = state["s_i"][k]
                        rec_buf[f"gi_e_{i}"][t] = state["gi_e"][k]
                        rec_buf[f"v_i_{i}"][t] = state["v_i"][k]
                        rec_buf[f"ge_i_{i}"][t] = state["ge_i"][k]
                        rec_buf[f"gi_i_{i}"][t] = state["gi_i"][k]
                rec_buf["out"][t] = logits_t

        sizes = {}
        for i in range(1, self.n_layers + 1):
            sizes[self._hid_key(i)] = self.hidden_sizes[i - 1]
            if i in self.ei_layers:
                sizes[self._inh_key(i)] = self.hidden_sizes[i - 1] // 4
        sizes["out"] = N_OUT
        n_spk = {k: v.item() for k, v in n_spk_tensors.items()}
        rec = None
        if rec_buf is not None:
            rec = {
                k: (v.squeeze(1).cpu() if B == 1 else v.cpu())
                for k, v in rec_buf.items()
            }
        self._set_meta(B, n_spk, rec, sizes)
        # Expose grad-attached per-neuron spike counts so the trainer's
        # firing-rate regulariser (train.py) can build its loss.
        self.last_spike_counts = rate_counts
        if self.readout_mode == "li":
            return state["logits_max"]
        if self.readout_mode == "spike-count":
            return state["s_count"]
        if self.readout_mode == "mem-mean":
            return state["mem_sum"] / float(T_steps)
        return logits_t

    def _step_body(self, slc, cfg, state):
        """One timestep: all PING layers + readout. The compile target.

        slc:    per-t inputs (in_t, ext_t) sliced in _forward_loop.
        cfg:    per-call constants (W_ff, drive_gains, ei_layers, flags,
                spike accumulators).
        state:  per-call mutable dicts (membrane voltages, conductances,
                spike outputs, readout accumulators). Mutated in place.

        Returns logits_t for the rec_buf write in _forward_loop. Per-t
        rec_buf writes and Python int `t` are kept out so the compiled
        graph reuses across all T_steps invocations.
        """
        W_ff = cfg["W_ff"]
        ei_layers = cfg["ei_layers"]
        drive_gains = cfg["drive_gains"]
        has_input_spikes = cfg["has_input_spikes"]
        has_ext_g = cfg["has_ext_g"]
        n_spk_tensors = cfg["n_spk_tensors"]

        prev_spk = None
        for i in range(1, self.n_layers + 1):
            k = str(i)
            W = W_ff[i - 1]
            is_ei = i in ei_layers

            if is_ei:
                # exp_synapse inlined: when called as a separate function
                # Dynamo compiles it as its own trace boundary and hits the
                # recompile limit (3 different W shapes × inner-loop call
                # sites > recompile_limit=8), silently falling back to
                # eager. Inlining keeps everything inside _step_body's
                # single compiled graph where the three weight shapes are
                # specialized once per (W_ee, W_ei, W_ie) tuple.
                ee_drive = state["s_e"][k] @ self.W_ee[k]
                state["ge_e"][k] = state["ge_e"][k] * decay_ampa + ee_drive
                ei_drive = state["s_e"][k] @ self.W_ei[k]
                if k == "1" and cfg["has_ext_g_i"]:
                    ei_drive = ei_drive + slc["ext_t_i"]
                state["ge_i"][k] = state["ge_i"][k] * decay_ampa + ei_drive
                state["gi_e"][k] = (
                    state["gi_e"][k] * decay_gaba + state["s_i"][k] @ self.W_ie[k]
                )
                state["gi_i"][k] = (
                    state["gi_i"][k] * decay_gaba + state["s_i"][k] @ self.W_ii[k]
                )
            else:
                state["ge_e"][k] = state["ge_e"][k] * decay_ampa

            if i == 1:
                if has_input_spikes:
                    g_ext = slc["in_t"] @ W
                    if k in drive_gains:
                        g_ext = g_ext * drive_gains[k]
                    state["ge_e"][k] = state["ge_e"][k] + g_ext
                if has_ext_g:
                    state["ge_e"][k] = state["ge_e"][k] + slc["ext_t"]
            else:
                ff_drive = prev_spk @ W
                state["ge_e"][k] = state["ge_e"][k] + ff_drive

            g_e_for_step = state["ge_e"][k]
            g_i_for_e = state["gi_e"][k] if is_ei else None
            g_e_for_i = state["ge_i"][k] if is_ei else None
            g_i_for_i = state["gi_i"][k] if is_ei else None
            v_noise = cfg["v_noise_std"]
            v_noise_i = v_noise if cfg["noise_on_inh"] else 0.0
            if is_ei:
                state["v_e"][k], state["s_e"][k], state["ref_e"][k] = e_step_coba(
                    state["v_e"][k],
                    state["ref_e"][k],
                    g_e_for_step,
                    g_i_for_e,
                    v_noise_std=v_noise,
                )
                state["v_i"][k], state["s_i"][k], state["ref_i"][k] = i_step_coba(
                    state["v_i"][k], state["ref_i"][k], g_e_for_i, g_i_for_i,
                    v_noise_std=v_noise_i,
                )
            else:
                state["v_e"][k], state["s_e"][k], state["ref_e"][k] = e_step_coba(
                    state["v_e"][k], state["ref_e"][k], g_e_for_step,
                    v_noise_std=v_noise,
                )

            if self._hidden_perturb_fn is not None:
                new_s_e, new_s_i = self._hidden_perturb_fn(
                    state["s_e"][k],
                    state["s_i"].get(k) if is_ei else None,
                    i,
                )
                state["s_e"][k] = new_s_e
                if is_ei and new_s_i is not None:
                    state["s_i"][k] = new_s_i

            prev_spk = state["s_e"][k]

            hk = self._hid_key(i)
            n_spk_tensors[hk] += state["s_e"][k].detach().sum()
            if is_ei:
                ik = self._inh_key(i)
                n_spk_tensors[ik] += state["s_i"][k].detach().sum()

        if cfg["readout_mode"] == "li":
            I_out = prev_spk @ W_ff[-1]
            state["v_out"] = beta_snn * state["v_out"] + (1.0 - beta_snn) * I_out
            state["logits_max"] = torch.maximum(state["logits_max"], state["v_out"])
            return state["v_out"]
        if cfg["readout_mode"] in ("spike-count", "mem-mean"):
            # Exp-Euler ZOH on output LIF + subtract reset. COBANet's W_ff has
            # no bias term — bias scaling is moot here.
            one_minus_beta = 1.0 - beta_out
            spike_scale = one_minus_beta / dt
            I_out = spike_scale * (prev_spk @ W_ff[-1])
            state["v_out"] = beta_out * state["v_out"] + I_out
            s_out = fast_sigmoid_spike(state["v_out"] - thr_snn, SURROGATE_SLOPE)
            state["s_count"] = state["s_count"] + s_out
            state["mem_sum"] = state["mem_sum"] + state["v_out"]
            state["v_out"] = state["v_out"] - s_out * thr_snn
            if cfg["readout_mode"] == "mem-mean":
                return state["mem_sum"] / float(T_steps)
            return state["s_count"]
        state["hidden_accum"] = state["hidden_accum"] + prev_spk
        return state["hidden_accum"] @ W_ff[-1]

    def _dale_params(self):
        """Trainable weights subject to Dale's law: the feedforward stack plus
        any recurrent E↔I matrices flipped to trainable (--trainable-w-*).
        Non-trainable recurrent buffers are skipped — they are init'd
        non-negative and never updated."""
        dicts = (self.W_ee, self.W_ei, self.W_ie, self.W_ii)
        params = list(self.W_ff) + [p for d in dicts for p in d.values()]
        return [p for p in params if p.requires_grad]

    @torch.no_grad()
    def project_dales(self) -> None:
        """Projected-gradient step for Dale's law: clamp every trainable
        constrained weight back onto the non-negative orthant. Registered as an
        optimiser step-post-hook (train.py), so it runs automatically after each
        opt.step(); no-op when signed weights are allowed. Eager and outside the
        compiled per-timestep graph, so it does not affect torch.compile."""
        if self.signed_weights:
            return
        for p in self._dale_params():
            p.clamp_(min=0)
