"""SNN model definitions and layer primitives.

All constants are hardcoded defaults. Override via module-level assignment
or by passing arguments to model constructors.

Models: CUBANet, COBANet.
Layer primitives: exp_synapse, lif_step, snn_lif_step.
"""

from __future__ import annotations

import math

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
# Slow excitatory ("NMDA-like") decay. Used only when COBANet is built
# with slow_synapse=True. ~100 ms matches the Wang / Compte working-
# memory NMDA timescale.
tau_nmda = 100.0  # ms — slow excitatory decay (off by default)
# Adaptation timescale for the optional ALIF neuron. ~700 ms is the
# LSNN paper default; tuned to bridge the working-memory delays the
# slow-NMDA channel struggles with.
tau_adapt = 700.0  # ms — slow adaptation decay (off by default)

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
W_STD_CUBA = 32.0  # nA — CUBA weight init std (pre-fan-in)
W_STD_COBA = 6.4  # uS — COBA weight init std (pre-fan-in)
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

# ── Training ──────────────────────────────────────────────────────────────
BATCH_SIZE = 64
GRAD_CLIP = 1.0
# Surrogate-gradient steepness (fast-sigmoid slope). 5 is the stable
# end-to-end value at pinglab's current BPTT depth / clip / optimizer
# config — slope=10 (Neftci canonical) and slope=40 (Cramer) both blow
# up to gn 1e10+ even with --optimizer adamax + --grad-clip 100.
SURROGATE_SLOPE = 5.0
READOUT_SCALE = 0.0
PATIENCE = 15
V_GRAD_DAMPEN = 80.0

# Derived
decay_ampa = np.exp(-dt / tau_ampa)
decay_gaba = np.exp(-dt / tau_gaba)
decay_nmda = np.exp(-dt / tau_nmda)
decay_adapt = np.exp(-dt / tau_adapt)
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


def spike_snn(v):
    # Dimensionless membrane (threshold=1). slope=1 (pinglab default; not
    # snntorch's 25) gives a wide active window so silent neurons keep gradient
    # support. Override via SURROGATE_SLOPE (oscilloscope --surrogate-slope) to
    # match a specific reference (e.g. Cramer β=40).
    return fast_sigmoid_spike(v - thr_snn, SURROGATE_SLOPE)


def _scale_grad(x, scale):
    """Return x unchanged in forward, but multiply gradient by scale in backward."""
    return x * scale + x.detach() * (1.0 - scale)


# ── Layer primitives ─────────────────────────────────────────────────────


def exp_synapse(g, spikes, W, decay):
    """Exponential synapse: spike kicks first, then decay."""
    return (g + spikes @ W) * decay


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


def snn_lif_step(mem, I, beta, spike_fn, reset="zero", can_fire=None):
    """snnTorch-style LIF step: decay + input, spike, reset. Returns (mem, s).

    Caller is responsible for dt-scaling of the input. Same primitive for
    standard-snn and cuba; what differs is how I is constructed.

    can_fire: optional bool mask. Where False, neuron is refractory — mem is
              clamped to V_reset=0 (no integration), spike output is 0.

    reset: "zero" — hard reset to 0 on spike (overshoot discarded)
           "subtract" — subtract threshold on spike (preserves overshoot)
    """
    mem = beta * mem + I
    if can_fire is not None:
        # Refractory: clamp mem to V_reset so spike_fn returns ~0.
        mem = torch.where(can_fire, mem, torch.zeros_like(mem))
    s = spike_fn(mem)
    if reset == "subtract":
        mem = mem - thr_snn * s
    else:
        mem = torch.where(s.bool(), 0.0, mem)
    return mem, s


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


# ── E-step and I-step composites ─────────────────────────────────────────

COBA_INTEGRATOR = "expeuler"  # "expeuler" | "fwd"  — parity toggle for COBA integration


def e_step_coba(v, ref, g_e, g_i=None, ref_steps=None, threshold_offset=None):
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


def i_step_coba(v, ref, g_e, threshold_offset=None):
    """One I-neuron LIF step with COBA driving force."""
    if COBA_INTEGRATOR == "expeuler":
        return lif_step_expeuler(
            v,
            ref,
            g_e,
            None,
            C_m_I,
            g_L_I,
            ref_steps_I,
            spike_biophysical,
            v_grad_dampen=V_GRAD_DAMPEN,
            threshold_offset=threshold_offset,
        )
    return lif_step(
        v,
        coba_current(g_e, v),
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


class CUBANet(SNNBase):
    # Biophysical defaults (Dale's law + hard reset + membrane readout).
    # Tutorial mode overrides every one of these.
    signed_weights = False
    beta_override = None
    reset_mode = "zero"
    randomize_init = False

    def __init__(
        self,
        discretisation="snntorch",
        w_in=None,
        w_hid=None,
        w_rec=None,
        dist="normal",
        sparsity=0.0,
        tutorial_mode=False,
        dales_law=True,
        hidden_sizes=None,
        rec_layers=None,
        exponential_synapse=False,
        ref_ms=0.0,
        reset_mode=None,
        readout_mode="rate",
    ):
        super().__init__()
        if discretisation not in ("snntorch", "zoh"):
            raise ValueError(
                f"discretisation must be 'snntorch' or 'zoh', got {discretisation!r}"
            )
        if readout_mode not in ("rate", "li", "spike-count", "mem-mean"):
            raise ValueError(
                f"readout_mode must be 'rate', 'li', 'spike-count', or "
                f"'mem-mean', got {readout_mode!r}"
            )
        self.discretisation = discretisation
        self.exponential_synapse = exponential_synapse
        self.ref_ms = ref_ms
        self._reset_mode_override = reset_mode
        self.readout_mode = readout_mode
        self.signed_weights = not dales_law
        # Lazy-compiled per-timestep step body. Device-agnostic — tried on
        # MPS, CUDA, CPU uniformly (no `if device.type == "cuda"` guard).
        # Filled on first forward; reused across all subsequent timesteps.
        self._compiled_cache: dict = {}

        # Layer sizes: [N_IN, H1, H2, ..., HN, N_OUT]
        sizes = hidden_sizes if hidden_sizes is not None else HIDDEN_SIZES
        self.hidden_sizes = list(sizes)
        self.n_layers = len(sizes)
        all_sizes = [N_IN] + list(sizes) + [N_OUT]
        self.all_sizes = all_sizes

        # Which layers get recurrence (1-indexed; default: all if w_rec set)
        if rec_layers is not None:
            self.rec_layers = set(rec_layers)
        elif w_rec is not None:
            self.rec_layers = set(range(1, self.n_layers + 1))
        else:
            self.rec_layers = set()

        if tutorial_mode:
            self.signed_weights = True
            # β falls through to module-level beta_snn (dt-aware, τ=tau_snn).
            # The snnTorch tutorial's hardcoded 0.95 assumes dt=1ms; using it
            # at dt≠1 gives an inconsistent τ_mem vs SNNTorchLibraryNet which
            # also uses beta_snn. Leaving beta_override=None keeps both paths
            # calibrated to the same τ_mem at every dt.
            self.reset_mode = self._reset_mode_override or "subtract"
            self._init_tutorial_weights(all_sizes, w_rec)
        else:
            self._init_biophysical_weights(
                all_sizes, w_in, w_hid, w_rec, dist, sparsity
            )
            if self._reset_mode_override is not None:
                self.reset_mode = self._reset_mode_override

    def _init_tutorial_weights(self, all_sizes, w_rec=None):
        # Match torch.nn.Linear.reset_parameters exactly so the same seed
        # gives bit-identical weights here and in SNNTorchLibraryNet (which
        # uses nn.Linear). nn.Linear stores weight as (n_post, n_pre) and
        # calls kaiming_uniform_(weight, a=sqrt(5)) — equivalent to
        # uniform(±1/sqrt(fan_in)) with fan_in=n_pre — then uniform_(bias,
        # ±1/sqrt(fan_in)). Our storage convention is (n_pre, n_post), so
        # we allocate a matching (n_post, n_pre) scratch tensor, run the
        # same inits in the same order, and transpose. Without this the
        # two paths draw the same *count* of samples but scatter them to
        # different weight positions, and the trained firing rates drift
        # apart even though accuracy calibrates.
        self.W_ff = nn.ParameterList()
        self.b_ff = nn.ParameterList()
        for n_pre, n_post in zip(all_sizes[:-1], all_sizes[1:]):
            W_t = torch.empty(n_post, n_pre)
            b = torch.empty(n_post)
            nn.init.kaiming_uniform_(W_t, a=math.sqrt(5))
            bound = 1.0 / math.sqrt(n_pre) if n_pre > 0 else 0
            nn.init.uniform_(b, -bound, bound)
            self.W_ff.append(nn.Parameter(W_t.t().contiguous()))
            self.b_ff.append(nn.Parameter(b))
        # Recurrent weights for selected layers
        self.W_rec = nn.ParameterDict()
        for i in self.rec_layers:
            n = all_sizes[i]  # hidden layer i size
            if w_rec is None:
                w_rec_spec = (0, 0.1)
            else:
                w_rec_spec = w_rec
            p1, p2, _, s = _parse_weight_spec(w_rec_spec, "signed_normal", 0.0)
            self.W_rec[str(i)] = nn.Parameter(
                init_weight((n, n), "signed_normal", p1, p2, s)
            )

    def _init_biophysical_weights(self, all_sizes, w_in, w_hid, w_rec, dist, sparsity):
        default_std = W_STD_CUBA
        if w_in is None:
            w_in = (0, default_std)
        if w_hid is None:
            w_hid = (0, default_std)
        if self.signed_weights:
            dist = "signed_normal"
        self.W_ff = nn.ParameterList()
        self.b_ff = nn.ParameterList()
        for idx, (n_pre, n_post) in enumerate(zip(all_sizes[:-1], all_sizes[1:])):
            # First projection uses w_in spec, last uses w_hid, middle uses w_hid
            spec = w_in if idx == 0 else w_hid
            p1, p2, d, s = _parse_weight_spec(spec, dist, sparsity)
            self.W_ff.append(nn.Parameter(init_weight((n_pre, n_post), d, p1, p2, s)))
            self.b_ff.append(nn.Parameter(torch.zeros(n_post)))
        # Recurrent weights for selected layers
        self.W_rec = nn.ParameterDict()
        for i in self.rec_layers:
            n = all_sizes[i]
            if w_rec is None:
                w_rec_spec = (0, 0.1)
            else:
                w_rec_spec = w_rec
            rec_dist = "signed_normal" if self.signed_weights else dist
            p1, p2, d, s = _parse_weight_spec(w_rec_spec, rec_dist, sparsity)
            self.W_rec[str(i)] = nn.Parameter(init_weight((n, n), d, p1, p2, s))

    def _hid_key(self, layer_idx):
        """Recording key for hidden layer (1-indexed). Single layer uses 'hid'."""
        if self.n_layers == 1:
            return "hid"
        return f"hid_{layer_idx}"

    def forward(self, ext_g=None, input_spikes=None):
        cfg = self._prepare_forward(ext_g, input_spikes)
        state = self._init_hidden_state(cfg)
        readout = self._init_readout_state(cfg)

        # Lazy-init torch.compile on the per-timestep body. Device-agnostic
        # for accelerators (MPS, CUDA); skipped on CPU because Inductor's
        # cpp_builder fails on some hosts (no compatible toolchain) and the
        # error only surfaces at first call, not at torch.compile() time —
        # past our try/except. CPU forward passes are rare in practice
        # (Mac→MPS, Modal→CUDA) so eager there has no real cost.
        # Opt out entirely via PINGLAB_NO_COMPILE=1.
        if (
            "step" not in self._compiled_cache
            and not _env_no_compile()
            and cfg["device"].type != "cpu"
        ):
            try:
                self._compiled_cache["step"] = torch.compile(
                    self._step_body, dynamic=False
                )
            except Exception as exc:  # noqa: BLE001
                self._compiled_cache["step"] = self._step_body
                self._compiled_cache["compile_error"] = str(exc)

        self._forward_loop(cfg, state, readout)

        self._finalise_meta(cfg, readout, state)
        if self.readout_mode == "li":
            return readout["logits_max"]
        if self.readout_mode == "spike-count":
            return readout["s_count"]
        if self.readout_mode == "mem-mean":
            return readout["mem_sum"] / float(T_steps)
        return readout["logits_t"]

    def _forward_loop(self, cfg, state, readout):
        """Outer time loop. Per-t tensor slicing happens here; rec_buf writes
        stay here too. _step_body sees only tensors, no Python int `t`."""
        step = self._compiled_cache.get("step", self._step_body)
        has_in = cfg["has_input_spikes"]
        has_ext = cfg["has_ext_g"]
        in_all = cfg["input_spikes"] if has_in else None
        ida_all = cfg["input_drive_all"]
        ext_all = cfg["ext_g"] if has_ext else None
        rec_buf = cfg["rec_buf"]
        for t in range(T_steps):
            slc = {
                "in_t": (
                    None
                    if in_all is None
                    else in_all[t].unsqueeze(0)
                    if in_all.dim() == 2
                    else in_all[t]
                ),
                "drive0_t": (
                    None
                    if ida_all is None
                    else ida_all[t].unsqueeze(0)
                    if ida_all.dim() == 2
                    else ida_all[t]
                ),
                "ext_t": (
                    None
                    if ext_all is None
                    else ext_all[t].unsqueeze(0)
                    if ext_all.dim() == 2
                    else ext_all[t]
                ),
            }
            step(slc, cfg, state, readout)

            if rec_buf is not None:
                if slc["in_t"] is not None and "input" in rec_buf:
                    rec_buf["input"][t] = slc["in_t"]
                for i in range(self.n_layers):
                    rec_buf[self._hid_key(i + 1)][t] = state["s_prevs"][i]
                rec_buf["out"][t] = readout["logits_t"]

    def _step_body(self, slc, cfg, state, readout):
        """One timestep: all hidden layers + readout. The compile target.

        `slc` carries pre-sliced per-t tensors (in_t, drive0_t, ext_t) so the
        Python int `t` never appears inside the compiled graph. Mutates
        cfg["n_spk_tensors"], state, readout in place.
        """
        prev_spk = None
        for i in range(self.n_layers):
            drive, spike_kick = self._compute_layer_drive(i, prev_spk, slc, state, cfg)

            if self.exponential_synapse:
                if spike_kick is None:
                    raise RuntimeError(
                        "input_drive_all precomputed but exp_synapse is on"
                    )
                state["g_exps"][i] = state["g_exps"][i] * decay_ampa + spike_kick
                drive = (1.0 - cfg["beta"]) * state["g_exps"][i] + cfg[
                    "bias_scale"
                ] * cfg["b_ff"][i]

            s = self._hidden_lif_step(i, drive, state, cfg)
            prev_spk = s
            # n_spk accumulates (pure tensor op, compile-safe); rec_buf
            # writes are keyed by t and live in _forward_loop.
            cfg["n_spk_tensors"][self._hid_key(i + 1)] += s.detach().sum()

        self._readout_step(prev_spk, readout, cfg)

    # -- forward helpers ---------------------------------------------------

    def _prepare_forward(self, ext_g, input_spikes):
        """Resolve shapes, weights, scales, recording buffers, spike counters."""
        has_ext_g = ext_g is not None
        has_input_spikes = input_spikes is not None

        if has_input_spikes and input_spikes.dim() == 3:
            B, device = input_spikes.shape[1], input_spikes.device
        elif has_ext_g and ext_g.dim() == 3:
            B, device = ext_g.shape[1], ext_g.device
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

        W_ff = [W if self.signed_weights else W.clamp(min=0) for W in self.W_ff]
        b_ff = list(self.b_ff)
        W_rec = {
            k: (W if self.signed_weights else W.clamp(min=0))
            for k, W in self.W_rec.items()
        }

        beta = self.beta_override if self.beta_override is not None else beta_snn

        # ZOH (zero-order hold) discretisation: exact integration of
        # τ·dV/dt = -V + I with delta-function spike inputs gives
        # I_step = W·s/Δt + b, so the forward becomes
        # mem = β·mem + (1-β)·(W·s/Δt + b). Per-spike kick is
        # (1-β)/Δt · W ≈ W/τ — dt-invariant.
        if self.discretisation == "zoh":
            spike_scale = (1.0 - beta) / dt
            bias_scale = 1.0 - beta
        else:
            spike_scale = 1.0
            bias_scale = 1.0

        # Per-step matmul on every device — single code path. The previous
        # CUDA-only big-matmul fast-path is removed: it added ~2 GB autograd
        # memory at T=2000 B=256 N_hid=1024, and the original MPS gradient
        # bug it worked around (commit 6cee33e) is no longer the live issue
        # — see the perf scope discussion in nb000 for context.
        input_drive_all = None

        rec_buf = None
        if self.recording:
            rec_buf = {"out": torch.zeros(T_steps, B, N_OUT, device=device)}
            for i in range(self.n_layers):
                rec_buf[self._hid_key(i + 1)] = torch.zeros(
                    T_steps, B, self.hidden_sizes[i], device=device
                )
            if has_input_spikes:
                rec_buf["input"] = torch.zeros(T_steps, B, N_IN, device=device)

        n_spk_tensors = {
            self._hid_key(i + 1): torch.zeros(1, device=device)
            for i in range(self.n_layers)
        }
        n_spk_tensors["out"] = torch.zeros(1, device=device)

        return {
            "B": B,
            "device": device,
            "has_ext_g": has_ext_g,
            "has_input_spikes": has_input_spikes,
            "ext_g": ext_g,
            "input_spikes": input_spikes,
            "input_drive_all": input_drive_all,
            "W_ff": W_ff,
            "b_ff": b_ff,
            "W_rec": W_rec,
            "beta": beta,
            "spike_scale": spike_scale,
            "bias_scale": bias_scale,
            "reset_mode": self.reset_mode,
            "ref_steps_snn": (
                max(1, int(round(self.ref_ms / dt))) if self.ref_ms > 0 else 0
            ),
            "rec_buf": rec_buf,
            "n_spk_tensors": n_spk_tensors,
        }

    def _init_hidden_state(self, cfg):
        B, device = cfg["B"], cfg["device"]
        mems, s_prevs = [], []
        g_exps = [] if self.exponential_synapse else None
        refs = [] if self.ref_ms > 0 else None
        for i, n in enumerate(self.hidden_sizes):
            if self.randomize_init and i == 0:
                mems.append(thr_snn * torch.rand(B, n, device=device))
            else:
                mems.append(init_conductance(B, n, device))
            s_prevs.append(torch.zeros(B, n, device=device))
            if g_exps is not None:
                g_exps.append(init_conductance(B, n, device))
            if refs is not None:
                refs.append(torch.zeros(B, n, device=device))
        rate_counts = [torch.zeros(B, n, device=device) for n in self.hidden_sizes]
        return {
            "mems": mems,
            "s_prevs": s_prevs,
            "g_exps": g_exps,
            "refs": refs,
            "rate_counts": rate_counts,
        }

    def _init_readout_state(self, cfg):
        B, device = cfg["B"], cfg["device"]
        return {
            "hidden_accum": init_conductance(B, self.hidden_sizes[-1], device),
            "v_out": torch.zeros(B, N_OUT, device=device),
            "logits_max": torch.full((B, N_OUT), float("-inf"), device=device),
            "s_count": torch.zeros(B, N_OUT, device=device),
            "mem_sum": torch.zeros(B, N_OUT, device=device),
            "logits_t": None,
        }

    def _compute_layer_drive(self, i, prev_spk, slc, state, cfg):
        """Return (drive, spike_kick).

        spike_kick is None on the CUDA big-matmul fast path (i=0 only).
        Per-t tensors come in via `slc` so the Python int `t` never enters
        this function (required for a stable compile guard).
        """
        B, device = cfg["B"], cfg["device"]
        W, b = cfg["W_ff"][i], cfg["b_ff"][i]
        n = self.hidden_sizes[i]
        spike_scale, bias_scale = cfg["spike_scale"], cfg["bias_scale"]

        if i == 0:
            if slc["in_t"] is not None:
                if slc["drive0_t"] is not None:
                    drive = slc["drive0_t"]
                    spike_kick = None
                else:
                    spike_kick = slc["in_t"] @ W
                    drive = spike_scale * spike_kick + bias_scale * b
            else:
                spike_kick = torch.zeros(B, n, device=device)
                drive = spike_kick
            if slc["ext_t"] is not None:
                drive = drive + slc["ext_t"]
                if spike_kick is not None:
                    spike_kick = spike_kick + slc["ext_t"]
        else:
            spike_kick = prev_spk @ W
            drive = spike_scale * spike_kick + bias_scale * b

        rec_key = str(i + 1)
        if rec_key in cfg["W_rec"]:
            rec_drive = state["s_prevs"][i] @ cfg["W_rec"][rec_key]
            drive = drive + spike_scale * rec_drive
            if spike_kick is not None:
                spike_kick = spike_kick + rec_drive

        return drive, spike_kick

    def _hidden_lif_step(self, i, drive, state, cfg):
        """LIF step + refractory update + state book-keeping. Returns s."""
        refs = state["refs"]
        can_fire = (refs[i] == 0) if refs is not None else None
        state["mems"][i], s = snn_lif_step(
            state["mems"][i],
            drive,
            cfg["beta"],
            spike_snn,
            reset=cfg["reset_mode"],
            can_fire=can_fire,
        )
        if refs is not None:
            refs[i] = torch.where(
                s.bool(),
                torch.full_like(refs[i], float(cfg["ref_steps_snn"])),
                torch.clamp(refs[i] - 1.0, min=0.0),
            )
        state["s_prevs"][i] = s
        state["rate_counts"][i] = state["rate_counts"][i] + s
        return s

    def _readout_step(self, prev_spk, readout, cfg):
        """li: leaky-integrator per class, track max-over-time.
        rate: accumulate last-hidden spikes and project once per step.
        spike-count: spiking output LIF, accumulate output spikes per class
        (snnTorch tutorial 3 pattern).
        rec_buf writes are keyed by t so they happen in _forward_loop."""
        W_out, b_out = cfg["W_ff"][-1], cfg["b_ff"][-1]
        if self.readout_mode == "li":
            I_out = prev_spk @ W_out + b_out
            readout["v_out"] = (
                cfg["beta"] * readout["v_out"] + (1.0 - cfg["beta"]) * I_out
            )
            readout["logits_max"] = torch.maximum(
                readout["logits_max"], readout["v_out"]
            )
            readout["logits_t"] = readout["v_out"]
        elif self.readout_mode in ("spike-count", "mem-mean"):
            # Exp-Euler ZOH on the output LIF (per-spike kick to v_out is
            # (1-β_out)/dt · W ≈ W/τ_out, dt-invariant). Subtract-reset on
            # spike — multiplicative `v_out * (1-s_out)` produces a
            # multiplicative gate in the backward path that overflows
            # through 2000 BPTT steps; subtract is what cuba's hidden
            # layers use and has clean linear backward.
            one_minus_beta = 1.0 - beta_out
            spike_scale = one_minus_beta / dt
            bias_scale = one_minus_beta
            I_out = spike_scale * (prev_spk @ W_out) + bias_scale * b_out
            readout["v_out"] = beta_out * readout["v_out"] + I_out
            s_out = fast_sigmoid_spike(readout["v_out"] - thr_snn, SURROGATE_SLOPE)
            readout["s_count"] = readout["s_count"] + s_out
            readout["mem_sum"] = readout["mem_sum"] + readout["v_out"]
            readout["v_out"] = readout["v_out"] - s_out * thr_snn
            if self.readout_mode == "mem-mean":
                readout["logits_t"] = readout["mem_sum"] / float(T_steps)
            else:
                readout["logits_t"] = readout["s_count"]
        else:
            readout["hidden_accum"] = readout["hidden_accum"] + prev_spk
            readout["logits_t"] = readout["hidden_accum"] @ W_out + b_out

    def _finalise_meta(self, cfg, readout, state):
        sizes = {
            self._hid_key(i + 1): self.hidden_sizes[i] for i in range(self.n_layers)
        }
        sizes["out"] = N_OUT
        n_spk = {k: v.item() for k, v in cfg["n_spk_tensors"].items()}
        rec = None
        if cfg["rec_buf"] is not None:
            rec = {
                k: (v.squeeze(1).cpu() if cfg["B"] == 1 else v.cpu())
                for k, v in cfg["rec_buf"].items()
            }
        self._set_meta(cfg["B"], n_spk, rec, sizes)
        # Gradients intentionally still attached — trainer uses these counts
        # to build the firing-rate regularisation loss.
        self.last_spike_counts = state["rate_counts"]

    def project_dales(self) -> None:
        """Project W_ff and W_rec back onto the non-negative orthant when
        Dale's law is enforced. Called after every optimiser step so the
        stored parameter values match what forward() actually uses."""
        if self.signed_weights:
            return
        with torch.no_grad():
            for W in self.W_ff:
                W.data.clamp_(min=0)
            for W in self.W_rec.values():
                W.data.clamp_(min=0)


def _parse_weight_spec(w, default_dist, default_sparsity):
    """Parse a weight spec tuple: (p1, p2), (p1, p2, dist), or (p1, p2, dist, sparsity)."""
    if len(w) >= 4:
        return w[0], w[1], w[2], w[3]
    elif len(w) == 3:
        return w[0], w[1], w[2], default_sparsity
    return w[0], w[1], default_dist, default_sparsity


def init_weight(shape, dist="normal", p1=0.0, p2=0.1, sparsity=0.0):
    """Initialize a non-negative weight tensor with fan-in normalization.

    Weights are scaled by 1/N_pre so total synaptic input per postsynaptic
    neuron stays constant regardless of presynaptic population size.
    When sparsity > 0, weights are further scaled by 1/(1-sparsity) to
    compensate for zeroed connections (Börgers-style normalization).

    Args:
        shape: (N_pre, N_post) tensor shape
        dist: "signed_normal" — N(p1, p2), no clamp (p1=mean, p2=std)
              "normal"   — N(p1, p2) clamped ≥0  (p1=mean, p2=std)
              "uniform"  — U(p1, p2)              (p1=lo, p2=hi)
              "constant" — all values = p1
              "zeros"    — all zeros
        sparsity: fraction of weights zeroed out (0.0 = dense, 0.9 = 90% zeros)
    """
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
        w = w * (torch.rand(*shape) > sparsity).float()
        w = w / (1.0 - sparsity)  # compensate for zeroed connections
    w = w / n_pre
    return w


class SNNTorchLibraryNet(SNNBase):
    """Parity reference using snn.Leaky from the snnTorch package directly.

    Same Kaiming-uniform feedforward init + linear-decoder readout as
    CUBANet(tutorial_mode=True), but the LIF step and surrogate gradient
    come from snntorch.snn.Leaky + snntorch.surrogate.fast_sigmoid. The only
    thing that can differ between this model and `standard-snn` at
    matched config is the LIF update + surrogate — so accuracy gaps
    localise the difference.

    β is computed from the module-level beta_snn (= exp(-dt/τ)), so the
    dt-sweep reinterpretation still applies.
    """

    randomize_init = False

    def __init__(
        self,
        hidden_sizes=None,
        w_rec=None,
        rec_layers=None,
        readout_mode="rate",
        **_ignored,
    ):
        super().__init__()
        if readout_mode not in ("rate", "li", "spike-count", "mem-mean"):
            raise ValueError(
                f"readout_mode must be 'rate', 'li', 'spike-count', or "
                f"'mem-mean', got {readout_mode!r}"
            )
        self.readout_mode = readout_mode
        import snntorch as snn
        from snntorch import surrogate

        self._snn = snn
        # slope=1 matches pinglab's SurrogateSpike (used by standard-snn and
        # cuba); snnTorch's default is slope=25. Unifying here makes
        # snntorch-library a pure update-rule parity probe vs standard-snn.
        self._surrogate = surrogate.fast_sigmoid(slope=SURROGATE_SLOPE)

        sizes = hidden_sizes if hidden_sizes is not None else HIDDEN_SIZES
        self.hidden_sizes = list(sizes)
        self.n_layers = len(sizes)
        all_sizes = [N_IN] + list(sizes) + [N_OUT]
        self.all_sizes = all_sizes
        self.signed_weights = True  # library uses signed weights

        if rec_layers is not None:
            self.rec_layers = set(rec_layers)
        elif w_rec is not None:
            self.rec_layers = set(range(1, self.n_layers + 1))
        else:
            self.rec_layers = set()

        # Kaiming-uniform nn.Linear layers — matches snnTorch tutorial 5.
        self.fc_ff = nn.ModuleList(
            [
                nn.Linear(n_pre, n_post)
                for n_pre, n_post in zip(all_sizes[:-1], all_sizes[1:])
            ]
        )
        # Mirror under W_ff/b_ff so extract_weights() and downstream consumers
        # that iterate net.W_ff work unchanged.
        self.W_ff = nn.ParameterList([lin.weight for lin in self.fc_ff])
        self.b_ff = nn.ParameterList([lin.bias for lin in self.fc_ff])

        self.W_rec = nn.ParameterDict()
        for i in self.rec_layers:
            n = all_sizes[i]
            w_rec_spec = w_rec if w_rec is not None else (0, 0.1)
            p1, p2, _, s = _parse_weight_spec(w_rec_spec, "signed_normal", 0.0)
            self.W_rec[str(i)] = nn.Parameter(
                init_weight((n, n), "signed_normal", p1, p2, s)
            )

    def _hid_key(self, layer_idx):
        if self.n_layers == 1:
            return "hid"
        return f"hid_{layer_idx}"

    def forward(self, ext_g=None, input_spikes=None):
        has_input_spikes = input_spikes is not None
        has_ext_g = ext_g is not None

        if has_input_spikes and input_spikes.dim() == 3:
            B, device = input_spikes.shape[1], input_spikes.device
        elif has_ext_g and ext_g.dim() == 3:
            B, device = ext_g.shape[1], ext_g.device
        else:
            B, device = (
                1,
                (
                    input_spikes.device
                    if has_input_spikes
                    else ext_g.device
                    if has_ext_g
                    else torch.device("cpu")
                ),
            )

        # snn.Leaky wraps mem state per forward. Fresh modules each call so
        # there's no stale state across batches or infer passes. β honours
        # the current module-level value (patch_dt updates it).
        lifs = [
            self._snn.Leaky(
                beta=float(beta_snn),
                threshold=float(thr_snn),
                spike_grad=self._surrogate,
                reset_mechanism="subtract",
            )
            for _ in range(self.n_layers)
        ]
        # Move to target device once (snn.Leaky is otherwise state-free).
        for lif in lifs:
            lif.to(device)
        mems = [torch.zeros(B, n, device=device) for n in self.hidden_sizes]
        # Randomised init matches CUBANet's option for symmetry breaking.
        if self.randomize_init:
            mems[0] = thr_snn * torch.rand(B, self.hidden_sizes[0], device=device)

        s_prevs = [torch.zeros(B, n, device=device) for n in self.hidden_sizes]
        hidden_accum = torch.zeros(B, self.hidden_sizes[-1], device=device)
        v_out = torch.zeros(B, N_OUT, device=device)
        logits_max = torch.full((B, N_OUT), float("-inf"), device=device)
        s_count = torch.zeros(B, N_OUT, device=device)
        mem_sum = torch.zeros(B, N_OUT, device=device)

        rec_buf = None
        if self.recording:
            rec_buf = {"out": torch.zeros(T_steps, B, N_OUT, device=device)}
            for i in range(self.n_layers):
                rec_buf[self._hid_key(i + 1)] = torch.zeros(
                    T_steps, B, self.hidden_sizes[i], device=device
                )
            if has_input_spikes:
                rec_buf["input"] = torch.zeros(T_steps, B, N_IN, device=device)
        n_spk_tensors = {
            self._hid_key(i + 1): torch.zeros(1, device=device)
            for i in range(self.n_layers)
        }
        n_spk_tensors["out"] = torch.zeros(1, device=device)

        for t in range(T_steps):
            prev_spk: torch.Tensor = torch.empty(0)
            for i in range(self.n_layers):
                if i == 0:
                    if has_input_spikes:
                        spk_t = (
                            input_spikes[t].unsqueeze(0)
                            if input_spikes.dim() == 2
                            else input_spikes[t]
                        )
                        drive = self.fc_ff[0](spk_t)
                        if rec_buf is not None and "input" in rec_buf:
                            rec_buf["input"][t] = spk_t
                    else:
                        drive = torch.zeros(B, self.hidden_sizes[0], device=device)
                    if has_ext_g:
                        ext = ext_g[t].unsqueeze(0) if ext_g.dim() == 2 else ext_g[t]
                        drive = drive + ext
                else:
                    drive = self.fc_ff[i](prev_spk)

                rec_key = str(i + 1)
                if rec_key in self.W_rec:
                    drive = drive + s_prevs[i] @ self.W_rec[rec_key]

                spk, mems[i] = lifs[i](drive, mems[i])
                s_prevs[i] = spk
                prev_spk = spk

                hk = self._hid_key(i + 1)
                n_spk_tensors[hk] = n_spk_tensors[hk] + spk.detach().sum()
                if rec_buf is not None:
                    rec_buf[hk][t] = spk

            if self.readout_mode == "li":
                I_out = self.fc_ff[-1](prev_spk)
                v_out = float(beta_snn) * v_out + (1.0 - float(beta_snn)) * I_out
                logits_max = torch.maximum(logits_max, v_out)
                logits_t = v_out
            elif self.readout_mode in ("spike-count", "mem-mean"):
                # Exp-Euler ZOH on output LIF + subtract reset (see
                # CUBANet._readout_step for rationale).
                one_minus_beta = 1.0 - float(beta_out)
                spike_scale = one_minus_beta / float(dt)
                bias_scale = one_minus_beta
                fc_out = self.fc_ff[-1]
                assert isinstance(fc_out, nn.Linear)
                W_out: torch.Tensor = fc_out.weight
                b_out: torch.Tensor = fc_out.bias
                I_out = spike_scale * (prev_spk @ W_out.t()) + bias_scale * b_out
                v_out = float(beta_out) * v_out + I_out
                s_out = fast_sigmoid_spike(v_out - thr_snn, SURROGATE_SLOPE)
                s_count = s_count + s_out
                mem_sum = mem_sum + v_out
                v_out = v_out - s_out * thr_snn
                if self.readout_mode == "mem-mean":
                    logits_t = mem_sum / float(T_steps)
                else:
                    logits_t = s_count
            else:
                hidden_accum = hidden_accum + prev_spk
                logits_t = self.fc_ff[-1](hidden_accum)
            if rec_buf is not None:
                rec_buf["out"][t] = logits_t

        sizes = {
            self._hid_key(i + 1): self.hidden_sizes[i] for i in range(self.n_layers)
        }
        sizes["out"] = N_OUT
        n_spk = {k: v.item() for k, v in n_spk_tensors.items()}
        rec = None
        if rec_buf is not None:
            rec = {
                k: (v.squeeze(1).cpu() if B == 1 else v.cpu())
                for k, v in rec_buf.items()
            }
        self._set_meta(B, n_spk, rec, sizes)
        if self.readout_mode == "li":
            return logits_max
        if self.readout_mode == "spike-count":
            return s_count
        if self.readout_mode == "mem-mean":
            return mem_sum / float(T_steps)
        return logits_t


class COBANet(SNNBase):
    signed_weights = False

    def __init__(
        self,
        w_in=(W_IN_MEAN, W_IN_STD),
        w_hid=(W_HID_MEAN, W_HID_STD),
        w_ee=(W_EE_MEAN, W_EE_STD),
        w_ei=(W_EI_MEAN, W_EI_STD),
        w_ie=(W_IE_MEAN, W_IE_STD),
        dist="normal",
        sparsity=0.0,
        dales_law=True,
        hidden_sizes=None,
        ei_layers=None,
        readout_mode="rate",
        trainable_w_ee=False,
        slow_synapse=False,
        slow_syn_gain=0.5,
        alif=False,
        alif_beta=1.7,
        sgcc=False,
        sgcc_alpha=0.5,
    ):
        super().__init__()
        if readout_mode not in ("rate", "li", "spike-count", "mem-mean"):
            raise ValueError(
                f"readout_mode must be 'rate', 'li', 'spike-count', or "
                f"'mem-mean', got {readout_mode!r}"
            )
        self.readout_mode = readout_mode
        self.signed_weights = not dales_law
        # Same lazy-compile pattern as CUBANet — see CUBANet for rationale.
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

        # Feedforward weights: input→H1, H1→H2, ..., HN→output
        self.W_ff = nn.ParameterList()
        for idx, (n_pre, n_post) in enumerate(zip(all_sizes[:-1], all_sizes[1:])):
            spec = w_in if idx == 0 else w_hid
            p1, p2, d, s = _parse_weight_spec(spec, dist, sparsity)
            self.W_ff.append(nn.Parameter(init_weight((n_pre, n_post), d, p1, p2, s)))

        # E-I weights per E-I layer. W_ei / W_ie are fixed anatomical
        # connectivity (the recurrent inhibitory circuit is a substrate the
        # readout learns to read, not a trainable thing). W_ee defaults to
        # fixed too, but flipping `trainable_w_ee=True` makes E→E gradient-
        # carrying so the network can learn a Hopfield-style E attractor on
        # top of the COBA biophysics — useful for working-memory tasks like
        # DMTS where activity has to persist across silent delays.
        self.trainable_w_ee = trainable_w_ee
        # Optional slow excitatory ("NMDA-like") channel running in parallel
        # with AMPA on every E-driving projection. When enabled, sample-
        # period spikes leave a long-decay residue (tau_nmda ~ 100 ms) on
        # E neurons via the same W matrices; useful for working-memory
        # tasks where activity has to outlast the membrane / AMPA window.
        # Gain is the multiplier on the slow channel's drive relative to
        # the fast channel; 0.0 makes the network behaviour identical to
        # plain COBA.
        self.slow_synapse = slow_synapse
        self.slow_syn_gain = float(slow_syn_gain)
        # Adaptive LIF: a per-neuron slow variable that rises with each
        # spike and raises the cell's own threshold. Provides a long
        # (tau_adapt ~ 700 ms) timescale on the OUTPUT side of the cell,
        # complementing slow_synapse which is on the input side. β scales
        # the bump per accumulated spike, in mV (since V_th is in mV).
        # LSNN paper default β ≈ 1.7 mV.
        self.alif = alif
        self.alif_beta = float(alif_beta)
        # SGCC (Surrogate Gradients by Costate Control, Burghi et al. 2024)
        # — a surgical replacement for v_grad_dampen. Instead of muting
        # every voltage gradient uniformly, scale only the cross-coupling
        # gradient v ↔ g (the path responsible for the conductance Jacobian
        # explosion in BPTT). alpha is the retained fraction; alpha=1 is
        # no-op, alpha=0 kills the cross-coupling entirely. Paper's example
        # uses alpha around 0.5–0.7.
        self.sgcc = sgcc
        self.sgcc_alpha = float(sgcc_alpha)
        self.W_ee = nn.ParameterDict()
        self.W_ei = nn.ParameterDict()
        self.W_ie = nn.ParameterDict()
        for i in self.ei_layers:
            n_e = sizes[i - 1]
            n_i = n_e // 4  # standard E:I ratio
            k = str(i)
            p1, p2, d, s = _parse_weight_spec(w_ee, dist, sparsity)
            w_ee_t = nn.Parameter(
                init_weight((n_e, n_e), d, p1, p2, s),
                requires_grad=trainable_w_ee,
            )
            p1, p2, d, s = _parse_weight_spec(w_ei, dist, sparsity)
            w_ei_t = nn.Parameter(
                init_weight((n_e, n_i), d, p1, p2, s), requires_grad=False
            )
            p1, p2, d, s = _parse_weight_spec(w_ie, dist, sparsity)
            w_ie_t = nn.Parameter(
                init_weight((n_i, n_e), d, p1, p2, s), requires_grad=False
            )
            self.W_ee[k] = w_ee_t
            self.W_ei[k] = w_ei_t
            self.W_ie[k] = w_ie_t

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
        drive_sigma=0.0,
        input_spikes=None,
    ):
        has_ext_g = ext_g is not None
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
        g_noise = noise_std / (E_e - E_L) if noise_std > 0 else 0.0

        # Per-step matmul on every device — single code path (no CUDA-only
        # fast-path). See CUBANet for the same change and rationale.
        input_drive_all = None

        # Per-layer state
        v_e, ref_e, ge_e, gi_e, s_e = {}, {}, {}, {}, {}
        v_i, ref_i, ge_i, s_i = {}, {}, {}, {}
        ge_e_slow: dict = {}  # populated only when slow_synapse is set
        a_e: dict = {}        # populated only when alif is set
        a_i: dict = {}
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
            if self.slow_synapse:
                ge_e_slow[k] = init_conductance(B, n_e, device)
            if self.alif:
                a_e[k] = torch.zeros(B, n_e, device=device)
            s_e[k] = torch.zeros(B, n_e, device=device)
            if i in self.ei_layers:
                n_i = n_e // 4
                gi_e[k] = init_conductance(B, n_e, device)
                v_i[k], ref_i[k] = init_lif_state(
                    B, n_i, device, randomize=randomize_init
                )
                ge_i[k] = init_conductance(B, n_i, device)
                s_i[k] = torch.zeros(B, n_i, device=device)
                if self.alif:
                    a_i[k] = torch.zeros(B, n_i, device=device)
            if drive_sigma > 0 and i == 1:
                drive_gains[k] = (
                    1.0 + drive_sigma * torch.randn(B, n_e, device=device)
                ).clamp(min=0)

        # Output: cumulative last-hidden-layer spikes → linear decoder
        # (Same readout as CUBANet — no output spiking neurons, no
        # per-model confound at the output.)
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
                    n_inh = n_e // 4
                    rec_buf[self._inh_key(i)] = torch.zeros(
                        T_steps, B, n_inh, device=device
                    )
                    rec_buf[f"gi_e_{i}"] = torch.zeros(T_steps, B, n_e, device=device)
                    rec_buf[f"v_i_{i}"] = torch.zeros(T_steps, B, n_inh, device=device)
                    rec_buf[f"ge_i_{i}"] = torch.zeros(T_steps, B, n_inh, device=device)
        # GPU-side spike accumulators
        n_spk_tensors = {}
        for i in range(1, self.n_layers + 1):
            n_spk_tensors[self._hid_key(i)] = torch.zeros(1, device=device)
            if i in self.ei_layers:
                n_spk_tensors[self._inh_key(i)] = torch.zeros(1, device=device)
        n_spk_tensors["out"] = torch.zeros(1, device=device)
        # Per-layer (B, n_e) spike-count accumulator for the firing-rate
        # regulariser — must keep gradient attached, so it sums state["s_e"]
        # post-step (CUBANet does the same in its rate_counts list).
        rate_counts = [
            torch.zeros(B, n, device=device) for n in self.hidden_sizes
        ]

        # Bundle mutating state and per-call config so _step_body can be
        # compiled per-timestep (the same boundary CUBANet uses). The Python
        # int `t` and rec_buf writes stay in _forward_loop so the compiled
        # graph never has to re-trace per-t.
        state = {
            "v_e": v_e,
            "ref_e": ref_e,
            "ge_e": ge_e,
            "ge_e_slow": ge_e_slow,
            "gi_e": gi_e,
            "s_e": s_e,
            "v_i": v_i,
            "ref_i": ref_i,
            "ge_i": ge_i,
            "s_i": s_i,
            "a_e": a_e,
            "a_i": a_i,
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
            "readout_mode": self.readout_mode,
            "g_noise": g_noise,
            "n_e0": self.hidden_sizes[0],
            "n_spk_tensors": n_spk_tensors,
            "slow_synapse": self.slow_synapse,
            "slow_syn_gain": self.slow_syn_gain,
            "alif": self.alif,
            "alif_beta": self.alif_beta,
            "sgcc": self.sgcc,
            "sgcc_alpha": self.sgcc_alpha,
        }

        # Lazy-init torch.compile on the per-timestep body. Same pattern,
        # rationale, CPU-skip, and PINGLAB_NO_COMPILE escape hatch as
        # CUBANet. CPU is skipped because Inductor's cpp build fails on
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
        # firing-rate regulariser (oscilloscope.train) can build its loss.
        # Mirrors CUBANet.last_spike_counts.
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
        g_noise = cfg["g_noise"]
        n_spk_tensors = cfg["n_spk_tensors"]
        slow_on = cfg["slow_synapse"]
        slow_gain = cfg["slow_syn_gain"]
        alif_on = cfg["alif"]
        alif_beta = cfg["alif_beta"]
        sgcc_on = cfg["sgcc"]
        sgcc_alpha = cfg["sgcc_alpha"]

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
                state["ge_e"][k] = (state["ge_e"][k] + ee_drive) * decay_ampa
                state["ge_i"][k] = (
                    state["ge_i"][k] + state["s_e"][k] @ self.W_ei[k]
                ) * decay_ampa
                state["gi_e"][k] = (
                    state["gi_e"][k] + state["s_i"][k] @ self.W_ie[k]
                ) * decay_gaba
                if slow_on:
                    state["ge_e_slow"][k] = (
                        state["ge_e_slow"][k] + slow_gain * ee_drive
                    ) * decay_nmda
            else:
                state["ge_e"][k] = state["ge_e"][k] * decay_ampa
                if slow_on:
                    state["ge_e_slow"][k] = state["ge_e_slow"][k] * decay_nmda

            if i == 1:
                if has_input_spikes:
                    g_ext = slc["in_t"] @ W
                    if k in drive_gains:
                        g_ext = g_ext * drive_gains[k]
                    state["ge_e"][k] = state["ge_e"][k] + g_ext
                    if slow_on:
                        state["ge_e_slow"][k] = (
                            state["ge_e_slow"][k] + slow_gain * g_ext
                        )
                if has_ext_g:
                    state["ge_e"][k] = state["ge_e"][k] + slc["ext_t"]
                    if slow_on:
                        state["ge_e_slow"][k] = (
                            state["ge_e_slow"][k] + slow_gain * slc["ext_t"]
                        )
            else:
                ff_drive = prev_spk @ W
                state["ge_e"][k] = state["ge_e"][k] + ff_drive
                if slow_on:
                    state["ge_e_slow"][k] = (
                        state["ge_e_slow"][k] + slow_gain * ff_drive
                    )

            if g_noise > 0 and i == 1:
                state["ge_e"][k] = state["ge_e"][k] + (
                    g_noise * torch.randn(cfg["B"], cfg["n_e0"], device=cfg["device"])
                ).clamp(min=0)

            # Combined fast + slow conductance drives the membrane.
            g_e_total = (
                state["ge_e"][k] + state["ge_e_slow"][k]
                if slow_on
                else state["ge_e"][k]
            )
            # SGCC (Burghi et al.): scale the gradient on the
            # voltage↔conductance cross-coupling by sgcc_alpha. forward
            # value is unchanged; backward gradient through this path is
            # multiplied by sgcc_alpha. Surgically tames the conductance
            # Jacobian explosion without uniformly damping all gradients.
            if sgcc_on:
                g_e_for_step = _scale_grad(g_e_total, sgcc_alpha)
                g_i_for_e = (
                    _scale_grad(state["gi_e"][k], sgcc_alpha) if is_ei else None
                )
                g_e_for_i = (
                    _scale_grad(state["ge_i"][k], sgcc_alpha) if is_ei else None
                )
            else:
                g_e_for_step = g_e_total
                g_i_for_e = state["gi_e"][k] if is_ei else None
                g_e_for_i = state["ge_i"][k] if is_ei else None
            # ALIF effective threshold: V_th_eff = V_th + β · a. Built
            # as a per-neuron offset tensor so the spike step compares
            # v against the right per-cell threshold.
            e_thresh_offset = (
                alif_beta * state["a_e"][k] if alif_on else None
            )
            i_thresh_offset = (
                alif_beta * state["a_i"][k] if alif_on and is_ei else None
            )
            if is_ei:
                state["v_e"][k], state["s_e"][k], state["ref_e"][k] = e_step_coba(
                    state["v_e"][k],
                    state["ref_e"][k],
                    g_e_for_step,
                    g_i_for_e,
                    threshold_offset=e_thresh_offset,
                )
                state["v_i"][k], state["s_i"][k], state["ref_i"][k] = i_step_coba(
                    state["v_i"][k], state["ref_i"][k], g_e_for_i,
                    threshold_offset=i_thresh_offset,
                )
            else:
                state["v_e"][k], state["s_e"][k], state["ref_e"][k] = e_step_coba(
                    state["v_e"][k], state["ref_e"][k], g_e_for_step,
                    threshold_offset=e_thresh_offset,
                )
            # ALIF adaptation update: a_{t+1} = decay_adapt · a_t + s_t.
            # Uses the FRESH spikes (post-step), so the threshold offset
            # bumps up *for the next step*, not the current one.
            if alif_on:
                state["a_e"][k] = (
                    decay_adapt * state["a_e"][k] + state["s_e"][k]
                )
                if is_ei:
                    state["a_i"][k] = (
                        decay_adapt * state["a_i"][k] + state["s_i"][k]
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
            # Exp-Euler ZOH on output LIF + subtract reset (see
            # CUBANet._readout_step for rationale). COBANet's W_ff has
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

    def project_dales(self) -> None:
        """Project W_ff back onto the non-negative orthant when Dale's
        law is enforced. The recurrent W_ee / W_ei / W_ie are stored as
        non-trainable buffers (requires_grad=False) so they are never
        touched by the optimiser and need no projection."""
        if self.signed_weights:
            return
        with torch.no_grad():
            for W in self.W_ff:
                W.data.clamp_(min=0)
