"""SNN model definitions and layer primitives.

All constants are hardcoded defaults. Override via module-level assignment
or by passing arguments to model constructors.

Models: SNNTorchNet, PINGNet.
Layer primitives: exp_synapse, lif_step, snn_lif_step.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── Simulation ────────────────────────────────────────────────────────────
dt            = 0.25      # ms — integration timestep
T_ms          = 1000.0    # ms — total simulation time per sample
T_steps       = int(T_ms / dt)

# ── Biophysics ────────────────────────────────────────────────────────────
tau_m_E       = 20.0      # ms — excitatory membrane time constant
tau_m_ratio   = 4.0       # tau_m_E / tau_m_I (Börgers: 20ms / 5ms)
C_m_E         = 1.0       # nF — excitatory capacitance (fixed reference)
_CM_RATIO     = 2.0       # C_m_E / C_m_I (fixed)
g_L_E         = C_m_E / tau_m_E           # 0.05 uS
tau_m_I       = tau_m_E / tau_m_ratio     # 5.0 ms
C_m_I         = C_m_E / _CM_RATIO        # 0.5 nF
g_L_I         = C_m_I / tau_m_I           # 0.1 uS
E_L           = -65.0     # mV — leak / resting potential
E_e           = 0.0       # mV — excitatory (AMPA) reversal
E_i           = -80.0     # mV — inhibitory (GABA) reversal
V_th          = -50.0     # mV — spike threshold
V_reset       = -65.0     # mV — post-spike reset voltage
V_floor       = -200.0    # mV — hard lower clamp
ref_ms_E      = 3.0       # ms — excitatory refractory period
_REF_RATIO    = 2.0       # ref_ms_E / ref_ms_I (Börgers)
ref_ms_I      = ref_ms_E / _REF_RATIO   # 1.5 ms
tau_ampa      = 2.0       # ms — AMPA decay
tau_gaba      = 9.0       # ms — GABA decay (Börgers: 9 ms; Buzsaki & Wang: 8-12 ms)

# ── Input encoding ────────────────────────────────────────────────────────
max_rate_hz   = 25.0      # Hz — max Poisson rate for fully-on pixel (sensory-input scale, LGN-ish)
input_scale   = 20.0      # nA per input spike (CUBA only)

# ── snnTorch ──────────────────────────────────────────────────────────────
tau_snn       = 10.0      # ms — membrane time constant
thr_snn       = 1.0       # spike threshold

# ── Architecture ──────────────────────────────────────────────────────────
N_IN          = 64        # input neurons (8×8 scikit-digits)
N_HID         = 64        # hidden excitatory neurons (last layer size for compat)
N_INH         = 16        # inhibitory neurons (PING only, per E-I layer)
N_OUT         = 10        # output neurons (one per digit class)
HIDDEN_SIZES  = [64]      # list of hidden layer sizes (N_HID is always last entry)

# ── Weight init ───────────────────────────────────────────────────────────
# Weight init — p1/p2 are pre-fan-in values (init_weight divides by N_pre)
W_STD_CUBA    = 32.0      # nA — CUBA weight init std (pre-fan-in)
W_STD_COBA    = 6.4       # uS — COBA weight init std (pre-fan-in)
W_FF_MEAN     = 5.1       # uS — feedforward init mean (pre-fan-in)
W_FF_STD      = 3.8       # uS — feedforward init std (pre-fan-in)
W_IN_MEAN     = W_FF_MEAN  # alias
W_IN_STD      = W_FF_STD   # alias
W_HID_MEAN    = W_FF_MEAN  # alias
W_HID_STD     = W_FF_STD   # alias
W_EE_MEAN     = 0.0       # uS — E→E recurrent init mean (0 per Börgers: PING needs no E→E)
W_EE_STD      = 0.0       # uS — E→E recurrent init std (0 per Börgers: PING needs no E→E)
W_EI_MEAN     = 1.0       # uS — E→I init mean (just suprathreshold, Börgers)
W_EI_STD      = 0.5       # uS — E→I init std (pre-fan-in)
W_IE_MEAN     = 3.0       # uS — I→E init mean (2-3× E→I, Viriyopase et al.)
W_IE_STD      = 1.5       # uS — I→E init std (pre-fan-in)
delay_ei_ms   = 1.0       # ms — E→I synaptic delay
delay_ie_ms   = 1.0       # ms — I→E synaptic delay

# ── Training ──────────────────────────────────────────────────────────────
BATCH_SIZE    = 64
GRAD_CLIP     = 1.0
READOUT_SCALE = 0.0
PATIENCE      = 15
CM_BACK_SCALE = 80.0

# Derived
decay_ampa   = np.exp(-dt / tau_ampa)
decay_gaba   = np.exp(-dt / tau_gaba)
ref_steps_E  = max(1, int(round(ref_ms_E / dt)))
ref_steps_I  = max(1, int(round(ref_ms_I / dt)))
p_scale      = max_rate_hz * dt / 1000.0
beta_snn     = np.exp(-dt / tau_snn)
delay_ei_steps = max(1, int(round(delay_ei_ms / dt)))
delay_ie_steps = max(1, int(round(delay_ie_ms / dt)))


# ── Surrogate gradient ───────────────────────────────────────────────────

class SurrogateSpike(torch.autograd.Function):
    """Fast sigmoid surrogate: grad = slope / (1 + slope·|x|)^2.

    Matches snntorch.surrogate.FastSigmoid: slope controls how sharply the
    pseudo-derivative peaks around u=0. slope=25 (snntorch default) yields a
    narrow window of active gradient (~±0.04 around threshold); slope=1 gives
    a much broader window (~±1). The biophysical path operates on mV-scale
    membrane potentials and needs slope=1; the snn/tutorial path matches the
    snntorch default of slope=25 to keep the parity reference honest.
    """
    @staticmethod
    def forward(ctx, u, slope):
        ctx.save_for_backward(u)
        ctx.slope = slope
        return (u >= 0).float()

    @staticmethod
    def backward(ctx, grad_out):
        (u,) = ctx.saved_tensors
        slope = ctx.slope
        grad = slope / (1.0 + slope * u.abs()) ** 2
        return grad_out * grad, None

def spike_biophysical(v):
    # mV-scale membrane: slope=1 keeps gradient support at the ~mV width of
    # typical threshold crossings.
    return SurrogateSpike.apply(v - V_th, 1.0)

def spike_snn(v):
    # Dimensionless membrane (threshold=1). NOTE: slope=1 (not snntorch's default
    # 25) — slope=25 starves gradients from silent neurons (|u|>>0), and pinglab's
    # init can land in a silent regime at some seeds, preventing training from
    # bootstrapping. The permissive slope=1 surrogate reliably escapes. The
    # snntorch-library parity path keeps slope=25 internally; part of what
    # calibration has to explain is this slope asymmetry.
    return SurrogateSpike.apply(v - thr_snn, 1.0)


def _scale_grad(x, scale):
    """Return x unchanged in forward, but multiply gradient by scale in backward."""
    return x * scale + x.detach() * (1.0 - scale)


# ── Layer primitives ─────────────────────────────────────────────────────

def exp_synapse(g, spikes, W, decay):
    """Exponential synapse: spike kicks first, then decay."""
    return (g + spikes @ W) * decay

def lif_step(v, I_total, ref, C_m, g_L, ref_steps, spike_fn, V_floor=V_floor, V_max=None, cm_back=1.0):
    """One LIF timestep: voltage update, spike decision, then reset.
    Returns (v, s, ref)."""
    dv = (dt / C_m) * (-g_L * (v - E_L) + I_total)
    if cm_back != 1.0:
        dv = _scale_grad(dv, 1.0 / cm_back)
    v = v + dv
    v = v.clamp(min=V_floor) if V_max is None else v.clamp(min=V_floor, max=V_max)
    ref = (ref - 1).clamp(min=0)
    can_spike = ref == 0
    s = spike_fn(v) * can_spike.float()
    spiked_or_ref = s.bool() | (~can_spike)
    v = torch.where(spiked_or_ref, torch.full_like(v, V_reset), v)
    ref = torch.where(s.bool(), torch.full_like(ref, ref_steps), ref)
    return v, s, ref

def snn_lif_step(mem, I, beta, spike_fn, reset="zero", can_fire=None):
    """snnTorch-style LIF step: decay + input, spike, reset. Returns (mem, s).

    Caller is responsible for dt-scaling of the input. Same primitive for
    snntorch and cuba; what differs is how I is constructed.

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

def hybrid_readout(spike_counts, v_sum):
    """Spike count + voltage readout."""
    return spike_counts + READOUT_SCALE * v_sum / T_steps

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
        ref = (torch.randn(B, N, device=device) * ref_std + ref_mean).clamp(min=0).long()
    else:
        ref = torch.zeros(B, N, device=device, dtype=torch.long)
    return v, ref

def init_conductance(B, N, device):
    """Initialise a conductance variable to zero."""
    return torch.zeros(B, N, device=device)


# ── E-step and I-step composites ─────────────────────────────────────────

def e_step_coba(v, ref, g_e, g_i=None, ref_steps=None):
    """One E-neuron LIF step with COBA driving force."""
    if ref_steps is None:
        ref_steps = ref_steps_E
    return lif_step(v, coba_current(g_e, v, g_i), ref, C_m_E, g_L_E, ref_steps, spike_biophysical, cm_back=CM_BACK_SCALE)

def i_step_coba(v, ref, g_e):
    """One I-neuron LIF step with COBA driving force."""
    return lif_step(v, coba_current(g_e, v), ref, C_m_I, g_L_I, ref_steps_I, spike_biophysical, cm_back=CM_BACK_SCALE)



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


# ── Model classes ────────────────────────────────────────────────────────

class SNNTorchNet(SNNBase):
    # Biophysical defaults (Dale's law + hard reset + membrane readout).
    # Tutorial mode overrides every one of these.
    signed_weights = False
    beta_override = None
    reset_mode = "zero"
    tutorial_readout = False
    randomize_init = False

    def __init__(self, discretisation="canonical",
                 w_in=None, w_hid=None, w_rec=None,
                 dist="normal", sparsity=0.0,
                 tutorial_mode=False, dales_law=True,
                 hidden_sizes=None, rec_layers=None,
                 exponential_synapse=False,
                 ref_ms=0.0,
                 reset_mode=None):
        super().__init__()
        if discretisation not in ("canonical", "continuous"):
            raise ValueError(
                f"discretisation must be 'canonical' or 'continuous', "
                f"got {discretisation!r}")
        self.discretisation = discretisation
        self.exponential_synapse = exponential_synapse
        self.ref_ms = ref_ms
        self._reset_mode_override = reset_mode
        self.signed_weights = not dales_law

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
            self.beta_override = 0.95
            self.reset_mode = self._reset_mode_override or "subtract"
            self.tutorial_readout = True
            self._init_tutorial_weights(all_sizes, w_rec)
        else:
            self._init_biophysical_weights(all_sizes, w_in, w_hid, w_rec, dist, sparsity)
            if self._reset_mode_override is not None:
                self.reset_mode = self._reset_mode_override

    def _init_tutorial_weights(self, all_sizes, w_rec=None):
        # Kaiming-uniform with a=sqrt(5) reduces to uniform(-1/sqrt(fan_in),
        # 1/sqrt(fan_in)). W is stored as (n_pre, n_post), but the semantic
        # fan_in is n_pre (each post-neuron receives n_pre inputs). Torch's
        # kaiming_uniform_ on a 2D tensor defaults to fan_in = size(1) = n_post,
        # which mis-scales the output projection (1024→10: fan_in would become
        # 10 and give a ~10× too-large init vs. nn.Linear's parity). Compute
        # the bound explicitly from n_pre to match nn.Linear semantics.
        self.W_ff = nn.ParameterList()
        self.b_ff = nn.ParameterList()
        for n_pre, n_post in zip(all_sizes[:-1], all_sizes[1:]):
            W = nn.Parameter(torch.empty(n_pre, n_post))
            b = nn.Parameter(torch.empty(n_post))
            bound = 1.0 / math.sqrt(n_pre) if n_pre > 0 else 0
            nn.init.uniform_(W, -bound, bound)
            nn.init.uniform_(b, -bound, bound)
            self.W_ff.append(W)
            self.b_ff.append(b)
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
                init_weight((n, n), "signed_normal", p1, p2, s))

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
            self.W_rec[str(i)] = nn.Parameter(
                init_weight((n, n), d, p1, p2, s))

    def _hid_key(self, layer_idx):
        """Recording key for hidden layer (1-indexed). Single layer uses 'hid'."""
        if self.n_layers == 1:
            return "hid"
        return f"hid_{layer_idx}"

    def forward(self, ext_g=None, input_spikes=None):
        has_ext_g = ext_g is not None
        has_input_spikes = input_spikes is not None

        if has_input_spikes and input_spikes.dim() == 3:
            B, device = input_spikes.shape[1], input_spikes.device
        elif has_ext_g and ext_g.dim() == 3:
            B, device = ext_g.shape[1], ext_g.device
        else:
            B, device = 1, (ext_g.device if has_ext_g
                            else input_spikes.device if has_input_spikes
                            else torch.device("cpu"))

        # Resolve weights (clamp if Dale's law)
        W_ff = [W if self.signed_weights else W.clamp(min=0) for W in self.W_ff]
        b_ff = list(self.b_ff)
        W_rec = {}
        for k, W in self.W_rec.items():
            W_rec[k] = W if self.signed_weights else W.clamp(min=0)

        beta = self.beta_override if self.beta_override is not None else beta_snn
        reset_mode = self.reset_mode

        # Continuous-time discretisation: exact integration of τ·dV/dt = -V + I
        # with delta-function spike inputs gives I_step = W·s/Δt + b, so the
        # forward becomes  mem = β·mem + (1-β)·(W·s/Δt + b). This yields a
        # per-spike kick of (1-β)/Δt · W ≈ W/τ — dt-invariant.
        # Equivalent factorisation: spike_drive × (1-β)/Δt + bias × (1-β).
        if self.discretisation == "continuous":
            spike_scale = (1.0 - beta) / dt
            bias_scale = (1.0 - beta)
        else:
            spike_scale = 1.0
            bias_scale = 1.0

        # Pre-compute input drive for all timesteps as one big matmul.
        # Only on CUDA where big matmuls are efficient. On CPU, the (T, B, N_hid)
        # intermediate blows the cache and swaps — per-step is faster. On MPS,
        # holding a (T, B, N_hid) tensor (hundreds of MB) in the autograd graph
        # triggers a per-step .item() sync storm on backward — per-step matmul
        # mirrors what the snntorch library path does and stays fast.
        # Exp-synapse path needs raw (W·s) to accumulate into g; skip fast-path.
        input_drive_all = None
        if (has_input_spikes and device.type == "cuda"
                and not self.exponential_synapse):
            input_drive_all = (spike_scale * (input_spikes @ W_ff[0])
                               + bias_scale * b_ff[0])

        # Hidden layer state: membrane, prev spikes, optional exp-synapse g,
        # optional refractory counter.
        mems = []
        s_prevs = []
        g_exps = [] if self.exponential_synapse else None
        refs = [] if self.ref_ms > 0 else None
        ref_steps_snn = max(1, int(round(self.ref_ms / dt))) if self.ref_ms > 0 else 0
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

        # Output: cumulative last-hidden-layer spikes → linear decoder
        hidden_accum = init_conductance(B, self.hidden_sizes[-1], device)

        # Recording — pre-allocate GPU buffers to avoid per-step CPU transfers
        rec_buf = None
        if self.recording:
            rec_buf = {"out": torch.zeros(T_steps, B, N_OUT, device=device)}
            for i in range(self.n_layers):
                rec_buf[self._hid_key(i + 1)] = torch.zeros(
                    T_steps, B, self.hidden_sizes[i], device=device)
            if has_input_spikes:
                rec_buf["input"] = torch.zeros(
                    T_steps, B, N_IN, device=device)
        # Spike counts accumulated as GPU tensors (avoid .item() per step)
        n_spk_tensors = {self._hid_key(i + 1): torch.zeros(1, device=device)
                         for i in range(self.n_layers)}
        n_spk_tensors["out"] = torch.zeros(1, device=device)

        for t in range(T_steps):
            # Process each hidden layer
            prev_spk = None  # output of previous layer for this timestep
            for i in range(self.n_layers):
                W, b = W_ff[i], b_ff[i]
                n = self.hidden_sizes[i]

                # Compute per-step spike-derived input kick (unscaled, just W·s)
                # and bias separately. spike_scale / bias_scale apply below.
                if i == 0:
                    if has_input_spikes:
                        spk_t = (input_spikes[t].unsqueeze(0)
                                 if input_spikes.dim() == 2 else input_spikes[t])
                        if input_drive_all is not None:
                            # Fast path: drive already includes spike-scale × (s @ W) + bias-scale × b
                            drive = (input_drive_all[t].unsqueeze(0)
                                     if input_drive_all.dim() == 2
                                     else input_drive_all[t])
                            spike_kick = None  # not used when fast-path active
                        else:
                            spike_kick = spk_t @ W
                            drive = spike_scale * spike_kick + bias_scale * b
                        if rec_buf is not None and "input" in rec_buf:
                            rec_buf["input"][t] = spk_t
                    else:
                        spike_kick = torch.zeros(B, n, device=device)
                        drive = spike_kick
                    if has_ext_g:
                        ext = ext_g[t].unsqueeze(0) if ext_g.dim() == 2 else ext_g[t]
                        drive = drive + ext
                        if spike_kick is not None:
                            spike_kick = spike_kick + ext
                else:
                    spike_kick = prev_spk @ W
                    drive = spike_scale * spike_kick + bias_scale * b

                # Recurrent drive (detached to prevent gradient explosion)
                rec_key = str(i + 1)
                if rec_key in W_rec:
                    rec_drive = s_prevs[i].detach() @ W_rec[rec_key]
                    drive = drive + spike_scale * rec_drive
                    if spike_kick is not None:
                        spike_kick = spike_kick + rec_drive

                # Exponential synapse: spikes feed a conductance g that decays
                # with τ_AMPA; V is driven by (1-β)·g + bias. Overrides the
                # normal drive computation. Bias path unchanged (still (1-β)·b).
                if self.exponential_synapse:
                    if spike_kick is None:
                        # Fast-path was active but exp_synapse was requested —
                        # we guarded against this earlier, so this shouldn't hit.
                        raise RuntimeError(
                            "input_drive_all precomputed but exp_synapse is on")
                    g_exps[i] = g_exps[i] * decay_ampa + spike_kick
                    drive = (1.0 - beta) * g_exps[i] + bias_scale * b

                # Refractory mask
                can_fire = (refs[i] == 0) if refs is not None else None

                # LIF step
                mems[i], s = snn_lif_step(mems[i], drive, beta,
                                          spike_snn, reset=reset_mode,
                                          can_fire=can_fire)

                # Update refractory counter
                if refs is not None:
                    refs[i] = torch.where(
                        s.bool(),
                        torch.full_like(refs[i], float(ref_steps_snn)),
                        torch.clamp(refs[i] - 1.0, min=0.0))

                s_prevs[i] = s
                prev_spk = s

                hk = self._hid_key(i + 1)
                # Detach: spike counts are metadata, not training signal.
                # Without detach, this += chains a 2400-step grad graph that
                # makes any later .item() stall on graph traversal.
                n_spk_tensors[hk] += s.detach().sum()
                if rec_buf is not None:
                    rec_buf[hk][t] = s

            # Linear decoder: accumulate last-hidden spikes, project via W_out.
            # Same readout across all models in the ladder — no output-layer
            # spiking dynamics, no model-specific confound at the output.
            hidden_accum = hidden_accum + prev_spk
            logits_t = hidden_accum @ W_ff[-1] + b_ff[-1]
            if rec_buf is not None:
                rec_buf["out"][t] = logits_t

        sizes = {self._hid_key(i + 1): self.hidden_sizes[i]
                 for i in range(self.n_layers)}
        sizes["out"] = N_OUT
        n_spk = {k: v.item() for k, v in n_spk_tensors.items()}
        rec = None
        if rec_buf is not None:
            rec = {k: (v.squeeze(1).cpu() if B == 1 else v.cpu())
                   for k, v in rec_buf.items()}
        self._set_meta(B, n_spk, rec, sizes)
        return logits_t



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
    SNNTorchNet(tutorial_mode=True), but the LIF step and surrogate gradient
    come from snntorch.snn.Leaky + snntorch.surrogate.fast_sigmoid. The only
    thing that can differ between this model and `snntorch` at
    matched config is the LIF update + surrogate — so accuracy gaps
    localise the difference.

    β is computed from the module-level beta_snn (= exp(-dt/τ)), so the
    dt-sweep reinterpretation still applies.
    """

    randomize_init = False

    def __init__(self, hidden_sizes=None, w_rec=None, rec_layers=None,
                 **_ignored):
        super().__init__()
        import snntorch as snn
        from snntorch import surrogate
        self._snn = snn
        self._surrogate = surrogate.fast_sigmoid()

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
        self.fc_ff = nn.ModuleList([
            nn.Linear(n_pre, n_post)
            for n_pre, n_post in zip(all_sizes[:-1], all_sizes[1:])
        ])
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
                init_weight((n, n), "signed_normal", p1, p2, s))

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
            B, device = 1, (input_spikes.device if has_input_spikes
                            else ext_g.device if has_ext_g
                            else torch.device("cpu"))

        # snn.Leaky wraps mem state per forward. Fresh modules each call so
        # there's no stale state across batches or infer passes. β honours
        # the current module-level value (patch_dt updates it).
        lifs = [self._snn.Leaky(beta=float(beta_snn), threshold=float(thr_snn),
                                spike_grad=self._surrogate,
                                reset_mechanism="subtract")
                for _ in range(self.n_layers)]
        # Move to target device once (snn.Leaky is otherwise state-free).
        for lif in lifs:
            lif.to(device)
        mems = [torch.zeros(B, n, device=device) for n in self.hidden_sizes]
        # Randomised init matches SNNTorchNet's option for symmetry breaking.
        if self.randomize_init:
            mems[0] = thr_snn * torch.rand(B, self.hidden_sizes[0], device=device)

        s_prevs = [torch.zeros(B, n, device=device) for n in self.hidden_sizes]
        hidden_accum = torch.zeros(B, self.hidden_sizes[-1], device=device)

        rec_buf = None
        if self.recording:
            rec_buf = {"out": torch.zeros(T_steps, B, N_OUT, device=device)}
            for i in range(self.n_layers):
                rec_buf[self._hid_key(i + 1)] = torch.zeros(
                    T_steps, B, self.hidden_sizes[i], device=device)
            if has_input_spikes:
                rec_buf["input"] = torch.zeros(T_steps, B, N_IN, device=device)
        n_spk_tensors = {self._hid_key(i + 1): torch.zeros(1, device=device)
                         for i in range(self.n_layers)}
        n_spk_tensors["out"] = torch.zeros(1, device=device)

        for t in range(T_steps):
            prev_spk = None
            for i in range(self.n_layers):
                if i == 0:
                    if has_input_spikes:
                        spk_t = (input_spikes[t].unsqueeze(0)
                                 if input_spikes.dim() == 2 else input_spikes[t])
                        drive = self.fc_ff[0](spk_t)
                        if rec_buf is not None and "input" in rec_buf:
                            rec_buf["input"][t] = spk_t
                    else:
                        drive = torch.zeros(B, self.hidden_sizes[0], device=device)
                    if has_ext_g:
                        ext = (ext_g[t].unsqueeze(0) if ext_g.dim() == 2 else ext_g[t])
                        drive = drive + ext
                else:
                    drive = self.fc_ff[i](prev_spk)

                rec_key = str(i + 1)
                if rec_key in self.W_rec:
                    drive = drive + s_prevs[i].detach() @ self.W_rec[rec_key]

                spk, mems[i] = lifs[i](drive, mems[i])
                s_prevs[i] = spk
                prev_spk = spk

                hk = self._hid_key(i + 1)
                n_spk_tensors[hk] = n_spk_tensors[hk] + spk.detach().sum()
                if rec_buf is not None:
                    rec_buf[hk][t] = spk

            hidden_accum = hidden_accum + prev_spk
            logits_t = self.fc_ff[-1](hidden_accum)
            if rec_buf is not None:
                rec_buf["out"][t] = logits_t

        sizes = {self._hid_key(i + 1): self.hidden_sizes[i]
                 for i in range(self.n_layers)}
        sizes["out"] = N_OUT
        n_spk = {k: v.item() for k, v in n_spk_tensors.items()}
        rec = None
        if rec_buf is not None:
            rec = {k: (v.squeeze(1).cpu() if B == 1 else v.cpu())
                   for k, v in rec_buf.items()}
        self._set_meta(B, n_spk, rec, sizes)
        return logits_t


class PINGNet(SNNBase):
    signed_weights = False

    def __init__(self, w_in=(W_IN_MEAN, W_IN_STD), w_hid=(W_HID_MEAN, W_HID_STD),
                 w_ee=(W_EE_MEAN, W_EE_STD), w_ei=(W_EI_MEAN, W_EI_STD),
                 w_ie=(W_IE_MEAN, W_IE_STD), dist="normal", sparsity=0.0,
                 dales_law=True, hidden_sizes=None, ei_layers=None):
        super().__init__()
        self.signed_weights = not dales_law

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

        # E-I weights per E-I layer — fixed anatomical connectivity
        # (buffers, not parameters: gradients don't flow through these, so
        # the recurrent E-I circuit is a fixed substrate that the network
        # learns to read out via trainable W_in / W_out only).
        self.W_ee = nn.ParameterDict()
        self.W_ei = nn.ParameterDict()
        self.W_ie = nn.ParameterDict()
        for i in self.ei_layers:
            n_e = sizes[i - 1]
            n_i = n_e // 4  # standard E:I ratio
            k = str(i)
            p1, p2, d, s = _parse_weight_spec(w_ee, dist, sparsity)
            w_ee_t = nn.Parameter(init_weight((n_e, n_e), d, p1, p2, s),
                                  requires_grad=False)
            p1, p2, d, s = _parse_weight_spec(w_ei, dist, sparsity)
            w_ei_t = nn.Parameter(init_weight((n_e, n_i), d, p1, p2, s),
                                  requires_grad=False)
            p1, p2, d, s = _parse_weight_spec(w_ie, dist, sparsity)
            w_ie_t = nn.Parameter(init_weight((n_i, n_e), d, p1, p2, s),
                                  requires_grad=False)
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

    def forward(self, noise_std=0.0, randomize_init=False,
                ref_mean=0.0, ref_std=0.0, ext_g=None, drive_sigma=0.0,
                input_spikes=None):
        has_ext_g = ext_g is not None
        has_input_spikes = input_spikes is not None

        if has_ext_g and ext_g.dim() == 3:
            B, device = ext_g.shape[1], ext_g.device
        elif has_input_spikes and input_spikes.dim() == 3:
            B, device = input_spikes.shape[1], input_spikes.device
        else:
            B, device = 1, (ext_g.device if has_ext_g
                            else input_spikes.device if has_input_spikes
                            else torch.device("cpu"))

        if self.signed_weights:
            W_ff = list(self.W_ff)
        else:
            W_ff = [W.clamp(min=0) for W in self.W_ff]
        g_noise = noise_std / (E_e - E_L) if noise_std > 0 else 0.0

        # Pre-compute input drive for all timesteps — CUDA only. On MPS the
        # (T, B, N_hid) intermediate inflates the autograd graph and triggers
        # a per-step .item() sync storm on backward. On CPU the big matmul
        # blows the cache.
        input_drive_all = None
        if has_input_spikes and device.type == "cuda":
            input_drive_all = input_spikes @ W_ff[0]

        # Per-layer state
        v_e, ref_e, ge_e, gi_e, s_e = {}, {}, {}, {}, {}
        v_i, ref_i, ge_i, s_i = {}, {}, {}, {}
        drive_gains = {}
        for i in range(1, self.n_layers + 1):
            n_e = self.hidden_sizes[i - 1]
            k = str(i)
            v_e[k], ref_e[k] = init_lif_state(B, n_e, device,
                                               randomize=randomize_init,
                                               ref_mean=ref_mean, ref_std=ref_std)
            ge_e[k] = init_conductance(B, n_e, device)
            s_e[k] = torch.zeros(B, n_e, device=device)
            if i in self.ei_layers:
                n_i = n_e // 4
                gi_e[k] = init_conductance(B, n_e, device)
                v_i[k], ref_i[k] = init_lif_state(B, n_i, device,
                                                   randomize=randomize_init)
                ge_i[k] = init_conductance(B, n_i, device)
                s_i[k] = torch.zeros(B, n_i, device=device)
            if drive_sigma > 0 and i == 1:
                drive_gains[k] = (1.0 + drive_sigma * torch.randn(
                    B, n_e, device=device)).clamp(min=0)

        # Output: cumulative last-hidden-layer spikes → linear decoder
        # (Same readout as SNNTorchNet — no output spiking neurons, no
        # per-model confound at the output.)
        hidden_accum = init_conductance(B, self.hidden_sizes[-1], device)

        # Pre-allocate recording buffers on GPU
        rec_buf = None
        if self.recording:
            rec_buf = {"out": torch.zeros(T_steps, B, N_OUT, device=device)}
            if has_input_spikes:
                rec_buf["input"] = torch.zeros(T_steps, B, N_IN, device=device)
            for i in range(1, self.n_layers + 1):
                n_e = self.hidden_sizes[i - 1]
                rec_buf[self._hid_key(i)] = torch.zeros(
                    T_steps, B, n_e, device=device)
                if i in self.ei_layers:
                    rec_buf[self._inh_key(i)] = torch.zeros(
                        T_steps, B, n_e // 4, device=device)
        # GPU-side spike accumulators
        n_spk_tensors = {}
        for i in range(1, self.n_layers + 1):
            n_spk_tensors[self._hid_key(i)] = torch.zeros(1, device=device)
            if i in self.ei_layers:
                n_spk_tensors[self._inh_key(i)] = torch.zeros(1, device=device)
        n_spk_tensors["out"] = torch.zeros(1, device=device)

        for t in range(T_steps):
            prev_spk = None
            for i in range(1, self.n_layers + 1):
                k = str(i)
                W = W_ff[i - 1]
                is_ei = i in self.ei_layers

                # E-I recurrent dynamics (spike then decay)
                if is_ei:
                    ge_e[k] = exp_synapse(ge_e[k], s_e[k], self.W_ee[k], decay_ampa)
                    ge_i[k] = exp_synapse(ge_i[k], s_e[k], self.W_ei[k], decay_ampa)
                    gi_e[k] = exp_synapse(gi_e[k], s_i[k], self.W_ie[k], decay_gaba)
                else:
                    ge_e[k] = ge_e[k] * decay_ampa  # just decay, no recurrence

                # Feedforward drive
                if i == 1:
                    if has_input_spikes:
                        spk_t = (input_spikes[t].unsqueeze(0)
                                 if input_spikes.dim() == 2 else input_spikes[t])
                        if input_drive_all is not None:
                            g_ext = (input_drive_all[t].unsqueeze(0)
                                     if input_drive_all.dim() == 2
                                     else input_drive_all[t])
                        else:
                            g_ext = spk_t @ W
                        if k in drive_gains:
                            g_ext = g_ext * drive_gains[k]
                        ge_e[k] = ge_e[k] + g_ext
                        if rec_buf is not None and "input" in rec_buf:
                            rec_buf["input"][t] = spk_t
                    if has_ext_g:
                        ge_e[k] = ge_e[k] + (ext_g[t].unsqueeze(0)
                                              if ext_g.dim() == 2 else ext_g[t])
                else:
                    ge_e[k] = ge_e[k] + prev_spk @ W

                # Background noise (layer 1 only)
                if g_noise > 0 and i == 1:
                    n_e = self.hidden_sizes[0]
                    ge_e[k] = ge_e[k] + (g_noise * torch.randn(
                        B, n_e, device=device)).clamp(min=0)

                # Voltage steps
                if is_ei:
                    v_e[k], s_e[k], ref_e[k] = e_step_coba(
                        v_e[k], ref_e[k], ge_e[k], gi_e[k])
                    v_i[k], s_i[k], ref_i[k] = i_step_coba(
                        v_i[k], ref_i[k], ge_i[k])
                else:
                    v_e[k], s_e[k], ref_e[k] = e_step_coba(
                        v_e[k], ref_e[k], ge_e[k])

                prev_spk = s_e[k]

                hk = self._hid_key(i)
                n_spk_tensors[hk] += s_e[k].detach().sum()
                if rec_buf is not None:
                    rec_buf[hk][t] = s_e[k]
                if is_ei:
                    ik = self._inh_key(i)
                    n_spk_tensors[ik] += s_i[k].detach().sum()
                    if rec_buf is not None:
                        rec_buf[ik][t] = s_i[k]

            # Linear decoder on cumulative last-hidden spikes
            hidden_accum = hidden_accum + prev_spk
            logits_t = hidden_accum @ W_ff[-1]
            if rec_buf is not None:
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
            rec = {k: (v.squeeze(1).cpu() if B == 1 else v.cpu())
                   for k, v in rec_buf.items()}
        self._set_meta(B, n_spk, rec, sizes)
        return logits_t


