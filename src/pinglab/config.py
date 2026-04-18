"""Configuration, network construction, and simulation runners.

Contains the Config dataclass, make_net, extract_weights, run_sim,
run_sim_batch, run_sim_image, and backward-compat module globals.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/pinglab/ is first on sys.path
_pkg_dir = str(Path(__file__).parent)
if _pkg_dir in sys.path:
    sys.path.remove(_pkg_dir)
sys.path.insert(0, _pkg_dir)

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

import models as M
from models import PINGNet, SNNTorchNet, SNNTorchLibraryNet
from inputs import (
    make_spike_drive,
    patch_dt as _patch_dt,
    make_step_drive,
    make_reference_noise,
    make_step_drive_from_ref,
    DT_CAL,
)


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    n_e: int = 1024
    n_i: int = 256
    fps: int = 60
    seed: int = 42
    sim_ms: float = 600.0
    burn_in_ms: float = 100.0
    step_on_ms: float = 200.0
    step_off_ms: float = 300.0
    t_e_async: float = 0.0006
    sigma_e: float = 0.05
    w_ei: tuple = (0.5, 0.05)
    w_ie: tuple = (1.0, 0.1)
    sparsity: float = 0.2
    noise_sigma: float = 0.001
    noise_tau: float = 3.0
    spike_rate_base: float = 10.0
    w_in_spikes: tuple = (0.3, 0.06)
    w_in_sparsity: float = 0.95
    bias: float = 0.0002
    ei_ratio: float = 2.0
    device: str = "cpu"
    raster_mode: str = "scatter"
    active_panels: list = field(default_factory=lambda: [
        "header", "progress", "sweep", "e_raster",
        "drive", "weights", "i_raster", "participation",
        "output", "psd"])
    artifact_root: str = ""

    @property
    def visible_ms(self):
        return self.sim_ms - self.burn_in_ms

    @property
    def torch_device(self):
        return torch.device(self.device)


cfg = Config(artifact_root=str(Path(__file__).parent.parent / "artifacts" / "oscilloscope"))


# =============================================================================
# Model Registry
# =============================================================================

MODEL_REGISTRY = {
    "ping":                 lambda **kw: PINGNet(w_in=(0, 0), w_hid=(5.1, 3.8),
                                                  w_ei=(*cfg.w_ei, "normal", cfg.sparsity),
                                                  w_ie=(*cfg.w_ie, "normal", cfg.sparsity),
                                                  **kw),
    "snntorch-clone":       lambda **kw: SNNTorchNet(w_in=(0, 0), w_hid=(0, 0.1), **kw),
    "cuba":                 lambda **kw: SNNTorchNet(discretisation="continuous",
                                                      w_in=(0, 0), w_hid=(0, 0.1), **kw),
    "cuba-exp":             lambda **kw: SNNTorchNet(discretisation="continuous",
                                                      exponential_synapse=True,
                                                      w_in=(0, 0), w_hid=(0, 0.1), **kw),
    "cuba-exp-hard":        lambda **kw: SNNTorchNet(discretisation="continuous",
                                                      exponential_synapse=True,
                                                      reset_mode="zero",
                                                      w_in=(0, 0), w_hid=(0, 0.1), **kw),
    "cuba-exp-hard-refrac": lambda **kw: SNNTorchNet(discretisation="continuous",
                                                      exponential_synapse=True,
                                                      reset_mode="zero",
                                                      ref_ms=2.0,
                                                      w_in=(0, 0), w_hid=(0, 0.1), **kw),
    "snntorch-library":     lambda **kw: SNNTorchLibraryNet(**kw),
}

HAS_INH = {"ping"}
IS_COBA = {"ping"}

# All CUBA-family variants (shared SNNTorchNet class, distinct discretisation
# / synaptic / reset / refractory settings). Used by build_net guards.
# The headline 5-model ladder is {snntorch-clone, cuba, cuba-exp, coba, ping};
# cuba-exp-hard and cuba-exp-hard-refrac are retained in the registry only
# to load legacy artifacts (they are no longer dispatched by any experiment).
CUBA_MODELS = {
    "snntorch-clone",
    "snntorch-library",
    "cuba",
    "cuba-exp",
    "cuba-exp-hard",           # kept in registry for legacy-artifact loading
    "cuba-exp-hard-refrac",    # kept in registry for legacy-artifact loading
}

# Headline ladder: the 5 models that appear in the main dt-sweep figure and
# the ablation waterfall. snntorch-library is a parity reference for snntorch-clone
# and deliberately *not* in the ladder. cuba-exp-hard and cuba-exp-hard-refrac
# stay in MODEL_REGISTRY for loading legacy artifacts but are not dispatched.
HEADLINE_LADDER = [
    "snntorch-clone",
    "cuba",
    "cuba-exp",
    "coba",
    "ping",
]

_MODEL_CLASSES = {
    "ping":                 (PINGNet,     {}),
    "snntorch-clone":       (SNNTorchNet, {}),
    "cuba":                 (SNNTorchNet, {"discretisation": "continuous"}),
    "cuba-exp":             (SNNTorchNet, {"discretisation": "continuous",
                                            "exponential_synapse": True}),
    "cuba-exp-hard":        (SNNTorchNet, {"discretisation": "continuous",
                                            "exponential_synapse": True,
                                            "reset_mode": "zero"}),
    "cuba-exp-hard-refrac": (SNNTorchNet, {"discretisation": "continuous",
                                            "exponential_synapse": True,
                                            "reset_mode": "zero",
                                            "ref_ms": 2.0}),
    "snntorch-library":     (SNNTorchLibraryNet, {}),
}

# Back-compat aliases: older config.json files record legacy model names.
# Resolved to current primary key before lookup in MODEL_REGISTRY / build_net.
LEGACY_MODEL_ALIASES = {
    "snntorch":           "snntorch-clone",   # renamed 2026-04-18
    "snntorch-canonical": "snntorch-clone",   # renamed 2026-04-17
    "snntorch-exp":       "snntorch-clone",   # exp removed; loads as canonical
}


def build_net(model_name, w_in=None, w_in_sparsity=0.0,
              w_ei=None, w_ie=None,
              ei_strength=None, ei_ratio=2.0, sparsity=0.0,
              device=None, randomize_init=False,
              kaiming_init=False, dales_law=True,
              w_rec=None, hidden_sizes=None,
              rec_layers=None, ei_layers=None):
    """Construct a network with the given config.

    Single canonical builder used by every mode. Same args produce the same
    network — no drift between train/infer/image/video/sim.

    hidden_sizes: list of hidden layer sizes (e.g. [128, 256] for 2 layers).
    w_rec: if set, enables recurrence on all layers (or rec_layers subset).
    ei_layers: which layers get E-I structure (1-indexed). Default: all.
    """
    model_name = LEGACY_MODEL_ALIASES.get(model_name, model_name)
    if model_name not in _MODEL_CLASSES:
        raise ValueError(f"Unknown model {model_name!r}; "
                         f"choose from {list(_MODEL_CLASSES)}")
    cls, base_kwargs = _MODEL_CLASSES[model_name]
    kwargs = {**base_kwargs}

    # Set module-level hidden sizes
    if hidden_sizes is not None:
        M.HIDDEN_SIZES = list(hidden_sizes)
        M.N_HID = hidden_sizes[-1]
        kwargs["hidden_sizes"] = list(hidden_sizes)

    if model_name in CUBA_MODELS or model_name == "ping":
        kwargs["dales_law"] = dales_law
    # Recurrence: inferred from w_rec being set
    if w_rec is not None and model_name in CUBA_MODELS:
        kwargs["w_rec"] = (*w_rec, "signed_normal", sparsity)
        if rec_layers is not None:
            kwargs["rec_layers"] = set(rec_layers)
    # E-I layer selection (PING)
    if ei_layers is not None and model_name in HAS_INH:
        kwargs["ei_layers"] = set(ei_layers)
    if kaiming_init and model_name in CUBA_MODELS:
        kwargs["tutorial_mode"] = True
    elif w_in is not None:
        kwargs["w_in"] = (*w_in, "normal", w_in_sparsity)
    if model_name in HAS_INH:
        if w_ei is not None:
            kwargs["w_ei"] = (*w_ei, "normal", sparsity)
        elif ei_strength is not None:
            s = ei_strength
            kwargs["w_ei"] = (s, s * 0.1, "normal", sparsity)
        if w_ie is not None:
            kwargs["w_ie"] = (*w_ie, "normal", sparsity)
        elif ei_strength is not None:
            s = ei_strength
            kwargs["w_ie"] = (s * ei_ratio, s * ei_ratio * 0.1, "normal", sparsity)
        if w_ei is None and w_ie is None and ei_strength is None and sparsity > 0:
            kwargs.setdefault("sparsity", sparsity)
    net = cls(**kwargs)
    if device is not None:
        net = net.to(device)
    if randomize_init and hasattr(net, "randomize_init"):
        net.randomize_init = True
    return net


# =============================================================================
# Backward-compat aliases (read from cfg, mutated in-place by CLI / scan fns)
# =============================================================================

N_E = cfg.n_e
N_I = cfg.n_i
FPS = cfg.fps
SEED = cfg.seed
SIM_MS = cfg.sim_ms
BURN_IN_MS = cfg.burn_in_ms
VISIBLE_MS = cfg.visible_ms

T_E_ASYNC_DEFAULT = cfg.t_e_async
SIGMA_E = cfg.sigma_e
STEP_ON_MS = cfg.step_on_ms
STEP_OFF_MS = cfg.step_off_ms

W_EI = cfg.w_ei
W_IE = cfg.w_ie
SPARSITY = cfg.sparsity

NOISE_SIGMA = cfg.noise_sigma
NOISE_TAU = cfg.noise_tau

SPIKE_RATE_BASE = cfg.spike_rate_base

ARTIFACT_ROOT = Path(cfg.artifact_root)

W_IN_SPIKES = cfg.w_in_spikes
W_IN_SPARSITY = cfg.w_in_sparsity
BIAS = cfg.bias
EI_RATIO = cfg.ei_ratio


# =============================================================================
# Device
# =============================================================================

def _get_device():
    """Pick the best available device.

    Defaults to CPU -- MPS/CUDA add overhead for the sequential timestep loop
    with per-step recording. Use --device mps/cuda to override.
    """
    return torch.device("cpu")

DEVICE = _get_device()


# =============================================================================
# Simulation helpers
# =============================================================================

def patch_dt(dt_new):
    """Apply a new dt. Uses M.T_ms (set by dispatch from --t-ms)."""
    _patch_dt(dt_new, M.T_ms)


def _extract_records(net):
    """Convert a network's spike_record to a dict of numpy arrays (on CPU)."""
    rec = {}
    for k, v in net.spike_record.items():
        if isinstance(v, list):
            rec[k] = torch.stack(v).cpu().numpy()
        else:
            rec[k] = v.cpu().numpy() if hasattr(v, "cpu") else np.array(v)
    return rec


def extract_weights(net):
    """Extract weight arrays from a network for display.

    Handles both old-style named parameters (W_in, W_hid) and new-style
    ParameterList/Dict (W_ff, W_rec, W_ee, W_ei, W_ie).
    """
    weights = {}

    def _extract(w):
        if isinstance(w, nn.Parameter):
            return w.data.cpu().numpy().ravel()
        return w.cpu().numpy().ravel()

    # New-style: ParameterList W_ff
    if hasattr(net, "W_ff"):
        for i, w in enumerate(net.W_ff):
            if i == 0:
                weights["W_in"] = _extract(w)
            elif i == len(net.W_ff) - 1:
                weights["W_out"] = _extract(w)
            else:
                weights[f"W_ff_{i+1}"] = _extract(w)

    # New-style: ParameterDict W_rec, W_ee, W_ei, W_ie
    for dict_name in ["W_rec", "W_ee", "W_ei", "W_ie"]:
        d = getattr(net, dict_name, None)
        if d is not None and isinstance(d, nn.ParameterDict) and len(d) > 0:
            if len(d) == 1:
                weights[dict_name] = _extract(list(d.values())[0])
            else:
                for k, w in d.items():
                    weights[f"{dict_name}_{k}"] = _extract(w)

    # Legacy fallback: old-style named params
    if not weights:
        for name in ["W_in", "W_hid", "W_rec", "W_ee", "W_ei", "W_ie"]:
            if hasattr(net, name):
                w = getattr(net, name)
                weights[name] = _extract(w)

    return weights


def make_net(cfg_obj, w_in=None, w_hid=(5.1, 3.8), model_name="ping"):
    """Build a network from a Config object — thin wrapper over build_net.

    Maps cfg fields to build_net args, then sets recording=True for the
    legacy generate_* / scan code paths that need spike recording.

    The w_in arg overrides cfg.w_in_spikes — supports the legacy
    (mean, std, dist, sparsity) 4-tuple form used by callers.
    """
    M.N_HID = cfg_obj.n_e
    M.N_INH = cfg_obj.n_i
    torch.manual_seed(cfg_obj.seed)

    # Resolve w_in: legacy 4-tuple (mean, std, dist, sparsity) → (mean, std), sparsity
    if w_in is None:
        w_in_pair = None
        w_in_sparsity = cfg_obj.w_in_sparsity
    elif len(w_in) >= 4:
        w_in_pair = (w_in[0], w_in[1])
        w_in_sparsity = w_in[3]
    else:
        w_in_pair = (w_in[0], w_in[1])
        w_in_sparsity = cfg_obj.w_in_sparsity

    net = build_net(
        model_name,
        w_in=w_in_pair,
        w_in_sparsity=w_in_sparsity,
        w_ei=cfg_obj.w_ei,
        w_ie=cfg_obj.w_ie,
        sparsity=cfg_obj.sparsity,
        device=cfg_obj.torch_device,
        hidden_sizes=[cfg_obj.n_e],
    )
    net.recording = True
    return net


# Backward compat alias
make_ping_net = make_net


def run_sim(dt, t_e_ping, *, ext_g_override=None, model_name="ping",
            t_e_async=None, input_spikes=None):
    """Run a single simulation with any registered model.

    Returns (rec, ext_g_or_spikes_numpy, weights).
    """
    if t_e_async is None:
        t_e_async = T_E_ASYNC_DEFAULT
    M.N_HID = N_E
    M.N_INH = N_I
    patch_dt(dt)
    T_steps = M.T_steps

    if input_spikes is not None:
        ext_g_tensor = None
        input_spikes = input_spikes.to(DEVICE)
        M.T_steps = min(M.T_steps, len(input_spikes))
    elif ext_g_override is not None:
        ext_g_tensor = (ext_g_override.clone().detach()
                        if isinstance(ext_g_override, torch.Tensor)
                        else torch.tensor(ext_g_override, dtype=torch.float32)
                        ).to(DEVICE)
        M.T_steps = min(M.T_steps, len(ext_g_tensor))
    else:
        ext_g_tensor, _ = make_step_drive(
            N_E, T_steps, dt, t_e_async, t_e_ping,
            STEP_ON_MS, STEP_OFF_MS, SIGMA_E, NOISE_SIGMA, NOISE_TAU, SEED,
        )
        ext_g_tensor = ext_g_tensor.to(DEVICE)

    torch.manual_seed(SEED)
    net = MODEL_REGISTRY[model_name](hidden_sizes=[M.N_HID])
    net.to(DEVICE)
    net.recording = True

    with torch.no_grad():
        if input_spikes is not None:
            net.forward(input_spikes=input_spikes)
        else:
            net.forward(ext_g=ext_g_tensor)

    rec = _extract_records(net)
    weights = extract_weights(net)

    display = (input_spikes.cpu().numpy() if input_spikes is not None
               else ext_g_tensor.cpu().numpy())
    return rec, display, weights


def run_sim_batch(dt, ext_g_list, w_hid=(5.1, 3.8), chunk_size=100,
                  model_name="ping"):
    """Run multiple simulations in one batched forward pass.

    Returns list of (rec, ext_g_raw) tuples, one per sim.
    """
    M.N_HID = N_E
    M.N_INH = N_I
    patch_dt(dt)

    results = []
    for start in range(0, len(ext_g_list), chunk_size):
        chunk = ext_g_list[start:start + chunk_size]
        B = len(chunk)
        T = len(chunk[0])

        ext_g_batch = torch.tensor(np.stack(chunk, axis=1),
                                    dtype=torch.float32).to(DEVICE)
        M.T_steps = min(M.T_steps, T)

        torch.manual_seed(SEED)
        net = MODEL_REGISTRY[model_name](hidden_sizes=[M.N_HID])
        net.to(DEVICE)
        net.recording = True

        with torch.no_grad():
            net.forward(ext_g=ext_g_batch, randomize_init=False)

        rec_stacked = {}
        for k, v in net.spike_record.items():
            if isinstance(v, list):
                rec_stacked[k] = torch.stack(v).cpu().numpy()
            else:
                rec_stacked[k] = v.cpu().numpy()

        for i in range(B):
            rec = {k: v[:, i, :] for k, v in rec_stacked.items()}
            results.append((rec, chunk[i]))

    return results


def run_sim_image(dt, image, model_name="ping", load_weights=None,
                  w_in_overdrive=1.0):
    """Run a simulation with image input using current config.

    Returns (rec, predicted_class, net).
    """
    M.N_IN = image.shape[0]
    patch_dt(dt)

    w_in = (*cfg.w_in_spikes, "normal", cfg.w_in_sparsity)
    net = make_net(cfg, w_in=w_in, model_name=model_name)
    if load_weights is not None:
        state = torch.load(load_weights, map_location=DEVICE)
        net.load_state_dict(state, strict=False)
        net.eval()
    if w_in_overdrive != 1.0:
        # Scale the input projection to amplify drive without retraining
        with torch.no_grad():
            if hasattr(net, "W_ff") and len(net.W_ff) > 0:
                net.W_ff[0].mul_(w_in_overdrive)
            elif hasattr(net, "W_in"):
                net.W_in.mul_(w_in_overdrive)

    # Pre-encode image as Poisson spikes — same canonical path as train/infer
    img_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    pixels = img_tensor.clamp(0, 1)
    p = M.max_rate_hz * dt / 1000.0
    input_spikes = (torch.rand(M.T_steps, 1, M.N_IN, device=DEVICE)
                    < pixels.unsqueeze(0) * p).float().squeeze(1)

    with torch.no_grad():
        kwargs = {"input_spikes": input_spikes}
        if model_name in HAS_INH:
            kwargs["randomize_init"] = True
        out = net.forward(**kwargs)

    rec = _extract_records(net)
    pred = int(out.argmax(dim=1).cpu().item())
    return rec, pred, net


def _run_sim_with_net(net, dt, t_e_ping, t_e_async, noise_seed=None):
    """Run a simulation reusing an existing network (resets state each call).

    Returns (rec, ext_g_numpy, weights_dict).
    """
    M.N_HID = N_E
    M.N_INH = N_I
    patch_dt(dt)
    T_steps = M.T_steps

    ext_g_tensor, _ = make_step_drive(
        N_E, T_steps, dt, t_e_async, t_e_ping,
        STEP_ON_MS, STEP_OFF_MS, SIGMA_E, NOISE_SIGMA, NOISE_TAU, SEED,
        noise_seed=noise_seed,
    )
    ext_g_tensor = ext_g_tensor.to(DEVICE)

    net.recording = True
    for attr in ["rec_hid", "rec_inh", "rec_out", "rec_in"]:
        if hasattr(net, attr):
            setattr(net, attr, [])

    with torch.no_grad():
        net.forward(ext_g=ext_g_tensor)

    rec = _extract_records(net)
    weights = extract_weights(net)

    return rec, ext_g_tensor.cpu().numpy(), weights


# =============================================================================
# Config builder + globals sync
# =============================================================================

def build_config(args):
    """Build Config from CLI args."""
    from plot import LAYOUT_PRESETS  # late import to avoid circular
    c = Config()
    if hasattr(args, 'out_dir') and args.out_dir is not None:
        c.artifact_root = args.out_dir
    else:
        c.artifact_root = str(Path(__file__).parent.parent / "artifacts" / "oscilloscope")
    c.fps = getattr(args, 'frame_rate', 10)
    if hasattr(args, 'n_hidden') and args.n_hidden is not None:
        # args.n_hidden may be an int or a list (multi-layer). For legacy Config,
        # use the last hidden size (the E-I / output-feeding layer).
        n_e = args.n_hidden[-1] if isinstance(args.n_hidden, list) else args.n_hidden
        c.n_e = n_e
        c.n_i = n_e // 4
    if hasattr(args, 'device') and args.device is not None:
        c.device = args.device
    c.raster_mode = getattr(args, 'raster', 'scatter')
    if hasattr(args, 'panels') and args.panels is not None:
        c.active_panels = [p.strip() for p in args.panels.split(",")]
    else:
        c.active_panels = list(LAYOUT_PRESETS[getattr(args, 'layout', 'full')])
    if hasattr(args, 'drive') and args.drive is not None:
        c.t_e_async = args.drive
    c.ei_ratio = getattr(args, 'ei_ratio', 2.0)
    ei_strength = getattr(args, 'ei_strength', None)
    if ei_strength is not None:
        s = ei_strength
        c.w_ei = (s, s * 0.1)
        c.w_ie = (s * c.ei_ratio, s * c.ei_ratio * 0.1)
    if hasattr(args, 'w_ei') and args.w_ei is not None:
        c.w_ei = tuple(args.w_ei)
    if hasattr(args, 'w_ie') and args.w_ie is not None:
        c.w_ie = tuple(args.w_ie)
    if hasattr(args, 'w_in') and args.w_in is not None:
        w = args.w_in
        if len(w) == 1:
            w = [w[0], w[0] * 0.1]  # std = 10% of mean
        c.w_in_spikes = tuple(w[:2])
    input_mode = getattr(args, 'input', 'synthetic-spikes')
    if hasattr(args, 'n_input') and args.n_input is not None:
        M.N_IN = args.n_input
    elif input_mode == "synthetic-spikes":
        M.N_IN = c.n_e
    sparsity = getattr(args, 'sparsity', None)
    if sparsity is not None:
        c.sparsity = sparsity
    w_in_sparsity = getattr(args, 'w_in_sparsity', None)
    if w_in_sparsity is not None:
        c.w_in_sparsity = w_in_sparsity
    bias = getattr(args, 'bias', None)
    if bias is not None:
        c.bias = bias
    # For image input, --input-rate sets the max Poisson encoding rate
    if input_mode == "dataset":
        spike_rate = getattr(args, 'spike_rate', M.max_rate_hz)
        M.max_rate_hz = spike_rate
        M.p_scale = M.max_rate_hz * M.dt / 1000.0
    return c


def _sync_globals_from_cfg(c):
    """Update module-level backward-compat aliases from a Config object."""
    global cfg, N_E, N_I, FPS, SEED, SIM_MS, BURN_IN_MS, VISIBLE_MS
    global T_E_ASYNC_DEFAULT, SIGMA_E, STEP_ON_MS, STEP_OFF_MS
    global W_EI, W_IE, SPARSITY, NOISE_SIGMA, NOISE_TAU
    global SPIKE_RATE_BASE, ARTIFACT_ROOT, DEVICE, EI_RATIO
    global W_IN_SPIKES, W_IN_SPARSITY, BIAS, OUT_NAME
    cfg = c
    N_E = c.n_e
    N_I = c.n_i
    FPS = c.fps
    SEED = c.seed
    SIM_MS = c.sim_ms
    BURN_IN_MS = c.burn_in_ms
    VISIBLE_MS = c.visible_ms
    T_E_ASYNC_DEFAULT = c.t_e_async
    SIGMA_E = c.sigma_e
    STEP_ON_MS = c.step_on_ms
    STEP_OFF_MS = c.step_off_ms
    W_EI = c.w_ei
    W_IE = c.w_ie
    SPARSITY = c.sparsity
    NOISE_SIGMA = c.noise_sigma
    NOISE_TAU = c.noise_tau
    SPIKE_RATE_BASE = c.spike_rate_base
    ARTIFACT_ROOT = Path(c.artifact_root)
    DEVICE = c.torch_device
    W_IN_SPIKES = c.w_in_spikes
    W_IN_SPARSITY = c.w_in_sparsity
    BIAS = c.bias
    EI_RATIO = c.ei_ratio
