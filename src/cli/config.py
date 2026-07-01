"""Configuration, network construction, and simulation runners.

Contains the Config dataclass, make_net, extract_weights, run_sim,
run_sim_batch, run_sim_image, and backward-compat module globals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

import models as M
from models import COBANet
from inputs import (
    make_step_drive,
)

# Default output directory when no --out-dir is given: where the oscilloscope
# snapshot figures land. Defined once and reused by cfg, build_config, and the
# CLI entrypoint so the path lives in exactly one place.
DEFAULT_ARTIFACT_ROOT = Path(__file__).parent.parent / "artifacts" / "oscilloscope"


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
    w_ee: tuple = (0.0, 0.0)
    w_ii: tuple = (0.0, 0.0)
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
    active_panels: list = field(
        default_factory=lambda: [
            "header",
            "progress",
            "sweep",
            "e_raster",
            "drive",
            "weights",
            "i_raster",
            "participation",
            "output",
            "psd",
        ]
    )
    artifact_root: str = ""

    @property
    def visible_ms(self):
        return self.sim_ms - self.burn_in_ms

    @property
    def torch_device(self):
        return torch.device(self.device)

    def apply_frame_param(self, param_name: str, value: float) -> None:
        """Apply a single scan-frame parameter override.

        Valid params: w_ei_mean, w_ie_mean, ei_strength, bias.
        Does not include tau_gaba/tau_ampa (those are M globals, handled separately).
        """
        if param_name == "w_ei_mean":
            self.w_ei = (value, self.w_ei[1])
        elif param_name == "w_ie_mean":
            self.w_ie = (value, self.w_ie[1])
        elif param_name == "ei_strength":
            self.w_ei = (value, value * 0.1)
            self.w_ie = (value * self.ei_ratio, value * self.ei_ratio * 0.1)
        elif param_name == "bias":
            self.bias = value
        else:
            raise ValueError(f"Unknown frame param: {param_name!r}")

    def sync_from_model(self, n_hid: int, n_inh: int) -> None:
        """After network construction, sync actual network size back to config.

        Called after build_net to keep n_e/n_i in sync with the actual
        network topology. Ensures config reflects what the network actually is.
        """
        self.n_e = n_hid
        self.n_i = n_inh


cfg = Config(artifact_root=str(DEFAULT_ARTIFACT_ROOT))


def setup_model_globals(hidden_sizes):
    """Initialize M module globals from hidden_sizes.

    Sets M.N_HID, M.N_INH (computed as N_HID // 4), and M.HIDDEN_SIZES.
    Call this before building/loading networks to ensure module globals are consistent.

    WHY: The models.py module uses global state (M.N_HID, M.N_INH, M.HIDDEN_SIZES)
    during forward() and weight initialization. These must be set BEFORE the network
    is built or weights are loaded, otherwise the network's internal size assumptions
    will be wrong. This function centralizes the init logic (was duplicated in 4 places).

    Args:
        hidden_sizes: List of hidden layer sizes. E.g. [256] for 1 layer, [128, 256]
                      for 2 layers. Uses the LAST element as the primary N_HID.

    Side effects:
        Mutates: M.N_HID (= hidden_sizes[-1]), M.N_INH (= N_HID // 4),
                 M.HIDDEN_SIZES (= list copy of hidden_sizes)
    """
    hidden_sizes = list(hidden_sizes) if hidden_sizes else [256]
    M.N_HID = hidden_sizes[-1]
    M.N_INH = hidden_sizes[-1] // 4
    M.HIDDEN_SIZES = list(hidden_sizes)


def save_snapshot_npz(out_path, rec, dt, n_e, n_i, display=None, primary_hid_key_fn=None, primary_inh_key_fn=None):
    """Save spike recording and metadata to NPZ file for notebook analysis.

    SINGLE SOURCE OF TRUTH: All snapshot saving across train/infer/sim paths must
    use this function (was previously duplicated in ~4 places with inconsistent field
    names and missing metadata). Ensures all snapshots have consistent structure.

    WHY: Recording dicts from different paths use different key names:
    - train.py records to 'hid', 'inh', 'input' keys
    - Multi-layer networks record to 'hid_0', 'hid_1', etc.
    - Notebooks always expect 'spk_e', 'spk_i' (excitatory/inhibitory spikes)
    This function handles all the mapping and ensures notebooks always get consistent
    field names.

    Args:
        out_path: Path to output NPZ file
        rec: Recording dict from network.spike_record with spike tensors keyed by
             layer name (e.g. 'hid', 'inh', 'hid_0', 'hid_1', 'input')
        dt: Timestep (ms) — stored as metadata
        n_e: Number of excitatory neurons — stored as metadata
        n_i: Number of inhibitory neurons — stored as metadata
        display: Optional stimulus array (ext_g_override or input_spikes tensor).
                 Used as fallback for 'input_spikes' field if rec doesn't contain it.
        primary_hid_key_fn: Function that finds the deepest hidden layer key in rec.
                           Defaults to scan.primary_hid_key (handles multi-layer).
        primary_inh_key_fn: Function that finds the deepest inhibitory layer key in rec.
                           Defaults to scan.primary_inh_key.

    Output NPZ fields:
        Metadata:
        - dt (float32): Timestep (ms)
        - n_e (int32): Number of excitatory neurons
        - n_i (int32): Number of inhibitory neurons

        Spike data (canonicalized names):
        - spk_e: Excitatory spike raster (T, n_e) — from rec[hid_key]
        - spk_i: Inhibitory spike raster (T, n_i) — from rec[inh_key]
        - input_spikes: Input stimulus (T, n_in) — from rec['input'] or display arg

        All other recorded fields (voltages, conductances, etc.):
        - v_e_<layer>, v_i_<layer>, g_e_<layer>, g_i_<layer>, etc.
          (all other keys in rec, minus hid/inh/input)
    """
    from scan import primary_hid_key as _primary_hid_key, primary_inh_key as _primary_inh_key

    if primary_hid_key_fn is None:
        primary_hid_key_fn = _primary_hid_key
    if primary_inh_key_fn is None:
        primary_inh_key_fn = _primary_inh_key

    # Start with metadata (dt, n_e, n_i are always present and consistent)
    npz_data = {
        "dt": np.float32(dt),
        "n_e": np.int32(n_e),
        "n_i": np.int32(n_i),
    }

    # Resolve spike recording keys: find the deepest/primary hidden and inhibitory layers
    # primary_hid_key returns 'hid' for single-layer, 'hid_1' for the deepest multi-layer
    hid_key = primary_hid_key_fn(rec)
    inh_key = primary_inh_key_fn(rec)

    # Save excitatory spikes under canonical name 'spk_e' (from whatever key they're in)
    if hid_key:
        spk_e = rec[hid_key]
        # Convert torch tensors to numpy; handle numpy arrays directly
        npz_data["spk_e"] = spk_e.numpy() if hasattr(spk_e, "numpy") else spk_e

    # Save inhibitory spikes under canonical name 'spk_i' or create empty array if absent
    if inh_key:
        spk_i = rec[inh_key]
        npz_data["spk_i"] = spk_i.numpy() if hasattr(spk_i, "numpy") else spk_i
    else:
        # Networks without inhibition: save empty (T, 0) array so downstream code
        # doesn't break when trying to access spk_i
        T = npz_data.get("spk_e", rec[hid_key]).shape[0] if hid_key else 0
        npz_data["spk_i"] = np.zeros((T, 0), dtype=np.float32)

    # Save all other recorded fields (voltages, conductances, etc.) under their
    # original names, except 'input' which we rename to 'input_spikes' for clarity
    for key, val in rec.items():
        if key not in (hid_key, inh_key) and val is not None:
            # 'input' key is confusing; notebooks always expect 'input_spikes'
            save_key = "input_spikes" if key == "input" else key
            npz_data[save_key] = val.numpy() if hasattr(val, "numpy") else val

    # Fallback for input stimulus: if rec doesn't have 'input' or 'input_spikes',
    # and display was passed, save it as 'input_spikes'. This handles the
    # synthetic-spikes inference mode where we don't record the input to rec.
    if display is not None:
        display_arr = display.numpy() if hasattr(display, "numpy") else display
        # Only add if not already present (rec['input'] takes precedence)
        if "input_spikes" not in npz_data:
            npz_data["input_spikes"] = display_arr

    # Write all fields to NPZ (automatic gzip compression)
    np.savez(out_path, **npz_data)


# =============================================================================
# Model Registry
# =============================================================================

# Single source of truth for the model set: name → (class, base kwargs).
# build_net (train/infer) and _build_sim_net (sim/scan/snapshot) both read it.
_MODEL_CLASSES = {
    "ping": (COBANet, {}),
}

HAS_INH = {"ping"}
IS_COBA = {"ping"}


def _build_sim_net(model_name, **kwargs):
    """Construct a network for the sim/scan/snapshot path from _MODEL_CLASSES.

    The ping path additionally wires the cfg-derived recurrent weight specs;
    the train/infer path sets those from CLI args via build_net instead.
    """
    cls, base_kwargs = _MODEL_CLASSES[model_name]
    kwargs = {**base_kwargs, **kwargs}
    if model_name == "ping":
        kwargs.update(
            w_in=(0, 0),
            w_hid=(5.1, 3.8),
            w_ee=(*cfg.w_ee, "normal", cfg.sparsity),
            w_ei=(*cfg.w_ei, "normal", cfg.sparsity),
            w_ie=(*cfg.w_ie, "normal", cfg.sparsity),
        )
    return cls(**kwargs)


LEGACY_MODEL_ALIASES: dict[str, str] = {}


def build_net(
    model_name,
    w_in=None,
    w_in_sparsity=0.0,
    w_ee=None,
    w_ei=None,
    w_ie=None,
    w_ii=None,
    ei_strength=None,
    ei_ratio=2.0,
    sparsity=0.0,
    device=None,
    randomize_init=False,
    dales_law=True,
    hidden_sizes=None,
    ei_layers=None,
    readout_mode="rate",
    trainable_w_ei=False,
    trainable_w_ie=False,
    trainable_w_ii=False,
    n_inh_per_layer=None,
):
    """Construct a network with the given config.

    Single canonical builder used by every mode. Same args produce the same
    network — no drift between train/infer/image/video/sim.

    hidden_sizes: list of hidden layer sizes (e.g. [128, 256] for 2 layers).
    w_rec: if set, enables recurrence on all layers (or rec_layers subset).
    ei_layers: which layers get E-I structure (1-indexed). Default: all.
    """
    if model_name not in _MODEL_CLASSES:
        raise ValueError(
            f"Unknown model {model_name!r}; choose from {list(_MODEL_CLASSES)}"
        )
    cls, base_kwargs = _MODEL_CLASSES[model_name]
    kwargs = {**base_kwargs}
    kwargs["readout_mode"] = readout_mode

    # Set module-level hidden sizes
    if hidden_sizes is not None:
        M.HIDDEN_SIZES = list(hidden_sizes)
        M.N_HID = hidden_sizes[-1]
        kwargs["hidden_sizes"] = list(hidden_sizes)

    kwargs["dales_law"] = dales_law
    if ei_layers is not None:
        kwargs["ei_layers"] = set(ei_layers)
    if trainable_w_ei:
        kwargs["trainable_w_ei"] = True
    if trainable_w_ie:
        kwargs["trainable_w_ie"] = True
    if trainable_w_ii:
        kwargs["trainable_w_ii"] = True
    if w_in is not None:
        kwargs["w_in"] = (*w_in, "normal", w_in_sparsity)
    if w_ee is not None:
        kwargs["w_ee"] = (*w_ee, "normal", sparsity)
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
    if w_ii is not None:
        kwargs["w_ii"] = (*w_ii, "normal", sparsity)
    if w_ei is None and w_ie is None and ei_strength is None and sparsity > 0:
        kwargs.setdefault("sparsity", sparsity)
    if n_inh_per_layer is not None:
        kwargs["n_inh_per_layer"] = dict(n_inh_per_layer)
    net = cls(**kwargs)
    if device is not None:
        net = net.to(device)
    if randomize_init and hasattr(net, "randomize_init"):
        setattr(net, "randomize_init", True)
    return net


# =============================================================================
# Backward-compat aliases (read from cfg, mutated in-place by CLI / scan fns)
# =============================================================================









# =============================================================================
# Device
# =============================================================================




# =============================================================================
# Simulation helpers
# =============================================================================


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
                weights[f"W_ff_{i + 1}"] = _extract(w)

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

    # Honour an explicit N_I that differs from the default n_e//4 (e.g. the
    # equal-fan-in V&S configs where K_I = K_E). Single E-I layer → index 1.
    # Default (n_i == n_e//4) leaves n_inh_per_layer=None, so behaviour is
    # byte-identical for every existing caller.
    n_inh_per_layer = (
        {1: cfg_obj.n_i} if cfg_obj.n_i != cfg_obj.n_e // 4 else None
    )
    net = build_net(
        model_name,
        w_in=w_in_pair,
        w_in_sparsity=w_in_sparsity,
        w_ee=cfg_obj.w_ee,
        w_ei=cfg_obj.w_ei,
        w_ie=cfg_obj.w_ie,
        w_ii=cfg_obj.w_ii if cfg_obj.w_ii != (0.0, 0.0) else None,
        sparsity=cfg_obj.sparsity,
        device=cfg_obj.torch_device,
        hidden_sizes=[cfg_obj.n_e],
        n_inh_per_layer=n_inh_per_layer,
    )
    net.recording = True
    return net


# Backward compat alias
make_ping_net = make_net


def run_sim(
    dt,
    t_e_ping,
    *,
    ext_g_override=None,
    model_name="ping",
    t_e_async=None,
    input_spikes=None,
):
    """Run a single simulation with any registered model.

    Returns (rec, ext_g_or_spikes_numpy, weights).
    """
    if t_e_async is None:
        t_e_async = cfg.t_e_async
    M.N_HID = cfg.n_e
    M.N_INH = cfg.n_i
    T_steps = int(cfg.sim_ms / dt)

    if input_spikes is not None:
        ext_g_tensor = None
        input_spikes = input_spikes.to(cfg.torch_device)
        M.T_steps = min(M.T_steps, len(input_spikes))
    elif ext_g_override is not None:
        ext_g_tensor = (
            ext_g_override.clone().detach()
            if isinstance(ext_g_override, torch.Tensor)
            else torch.tensor(ext_g_override, dtype=torch.float32)
        ).to(cfg.torch_device)
        M.T_steps = min(M.T_steps, len(ext_g_tensor))
    else:
        ext_g_tensor, _ = make_step_drive(
            cfg.n_e,
            T_steps,
            dt,
            t_e_async,
            t_e_ping,
            cfg.step_on_ms,
            cfg.step_off_ms,
            cfg.sigma_e,
            cfg.noise_sigma,
            cfg.noise_tau,
            cfg.seed,
        )
        ext_g_tensor = ext_g_tensor.to(cfg.torch_device)
        M.T_steps = T_steps

    torch.manual_seed(cfg.seed)
    net = _build_sim_net(model_name, hidden_sizes=[M.N_HID])
    net.to(cfg.torch_device)
    net.recording = True

    with torch.no_grad():
        if input_spikes is not None:
            net.forward(input_spikes=input_spikes)
        else:
            net.forward(ext_g=ext_g_tensor)

    rec = _extract_records(net)
    weights = extract_weights(net)

    if input_spikes is not None:
        display = input_spikes.cpu().numpy()
    else:
        assert ext_g_tensor is not None
        display = ext_g_tensor.cpu().numpy()
    return rec, display, weights


def run_sim_batch(dt, ext_g_list, w_hid=(5.1, 3.8), chunk_size=100, model_name="ping"):
    """Run multiple simulations in one batched forward pass.

    Returns list of (rec, ext_g_raw) tuples, one per sim.
    """
    M.N_HID = cfg.n_e
    M.N_INH = cfg.n_i
    M.T_steps = int(C.SIM_MS / dt)

    results = []
    for start in range(0, len(ext_g_list), chunk_size):
        chunk = ext_g_list[start : start + chunk_size]
        B = len(chunk)
        T = len(chunk[0])

        ext_g_batch = torch.tensor(np.stack(chunk, axis=1), dtype=torch.float32).to(
            cfg.torch_device
        )
        M.T_steps = min(M.T_steps, T)

        torch.manual_seed(cfg.seed)
        net = _build_sim_net(model_name, hidden_sizes=[M.N_HID])
        net.to(cfg.torch_device)
        net.recording = True

        with torch.no_grad():
            kwargs: dict = {"ext_g": ext_g_batch}
            if model_name in HAS_INH:
                kwargs["randomize_init"] = False
            net.forward(**kwargs)

        rec_stacked = {}
        for k, v in net.spike_record.items():
            if isinstance(v, list):
                rec_stacked[k] = torch.stack(v).cpu().numpy()  # ty: ignore[invalid-argument-type]
            else:
                rec_stacked[k] = v.cpu().numpy()

        for i in range(B):
            rec = {k: v[:, i, :] for k, v in rec_stacked.items()}
            results.append((rec, chunk[i]))

    return results


def run_sim_image(dt, image, model_name="ping", load_weights=None):
    """Run a simulation with image input using current config.

    Returns (rec, predicted_class, net).
    """
    M.N_IN = image.shape[0]
    # dt-dependent constants are computed locally in model.forward

    w_in = (*cfg.w_in_spikes, "normal", cfg.w_in_sparsity)
    net = make_net(cfg, w_in=w_in, model_name=model_name)
    if load_weights is not None:
        state = torch.load(load_weights, map_location=cfg.torch_device)
        net.load_state_dict(state, strict=False)
        net.eval()

    # Pre-encode image as Poisson spikes — same canonical path as train/infer
    img_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(cfg.torch_device)
    pixels = img_tensor.clamp(0, 1)
    p = M.max_rate_hz * dt / 1000.0
    input_spikes = (
        (torch.rand(M.T_steps, 1, M.N_IN, device=cfg.torch_device) < pixels.unsqueeze(0) * p)
        .float()
        .squeeze(1)
    )

    with torch.no_grad():
        kwargs: dict = {"input_spikes": input_spikes}
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
    M.N_HID = cfg.n_e
    M.N_INH = cfg.n_i
    T_steps = int(cfg.sim_ms / dt)

    ext_g_tensor, _ = make_step_drive(
        cfg.n_e,
        T_steps,
        dt,
        t_e_async,
        t_e_ping,
        cfg.step_on_ms,
        cfg.step_off_ms,
        cfg.sigma_e,
        cfg.noise_sigma,
        cfg.noise_tau,
        cfg.seed,
        noise_seed=noise_seed,
    )
    ext_g_tensor = ext_g_tensor.to(cfg.torch_device)

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
    c = Config()
    if hasattr(args, "out_dir") and args.out_dir is not None:
        c.artifact_root = args.out_dir
    else:
        c.artifact_root = str(DEFAULT_ARTIFACT_ROOT)
    c.fps = getattr(args, "frame_rate", 10)
    if hasattr(args, "n_hidden") and args.n_hidden is not None:
        # args.n_hidden may be an int or a list (multi-layer). For legacy Config,
        # use the last hidden size (the E-I / output-feeding layer).
        n_e = args.n_hidden[-1] if isinstance(args.n_hidden, list) else args.n_hidden
        c.n_e = n_e
        c.n_i = n_e // 4
    if hasattr(args, "device") and args.device is not None:
        c.device = args.device
    c.raster_mode = getattr(args, "raster", "scatter")
    if hasattr(args, "drive") and args.drive is not None:
        c.t_e_async = args.drive
    c.ei_ratio = getattr(args, "ei_ratio", 2.0)
    ei_strength = getattr(args, "ei_strength", None)
    if ei_strength is not None:
        s = ei_strength
        c.w_ei = (s, s * 0.1)
        c.w_ie = (s * c.ei_ratio, s * c.ei_ratio * 0.1)
    if hasattr(args, "w_ei") and args.w_ei is not None:
        c.w_ei = tuple(args.w_ei)
    if hasattr(args, "w_ie") and args.w_ie is not None:
        c.w_ie = tuple(args.w_ie)
    if hasattr(args, "w_ee") and args.w_ee is not None:
        c.w_ee = tuple(args.w_ee)
    if hasattr(args, "burn_in") and args.burn_in is not None:
        c.burn_in_ms = args.burn_in
    if hasattr(args, "w_in") and args.w_in is not None:
        w = args.w_in
        if len(w) == 1:
            w = [w[0], w[0] * 0.1]  # std = 10% of mean
        c.w_in_spikes = tuple(w[:2])
    input_mode = getattr(args, "input", "synthetic-spikes")
    if hasattr(args, "n_input") and args.n_input is not None:
        M.N_IN = args.n_input
    elif input_mode == "synthetic-spikes":
        M.N_IN = c.n_e
    sparsity = getattr(args, "sparsity", None)
    if sparsity is not None:
        c.sparsity = sparsity
    w_in_sparsity = getattr(args, "w_in_sparsity", None)
    if w_in_sparsity is not None:
        c.w_in_sparsity = w_in_sparsity
    bias = getattr(args, "bias", None)
    if bias is not None:
        c.bias = bias
    # Honour --t-ms by syncing cfg.sim_ms; otherwise the scan path sizes the
    # input array from cfg.SIM_MS (default 600) while the network loop uses
    # args.t_ms, producing an off-by-one IndexError when they disagree.
    t_ms = getattr(args, "t_ms", None)
    if t_ms is not None:
        c.sim_ms = float(t_ms)
    # For image input, --input-rate sets the max Poisson encoding rate
    if input_mode == "dataset":
        spike_rate = getattr(args, "spike_rate", M.max_rate_hz)
        M.max_rate_hz = spike_rate
        M.p_scale = M.max_rate_hz * M.dt / 1000.0
    # Sync the module-level aliases here so callers can't forget to.
    _sync_globals_from_cfg(c)
    return c


def _sync_globals_from_cfg(c):
    """Install a Config as the module-wide source of truth.

    Every alias (C.N_E, C.W_EI, C.STEP_ON_MS, …) resolves to this cfg via
    the module __getattr__ below, so there is nothing else to mirror.
    """
    global cfg
    cfg = c
# Every config alias (C.N_E, C.W_EI, C.STEP_ON_MS, …) resolves to the live
# Config — one source of truth, no mirror globals, no sync step.
_CFG_ALIASES = {
    "BURN_IN_MS": "burn_in_ms",
    "DEVICE": "torch_device",
    "EI_RATIO": "ei_ratio",
    "FPS": "fps",
    "NOISE_SIGMA": "noise_sigma",
    "NOISE_TAU": "noise_tau",
    "SEED": "seed",
    "SIGMA_E": "sigma_e",
    "SIM_MS": "sim_ms",
    "SPIKE_RATE_BASE": "spike_rate_base",
    "STEP_OFF_MS": "step_off_ms",
    "STEP_ON_MS": "step_on_ms",
    "T_E_ASYNC_DEFAULT": "t_e_async",
    "W_IN_SPARSITY": "w_in_sparsity",
    "W_IN_SPIKES": "w_in_spikes",
    "N_E": "n_e",
    "N_I": "n_i",
    "W_EI": "w_ei",
    "W_IE": "w_ie",
    "SPARSITY": "sparsity",
    "BIAS": "bias",
}


def __getattr__(name):
    field = _CFG_ALIASES.get(name)
    if field is not None:
        return getattr(cfg, field)
    if name == "ARTIFACT_ROOT":
        return Path(cfg.artifact_root)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")