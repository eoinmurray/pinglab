"""Scan-mode utilities (video generation code removed).

Provides shared utilities for scanning parameters and recording-key resolution.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

import models as M
from config import W_EI, W_IE

log = logging.getLogger("cli")


# ── Shared utilities (imported by train/infer/__main__) ──────────
def _auto_device() -> torch.device:
    """Auto-detect the fastest available compute device.

    Priority: CUDA (Nvidia GPU) > MPS (Apple Silicon) > CPU.

    Called when the user doesn't explicitly set --device on the CLI. This provides
    sensible defaults: use GPU if available (training is ~3x faster), fall back to
    CPU on machines without accelerators.

    Returns:
        torch.device: cuda, mps, or cpu, in order of speed.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Recording-key resolvers (shared with cli.py, config.py, and notebooks) ──
# WHY THESE EXIST: Recording dicts use variable key names depending on network
# architecture (single-layer 'hid' vs multi-layer 'hid_1', 'hid_2', etc.).
# Snapshots and notebooks always need to find the DEEPEST/PRIMARY hidden and
# inhibitory spike trains. These functions abstract that lookup.

def primary_hid_key(rec):
    """Return the recording key for the deepest hidden layer.

    WHY: Multi-layer networks record each layer separately ('hid_0', 'hid_1', etc.).
    For analysis, we typically care about the DEEPEST/FINAL hidden layer
    (the one with the most relevant representation before readout).

    Returns:
        str: Recording key for the deepest excitatory layer.
             Single-layer models: 'hid'
             Multi-layer models: 'hid_1' (layer 1 is the deepest; layer 0 is input→hid_0)
             Networks with no hidden layer: 'hid' (default fallback)

    Args:
        rec: Recording dict from network.spike_record
    """
    hid_keys = sorted(k for k in rec if k.startswith("hid"))
    return hid_keys[-1] if hid_keys else "hid"


def primary_inh_key(rec):
    """Return the recording key for the deepest inhibitory layer, or None if no inhibition.

    WHY: Like primary_hid_key, but for inhibitory spikes. Networks without E-I
    structure return None (e.g., standard feedforward networks).

    Returns:
        str or None: Recording key for the deepest inhibitory layer (e.g. 'inh', 'inh_1'),
                     or None if the network doesn't have inhibitory populations.

    Args:
        rec: Recording dict from network.spike_record
    """
    inh_keys = sorted(k for k in rec if k.startswith("inh"))
    return inh_keys[-1] if inh_keys else None


# Scannable variables
# =============================================================================

SCAN_DEFAULTS = {
    "stim-overdrive": (1.0, "x"),
    "tau_gaba": (9.0, "ms"),
    "tau_ampa": (2.0, "ms"),
    "w_ei_mean": (W_EI[0], "μS"),
    "w_ie_mean": (W_IE[0], "μS"),
    "ei_strength": (1.0, ""),
    "spike_rate": (1.0, "Hz"),
    "bias": (1.0, "μS"),
    "dt": (1.0, "ms"),
    "digit": (0, ""),  # dataset digit class (0-9)
    "noise": (0, "Hz"),  # Poisson noise rate added to input
}


def _apply_scan_var(var_name, value):
    """Apply a scan variable that mutates global state (M globals or cfg).

    Scan loops sweep over parameters (tau_gaba, w_ei_mean, etc.), running simulations
    for each value. Some parameters require mutating global state (M.tau_gaba, cfg.w_ei),
    while others are passed directly to run_sim (stim-overdrive, spike_rate, etc.).

    This function handles the GLOBAL-MUTATING parameters. Loop-passed parameters are
    handled by the scan loop itself (e.g., stim-overdrive, spike_rate, noise, digit).

    WHY: Separates concerns — global mutations go here, local parameters stay in loops.

    Args:
        var_name: Scan variable name (tau_gaba, w_ei_mean, ei_strength, etc.)
        value: Value to apply

    Handled mutations:
        tau_gaba, tau_ampa: Time constants → compute discrete decay constants
        w_ei_mean, w_ie_mean, ei_strength, bias: Network params → apply via cfg
    """
    import config as C

    if var_name == "tau_gaba":
        # τ_GABA (inhibitory synaptic time constant) affects E-I gamma oscillation frequency
        M.tau_gaba = value
        M.decay_gaba = np.exp(-M.dt / value)  # Exponential decay: exp(-dt/τ)
    elif var_name == "tau_ampa":
        # τ_AMPA (excitatory synaptic time constant) affects input timescale
        M.tau_ampa = value
        M.decay_ampa = np.exp(-M.dt / value)
    elif var_name in ("w_ei_mean", "w_ie_mean", "ei_strength", "bias"):
        # Network weight/bias mutations: delegate to Config.apply_frame_param
        # (It handles parameter interactions, e.g., ei_strength affects both w_ei and w_ie)
        C.cfg.apply_frame_param(var_name, value)
    # Other scan vars (stim-overdrive, spike_rate, noise, digit) don't mutate globals;
    # they're passed as kwargs to run_sim directly in the scan loop
