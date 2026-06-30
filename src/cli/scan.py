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
    """Pick the fastest available device: cuda > mps > cpu.

    Called when the user doesn't explicitly set --device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Recording-key resolvers (shared with cli.py and future modules) ──
def primary_hid_key(rec):
    """Return the recording key for the deepest hidden layer.

    Single-layer models use 'hid'; multi-layer use 'hid_1', 'hid_2', etc.
    Returns the highest-numbered key, or 'hid' for single-layer.
    """
    hid_keys = sorted(k for k in rec if k.startswith("hid"))
    return hid_keys[-1] if hid_keys else "hid"


def primary_inh_key(rec):
    """Return the recording key for the deepest inhibitory layer, or None."""
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

    Handles tau_* overrides for the M module. Other scan variables like
    stim-overdrive, spike_rate, noise are handled directly in scan loops.
    """
    import config as C

    if var_name == "tau_gaba":
        M.tau_gaba = value
        M.decay_gaba = np.exp(-M.dt / value)
    elif var_name == "tau_ampa":
        M.tau_ampa = value
        M.decay_ampa = np.exp(-M.dt / value)
    elif var_name in ("w_ei_mean", "w_ie_mean", "ei_strength", "bias"):
        # Config param mutations
        C.cfg.apply_frame_param(var_name, value)
    # Other scan vars (stim-overdrive, spike_rate, noise, digit) don't mutate globals
