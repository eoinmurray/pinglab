"""Video-mode helper: dt-sweep evaluation video for trained models.

Re-evaluates a trained network across a range of dt values, optionally
re-training at each dt, and renders a side-by-side accuracy plot plus a
per-dt training/eval video.
"""

from __future__ import annotations

import sys
from pathlib import Path

_pkg_dir = str(Path(__file__).resolve().parent.parent)
if _pkg_dir in sys.path:
    sys.path.remove(_pkg_dir)
sys.path.insert(0, _pkg_dir)
_cli_dir = str(Path(__file__).resolve().parent)
if _cli_dir in sys.path:
    sys.path.remove(_cli_dir)
sys.path.insert(0, _cli_dir)

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter

import models as M
from config import extract_weights, patch_dt
from plot import draw_transient_frame, make_transient_fig, reset_weight_xlims

from cli.datasets import _load_dataset_image
from cli.encoders import (
    encode_batch,
    encode_images_poisson,
    encode_smnist,
    transport_spikes_bin,
)
from cli.scan import primary_hid_key, primary_inh_key
from cli.train import train

log = logging.getLogger("oscilloscope")


def _plot_dt_sweep(sweep_results, train_dt, model_name, out_dir):
    """Plot accuracy vs dt with the training dt marked."""
    dts = [r["dt"] for r in sweep_results]
    accs = [r["acc"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(dts, accs, "o-", color="#2a2a2a", linewidth=1.5, markersize=6)
    if train_dt in dts:
        ref_acc = accs[dts.index(train_dt)]
        ax.axvline(
            train_dt,
            color="#cc4444",
            linestyle="--",
            linewidth=1,
            label=f"train dt={train_dt}",
        )
        ax.plot(train_dt, ref_acc, "s", color="#cc4444", markersize=10, zorder=5)
    ax.set_xlabel("dt (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"dt inference stability — {model_name}")
    ax.legend(loc="lower left")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "dt_sweep.png", dpi=150)
    plt.close(fig)


def _render_dt_sweep_video(
    net,
    dt_values,
    sweep_results,
    train_dt,
    model_name,
    dataset,
    out_dir,
    burn_in_ms=20.0,
    frozen_inputs=False,
):
    """Render an oscilloscope video with one frame per dt value."""
    import config as C

    C.N_E = M.N_HID
    C.N_I = M.N_INH

    loader_dataset = "mnist" if dataset in ("mnist", "smnist") else dataset
    ref_pixel_vec, ref_image = _load_dataset_image(loader_dataset, 0, 0)
    ref_input = torch.from_numpy(ref_pixel_vec).unsqueeze(0)
    use_smnist = dataset == "smnist"

    frozen_ref = None
    if frozen_inputs:
        dt_ref = dt_values[0]
        g = torch.Generator()
        g.manual_seed(99)
        if use_smnist:
            frozen_ref = encode_smnist(ref_input, dt_ref, M.max_rate_hz, generator=g)
        else:
            T_ref = int(M.T_ms / dt_ref)
            frozen_ref = encode_images_poisson(
                ref_input, T_ref, dt_ref, M.max_rate_hz, generator=g
            )

    reset_weight_xlims()
    fig, axes = make_transient_fig(layout="train")
    writer = FFMpegWriter(fps=2, metadata=dict(title=f"dt sweep — {model_name}"))
    writer.setup(fig, str(out_dir / "dt_sweep.mp4"), dpi=120)

    for i, sweep_dt in enumerate(dt_values):
        patch_dt(sweep_dt)
        if frozen_ref is not None:
            spk = transport_spikes_bin(frozen_ref, dt_values[0], sweep_dt)
        else:
            spk = encode_batch(ref_input, sweep_dt, use_smnist)

        net.recording = True
        with torch.no_grad():
            net(input_spikes=spk)
        net.recording = False

        rec = net.spike_record
        burn = int(burn_in_ms / sweep_dt)

        def _to_np(v):
            if isinstance(v, torch.Tensor):
                return v.cpu().numpy()
            return torch.stack(v).numpy()

        spk_e = _to_np(rec[primary_hid_key(rec)])[burn:]
        _ik = primary_inh_key(rec)
        spk_i = _to_np(rec[_ik])[burn:] if _ik else None
        spk_h1 = _to_np(rec["hid_1"])[burn:] if "hid_1" in rec else None
        spk_o = _to_np(rec["out"])[burn:] if "out" in rec else None
        ext_g = (
            _to_np(rec["input"])[burn:]
            if "input" in rec
            else np.zeros((len(spk_e), spk_e.shape[1]))
        )

        acc = sweep_results[i]["acc"]
        marker = " ◄train" if sweep_dt == train_dt else ""
        title = f"dt={sweep_dt:.3f}ms  acc={acc:.1f}%{marker}"

        draw_transient_frame(
            axes,
            1.0,
            spk_e,
            spk_i,
            ext_g,
            sweep_dt,
            title,
            spk_o=spk_o,
            weights=extract_weights(net),
            model_name=model_name,
            sweep_frame_idx=i,
            n_e=M.N_HID,
            n_i=M.N_INH,
            acc=acc,
            digit_image=ref_image,
            total_epochs=len(dt_values),
            spk_h1=spk_h1,
        )
        writer.grab_frame()

    writer.finish()
    plt.close(fig)

