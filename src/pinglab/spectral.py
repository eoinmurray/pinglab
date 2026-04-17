"""Spectral analysis of the E-I recurrent connectivity matrix.

Constructs the combined (N_E + N_I) × (N_E + N_I) recurrent matrix from
a PINGNet's frozen buffers, computes eigenvalues, and predicts oscillation
frequency from the dominant complex pair. Compares predicted vs measured f0.

Usage:
    uv run python src/pinglab/spectral.py [--from-dir DIR] [--ei-strength S]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))


def build_recurrent_matrix(net):
    """Extract the combined E-I recurrent matrix from a PINGNet.

    Returns the (N_E + N_I) × (N_E + N_I) matrix J:
        J = [[W_ee,   -W_ie^T],
             [W_ei^T,  0     ]]

    Sign convention: E columns are positive (excitatory),
    I columns are negative (inhibitory). The minus on W_ie
    encodes Dale's law — I neurons suppress E neurons.
    """
    W_ee = net.W_ee.cpu().numpy()  # (N_E, N_E)
    W_ei = net.W_ei.cpu().numpy()  # (N_E, N_I) — E excites I
    W_ie = net.W_ie.cpu().numpy()  # (N_I, N_E) — I inhibits E

    N_E = W_ee.shape[0]
    N_I = W_ei.shape[1]
    N = N_E + N_I

    J = np.zeros((N, N))
    J[:N_E, :N_E] = W_ee          # E→E (excitatory)
    J[:N_E, N_E:] = -W_ie.T       # I→E (inhibitory, negative)
    J[N_E:, :N_E] = W_ei.T        # E→I (excitatory)
    # J[N_E:, N_E:] = 0           # I→I (none in PING)

    return J, N_E, N_I


def analyze_spectrum(J, dt, N_E):
    """Compute eigenvalues and extract dynamical predictions."""
    eigs = np.linalg.eigvals(J)

    spectral_radius = np.max(np.abs(eigs))

    # Dominant complex pair (largest imaginary part)
    complex_mask = np.abs(eigs.imag) > 1e-10
    if complex_mask.any():
        complex_eigs = eigs[complex_mask]
        dom_idx = np.argmax(np.abs(complex_eigs.imag))
        dom = complex_eigs[dom_idx]
        # Predicted oscillation frequency from imaginary part
        # For discrete system: angle of eigenvalue / (2π * dt_seconds)
        # For continuous-rate approx: imag(λ) / (2π) gives Hz
        pred_f0_linear = np.abs(dom.imag) / (2 * np.pi) * 1000  # ms→Hz
    else:
        dom = None
        pred_f0_linear = 0.0

    # Nonlinear correction: the linear eigenvalue predicts oscillation
    # frequency for a smooth rate model. The spiking network has additional
    # timescales that stretch each gamma cycle:
    #   T_cycle ≈ ref_E + tau_GABA + T_integration
    # where T_integration is the time for E neurons to reach threshold
    # after inhibition wears off (depends on drive strength).
    # Minimum cycle period from refractory + GABA decay alone:
    from models import ref_ms_E, tau_gaba
    T_min_ms = ref_ms_E + tau_gaba  # ~3 + 9 = 12ms
    pred_f0_corrected = 1000.0 / T_min_ms if T_min_ms > 0 else pred_f0_linear

    # Classify eigenvalues by E/I participation
    _, vecs = np.linalg.eig(J)

    return {
        "eigenvalues": eigs,
        "eigenvectors": vecs,
        "spectral_radius": spectral_radius,
        "dominant_complex": dom,
        "predicted_f0_linear": pred_f0_linear,
        "predicted_f0_corrected": pred_f0_corrected,
        "T_min_ms": T_min_ms,
    }


def plot_spectrum(result, N_E, N_I, title="", out_path=None, measured_f0=None):
    """Plot eigenvalue spectrum in the complex plane."""
    eigs = result["eigenvalues"]
    sr = result["spectral_radius"]
    pred_f0_lin = result["predicted_f0_linear"]
    pred_f0_cor = result["predicted_f0_corrected"]
    dom = result["dominant_complex"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Full spectrum in complex plane
    ax = axes[0]
    ax.scatter(eigs.real, eigs.imag, s=8, c="#2a2a2a", alpha=0.6, zorder=2)
    if dom is not None:
        ax.scatter([dom.real, dom.real], [dom.imag, -dom.imag],
                   s=60, c="#cc4444", marker="*", zorder=3,
                   label=f"dominant: {dom.real:.3f}±{abs(dom.imag):.3f}i")
    circle = plt.Circle((0, 0), sr, fill=False, color="#888",
                         linestyle="--", linewidth=1)
    ax.add_patch(circle)
    ax.axhline(0, color="#ccc", linewidth=0.5)
    ax.axvline(0, color="#ccc", linewidth=0.5)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title(f"Eigenvalue spectrum (ρ={sr:.3f})")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)

    # Panel 2: Eigenvalue magnitudes sorted
    ax = axes[1]
    mags = np.sort(np.abs(eigs))[::-1]
    ax.plot(mags, color="#2a2a2a", linewidth=1.5)
    ax.axhline(1.0, color="#cc4444", linestyle="--", linewidth=1,
               label="unit circle")
    ax.set_xlabel("Eigenvalue index (sorted)")
    ax.set_ylabel("|λ|")
    ax.set_title("Eigenvalue magnitudes")
    ax.legend(fontsize=8)
    ax.set_xlim(0, len(mags))

    # Panel 3: Frequency prediction
    ax = axes[2]
    freqs = np.abs(eigs.imag) / (2 * np.pi) * 1000
    freqs_nonzero = freqs[freqs > 0.1]
    if len(freqs_nonzero) > 0:
        ax.hist(freqs_nonzero, bins=30, color="#2a2a2a", alpha=0.7,
                edgecolor="white", linewidth=0.3)
    if pred_f0_lin > 0:
        ax.axvline(pred_f0_lin, color="#cc4444", linewidth=2, alpha=0.4,
                   label=f"linear f0={pred_f0_lin:.0f}Hz")
    if pred_f0_cor > 0:
        ax.axvline(pred_f0_cor, color="#cc4444", linewidth=2,
                   label=f"corrected f0={pred_f0_cor:.0f}Hz")
    if measured_f0 is not None and measured_f0 > 0:
        ax.axvline(measured_f0, color="#4444cc", linewidth=2, linestyle="--",
                   label=f"measured f0={measured_f0:.0f}Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Count")
    ax.set_title("Oscillation modes")
    ax.legend(fontsize=8)

    fig.suptitle(title or "E-I Recurrent Matrix Spectrum", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"  → {out_path}")
    plt.close(fig)
    return fig


def plot_eigenmodes(result, N_E, N_I, title="", out_path=None, n_modes=6):
    """Plot the spatial patterns of the top eigenmodes.

    For each mode: shows E and I neuron participation (magnitude),
    and for complex pairs shows the phase relationship.
    """
    eigs = result["eigenvalues"]
    vecs = result["eigenvectors"]

    # Sort by eigenvalue magnitude, take top n_modes
    order = np.argsort(np.abs(eigs))[::-1]
    # Deduplicate conjugate pairs — keep only positive imaginary
    seen = set()
    selected = []
    for idx in order:
        if len(selected) >= n_modes:
            break
        # Skip if we already have its conjugate
        conj_idx = None
        for s in seen:
            if (abs(eigs[idx].real - eigs[s].real) < 1e-10 and
                    abs(eigs[idx].imag + eigs[s].imag) < 1e-10):
                conj_idx = s
                break
        if conj_idx is not None:
            continue
        selected.append(idx)
        seen.add(idx)

    n = len(selected)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(selected):
        lam = eigs[idx]
        vec = vecs[:, idx]
        freq = abs(lam.imag) / (2 * np.pi) * 1000

        e_part = vec[:N_E]
        i_part = vec[N_E:]

        # Panel 1: Participation magnitude
        ax = axes[row, 0]
        ax.bar(range(N_E), np.abs(e_part), color="#2a2a2a", alpha=0.7,
               width=1.0, label="E")
        ax.bar(range(N_E, N_E + N_I), np.abs(i_part), color="#cc4444",
               alpha=0.7, width=1.0, label="I")
        ax.set_ylabel("|v|")
        if row == 0:
            ax.legend(fontsize=8)
        lam_str = f"{lam.real:.3f}{lam.imag:+.3f}i" if abs(lam.imag) > 1e-10 else f"{lam.real:.3f}"
        ax.set_title(f"Mode {row+1}: λ={lam_str}  |λ|={abs(lam):.3f}  f={freq:.0f}Hz",
                     fontsize=9)
        if row == n - 1:
            ax.set_xlabel("Neuron index")

        # Panel 2: Phase (angle of complex eigenvector components)
        ax = axes[row, 1]
        if abs(lam.imag) > 1e-10:
            e_phase = np.angle(e_part)
            i_phase = np.angle(i_part)
            ax.scatter(range(N_E), e_phase, s=3, c="#2a2a2a", alpha=0.5)
            ax.scatter(range(N_E, N_E + N_I), i_phase, s=3, c="#cc4444",
                       alpha=0.5)
            ax.set_ylabel("Phase (rad)")
            ax.set_ylim(-np.pi, np.pi)
            ax.axhline(0, color="#ccc", linewidth=0.5)
            # Mean phase difference
            e_mean = np.angle(np.mean(e_part))
            i_mean = np.angle(np.mean(i_part))
            delta = np.degrees((i_mean - e_mean + np.pi) % (2*np.pi) - np.pi)
            ax.set_title(f"Phase  (E→I lag: {delta:.0f}°)", fontsize=9)
        else:
            ax.set_title("(real mode — no phase)", fontsize=9)
            ax.axis("off")
        if row == n - 1:
            ax.set_xlabel("Neuron index")

        # Panel 3: E vs I participation in complex plane
        ax = axes[row, 2]
        if abs(lam.imag) > 1e-10:
            ax.scatter(e_part.real, e_part.imag, s=5, c="#2a2a2a",
                       alpha=0.5, label="E")
            ax.scatter(i_part.real, i_part.imag, s=5, c="#cc4444",
                       alpha=0.5, label="I")
            ax.axhline(0, color="#ccc", linewidth=0.5)
            ax.axvline(0, color="#ccc", linewidth=0.5)
            ax.set_aspect("equal")
            ax.set_title("Eigenvector components", fontsize=9)
            if row == 0:
                ax.legend(fontsize=8)
        else:
            ax.bar(range(N_E), e_part.real, color="#2a2a2a", alpha=0.7,
                   width=1.0)
            ax.bar(range(N_E, N_E + N_I), i_part.real, color="#cc4444",
                   alpha=0.7, width=1.0)
            ax.set_title("Eigenvector (real)", fontsize=9)
        if row == n - 1:
            ax.set_xlabel("Re" if abs(lam.imag) > 1e-10 else "Neuron index")

    fig.suptitle(f"Eigenmodes — {title}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"  → {out_path}")
    plt.close(fig)
    return fig


def main():
    ap = argparse.ArgumentParser(
        description="Spectral analysis of the PING E-I recurrent matrix.")
    ap.add_argument("--from-dir", type=str, default=None,
                    help="Load trained PINGNet from this directory")
    ap.add_argument("--ei-strength", type=float, default=0.5,
                    help="E-I coupling strength for untrained analysis (default: 0.5)")
    ap.add_argument("--ei-ratio", type=float, default=2.0)
    ap.add_argument("--sparsity", type=float, default=0.2)
    ap.add_argument("--n-hidden", type=int, default=256)
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    import models as M
    from config import build_net

    measured_f0 = None

    if args.from_dir:
        from_dir = Path(args.from_dir)
        cfg_path = from_dir / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            args.ei_strength = cfg.get("ei_strength", args.ei_strength)
            args.ei_ratio = cfg.get("ei_ratio", args.ei_ratio)
            args.n_hidden = cfg.get("n_hidden", args.n_hidden)
            args.dt = cfg.get("dt", args.dt)
            args.sparsity = cfg.get("sparsity", args.sparsity)
        metrics_path = from_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            end = metrics.get("end", {})
            measured_f0 = end.get("f0", 0)
        weights_path = from_dir / "weights.pth"

        M.N_IN = cfg.get("n_in", 64)
        M.N_HID = args.n_hidden
        M.N_INH = args.n_hidden // 4
        net = build_net("ping", ei_strength=args.ei_strength,
                        ei_ratio=args.ei_ratio, sparsity=args.sparsity)
        if weights_path.exists():
            net.load_state_dict(torch.load(weights_path, map_location="cpu"),
                                strict=False)
            print(f"  loaded {weights_path}")
        title = f"PING spectrum — {from_dir.name}"
    else:
        M.N_HID = args.n_hidden
        M.N_INH = args.n_hidden // 4
        s = args.ei_strength
        net = build_net("ping",
                        ei_strength=s, ei_ratio=args.ei_ratio,
                        sparsity=args.sparsity)
        title = (f"PING spectrum — ei={args.ei_strength} "
                 f"ratio={args.ei_ratio} N={args.n_hidden}")

    J, N_E, N_I = build_recurrent_matrix(net)
    print(f"  J: ({N_E}+{N_I}) × ({N_E}+{N_I}) = {J.shape}")
    print(f"  E→E: nnz={np.count_nonzero(J[:N_E,:N_E])}")
    print(f"  E→I: nnz={np.count_nonzero(J[N_E:,:N_E])}")
    print(f"  I→E: nnz={np.count_nonzero(J[:N_E,N_E:])}")

    result = analyze_spectrum(J, args.dt, N_E)
    print(f"\n  spectral radius: {result['spectral_radius']:.4f}")
    if result["dominant_complex"] is not None:
        d = result["dominant_complex"]
        print(f"  dominant complex: {d.real:.4f} ± {abs(d.imag):.4f}i")
        print(f"  predicted f0 (linear):    {result['predicted_f0_linear']:.1f} Hz")
        print(f"  predicted f0 (corrected): {result['predicted_f0_corrected']:.1f} Hz"
              f"  (T_min = ref_E + tau_GABA = {result['T_min_ms']:.1f}ms)")
    if measured_f0 is not None and measured_f0 > 0:
        print(f"  measured f0:              {measured_f0:.1f} Hz")

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(args.from_dir) if args.from_dir else Path("src/artifacts"))
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_spectrum(result, N_E, N_I, title=title,
                  out_path=out_dir / "spectrum.png",
                  measured_f0=measured_f0)
    plot_eigenmodes(result, N_E, N_I, title=title,
                    out_path=out_dir / "eigenmodes.png")

    # Save raw eigenvalues for further analysis
    np.savez(out_dir / "spectrum.npz",
             eigenvalues=result["eigenvalues"],
             spectral_radius=result["spectral_radius"],
             predicted_f0_linear=result["predicted_f0_linear"],
             predicted_f0_corrected=result["predicted_f0_corrected"],
             T_min_ms=result["T_min_ms"],
             measured_f0=measured_f0 or 0,
             J=J)
    print(f"  → {out_dir / 'spectrum.npz'}")


if __name__ == "__main__":
    main()
