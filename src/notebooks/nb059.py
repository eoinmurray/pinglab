"""Notebook runner for entry 059 — does the gamma clock make spiking
computation noise-robust?

The first concrete test of the PING-as-a-clock idea (article ar062).
Trains matched PING and COBA classifiers, then at inference sweeps
per-cell membrane noise (the simulator's noise_std knob, injected on
g_E) and measures, for each model:

- accuracy A(sigma) under noise;
- the trial-to-trial readout dispersion D_bar(sigma) — how much the
  logits wobble when the SAME input is presented K times under
  independent noise draws; and
- the mean inhibitory firing rate r_I, which is the energy the
  self-organised clock actually costs, turned into a back-of-envelope
  ledger against a conventional clock tree.

Membrane noise is wired into the CLI's single-trial scope path
(generate_spike_snapshot), NOT into the batched infer() evaluator, so
this runner drives the forward pass in-process — the nb038 pattern —
and passes noise_std straight to net.forward().

Figures land in /figures/notebooks/nb059/ and the success-criteria
summary in nb059/numbers.json.

Notebook entry: src/docs/content/notebooks/nb059.mdx
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from helpers.fmt import format_duration  # noqa: E402
from helpers.modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402
from helpers.tier import parse_tier  # noqa: E402
from cli import theme  # noqa: E402

SLUG = "nb059"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
OSCILLOSCOPE = REPO / "src" / "cli/cli.py"

TIER_CONFIG = {
    "extra small": dict(max_samples=100, epochs=1, n_eval=40),
    "small": dict(max_samples=500, epochs=5, n_eval=100),
    "medium": dict(max_samples=2000, epochs=10, n_eval=200),
    "large": dict(max_samples=5000, epochs=40, n_eval=400),
    "extra large": dict(max_samples=10000, epochs=40, n_eval=400),
}
DEFAULT_TIER = "small"
T_MS = 200.0
DT_TRAIN = 0.1
SEED = 42

# Reliability sweep: per-step white noise added to g_E (mV-scale), and
# the number of independent noisy repeats of each input. K_REPEATS > 1
# is what lets us measure trial-to-trial dispersion; sigma = 0 must give
# zero dispersion for both models (sanity check).
NOISE_STD_GRID: list[float] = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
K_REPEATS: int = 8
RASTER_SIGMAS: list[float] = [0.0, 8.0, 32.0]  # for Figure 3
RASTER_SAMPLE_IDX: int = 0
RASTER_N_E_PLOT: int = 200
RASTER_N_I_PLOT: int = 64

# Energy-ledger constants (illustrative; see ar062 / nb059.mdx).
# c_node: per-endpoint switched capacitance of a clock tree (F);
# V_dd: supply (V); E_sp: energy per spike/synaptic op (J).
LEDGER_C_NODE_F: float = 5e-15        # 5 fF per clock endpoint
LEDGER_V_DD: float = 0.8             # 0.8 V
LEDGER_E_SP_J: float = 1e-13         # 100 pJ per spike op
LEDGER_N_GRID = np.logspace(2, 7, 200)  # array sizes 1e2 .. 1e7

MODELS = ["coba", "ping"]

# Recipes copied from nb038 — same architecture, COBA is the loop-off
# limit. Both train WITH their own loop setting so each is competent in
# its own configuration (removes the nb038 transfer-accuracy confound).
MODEL_RECIPES: dict[str, dict] = {
    "coba": {
        "__build_as": "ping",
        "--ei-strength": "0",
        "--v-grad-dampen": "1000",
        "--w-in": "0.3",
        "--w-in-sparsity": "0.95",
        "--readout": "mem-mean",
        "--surrogate-slope": "1",
        "--readout-w-out-scale": "100",
        "--lr": "0.0004",
        "--batch-size": "256",
    },
    "ping": {
        "__build_as": "ping",
        "--ei-strength": "1",
        "--v-grad-dampen": "1000",
        "--w-in": "1.2",
        "--w-in-sparsity": "0.95",
        "--readout": "mem-mean",
        "--surrogate-slope": "1",
        "--readout-w-out-scale": "500",
        "--lr": "0.0004",
        "--batch-size": "256",
    },
}

MODEL_COLORS = {"coba": theme.DEEP_RED, "ping": theme.INK_BLACK}
MODEL_LABELS = {"coba": "COBA (loop off)", "ping": "PING (loop on)"}


def baseline_dir(model: str) -> Path:
    return ARTIFACTS / f"{model}__seed{SEED}"


def build_train_args(model: str, tier: str, out_dir: Path) -> list[str]:
    recipe = MODEL_RECIPES[model]
    args = [
        "train",
        "--model", recipe["__build_as"],
        "--dataset", "mnist",
        "--max-samples", str(TIER_CONFIG[tier]["max_samples"]),
        "--epochs", str(TIER_CONFIG[tier]["epochs"]),
        "--t-ms", str(T_MS),
        "--dt", str(DT_TRAIN),
        "--seed", str(SEED),
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]
    for k, v in recipe.items():
        if k.startswith("__"):
            continue
        if v is True:
            args.append(k)
        elif v is not None:
            args += [k, v]
    return args


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


# ── In-process trained-net loader (nb038 pattern) ────────────────────


def _load_trained(train_dir: Path, n_eval: int):
    """Load a trained net plus a fixed evaluation batch. Returns
    (net, cfg, X_eval, y_eval, device). Recording is left off; callers
    flip net.recording per need."""
    import torch

    import models as M
    from cli.config import build_net, patch_dt
    from cli import _auto_device, load_dataset, seed_everything

    cfg = json.loads((train_dir / "config.json").read_text())
    seed_everything(int(cfg.get("seed", SEED)))
    M.T_ms = float(cfg["t_ms"])
    patch_dt(float(cfg["dt"]))
    hidden_sizes = cfg.get("hidden_sizes") or [int(cfg["n_hidden"])]
    M.N_HID = hidden_sizes[-1]
    M.N_INH = hidden_sizes[-1] // 4
    M.HIDDEN_SIZES = list(hidden_sizes)

    device = _auto_device()
    _, X_te, _, y_te = load_dataset(
        cfg["dataset"], max_samples=int(cfg["max_samples"]), split=True
    )
    M.N_IN = 784 if cfg["dataset"] == "mnist" else 64
    n_eval = min(n_eval, X_te.shape[0])
    X_eval = X_te[:n_eval]
    y_eval = y_te[:n_eval]

    w_in_cfg = cfg.get("w_in")
    w_in_arg = (
        (float(w_in_cfg[0]), float(w_in_cfg[1]))
        if isinstance(w_in_cfg, list) and len(w_in_cfg) >= 2
        else None
    )
    net = build_net(
        cfg["model"],
        w_in=w_in_arg,
        w_in_sparsity=float(cfg.get("w_in_sparsity") or 0.0),
        ei_strength=float(cfg.get("ei_strength") or 1.0),
        ei_ratio=float(cfg.get("ei_ratio") or 2.0),
        sparsity=float(cfg.get("sparsity") or 0.0),
        device=device,
        randomize_init=not bool(cfg.get("kaiming_init", False)),
        dales_law=bool(cfg.get("dales_law", True)),
        hidden_sizes=hidden_sizes,
    )
    if hasattr(net, "readout_mode"):
        net.readout_mode = cfg.get("readout_mode", "mem-mean")

    state = torch.load(train_dir / "weights.pth", map_location=device)
    net.load_state_dict(state, strict=False)
    net.eval()
    return net, cfg, X_eval, y_eval, device


def run_noise_sweep(model: str, n_eval: int) -> list[dict]:
    """For each noise level, present the SAME eval batch K times under
    independent membrane-noise draws; record accuracy, readout
    dispersion, and E/I firing rates."""
    import torch

    import models as M
    from cli import encode_batch

    train_dir = baseline_dir(model)
    if not (train_dir / "weights.pth").exists():
        raise SystemExit(f"noise sweep needs trained {model} weights at {train_dir}")
    net, cfg, X_eval, y_eval, device = _load_trained(train_dir, n_eval)
    net.recording = True

    n_e = M.N_HID
    n_i = M.N_INH or 1
    t_sec = float(cfg["t_ms"]) / 1000.0
    y_t = torch.from_numpy(y_eval).to(device)
    X_t = torch.from_numpy(X_eval).to(device)
    n_total = y_eval.shape[0]

    # Encode the input ONCE (fixed generator) so the only thing varying
    # across repeats is the membrane noise — the input spikes are held
    # identical. encode_batch is deterministic given the generator seed.
    enc_gen = torch.Generator().manual_seed(SEED)
    spk = encode_batch(X_t, M.dt, False, generator=enc_gen)

    rows: list[dict] = []
    for sigma in NOISE_STD_GRID:
        logits_repeats = np.zeros((K_REPEATS, n_total, 10), dtype=np.float64)
        accs = []
        e_rates = []
        i_rates = []
        for k in range(K_REPEATS):
            # Reseed the GLOBAL torch RNG per repeat: the forward's g_noise
            # draw uses it, so each repeat sees an independent noise stream
            # while the (already-encoded) input stays fixed.
            torch.manual_seed(1000 + k)
            with torch.no_grad():
                logits = net(input_spikes=spk, noise_std=float(sigma))
            logits_np = logits.detach().cpu().numpy()
            logits_repeats[k] = logits_np[:, :10]
            accs.append(float((logits.argmax(1) == y_t).float().mean().item()) * 100.0)
            e_sum = float(net.spike_record["hid"].sum().item())
            i_sum = (
                float(net.spike_record["inh"].sum().item())
                if "inh" in net.spike_record
                else 0.0
            )
            e_rates.append(e_sum / (n_total * n_e * t_sec))
            i_rates.append(i_sum / (n_total * n_i * t_sec) if i_sum else 0.0)

        # Trial-to-trial readout dispersion: var across repeats (axis 0),
        # mean over classes, sqrt, mean over inputs (RMS logit jitter).
        var_kc = logits_repeats.var(axis=0)          # (n_total, 10)
        d_per_input = np.sqrt(var_kc.mean(axis=1))     # (n_total,)
        d_bar = float(d_per_input.mean())

        row = {
            "model": model,
            "sigma": float(sigma),
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "dispersion": d_bar,
            "e_rate_hz": float(np.mean(e_rates)),
            "i_rate_hz": float(np.mean(i_rates)),
        }
        rows.append(row)
        print(
            f"  {model:<5} sigma={sigma:>5.1f}: acc={row['acc_mean']:5.1f}% "
            f"D={d_bar:7.4f}  E={row['e_rate_hz']:.1f}Hz I={row['i_rate_hz']:.1f}Hz"
        )
    return rows


def capture_noise_raster(model: str, sigma: float, n_eval: int) -> dict:
    """Single-trial raster for one input under a given noise level."""
    import torch

    import models as M
    from cli import encode_batch

    net, cfg, X_eval, y_eval, device = _load_trained(model_train_dir(model), n_eval)
    net.recording = True
    idx = min(RASTER_SAMPLE_IDX, X_eval.shape[0] - 1)
    X_b = torch.from_numpy(X_eval[idx : idx + 1]).to(device)
    enc_gen = torch.Generator().manual_seed(SEED)
    spk = encode_batch(X_b, M.dt, False, generator=enc_gen)
    torch.manual_seed(2025)
    with torch.no_grad():
        _ = net(input_spikes=spk, noise_std=float(sigma))
    e_full = net.spike_record["hid"].cpu().numpy()
    i_full = (
        net.spike_record["inh"].cpu().numpy()
        if "inh" in net.spike_record
        else np.zeros((e_full.shape[0], 1), dtype=e_full.dtype)
    )
    rng = np.random.default_rng(0)
    e_idx = np.sort(rng.choice(e_full.shape[1], min(RASTER_N_E_PLOT, e_full.shape[1]), replace=False))
    i_idx = np.sort(rng.choice(i_full.shape[1], min(RASTER_N_I_PLOT, i_full.shape[1]), replace=False))
    return {
        "model": model,
        "sigma": float(sigma),
        "label": int(y_eval[idx]),
        "e": e_full[:, e_idx].astype(bool),
        "i": i_full[:, i_idx].astype(bool),
        "dt": float(cfg["dt"]),
        "t_ms": float(cfg["t_ms"]),
    }


def model_train_dir(model: str) -> Path:
    return baseline_dir(model)


# ── Figures ──────────────────────────────────────────────────────────


def _despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot_reliability_compound(rows: list[dict], out_path: Path, run_id: str) -> None:
    """Figure 1: accuracy (left) and readout dispersion (right) vs noise,
    PING vs COBA."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"
    fig, (ax_a, ax_d) = plt.subplots(1, 2, figsize=(12.0, 6.75), dpi=150)
    for model in MODELS:
        msel = sorted((r for r in rows if r["model"] == model), key=lambda r: r["sigma"])
        xs = [r["sigma"] for r in msel]
        ax_a.plot(
            xs, [r["acc_mean"] for r in msel], marker="o", lw=1.5,
            color=MODEL_COLORS[model], label=MODEL_LABELS[model],
        )
        ax_d.plot(
            xs, [r["dispersion"] for r in msel], marker="o", lw=1.5,
            color=MODEL_COLORS[model], label=MODEL_LABELS[model],
        )
    ax_a.set_xlabel("membrane noise σ (mV-scale on g_E)")
    ax_a.set_ylabel("test accuracy (%)")
    ax_a.set_title("Accuracy under noise", loc="left", fontweight="semibold")
    ax_a.set_ylim(0, 100)
    ax_a.legend(frameon=False, fontsize=theme.SIZE_LEGEND, loc="lower left")
    _despine(ax_a)

    ax_d.set_xlabel("membrane noise σ (mV-scale on g_E)")
    ax_d.set_ylabel("readout dispersion  D̄(σ)  (RMS logit jitter)")
    ax_d.set_title("Trial-to-trial readout wobble", loc="left", fontweight="semibold")
    ax_d.legend(frameon=False, fontsize=theme.SIZE_LEGEND, loc="upper left")
    _despine(ax_d)

    fig.suptitle(
        "Does the gamma clock make computation noise-robust?",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_energy_ledger(rows: list[dict], out_path: Path, run_id: str) -> dict:
    """Figure 2: self-clock power (flat) vs clock-tree power (linear in N)
    against array size. Returns the crossover summary."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"

    # Self-clock cost uses the PING inhibitory rate at sigma = 0.
    ping0 = next(
        r for r in rows if r["model"] == "ping" and r["sigma"] == 0.0
    )
    r_i = max(ping0["i_rate_hz"], 1e-9)
    # Frame rate ~ gamma; approximate by I-volley rate is not f_gamma, so
    # use a nominal 25 Hz clock for the tree (matches the project default
    # operating point). The comparison is order-of-magnitude.
    f_clk = 25.0

    n = LEDGER_N_GRID
    # I-population is 1/5 of the array (N_I = N/5 at the 4:1 E:I default).
    n_i_frac = 1.0 / 5.0
    p_ping = r_i * (n * n_i_frac) * LEDGER_E_SP_J
    p_clk = 1.0 * (LEDGER_C_NODE_F * n) * (LEDGER_V_DD ** 2) * f_clk

    cross_mask = p_ping < p_clk
    n_star = float(n[cross_mask][0]) if cross_mask.any() else None

    fig, ax = plt.subplots(figsize=(12.0, 6.75), dpi=150)
    ax.plot(n, p_clk, color=theme.DEEP_RED, lw=1.8, label="clock tree  P_clk = α C_node N V² f")
    ax.plot(n, p_ping, color=theme.INK_BLACK, lw=1.8, label="PING self-clock  P_ping = r_I N_I E_sp")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if n_star is not None:
        ax.axvline(n_star, color=theme.LABEL, ls="--", lw=1.0)
        ax.annotate(
            f"crossover N* ≈ {n_star:.0e}",
            xy=(n_star, p_clk[cross_mask][0]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=theme.SIZE_ANNOTATION,
        )
    ax.set_xlabel("array size N (endpoints)")
    ax.set_ylabel("clock power (W)")
    ax.set_title(
        f"Energy ledger — self-clock vs clock tree "
        f"(r_I = {r_i:.1f} Hz, E_sp = {LEDGER_E_SP_J:.0e} J, "
        f"c_node = {LEDGER_C_NODE_F:.0e} F)",
        loc="left", fontsize=theme.SIZE_LABEL,
    )
    ax.legend(frameon=False, fontsize=theme.SIZE_LEGEND, loc="upper left")
    _despine(ax)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "r_i_hz": r_i,
        "f_clk_hz": f_clk,
        "n_star": n_star,
        "c_node_f": LEDGER_C_NODE_F,
        "v_dd": LEDGER_V_DD,
        "e_sp_j": LEDGER_E_SP_J,
    }


def plot_resync_rasters(samples: list[dict], out_path: Path, run_id: str) -> None:
    """Figure 3: E/I rasters at increasing noise, one row per (model, σ)."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"
    n = len(samples)
    n_e, n_i, gap = RASTER_N_E_PLOT, RASTER_N_I_PLOT, 6
    fig, axes = plt.subplots(
        n, 1, figsize=(10.0, 5.625), sharex=True,
        gridspec_kw={"hspace": 0.25},
    )
    if n == 1:
        axes = [axes]
    for i, (ax, s) in enumerate(zip(axes, samples)):
        T = s["e"].shape[0]
        t_axis = np.arange(T) * s["dt"]
        e_t, e_n = np.where(s["e"])
        i_t, i_n = np.where(s["i"])
        ax.scatter(t_axis[e_t], e_n, s=1.6, c=theme.INK_BLACK, marker="|", linewidths=0.4)
        ax.scatter(t_axis[i_t], i_n + n_e + gap, s=1.6, c=theme.DEEP_RED, marker="|", linewidths=0.4)
        ax.set_ylim(-2, n_e + n_i + gap + 2)
        ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
        ax.set_yticklabels(["E", "I"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, s["t_ms"])
        ax.text(
            1.012, 0.5, f"{s['model']}\nσ = {s['sigma']:g}",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=theme.SIZE_ANNOTATION,
        )
        if i == 0:
            ax.set_title("E (black) / I (red) spikes under rising membrane noise")
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    skip_training = "--skip-training" in sys.argv
    wipe_dir = "--no-wipe-dir" not in sys.argv
    n_eval = TIER_CONFIG[tier]["n_eval"]

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(
        f"notebook_run_id = {notebook_run_id} tier={tier} "
        f"n_eval={n_eval} K={K_REPEATS}"
        + ("  [skip-training]" if skip_training else "")
    )

    prepare_run_dirs(
        SLUG, notebook_run_id, wipe=wipe_dir, skip_training=skip_training,
        make_artifacts=False,
    )

    if not skip_training:
        dispatcher = BatchDispatcher(modal_gpu, REPO, OSCILLOSCOPE)
        for model in MODELS:
            build_as = MODEL_RECIPES[model]["__build_as"]
            gpu_override = None
            if modal_gpu in ("T4", "L4", "A10G") and build_as == "ping":
                gpu_override = "A100"
            out = baseline_dir(model)
            print(f"[train] {model} → {out.relative_to(REPO)}"
                  + (f"  [modal:{modal_gpu}]" if modal_gpu else ""))
            dispatcher.submit(
                build_train_args(model, tier, out), out, gpu_override=gpu_override
            )
        dispatcher.drain()

    # Reliability sweep — the headline.
    noise_rows: list[dict] = []
    for model in MODELS:
        print(f"[noise-sweep] {model}")
        noise_rows += run_noise_sweep(model, n_eval)

    plot_reliability_compound(
        noise_rows, FIGURES / "reliability_compound.png", notebook_run_id
    )
    print(f"wrote {FIGURES / 'reliability_compound.png'}")

    # Energy ledger.
    ledger = plot_energy_ledger(
        noise_rows, FIGURES / "energy_ledger.png", notebook_run_id
    )
    print(f"wrote {FIGURES / 'energy_ledger.png'}")

    # Mechanism rasters: both models at a few noise levels.
    raster_samples = [
        capture_noise_raster(model, sigma, n_eval)
        for model in MODELS
        for sigma in RASTER_SIGMAS
    ]
    plot_resync_rasters(
        raster_samples, FIGURES / "resync_rasters.png", notebook_run_id
    )
    print(f"wrote {FIGURES / 'resync_rasters.png'}")

    duration_s = time.monotonic() - t_start
    train_cfg = load_config(baseline_dir(MODELS[0]))
    summary = {
        "notebook_run_id": notebook_run_id,
        "git_sha": train_cfg.get("git_sha"),
        "duration_s": round(duration_s, 1),
        "duration": format_duration(duration_s),
        "tier": tier,
        "config": {
            "tier": tier,
            "dataset": "mnist",
            "models": MODELS,
            "noise_std_grid": NOISE_STD_GRID,
            "k_repeats": K_REPEATS,
            "n_eval": n_eval,
            "t_ms": T_MS,
            "dt": DT_TRAIN,
            "seed": SEED,
        },
        "noise_sweep": noise_rows,
        "energy_ledger": ledger,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")


if __name__ == "__main__":
    main()
