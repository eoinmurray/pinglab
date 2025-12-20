
from __future__ import annotations
import sys
import yaml
import shutil
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from pinglab.plots.styles import save_both, figsize
from pinglab.analysis import base_metrics
from pinglab.plots import save_raster
from pinglab.utils import slice_spikes

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import ARTIFACTS_ROOT

from lib.model import LocalConfig
from lib.image import synth_blobs, synth_bars, synth_checker
from lib.encoding import build_pixel_groups, image_to_group_currents
from lib.hotloop import hotloop_single_image
from lib.homeostatic import tune_I_E_for_target_rate


def save_image_png(img: np.ndarray, path: Path) -> None:
    """
    Minimal dependency save: use matplotlib only.
    """
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(img, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def reconstruct_from_spikes(
    *,
    img_true: np.ndarray,
    spikes,
    groups: list[np.ndarray],
    stim_start: float,
    stim_stop: float,
) -> tuple[np.ndarray, dict]:
    """
    Reconstruct an image from spike counts during the stimulus window.

    For each pixel, counts spikes from its assigned neuron group during
    [stim_start, stim_stop], then linearly rescales the count vector to [0,1].

    Args:
        img_true: Ground truth image, shape (H, W), values in [0,1].
        spikes: Spikes object from network simulation.
        groups: List of P arrays, where groups[p] contains neuron IDs for pixel p.
        stim_start: Start of stimulus window (ms).
        stim_stop: End of stimulus window (ms).

    Returns:
        img_hat: Reconstructed image, shape (H, W), values in [0,1].
            Each pixel value is the normalized spike count from its neuron group.
        metrics: Dictionary with reconstruction quality measures:
            mse: Mean squared error between img_true and img_hat.
            corr: Pearson correlation between flattened images. Zero if either
                image has zero variance.
            r_min: Minimum raw spike count across pixel groups (before rescaling).
            r_max: Maximum raw spike count across pixel groups (before rescaling).
                The range [r_min, r_max] indicates dynamic range of the readout.
    """
    h, w = img_true.shape
    P = h * w

    # Slice spikes to stim window for speed
    sp = slice_spikes(spikes, start_time=stim_start, stop_time=stim_stop)

    # Build counts per neuron id in window
    # ids in sp.ids are neuron indices in [0, N_E+N_I)
    # We only used E-neuron groups, so fine.
    counts = {}
    for nid in sp.ids:
        counts[int(nid)] = counts.get(int(nid), 0) + 1

    r = np.zeros((P,), dtype=np.float32)
    for p, ids in enumerate(groups):
        c = 0
        for nid in ids:
            c += counts.get(int(nid), 0)
        r[p] = float(c)

    # Rescale readout to [0,1]
    r0 = r.min()
    r1 = r.max()
    if r1 - r0 > 1e-8:
        r = (r - r0) / (r1 - r0)
    else:
        r = np.zeros_like(r)

    img_hat = r.reshape(h, w)

    # Metrics
    x = img_true.reshape(-1).astype(np.float32)
    y = img_hat.reshape(-1).astype(np.float32)
    mse = float(np.mean((x - y) ** 2))
    # corr: guard zero variance
    if float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
        corr = 0.0
    else:
        corr = float(np.corrcoef(x, y)[0, 1])

    metrics = dict(mse=mse, corr=corr, r_min=float(r0), r_max=float(r1))
    return img_hat, metrics


def main() -> None:
    data_path = ARTIFACTS_ROOT / Path(__file__).parent.name
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    config_path = Path(__file__).parent / "config.yaml"
    with config_path.open() as f:
        data = yaml.safe_load(f)

    config = LocalConfig.model_validate(data)

    cfg = config.experiment_1
    h, w = cfg.image_h, cfg.image_w

    # 1) generate image
    if cfg.image_type == "blobs":
        img = synth_blobs(h, w, n=2, seed=cfg.image_seed)
    elif cfg.image_type == "bars":
        img = synth_bars(h, w, seed=cfg.image_seed)
    elif cfg.image_type == "checker":
        img = synth_checker(h, w, seed=cfg.image_seed)
    else:
        raise ValueError(f"Unknown image_type: {cfg.image_type}")

    img = np.clip(cfg.image_contrast * img, 0.0, 1.0).astype(np.float32)

    # 2) pixel groups (disjoint E-neuron groups)
    groups = build_pixel_groups(
        N_E=config.base.N_E,
        h=h,
        w=w,
        group_size=cfg.group_size,
        seed=cfg.mapping_seed,
    )

    # 3) convert image -> added currents
    added = image_to_group_currents(
        img=img,
        groups=groups,
        scale=cfg.image_current_scale,
        value_range=tuple(cfg.pixel_value_range),
    )

    # 4) run network sweep over g_ei with homeostatic rate control
    g_ei_values = np.linspace(0.0, 5.0, 20)
    mse_values = []
    mean_E_rate = []
    I_E_tuned_values = []

    for g_ei in g_ei_values:
        config_run = config.model_copy(deep=True)
        config_run.base = config_run.base.model_copy(update={"g_ei": float(g_ei)})

        # Tune I_E to maintain target E rate
        I_E_tuned, tuning_history = tune_I_E_for_target_rate(
            config=config_run,
            target_rate_E=config.homeostasis.target_rate_E,
            added_current_E=added,
            seed=config_run.base.seed,
            burnin_ms=config.homeostasis.burnin_ms,
            max_iters=config.homeostasis.max_iters,
            eta=config.homeostasis.eta,
            tol=config.homeostasis.tol,
        )

        # Update config with tuned I_E
        config_run.default_inputs = config_run.default_inputs.model_copy(
            update={"I_E": I_E_tuned}
        )
        I_E_tuned_values.append(I_E_tuned)

        print(f"g_ei={g_ei:.2f}: tuned I_E={I_E_tuned:.3f} after {len(tuning_history)} iters")

        result, meta = hotloop_single_image(
            config=config_run,
            warmup_ms=cfg.warmup_ms,
            stim_ms=cfg.stim_ms,
            added_current_E=added,
            seed=config_run.base.seed,
        )

        metrics_base = base_metrics(
            config=config_run,
            run_result=result,
            data_path=data_path,
            label=f"g_ei_{g_ei:.2f}",
        )

        print(f"g_ei: {g_ei:.2f}")
        print(f"mean_rate_E: {metrics_base['mean_rate_E']:.2f} Hz")
        print(f"mean_rate_I: {metrics_base['mean_rate_I']:.2f} Hz")
        print(f"regime: {metrics_base['regime']}")
        print(f"regime_reason: {metrics_base['regime_reason']}")

        stim_start = float(meta["stim_start_ms"])
        stim_stop = float(meta["stim_stop_ms"])

        # 5) reconstruct
        img_hat, metrics = reconstruct_from_spikes(
            img_true=img,
            spikes=result.spikes,
            groups=groups,
            stim_start=stim_start,
            stim_stop=stim_stop,
        )

        # 6) save artifacts (tagged by g_ei)
        tag = f"g_ei_{g_ei:.2f}"
        save_image_png(img, data_path / f"image_true_{tag}.png")
        save_image_png(img_hat, data_path / f"image_recon_{tag}.png")

        # Raster slice for the presentation window
        sliced = slice_spikes(
            result.spikes,
            start_time=config_run.plotting.raster.start_time,
            stop_time=config_run.plotting.raster.stop_time,
        )

        save_raster(
            sliced,
            path=data_path / f"raster_{tag}.png",
            label=f"single-image ({cfg.image_type}, g_ei={g_ei:.2f})",
        )

        out = {
            "g_ei": float(g_ei),
            "stim_start_ms": stim_start,
            "stim_stop_ms": stim_stop,
            "mse": metrics["mse"],
            "corr": metrics["corr"],
            "r_min": metrics["r_min"],
            "r_max": metrics["r_max"],
            "h": h,
            "w": w,
            "type": cfg.image_type,
            "contrast": cfg.image_contrast,
            "group_size": cfg.group_size,
            "mapping_seed": cfg.mapping_seed,
            "image_current_scale": cfg.image_current_scale,
            "I_E_tuned": float(I_E_tuned),
            "I_E_initial": float(config.default_inputs.I_E),
            "I_I": float(config_run.default_inputs.I_I),
            "noise_std": float(config_run.default_inputs.noise),
            "target_rate_E": float(config.homeostasis.target_rate_E),
        }

        with (data_path / f"recon_metrics_{tag}.yaml").open("w") as f:
            yaml.safe_dump(out, f, sort_keys=False)

        mse_values.append(metrics["mse"])
        mean_E_rate.append(metrics_base["mean_rate_E"])

    print("g_ei values:", g_ei_values)
    print("MSE values:", mse_values)

    def plot_fn():
        plt.figure(figsize=figsize)
        plt.plot(g_ei_values, mse_values, marker="o")
        plt.xlabel("g_ei")
        plt.ylabel("Reconstruction MSE")
        plt.title("Reconstruction MSE vs g_ei")
        plt.grid(True)
        plt.tight_layout()

    save_both(
        data_path / "mse_vs_g_ei.png",
        plot_fn,
    )

    def plot_fn():
        plt.figure(figsize=figsize)
        plt.plot(g_ei_values, mean_E_rate, marker="o")
        plt.xlabel("g_ei")
        plt.ylabel("Mean E Rate (Hz)")
        plt.title("Mean E Rate vs g_ei")
        plt.grid(True)
        plt.tight_layout()

    save_both(
        data_path / "mean_E_rate_vs_g_ei.png",
        plot_fn,
    )

    def plot_fn():
        plt.figure(figsize=figsize)
        plt.plot(g_ei_values, I_E_tuned_values, marker="o")
        plt.axhline(config.default_inputs.I_E, color="gray", linestyle="--", label="initial I_E")
        plt.xlabel("g_ei")
        plt.ylabel("Tuned I_E")
        plt.title(f"Tuned I_E vs g_ei (target rate = {config.homeostasis.target_rate_E} Hz)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    save_both(
        data_path / "I_E_tuned_vs_g_ei.png",
        plot_fn,
    )

if __name__ == "__main__":
    main()
