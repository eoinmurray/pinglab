"""Analyse explicit exp049 recordings and pinned TR-05 histories.

Invoke this stage explicitly from the repository root. Scientific outputs
and input provenance belong to its independently completed Pingstore run.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import argparse
import warnings

import numpy as np
from experiments.exp049 import recipe
from experiments.exp049.custom import evidence, inputs
from experiments.exp049.recipe import F_GAMMA_BAND_HZ
from pingstore.contracts import PingstoreError, load_json, write_json_atomic

MEASUREMENT = {
    "schema": "exp049.measurement/v1",
    "final_outcomes": "endpoint means and sample-SD/sqrt(n) SEM across seeds; unsmoothed final reference-image contrast",
    "endpoint": "final-epoch checkpoint; official MNIST test accuracy and exact hid/inh rate keys",
    "psd": "per-trial full-length Welch density; demean; detrend=False; skip constant trials; average PSD; maximum raw bin in inclusive 5-150 Hz band",
    "weights": "native-dtype means; <=0 zero fraction; positive-only mean; retain per-network summaries and pool equal-sized matrices before 49-bin histograms",
    "cards": "raw training rate_e/rate_i; per-seed trajectories and nanmean; mean endpoint PSD and scalars",
    "trajectories": "prefer test_rate_e/test_rate_i keys even if null; exclude incomplete curves; mean and min/max then five-epoch edge-padded moving average",
    "phase": "unsmoothed seed means; segment contrast averages adjacent endpoints",
    "rhythmicity": "frozen final mean; trainable first available contrast mean and final minimum/maximum",
    "raster": "seed 42, image index 0; RNG(42) samples up to 200 E then 50 I cells without replacement",
}


def analyse(identity, *, run_id=None, raster_source=None):
    source = inputs.source(REPO, identity, "compute")
    cfg, bank, contract = inputs.compute_evidence(REPO, source)
    histories = evidence.histories(bank.export, contract)
    sources = {"compute": source, "bank": bank}
    paired = None
    if raster_source is not None:
        sources["paired"] = inputs.source(REPO, raster_source, "compute")
        paired = inputs.paired_evidence(REPO, sources["paired"], source, bank)
    with inputs.execution(
        REPO,
        "analyse",
        sources=sources,
        run_id=run_id,
        configuration=MEASUREMENT,
    ) as run:
        for job in recipe.jobs(cfg):
            train = contract["configs"][job["cell_name"]]
            artifact = (
                source.unit(job["path"])
                if job["kind"] == "infer"
                else source.file(
                    job["path"],
                    "weights_dump.npz"
                    if job["kind"] == "weights_dump"
                    else "recording.npz",
                )
            )
            evidence.recordings(artifact, train, job)
        summary, curves, cards, weights, rasters, attractor = [], {}, {}, {}, [], {}
        outcomes = {}
        for cond in recipe.COND_ORDER:
            endpoints, matrices, metrics = [], [], []
            for seed in recipe.SEEDS:
                name = recipe.cell_name(cond, seed)
                train = contract["configs"][name]
                directory = source.unit("infer", name)
                with np.load(
                    source.file("weights_dump", name, "weights_dump.npz"),
                    allow_pickle=False,
                ) as raw:
                    arrays = tuple(raw[k] for k in recipe.WEIGHT_ARRAYS)
                matrices.append(arrays)
                with np.load(directory / "pop_traces.npz", allow_pickle=False) as raw:
                    result = endpoint(
                        train,
                        load_json(directory / "metrics.json"),
                        raw["pop_e"],
                        arrays,
                    )
                endpoints.append(result)
                summary.append(
                    {
                        "condition": cond,
                        "seed": seed,
                        **{
                            k: v
                            for k, v in result.items()
                            if k not in ("psd", "freqs_hz")
                        },
                    }
                )
                metrics.append(histories[name])
                curves[name] = epoch_curve(histories[name]["epochs"], cond)
            outcomes[cond] = final_outcomes(endpoints, metrics)
            cards[cond] = card(metrics, endpoints)
            weights[cond] = {
                "seeds": recipe.SEEDS,
                "weights": weight_distributions(matrices),
            }
            attractor[cond] = {
                "e": [r["e_rate_hz"] for r in endpoints],
                "i": [r["i_rate_hz"] for r in endpoints],
                "acc": float(np.nanmean([r["acc"] for r in endpoints])),
            }
            name = recipe.cell_name(cond, cfg["snapshot_seed"])
            filename = f"raster-{cond}.npz"
            np.savez_compressed(
                run.export / filename,
                **raster(
                    source.file("snapshot", name, "recording.npz"),
                    contract["configs"][name],
                ),
            )
            rasters.append({"condition": cond, "file": filename})
        paired_rasters = []
        if paired is not None:
            for entry in paired["entries"]:
                name = recipe.cell_name(entry["condition"], paired["config"]["seed"])
                filename = f"raster-{entry['stage']}-{entry['condition']}.npz"
                np.savez_compressed(
                    run.export / filename,
                    **raster(
                        sources["paired"].export / entry["file"],
                        contract["configs"][name],
                    ),
                )
                paired_rasters.append({**entry, "file": filename})
            with np.load(
                sources["paired"].export / "input.npz", allow_pickle=False
            ) as raw:
                np.savez_compressed(
                    run.export / "raster-input.npz",
                    pixels=raw["pixels"],
                    label=raw["label"],
                )
        write_json_atomic(
            run.export / "results.json",
            clean(
                {
                    "schema": "exp049.analysis/v1",
                    "recipe": cfg,
                    "measurement": MEASUREMENT,
                    "checkpoint_policy": recipe.CHECKPOINT_POLICY,
                    "checkpoint_provenance": contract["checkpoints"],
                    "config": {
                        "epochs": recipe.EPOCHS,
                        "max_samples": recipe.MAX_SAMPLES,
                        "evaluation_samples": cfg["evaluation_samples"],
                        "seeds": recipe.SEEDS,
                        "conditions": recipe.CONDITIONS,
                        "common_recipe": recipe.COMMON_RECIPE,
                    },
                    "summary": summary,
                    "rhythmicity": rhythmicity(curves),
                    "epoch_curves": curves,
                    "plot_data": {
                        "outcomes": outcomes,
                        "cards": cards,
                        "weights": weights,
                        "attractor": attractor,
                        "trajectories": trajectories(curves),
                        "last_epoch": max(max(c["ep"]) for c in curves.values()),
                    },
                    "rasters": rasters,
                    **(
                        {
                            "paired_rasters": paired_rasters,
                            "raster_protocol": paired["config"],
                        }
                        if paired is not None
                        else {}
                    ),
                }
            ),
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--raster-source", help="Explicit paired-raster compute run")
    args = parser.parse_args()
    try:
        analyse(args.source, run_id=args.run_id, raster_source=args.raster_source)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"exp049 analyse: {exc}\n")


def weight_summary(init_arr: np.ndarray, trained_arr: np.ndarray) -> dict[str, float]:
    """Return matrix-specific conductance statistics for numbers.json."""
    init = init_arr.ravel()
    trained = trained_arr.ravel()

    def _positive_mean(values: np.ndarray) -> float:
        positive = values[values > 0]
        return float(positive.mean()) if positive.size else 0.0

    return {
        "init_mean": float(init.mean()),
        "trained_mean": float(trained.mean()),
        "init_zero_fraction": float((init <= 0).mean()),
        "trained_zero_fraction": float((trained <= 0).mean()),
        "init_positive_mean": _positive_mean(init),
        "trained_positive_mean": _positive_mean(trained),
    }


def endpoint(config, metrics, pop_e, weights) -> dict:
    from scipy import signal as sp_signal

    cfg = config
    m = metrics
    rates = m.get("rates_hz", {})
    w_ei_init, w_ei_trained, w_ie_init, w_ie_trained = weights
    w_ei_summary = weight_summary(w_ei_init, w_ei_trained)
    w_ie_summary = weight_summary(w_ie_init, w_ie_trained)

    # f_γ via Welch PSD on the per-trial E-population trace (from pop_traces.npz).
    pop_e_traces = list(pop_e)
    fs_hz = 1000.0 / float(cfg["dt"])
    nperseg = pop_e_traces[0].size
    psds: list[np.ndarray] = []
    freqs: np.ndarray | None = None
    for tr in pop_e_traces:
        if tr.std() == 0:
            continue
        f, p = sp_signal.welch(
            tr - tr.mean(),
            fs=fs_hz,
            nperseg=nperseg,
            scaling="density",
            detrend=False,
        )
        psds.append(p)
        freqs = f
    if psds and freqs is not None:
        psd_mean = np.mean(np.stack(psds, axis=0), axis=0)
        band = (freqs >= F_GAMMA_BAND_HZ[0]) & (freqs <= F_GAMMA_BAND_HZ[1])
        if band.any() and psd_mean[band].max() > 0:
            peak_local = int(psd_mean[band].argmax())
            peak_idx = int(np.where(band)[0][peak_local])
            f_gamma = float(freqs[peak_idx])
            psd_used = psd_mean
            freqs_used = freqs
        else:
            f_gamma = float("nan")
            psd_used = psd_mean
            freqs_used = freqs
    else:
        f_gamma = float("nan")
        psd_used = np.zeros(1)
        freqs_used = np.zeros(1)

    return {
        "acc": float(m["best_acc"]),
        "e_rate_hz": float(rates.get("hid", 0.0)),
        "i_rate_hz": float(rates.get("inh", 0.0)),
        "f_gamma_hz": f_gamma,
        "w_ei_mean": w_ei_summary["trained_mean"],
        "w_ie_mean": w_ie_summary["trained_mean"],
        "w_ei": w_ei_summary,
        "w_ie": w_ie_summary,
        "psd": psd_used.tolist(),
        "freqs_hz": freqs_used.tolist(),
    }


def rhythmicity(curves) -> dict:
    trainable = {"trainable_ping_init", "trainable_zero_init", "trainable_small_init"}
    ep1: list[float] = []
    fin_tr: list[float] = []
    fin_fz: list[float] = []
    for c in curves.values():
        pairs = sorted(
            (e, v) for e, v in zip(c["ep"], c["contrast"], strict=True) if v is not None
        )
        if not pairs:
            continue
        if c["cond"] in trainable:
            ep1.append(pairs[0][1])
            fin_tr.append(pairs[-1][1])
        elif c["cond"] == "frozen_ping":
            fin_fz.append(pairs[-1][1])
    return {
        "canonical_contrast": float(np.mean(fin_fz)) if fin_fz else None,
        "epoch1_contrast_trainable": float(np.mean(ep1)) if ep1 else None,
        "final_contrast_trainable_min": float(np.min(fin_tr)) if fin_tr else None,
        "final_contrast_trainable_max": float(np.max(fin_tr)) if fin_tr else None,
    }


def clean(o):
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


def epoch_curve(rows, condition):
    return {
        "cond": condition,
        "ep": [r["ep"] for r in rows],
        "acc": [r["acc"] for r in rows],
        "rate_e": [r.get("test_rate_e", r.get("rate_e")) for r in rows],
        "rate_i": [r.get("test_rate_i", r.get("rate_i")) for r in rows],
        "contrast": [r.get("contrast") for r in rows],
    }


def smooth(y, w=5):
    y = np.asarray(y, dtype=float)
    if w <= 1 or y.size < w:
        return y
    return np.convolve(np.pad(y, w // 2, mode="edge"), np.ones(w) / w, mode="valid")[
        : y.size
    ]


def trajectories(curves):
    from experiments.exp049.recipe import COND_ORDER

    result = {}
    for cond in COND_ORDER:
        seeds = [c for c in curves.values() if c["cond"] == cond]
        panels = {}
        for key in ("acc", "rate_e", "rate_i", "contrast"):
            valid = [c for c in seeds if not any(v is None for v in c[key])]
            if valid:
                stack = np.array([c[key] for c in valid], dtype=float)
                panels[key] = {
                    "ep": valid[0]["ep"],
                    "mean": smooth(stack.mean(axis=0)).tolist(),
                    "lo": smooth(stack.min(axis=0)).tolist(),
                    "hi": smooth(stack.max(axis=0)).tolist(),
                }
        valid = [
            c
            for c in seeds
            if all(
                not any(v is None for v in c[k]) for k in ("rate_e", "contrast", "acc")
            )
        ]
        phase = None
        if valid:
            stacks = {
                k: np.array([c[k] for c in valid], dtype=float)
                for k in ("rate_e", "contrast", "acc")
            }
            means = {k: a.mean(axis=0) for k, a in stacks.items()}
            e, p, a = (means[k] for k in ("rate_e", "contrast", "acc"))
            phase = {
                "e": e.tolist(),
                "p": p.tolist(),
                "a": a.tolist(),
                "final_e": stacks["rate_e"][:, -1].tolist(),
                "final_p": stacks["contrast"][:, -1].tolist(),
                "final_acc": float(stacks["acc"][:, -1].mean()),
                "max_e": float(e.max()),
                "segment_pings": (0.5 * (p[:-1] + p[1:])).tolist(),
            }
        result[cond] = {"panels": panels, "phase": phase}
    return result


def card(metrics_list, final_cells):
    header = {}
    for key in ("acc", "e_rate_hz", "i_rate_hz", "f_gamma_hz"):
        values = [
            float(c[key])
            for c in final_cells
            if c.get(key) is not None and np.isfinite(c[key])
        ]
        header[key] = float(np.mean(values)) if values else None
    curves = {}
    for key, sub in (
        ("weight_norms", "W_ei.1"),
        ("weight_norms", "W_ie.1"),
        ("rate_e", None),
        ("rate_i", None),
        ("acc", None),
    ):
        per_seed = []
        for m in metrics_list:
            values = []
            for row in m["epochs"]:
                v = (row.get(key) or {}).get(sub) if sub else row.get(key)
                values.append(float(v) if v is not None else np.nan)
            if values:
                per_seed.append(values)
        length = min(map(len, per_seed))
        arr = np.array([c[:length] for c in per_seed], dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(arr, axis=0)
            last = float(np.nanmean(arr[:, -1]))
        curves[sub or key] = {
            "ep": list(range(1, length + 1)),
            "rows": arr.tolist(),
            "mean": mean.tolist(),
            "last": last,
            "visible": bool(np.isfinite(arr).any() and arr.max() > 0),
        }
    # Preserve the legacy degenerate all-constant result while allowing mixed seeds.
    frequencies = next(
        (np.array(c["freqs_hz"]) for c in final_cells if len(c["freqs_hz"]) > 1),
        np.zeros(1),
    )
    psds = np.stack(
        [
            np.array(c["psd"])
            if len(c["freqs_hz"]) == len(frequencies)
            else np.zeros_like(frequencies)
            for c in final_cells
        ]
    )
    band = (frequencies >= F_GAMMA_BAND_HZ[0]) & (frequencies <= F_GAMMA_BAND_HZ[1])
    return clean(
        {
            "header": header,
            "curves": curves,
            "psd": {
                "frequencies": frequencies[band].tolist(),
                "rows": psds[:, band].tolist(),
                "mean": psds.mean(axis=0)[band].tolist(),
            },
        }
    )


def weight_distributions(arrays):
    result = {}
    for direction, offset, canonical in (("ei", 0, 1 / 1024), ("ie", 2, 2 / 256)):
        per_network = [weight_summary(a[offset], a[offset + 1]) for a in arrays]
        initial = np.concatenate([a[offset].ravel() for a in arrays])
        trained = np.concatenate([a[offset + 1].ravel() for a in arrays])
        ni, nt = initial[initial > 0], trained[trained > 0]
        hi = float(
            max(
                ni.max() if ni.size else 0,
                nt.max() if nt.size else 0,
                canonical * 1.2,
                1e-12,
            )
        )
        bins = np.linspace(0, hi * 1.05, 50)
        result[direction] = {
            "bins": bins.tolist(),
            "initial": np.histogram(ni, bins)[0].tolist(),
            "trained": np.histogram(nt, bins)[0].tolist(),
            "has_initial": bool(ni.size),
            "has_trained": bool(nt.size),
            "stats": weight_summary(initial, trained),
            "per_network": per_network,
            "canonical": canonical,
        }
    return result


def raster(directory, config):
    path = directory if directory.is_file() else directory / "recording.npz"
    with np.load(path, allow_pickle=False) as raw:
        e, i = (raw[k] for k in ("spk_e", "spk_i"))
        if e.ndim == 3:
            e = e[:, 0, :]
        if i.ndim == 3:
            i = i[:, 0, :]
        e, i = e.astype(bool), i.astype(bool)
        rng = np.random.default_rng(42)
        ne, ni = min(200, e.shape[1]), min(50, i.shape[1])
        ei = np.sort(rng.choice(e.shape[1], ne, replace=False))
        ii = np.sort(rng.choice(i.shape[1], ni, replace=False))
        et, en = np.where(e[:, ei])
        it, inn = np.where(i[:, ii])
        # The old capture used training-config dt, not the rounded float32 sidecar.
        dt = float(config["dt"])
        return {
            "e_t": et * dt,
            "e_n": en,
            "i_t": it * dt,
            "i_n": inn,
            "n_e": e.shape[1],
            "n_i": i.shape[1],
            "n_e_plot": ne,
            "n_i_plot": ni,
            "label": int(raw["label"]),
            "t_ms": float(config["t_ms"]),
            "dt_ms": dt,
        }


def final_outcomes(endpoints, metrics):
    """Across-replicate SEM; contrast uses the unsmoothed final diagnostic."""
    values = {
        key: [row[key] for row in endpoints]
        for key in ("acc", "e_rate_hz", "i_rate_hz")
    }
    values["contrast"] = [m["epochs"][-1]["contrast"] for m in metrics]
    result = {}
    for key, rows in values.items():
        a = np.asarray(rows, dtype=float)
        if a.size < 2 or not np.isfinite(a).all():
            raise ValueError("final outcomes require at least two finite replicates")
        result[key] = {
            "values": a.tolist(),
            "n": int(a.size),
            "mean": float(a.mean()),
            "sem": float(a.std(ddof=1) / np.sqrt(a.size)),
        }
    return result


if __name__ == "__main__":
    main()
