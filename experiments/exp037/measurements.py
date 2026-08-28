"""Retained exp037 estimators and raster coordinate selection."""

import numpy as np

from . import recipe


def summarize_accuracy(rows: list[dict], x_key: str) -> tuple[np.ndarray, ...]:
    """Return x, across-seed mean accuracy, and sample SD for a curve."""
    xs = sorted({float(row[x_key]) for row in rows})
    means = []
    sds = []
    for x in xs:
        values = [float(row["acc"]) for row in rows if float(row[x_key]) == x]
        means.append(float(np.mean(values)))
        sds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
    return np.asarray(xs), np.asarray(means), np.asarray(sds)


def summarize_perturbation_rows(rows: list[dict]) -> list[dict]:
    """Create publication rows from the per-seed perturbation measurements."""
    summary = []
    keys = sorted({(row["model"], row["mode"], float(row["level"])) for row in rows})
    for model, mode, level in keys:
        selected = [
            row
            for row in rows
            if row["model"] == model
            and row["mode"] == mode
            and float(row["level"]) == level
        ]
        acc = np.asarray([float(row["acc"]) for row in selected])
        rate = np.asarray([float(row["e_rate_hz"]) for row in selected])
        summary.append(
            {
                "model": model,
                "mode": mode,
                "level": level,
                "acc": float(acc.mean()),
                "acc_sd": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
                "e_rate_hz": float(rate.mean()),
                "e_rate_hz_sd": float(rate.std(ddof=1)) if len(rate) > 1 else 0.0,
                "seeds": [int(row["seed"]) for row in selected],
                "n_total_per_seed": [int(row["n_total"]) for row in selected],
            }
        )
    return summary


def baseline_rows(histories):
    rows = []
    for cell in recipe.bank_cells():
        name = cell["cell_name"]
        m = histories[name]
        last = m["epochs"][-1]
        rows.append(
            {
                "cell_name": name,
                "model": cell["model"],
                "rate_target_display": recipe.rate_target_display(
                    cell["rate_target_hz"]
                ),
                "rate_target_hz": cell["rate_target_hz"],
                "seed": cell["seed"],
                "best_acc": float(m["best_acc"]),
                "best_epoch": int(m["best_epoch"]),
                "final_acc": float(last["acc"]),
                "rate_e": float(last.get("rate_e") or 0.0),
            }
        )
    return rows


def perturbation_row(metrics, job):
    rates = metrics.get("rates_hz", {})
    hid = max((k for k in rates if k.startswith("hid")), default=None)
    return {
        "mode": job["mode"],
        "level": job["level"],
        "acc": float(metrics["best_acc"]),
        "e_rate_hz": float(rates.get(hid, 0.0)) if hid else 0.0,
        "n_total": int(metrics.get("n_total", 0)),
        "model": job["model"],
        "seed": job["seed"],
    }


def plot_data(rows, points):
    base = {}
    for model in recipe.MODELS:
        rs = [
            r["rate_e"]
            for r in rows
            if r["model"] == model and r["rate_target_hz"] is None
        ]
        base[model] = sum(rs) / len(rs) if rs else 0.0
    add = [
        {"model": r["model"], "pct": r["level"] / base[r["model"]], "acc": r["acc"]}
        for r in points
        if r["mode"] == "add" and base.get(r["model"], 0.0) > 0
    ]
    use_pct = bool(add)
    panels = {}
    for mode in ("drop", "add"):
        panel = {}
        for model in recipe.MODELS:
            rs = [
                r
                for r in (add if mode == "add" and use_pct else points)
                if r["model"] == model and ("mode" not in r or r["mode"] == mode)
            ]
            xs, means, sds = summarize_accuracy(
                rs, "pct" if mode == "add" and use_pct else "level"
            )
            if mode == "drop" or use_pct:
                xs = xs * 100
            panel[model] = {
                "x": xs.tolist(),
                "mean": means.tolist(),
                "lo": (means - sds).tolist(),
                "hi": (means + sds).tolist(),
            }
        panels[mode] = panel
    return {
        "baseline_e_rate_hz": base,
        "add_pct_rows": add,
        "use_pct": use_pct,
        "panels": panels,
    }


def raster(directory, train, job):
    with np.load(directory / "snapshot.npz", allow_pickle=False) as d:
        e_full, i_full = d["spk_e"], d["spk_i"]
        if e_full.ndim == 3:
            e_full = e_full[:, 0, :]
        if i_full.ndim == 3:
            i_full = i_full[:, 0, :]
        rate = float(e_full.sum() / (e_full.shape[1] * (float(train["t_ms"]) / 1000.0)))
        rng = np.random.default_rng(0)
        ei = np.sort(
            rng.choice(e_full.shape[1], recipe.EI_RASTER_N_E_PLOT, replace=False)
        )
        ii = np.sort(
            rng.choice(i_full.shape[1], recipe.EI_RASTER_N_I_PLOT, replace=False)
        )
        et, en = np.where(e_full[:, ei].astype(bool))
        it, inn = np.where(i_full[:, ii].astype(bool))
        t = np.arange(e_full.shape[0]) * float(train["dt"])
        return {
            "model": job["model"],
            "seed": job["seed"],
            "mode": job["mode"],
            "level": float(job["level"]),
            "e_rate_hz": rate,
            "label": int(d["label"]),
            "dt": float(train["dt"]),
            "t_ms": float(train["t_ms"]),
            "e_t": t[et],
            "e_n": en,
            "i_t": t[it],
            "i_n": inn + recipe.EI_RASTER_N_E_PLOT + 6,
        }
