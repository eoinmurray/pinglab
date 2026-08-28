"""Preserved paired pool-size controls, with no execution on import."""

SLUG = "exp047"
FIGURES = ("pool_size_controls.svg", "pool_size_controls.pdf")
DEFINITION = "j_ie_synapse = g_ie_total / n_i"
MEASUREMENT = {
    "schema": "exp047.measurement/v1",
    "aggregation": "seed mean and sample SD",
    "sd_ddof": 1,
}


def configuration(*, smoke=False):
    return {
        "schema": "exp047.recipe/v1",
        "profile": "smoke" if smoke else "production",
        "n_e": 1024,
        "n_in": 784,
        "n_i_sweep": [16, 64, 256],
        "n_i_reference": 256,
        "reference_g_ie": [1.0, 2.0, 4.0],
        "reference_j_ie": [1.0 / 256, 2.0 / 256, 4.0 / 256],
        "g_ei_total": 1.0,
        "input_rate_hz": 25.0,
        "t_ms": 200.0 if smoke else 500.0,
        "dt_ms": 0.1,
        "n_batch": 2 if smoke else 8,
        "seeds": [40, 41] if smoke else [40, 41, 42],
        "w_in_mean": 1.2,
        "w_in_initial_zero_fraction": 0.95,
    }


def report_config(cfg):
    """Keep the historical article's configuration interface unchanged."""
    return {
        k: v
        for k, v in cfg.items()
        if k not in {"schema", "profile", "w_in_mean", "w_in_initial_zero_fraction"}
    }


def conditions(cfg):
    for control, key in (
        ("fixed_total", "reference_g_ie"),
        ("fixed_synapse", "reference_j_ie"),
    ):
        for level in cfg[key]:
            for n_i in cfg["n_i_sweep"]:
                g_ie = level if control == "fixed_total" else n_i * level
                yield control, f"{level:.12g}", n_i, g_ie


def job(n_i, g_ie, seed):
    return {
        "id": f"nI{n_i}_g{g_ie:.8g}_s{seed}",
        "n_i": n_i,
        "g_ie_total": g_ie,
        "seed": seed,
    }


def jobs(cfg):
    seen, result = set(), []
    for _, _, n_i, g_ie in conditions(cfg):
        for seed in cfg["seeds"]:
            key = (n_i, round(g_ie, 12), seed)
            if key not in seen:
                seen.add(key)
                result.append(job(n_i, g_ie, seed))
    return result


def simulation_args(cfg, item, output):
    return [
        "sim",
        "--input",
        "synthetic-spikes",
        "--model",
        "ping",
        "--n-hidden",
        str(cfg["n_e"]),
        "--n-in",
        str(cfg["n_in"]),
        "--n-inh",
        str(item["n_i"]),
        "--ei-strength",
        str(cfg["g_ei_total"]),
        "--ei-ratio",
        str(item["g_ie_total"] / cfg["g_ei_total"]),
        "--w-in",
        str(cfg["w_in_mean"]),
        "--w-in-initial-zero-fraction",
        str(cfg["w_in_initial_zero_fraction"]),
        "--input-rate",
        str(cfg["input_rate_hz"]),
        "--n-batch",
        str(cfg["n_batch"]),
        "--t-ms",
        str(cfg["t_ms"]),
        "--dt",
        str(cfg["dt_ms"]),
        "--seed",
        str(item["seed"]),
        "--out-dir",
        str(output),
    ]
