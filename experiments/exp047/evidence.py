"""Validate the retained recipe, per-simulation metrics and report grids."""

import math

from pingstore.contracts import PingstoreError, load_json

from . import recipe


def configuration(cfg):
    if (
        not isinstance(cfg, dict)
        or cfg.get("profile") not in ("smoke", "production")
        or cfg != recipe.configuration(smoke=cfg["profile"] == "smoke")
    ):
        raise PingstoreError("exp047 scientific recipe differs")
    return cfg


def finite_rate(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise PingstoreError("exp047 rate must be finite and nonnegative")
    return float(value)


def simulation_config(document, cfg, item):
    expected = {
        "mode": "sim",
        "model": "ping",
        "input": "synthetic-spikes",
        "n_hidden": [cfg["n_e"]],
        "n_in": cfg["n_in"],
        "n_inh": item["n_i"],
        "ei_strength": cfg["g_ei_total"],
        "ei_ratio": item["g_ie_total"] / cfg["g_ei_total"],
        "w_in": [cfg["w_in_mean"]],
        "w_in_initial_zero_fraction": cfg["w_in_initial_zero_fraction"],
        "recurrent_initial_zero_fraction": 0.0,
        "spike_rate": cfg["input_rate_hz"],
        "n_batch": cfg["n_batch"],
        "t_ms": cfg["t_ms"],
        "dt": cfg["dt_ms"],
        "seed": item["seed"],
        "dales_law": True,
        "private_w_in": False,
        "scale_w_in": 1.0,
        "scale_w_ei": 1.0,
        "scale_w_ie": 1.0,
    }
    if (
        any(document.get(k) != v for k, v in expected.items())
        or document.get("load_weights") is not None
    ):
        raise PingstoreError(f"exp047 simulation configuration differs: {item['id']}")


def metric(document, cfg, item):
    expected = {
        "dt": cfg["dt_ms"],
        "t_ms": cfg["t_ms"],
        "n_in": cfg["n_in"],
        "n_hidden": cfg["n_e"],
        "n_inh": item["n_i"],
        "ei_strength": cfg["g_ei_total"],
        "ei_ratio": item["g_ie_total"] / cfg["g_ei_total"],
        "input_rate_hz": cfg["input_rate_hz"],
        "n_batch": cfg["n_batch"],
        "load_weights": None,
    }
    if (
        document.get("mode") != "probe"
        or document.get("model") != "ping"
        or document.get("config") != expected
    ):
        raise PingstoreError(f"exp047 metrics configuration differs: {item['id']}")
    e, i = finite_rate(document["rate_e_hz"]), finite_rate(document["rate_i_hz"])
    if document.get("rates_hz") != {"hid": e, "inh": i}:
        raise PingstoreError("exp047 population rate aliases disagree")
    return {
        "n_i": item["n_i"],
        "g_ie_total": item["g_ie_total"],
        "j_ie_synapse": item["g_ie_total"] / item["n_i"],
        "seed": item["seed"],
        "r_e_hz": e,
        "r_i_hz": i,
    }


def compute_contract(run):
    cfg = configuration(run.record["execution"].get("configuration"))
    if run.record["inputs"]:
        raise PingstoreError("exp047 initial compute must not have upstream inputs")
    expected = {"schema": "exp047.compute/v1", "recipe": cfg, "jobs": recipe.jobs(cfg)}
    if load_json(run.export / "evidence.json") != expected:
        raise PingstoreError("exp047 compute evidence differs from the recipe")
    expected_files = {
        "evidence.json",
        *(f"probe/{j['id']}/metrics.json" for j in recipe.jobs(cfg)),
    }
    actual_files = {
        str(p.relative_to(run.export))
        for p in run.export.rglob("*")
        if p.is_file() and not p.is_relative_to(run.export / "evidence")
    }
    if actual_files != expected_files:
        raise PingstoreError("exp047 compute metric grid differs")
    return cfg


def rows(export, provenance, cfg):
    result = {}
    for item in recipe.jobs(cfg):
        simulation_config(
            load_json(provenance / "simulations" / item["id"] / "config.json"),
            cfg,
            item,
        )
        result[item["id"]] = metric(
            load_json(export / "probe" / item["id"] / "metrics.json"), cfg, item
        )
    return result


def analysis(document, cfg):
    """Check saved table shape and numeric validity without recomputing estimators."""
    if (
        document.get("schema") != "exp047.analysis/v1"
        or document.get("recipe") != cfg
        or document.get("measurement") != recipe.MEASUREMENT
        or document.get("config") != recipe.report_config(cfg)
        or document.get("definition") != recipe.DEFINITION
    ):
        raise PingstoreError("exp047 analysis contract differs")
    controls = {
        "fixed_total": cfg["reference_g_ie"],
        "fixed_synapse": cfg["reference_j_ie"],
    }
    seen = {}
    for key in ("raw", "summary"):
        table = document.get(key, {})
        if set(table) != set(controls):
            raise PingstoreError("exp047 analysis control grid differs")
        for control, levels in controls.items():
            if set(table[control]) != {f"{level:.12g}" for level in levels}:
                raise PingstoreError("exp047 analysis coupling grid differs")
            for cells in table[control].values():
                if set(cells) != {str(n) for n in cfg["n_i_sweep"]}:
                    raise PingstoreError("exp047 analysis pool grid differs")
    for control, level, n_i, g_ie in recipe.conditions(cfg):
        rows_ = document["raw"][control][level][str(n_i)]
        if len(rows_) != len(cfg["seeds"]):
            raise PingstoreError("exp047 analysis seed grid differs")
        for row, seed in zip(rows_, cfg["seeds"], strict=True):
            expected = {
                "n_i": n_i,
                "g_ie_total": g_ie,
                "j_ie_synapse": g_ie / n_i,
                "seed": seed,
            }
            if set(row) != {*expected, "r_e_hz", "r_i_hz"} or any(
                row[k] != v for k, v in expected.items()
            ):
                raise PingstoreError("exp047 analysis row identity differs")
            finite_rate(row["r_e_hz"])
            finite_rate(row["r_i_hz"])
            identity = recipe.job(n_i, g_ie, seed)["id"]
            if identity in seen and seen[identity] != row:
                raise PingstoreError("exp047 shared simulation rows disagree")
            seen[identity] = row
        summary = document["summary"][control][level][str(n_i)]
        if set(summary) != {
            f"{metric_}_{stat}"
            for metric_ in ("r_e_hz", "r_i_hz")
            for stat in ("mean", "sd")
        }:
            raise PingstoreError("exp047 analysis summary fields differ")
        for value in summary.values():
            finite_rate(value)
    return document
