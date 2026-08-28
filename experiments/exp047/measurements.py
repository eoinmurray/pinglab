"""Pure aggregation of retained per-seed rates, preserving historical ordering."""

import numpy as np

from . import recipe


def summarise(rows):
    result = {}
    for key in ("r_e_hz", "r_i_hz"):
        values = np.asarray([row[key] for row in rows], dtype=float)
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_sd"] = float(values.std(ddof=1))
    return result


def analyse_rows(rows, cfg):
    raw, summary = (
        {"fixed_total": {}, "fixed_synapse": {}},
        {"fixed_total": {}, "fixed_synapse": {}},
    )
    for control, level, n_i, g_ie in recipe.conditions(cfg):
        selected = [rows[recipe.job(n_i, g_ie, seed)["id"]] for seed in cfg["seeds"]]
        raw[control].setdefault(level, {})[str(n_i)] = selected
        summary[control].setdefault(level, {})[str(n_i)] = summarise(selected)
    return {
        "schema": "exp047.analysis/v1",
        "recipe": cfg,
        "measurement": recipe.MEASUREMENT,
        "config": recipe.report_config(cfg),
        "definition": recipe.DEFINITION,
        "raw": raw,
        "summary": summary,
    }
