"""Export a sampled browser replay of recorded activity, without simulation."""

import numpy as np


def circuit_data(recording, weights, cfg, view, *, measurements):
    dt = cfg["dt_ms"]
    first = round(view["start_ms"] / dt)
    stop = round(view["end_ms"] / dt)
    selections = {
        pop: np.linspace(0, cfg[f"n_{pop}"] - 1, count, dtype=int)
        for pop, count in (("e", min(160, cfg["n_e"])), ("i", min(40, cfg["n_i"])))
    }
    groups = {}
    for pop, indices in selections.items():
        for name, key in ((pop, f"spk_{pop}"), (f"private_{pop}", f"private_{pop}")):
            values = recording[key][first:stop, indices]
            steps, cells = np.nonzero(values)
            groups[name] = {
                "indices": indices.tolist(),
                # Integer steps preserve exact event timing and simultaneous counts.
                "events": np.column_stack((steps, cells, values[steps, cells]))
                .astype(int)
                .tolist(),
            }
    edges = []
    for src in ("e", "i"):
        for dst in ("e", "i"):
            matrix = weights[f"{src.upper()}_to_{dst.upper()}.weight"]
            if matrix.shape != (cfg[f"n_{src}"], cfg[f"n_{dst}"]):
                raise ValueError("connection matrix must have source then target axes")
            selected = matrix[np.ix_(selections[src], selections[dst])]
            source, target = np.nonzero(selected)
            # A deterministic display subset; every displayed edge actually exists.
            take = np.linspace(0, len(source) - 1, min(80, len(source)), dtype=int)
            edges.extend([src, int(source[k]), dst, int(target[k])] for k in take)
    means = {}
    for key in ("mean_g_e", "mean_g_i", "mean_v_e", "mean_v_i"):
        values = np.asarray(measurements[key])
        if values.shape != (round(cfg["t_ms"] / dt),) or not np.isfinite(values).all():
            raise ValueError(f"invalid recorded population mean: {key}")
        means[key] = values[first:stop].tolist()
    conductance_max = max(
        0.001, max(max(means[key]) for key in ("mean_g_e", "mean_g_i")) * 1.15
    )
    rates = {}
    for pop in ("e", "i"):
        values = np.asarray(measurements[f"rate_private_{pop}"])
        if (
            values.shape != (round(cfg["t_ms"] / dt),)
            or not np.isfinite(values).all()
            or (values < 0).any()
        ):
            raise ValueError(f"invalid recorded afferent-rate protocol: {pop}")
        rates[pop] = values[first:stop].tolist()
    return {
        "schema": "exp130.circuit/v3",
        "dt_ms": dt,
        "duration_ms": (stop - first) * dt,
        "start_ms": view["start_ms"] - cfg.get("burn_in_ms", 0),
        "delay_ms": cfg["delay_ms"],
        "population_sizes": {pop: cfg[f"n_{pop}"] for pop in ("e", "i")},
        "afferents_per_neuron": cfg["external_afferents_per_neuron"],
        "groups": groups,
        "edges": edges,
        "means": means,
        "piston_scales": {
            "conductance_us": [0, conductance_max],
            "voltage_mv": [-80, -50],
        },
        "afferent_rates_hz": rates,
    }
