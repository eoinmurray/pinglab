"""Cost/latency frontiers at a population-accuracy target."""

import numpy as np


def measure_target_accuracy(tradeoffs, trials, target_pct):
    """Select empirical first crossings, then filter the cost/time frontier."""
    settings = []
    for tau in tradeoffs["design"]["tau_gaba_ms"]:
        frequencies = trials["frequency_hz"][trials["tau_gaba_ms"] == tau]
        accepted = frequencies[np.isfinite(frequencies)]
        if not len(accepted):
            raise ValueError(f"No accepted I-volley frequency at decay {tau:g}")
        settings.append(dict(tau_gaba_ms=tau, mean_frequency_hz=float(accepted.mean()),
                             accepted_frequency_trials=len(accepted), total_trials=len(frequencies)))
    points = []
    for setting in settings:
        reached = [r for r in tradeoffs["rows"]
                   if r["tau_gaba_ms"] == setting["tau_gaba_ms"]
                   and r["accuracy"] * 100 >= target_pct]
        if reached:
            points.append(dict(min(reached, key=lambda r: r["deadline_ms"]), **{
                key: value for key, value in setting.items() if key != "tau_gaba_ms"}))
    for a in points:
        a["on_frontier"] = not any(
            b["mean_network_spikes"] <= a["mean_network_spikes"]
            and b["deadline_ms"] <= a["deadline_ms"]
            and (b["mean_network_spikes"] < a["mean_network_spikes"]
                 or b["deadline_ms"] < a["deadline_ms"])
            for b in points)
    return dict(
        schema="exp123.target.accuracy/v1", target_accuracy_pct=target_pct,
        frequency_by_setting=settings, points=points,
        frontier=sorted([p for p in points if p["on_frontier"]],
                        key=lambda p: p["mean_network_spikes"]),
        unreachable_tau_gaba_ms=[s["tau_gaba_ms"] for s in settings
                                 if not any(p["tau_gaba_ms"] == s["tau_gaba_ms"] for p in points)],
        measurement=dict(
            crossing="first measured 1 ms deadline with population accuracy >= target; no interpolation; accuracy need not stay above target",
            cost="mean cumulative E+I+output spikes excluding input at the common crossing deadline; all equally sampled images and draws, including incorrect trials",
            frontier="no other first-crossing setting has both no greater mean spikes and no greater elapsed time, with at least one strict improvement; only frontier settings displayed",
            frequency="mean accepted per-trial I-volley interval frequency over full 200 ms; missing frequencies excluded from colour summary only",
            limitations="one frozen network; target and frontier evaluated on the same test sample; no individual stopping rule, physical energy estimate or uncertainty intervals",
        ),
    )
