"""Adapt validated digit measurements to the shared physical-time renderer.

Read native snapshots to verify each output clock and assemble figure arrays
without changing spike times. Exclude output neurons from the E/I error panels
while retaining their classification counts in the analysis evidence."""


def output_clock(source, mode, case, cfg):
    """Read the actual native graph clock; all COBA-LIF populations share it."""
    from pingstore.contracts import load_json

    if case.startswith("continuous"):
        diagnostics = source.record["execution"]["cases"][f"{mode}-{case}"]
        if "COBA-LIF" not in diagnostics["readout"]:
            raise ValueError("continuous case lacks the continuous COBA-LIF output")
        return dict(network_dt_ns=None, output_dt_ns=None, kind="continuous COBA-LIF")
    expected = (
        cfg["input_grid_ns"]
        if case == "native"
        else next(row["dt_ns"] for row in cfg["cases"] if row["id"] == case)
    )
    snapshot = load_json(source.file(f"{mode}-{case}", "start", "snapshot.json"))
    graph = snapshot["network"]
    populations = {row["id"]: row["neuron"] for row in graph["populations"]}
    if populations["readout"]["kind"] != "coba_lif":
        raise ValueError("expected the trained COBA-LIF readout")
    for actual in (
        graph["timebase"]["dt"]["value"],
        snapshot["execution"]["timestep_ms"],
    ):
        if actual * 1e6 != expected:
            raise ValueError("readout/network timestep differs from the requested case")
    return dict(network_dt_ns=expected, output_dt_ns=expected, kind="discrete COBA-LIF")


def figure_summary(cfg, audit):
    cases = [dict(**row, label=f"{row['dt_ns'] / 1e6:g} ms") for row in cfg["cases"]]
    cases.append(dict(id="continuous", dt_ns=None, label="Continuous"))
    return dict(
        schema="exp142.physical-time-figures/v1",
        configuration=cfg,
        target_e=cfg["target_e"],
        reset_guard_ms=cfg["guard_ms"],
        cases=cases,
        input_image=dict(shape=[8, 8], name="Scikit digits image"),
        native_anchor_label="Native",
        native_anchor_legend="Native 0.2 ms",
        input={
            **audit,
            "plot_annotation": f"{cfg['input_rate_hz']:g} Hz maximum per pixel; {audit['event_count']} saved spikes; exact replay in both models and all {len(audit['cases'])} executions",
        },
        readout_policy="matched-spiking",
        models={},
    )


def add_model(summary, destination, mode, cfg, ni, models, recordings, measurements):
    """Exclude readout neurons from E/I error markers; keep their trained scores."""
    hidden = cfg["n_e"] + ni
    alias = dict(
        g_exc_guarded_mae_ns="ampa_mae_ns",
        g_inh_guarded_mae_ns="gaba_mae_ns",
        v_reset_aware_mae_mv="voltage_mae_mv",
        spike_median_shift_ms="spike_shift_ms",
        g_retained_sample_fraction="conductance_fraction",
        v_retained_sample_fraction="voltage_fraction",
    )
    rows = []
    for case in [
        dict(id="native", dt_ns=cfg["input_grid_ns"], label="Native 0.2 ms"),
        *summary["cases"],
    ]:
        name = case["id"]
        recording, metrics = recordings[name], measurements[name]
        populations = {
            key: value
            for key, value in models[name]["populations"].items()
            if key in ("E", "I")
        }
        if not ni:
            populations["I"] = dict(neurons=0, spikes=0, timing_eligible_neurons=0)
        rows.append(
            dict(
                **case,
                classification=models[name]["classification"],
                populations=populations,
                spike_count=sum(row["spikes"] for row in populations.values()),
            )
        )
        stem = f"{mode}__{name}__"
        for key in ("g_time_ns", "v_time_ns", "readout_scores", "class_scores"):
            destination[stem + key] = recording[key]
        for key in ("g_exc_ns", "g_inh_ns", "voltage_mv"):
            destination[stem + key] = recording[key][:, :hidden]
        selected = recording["spike_neurons"] < hidden
        destination[stem + "spike_time_ns"] = recording["spike_time_ns"][selected]
        destination[stem + "spike_neurons"] = recording["spike_neurons"][selected]
        for output, input_key in alias.items():
            destination[stem + output] = metrics[input_key][:hidden]
        # Preserve full root resolution. Rendering never rounds these events.
        if (
            name == "continuous"
            and destination[stem + "spike_time_ns"].dtype.kind != "f"
        ):
            raise ValueError("continuous roots must remain floating point")
    summary["models"][mode] = dict(native_anchor=rows[0], cases=rows[1:], n_i=ni)
