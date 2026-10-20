"""Execute frozen-model controls, timestep sweeps and continuous refinements.

The compute stage supplies validated inputs and a fresh Pingstore run. Each
execution retains its own evidence, while source replay and refinement checks
must pass before the compute stage can complete.
"""

import numpy as np
from experiments.exp142.custom.continuous import simulate_continuous
from experiments.exp142.custom.discrete import execute
from experiments.exp142.custom.inputs import event_digest
from experiments.exp142.custom.metrics import (
    arrays,
    check_anchor,
    refinement_check,
    validate,
)
from experiments.exp142.custom.source import source_recording
from experiments.exp142.custom.external_artifacts import external_execution


def simulate_models(run, source, models, source_events, cfg, audit, label):
    anchors, refinements = {}, {}
    for mode, weights in models.items():
        ni = cfg["n_i"] if mode == "ping" else 0
        originals = source_recording(source, mode)
        # Preserve the original 25 Hz reproduction control separately from
        # all requested-rate inference, including the native-clock case.
        control_dir = run.export / f"{mode}-source-control"
        control, control_diagnostics = execute(
            source_events,
            weights,
            mode,
            cfg,
            cfg["input_grid_ns"],
            control_dir,
            matched=False,
        )
        validate(control, cfg, ni, sample_dt_ns=cfg["native_sample_dt_ns"])
        anchors[mode] = check_anchor(control, originals, cfg, ni)
        np.savez_compressed(control_dir / "recording.npz", **control)
        run.record["execution"].setdefault("cases", {})[f"{mode}-source-control"] = (
            control_diagnostics
        )
        conditions = [("native", cfg["input_grid_ns"], False)] + [
            (row["id"], row["dt_ns"], True) for row in cfg["cases"]
        ]
        continuous = {}
        for case, dt_ns, matched in conditions + [
            ("continuous", None, True),
            ("continuous-refined", None, True),
        ]:
            unit = f"{mode}-{case}"
            directory = run.export / unit
            saved = arrays(run.export / "input.npz")
            replay = {key: saved[key] for key in ("time_ns", "batches", "channels")}
            if event_digest(replay) != audit["input_event_sha256"]:
                raise ValueError("physical input coordinates changed")
            if dt_ns is not None:
                recording, diagnostics = execute(
                    replay,
                    weights,
                    mode,
                    cfg,
                    dt_ns,
                    directory,
                    matched=matched,
                    diagram=case == cfg["cases"][0]["id"],
                )
            else:
                solver = cfg["solver"] if case == "continuous" else cfg["refinement"]
                w = {key: value.copy() for key, value in weights.items()}
                if not ni:
                    w.update(
                        {
                            "E_to_I.weight": np.zeros((cfg["n_e"], 0), np.float32),
                            "I_to_E.weight": np.zeros((0, cfg["n_e"]), np.float32),
                        }
                    )
                promoted = {key: value.astype(np.float64) for key, value in w.items()}
                promoted.update(
                    {key: np.float64(value) for key, value in cfg["physics"].items()}
                )
                with external_execution(
                    directory,
                    backend="SciPy DOP853 hybrid COBA-LIF (E/I/readout)",
                    configuration={"physics": cfg["physics"], "solver": solver},
                    parameters=lambda: promoted,
                ):
                    recording, diagnostics = simulate_continuous(
                        replay,
                        w,
                        defaults=cfg["physics"],
                        duration_ms=cfg["duration_ns"] / 1e6,
                        sample_dt_ms=cfg["sample_dt_ns"] / 1e6,
                        **solver,
                    )
                continuous[case] = recording
            validate(
                recording,
                cfg,
                ni,
                sample_dt_ns=cfg["native_sample_dt_ns"] if case == "native" else None,
            )
            if (
                dt_ns is not None
                and diagnostics["input_event_sha256"] != audit["input_event_sha256"]
            ):
                raise ValueError("native input resolution changed physical coordinates")
            np.savez_compressed(directory / "recording.npz", **recording)
            audit["cases"][unit] = dict(
                event_count=len(replay["time_ns"]),
                input_equivalent=True,
                input_event_sha256=event_digest(replay),
                max_time_error_ns=0,
            )
            run.record["execution"].setdefault("cases", {})[unit] = diagnostics
            print(
                f"{unit}: prediction={recording['class_scores'].argmax()}, label={label}, counts={recording['class_scores'].tolist()}",
                flush=True,
            )
        refinements[mode] = refinement_check(
            continuous["continuous"], continuous["continuous-refined"], cfg, ni
        )
    return dict(anchors=anchors, refinement=refinements)
