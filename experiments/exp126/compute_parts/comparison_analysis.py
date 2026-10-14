"""Compare explicit paired completed runs; never simulate or train."""

import numpy as np
from experiments.exp126.compute import REPO
from experiments.exp126.compute_parts.assembly import assembled_conditions
from experiments.exp126.recipe import SLUG
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run


def compare(identity):
    full = source_run(REPO / ".pingstore", identity, stage="compute", experiment=SLUG)
    sources = assembled_conditions(REPO, full)
    pairs = {}
    for source in sources.values():
        cfg = source.record["execution"]["configuration"]
        if (
            cfg.get("schema") not in ("exp126.matched/v2", "exp126.matched/v3")
            or source.record.get("scope") != "phase"
        ):
            raise ValueError(
                "comparison requires complete matched-protocol runs, not the original baseline"
            )
        seed, implementation = cfg["seed"], source.record["implementation"]
        if implementation not in (
            "snntorch",
            "snnlab",
        ) or implementation in pairs.setdefault(seed, {}):
            raise ValueError("unknown or duplicate implementation for seed")
        pairs[seed][implementation] = source
    if any(set(pair) != {"snntorch", "snnlab"} for pair in pairs.values()):
        raise ValueError("explicitly provide both implementations for every seed")
    records, table, retained = [], [], {}
    protocol = None
    for seed, pair in sorted(pairs.items()):
        torch_source, lab_source = pair["snntorch"], pair["snnlab"]
        cfg = torch_source.record["execution"]["configuration"]
        if cfg != lab_source.record["execution"]["configuration"]:
            raise ValueError("paired recipes differ")
        normalized = {key: value for key, value in cfg.items() if key != "seed"}
        if protocol is not None and protocol != normalized:
            raise ValueError("replicate recipes differ beyond seed")
        protocol = normalized
        executions = {name: source.record["execution"] for name, source in pair.items()}
        for field in (
            "device",
            "device_name",
            "cpu_threads",
            "versions",
            "dataset_sha256",
            "encoding_sha256",
        ):
            if executions["snntorch"][field] != executions["snnlab"][field]:
                raise ValueError(f"paired execution mismatch: {field}")
        if executions["snntorch"]["device"] != "cuda":
            raise ValueError("this comparison requires matched CUDA devices")
        for filename in ("initial_weights.npz", "dataset.npz"):
            with (
                np.load(torch_source.file(filename), allow_pickle=False) as a,
                np.load(lab_source.file(filename), allow_pickle=False) as b,
            ):
                if set(a.files) != set(b.files) or any(
                    not np.array_equal(a[key], b[key]) for key in a.files
                ):
                    raise ValueError(f"paired {filename} differs")
        differences = {}
        with (
            np.load(torch_source.file("agreement.npz"), allow_pickle=False) as a,
            np.load(lab_source.file("agreement.npz"), allow_pickle=False) as b,
        ):
            if (
                set(a.files) != set(b.files)
                or not np.array_equal(a["events"], b["events"])
                or not np.array_equal(a["labels"], b["labels"])
            ):
                raise ValueError("parity probe inputs or trace keys differ")
            for key in a.files:
                if key in ("events", "labels"):
                    continue
                if not np.isfinite(a[key]).all() or not np.isfinite(b[key]).all():
                    raise ValueError("nonfinite parity probe")
                differences[key] = dict(
                    max_abs=float(np.max(np.abs(a[key] - b[key]))),
                    within_tolerance=bool(
                        np.allclose(
                            a[key],
                            b[key],
                            atol=cfg["state_atol"],
                            rtol=cfg["state_rtol"],
                        )
                    ),
                )
                if key.endswith("spikes"):
                    differences[key]["disagreements"] = int(
                        np.count_nonzero(a[key] != b[key])
                    )
            if seed == min(pairs):
                for key in a.files:
                    if key not in ("events", "labels"):
                        retained["snntorch_" + key] = a[key].copy()
                        retained["snnlab_" + key] = b[key].copy()
        gradients = {}
        with (
            np.load(torch_source.file("gradients.npz"), allow_pickle=False) as a,
            np.load(lab_source.file("gradients.npz"), allow_pickle=False) as b,
        ):
            if set(a.files) != set(b.files):
                raise ValueError("gradient parameter sets differ")
            for key in a.files:
                delta = b[key].astype(float) - a[key].astype(float)
                if not np.isfinite(delta).all():
                    raise ValueError("nonfinite parameter gradient")
                relative = float(
                    np.linalg.norm(delta) / max(np.linalg.norm(a[key]), 1e-12)
                )
                gradients[key] = dict(
                    relative_l2=relative,
                    max_abs=float(np.max(np.abs(delta))),
                    within_tolerance=bool(
                        np.allclose(
                            a[key],
                            b[key],
                            atol=cfg["gradient_atol"],
                            rtol=cfg["gradient_rtol"],
                        )
                    ),
                )
                if seed == min(pairs):
                    retained["snntorch_gradient_" + key] = a[key].copy()
                    retained["snnlab_gradient_" + key] = b[key].copy()
        passed = (
            all(value["within_tolerance"] for value in differences.values())
            and all(
                value.get("disagreements", 0) == 0 for value in differences.values()
            )
            and all(value["within_tolerance"] for value in gradients.values())
        )
        learning = {
            name: load_json(source.file("learning.json"))
            for name, source in pair.items()
        }
        metrics = {}
        for name, source in pair.items():
            rows = learning[name]["epochs"]
            if [row["epoch"] for row in rows] != list(range(cfg["epochs"] + 1)):
                raise ValueError("incomplete training history")
            selected = min(rows[1:], key=lambda row: row["validation"]["loss"])["epoch"]
            if selected != learning[name]["selected_epoch"]:
                raise ValueError("checkpoint selection disagrees with validation")
            for row in rows:
                for split, count in (("train", 6000), ("validation", 1000)):
                    metric = row[split]
                    if (
                        metric["samples"] != count
                        or not np.isfinite(metric["loss"])
                        or not 0 <= metric["accuracy"] <= 1
                    ):
                        raise ValueError("invalid learning metrics")
            with np.load(
                source.file("test_predictions.npz"), allow_pickle=False
            ) as test:
                if test["predictions"].shape != (10000,) or not np.array_equal(
                    test["indices"], np.arange(10000)
                ):
                    raise ValueError("incomplete test predictions")
                accuracy = float(np.mean(test["labels"] == test["predictions"]))
                if abs(accuracy - learning[name]["test"]["accuracy"]) > 1e-12:
                    raise ValueError("test accuracy disagrees with predictions")
                confusion = np.zeros((10, 10), dtype=int)
                np.add.at(confusion, (test["labels"], test["predictions"]), 1)
                learning[name]["confusion_matrix"] = confusion.tolist()
            timing = executions[name]["timing"]
            times = np.asarray(timing["inference_seconds"])
            if (
                times.shape != (cfg["inference_repeats"],)
                or not np.isfinite(times).all()
                or (times <= 0).any()
            ):
                raise ValueError("invalid inference timings")
            metrics[name] = dict(
                test_accuracy=accuracy,
                selected_epoch=selected,
                optimization_seconds=timing["optimization_seconds"],
                training_protocol_seconds=timing["training_protocol_seconds"],
                test_inference_seconds=float(np.median(times)),
                test_images_per_second=10000 / float(np.median(times)),
            )
            for operation, values in executions[name]["microbenchmark"][
                "seconds"
            ].items():
                values = np.asarray(values)
                if (
                    values.shape != (cfg["timing_repeats"],)
                    or not np.isfinite(values).all()
                    or (values <= 0).any()
                ):
                    raise ValueError("invalid microbenchmark repeats")
                metrics[name][operation + "_seconds"] = float(np.median(values))
            metrics[name]["peak_allocated_bytes"] = timing["peak_allocated_bytes"]
        speedups = {
            key: metrics["snntorch"][key] / metrics["snnlab"][key]
            for key in metrics["snntorch"]
            if key.endswith("_seconds")
        }
        validation_policy = None
        if cfg.get("schema") == "exp126.matched/v3":
            execution = executions["snnlab"]
            if execution.get("extension_validation") != "structural":
                raise ValueError(
                    "upgraded comparison did not use structural validation"
                )
            agreement = execution.get("validation_policy_agreement")
            if agreement != {"scores_equal": True, "gradients_equal": True}:
                raise ValueError("strict and default policy computations did not agree")
            strict = execution.get("strict_microbenchmark", {}).get("seconds", {})
            if set(strict) != {
                "encoding",
                "forward",
                "forward_backward",
                "optimization_step",
            }:
                raise ValueError("missing strict-policy timing reference")
            for values in strict.values():
                if (
                    len(values) != cfg["timing_repeats"]
                    or not np.isfinite(values).all()
                    or min(values) <= 0
                ):
                    raise ValueError("invalid strict-policy timing repeats")
            validation_policy = {
                "default": execution["extension_validation"],
                "initial_agreement": agreement,
                "strict_median_seconds": {
                    name: float(np.median(values)) for name, values in strict.items()
                },
                "profiles": execution["validation_policy_profiles"],
            }
        for key in metrics["snntorch"]:
            a, b = metrics["snntorch"][key], metrics["snnlab"][key]
            table.append(
                dict(
                    seed=seed,
                    measurement=key,
                    snntorch=a,
                    snnlab=b,
                    speedup=speedups.get(key) if passed else None,
                )
            )
        for key, agreement in differences.items():
            table.append(
                dict(
                    seed=seed,
                    measurement="initial_" + key + "_max_abs_error",
                    snntorch=0,
                    snnlab=agreement["max_abs"],
                    speedup=None,
                )
            )
            if "disagreements" in agreement:
                table.append(
                    dict(
                        seed=seed,
                        measurement="initial_" + key + "_disagreements",
                        snntorch=0,
                        snnlab=agreement["disagreements"],
                        speedup=None,
                    )
                )
        for key, agreement in gradients.items():
            table.append(
                dict(
                    seed=seed,
                    measurement=key + "_gradient_relative_l2_error",
                    snntorch=0,
                    snnlab=agreement["relative_l2"],
                    speedup=None,
                )
            )
        records.append(
            dict(
                seed=seed,
                agreement_passed=passed,
                state_agreement=differences,
                gradient_agreement=gradients,
                learning=learning,
                metrics=metrics,
                validation_policy=validation_policy,
                speedups=speedups if passed else {},
                run_ids={
                    name: source.record["run_id"] for name, source in pair.items()
                },
            )
        )
    result = dict(
        schema="exp126.comparison/v1",
        pairs=records,
        all_agreement_passed=all(r["agreement_passed"] for r in records),
        protocol=protocol,
        table=table,
        probe_seed=min(pairs),
        timing_interpretation="custom snnlab graph executor versus snnTorch; declared extension validation policy; separate equal-model GPU workers",
    )
    with stage_run(
        REPO,
        SLUG,
        "analyse",
        inputs={"compute": full},
        configuration={"schema": "exp126.comparison/v1", "seeds": sorted(pairs)},
    ) as run:
        write_json_atomic(run.export / "results.json", result)
        np.savez_compressed(run.export / "agreement.npz", **retained)
        # Raw execution measurements remain metadata; results contain display projections.
        run.record["execution"]["source_benchmarks"] = {
            source.record["run_id"]: source.record["execution"]
            for source in sources.values()
        }
    return run.run_id
