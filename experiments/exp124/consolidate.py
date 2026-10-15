"""Explicitly consolidate retained SHD training and inference for source pruning.

This is a user-authorized replacement of retained evidence, not new training or
ordinary phase assembly. Original records become historical execution metadata;
all scientific payloads needed downstream are copied into the new full run.
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
from experiments.exp124 import recipe
from experiments.exp124.rasters import measure
from pingstore.contracts import file_sha256, load_json
from pingstore.stages import operation_lock, source_run, stage_run
from snnlab.sim import load_training_checkpoint


def consolidate(identities, raster_identity, *, replace_sources=False):
    if not replace_sources:
        raise ValueError("self-contained consolidation requires explicit --replace-sources")
    with operation_lock(REPO / ".pingstore", exclusive=False):
        sources = {}
        for identity in identities:
            source = source_run(REPO / ".pingstore", identity, stage="compute", experiment=recipe.SLUG)
            models = source.record["execution"]["configuration"]["models"]
            if len(models) != 1 or models[0] in sources:
                raise ValueError("select exactly one separate training run per model")
            sources[models[0]] = source
        if set(sources) != set(recipe.MODELS):
            raise ValueError("COBA and PING training sources are required")
        first = sources["coba"]
        cfg = dict(first.record["execution"]["configuration"])
        cfg["models"] = list(recipe.MODELS)
        if not cfg.get("full_dataset") or cfg.get("training_engine") not in (
            "snnlab.sim.train", "snnlab.sim.execution.train"
        ):
            raise ValueError("expected full-data native training")
        for model, source in sources.items():
            candidate = dict(source.record["execution"]["configuration"], models=list(recipe.MODELS))
            if candidate != cfg:
                raise ValueError("training scientific configurations differ")
            for key in ("dataset", "snnlab_version"):
                if source.record["execution"][key] != first.record["execution"][key]:
                    raise ValueError(f"training sources have different {key}")
            if file_sha256(source.file("spikes.npz")) != file_sha256(first.file("spikes.npz")):
                raise ValueError("training stimulus payloads differ")
            unit = source.unit(model)
            checkpoint = load_training_checkpoint(unit)
            expected_updates = cfg["epochs"] * ((cfg["train_examples"] + cfg["batch_size"] - 1) // cfg["batch_size"])
            if checkpoint.completed_updates != expected_updates:
                raise ValueError("incomplete training checkpoint")
            rows = load_json(unit / "learning.json")["epochs"]
            if [row["epoch"] for row in rows] != list(range(cfg["epochs"] + 1)):
                raise ValueError("incomplete training history")
            with np.load(unit / "initial_weights.npz") as initial, np.load(unit / "final_weights.npz") as final:
                if set(final.files) != set(checkpoint.parameters) or set(final.files) != set(initial.files):
                    raise ValueError("checkpoint parameter coverage differs")
                for name in final.files:
                    if not np.isfinite(final[name]).all() or not np.array_equal(final[name], checkpoint.parameters[name].cpu().numpy()):
                        raise ValueError(f"invalid final checkpoint parameter: {name}")
                    if (not np.array_equal(initial[name], final[name])) != (name in recipe.TRAINABLE):
                        raise ValueError(f"unexpected learned/fixed parameter change: {name}")
        with np.load(first.unit("coba") / "initial_weights.npz") as a, np.load(sources["ping"].unit("ping") / "initial_weights.npz") as b:
            if any(not np.array_equal(a[name], b[name]) for name in recipe.TRAINABLE):
                raise ValueError("initial learned matrices differ")
        raster = source_run(REPO / ".pingstore", raster_identity, stage="compute",
                            experiment=recipe.SLUG, phase="sample-raster")
        if raster.record["inputs"] != {f"{m}_weights": s.reference for m, s in sources.items()}:
            raise ValueError("inference was not recorded from these exact trained weights")
        raster_cfg = raster.record["execution"]["configuration"]
        measure(raster, raster_cfg)
        roles = {**sources, "sample_raster": raster}
        with stage_run(REPO, recipe.SLUG, "compute", configuration=cfg, operation="consolidate") as run:
            mapping = [(first.file("spikes.npz"), "spikes.npz"), (raster.file("rasters.npz"), "rasters.npz")]
            for model, source in sources.items():
                for path in source.unit(model).iterdir():
                    if not path.is_file():
                        raise ValueError("expected a flat native checkpoint bundle")
                    mapping.append((path, f"{model}/{path.name}"))
            copied = {}
            for path, relative in mapping:
                target = run.export / relative
                if target.exists():
                    raise ValueError(f"duplicate consolidated output: {relative}")
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(path, target)
                digest = file_sha256(path)
                if file_sha256(target) != digest:
                    raise ValueError(f"copy verification failed: {relative}")
                copied[relative] = {"source_run": next(s.record["run_id"] for s in roles.values() if path.is_relative_to(s.export)),
                                    "source_file": str(path.relative_to(next(s.export for s in roles.values() if path.is_relative_to(s.export)))),
                                    "sha256": digest}
            run.record["execution"].update(
                dataset=first.record["execution"]["dataset"],
                snnlab_version=first.record["execution"]["snnlab_version"],
                models={m: s.record["execution"]["models"][m] for m, s in sources.items()},
                scientific_origin="paired-independent-workers",
                condition_origins={m: s.record["origin"] for m, s in sources.items()},
                sample_raster_configuration=raster_cfg,
                inference_metrics=raster.record["execution"].get("inference_metrics", {}),
                consolidation={"authorization": "explicit user request to merge and retain only compute/analyse/present",
                               "policy": "self-contained replacement; historical records are not live input dependencies",
                               "source_records": {role: s.record for role, s in roles.items()},
                               "copied_files": copied, "training_executed": False, "inference_executed": False})
            with (run.directory / "README.md").open("a") as handle:
                handle.write("\n## Consolidated evidence\n\nOriginal COBA/PING workers executed separately; this operation did not train or infer.\n"
                             "The user explicitly authorized replacing the source runs and retaining only the new three-stage chain.\n"
                             "All checkpoint, stimulus and raster payloads are self-contained; complete source records and byte mappings are preserved in execution metadata.\n"
                             + "".join(f"- {role}: `{s.record['run_id']}` ({s.record['payload_digest']})\n" for role, s in roles.items()))
            for source in roles.values():
                source.check_unchanged()
        return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--raster-source", required=True)
    parser.add_argument("--replace-sources", action="store_true")
    args = parser.parse_args()
    consolidate(args.source, args.raster_source, replace_sources=args.replace_sources)
