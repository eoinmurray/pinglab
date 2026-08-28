"""Import the explicitly approved Gold-2 selection without simulation or publication."""

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp033.import_gold2 import producer
from experiments.exp041.import_gold2 import safe_path
from experiments.exp054 import evidence, historical, inputs, measurements, recipe
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)

BASE = "provenance/source-records/base"
DERIVED = "derived/artifacts/data/exp054"
PROBE = "state/experiments/exp054/probe"
METADATA = ("run.json", "inventory.json", "lineage.json")


def selected_paths():
    return {
        *METADATA,
        *(f"{DERIVED}/{n}" for n in ("numbers.json", "_manifest.json", "run.sh")),
        *(
            f"{BASE}/{n}"
            for n in (
                "run.json",
                "collection-plan.json",
                "submissions/collection-submission.json",
                "collection-status/exp054.json",
                "logs/collection/ggs-exp054_33913631.out",
                "logs/collection/ggs-exp054_33913631.err",
            )
        ),
        *(
            f"{PROBE}/{j['id']}/{n}"
            for j in recipe.jobs(recipe.configuration())
            for n in (
                "config.json",
                "metrics.json",
                "output.log",
                "rasters.npz",
                "run.jsonl",
                "run.sh",
            )
        ),
    }


def verify_files(archive, plan):
    rows = plan["source_files"]
    if (
        len(rows) != len(selected_paths())
        or {r["path"] for r in rows} != selected_paths()
        or plan["source_file_count"] != len(rows)
        or sum(r["size_bytes"] for r in rows) != plan["source_bytes"]
    ):
        raise PingstoreError("selection differs from approved exp054 scope")
    indexed = {
        r["path"]: r for r in load_json(safe_path(archive, "inventory.json"))["files"]
    }
    for row in rows:
        path = safe_path(archive, row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != row["size_bytes"]
            or file_sha256(path) != row["sha256"]
        ):
            raise PingstoreError("changed archive evidence: " + row["path"])
        if row["path"] not in METADATA and (
            row["path"] not in indexed
            or any(indexed[row["path"]][k] != row[k] for k in ("size_bytes", "sha256"))
        ):
            raise PingstoreError("archive inventory disagrees with selection")


def members(path):
    with zipfile.ZipFile(path) as z:
        return [
            {
                "name": n,
                "size_bytes": len(z.read(n)),
                "sha256": hashlib.sha256(z.read(n)).hexdigest(),
            }
            for n in z.namelist()
        ]


def prepare(archive, plan, code_directory, live_directory):
    if (
        plan.get("schema") != "exp054.selective-import-plan/v1"
        or plan["archive"] != "r2://pinglab/campaigns/gold-2"
        or load_json(safe_path(archive, "run.json")).get("archive", {}).get("uri")
        != plan["archive"]
    ):
        raise PingstoreError("unexpected exp054 historical archive")
    verify_files(archive, plan)
    for name in METADATA:
        if (
            safe_path(live_directory, "live-" + name).read_bytes()
            != safe_path(archive, name).read_bytes()
        ):
            raise PingstoreError("live R2 metadata differs from cache")
    for row in plan["producer_code"]:
        path = safe_path(code_directory, row["audit_file"])
        if (
            path.stat().st_size != row["size_bytes"]
            or file_sha256(path) != row["sha256"]
        ):
            raise PingstoreError("historical producer code changed")
    prod = producer(archive, "exp054", "33913631", plan["producer"])
    if prod != plan["producer"]:
        raise PingstoreError("historical producer differs from approved evidence")
    old = load_json(archive / DERIVED / "numbers.json")
    provenance = old["collection_provenance"]
    if (
        provenance["source_git_commit"] != prod["git_commit"]
        or provenance["campaign_id"] != prod["campaign"]
    ):
        raise PingstoreError("historical summary producer differs")
    cfg = recipe.configuration()
    recordings = {r["path"]: r for r in plan["retained_recordings"]}
    expected = {f"{PROBE}/{j['id']}/rasters.npz" for j in recipe.jobs(cfg)}
    if (
        len(recordings) != len(plan["retained_recordings"])
        or set(recordings) != expected
    ):
        raise PingstoreError("historical recordings differ from complete recipe")
    for item in recipe.jobs(cfg):
        directory = archive / PROBE / item["id"]
        evidence.simulation_config(load_json(directory / "config.json"), cfg, item)
        evidence.raster(directory / "rasters.npz", cfg)
        if (
            members(directory / "rasters.npz")
            != recordings[f"{PROBE}/{item['id']}/rasters.npz"]["members"]
        ):
            raise PingstoreError("historical NPY members differ from approved plan")
    return prod


def import_subset(archive, plan_path, plan_sha256, code_directory, live_directory):
    archive, plan_path, code_directory, live_directory = map(
        Path, (archive, plan_path, code_directory, live_directory)
    )
    if file_sha256(plan_path) != plan_sha256:
        raise PingstoreError("approved exp054 import plan changed")
    plan = load_json(plan_path)
    prod = prepare(archive, plan, code_directory, live_directory)
    sources, ancestry = {}, {}
    if set(plan["upstream_references"]) != {"mean_field", "frequencies"}:
        raise PingstoreError("unexpected exp054 import upstream roles")
    for role, stage, experiment in (
        ("mean_field", "compute", "exp033"),
        ("frequencies", "analyse", "exp041"),
    ):
        ref = plan["upstream_references"][role]
        sources[role] = inputs.source(
            REPO, ref["run_id"], stage, experiment=experiment, reference=ref
        )
        ancestry.update(
            {
                k: v.reference
                for k, v in inputs.lineage(REPO, ref["run_id"], ref).items()
            }
        )
    if ancestry != plan["ancestry"]:
        raise PingstoreError("exp054 import ancestry differs from approval")
    _, theory_check = historical.mean_field(
        sources["mean_field"], sources["frequencies"]
    )
    if theory_check["frequency_deltas_hz"] != plan["historical_frequency_deltas_hz"]:
        raise PingstoreError("historical frequency comparison changed")
    cfg = recipe.configuration()
    with inputs.execution(
        REPO,
        "compute",
        sources=sources,
        configuration=cfg,
        operation="historical-import",
    ) as run:
        mapping = []
        recording_rows = {r["path"]: r for r in plan["retained_recordings"]}
        for row in plan["source_files"]:
            name = row["path"]
            source = safe_path(archive, name)
            if name in recording_rows:
                target = run.export / "probe" / Path(name).parent.name / "rasters.npz"
                target.parent.mkdir(parents=True, exist_ok=True)
                evidence.repack(source, target)
                if members(target) != recording_rows[name]["members"]:
                    raise PingstoreError("repacked NPY evidence differs")
                operation = "lossless ZIP; every NPY member byte preserved"
            else:
                target = (
                    run.export / "historical-numbers.json"
                    if name == f"{DERIVED}/numbers.json"
                    else run.provenance / "gold-2" / name
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if file_sha256(target) != row["sha256"]:
                    raise PingstoreError("copied historical evidence differs")
                operation = "byte-for-byte copy"
            mapping.append(
                {
                    "source": name,
                    "source_sha256": row["sha256"],
                    "source_bytes": row["size_bytes"],
                    "target": str(target.relative_to(run.directory)),
                    "target_sha256": file_sha256(target),
                    "target_bytes": target.stat().st_size,
                    "operation": operation,
                }
            )
        for row in plan["producer_code"]:
            target = safe_path(run.provenance / "producer-code", row["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_path(code_directory, row["audit_file"]), target)
            if file_sha256(target) != row["sha256"]:
                raise PingstoreError("copied producer code differs")
        shutil.copyfile(plan_path, run.provenance / "import-plan.json")
        shutil.copyfile(__file__, run.provenance / "import_gold2.py")
        write_json_atomic(
            run.export / "recordings.json",
            {"schema": "exp054.recordings/v1", "recipe": cfg, "jobs": recipe.jobs(cfg)},
        )
        staged = SimpleNamespace(
            export=run.export, record={"execution": {"configuration": cfg}}
        )
        evidence.compute_contract(staged)
        check = historical.compare_numbers(
            measurements.summary(measurements.recordings(staged, cfg), cfg),
            load_json(run.export / "historical-numbers.json"),
        )
        write_json_atomic(run.provenance / "file-mapping.json", {"files": mapping})
        write_json_atomic(
            run.provenance / "verification.json",
            {
                "source_files_verified": len(mapping),
                "live_metadata_matches": True,
                "live_metadata_sha256": {
                    n: file_sha256(live_directory / ("live-" + n)) for n in METADATA
                },
                "npy_member_bytes_exact": True,
                "recordings": len(recording_rows),
                "empirical_recheck": check,
                "mean_field_recheck": theory_check,
            },
        )
        run.record["historical_import"] = {
            "archive_uri": plan["archive"],
            "approved_plan_sha256": plan_sha256,
            "authorization": "author approved the pinned selective plan before execution",
            "producer": prod,
            "simulation_executed": False,
            "mapping": "provenance/file-mapping.json",
            "missing_evidence": plan["missing_evidence"],
            "historical_inconsistency": "derived manifest says host=local; scheduler identifies gpu-q-35",
            "source_preservation": "archive unchanged; all spike records retained; no bank copies",
        }
        (run.directory / "README.md").write_text(
            "# exp054: local historical import\n\n"
            "This is an import, not a new simulation. Scientific producer: Slurm job\n"
            "33913631 on gpu-q-35, campaign ggs-production-20260818-4ad223d3. Original\n"
            "commands, source hashes, configuration, logs and completion evidence are\n"
            "retained. The historical derived manifest incorrectly labels its host local;\n"
            "that conflicting evidence is preserved, not silently repaired.\n\n"
            "All 136 complete sparse recordings retain every NPY member byte through\n"
            "lossless ZIP compression, including pre-burn and output spikes. Mean-field\n"
            "evidence and frequency measurements reference validated exp033/exp041 runs\n"
            "and their ancestry; no model bank is copied. Missing historical ODE\n"
            "trajectories are not reconstructed. Analysis and presentation run separately.\n"
        )
        prepare(archive, plan, code_directory, live_directory)
        if file_sha256(plan_path) != plan_sha256:
            raise PingstoreError("approved import plan changed during execution")
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--producer-code-directory", type=Path, required=True)
    parser.add_argument("--live-metadata-directory", type=Path, required=True)
    args = parser.parse_args()
    print(
        import_subset(
            args.archive,
            args.plan,
            args.plan_sha256,
            args.producer_code_directory,
            args.live_metadata_directory,
        )
    )


if __name__ == "__main__":
    main()
