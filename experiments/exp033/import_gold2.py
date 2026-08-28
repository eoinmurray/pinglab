"""Execute only the approved selective Gold-2 import; no model execution."""

import argparse
import hashlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp033 import historical, inputs, recipe
from experiments.exp041.import_gold2 import safe_path
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)
from pingstore.stages import stage_run

BASE = "provenance/source-records/base"
DERIVED = "derived/artifacts/data/exp033"
CACHE = "state/experiments/exp054/super_compound_cache.npz"


def selected_paths():
    return {
        "run.json",
        "inventory.json",
        "lineage.json",
        CACHE,
        f"{DERIVED}/numbers.json",
        *(f"{DERIVED}/{n}" for n in historical.CARRY),
        f"{BASE}/run.json",
        f"{BASE}/collection-plan.json",
        f"{BASE}/submissions/collection-submission.json",
        f"{BASE}/logs/exp033/exp033.jsonl",
        *(f"{BASE}/collection-status/{s}.json" for s in ("exp033", "exp054")),
        *(
            f"{BASE}/logs/collection/ggs-{s}_{j}.{ext}"
            for s, j in (("exp033", "33913627"), ("exp054", "33913631"))
            for ext in ("out", "err")
        ),
    }


def verify_files(archive, plan):
    rows = plan["source_files"]
    if (
        len(rows) != len(selected_paths())
        or {r["path"] for r in rows} != selected_paths()
        or plan["source_file_count"] != len(rows)
        or sum(r["size_bytes"] for r in rows) != plan["source_bytes"]
        or plan["carry_forward_figures"] != list(historical.CARRY)
    ):
        raise PingstoreError("selection differs from the approved exp033 scope")
    inventory = load_json(safe_path(archive, "inventory.json"))
    indexed = {r["path"]: r for r in inventory["files"]}
    for row in rows:
        path = safe_path(archive, row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != row["size_bytes"]
            or file_sha256(path) != row["sha256"]
        ):
            raise PingstoreError("changed archive evidence: " + row["path"])
        if row["path"] not in ("run.json", "inventory.json", "lineage.json"):
            if any(indexed[row["path"]][k] != row[k] for k in ("sha256", "size_bytes")):
                raise PingstoreError("selection and archive inventory disagree")


def producer(archive, slug, job, expected):
    base = load_json(archive / BASE / "run.json")
    status = load_json(archive / BASE / "collection-status" / (slug + ".json"))
    log = (
        (archive / BASE / "logs/collection" / f"ggs-{slug}_{job}.out")
        .read_text()
        .splitlines()
    )
    plan = load_json(archive / BASE / "collection-plan.json")
    rows = [r for s in plan["stages"] for r in s["experiments"] if r["slug"] == slug]
    submission = load_json(archive / BASE / "submissions/collection-submission.json")

    def jobs(value):
        if isinstance(value, dict):
            if value.get("name") == "ggs-" + slug:
                yield value
            else:
                for v in value.values():
                    yield from jobs(v)
        elif isinstance(value, list):
            for v in value:
                yield from jobs(v)

    records = list(jobs(submission))
    if (
        base["run_id"] != expected["campaign"]
        or base["source"]["git_commit"] != expected["git_commit"]
        or status.get("state") != "complete"
        or status.get("experiment") != slug
        or len(rows) != 1
        or len(records) != 1
        or records[0].get("job_id") != job
        or not log[0].startswith(f"job={job} host=")
        or not log[0].endswith(f"action=run-experiment experiment={slug}")
    ):
        raise PingstoreError("historical producer identity is inconsistent")
    return {
        "origin": "slurm",
        "experiment": slug,
        "campaign": base["run_id"],
        "git_commit": base["source"]["git_commit"],
        "job_id": job,
        "host_record": log[0],
        "device_record": log[1],
        "status": status,
        "command": rows[0]["command"],
        "submission": records[0],
    }


def prepare(archive, plan, producer_code, live_directory):
    verify_files(archive, plan)
    if (
        plan.get("schema") != "exp033.selective-import-plan/v1"
        or load_json(archive / "run.json").get("archive", {}).get("uri")
        != plan["archive"]
        or file_sha256(producer_code) != plan["producer_code_sha256"]
    ):
        raise PingstoreError("archive or producer code differs from approved plan")
    for name in ("run.json", "inventory.json", "lineage.json"):
        if (live_directory / ("live-" + name)).read_bytes() != (
            archive / name
        ).read_bytes():
            raise PingstoreError("live R2 metadata differs from cached evidence")
    prod = producer(archive, "exp033", plan["producer"]["exp033_job"], plan["producer"])
    cache_prod = producer(
        archive, "exp054", plan["producer"]["exp054_cache_job"], plan["producer"]
    )
    old = load_json(archive / DERIVED / "numbers.json")
    provenance = old["collection_provenance"]
    if (
        provenance["source_git_commit"] != prod["git_commit"]
        or provenance["campaign_id"] != prod["campaign"]
        or provenance["dependencies"] != ["exp041"]
    ):
        raise PingstoreError("exp033 summary producer differs from its evidence")
    # Only this hash-verified, explicitly approved historical archive is unpickled.
    # Operational exports use ordinary JSON/ZIP and never pickle.
    with np.load(archive / CACHE, allow_pickle=True) as f:
        if f.files != ["payload"] or f["payload"].shape != (6,):
            raise PingstoreError("unexpected historical cache layout")
        _, sweep, hopf, crit, mf, meas = f["payload"]
    subset = {
        "sweep": sweep,
        "hopf": hopf,
        "criticality": crit,
        "frequency_vs_tau_gaba": mf,
        "spiking_exp041": {str(k): v for k, v in meas.items()},
    }
    raw = json.dumps(
        subset, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if hashlib.sha256(raw).hexdigest() != plan["borrowed_cache"][
        "selected_json_sha256"
    ] or not historical.exact_values(subset, json.loads(raw)):
        raise PingstoreError("selected historical values differ from approved plan")
    historical.validate_summary(old, subset)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("numerical-evidence.json", raw)
    return buffer.getvalue(), prod, cache_prod


def import_subset(archive, plan_path, plan_sha256, producer_code, live_directory):
    archive, plan_path, producer_code, live_directory = map(
        Path, (archive, plan_path, producer_code, live_directory)
    )
    if file_sha256(plan_path) != plan_sha256:
        raise PingstoreError("approved import plan changed")
    plan = load_json(plan_path)
    packed, prod, cache_prod = prepare(archive, plan, producer_code, live_directory)
    ancestors = {}
    for identity, ref in plan["upstream_references"].items():
        ancestors.update(inputs.lineage(REPO, identity, ref))
    frequency = next(
        s
        for s in ancestors.values()
        if s.record["experiment"] == "exp041" and s.record["stage"] == "analyse"
    )
    from experiments.exp033.measurements import spiking_medians

    current = spiking_medians(load_json(frequency.export / "results.json"))
    old = load_json(archive / DERIVED / "numbers.json")["results"]
    deltas = {
        str(t): current[t] - old["frequency_vs_tau_gaba"]["spiking_exp041"][str(t)]
        for t in current
    }
    if deltas != plan["frequency_comparison"]["deltas_hz"]:
        raise PingstoreError("upstream frequencies differ from approved comparison")
    measured = historical.verify_amplitudes(
        old["criticality"], old["hopf"]["I_ext_star"]
    )
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        inputs={"frequencies": frequency},
        configuration=recipe.configuration(),
        operation="historical-import",
    ) as run:
        mapping = []
        for row in plan["source_files"]:
            name = row["path"]
            if name == CACHE:
                target = run.export / "mean-field.zip"
                target.write_bytes(packed)
                operation = "all five mean-field cache entries; exact scalar JSON roundtrip; lossless ZIP"
            else:
                if name == f"{DERIVED}/numbers.json":
                    target = run.export / "historical-numbers.json"
                elif name.startswith(DERIVED + "/"):
                    target = run.export / "retained-figures" / Path(name).name
                else:
                    target = run.provenance / "gold-2" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(safe_path(archive, name), target)
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
                    "operation": operation,
                }
            )
        shutil.copyfile(producer_code, run.provenance / "producer-exp033.py")
        shutil.copyfile(__file__, run.provenance / "import_gold2.py")
        shutil.copyfile(plan_path, run.provenance / "import-plan.json")
        write_json_atomic(run.provenance / "file-mapping.json", {"files": mapping})
        run.record["historical_import"] = {
            "archive_uri": plan["archive"],
            "approved_plan_sha256": plan_sha256,
            "producer": prod,
            "cache_producer": cache_prod,
            "carry_forward_figures": list(historical.CARRY),
            "frequency_deltas_hz": plan["frequency_comparison"]["deltas_hz"],
            "simulation_executed": False,
            "mapping": "provenance/file-mapping.json",
            "missing_evidence": plan["missing_evidence"],
            "source_preservation": "Gold-2 unchanged; no bank copying or numerical subsampling",
        }
        write_json_atomic(
            run.provenance / "verification.json",
            {
                "source_files_verified": len(mapping),
                "live_metadata_matches": True,
                "numeric_scalar_roundtrip_exact": True,
                "local_hysteresis_recomputation": measured,
                "regression_tolerance": {"rtol": 1e-12, "atol": 1e-15},
                "sweep_points": 401,
                "scientific_producer_is_not_import_host": True,
            },
        )
        (run.directory / "README.md").write_text(
            "# exp033: local historical import\n\n"
            "This operation imports Gold-2 evidence, not a new simulation. Original\n"
            "exp033 producer: Slurm job 33913627, base campaign 4ad223d3. The matching\n"
            "401-point sweep is separately produced exp054 evidence, job 33913631.\n"
            "Both original identities, commands, source hashes and completion records\n"
            "are retained in run.json and provenance. The exp041 analysis and its bank\n"
            "ancestry are referenced, not copied.\n\n"
            "Export retains original numbers and four historical SVGs whose raw\n"
            "trajectories are missing; these are not reconstructed waveforms. The\n"
            "five mean-field cache entries are losslessly encoded as JSON/ZIP; the\n"
            "unrelated empirical grid remains in the unchanged archive. No numerical\n"
            "points were subsampled. Analysis and presentation execute separately.\n"
        )
        # Recheck all source evidence before exposing the completed directory.
        verify_files(archive, plan)
        if (
            file_sha256(plan_path) != plan_sha256
            or file_sha256(producer_code) != plan["producer_code_sha256"]
        ):
            raise PingstoreError("import sources changed during execution")
        for ancestor in ancestors.values():
            ancestor.check_unchanged()
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--producer-code", required=True)
    parser.add_argument("--live-metadata", required=True)
    args = parser.parse_args()
    import_subset(
        args.archive,
        args.plan,
        args.plan_sha256,
        args.producer_code,
        args.live_metadata,
    )


if __name__ == "__main__":
    main()
