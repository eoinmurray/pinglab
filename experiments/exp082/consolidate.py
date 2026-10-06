"""Explicitly consolidate validated evaluation and showcase evidence without inference."""

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp082 import evidence, inputs
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)


def consolidate(evaluation_identity, showcase_identity, *, run_id=None):
    evaluation = inputs.source(REPO, evaluation_identity, "compute")
    showcase = inputs.source(REPO, showcase_identity, "compute")
    cfg, bank, _ = inputs.compute_evidence(REPO, evaluation)
    showcase_bank, selection = evidence.showcase_evidence(REPO, showcase)
    if showcase_bank.reference != bank.reference:
        raise PingstoreError("evaluation and showcase use different banks")
    saved = load_json(evaluation.export / "evidence.json")
    if saved.get("schema") != "exp082.compute/v2":
        raise PingstoreError("consolidation requires an uncombined current evaluation")
    with inputs.execution(
        REPO,
        "compute",
        sources={"bank": bank},
        run_id=run_id,
        configuration=cfg,
        operation="import",
    ) as run:
        mappings = {}
        for role, source in (("evaluation", evaluation), ("showcase", showcase)):
            files = {}
            for path in sorted(source.export.iterdir()):
                if path.name == "evidence.json":
                    continue
                destination = run.export / path.name
                if destination.exists():
                    raise PingstoreError("overlapping scientific export: " + path.name)
                if path.is_dir():
                    shutil.copytree(path, destination)
                else:
                    shutil.copyfile(path, destination)
                children = sorted(path.iterdir()) if path.is_dir() else [path]
                for original in children:
                    relative = original.relative_to(source.export)
                    checksum = file_sha256(original)
                    if file_sha256(run.export / relative) != checksum:
                        raise PingstoreError(
                            "copied scientific bytes differ: " + str(relative)
                        )
                    files[str(relative)] = checksum
            mappings[role] = {
                "source_record": source.record,
                "source_readme": (source.directory / "README.md").read_text(),
                "evidence": load_json(source.export / "evidence.json"),
                "copied_file_sha256": files,
            }
        run.record["consolidation"] = {
            "schema": "exp082.consolidation/v1",
            "meaning": (
                "Self-contained byte-preserving consolidation; source identities are "
                "historical provenance, not operational inputs. No inference was run. "
                "Original evidence summaries and file hashes reconstruct both exports."
            ),
            "sources": mappings,
        }
        run.record["historical_import"] = {
            "schema": "exp082.consolidation/v1",
            "method": "consolidate-completed-evidence",
            "producer_origin": " and ".join(
                sorted({evaluation.record["origin"], showcase.record["origin"]})
            ),
            "note": "Separate original executions; no single combined scientific wall time.",
            "components": [
                {
                    "role": role,
                    "origin": source.record["origin"],
                    "execution": source.record["execution"],
                    "provenance": source.record["provenance"],
                }
                for role, source in (("evaluation", evaluation), ("showcase", showcase))
            ],
        }
        run.record["showcase_configuration"] = selection["configuration"]
        write_json_atomic(
            run.export / "evidence.json",
            {**saved, "schema": "exp082.compute/v3", "showcase": selection},
        )
        evidence.validate_compute(run.export, cfg)
        evidence.validate_showcase(run.export)
        evaluation.check_unchanged()
        showcase.check_unchanged()
        with (run.directory / "README.md").open("a") as history:
            history.write(
                f"- {run.record['created_at']}: consolidated evaluation `{evaluation_identity}` "
                f"and showcase `{showcase_identity}` with unchanged scientific file bytes. "
                "Original records, histories, evidence summaries and file hashes are retained "
                "in run.json; both original exports are recoverable from this run. "
                "The two scientific recipes remain distinct. No training or inference ran.\n"
            )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--showcase", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        consolidate(args.evaluation, args.showcase, run_id=args.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"exp082 consolidate: {exc}\n")


if __name__ == "__main__":
    main()
