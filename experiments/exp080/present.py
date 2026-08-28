"""Draw explicit retained analysis and illustrations; never simulate or aggregate."""

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp080 import evidence, inputs, plots, recipe
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def present(identity: str, *, run_id: str | None = None) -> str:
    analysis = inputs.source(REPO, identity, "analyse")
    cfg = inputs.configuration(analysis)
    pin = analysis.record["inputs"]["compute"]
    compute = inputs.source(REPO, pin["run_id"], "compute", reference=pin)
    result = load_json(analysis.export / "results.json")
    if (
        result.get("schema") != "exp080.analysis/v1"
        or result.get("recipe") != cfg
        or result.get("parameters") != recipe.reported_parameters(cfg)
        or result.get("decision") != load_json(analysis.export / "decision.json")
    ):
        raise PingstoreError("exp080 analysis recipe or decision disagrees")
    with inputs.execution(
        REPO,
        "present",
        sources={"analysis": analysis, "compute": compute},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        historical = compute.record["execution"].get("operation") == "historical-import"
        if historical:
            evidence.require(
                bool(compute.record.get("historical_import", {}).get("producer")),
                "historical import lacks original producer identity",
            )
        illustration = result["illustration"]
        if illustration != load_json(compute.export / "evidence.json")["illustration"]:
            raise PingstoreError("exp080 illustration disagrees with compute source")
        evidence.illustration(compute.export, illustration, cfg, historical=historical)
        plots.plot_training(result["training"], run.export)
        plots.plot_psychometric(result["decision"], run.export, cfg)
        if illustration["kind"] == "historical-image":
            shutil.copyfile(
                compute.export / illustration["path"], run.export / "feature_images.png"
            )
            run.record["retained_figures"] = {
                "feature_images.png": {
                    "source": compute.reference,
                    "path": illustration["path"],
                    "regenerated": False,
                },
            }
        else:
            with np.load(
                compute.export / illustration["path"], allow_pickle=False
            ) as samples:
                plots.plot_feature_images(
                    samples["image"],
                    samples["features_mV"],
                    samples["rates_hz"],
                    run.export,
                )
        write_json_atomic(run.export / "numbers.json", result)
        write_json_atomic(run.export / "decision.json", result["decision"])
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="completed exp080 v3 analyse run ID"
    )
    parser.add_argument(
        "--run-id", help="source-neutral identity reserved before dispatch"
    )
    args = parser.parse_args()
    present(args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
