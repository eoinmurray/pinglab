"""Render saved exp083 measurements; never simulate, estimate or publish."""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp083 import evidence, inputs, plots
from experiments.helpers.fmt import format_duration
from pingstore.contracts import PingstoreError, write_json_atomic
from tools.snnlang.compiler import Bundle  # noqa: TID251


def present(identity, *, run_id=None):
    source = inputs.source(REPO, identity, "analyse")
    pin = source.record["inputs"]["compute"]
    compute = inputs.lineage(REPO, pin["run_id"], pin)[pin["run_id"]]
    result, graph, manifest = evidence.analysis_payload(source, compute)
    rasters = evidence.display_arrays(source, result["rasters"], "rasters")
    spectra = evidence.display_arrays(source, result["spectra"], "spectra")
    started = time.monotonic()
    with inputs.execution(
        REPO, "present", sources={"analysis": source}, run_id=run_id
    ) as run:
        Bundle(graph=graph, training=None, manifest=manifest, diagnostics=[]).visualise(
            run.export / "network.svg", view="circuit"
        )
        plots.plot_representative_rasters(
            rasters, result["conditions"], run.export / "representative_rasters.png"
        )
        plots.plot_response(result["conditions"], run.export / "response.png")
        plots.plot_spectra(spectra, run.export / "spectra.png")
        duration = time.monotonic() - started
        write_json_atomic(run.export / "protocol.json", result["config"])
        write_json_atomic(
            run.export / "numbers.json",
            {
                **{
                    key: result[key]
                    for key in (
                        "question",
                        "config",
                        "frequency_analysis",
                        "representative_rates_hz",
                        "graph",
                        "conditions",
                    )
                },
                "run_id": run.run_id,
                "git_sha": run.record["provenance"]["git_commit"],
                "duration_s": round(duration, 1),
                "duration": format_duration(duration),
            },
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        present(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"exp083 present: {exc}\n")


if __name__ == "__main__":
    main()
