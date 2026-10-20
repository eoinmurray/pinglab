"""Render saved true-class score trajectories without upstream execution.

The CLI requires explicit completed analysis evidence. Figures and browser
replays are written to a new presentation run; upstream execution is never
launched automatically.
"""

import argparse
import sys
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2]),
    str(Path(__file__).resolve().parents[2] / "tools"),
]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.exp123 import recipe
from experiments import theme
from pingstore.stages import source_run, stage_run


def save(fig, destination):
    with plt.rc_context({"savefig.bbox": None}):
        for ext in ("svg", "pdf"):
            fig.savefig(destination.with_suffix("." + ext), dpi=300)
    plt.close(fig)


def present(identity):
    source = source_run(
        recipe.REPO / ".pingstore", identity, stage="analyse", experiment=recipe.SLUG
    )
    if (
        source.record["execution"]["configuration"].get("schema")
        == "exp123.shared.measurement/v1"
    ):
        from experiments.exp123.custom.shared_plots import present_shared

        theme.set_paper_mode(True)
        theme.apply()
        plt.rcParams["svg.fonttype"] = "none"
        with stage_run(
            recipe.REPO,
            recipe.SLUG,
            "present",
            inputs={"analyse": source},
            configuration=dict(
                schema="exp123.shared.presentation/v1",
                paper_mode=True,
                figures="six figures from one uniform sweep; evidence traces coloured by mean I-volley interval frequency, higher darker, shared scale across nine panels and both coordinates; pooled sustained decisions in 5 ms / 1 volley by 5 Hz rectangular bins with shared log counts and frequency-bin medians; coverage-limited volleys; 80% population-accuracy first-crossing cost/latency frontier in house colours with matched mean I-frequency and nonlinear GABA-decay colour scales",
                show_uncertainty_estimates=False,
            ),
        ) as run:
            present_shared(source, run, save)
            result = run.run_id
        return result
    raise ValueError("expected shared-sweep analysis")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    print(present(parser.parse_args().source))
