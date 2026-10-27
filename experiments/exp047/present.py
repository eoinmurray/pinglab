"""Draw saved exp047 analysis; never simulate, aggregate or publish.

Invoke this stage explicitly from the repository root. Scientific outputs
and input provenance belong to its independently completed Pingstore run.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import theme
from experiments.exp047 import recipe
from experiments.exp047.custom import evidence, inputs
from experiments.exp047.custom.network_artifacts import copy_circuits
from pingstore.library import PingstoreError, load_json, write_json_atomic

DEFAULT_FORMATS = ("svg", "pdf")


def save_figure(fig, stem, formats=DEFAULT_FORMATS) -> list[Path]:
    """Write *fig* to <stem>.<ext> for each ext in *formats*.

    *stem* is a path without an extension (a trailing .png/.pdf/.svg is
    stripped, so existing call sites that pass "foo.png" keep working).
    Format is inferred from the extension; rcParams (dpi, bbox, fonttype)
    come from the active theme profile.
    """
    stem = Path(stem)
    if stem.suffix:
        stem = stem.with_suffix("")
    out: list[Path] = []
    for ext in formats:
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p)
        out.append(p)
    return out


def format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def analysis_source(repo, identity, reference=None):
    source = inputs.source(repo, identity, "analyse", reference=reference)
    refs = source.record["inputs"]
    if (
        set(refs) != {"compute"}
        or source.record["execution"].get("configuration") != recipe.MEASUREMENT
    ):
        raise PingstoreError(
            "exp047 analysis must pin compute and its measurement contract"
        )
    pin = refs["compute"]
    compute = inputs.source(repo, pin["run_id"], "compute", reference=pin)
    cfg = evidence.compute_contract(compute)
    result = evidence.analysis(load_json(source.export / "results.json"), cfg)
    return source, result


def present(identity, *, run_id=None):

    source, result = analysis_source(REPO, identity)
    started = time.monotonic()
    with inputs.execution(
        REPO,
        "present",
        sources={"analysis": source},
        run_id=run_id,
        configuration={"schema": "exp047.presentation/v1", "figure_run_stamps": False},
    ) as run:
        theme.set_paper_mode(True)
        plot_controls(
            result["summary"], result["config"], run.export / "pool_size_controls"
        )
        duration = time.monotonic() - started
        write_json_atomic(
            run.export / "numbers.json",
            {
                **result,
                "run_id": run.run_id,
                "duration_s": duration,
                "duration": format_duration(duration),
                "git_sha": run.record["provenance"]["git_commit"],
                "circuit_files": copy_circuits(run),
            },
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="explicit completed exp047 analysis ID"
    )
    parser.add_argument("--run-id", help="unused v4 identity reserved before dispatch")
    args = parser.parse_args()
    try:
        present(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"exp047 present: {exc}\n")


def plot_controls(summary: dict, cfg: dict, out_path: Path) -> None:
    theme.apply()
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.0), sharex=True)
    colors = [theme.GREY_MID, theme.DEEP_RED, theme.INK_BLACK]
    markers = ["s", "^", "o"]

    controls = [
        ("fixed_total", "Fixed summed coupling $G_{IE}$", cfg["reference_g_ie"]),
        ("fixed_synapse", "Fixed mean synapse $j_{IE}$", cfg["reference_j_ie"]),
    ]
    for col, (control, title, levels) in enumerate(controls):
        axes[0, col].set_title(title, fontsize=theme.SIZE_TITLE)
        for level, color, marker in zip(levels, colors, markers, strict=True):
            cells = [
                summary[control][f"{level:.12g}"][str(n)] for n in cfg["n_i_sweep"]
            ]
            if control == "fixed_total":
                label = f"$G_{{IE}}={level:g}$ μS"
            else:
                label = f"$j_{{IE}}={level * 1000:.2f}$ nS"
            for row, metric in enumerate(("r_e_hz", "r_i_hz")):
                means = [cell[f"{metric}_mean"] for cell in cells]
                sds = [cell[f"{metric}_sd"] for cell in cells]
                axes[row, col].errorbar(
                    cfg["n_i_sweep"],
                    means,
                    yerr=sds,
                    marker=marker,
                    color=color,
                    lw=1.5,
                    ms=5.5,
                    capsize=2,
                    label=label,
                )

    for row, ylabel in enumerate(("E rate  (Hz / cell)", "I rate  (Hz / cell)")):
        axes[row, 0].set_ylabel(ylabel, fontsize=theme.SIZE_LABEL)
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks(cfg["n_i_sweep"], labels=[str(n) for n in cfg["n_i_sweep"]])
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[1, :]:
        ax.set_xlabel("inhibitory pool size  $N_I$", fontsize=theme.SIZE_LABEL)
    axes[0, 0].legend(frameon=False, fontsize=theme.SIZE_LEGEND - 1)
    axes[0, 1].legend(frameon=False, fontsize=theme.SIZE_LEGEND - 1)

    theme.label_panels(axes.flat)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg", "pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
