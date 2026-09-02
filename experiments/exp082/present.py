"""Render explicit saved analysis and its pinned recordings; no upstream execution."""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp082 import evidence, inputs, plots, recipe
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def _panel_font(height: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(18, round(height * 0.052))
    try:
        return ImageFont.truetype("DejaVuSansMono-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def _relabel_summary_panel(image: Image.Image, label: str) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, round(width * 0.14), round(height * 0.10)), fill="white")
    draw.text(
        (round(width * 0.035), round(height * 0.014)),
        label,
        font=_panel_font(height),
        fill="black",
    )
    return image


def _fit_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), "white")
    panel.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    return panel


def build_continuous_stream_compound(
    hero_path: Path, summary_path: Path, output_stem: Path
) -> None:
    """Place the duration and rate panels vertically to the right of the stream."""
    with (
        Image.open(hero_path) as hero_source,
        Image.open(summary_path) as summary_source,
    ):
        hero = hero_source.convert("RGB")
        summary = summary_source.convert("RGB")
    left_end = round(summary.width * 0.575)
    right_start = round(summary.width * 0.56)
    summary_top = _relabel_summary_panel(
        summary.crop((0, 0, left_end, summary.height)), "E"
    )
    summary_bottom = _relabel_summary_panel(
        summary.crop((right_start, 0, summary.width, summary.height)), "F"
    )
    gap = round(hero.width * 0.02)
    column_width = round(hero.width * 0.52)
    cell_height = (hero.height - gap) // 2
    top = _fit_panel(summary_top, column_width, cell_height)
    bottom = _fit_panel(summary_bottom, column_width, cell_height)
    composite = Image.new(
        "RGB", (hero.width + gap + column_width, hero.height), "white"
    )
    composite.paste(hero, (0, 0))
    composite.paste(top, (hero.width + gap, 0))
    composite.paste(bottom, (hero.width + gap, cell_height + gap))
    composite.save(output_stem.with_suffix(".png"), dpi=(300, 300))
    composite.save(output_stem.with_suffix(".pdf"), "PDF", resolution=300)


def present(identity, *, run_id=None):
    source = inputs.source(REPO, identity, "analyse")
    if set(source.record["inputs"]) != {"compute", "showcase", "bank"}:
        raise PingstoreError("analysis must pin evaluation, showcase and bank")
    pin = source.record["inputs"]["compute"]
    compute = inputs.source(REPO, pin["run_id"], "compute", reference=pin)
    cfg, bank, _ = inputs.compute_evidence(REPO, compute)
    if source.record["inputs"]["bank"] != bank.reference:
        raise PingstoreError("analysis bank differs from compute ancestry")
    showcase_pin = source.record["inputs"]["showcase"]
    showcase = inputs.source(
        REPO, showcase_pin["run_id"], "compute", reference=showcase_pin
    )
    showcase_bank, showcase_record = evidence.showcase_evidence(REPO, showcase)
    if showcase_bank.reference != bank.reference:
        raise PingstoreError("showcase and evaluation use different banks")
    result = load_json(source.export / "numbers.json")
    expected = [
        {k: j[k] for k in ("seed", "duration_ms", "rate_hz")} for j in recipe.jobs(cfg)
    ]
    if (
        result.get("schema") != "exp082.analysis/v2"
        or [
            {k: r.get(k) for k in ("seed", "duration_ms", "rate_hz")}
            for r in result.get("grid_per_seed", [])
        ]
        != expected
    ):
        raise PingstoreError("analysis grid incomplete or inconsistent")
    arrays = evidence.arrays(source.export / "display.npz")
    streams = {}
    for name in ("matched", "variable"):
        raw, _ = evidence.stream(compute, name)
        streams[name] = {**raw, **result[name + "_stream"]}
    for name in recipe.SHOWCASE_TARGETS:
        raw, _ = evidence.stream(showcase, name, conditions=recipe.SHOWCASE_CONDITIONS)
        streams[name] = {**raw, **result[name + "_stream"]}
    if result.get("showcase_selection") != {
        key: showcase_record[key] for key in ("configuration", "candidates", "selected")
    }:
        raise PingstoreError("analysis showcase selection differs")
    index = result["single_trial_segment_index"]
    matched = streams["matched"]
    if (
        type(index) is not int
        or not 0 <= index < 5
        or matched["correct"][index] != 1
        or any(matched["correct"][:index])
    ):
        raise PingstoreError("analysis explanatory-trial selection differs")
    start, stop = matched["boundaries"][index : index + 2]
    streams["single_trial"] = {
        **result["single_trial"],
        "pixels": matched["pixels"][index : index + 1],
        **{k: matched[k][start:stop] for k in ("spikes_e", "spikes_i", "spikes_out")},
    }
    for name, stream in streams.items():
        for key in ("probabilities", "counts", "final_counts"):
            stream[key] = arrays[name + "_" + key]
        n = len(stream["spikes_out"])
        if (
            stream["probabilities"].shape != (n, 10)
            or stream["counts"].shape != (n, 10)
            or stream["final_counts"].shape != (10,)
        ):
            raise PingstoreError("analysis display dimensions differ")
        if not np.isfinite(stream["probabilities"]).all():
            raise PingstoreError("nonfinite display values")
    with inputs.execution(
        REPO,
        "present",
        sources={"analysis": source},
        run_id=run_id,
        configuration={"schema": "exp082.presentation/v3"},
    ) as run:
        out, rid = run.export, run.run_id
        plots.plot_variable_headline(streams["hero"], out / "hero_stream.png", rid)
        plots.plot_variable_headline(
            streams["alternative"], out / "alternative_stream.png", rid
        )
        plots.plot_single_trial(streams["single_trial"], out / "single_trial.png", rid)
        plots.plot_single_trial_transition(
            streams["single_trial"], out / "single_trial_transition.png", rid
        )
        plots.plot_stream(streams["matched"], out / "matched_stream.png", rid)
        plots.plot_variable_headline(
            streams["variable"], out / "variable_stream.png", rid
        )
        plots.plot_psychometric(
            result["plot_data"], out / "psychometric_200ms.svg", rid
        )
        plots.plot_duration_rate_summary(
            result["plot_data"], out / "duration_rate_summary.png", rid
        )
        build_continuous_stream_compound(
            out / "hero_stream.png",
            out / "duration_rate_summary.png",
            out / "continuous_stream_compound",
        )
        write_json_atomic(out / "numbers.json", {**result, "run_id": rid})
    return run.run_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--run-id")
    a = p.parse_args()
    try:
        present(a.source, run_id=a.run_id)
    except (PingstoreError, OSError, KeyError, ValueError, RuntimeError) as exc:
        p.exit(1, f"exp082 present: {exc}\n")


if __name__ == "__main__":
    main()
