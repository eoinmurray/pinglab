from __future__ import annotations

from pathlib import Path

from experiments.exp111 import measurements, plots, recipe


def _comparison(identifier: str, rows: list[dict]) -> dict:
    title, source = next(
        (title, source)
        for test_id, title, source in recipe.TESTS
        if test_id == identifier
    )
    return {
        "id": identifier,
        "title": title,
        "source_experiment": source,
        "x_label": "condition",
        "y_label": "rate (Hz)",
        "series": rows,
    }


def test_recipe_freezes_twenty_unique_comparisons_and_figures() -> None:
    identifiers = [identifier for identifier, _title, _source in recipe.TESTS]
    assert len(identifiers) == 20
    assert len(set(identifiers)) == 20
    assert recipe.FIGURES == tuple(f"{identifier}.svg" for identifier in identifiers)


def test_analysis_reports_distances_without_verdicts() -> None:
    first = _comparison(
        "projection-scaling",
        [
            {"label": "off", "x": 0, "snnsim": 10.0, "brian2": 10.4},
            {"label": "on", "x": 1, "snnsim": 20.0, "brian2": 20.5},
        ],
    )
    second = _comparison(
        "projection-scaling",
        [
            {"label": "off", "x": 0, "snnsim": 10.0, "brian2": 13.0},
            {"label": "on", "x": 1, "snnsim": 20.0, "brian2": 15.0},
        ],
    )
    result = measurements.evaluate([first, second])
    assert result["summary"] == {"total": 2}
    assert result["tests"][0]["maximum_absolute_difference"] == 0.5
    assert result["tests"][1]["maximum_absolute_difference"] == 5.0
    assert all(
        "passed" not in test and "checks" not in test and "condition" not in test
        for test in result["tests"]
    )
    assert round(result["tests"][0]["series"][0]["brian2_minus_snnsim"], 8) == 0.4


def test_plot_contains_two_named_panels_without_verdict(tmp_path: Path) -> None:
    comparison = _comparison(
        "projection-scaling",
        [
            {"label": "off", "x": 0, "snnsim": 10.0, "brian2": 10.2},
            {"label": "on", "x": 1, "snnsim": 20.0, "brian2": 20.4},
        ],
    )
    target = tmp_path / "comparison.svg"
    plots.comparison_figure(comparison, target)
    svg = target.read_text()
    assert all(text in svg for text in ("<!-- A -->", "<!-- B -->", "snnsim", "Brian2"))
    assert "PASS" not in svg and "FAIL" not in svg


def test_writing_has_one_card_and_figure_per_comparison() -> None:
    writing = Path(__file__).resolve().parents[2] / "writings" / "exp111.typ"
    source = writing.read_text()
    assert source.count("#result-card[") == 20
    assert source.count("#figure(") == 20
    assert 'status: "[▦ DATA | v34.0.1]"' in source
    assert "== Results" in source
    assert "== Methods" not in source
    assert "=== Compute" not in source
    assert "=== Analyse" not in source
    assert "=== Present" not in source
