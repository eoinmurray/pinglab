"""Compile the dataset table with real Typst and check its HTML contract."""

import json
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from xml.etree import ElementTree

import pytest
from demolab_cli import _paths

ROOT = Path(__file__).resolve().parents[1]


def run(identity, experiment="exp047", origin="local", stage="present"):
    return {
        "id": identity,
        "experiment": experiment,
        "basepath": f"/.pingstore/runs/{identity}/export",
        "created_at": "2026-08-28T13:31:23+00:00",
        "export_bytes": 2048,
        "origin": origin,
        "stage": stage,
    }


def render(tmp_path, runs, *, interactive=True, inputs=("exp047",), pdf=False, article_body=None, placed=False, dependencies=None):
    (tmp_path / "writings").mkdir()
    (tmp_path / ".demolab").mkdir()
    for name in ("run-inputs.typ", "run-view.typ"):
        shutil.copy2(ROOT / "writings" / name, tmp_path / "writings" / name)
    shutil.copy2(_paths.TYP / "lib.typ", tmp_path / ".demolab/lib.typ")
    (tmp_path / ".demolab/pinglab-inputs.json").write_text(json.dumps({
        "runs": [row for row in runs if row["stage"] == "present"],
        "display_runs": runs, "defaults": {}, "articles": {"report": list(inputs)},
        "experiment_dependencies": dependencies or {},
    }))
    keys = "(" + "".join(json.dumps(key) + "," for key in inputs) + ")"
    source = tmp_path / "view.typ"
    source.write_text(
        '#import "writings/run-view.typ": run-view, with-datasets\n'
        + (f'#run-view("report", {keys})\n' if article_body is None else
           f'#with-datasets("report", {keys}, placed: {str(placed).lower()})[\n{article_body}\n]\n')
    )
    output = tmp_path / ("view.pdf" if pdf else "view.html")
    command = [_paths.find_typst(ROOT), "compile", "--root", str(tmp_path),
               "--input", "demolab-bundle-root=/",
               "--input", f"demolab-dev={str(interactive).lower()}"]
    if not pdf:
        command += ["--features", "html", "--format", "html"]
    result = subprocess.run([*command, str(source), str(output)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    if pdf:
        assert output.read_bytes().startswith(b"%PDF")
        return None
    html = output.read_text()
    panel = re.search(r"<aside\b.*?</aside>", html, re.S)
    return ElementTree.fromstring(panel.group()) if panel else None


@pytest.mark.parametrize("origin", ["slurm", "modal", "runpod", "local", "mixed", "unknown"])
def test_run_table_has_name_date_duration_export_size_and_origin_columns(tmp_path, origin):
    panel = render(tmp_path, [run("exp047-r004-present", origin=origin)])
    row = panel.find("table/tbody/tr")
    assert list(panel.iter("details")) == []
    assert list(panel.iter("ul")) == []
    headers = panel.findall("table/thead/tr/th")
    assert [header.text for header in headers] == ["Run", "Date", "Duration", "Size", "Origin"]
    assert all(header.attrib["scope"] == "col" for header in headers)
    assert len(row.findall("td")) == 5
    assert " ".join(" ".join(row.itertext()).split()) == (
        f"exp047-r004-present 28 Aug 2026, 1:31 pm — 2 KB {origin}"
    )
    assert row.find("td/time").attrib["datetime"] == "2026-08-28T13:31:23+00:00"
    assert row.find("td/a").attrib["aria-current"] == "true"


@pytest.mark.parametrize("interactive", [True, False])
def test_timestamp_stays_on_one_line_in_scrollable_table(tmp_path, interactive):
    panel = render(tmp_path, [run("present")], interactive=interactive)
    assert panel.find("table/tbody/tr/td[@class='run-date']/time") is not None
    html = (tmp_path / "view.html").read_text()
    css = "\n".join(re.findall(r"<style\b[^>]*>(.*?)</style>", html, re.S))
    date_rule = re.search(r"\.run-view \.run-date\s*\{([^}]*)\}", css)
    assert date_rule is not None
    assert "white-space:nowrap;" in date_rule.group(1)
    panel_rule = re.search(r"\.run-view\s*\{([^}]*)\}", css)
    assert panel_rule is not None
    assert "overflow-x:auto;" in panel_rule.group(1)


def test_links_preserve_other_inputs_and_open_new_tabs(tmp_path):
    panel = render(tmp_path, [run("new"), run("old"), run("other", "exp025")],
                   inputs=("exp047", "exp025"))
    links = panel.findall("table/tbody/tr/td/a")
    assert [link.text for link in links] == ["new", "old", "other"]
    assert links[1].attrib["aria-current"] == "false"
    assert links[1].attrib["target"] == "_blank"
    assert links[1].attrib["rel"] == "noopener"
    assert parse_qs(urlsplit(links[1].attrib["href"]).query) == {
        "source.exp047": [".pingstore/runs/old/export"],
        "source.exp025": [".pingstore/runs/other/export"],
    }


def test_static_table_shows_only_selected_run_without_links(tmp_path):
    panel = render(tmp_path, [run("new"), run("old")], interactive=False)
    assert panel.findall(".//a") == []
    assert len(panel.findall("table/tbody/tr")) == 1
    selected = panel.find("table/tbody/tr/td/span")
    assert selected.text == "new"
    assert selected.attrib["class"] == "run-name run-stage-present"
    assert selected.attrib["aria-current"] == "true"


def test_empty_input_is_explicit(tmp_path):
    panel = render(tmp_path, [])
    cell = panel.find("table/tbody/tr/td")
    assert cell.attrib["colspan"] == "5"
    assert "".join(cell.itertext()).strip() == "exp047 — No presentation runs available."


def test_unknown_origin_fallback(tmp_path):
    panel = render(tmp_path, [run("new", origin="unrecognised")])
    assert panel.find("table/tbody/tr/td[@class='run-origin']").text == "unknown"


def test_no_inputs_shows_empty_datasets_table(tmp_path):
    panel = render(tmp_path, [], inputs=())
    assert panel.find("table/tbody/tr/td").text == "No datasets declared."
    assert panel.find("table/tbody/tr/td").attrib["colspan"] == "5"


@pytest.mark.parametrize("interactive", [True, False])
@pytest.mark.parametrize("with_runs", [True, False])
def test_dependencies_appear_once_for_page_above_runs(tmp_path, interactive, with_runs):
    rows = [run("compute", stage="compute"), run("present"), run("other", "exp025")] if with_runs else []
    panel = render(tmp_path, rows, interactive=interactive, inputs=("exp047", "exp025"),
                   dependencies={
                       "report": {"upstream": ["exp022", "exp025"], "downstream": ["exp049"]},
                       "exp047": {"upstream": ["exp099"], "downstream": []},
                   })
    assert [child.tag for child in panel] == ["dl", "table"]
    block = panel.find("dl")
    assert [label.text for label in block.findall("dt")] == ["Upstream", "Downstream"]
    for direction, experiments in (
        ("upstream", ["exp022", "exp025"]), ("downstream", ["exp049"]),
    ):
        cell = block.find(f"dd[@class='experiment-{direction}']")
        assert "".join(cell.itertext()) == ", ".join(experiments)
        assert [(link.text, link.attrib["href"]) for link in cell.findall("a")] == [
            (experiment, "/" + experiment) for experiment in experiments
        ]
    assert "report" not in "".join(block.itertext())
    assert "exp047" not in "".join(block.itertext())
    assert "exp099" not in "".join(panel.itertext())
    assert [header.text for header in panel.findall("table/thead/tr/th")] == [
        "Run", "Date", "Duration", "Size", "Origin",
    ]


def test_no_experiment_dependencies_shows_dashes(tmp_path):
    panel = render(tmp_path, [], inputs=())
    assert [cell.text for cell in panel.findall("dl/dd")] == ["—", "—"]


def test_lab_hook_uses_declared_graph_and_article_reuse():
    from pingstore.presentation_inputs import article_inputs, experiment_dependencies

    from writings.prepare import declared_dependencies

    data = experiment_dependencies(article_inputs(ROOT), declared_dependencies())
    assert data["exp037"]["upstream"] == ["exp022"]
    assert data["exp046"]["upstream"] == ["exp022", "exp041"]
    assert data["exp048"]["upstream"] == ["exp022"]
    assert data["exp023"]["upstream"] == []
    assert {"exp037", "exp048", "exp109"} <= set(data["exp022"]["downstream"])
    config = (ROOT / "demolab.yaml").read_text()
    assert "prepare: [uv, run, python, -m, writings.prepare]" in config


def test_pdf_omits_html_panel(tmp_path):
    render(tmp_path, [run("new")], pdf=True)


def test_dataset_heading_precedes_bordered_panel(tmp_path):
    panel = render(tmp_path, [run("present")])
    html = (tmp_path / "view.html").read_text()
    assert re.search(r"<h3\b[^>]*>Dataset</h3>", html)
    assert html.index("Dataset</h3>") < html.index("<aside")
    assert panel.attrib["aria-label"] == "Dataset"


@pytest.mark.parametrize("interactive", [True, False])
def test_compute_and_analyse_are_visible_but_not_clickable(tmp_path, interactive):
    rows = [run("compute", stage="compute"), run("analyse", stage="analyse"), run("present")]
    panel = render(tmp_path, rows, interactive=interactive)
    items = panel.findall("table/tbody/tr")
    assert len(items) == 3
    for item, identity in zip(items[:2], ("compute", "analyse")):
        assert item.find("td/a") is None
        label = item.find("td/span")
        assert label.text == identity
        assert label.attrib["class"] == f"run-name run-stage-{identity}"
        assert "style" not in label.attrib
    assert [a.text for a in panel.findall(".//a")] == (["present"] if interactive else [])
    present = items[2].find("td/a" if interactive else "td/span")
    assert present.attrib["class"] == "run-name run-stage-present"
    assert present.attrib["aria-current"] == "true"
    html = (tmp_path / "view.html").read_text()
    css = "\n".join(re.findall(r"<style\b[^>]*>(.*?)</style>", html, re.S))
    present_rule = re.search(r"\.run-view \.run-stage-present\s*\{([^}]*)\}", css)
    assert present_rule is not None
    assert "text-decoration:underline" in present_rule.group(1)


def test_compute_only_does_not_become_selected_input(tmp_path):
    panel = render(tmp_path, [run("compute", stage="compute")])
    assert panel.find("table/tbody/tr/td/span").text == "compute"
    assert panel.findall(".//a") == []


@pytest.mark.parametrize("size,label", [
    (0, "0 bytes"), (1, "1 byte"), (999, "999 bytes"), (1000, "1 KB"),
    (1234500, "1.2 MB"), (1500000000, "1.5 GB"), (2500000000000, "2.5 TB"),
])
def test_readable_sizes_and_exact_byte_tooltip(tmp_path, size, label):
    record = run("present") | {"export_bytes": size}
    panel = render(tmp_path, [record])
    size_span = panel.find("table/tbody/tr/td[@class='run-size']")
    assert size_span.text == label
    assert size_span.attrib["title"] == f"Export size: {size} bytes"


@pytest.mark.parametrize("timestamp,label", [
    ("2026-01-01T00:05:00+00:00", "1 Jan 2026, 12:05 am"),
    ("2026-12-31T12:00:59.123456+00:00", "31 Dec 2026, 12:00 pm"),
])
def test_readable_dates_keep_exact_timestamp(tmp_path, timestamp, label):
    panel = render(tmp_path, [run("present") | {"created_at": timestamp}])
    time = panel.find("table/tbody/tr/td/time")
    assert time.text == label
    assert time.attrib["datetime"] == timestamp


@pytest.mark.parametrize("seconds,label", [
    (None, "—"), (0, "<1s"), (0.25, "<1s"), (1, "1s"), (59, "59s"),
    (60, "1m"), (157.5, "2m 38s"), (3599.6, "1h"),
    (3661, "1h 1m 1s"), (90061, "1d 1h 1m 1s"),
])
def test_readable_duration_and_exact_tooltip(tmp_path, seconds, label):
    panel = render(tmp_path, [run("present") | {"duration_seconds": seconds}])
    cell = panel.find("table/tbody/tr/td[@class='run-duration']")
    assert cell.text == label
    if seconds is None:
        assert cell.attrib["title"] == "Execution timing not recorded"
    else:
        assert f"Recorded elapsed time: {seconds} seconds" in cell.attrib["title"]
        assert "excludes upstream runs" in cell.attrib["title"]


@pytest.mark.parametrize("stage", ["compute", "analyse", "present"])
@pytest.mark.parametrize("interactive", [True, False])
@pytest.mark.parametrize("operation", ["import", "historical-import"])
def test_import_duration_is_compact_with_explicit_tooltip_for_every_stage(tmp_path, stage, interactive, operation):
    panel = render(tmp_path, [run(stage, stage=stage) | {
        "duration_seconds": 3, "execution_operation": operation,
    }], interactive=interactive)
    cell = panel.find("table/tbody/tr/td[@class='run-duration']")
    assert cell.text == "3s"
    assert "excludes original training or simulation" in cell.attrib["title"]


@pytest.mark.parametrize("interactive", [True, False])
@pytest.mark.parametrize("operation", ["import", "historical-import"])
def test_scientific_duration_prefers_hpc_span_and_explains_job_total(tmp_path, interactive, operation):
    panel = render(tmp_path, [run("compute", stage="compute") | {
        "duration_seconds": 3, "execution_operation": operation,
        "scientific_timing": {
            "duration_seconds": 198256, "origin": "slurm", "jobs": 102,
            "job_seconds": 627332.174, "started_at": "2026-08-18T15:42:06Z",
            "completed_at": "2026-08-20T22:46:22Z",
        },
    }], interactive=interactive)
    cell = panel.find("table/tbody/tr/td[@class='run-duration']")
    assert cell.text == "2d 7h 4m 16s"
    assert "includes gaps between jobs" in cell.attrib["title"]
    assert "102 recorded completed attempts: 174.26 job-hours" in cell.attrib["title"]
    assert "excludes unrecorded attempts" in cell.attrib["title"]
    assert "Import operation: 3 seconds (excluded)" in cell.attrib["title"]
    assert panel.find("table/tbody/tr/td[@class='run-origin']").text == "local"


@pytest.mark.parametrize("abstract", ["Abstract", "1. Abstract", "*Abstract*"])
def test_dataset_follows_main_text_before_end_matter(tmp_path, abstract):
    render(tmp_path, [], article_body=(
        f"== Contents\n\nOpening navigation.\n\n== {abstract}\n\n"
        "First abstract paragraph.\n\nSecond abstract paragraph.\n\n"
        "== Results\n\nResults content.\n\n== Methods\n\nMethods content.\n\n"
        "== Appendix: Detail\n\nAppendix content.\n\n== References\n\nReferences content."
    ))
    html = (tmp_path / "view.html").read_text()
    assert html.count('aria-label="Dataset"') == 2  # aside and table
    positions = [html.index(text) for text in (
        "Contents", "First abstract paragraph", "Second abstract paragraph",
        "Results</h3>", "Methods</h3>", "Dataset</h3>",
        "Appendix: Detail</h3>", "References</h3>",
    )]
    assert positions == sorted(positions)


@pytest.mark.parametrize("body,before,after", [
    ("== Abstract\n\nSummary only.", "Summary only.", None),
    ("Introduction.\n\n== Reference\n\nDetails.", "Details.", None),
    ("A required run is unavailable.", "A required run is unavailable.", None),
])
def test_dataset_placement_without_end_matter(tmp_path, body, before, after):
    render(tmp_path, [], inputs=(), article_body=body)
    html = (tmp_path / "view.html").read_text()
    assert html.index(before) < html.index("Dataset</h3>")
    if after is not None:
        assert html.index("Dataset</h3>") < html.index(after)


def test_every_experiment_uses_shared_dataset_placement():
    articles = sorted((ROOT / "writings").glob("exp[0-9][0-9][0-9].typ"))
    assert articles
    for article in articles:
        source = article.read_text()
        assert re.search(
            r'^#import "run-view\.typ": [^\n]*\bwith-datasets\b', source, re.M
        ), article
        assert source.count(f'with-datasets("{article.stem}",') == 1, article
        abstract = re.search(r"^\s*== (?:\d+\. )?Abstract\s*$", source, re.M)
        if abstract:
            placement = source.index(f'#run-view("{article.stem}", inputs)')
            end_matter = re.search(
                r"^\s*(?:== Appendix(?:[.:]|\s)|== References\s*$|#reference-list\()",
                source, re.M,
            )
            if end_matter:
                assert placement < end_matter.start(), article
            main_headings = [match.start() for match in re.finditer(
                r"^\s*== (?!Appendix(?:[.:]|\s)|References\s*$)[^\n]+", source, re.M,
            )]
            assert main_headings and placement > max(main_headings), article
            assert source.count("#run-view(") == 1, article
            assert "placed: inputs-ready(data-file, inputs)" in source, article
        else:
            assert "#run-view(" not in source, article


def test_explicit_placement_preserves_styled_report_without_duplicate(tmp_path):
    render(tmp_path, [], placed=True, article_body=(
        '#show strong: it => html.elem("em", it.body)\n== Abstract\n\n*Summary.*\n\n'
        '== Results\n\nDetails.\n\n#run-view("report", ("exp047",))'
    ))
    html = (tmp_path / "view.html").read_text()
    assert html.count('aria-label="Dataset"') == 2
    assert html.index("Results</h3>") < html.index("Dataset</h3>")
    assert "<em>Summary.</em>" in html
