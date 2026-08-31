"""Typst integration checks for explicit inputs and empty experiment reports."""
from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from demolab_cli import _paths

ROOT = Path(__file__).resolve().parents[1]
WRITINGS = ROOT / "writings"
REPORTS = [
    path.stem for path in sorted(WRITINGS.glob("exp*.typ"))
    if '#import "run-inputs.typ"' in path.read_text()
]
NOTICE = "A required run is unavailable, so there is no content to display yet."
SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2"><rect width="2" height="2"/></svg>'
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aP1sAAAAASUVORK5CYII="
)


@pytest.fixture
def lab(tmp_path):
    shutil.copytree(WRITINGS, tmp_path / "writings")
    (tmp_path / ".demolab").mkdir()
    shutil.copy2(_paths.TYP / "lib.typ", tmp_path / ".demolab/lib.typ")
    return tmp_path


def evaluate(lab, expression, *, preview=None, inventory=None, error=None):
    command = [_paths.find_typst(ROOT), "eval", "--root", str(lab)]
    for name, value in (
        ("demolab-preview-file", preview), ("demolab-data-inputs", inventory)
    ):
        if value is not None:
            (lab / f"{name}.json").write_text(json.dumps(value))
            command += ["--input", f"{name}=/{name}.json"]
    result = subprocess.run(
        [*command, expression], capture_output=True, text=True, timeout=30
    )
    # Some Typst eval versions signal errors on stderr with a zero exit status.
    if error is not None:
        assert not result.stdout.strip(), result.stdout
        assert error in result.stderr, result.stderr
        return None
    assert result.returncode == 0 and result.stdout.strip(), result.stderr
    return json.loads(result.stdout)


def report_expression(article):
    return '{ import "/writings/' + article + '.typ": body; repr(body) }'


@pytest.mark.parametrize("article", REPORTS)
def test_reports_without_inputs_show_shared_notice(lab, article):
    # Even a stale legacy file must not become a fallback publication input.
    legacy = lab / ".artifacts" / article
    legacy.mkdir(parents=True)
    (legacy / "numbers.json").write_text("not valid JSON")
    assert NOTICE in evaluate(lab, report_expression(article))
    assert NOTICE in evaluate(
        lab, report_expression(article), preview={article: {article: None}}
    )


def test_writings_do_not_import_legacy_resolver_or_link_artifacts():
    for path in [*WRITINGS.glob("*.typ"), WRITINGS / "README.md"]:
        text = path.read_text()
        assert ".artifacts" not in text, path
        assert not re.search(r'^#import "/.demolab/lib.typ":.*\bdata-file\b', text, re.M), path


def test_input_keys_match_url_allowlist_and_legacy_attachments():
    config = subprocess.run(
        [_paths.find_typst(ROOT), "eval", "--root", str(ROOT), 'yaml("/demolab.yaml")'],
        capture_output=True, text=True, check=True, timeout=30,
    )
    settings = json.loads(config.stdout)
    attachments = settings["legacy_preview"]["articles"]
    for article in REPORTS:
        text = (WRITINGS / f"{article}.typ").read_text()
        declaration = re.search(r"^#let inputs = \((.*?)\)\n", text, re.M | re.S)
        assert declaration is not None, article
        assert set(re.findall(r'"([^"]+)"', declaration[1])) == set(attachments[article]), article
        for key in attachments[article]:
            assert "source." + key in settings["url_inputs"], (article, key)


RESOLVER = '{ import "/writings/run-inputs.typ": data-file, inputs-ready; '


def test_selected_numbers_are_read_and_article_scoped(lab):
    (lab / "selected").mkdir()
    (lab / "selected/numbers.json").write_text('{"value": 7}')
    expression = RESOLVER + 'json(data-file("input/numbers.json", article: "report")).value }'
    assert evaluate(lab, expression, preview={"report": {"input": "/selected"}}) == 7
    assert evaluate(
        lab, RESOLVER + 'data-file("input/numbers.json", article: "other") }',
        preview={"report": {"input": "/selected"}},
    ) is None


def test_exp081_selected_report_renders_all_sections(lab):
    from experiments.exp081 import recipe

    selected = lab / "selected"
    selected.mkdir()
    (selected / "numbers.json").write_text(json.dumps({
        "schema": "exp081.analysis/v1", "parameters": recipe.configuration(),
        "comparison": {
            "mean": {
                "pearson_r": 0.9,
                "median_predicted_empirical_ratio": 1.5,
            },
            "standard_deviation": {"pearson_r": 0.3},
        },
    }))
    for name in ("empirical_moments", "response_distributions", "frequency_response", "analytical_empirical"):
        (selected / f"{name}.svg").write_text(SVG)
    rendered = evaluate(lab, report_expression("exp081"), preview={"exp081": {"exp081": "/selected"}})
    assert NOTICE not in rendered
    headings = ["Abstract", "Results", "Methods"]
    positions = [rendered.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "stationary approximation" in rendered
    assert "spike-count and spike-time mixture" in rendered
    for removed in (
        "Discussion",
        "Inputs and outputs",
        "Design Scope",
        "Prior art",
        "Conclusion",
        "Limitations",
    ):
        assert removed not in rendered
    assert "derivation of the analytical filter" in rendered
    assert "uv run" not in rendered
    (selected / "numbers.json").write_text("broken")
    evaluate(lab, report_expression("exp081"), preview={"exp081": {"exp081": "/selected"}}, error="error:")


@pytest.mark.parametrize("contents,error", [(None, "file not found"), ("broken", "error:")])
def test_selected_missing_or_corrupt_numbers_remain_errors(lab, contents, error):
    (lab / "selected").mkdir()
    if contents is not None:
        (lab / "selected/numbers.json").write_text(contents)
    evaluate(
        lab, RESOLVER + 'json(data-file("input/numbers.json", article: "report")) }',
        preview={"report": {"input": "/selected"}}, error=error,
    )


def test_fixed_inventory_and_preview_precedence(lab):
    inventory = {
        "sources": {"report": {"input": "/fixed"}},
        "files": ["/fixed/figure.svg"],
    }
    expression = RESOLVER + 'data-file("input/figure.svg", article: "report") }'
    assert evaluate(lab, expression, inventory=inventory) == "/fixed/figure.svg"
    assert evaluate(
        lab, expression, inventory=inventory, preview={"report": {"input": None}}
    ) is None
    evaluate(
        lab, RESOLVER + 'data-file("other/figure.svg", article: "report") }',
        inventory=inventory, error="has no pin",
    )
    evaluate(
        lab, RESOLVER + 'data-file("input/absent.svg", article: "report") }',
        inventory=inventory, error="missing pinned data file",
    )


def test_image_only_input_needs_no_numbers_file(lab):
    (lab / "selected").mkdir()
    (lab / "selected/network.svg").write_text(SVG)
    (lab / "selected/richer-input-ai-to-intermittent-ping.mp4").write_bytes(b"fixture-video")
    selection = {"exp099": {"exp099": "/selected"}}
    rendered = evaluate(lab, report_expression("exp099"), preview=selection)
    assert NOTICE not in rendered
    headings = ["Abstract", "Results", "Methods", "References"]
    assert [rendered.index(name) for name in headings] == sorted(
        rendered.index(name) for name in headings
    )
    assert "Working media" not in rendered
    assert "controlled comparison remain planned" in rendered
    # Compile contexts too: the video must share the diagram's figure counter,
    # and the article must no longer depend on the standalone poster.
    (lab / "report.typ").write_text('#import "writings/exp099.typ": body\n#body\n')
    result = subprocess.run(
        [_paths.find_typst(ROOT), "compile", "--features", "html", "--format", "html",
         "--root", str(lab), "--input", "demolab-preview-file=/demolab-preview-file.json",
         str(lab / "report.typ"), str(lab / "report.html")],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    html = (lab / "report.html").read_text()
    captions = re.findall(r"<figcaption\b[^>]*>(.*?)</figcaption>", html, re.S)
    assert len(captions) == 2
    assert re.search(r"Figure\s+1", re.sub(r"<[^>]+>", "", captions[0]))
    assert re.search(r"Figure\s+2", re.sub(r"<[^>]+>", "", captions[1]))
    assert html.count("<video ") == 1
    assert "richer-input-ai-to-intermittent-ping.png" not in html
    (lab / "selected/network.svg").unlink()
    evaluate(lab, report_expression("exp099"), preview=selection, error="file not found")


@pytest.mark.parametrize("path", ["../input/a.json", "input/../a.json", "/input/a.json", "input\\a.json"])
def test_invalid_logical_paths_are_rejected(lab, path):
    evaluate(
        lab, RESOLVER + f'data-file({json.dumps(path)}, article: "report") }}',
        error="safe data key",
    )
