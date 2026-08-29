"""Article navigation stays above the abstract and within the current article."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from xml.etree import ElementTree

import pytest
from demolab_cli import _paths

ROOT = Path(__file__).resolve().parents[1]
ARTICLES = sorted((ROOT / "writings").glob("exp[0-9][0-9][0-9].typ"))
SECTION_ORDINAL = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?|[A-Z](?:\.\d+)+\.?)\s|^Appendix [A-Z][.:]\s"
)


@pytest.fixture
def lab(tmp_path):
    shutil.copytree(ROOT / "writings", tmp_path / "writings")
    (tmp_path / ".demolab").mkdir()
    for name in ("lib.typ", "style.css"):
        shutil.copy2(_paths.TYP / name, tmp_path / ".demolab" / name)
    (tmp_path / ".demolab/VERSION").write_text("test")
    return tmp_path


def compile_document(lab, source, *, pdf=False):
    path = lab / "check.typ"
    path.write_text(source)
    output = lab / ("check.pdf" if pdf else "check.html")
    command = [_paths.find_typst(ROOT), "compile", "--root", str(lab)]
    if not pdf:
        command += ["--features", "html", "--format", "html"]
    result = subprocess.run([*command, str(path), str(output)], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "did not converge" not in result.stderr, result.stderr
    if pdf:
        assert output.read_bytes().startswith(b"%PDF")
        return output
    return re.sub(r"<style\b.*?</style>", "", output.read_text(), flags=re.S)


def navigation(html):
    return [ElementTree.fromstring(nav) for nav in re.findall(
        r'<nav aria-label="Table of Contents">.*?</nav>', html, re.S
    )]


@pytest.mark.parametrize("article", ARTICLES, ids=lambda path: path.stem)
def test_article_sections_are_unnumbered_and_results_precede_methods(article):
    # Source checks include reports whose data-dependent body is unavailable.
    source = article.read_text()
    headings = re.findall(r"^[ \t]*(=+) (.+)$", source, re.M)
    assert headings or article.stem == "exp092"
    for _, title in headings:
        assert not SECTION_ORDINAL.match(title), (article, title)
    top = [title for level, title in headings if level == "=="]
    results = [i for i, title in enumerate(top) if re.match(r"Results(?:[: ]|$)", title)]
    methods = [i for i, title in enumerate(top) if re.match(r"Methods(?: |$)", title)]
    if results and methods:
        assert max(results) < min(methods), article
    for setting in re.findall(r'#set heading\(([^\n]*)\)', source):
        numbering = re.search(r'numbering:\s*([^,)]+)', setting)
        assert numbering is None or numbering[1].strip() == "none"
    labels = set(re.findall(r'^[ \t]*=+ .*? <([^>]+)>$', source, re.M))
    for target in re.findall(r'#link\(<(sec-[^>]+)>\)', source):
        assert target in labels, (article, target)
    assert not re.search(r'§\d|\bAppendix [A-Z](?:\.\d+)?\b', source)


@pytest.mark.parametrize("article", ARTICLES, ids=lambda path: path.stem)
def test_results_heading_is_plain(article):
    # Include reports whose data-dependent body is unavailable.
    source = article.read_text()
    for title in re.findall(r'^[ \t]*== (Results\b[^\n]*)', source, re.M):
        title = re.sub(r' <[^>]+>$', '', title)
        assert title == "Results", (article, title)


@pytest.mark.parametrize("article", ARTICLES, ids=lambda path: path.stem)
def test_every_article_has_one_toc_and_valid_links(lab, article):
    source = article.read_text()
    assert source.count('#import "contents.typ": with-contents') == 1
    assert source.rstrip().endswith('#let body = with-contents(body)')
    assert not re.search(r"^\s*== (Contents|Table of Contents)\s*$", source, re.M)
    html = compile_document(lab,
        '#import "/.demolab/lib.typ": entry-page\n'
        f'#import "/writings/{article.name}": meta, body\n'
        '#entry-page(meta, body)\n')
    navs = navigation(html)
    assert len(navs) == 1
    rendered_headings = re.findall(r'<h[3-6]\b[^>]*>(.*?)</h[3-6]>', html, re.S)
    for title in rendered_headings:
        assert not SECTION_ORDINAL.match(re.sub(r'<[^>]+>', '', title))
    headings = re.findall(r'<h3\b[^>]*>(.*?)</h3>', html, re.S)
    assert headings[0].startswith("Table of Contents")
    links = navs[0].findall(".//a")
    assert links
    assert len(links) == len(headings) - 1
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    for link in links:
        href = link.attrib["href"]
        assert href.startswith("#") and href[1:] in ids
        assert href != "#table-of-contents"
    abstract = re.search(r'<h3\b[^>]*>(?:[0-9]+\. )?Abstract', html)
    if abstract:
        assert html.index('</nav>') < abstract.start()


@pytest.mark.parametrize("pdf", [False, True])
def test_rendered_headings_are_scoped_and_include_generated_sections(lab, pdf):
    html = compile_document(lab, '''
#import "/writings/contents.typ": with-contents
#import "/.demolab/lib.typ": entry-page, reference-list
#set heading(numbering: "1.1")
#entry-page((title: "First", date: "2026-08-28"), with-contents[
  == Abstract
  Summary.
  #context heading(level: 2)[Datasets]
  == Results
  Evidence.
  === Detail
  Detail.
  == Methods
  Procedure.
  #reference-list(((text: [A source.],),))
])
#entry-page((title: "Second", date: "2026-08-28"), with-contents[
  == Reference
  No abstract is required for this existing reference.
])
''', pdf=pdf)
    if pdf:
        evaluated = subprocess.run(
            [_paths.find_typst(ROOT), "eval", "--root", str(lab),
             "--in", str(lab / "check.typ"),
             "query(heading).filter(h => h.level >= 2).map(h => h.numbering)"],
            capture_output=True, text=True, timeout=30,
        )
        assert evaluated.returncode == 0 and evaluated.stdout.strip(), evaluated.stderr
        numbers = json.loads(evaluated.stdout)
        assert numbers and all(number is None for number in numbers)
        return
    first, second = navigation(html)
    assert [a.text for a in first.findall('.//a')] == [
        'Abstract', 'Datasets',
        'Results',
        'Methods', 'References']
    results = first.findall('.//a')[2]
    assert results.attrib['href'] == '#results'
    assert 'id="' + results.attrib['href'][1:] + '"' in html
    assert [a.text for a in second.findall('.//a')] == ['Reference']
    assert html.index('</nav>') < html.index('id="abstract"')
