"""Unit tests for build.py's resilience helpers — the pure parsing bits behind stubbing a broken
entry instead of failing the whole site (no compile, no typst)."""
import build


def test_entry_from_error_parses_the_failing_id():
    err = (
        "error: file not found (searched at /x/artifacts/data/exp044/dt_sweep.svg)\n"
        "  ┌─ writings/exp044.typ:50:10\n"
        "while importing `/writings/exp044.typ` at demolab-engine/build/main.typ:32:2"
    )
    assert build._entry_from_error(err, {"exp044", "exp000"}) == "exp044"
    # only entries we can still drop are candidates
    assert build._entry_from_error(err, {"exp000"}) is None
    # an error not attributable to an entry
    assert build._entry_from_error("error: something broke in main.typ", {"exp044"}) is None


def test_error_excerpt_grabs_the_error_block():
    out = (
        "downloading @preview/cmarker\n"
        "warning: bundle export is experimental\n"
        "error: file not found (searched at /x/y.svg)\n"
        "  context line\n"
    )
    excerpt = build._error_excerpt(out)
    assert excerpt.startswith("error: file not found"), excerpt
    assert "context line" in excerpt
    assert "downloading" not in excerpt  # trimmed to the error block
