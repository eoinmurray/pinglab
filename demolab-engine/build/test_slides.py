"""Tests for slides.py's bundle-vs-standalone detection — the rule that decides which
writings/*.typ `task slides` compiles (the compile itself needs the typst CLI and is
covered by the engine build test)."""
import slides


def test_bundle_entry_has_meta_and_body():
    assert slides.is_bundle_entry("#let meta = (title: [x])\n#let body = [prose]\n")


def test_deck_with_meta_but_no_body_is_standalone():
    # Touying decks declare #let meta (title/date for the homepage) but no #let body.
    assert not slides.is_bundle_entry('#import "@preview/touying"\n#let meta = (title: [d])\n')


def test_mentions_in_prose_do_not_count():
    # The declarations must start a line — `#let meta` quoted mid-prose is not the contract.
    assert not slides.is_bundle_entry("This doc explains `#let meta` and `#let body` usage.\n")
