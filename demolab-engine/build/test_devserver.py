"""Unit tests for the dev server's pure logic (no HTTP, no build).

The end-to-end behaviour — hot-add of a new entry, the browser error overlay on a failed
compile — is exercised by hand against `task dev`; these cover the two string transforms the
server leans on, which are easy to break silently, plus the loopback bind (sockets only, no
HTTP request, no build) that an IPv4-only server gets wrong on Windows.
"""
import socket
import threading

import devserver


def test_sse_bytes_single_line():
    assert devserver.sse_bytes("reload") == b"data: reload\n\n"


def test_sse_bytes_multiline_frames_each_line():
    # A Typst error is multi-line; EventSource rejoins "data:" lines with "\n", so each source
    # line must get its own "data: " prefix or the browser sees a mangled message.
    out = devserver.sse_bytes("error\nunclosed delimiter\n  at foo.typ:1").decode()
    assert out == "data: error\ndata: unclosed delimiter\ndata:   at foo.typ:1\n\n"


def test_inject_reload_before_body_close():
    out = devserver.inject_reload("<html><body>hi</body></html>")
    assert "EventSource('/__dev')" in out
    # injected inside the document, immediately before </body>
    assert out.index("<script>") < out.index("</body>")
    assert out.count("</body>") == 1


def test_inject_reload_appends_when_no_body_tag():
    out = devserver.inject_reload("<p>fragment</p>")
    assert out.startswith("<p>fragment</p>")
    assert out.rstrip().endswith("</script>")


def test_benign_disconnects_are_swallowed():
    # Browser disconnects (reset / broken pipe / abort) are harmless and must not be logged as errors.
    for exc in (ConnectionResetError(), BrokenPipeError(), ConnectionAbortedError(), TimeoutError()):
        assert devserver._is_benign_disconnect(exc), exc
    # A real error is not swallowed.
    for exc in (ValueError(), RuntimeError(), KeyError()):
        assert not devserver._is_benign_disconnect(exc), exc


def test_within_blocks_traversal(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "ok.html").write_text("x")
    assert devserver._within(site / "ok.html", site)
    assert devserver._within(site / "sub" / "page.html", site)  # not-yet-existing, still contained
    # `..` that climbs out of the site must be rejected
    assert not devserver._within(site / ".." / "secret.html", site)
    assert not devserver._within(site / ".." / ".." / "etc" / "hosts.html", site)


def test_make_server_accepts_both_loopbacks():
    # The banner says http://localhost, but Windows resolves `localhost` to the IPv6 ::1 first
    # — an IPv4-only bind makes that URL dead there while 127.0.0.1 works. make_server must
    # therefore accept connections on BOTH loopbacks (dual-stack), not just 127.0.0.1.
    server = devserver.make_server(0)  # port 0 → OS assigns a free one
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass
        if server.address_family == socket.AF_INET6:  # IPv4-only fallback hosts skip this leg
            with socket.create_connection(("::1", port), timeout=2):
                pass
    finally:
        server.shutdown()
        server.server_close()


def test_deck_affecting_triggers_on_slide_and_data():
    # A deck PDF depends only on its own source and the data assets it embeds.
    assert devserver.deck_affecting({"/repo/writings/ar004.slide.typ"})
    assert devserver.deck_affecting({"/repo/artifacts/data/exp000/lif.svg"})
    # A prose / CSS / lib edit can't change a deck, so decks are skipped.
    assert not devserver.deck_affecting({"/repo/writings/exp000.typ"})
    assert not devserver.deck_affecting({"/repo/demolab-engine/build/lib.typ"})
    assert not devserver.deck_affecting(set())
