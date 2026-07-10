"""demolab dev server — rebuild on change, serve with live-reload + in-browser build errors.

Replaces `typst watch`'s built-in server. Typst still does all the compiling (via build.py);
this process owns HTTP, live-reload, and the error surface, which fixes two things `typst watch`
couldn't:

  1. Added / removed writings are picked up. build.py re-globs the filesystem on every rebuild
     (Typst can't list a directory), so a brand-new writings/expNNN.typ or a *.slide.typ deck
     appears without restarting the server.
  2. A failed compile shows up IN the browser. `typst watch` kept the last good site and printed
     the error only to the terminal; here a failed build paints a full-screen overlay in the page
     (and it clears on the next good build), so a broken edit can't masquerade as a working site.

Trade-off vs `typst watch`: a full build.py per save (~1s for a small lab) instead of Typst's
incremental recompile. Simpler, and it makes the dev path honest. No Node, no new dependencies.

Run: `uv run python demolab-engine/build/devserver.py [port]` (defaults to the first free port
from 3000). Ctrl-C to stop.
"""
import http.server
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import unquote

import build

ROOT = build.ROOT
ENGINE = build.ENGINE
BUILD_PY = ENGINE / "build.py"
SITE = build.SITE

# Source trees whose changes trigger a rebuild. SOURCES only — never artifacts/site (build.py
# writes it, which would loop). Add/remove within these dirs is detected too (the file set is
# part of the signature), which is what lets new entries appear.
if build.DEMO:
    WATCH_DIRS = [
        (build.CONTENT / "writings", "*.typ"),
        (ENGINE, "*.typ"),
        (ENGINE, "*.css"),
        (ENGINE, "*.js"),
        (ENGINE, "*.py"),
        (build.CONTENT / "artifacts" / "data", "**/*"),
    ]
    WATCH_FILES = [build.CONTENT / "demolab.yaml"]
else:
    WATCH_DIRS = [
        (ROOT / "writings", "*.typ"),                 # entries + decks (content + add/remove)
        (ENGINE, "*.typ"),                            # lib.typ, main.typ
        (ENGINE, "*.css"),
        (ENGINE, "*.js"),
        (ENGINE, "*.py"),                             # build.py / devserver.py themselves
        (ROOT / "artifacts" / "data", "**/*"),        # runner outputs (figures, videos, numbers)
    ]
    WATCH_FILES = [ROOT / "demolab.yaml"]             # optional brand config (may not exist)
POLL_SECONDS = 0.4
DEBOUNCE_SECONDS = 0.15
BUILD_TIMEOUT = 120  # a compile still running after this is stuck, not slow — surface it, don't hang

# --- live-reload + error-overlay client (injected into every served HTML page) ---
RELOAD_JS = r"""
(function () {
  var overlay;
  function show(msg) {
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.id = '__demolab_dev_error';
      overlay.style.cssText =
        'position:fixed;inset:0;z-index:2147483647;margin:0;padding:2.2rem 2.4rem;overflow:auto;' +
        'background:#1a1a1af2;color:#f4f4f2;font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;' +
        'white-space:pre-wrap;-webkit-font-smoothing:antialiased;backdrop-filter:blur(1px)';
      document.documentElement.appendChild(overlay);
    }
    overlay.textContent = '⚠  demolab build failed\n\n' + msg +
      '\n\nFix the source; this clears on the next good build.';
  }
  function clear() { if (overlay) { overlay.remove(); overlay = null; } }
  function connect() {
    var es = new EventSource('/__dev');
    es.onmessage = function (e) {
      if (e.data === 'reload') { location.reload(); }
      else if (e.data === 'ok') { clear(); }
      else if (e.data.slice(0, 5) === 'error') { show(e.data.slice(6)); }
    };
    es.onerror = function () { es.close(); setTimeout(connect, 1000); };
  }
  connect();
})();
"""

# --- broadcast state ---
_lock = threading.Lock()
_clients = []                 # list[queue.Queue] — one per connected browser tab
_last_error = [""]            # latest build error text ("" when the last build was clean)


def broadcast(msg: str) -> None:
    with _lock:
        targets = list(_clients)
    for q in targets:
        try:
            q.put_nowait(msg)
        except queue.Full:
            pass  # a stalled client — it catches up on the next reload, or the ping drops it


def sse_bytes(payload: str) -> bytes:
    # SSE frames each line with "data: "; EventSource rejoins multi-line data with "\n".
    body = "".join("data: " + line + "\n" for line in payload.split("\n"))
    return (body + "\n").encode("utf-8")


def inject_reload(html: str) -> str:
    """Splice the live-reload client into a page, before </body> when present (else appended),
    so even a stale page can receive the error overlay."""
    tag = "<script>" + RELOAD_JS + "</script>"
    return html.replace("</body>", tag + "</body>", 1) if "</body>" in html else html + tag


def snapshot() -> dict:
    """A signature of every watched source file: path -> mtime_ns. A changed value means an edit;
    a changed key set means an add/remove."""
    sig = {}
    for base, pattern in WATCH_DIRS:
        if not base.exists():
            continue
        for p in base.glob(pattern):
            if p.is_file():
                try:
                    sig[str(p)] = p.stat().st_mtime_ns
                except OSError:
                    pass  # vanished mid-scan
    for p in WATCH_FILES:
        try:
            sig[str(p)] = p.stat().st_mtime_ns
        except OSError:
            pass  # optional / absent
    return sig


def deck_affecting(changed: set) -> bool:
    """A deck PDF depends only on its own `.slide.typ` and the `artifacts/data` assets it embeds
    (decks import touying, not lib.typ). So a prose/CSS/lib edit can't change any deck — only a
    slide-source or data-asset change can, and just then do we pay to recompile decks."""
    return any(p.endswith(".slide.typ") or "/artifacts/data/" in p for p in changed)


def build(skip_decks: bool = False) -> tuple[bool, str]:
    """Run build.py with the current interpreter (already inside the uv env, so no uv spawn).
    Returns (ok, error_text). On failure the combined output carries Typst's error."""
    cmd = [sys.executable, str(BUILD_PY)]
    if skip_decks:
        cmd.append("--skip-decks")
    try:
        proc = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=BUILD_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False, f"build timed out after {BUILD_TIMEOUT}s — the compile looks stuck."
    except Exception as e:  # never let a spawn failure kill the watch loop
        return False, f"couldn't run build.py: {type(e).__name__}: {e}"
    if proc.returncode == 0:
        tail = proc.stdout.strip().splitlines()
        return True, (tail[-1] if tail else "built")
    return False, (proc.stdout + proc.stderr).strip() or f"build.py exited {proc.returncode}"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(SITE), **k)

    def log_message(self, *a):
        pass  # quiet — build status is printed by the watch loop instead

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/__dev":
            return self._sse()
        path = unquote(self.path.split("?", 1)[0])
        fs = SITE / path.lstrip("/")
        if path.endswith("/") or fs.is_dir():
            fs = fs / "index.html"
        # Serve in-tree .html through the reload-injecting path; everything else (assets, or a
        # crafted `..` path) falls to SimpleHTTPRequestHandler, whose translate_path already confines
        # to the served dir. _within re-checks so a `..%2f….html` can't escape SITE via this path.
        if fs.suffix == ".html" and _within(fs, SITE):
            return self._serve_html(fs)
        return super().do_GET()

    def _serve_html(self, fs: Path):
        # Inject the reload client so even a stale page can receive the error overlay. If the file
        # doesn't exist yet (first build failed, nothing ever built), synthesize a shell so the
        # overlay still has somewhere to render.
        if fs.is_file():
            html = fs.read_text()
        elif fs.name == "index.html":
            html = "<!doctype html><meta charset=utf-8><title>demolab dev</title><body></body>"
        else:
            return super().do_GET()
        data = inject_reload(html).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except _BENIGN_DISCONNECT:
            pass

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = queue.Queue(maxsize=64)  # bounded so a stalled tab can't grow it without limit
        with _lock:
            _clients.append(q)
            pending_error = _last_error[0]
        try:
            # A tab that connects while the build is broken should see the overlay immediately.
            if pending_error:
                self.wfile.write(sse_bytes("error\n" + pending_error))
                self.wfile.flush()
            while True:
                try:
                    self.wfile.write(sse_bytes(q.get(timeout=15)))
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")  # keep the connection warm through proxies
                self.wfile.flush()
        except (*_BENIGN_DISCONNECT, ValueError):  # ValueError: write to a closed connection
            pass
        finally:
            with _lock:
                if q in _clients:
                    _clients.remove(q)


# Browsers routinely reset dev connections — closing an SSE stream, reloading, navigating away —
# which raises these from deep inside http.server, below the handler's own try/excepts. They're
# harmless, so the server swallows them instead of dumping an alarming traceback the user will read
# as a crash.
_BENIGN_DISCONNECT = (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, TimeoutError)


def _is_benign_disconnect(exc) -> bool:
    return isinstance(exc, _BENIGN_DISCONNECT)


def _within(p: Path, base: Path) -> bool:
    """True if `p` resolves inside `base` — guards the .html serving path against a `..` traversal
    that would otherwise read files outside the site directory."""
    try:
        p.resolve().relative_to(base.resolve())
        return True
    except (ValueError, OSError):
        return False


class DevServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True  # rebind cleanly after a restart (no TIME_WAIT stall)

    def server_bind(self):
        if self.address_family == socket.AF_INET6:
            # Accept IPv4 on the same socket (dual-stack). Windows resolves `localhost` to the
            # IPv6 ::1 first, so an IPv4-only bind makes the printed URL unreachable there.
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except OSError:
                pass  # platform forces v6-only; IPv6 localhost still works
        return super().server_bind()

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if _is_benign_disconnect(exc):
            return  # the client just went away
        # A genuinely unexpected per-request error: one quiet line, never a traceback, and the
        # server keeps serving.
        print(f"  dev server: {type(exc).__name__}: {exc}", flush=True)


def pick_port(argv) -> int:
    if len(argv) > 1 and argv[1].strip():
        return int(argv[1].strip())
    port = 3000
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port  # nothing listening
        port += 1


def watch_loop():
    """Poll the source signature; on a settled change, rebuild and tell the browser."""
    ok, msg = build()  # first build always compiles decks so their PDFs exist for later skips
    _last_error[0] = "" if ok else msg
    print("  first build: " + ("ok" if ok else "FAILED\n" + msg), flush=True)
    last = snapshot()
    while True:
        try:
            time.sleep(POLL_SECONDS)
            cur = snapshot()
            if cur == last:
                continue
            # Debounce: wait for the filesystem to settle before building (editors write in bursts).
            while True:
                time.sleep(DEBOUNCE_SECONDS)
                nxt = snapshot()
                if nxt == cur:
                    break
                cur = nxt
            changed = {k for k in set(cur) | set(last) if cur.get(k) != last.get(k)}
            skip = not deck_affecting(changed)
            ok, msg = build(skip_decks=skip)
            last = snapshot()
            if ok:
                _last_error[0] = ""
                print("  rebuilt" + (" (decks skipped)" if skip else "") + ": " + msg, flush=True)
                broadcast("reload")
            else:
                _last_error[0] = msg
                print("  BUILD FAILED (shown in browser):\n" + msg, flush=True)
                broadcast("error\n" + msg)
        except KeyboardInterrupt:
            raise  # let main() print "stopped" and exit
        except Exception as e:
            # A transient error (a file vanishing mid-scan, an odd OS hiccup) must NOT kill the
            # watcher — that would silently stop every future rebuild, the most confusing failure
            # there is. Log one line and keep polling.
            print(f"  watch hiccup ({type(e).__name__}: {e}); still watching", flush=True)
            last = snapshot()


def make_server(port):
    """Bind dual-stack so both http://localhost (::1 on Windows) and 127.0.0.1 reach the
    server; fall back to IPv4-only where the machine has no IPv6 stack."""
    if socket.has_ipv6:
        try:
            DevServer.address_family = socket.AF_INET6
            return DevServer(("::", port), Handler)
        except OSError:
            pass  # no usable IPv6 (or the port is taken there) — try plain IPv4
    DevServer.address_family = socket.AF_INET
    return DevServer(("0.0.0.0", port), Handler)


def main():
    explicit = len(sys.argv) > 1 and bool(sys.argv[1].strip())
    port = pick_port(sys.argv)
    server = None
    while server is None:
        try:
            server = make_server(port)
        except OSError as e:
            # The chosen port raced (someone grabbed it between the free check and bind). For an
            # auto-picked port, step to the next; for an explicit one, fail clearly.
            if explicit:
                print(f"could not bind port {port}: {e}", flush=True)
                return
            port += 1
            if port > 3100:
                print("could not find a free port in 3000-3100", flush=True)
                return
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"→ serving on http://localhost:{port}  (Ctrl-C to stop)", flush=True)
    try:
        watch_loop()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)


if __name__ == "__main__":
    main()
