"""Console logging for notebook runners — the runlog look, notebook-side.

The pinglab-cli renders every run through cli/runlog.py (banner ▸ phases ▸
epoch table ▸ summary), so a CLI run reads like an instrument panel. A notebook
run, by contrast, has historically been a scatter of ad-hoc prints — [scope],
[f-I], "wrote X" — each runner inventing its own bracket-tag convention. This
module gives notebooks the SAME visual vocabulary (glyphs, palette, aligned
right-hand cost column) without importing cli: the notebook↔cli boundary is a
hard gate (src/notebooks/ruff.toml TID251), so the small shared look — the
ANSI palette, the ◆ ▸ → ✓ glyphs, the eta formatter — is deliberately
re-stated here rather than imported, exactly as helpers/provenance.py re-states
the git-capture logic. It stays stdlib-only (no torch, no cli, no models) so it
can never trip the gate.

Console only: this writes to the terminal, nothing else. A notebook's canonical
machine record is still the figures-dir _manifest.json (helpers/provenance.py);
nblog does not open a JSONL sidecar. Colour appears on a TTY and is stripped
when stdout is redirected (a log file, CI), so a captured run stays greppable.

Typical use — one Run per notebook, opened at the top of main():

    log = nblog.Run("nb023", run_id, subtitle="PING fundamentals")
    log.phase("architecture", "schematic")
    log.wrote(arch_path, "svg,pdf")

    bar = log.bar(len(cells) * len(rates), "f-I sweep")
    for cell in cells:
        for rate in rates:
            ...                       # do the work
            bar.tick(f"{cell} {rate}Hz -> E {e:.0f}")
    bar.done()

    log.result("f_gamma[ping]", "41.2 Hz")
    log.summary(duration_s, out_dir=figures)
"""

from __future__ import annotations

import atexit
import re
import sys
import time
from pathlib import Path

# ── palette ──────────────────────────────────────────────────────────────
# Mirrors cli/runlog.py so a notebook and a CLI run read the same. Colour is
# emitted only to a TTY; a redirected stream (log file, CI) gets plain text.

WIDTH = 70  # inner width of the banner rule and the right-aligned cost column
_IS_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if _IS_TTY else text


def bold(t: str) -> str:
    return _c("1", t)


def dim(t: str) -> str:
    return _c("2", t)


def cyan(t: str) -> str:
    return _c("36", t)


def green(t: str) -> str:
    return _c("32", t)


def yellow(t: str) -> str:
    return _c("33", t)


def red(t: str) -> str:
    return _c("31", t)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _vis_len(s: str) -> int:
    """Visible length — ANSI SGR codes are zero-width."""
    return len(_strip_ansi(s))


# Glyphs — one copy of the runlog visual language.
_BANNER = "◆"
_RULE = "─"
_RULE_HEAVY = "━"
_PHASE = "▸"
_OK = "✓"
_ARROW = "→"
_STATE = "◈"
_CLOCK = "◷"
_DOT = "·"
_FILL = "█"
_TRACK = "░"


def format_eta(seconds: float) -> str:
    """Compact duration: 45s, 8m30s, 1h12m — matches runlog.format_eta."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


# ── terminal cursor ──────────────────────────────────────────────────────
# The in-place progress bar leaves the cursor stranded on the bar line; hide it
# while a bar is live and restore it when the run ends. An atexit backstop
# guarantees the terminal is usable again even if a run crashes mid-bar.

_cursor_hidden = False


def _hide_cursor() -> None:
    global _cursor_hidden
    if _IS_TTY and not _cursor_hidden:
        sys.stdout.write("\x1b[?25l")
        sys.stdout.flush()
        _cursor_hidden = True


def _show_cursor() -> None:
    global _cursor_hidden
    if _IS_TTY and _cursor_hidden:
        sys.stdout.write("\x1b[?25h")
        sys.stdout.flush()
        _cursor_hidden = False


atexit.register(_show_cursor)


class ProgressBar:
    """A single in-place progress bar for a bounded loop of N steps.

    Two renderings, chosen by whether stdout is a TTY (identical policy to
    runlog.Heartbeat):

      • terminal (TTY) — one line rewritten in place on each tick (carriage
        return + clear-to-EOL), so a bar fills where it stands instead of
        scrolling a wall of lines. done() finalises the line and drops to the
        next row.
      • non-TTY (log file, CI, remote pod) — no control characters: a plain line is
        emitted at most every `log_interval` seconds, plus one at completion.

    The bar owns the terminal while live, so the caller should keep the wrapped
    work quiet (capture any subprocess stdout) — see nb023's sim loop.
    """

    def __init__(self, total: int, label: str, *, log_interval: float = 5.0):
        self.total = max(1, int(total))
        self.label = label
        self._n = 0
        self._start = time.monotonic()
        self._last_render = 0.0
        self._log_interval = log_interval
        self._live = False  # a bar line is on screen awaiting overwrite/clear

    def _line(self, note: str) -> str:
        frac = self._n / self.total
        width = 20
        filled = int(round(frac * width))
        bar = _FILL * filled + _TRACK * (width - filled)
        elapsed = time.monotonic() - self._start
        left = (
            f"  {cyan(_PHASE)} {self.label}  "
            f"{dim('[')}{cyan(bar)}{dim(']')}  "
            f"{self._n}/{self.total}"
        )
        right = format_eta(elapsed)
        note_s = f"  {dim(note)}" if note else ""
        pad = max(1, WIDTH - _vis_len(left) - _vis_len(note_s) - _vis_len(right))
        return left + note_s + " " * pad + dim(right)

    def tick(self, note: str = "") -> None:
        """Advance one step and repaint the bar. `note` is the trailing status
        (e.g. the item just finished and its headline number)."""
        self._n += 1
        line = self._line(note)
        if _IS_TTY:
            _hide_cursor()
            sys.stdout.write("\r\x1b[2K" + line)
            sys.stdout.flush()
            self._live = True
        else:
            now = time.monotonic()
            if now - self._last_render >= self._log_interval or self._n >= self.total:
                self._last_render = now
                print(_strip_ansi(line))

    def done(self, note: str = "") -> None:
        """Finalise: leave the completed bar on its own line and hand back the
        cursor. Safe to call once after the loop."""
        if _IS_TTY:
            if self._n < self.total:
                self._n = self.total
            sys.stdout.write("\r\x1b[2K" + self._line(note or "done") + "\n")
            sys.stdout.flush()
            self._live = False
            _show_cursor()
        else:
            print(_strip_ansi(self._line(note or "done")))


class Run:
    """A notebook run's console logger — banner, phases, wrote-lines, summary.

    Stateless beyond the identity header and a start clock; every method just
    prints. `wrote()` accumulates the paths it is told about so `summary()` can
    report a count without walking the figures dir.
    """

    def __init__(self, slug: str, run_id: str, *, subtitle: str = ""):
        self.slug = slug
        self.run_id = run_id
        self._start = time.monotonic()
        self._wrote: list[Path] = []
        self._banner(subtitle)

    # ── banner ───────────────────────────────────────────────────────────
    def _banner(self, subtitle: str) -> None:
        left = (
            f"{cyan(_BANNER)} {bold('pinglab')} {dim(_DOT)} "
            f"{cyan(self.slug)} {dim(_DOT)} {self.run_id}"
        )
        pad = max(1, WIDTH - _vis_len(left) - _vis_len(subtitle))
        print(left + " " * pad + dim(subtitle))
        print(dim(_RULE_HEAVY * WIDTH))

    # ── setup / step markers ─────────────────────────────────────────────
    def phase(self, name: str, detail: str = "") -> None:
        """One step: ▸ <name> .......... <detail>. The right column is the cost
        or descriptor (a param count, a cell's argv summary)."""
        left = f"  {cyan(_PHASE)} {name}"
        pad = max(1, WIDTH - _vis_len(left) - _vis_len(detail))
        print(left + " " * pad + dim(detail))

    def wrote(self, path: Path | str, formats: str = "") -> None:
        """An artifact landed: → <path>[.{fmts}]. Records the path for the
        summary count. `formats` is a bare comma list, e.g. "svg,pdf"."""
        p = Path(path)
        self._wrote.append(p)
        shown = f"{p}.{{{formats}}}" if formats else str(p)
        print(f"  {cyan(_ARROW)} {dim(shown)}")

    def result(self, label: str, value: str) -> None:
        """A headline number: ◈ <label>  <value>. For the few scalars a run
        wants echoed to the terminal (the full set lives in numbers.json)."""
        print(f"  {cyan(_STATE)} {dim(f'{label:<16}')} {value}")

    # ── progress ─────────────────────────────────────────────────────────
    def bar(self, total: int, label: str) -> ProgressBar:
        """Open a progress bar for a bounded loop of `total` steps."""
        return ProgressBar(total, label)

    # ── close ────────────────────────────────────────────────────────────
    def summary(self, duration_s: float | None = None, *, out_dir: Path | None = None) -> None:
        """Closing block: a rule, the elapsed time, and where output landed."""
        if duration_s is None:
            duration_s = time.monotonic() - self._start
        print(dim(_RULE * WIDTH))
        bits = [format_eta(duration_s)]
        if self._wrote:
            bits.append(f"{len(self._wrote)} figures")
        print(f"  {green(_OK)} {dim('done')}     " + f" {dim(_DOT)} ".join(bits))
        if out_dir is not None:
            print(f"  {cyan(_ARROW)} {dim(str(Path(out_dir).resolve()) + '/')}")
        print(dim(_RULE * WIDTH))
