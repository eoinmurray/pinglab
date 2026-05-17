"""Terminal color + width primitives. TTY-aware: stdout files get plain text."""

from __future__ import annotations

import re
import sys

WIDTH = 80
DIVIDER = "─" * 40

_IS_TTY = sys.stdout.isatty()


def c(code: str, text: str) -> str:
    """Apply ANSI color if stdout is a TTY; return plain otherwise."""
    if not _IS_TTY:
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def bold(text):
    return c("1", text)


def green(text):
    return c("32", text)


def yellow(text):
    return c("33", text)


def red(text):
    return c("31", text)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)
