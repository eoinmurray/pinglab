"""--tier CLI parsing for notebook runners.

Each runner passes its accepted tier names (keys of whatever per-tier config
dict it carries) and a default. Returns the selected tier.
"""
from __future__ import annotations

from collections.abc import Iterable


def parse_tier(argv: list[str], *, choices: Iterable[str], default: str) -> str:
    choices = list(choices)
    if default not in choices:
        raise ValueError(f"default tier {default!r} not in choices {choices}")
    if "--tier" not in argv:
        return default
    idx = argv.index("--tier")
    if idx + 1 >= len(argv):
        raise SystemExit("--tier requires a value")
    tier = argv[idx + 1]
    if tier not in choices:
        raise SystemExit(f"--tier: unknown tier {tier!r}, choose from {choices}")
    return tier
