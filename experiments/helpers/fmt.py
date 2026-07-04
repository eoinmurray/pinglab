"""Human-readable run timestamp + duration formatting for notebook summaries.

Both helpers are presentation-only (no scientific content): `format_duration`
renders an elapsed-seconds count as e.g. "2m 03s", and `format_run_datetime`
renders an aware datetime as e.g. "Monday, 16th June 26 at 14:05".
"""

from __future__ import annotations

from datetime import datetime


def format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def format_run_datetime(dt: datetime) -> str:
    day = dt.day
    suffix = (
        "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    )
    return dt.strftime(f"%A, {day}{suffix} %B %y at %H:%M")
