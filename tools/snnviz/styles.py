"""Styling values that compositions may adopt, override, or ignore."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    background: str = "#f3efe6"
    ink: str = "#20201e"
    accent: str = "#a62a24"
    muted: str = "#77726a"
