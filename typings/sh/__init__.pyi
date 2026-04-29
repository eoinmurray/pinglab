"""Minimal typing stub for the `sh` library.

`sh` exposes every executable on $PATH as a dynamic attribute on the module,
so static checkers can't see them. We declare the few we actually call so
`uvx ty check` passes without per-line ignores.
"""
from typing import Any

def __getattr__(name: str) -> Any: ...

class Command:
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

def uv(*args: Any, **kwargs: Any) -> Any: ...
def ffmpeg(*args: Any, **kwargs: Any) -> Any: ...
