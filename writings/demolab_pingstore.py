"""Demolab-Pingstore connector: declared dependencies and validated presentation runs."""

from pathlib import Path

from pingstore.presentation_inputs import prepare


def declared_dependencies() -> dict[str, tuple[str, ...]]:
    """Article data dependencies used to select compatible presentations.

    This is publication metadata, not an execution or scheduling graph.
    """
    return {
        "exp022": (),
        "exp023": (),
        "exp024": ("exp022",),
        "exp025": ("exp022",),
        "exp033": ("exp041",),
        "exp037": ("exp022",),
        "exp038": ("exp022",),
        "exp041": ("exp022",),
        "exp042": ("exp022",),
        "exp044": ("exp022",),
        "exp046": ("exp041", "exp022"),
        "exp047": (),
        "exp049": ("exp022",),
        "exp054": ("exp041",),
        "exp080": (),
        "exp081": (),
        "exp082": ("exp022",),
        "exp110": ("exp025", "exp037", "exp038", "exp041", "exp044", "exp046", "exp054"),
    }


if __name__ == "__main__":
    raise SystemExit(prepare(
        Path(__file__).resolve().parents[1],
        declared_dependencies=declared_dependencies(),
    ))
