"""Scientific dependency graph without duplicated experiment parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass

COLLECTION = "gamma-gated-sparsity"


@dataclass(frozen=True)
class Experiment:
    slug: str
    dependencies: tuple[str, ...] = ()
    training_run: str | None = None
    integrated: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


# Parameters remain owned by each experiment. This graph records only hard data
# dependencies and the exp022 training-run identity consumed downstream.
EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment("exp022", note="collection checkpoint bank"),
    Experiment("exp024", ("exp022",), "TR-02", integrated=True),
    Experiment("exp025", ("exp022",), "TR-02"),
    Experiment("exp037", ("exp022",), "TR-02", integrated=True),
    Experiment("exp038", ("exp022",), "TR-02", integrated=True),
    Experiment("exp041", ("exp022",), "TR-03"),
    Experiment("exp044", ("exp022",), "TR-04"),
    Experiment("exp049", ("exp022",), "TR-05"),
    Experiment("exp082", ("exp022",), "TR-06"),
    Experiment("exp033", ("exp041",)),
    Experiment("exp042", ("exp041",)),
    Experiment("exp046", ("exp041",)),
)

# These roots are deliberately not in the executable graph until the Stage 1
# shared-code audit records rerun or reuse for each one.
PENDING_ROOT_DECISIONS: tuple[str, ...] = (
    "exp023",
    "exp047",
    "exp054",
    "exp080",
    "exp081",
)


def ordered_experiments(
    experiments: tuple[Experiment, ...] = EXPERIMENTS,
) -> tuple[Experiment, ...]:
    """Return a stable topological ordering, rejecting malformed graphs."""
    by_slug = {experiment.slug: experiment for experiment in experiments}
    if len(by_slug) != len(experiments):
        raise ValueError("collection experiment IDs must be unique")
    unknown = {
        dependency
        for experiment in experiments
        for dependency in experiment.dependencies
        if dependency not in by_slug
    }
    if unknown:
        raise ValueError(f"unknown collection dependencies: {sorted(unknown)}")

    ordered: list[Experiment] = []
    remaining = list(experiments)
    while remaining:
        ready = [
            experiment
            for experiment in remaining
            if all(dep in {item.slug for item in ordered} for dep in experiment.dependencies)
        ]
        if not ready:
            raise ValueError("collection dependency graph contains a cycle")
        for experiment in ready:
            ordered.append(experiment)
            remaining.remove(experiment)
    return tuple(ordered)
