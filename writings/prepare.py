"""Lab-owned presentation hook: declared experiment dependencies and validated runs."""

from pathlib import Path

from pingstore.presentation_inputs import prepare

from experiments.collections.gamma_gated_sparsity.graph import ordered_experiments


def declared_dependencies() -> dict[str, tuple[str, ...]]:
    dependencies = {item.slug: item.dependencies for item in ordered_experiments()}
    # exp048 is outside the campaign graph but consumes the exp022 bank.
    dependencies["exp048"] = ("exp022",)
    # The scheduling graph reaches exp022 through exp041; exp046 also reads it directly.
    dependencies["exp046"] = (*dependencies["exp046"], "exp022")
    return dependencies


if __name__ == "__main__":
    raise SystemExit(prepare(
        Path(__file__).resolve().parents[1],
        declared_dependencies=declared_dependencies(),
    ))
