"""pinglab experiment harness — single CLI entry point.

Usage:
    uv run python src/pinglab/harness.py --help
    uv run python src/pinglab/harness.py mnist-dt-stability --help
    uv run python src/pinglab/harness.py mnist-dt-stability experiment --size standard
    uv run python src/pinglab/harness.py ping-smnist-advantage ladder --only coba
    uv run python src/pinglab/harness.py test
"""
from __future__ import annotations

import typer

from pinglab.experiments.mnist_dt_stability import runner as mnist_dt_stability
from pinglab.experiments.ping_smnist_advantage import runner as ping_smnist_advantage

app = typer.Typer(
    help="pinglab — SNN research harness.",
    no_args_is_help=True,
    add_completion=False,
)

app.add_typer(
    mnist_dt_stability.app,
    name="mnist-dt-stability",
    help="Goal 1: CUBA vs COBA dt-stability on MNIST",
)
app.add_typer(
    ping_smnist_advantage.app,
    name="ping-smnist-advantage",
    help="Goal 2: does PING's E-I loop help on sequential MNIST?",
)


@app.command()
def test(slow: bool = typer.Option(False, help="Include slow + regression tests")):
    """Run the unit test suite."""
    import subprocess
    args = ["uv", "run", "pytest"]
    if not slow:
        # Pass as a single -m expression so zsh doesn't word-split
        args += ["-m", "not slow and not regression"]
    raise typer.Exit(code=subprocess.call(args))


if __name__ == "__main__":
    app()
