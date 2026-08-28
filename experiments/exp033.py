"""Retired combined runner; pure helpers remain importable by exp054."""

from experiments.exp033 import *  # noqa: F403


def load_exp041_fgamma():
    raise RuntimeError(
        "exp033 requires an explicit exp041 analysis input; migrate this consumer"
    )


if __name__ == "__main__":
    raise SystemExit("exp033 requires an explicit compute, analyse or present stage")
