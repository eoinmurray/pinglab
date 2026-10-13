"""Internal native training worker invoked by the validated bank plan."""

import argparse
from pathlib import Path

from experiments.exp022.compute import _cell_by_name, _writable_compute_root
from experiments.exp022.training import train_cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-cell", required=True)
    parser.add_argument("--native-output", type=Path, required=True)
    parser.add_argument("--native-samples", type=int, required=True)
    parser.add_argument("--native-epochs", type=int, required=True)
    args = parser.parse_args()
    cell = _cell_by_name(args.native_cell)
    if cell is None:
        parser.error("unregistered cell")
    _writable_compute_root(args.native_output)
    train_cell(cell, args.native_output, args.native_samples, args.native_epochs)


if __name__ == "__main__":
    main()
