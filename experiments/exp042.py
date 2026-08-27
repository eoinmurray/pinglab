"""Compatibility launcher: exp042 requires an explicit independent stage."""

if __name__ == "__main__":
    import runpy
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("experiments.exp042", run_name="__main__")
