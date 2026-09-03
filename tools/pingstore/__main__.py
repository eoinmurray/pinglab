"""Allow python -m pingstore to use the same narrow command interface."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
