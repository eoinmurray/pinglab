# Python Code Review (uv-idiomatic)

Review Python code following modern uv-based workflows and idioms.

## When to use

- User asks to review Python code
- User asks about Python best practices
- User wants feedback on Python project structure

## Review checklist

### Project structure
- `pyproject.toml` is the single source of truth for dependencies and metadata
- No `requirements.txt`, `setup.py`, or `setup.cfg` unless absolutely necessary for compatibility
- Use `uv.lock` for reproducible installs (commit this file)

### Dependencies
- Declare dependencies in `pyproject.toml` under `[project.dependencies]`
- Use `uv add <package>` to add dependencies (not pip install)
- Use `uv add --dev <package>` for dev dependencies
- Pin versions appropriately: `>=1.0,<2.0` for libraries, exact pins in lockfile

### Virtual environments
- Let uv manage the venv (`.venv/` in project root)
- Run commands with `uv run <command>` instead of activating venv manually
- Use `uv run python script.py` or `uv run pytest`

### Scripts and tools
- Define scripts in `pyproject.toml` under `[project.scripts]`
- Use `uv tool install` for global CLI tools (ruff, mypy, etc.)
- Prefer `uv run ruff check .` over globally installed linters

### Code style
- Use `ruff` for linting and formatting (replaces black, isort, flake8)
- Configure ruff in `pyproject.toml` under `[tool.ruff]`
- Use type hints; check with `mypy` or `pyright`

### Common issues to flag
- `pip install` commands (should be `uv add` or `uv pip install`)
- `python -m venv` (should be `uv venv` or implicit)
- Separate `requirements-dev.txt` (use `[project.optional-dependencies]` or `[dependency-groups]`)
- Missing `uv.lock` in version control
- `pip freeze > requirements.txt` patterns

## Example pyproject.toml structure

```toml
[project]
name = "myproject"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "httpx>=0.27",
    "pydantic>=2.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.8",
    "mypy>=1.13",
]

[tool.ruff]
line-length = 88
select = ["E", "F", "I", "UP"]

[tool.ruff.format]
quote-style = "double"
```

## Running checks

```bash
uv run ruff check .
uv run ruff format .
uv run mypy .
uv run pytest
```
