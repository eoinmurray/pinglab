"""Gate: experiment runners expose only META flags — never science parameters.

The demolab invariant is "the runner is the recipe": a committed experiment must
run the same way every time, so its physics/training parameters (weights, rates,
τ_GABA, lr, epochs, seeds, dt, …) are hardcoded literals in the runner, not
overridable CLI flags. Only a small set of *meta* flags — which don't change the
science, just how/whether it runs or re-renders — may be accepted.

This scans each experiments/exp<NNN>.py for the flags it accepts (argparse
add_argument + bare `"--flag" in sys.argv` checks) and fails on anything outside
the meta allowlist. A new science flag lights up here until it is either hardcoded
or (if genuinely meta) added to ALLOWED with justification.

Scope note: only canonical runners exp<digits>.py are governed; helper/dispatch
scripts (e.g. exp022_runpod.py) are excluded by the filename pattern.
"""
import re
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1]

from experiments.helpers.cli import ALL_META_FLAGS

# Synced with helpers/cli.py — the closed meta vocabulary (+ legacy wipe/replot).
ALLOWED_EXACT = set(ALL_META_FLAGS) | {"--no-wipe-dir", "--wipe-dir", "--replot"}

# Canonical runners only — exp<digits>.py, so exp022_runpod.py et al. are skipped.
RUNNERS = sorted(p for p in EXPERIMENTS.glob("exp*.py") if re.fullmatch(r"exp\d+\.py", p.name))


def _is_meta(flag: str) -> bool:
    return flag in ALLOWED_EXACT


def _accepted_flags(src: str) -> set[str]:
    argparse_flags = set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', src))
    argv_flags = set(re.findall(r'"(--[a-z0-9-]+)"\s+(?:not\s+)?in\s+sys\.argv', src))
    return argparse_flags | argv_flags


def test_runners_exist():
    # Guards against a glob/path mistake silently passing the gate on zero files.
    assert RUNNERS, f"no exp<NNN>.py runners found under {EXPERIMENTS}"


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda p: p.name)
def test_runner_accepts_only_meta_flags(runner):
    offenders = sorted(f for f in _accepted_flags(runner.read_text()) if not _is_meta(f))
    assert not offenders, (
        f"{runner.name} exposes non-meta CLI flag(s) {offenders} — the runner is the "
        f"recipe, so science parameters must be hardcoded, not accepted as flags. "
        f"If a flag is genuinely meta, add it to ALLOWED in {Path(__file__).name}."
    )
