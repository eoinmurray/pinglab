"""Oscilloscope run logging: structured intro / progress / summary.

Three sections:
  1. Intro — grouped config dump (Data / Simulation / Network / Training / Output)
  2. Progress — per-epoch / per-frame compact line with ETA
  3. Summary — result / dynamics / files / warnings

Also handles:
  - Provenance metadata (git SHA, seed, torch version, run_id)
  - Warning detection (dead / saturated / no progress / NaN / grad-explosion)
  - TTY-aware color (stdout only, log files plain)
  - Enriched .running marker file
  - metrics.jsonl sidecar
  - test_predictions.json sidecar

All output fits in 80 columns.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


# ── 80-col layout ────────────────────────────────────────────────────────
WIDTH = 80
DIVIDER = "─" * 40


# ── Color (TTY-aware) ────────────────────────────────────────────────────
_IS_TTY = sys.stdout.isatty()


def c(code: str, text: str) -> str:
    """Apply ANSI color if stdout is a TTY; return plain otherwise."""
    if not _IS_TTY:
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def dim(text): return c("2", text)
def bold(text): return c("1", text)
def green(text): return c("32", text)
def yellow(text): return c("33", text)
def red(text): return c("31", text)


# ── Provenance ───────────────────────────────────────────────────────────

def _git_sha() -> str:
    """Return current git SHA with '(dirty)' suffix if uncommitted changes."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=2
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff-index", "--quiet", "HEAD"],
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=2
        )
        return f"{sha} (dirty)" if dirty else sha
    except Exception:
        return "unknown"


def _env_hash() -> str:
    """Hash of uv.lock (or pyproject.toml fallback) for env reproducibility."""
    for name in ("uv.lock", "pyproject.toml"):
        p = Path(name)
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
            return f"{name}:{h}"
    return "unknown"


def run_id() -> str:
    """Compact run ID: r-YYYYMMDD-HHMMSS."""
    now = datetime.datetime.now()
    return f"r-{now.strftime('%Y%m%d-%H%M%S')}"


def provenance() -> dict:
    """Return a provenance dict to embed in config.json."""
    import torch
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"
    return {
        "git_sha": _git_sha(),
        "torch_version": torch.__version__,
        "device": device,
        "python_env_hash": _env_hash(),
        "run_id": run_id(),
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


# ── .running marker ──────────────────────────────────────────────────────

def write_running_marker(out_dir: Path, run_id_str: str) -> Path:
    """Write enriched .running file with PID, start time, run_id, cmd.

    Deleted by atexit hook in caller.
    """
    marker = Path(out_dir) / ".running"
    try:
        marker.write_text(
            f"pid={os.getpid()}\n"
            f"started={datetime.datetime.now().isoformat(timespec='seconds')}\n"
            f"run_id={run_id_str}\n"
            f"cmd={' '.join(sys.argv)}\n"
        )
    except Exception:
        pass
    return marker


# ── Intro block ──────────────────────────────────────────────────────────

def _fmt_kv(key: str, val, width: int = 16) -> str:
    """Format a key: value pair with key padded to `width`."""
    return f"  {key:<{width}} {val}"


def print_intro(log, mode: str, model: str, dataset: str, sections: dict):
    """Print the structured intro block.

    sections: {"Data": {"dataset": "mnist", ...}, "Simulation": {...}, ...}
    Each section is printed as a header + indented key: value lines.
    """
    log.info(bold(f"{mode} | {model} on {dataset}"))
    log.info(DIVIDER)
    for section_name, items in sections.items():
        if not items:
            continue
        log.info(section_name)
        for k, v in items.items():
            if v is None or v == "":
                continue
            log.info(_fmt_kv(f"{k}:", v))
        log.info("")
    log.info(DIVIDER)


# ── Progress lines ───────────────────────────────────────────────────────

def format_eta(seconds: float) -> str:
    """Compact ETA string: 8m30s, 45s, 1h12m."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def print_progress_header(log):
    """Print column header before the per-epoch stream."""
    log.info(
        "  ep       acc      loss  |  E       I      CV    act   |  time   eta"
    )
    log.info("  " + "─" * (WIDTH - 2))


def print_epoch(log, ep: int, total: int, acc: float, loss: float,
                 e_rate: float, i_rate: float, cv: float, activity: float,
                 elapsed_s: float, eta_s: float,
                 new_best: bool = False, warnings: list = None):
    """One compact progress line per epoch. Fits in 80 cols."""
    arrow = green("↑") if new_best else " "
    i_str = f"{i_rate:3.0f}Hz" if i_rate is not None else "  -  "
    line = (
        f"  {ep:>3}/{total:<3}  {acc:>3.0f}%{arrow}  {loss:>6.2f}"
        f"  |  {e_rate:3.0f}Hz  {i_str}  {cv:>3.2f}  {activity:>3.0f}%"
        f"  |  {int(elapsed_s):>3}s  {format_eta(eta_s):>5}"
    )
    if warnings:
        tag = " " + " ".join(warnings)
        # Keep tag within total budget — truncate if overflow
        if len(_strip_ansi(line)) + len(_strip_ansi(tag)) > WIDTH:
            pass  # acceptable overflow — flags are more important than width
        line += tag
    log.info(line)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


# ── Warning detector ─────────────────────────────────────────────────────

class WarningTracker:
    """Tracks rolling dynamics state to flag dead / saturated / no-progress."""

    def __init__(self):
        self.dead_streak = 0
        self.saturated_streak = 0
        self.best_acc = 0.0
        self.best_epoch = 0
        self.no_progress_since = 0
        self.grad_clip_frac_last = 0.0
        self.observed_warnings = []  # (ep_start, ep_end, kind) aggregated

    def tick(self, ep: int, acc: float, activity: float, loss: float = None,
             grad_clip_frac: float = 0.0):
        flags = []
        # Activity flags only trigger when paired with no improvement —
        # extreme firing rates alone aren't pathological if the network
        # is still learning. Sparse-coding nets can run at ~1% activity;
        # gamma-locked PING can sustain >80% — both fine if accuracy climbs.
        stuck = self.no_progress_since >= 5

        # dead: silent for 3+ epochs AND not improving
        if activity < 1.0:
            self.dead_streak += 1
            if self.dead_streak >= 3 and stuck:
                flags.append(yellow("⚠ dead"))
                self._record(ep, "dead")
        else:
            self.dead_streak = 0

        # saturated: nearly-all-firing for 3+ epochs AND not improving
        if activity > 95.0:
            self.saturated_streak += 1
            if self.saturated_streak >= 3 and stuck:
                flags.append(yellow("⚠ saturated"))
                self._record(ep, "saturated")
        else:
            self.saturated_streak = 0

        # NaN
        if loss is not None and (loss != loss):  # NaN check
            flags.append(red("⚠ NaN"))
            self._record(ep, "NaN")

        # grad explosion
        if grad_clip_frac > 0.5:
            flags.append(yellow("⚠ grad-clip"))
            self._record(ep, "grad-clip")

        # new best tracking
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = ep
            self.no_progress_since = 0
        else:
            self.no_progress_since += 1

        if self.no_progress_since >= 10:
            flags.append(yellow("⚠ stuck"))
            self._record(ep, "stuck")

        return flags

    def _record(self, ep: int, kind: str):
        # Extend existing run of same kind or start new
        if self.observed_warnings and self.observed_warnings[-1][2] == kind \
                and self.observed_warnings[-1][1] == ep - 1:
            s, _, k = self.observed_warnings[-1]
            self.observed_warnings[-1] = (s, ep, k)
        else:
            self.observed_warnings.append((ep, ep, kind))

    def summary_lines(self) -> list:
        """Human-readable aggregated warnings for the summary block."""
        if not self.observed_warnings:
            return []
        lines = []
        for start, end, kind in self.observed_warnings:
            span = f"ep {start}" if start == end else f"ep {start}-{end}"
            lines.append(f"  ⚠ {kind} at {span}")
        return lines


# ── Summary block ────────────────────────────────────────────────────────

def format_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def list_output_files(out_dir: Path) -> list:
    """List all files in out_dir (recursively) with sizes."""
    out_dir = Path(out_dir)
    files = []
    if not out_dir.exists():
        return files
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(out_dir)
            files.append((str(rel), p.stat().st_size))
    return files


def print_summary(log, *, best_acc: float = None, final_acc: float = None,
                   best_epoch: int = None, runtime_s: float = None,
                   dynamics: dict = None, out_dir: Path = None,
                   warnings: list = None):
    """Print the structured summary block."""
    log.info(DIVIDER)
    if best_acc is not None:
        log.info("Result")
        log.info(_fmt_kv("best:", f"{best_acc:.0f}%"
                         + (f"  (ep {best_epoch})" if best_epoch else "")))
        if final_acc is not None and final_acc != best_acc:
            log.info(_fmt_kv("final:", f"{final_acc:.0f}%"))
        if runtime_s is not None:
            log.info(_fmt_kv("runtime:", format_eta(runtime_s)))
        log.info("")
    if dynamics:
        log.info("Dynamics (end state)")
        for k, v in dynamics.items():
            if v is None:
                continue
            log.info(_fmt_kv(f"{k}:", v))
        log.info("")
    if out_dir is not None:
        files = list_output_files(out_dir)
        if files:
            log.info("Files written")
            abs_dir = str(Path(out_dir).resolve())
            log.info(f"  → {abs_dir}/")
            # Group dirs: collapse frames/*.png into one line
            dir_groups = {}
            standalone = []
            for rel, sz in files:
                parts = rel.split(os.sep)
                if len(parts) > 1:
                    d = parts[0]
                    dir_groups.setdefault(d, []).append(sz)
                else:
                    standalone.append((rel, sz))
            for rel, sz in standalone:
                log.info(f"    {rel:<22} {format_bytes(sz):>10}")
            for d, sizes in dir_groups.items():
                total = sum(sizes)
                log.info(f"    {d + '/':<22} ({len(sizes)} files, "
                          f"{format_bytes(total)})")
        log.info("")
    if warnings:
        log.info("Warnings")
        for w in warnings:
            log.info(w)
        log.info("")
    log.info(DIVIDER)


# ── Sidecars: metrics.jsonl, test_predictions.json ──────────────────────

class MetricsJsonl:
    """Append-only JSONL writer for per-epoch (or per-step) metrics."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w")

    def write(self, **fields):
        fields.setdefault("timestamp", datetime.datetime.now().isoformat(
            timespec="seconds"))
        self._f.write(json.dumps(fields) + "\n")
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


def write_test_predictions(path: Path, predictions: list):
    """Save list of {idx, true, pred, correct, logits} records to JSON."""
    with open(path, "w") as f:
        json.dump(predictions, f, indent=2, default=float)
