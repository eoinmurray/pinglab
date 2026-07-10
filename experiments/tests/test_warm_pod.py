"""Unit tests for the warm-pod idle lease (helpers/warm_pod.py).

Deterministic — no real pods, no real processes. `ps axww` and the wall clock
are mocked so the watcher's renew/stop logic is exercised in-memory:
  • _runner_active parses ps output and matches only real runners (expNNN.py),
    never the watcher itself.
  • watch() stops the pod once the idle window passes with nothing running.
  • a live runner keeps renewing the lease, so a long run is never stopped.
  • the hard ceiling stops the pod even if activity never ceases (safety valve).
"""
from types import SimpleNamespace

import pytest
from experiments.helpers import warm_pod as w


def _fake_ps(stdout):
    """A subprocess.run stand-in that returns fixed `ps axww` output."""
    return lambda *a, **k: SimpleNamespace(stdout=stdout, returncode=0)


def test_runner_active_detects_runner(monkeypatch):
    monkeypatch.setattr(w.subprocess, "run", _fake_ps(
        "  1 ??  Ss  0:00 /usr/sbin/sshd -D\n"
        "123 pts/0 Rl 3:21 python experiments/exp042.py\n"))
    assert w._runner_active() is True


def test_runner_active_idle_ignores_watcher(monkeypatch):
    # The watcher's own cmdline must NOT count as a run, or it would hold itself
    # open forever.
    monkeypatch.setattr(w.subprocess, "run", _fake_ps(
        "  1 ??  Ss  0:00 /usr/sbin/sshd -D\n"
        " 42 pts/0 Sl 0:01 python experiments/helpers/warm_pod.py watch --hours 3\n"))
    assert w._runner_active() is False


def _fake_clock(monkeypatch, start=1000.0):
    """Patch _now() and time.sleep() onto a shared fake clock; sleeping advances
    it. Returns the clock dict so a test can read the elapsed time."""
    clock = {"t": start}
    monkeypatch.setattr(w, "_now", lambda: clock["t"])
    monkeypatch.setattr(w.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(w, "_write_lease", lambda *a, **k: None)
    return clock


def test_watch_stops_on_idle(monkeypatch):
    _fake_clock(monkeypatch)
    monkeypatch.setattr(w, "_runner_active", lambda: False)
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-1")
    stopped = []
    monkeypatch.setattr(w, "_stop_pod", lambda pid: stopped.append(pid))

    w.watch(hours=2 / 3600, poll_s=1, max_hours=24)  # 2-second idle window
    assert stopped == ["pod-1"]


def test_watch_renews_while_a_run_is_active(monkeypatch):
    clock = _fake_clock(monkeypatch)
    calls = {"n": 0}

    def active():
        calls["n"] += 1
        return calls["n"] <= 5  # a run occupies the first 5 polls, then finishes

    monkeypatch.setattr(w, "_runner_active", active)
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-1")
    stopped = []
    monkeypatch.setattr(w, "_stop_pod", lambda pid: stopped.append(pid))

    w.watch(hours=2 / 3600, poll_s=1, max_hours=24)  # 2s window, but 5s of activity
    assert stopped == ["pod-1"]
    # It ran well past the 2s idle window — the active run held it open (no
    # mid-flight kill), and it stopped only once activity ceased.
    assert clock["t"] - 1000.0 > 5


def test_watch_hard_ceiling_stops_even_if_always_busy(monkeypatch):
    _fake_clock(monkeypatch)
    monkeypatch.setattr(w, "_runner_active", lambda: True)  # never idle
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-1")
    stopped = []
    monkeypatch.setattr(w, "_stop_pod", lambda pid: stopped.append(pid))

    # Ceiling of 2 seconds despite perpetual activity — the safety valve fires.
    w.watch(hours=1, poll_s=60, max_hours=2 / 3600)
    assert stopped == ["pod-1"]


def test_watch_requires_pod_id(monkeypatch):
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    with pytest.raises(SystemExit):
        w.watch()
