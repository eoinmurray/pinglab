#!/usr/bin/env python
"""Warm (persistent) RunPod pod with a renewable idle lease.

The fan-out in runpod.py is fire-and-forget batch: pods self-run a CELLS
assignment and self-terminate. This is the opposite shape — ONE long-lived pod
you SSH into for short, interactive runs, kept alive only while you are actually
running experiments and STOPPED (not destroyed) after an idle window, so it
stops billing GPU but keeps its container disk and wakes fast.

Two halves, mirroring runpod.py:
  • launch (laptop side): create a Secure sshd-only pod on the shared volume,
    then arm the watcher on it over SSH so the watcher outlives your laptop.
  • watch (pod side): every POLL_S seconds, renew the lease to now+LEASE_HOURS
    whenever an experiment-runner process is alive; when the lease expires with
    nothing running, `runpodctl stop pod` self-stops the pod. `resume` re-arms it.

WHAT COUNTS AS "USING IT": a runner process — `experiments/expNNN.py` — is
running. That is the ONLY renewal signal (chosen over SSH-login or GPU-util):
merely connecting does not hold the pod open, and a long run can never be
stopped mid-flight, because every poll while it runs pushes the deadline out.
This needs NO runner edits and NO image rebuild — the watcher is injected over
SSH and detects activity by process, so every runner is covered for free.

Stop vs terminate: stop keeps the container disk (~cents/day storage) and wakes
fast; the /shared network volume survives either way. A hard ceiling
(MAX_WARM_HOURS from launch) stops the pod regardless, so a wedged activity
check can never bill forever.

CLI (run from the repo root):
    # laptop: create a warm 5090 pod with a 3-hour idle lease, arm the watcher
    uv run python experiments/helpers/warm_pod.py launch --gpu 5090 --hours 3
    uv run python experiments/helpers/warm_pod.py launch --dry-run     # plan only
    # laptop: wake a stopped pod and re-arm the same lease
    uv run python experiments/helpers/warm_pod.py resume <pod_id> --hours 3
    # laptop: stop it now / list warm pods
    uv run python experiments/helpers/warm_pod.py stop <pod_id>
    uv run python experiments/helpers/warm_pod.py status
    # pod side (armed by launch/resume — you do not call this yourself):
    uv run python experiments/helpers/warm_pod.py watch --hours 3
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Runner-style bootstrap: put experiments/ on sys.path so the laptop-side
# `launch`/`resume` can `from helpers import runpod`. The pod-side `watch` path
# imports nothing from the package (stdlib + runpodctl only), so it works even
# on an older baked image before this file's checkout lands.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Lease configuration ──────────────────────────────────────────────
LEASE_HOURS = 3.0            # idle window: stop after this long with no runner
POLL_S = 60                  # how often the watcher checks for activity
MAX_WARM_HOURS = 24.0        # hard ceiling from launch — stop regardless (safety)

# The lease + watcher log live on the shared network volume so they survive a
# stop and are visible to a collector pod. Overridable for local testing.
WARM_DIR = Path(os.environ.get("PINGLAB_WARM_DIR", "/shared/.warm"))

# A "runner is active" = a live `experiments/expNNN.py` process. The watcher's
# own cmdline is `experiments/helpers/warm_pod.py`, which does NOT match, so the
# watcher never counts itself as activity.
RUNNER_RE = re.compile(r"experiments/exp[0-9]{3}\.py")

# Reuse runpod.py's pod-side constant so the arm command uses the same uv.
UV = "/root/.local/bin/uv"
POD_REPO = "/workspace/pinglab"


def _now() -> float:
    """Wall-clock epoch seconds (the lease is wall-clock so it reads sanely in
    the file and survives a watcher restart)."""
    return time.time()


def _pod_id() -> str | None:
    """This pod's id, injected by RunPod as RUNPOD_POD_ID. None off a pod."""
    return os.environ.get("RUNPOD_POD_ID")


# ── Pod side: the watcher ────────────────────────────────────────────

def _runner_active() -> bool:
    """True iff an experiment-runner process is currently running on this pod.

    Lists every process with its full (untruncated, `ww`) command line and keeps
    only those matching RUNNER_RE. Uses `ps axww` rather than `pgrep -f` because
    pgrep's regex dialect and its `-a` flag are not portable (macOS pgrep has no
    `-a`), and this keeps the exact match in Python. The watcher's own cmdline
    (warm_pod.py) does not match RUNNER_RE, so it never counts itself."""
    out = subprocess.run(["ps", "axww"], capture_output=True, text=True)
    return any(RUNNER_RE.search(line) for line in out.stdout.splitlines())


def _write_lease(pod_id: str, deadline: float, hard_deadline: float) -> None:
    """Persist the current lease to WARM_DIR/<pod_id>.lease (human-readable, for
    `cat` on the pod). The watcher owns the live countdown in memory; this file
    is observability, not the source of truth."""
    WARM_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    (WARM_DIR / f"{pod_id}.lease").write_text(
        f"idle_deadline={int(deadline)}  # {time.strftime(fmt, time.gmtime(deadline))}\n"
        f"hard_deadline={int(hard_deadline)}  # {time.strftime(fmt, time.gmtime(hard_deadline))}\n"
        f"lease_hours={LEASE_HOURS}\n"
    )


def _stop_pod(pod_id: str) -> None:
    """Stop (not remove) this pod: releases the GPU and stops GPU billing while
    keeping the container disk for a fast `resume`. runpodctl auths via the
    injected RUNPOD_API_KEY env, exactly as pod_self_terminate does."""
    print(f"[warm-watch] idle → stopping pod {pod_id}", flush=True)
    r = subprocess.run(["runpodctl", "stop", "pod", pod_id],
                       capture_output=True, text=True)
    print(f"[warm-watch] runpodctl stop: {(r.stdout + r.stderr).strip()[:200]}",
          flush=True)


def watch(hours: float = LEASE_HOURS, poll_s: int = POLL_S,
          max_hours: float = MAX_WARM_HOURS) -> None:
    """Pod-side loop: renew while a runner is alive, else stop when the idle
    lease (or the hard ceiling) expires. Blocks until it stops the pod."""
    pod_id = _pod_id()
    if not pod_id:
        raise SystemExit("[warm-watch] no RUNPOD_POD_ID — this runs on a pod only")

    start = _now()
    hard_deadline = start + max_hours * 3600
    deadline = start + hours * 3600
    _write_lease(pod_id, deadline, hard_deadline)
    print(f"[warm-watch] armed on {pod_id}: idle={hours}h ceiling={max_hours}h "
          f"poll={poll_s}s", flush=True)

    while True:
        now = _now()
        if now >= hard_deadline:
            print(f"[warm-watch] hard ceiling ({max_hours}h) reached", flush=True)
            _stop_pod(pod_id)
            return
        if _runner_active():
            # "Using it" — push the idle deadline out (capped at the ceiling).
            deadline = min(now + hours * 3600, hard_deadline)
            _write_lease(pod_id, deadline, hard_deadline)
        elif now >= deadline:
            _stop_pod(pod_id)
            return
        time.sleep(poll_s)


# ── Laptop side: launch / resume / stop / status ─────────────────────

def _arm(host: str, port: int, sha: str, hours: float, max_hours: float) -> None:
    """Over SSH, check the pod's repo out to *sha* (so this file is present even
    on an older baked image), then start the watcher detached under nohup so it
    survives the SSH session and the laptop sleeping."""
    from helpers import runpod  # lazy: laptop side only

    cmd = (
        f"cd {POD_REPO} && git fetch --quiet origin && git checkout --quiet {sha} "
        f"&& mkdir -p {WARM_DIR} "
        f"&& nohup {UV} run python experiments/helpers/warm_pod.py watch "
        f"--hours {hours} --max-hours {max_hours} "
        f"> {WARM_DIR}/watch.log 2>&1 & echo armed"
    )
    out = runpod.run_on_pod(host, port, cmd, timeout=120)
    print(f"[warm] watcher armed ({out.strip() or 'ok'})")


def launch(gpu: str = "5090", hours: float = LEASE_HOURS,
           max_hours: float = MAX_WARM_HOURS, dry_run: bool = False) -> None:
    """Create a Secure sshd-only warm pod on the shared volume and arm the
    watcher. Spends money — dry_run prints the plan and creates nothing."""
    from helpers import runpod  # lazy: laptop side only

    sha = runpod.head_sha()
    print(f"WARM POD  gpu={gpu}  idle-lease={hours}h  hard-ceiling={max_hours}h")
    print(f"pinned sha : {sha}")
    if dry_run:
        print(f"\n(dry-run) would create 1 secure pod ({gpu} @ {runpod.DATACENTER}, "
              f"volume {runpod.VOLUME_ID}) with sshd only, then over SSH: checkout "
              f"{sha[:12]} and nohup the watcher (stop after {hours}h idle).")
        print("re-run without --dry-run to spend.")
        return

    # The watcher checks the pod's repo out to this sha; it must be on the remote
    # or `git checkout` on the pod fails (same guarantee the fan-out enforces).
    if not runpod.sha_is_pushed(sha):
        raise SystemExit(f"HEAD {sha[:12]} is not pushed to origin — the pod "
                         "fetches from GitHub. Commit + push, then re-run.")

    # RUNPOD_API_KEY lets the watcher stop the pod; no CELLS ⇒ the image runs
    # sshd only (the warm/interactive mode).
    pod_id = runpod.create_pod(
        "warm", gpu, datacenter=runpod.DATACENTER, volume_id=runpod.VOLUME_ID,
        env={"RUNPOD_API_KEY": runpod.api_key()},
    )
    print(f"[warm] created pod {pod_id} — waiting for SSH…")
    host, port = runpod.wait_for_ssh(pod_id)
    _arm(host, port, sha, hours, max_hours)

    print(f"\n=== warm pod {pod_id} up ({gpu}) ===")
    print(f"  ssh -i ~/.ssh/id_ed25519 -p {port} root@{host}")
    print(f"  cd {POD_REPO} && {UV} run python experiments/exp042.py   # your short runs")
    print(f"  it STOPS after {hours}h with no runner; each run resets the clock.")
    print(f"  resume : uv run python experiments/helpers/warm_pod.py resume {pod_id} --hours {hours}")
    print(f"  stop   : uv run python experiments/helpers/warm_pod.py stop {pod_id}")


def resume(pod_id: str, hours: float = LEASE_HOURS,
           max_hours: float = MAX_WARM_HOURS) -> None:
    """Wake a stopped warm pod (`runpodctl start pod`) and re-arm the watcher —
    a stop killed the previous nohup watcher, so resume restarts it."""
    from helpers import runpod  # lazy: laptop side only

    sha = runpod.head_sha()
    if not runpod.sha_is_pushed(sha):
        raise SystemExit(f"HEAD {sha[:12]} is not pushed to origin — commit + push first.")
    print(f"[warm] starting pod {pod_id}…")
    r = subprocess.run(["runpodctl", "start", "pod", pod_id],
                       capture_output=True, text=True)
    print(f"  runpodctl start: {(r.stdout + r.stderr).strip()[:200]}")
    host, port = runpod.wait_for_ssh(pod_id)
    _arm(host, port, sha, hours, max_hours)
    print(f"[warm] pod {pod_id} resumed and re-armed ({hours}h idle lease)")
    print(f"  ssh -i ~/.ssh/id_ed25519 -p {port} root@{host}")


def stop(pod_id: str) -> None:
    """Manually stop a warm pod now (keeps the disk; wake with `resume`)."""
    print(f"[warm] stopping pod {pod_id}…")
    r = subprocess.run(["runpodctl", "stop", "pod", pod_id],
                       capture_output=True, text=True)
    print(f"  runpodctl stop: {(r.stdout + r.stderr).strip()[:200]}")


def status() -> None:
    """List every pod on the account (warm pods included) so you can see what is
    running/stopped and grab an id for resume/stop."""
    from helpers import runpod  # lazy: laptop side only

    pods = runpod.running_pods()
    if not pods:
        print("no pods on the account.")
        return
    for p in pods:
        print(f"  {p.get('id')}  {p.get('name')}  {p.get('desiredStatus') or p.get('status')}")


# ── CLI ──────────────────────────────────────────────────────────────

def _main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Warm RunPod pod with a renewable idle lease.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_launch = sub.add_parser("launch", help="create a warm pod + arm the watcher (laptop)")
    p_launch.add_argument("--gpu", default="5090", choices=("4090", "5090"))
    p_launch.add_argument("--hours", type=float, default=LEASE_HOURS)
    p_launch.add_argument("--max-hours", type=float, default=MAX_WARM_HOURS)
    p_launch.add_argument("--dry-run", action="store_true")

    p_watch = sub.add_parser("watch", help="run the lease watcher (pod side; armed for you)")
    p_watch.add_argument("--hours", type=float, default=LEASE_HOURS)
    p_watch.add_argument("--max-hours", type=float, default=MAX_WARM_HOURS)
    p_watch.add_argument("--poll", type=int, default=POLL_S)

    p_resume = sub.add_parser("resume", help="wake a stopped warm pod + re-arm (laptop)")
    p_resume.add_argument("pod_id")
    p_resume.add_argument("--hours", type=float, default=LEASE_HOURS)
    p_resume.add_argument("--max-hours", type=float, default=MAX_WARM_HOURS)

    p_stop = sub.add_parser("stop", help="stop a warm pod now (laptop)")
    p_stop.add_argument("pod_id")

    sub.add_parser("status", help="list pods on the account (laptop)")

    a = ap.parse_args(argv)
    if a.cmd == "launch":
        launch(gpu=a.gpu, hours=a.hours, max_hours=a.max_hours, dry_run=a.dry_run)
    elif a.cmd == "watch":
        watch(hours=a.hours, poll_s=a.poll, max_hours=a.max_hours)
    elif a.cmd == "resume":
        resume(a.pod_id, hours=a.hours, max_hours=a.max_hours)
    elif a.cmd == "stop":
        stop(a.pod_id)
    elif a.cmd == "status":
        status()


if __name__ == "__main__":
    _main()
