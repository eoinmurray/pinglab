"""Generic RunPod fan-out backend — fire-and-forget over a network volume.

The counterpart to helpers/modal.py. Where the SSH-drive design babysat every
pod from the laptop, this is laptop-independent: pods are created with their work
assignment as env vars, self-run against a shared RunPod network volume, and
self-terminate when done. The laptop is only needed to *fire* the pods (seconds)
and, once they are gone, to *collect* the volume's contents (minutes). If the
laptop sleeps or dies mid-run, the pods carry on and the artifacts stay safe on
the volume.

Division of labour:
  • This module owns pod lifecycle: create (pinned to the volume's datacenter),
    fire the fleet, collect artifacts back via a throwaway pod, and — above all —
    teardown that VERIFIES a pod is gone (destroy_pod) plus a standalone kill
    switch (reap_all_pods). pod_self_terminate() runs ON a pod so it can remove
    itself at the end of its work.
  • The runner (exp022) owns the *what*: which cells, the pinned git sha, and the
    pod-side --pod-run entrypoint the image starts.

The pod's start script (see the repo Dockerfile) self-runs when the CELLS env var
is set: it checks out PIN_SHA, trains each cell in CELLS to the volume (writing
under PINGLAB_TRAINING_ROOT = <mount>/training), then self-terminates. With no
CELLS it just runs sshd — that mode is what collect() and any debugging use.

Nothing here knows about training; it is reusable by any runner that can express
its work as pods carrying a CELLS assignment.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

# ── Fleet configuration ──────────────────────────────────────────────
IMAGE = "ghcr.io/eoinmurray/pinglab:cu128"
GPU_IDS = {"4090": "NVIDIA GeForce RTX 4090", "5090": "NVIDIA GeForce RTX 5090"}
# $/hr ceilings (measured EU-RO-1 secure 2026-07-05: 4090 $0.69, 5090 $0.99),
# with a little margin so create never rejects on a small price wobble.
COST_CEILING = {"4090": 0.79, "5090": 1.09}
POD_REPO = "/workspace/pinglab"            # where the image cloned the code
UV = "/root/.local/bin/uv"
VOLUME_MOUNT = "/shared"                    # network volume mount point on pods
PUBKEY_PATH = Path.home() / ".ssh" / "id_ed25519.pub"
PRIVKEY_PATH = Path.home() / ".ssh" / "id_ed25519"

# Timeouts (seconds) — used only for the collect pod, which we DO wait on.
SSH_ATTEMPT_TIMEOUT = 600      # readiness spans 40s→>490s; wait out slow pods
PROVISION_ATTEMPTS = 3         # fresh-pod retries before giving up
PROVISION_BACKOFF_S = 20       # pause between attempts (let stock recover)


def _sh(cmd: list[str], timeout: float | None = None, check: bool = True) -> str:
    """Run a local command, return stdout. Raises on non-zero when check=True."""
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"$ {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return p.stdout


# ── Provenance ───────────────────────────────────────────────────────

def resolve_image_digest(image: str = IMAGE) -> str | None:
    """Resolve a :tag image to its immutable @sha256 digest (for the manifest).

    A tag is mutable — recording it is not reproducible; the digest is. Queries
    GHCR anonymously (the package is public). Returns 'ghcr.io/…@sha256:…' or
    None if the lookup fails (provenance is best-effort, never blocks a run).
    """
    import urllib.request

    repo, tag = image.rsplit(":", 1)
    name = repo.split("/", 1)[1]  # eoinmurray/pinglab
    try:
        tok_url = f"https://ghcr.io/token?scope=repository:{name}:pull"
        with urllib.request.urlopen(tok_url, timeout=15) as r:
            token = json.loads(r.read())["token"]
        req = urllib.request.Request(
            f"https://ghcr.io/v2/{name}/manifests/{tag}",
            headers={"Authorization": f"Bearer {token}",
                     "Accept": "application/vnd.oci.image.index.v1+json,"
                               "application/vnd.docker.distribution.manifest.v2+json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            digest = r.headers.get("Docker-Content-Digest")
        return f"{repo}@{digest}" if digest else None
    except Exception:  # noqa: BLE001 — provenance is best-effort
        return None


# ── Pod-side: self-terminate ─────────────────────────────────────────

def pod_self_terminate() -> None:
    """Terminate the pod this is running on. Called by exp022 --pod-run at the
    end of its work (and by the image's shell backstop on a hang). Uses the
    injected RUNPOD_POD_ID; runpodctl authenticates via the RUNPOD_API_KEY env.
    """
    import os

    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id:
        print("[self-terminate] no RUNPOD_POD_ID — skipping (not on a pod?)")
        return
    print(f"[self-terminate] removing pod {pod_id}")
    _sh(["runpodctl", "remove", "pod", pod_id], check=False)


# ── Pod lifecycle (laptop side) ──────────────────────────────────────

def create_pod(name: str, gpu: str, *, datacenter: str, volume_id: str,
               env: dict[str, str], cost: float | None = None) -> str:
    """Create one pod pinned to the volume's datacenter, return its id.

    env is injected as --env KEY=VALUE. When it contains CELLS the image's start
    script self-runs the assignment; without CELLS the pod just runs sshd (the
    collect / debug mode). PUBLIC_KEY is always injected so sshd accepts our key.
    Raises on no-stock / API error.
    """
    full_env = {"PUBLIC_KEY": PUBKEY_PATH.read_text().strip(), **env}
    cmd = [
        "runpodctl", "create", "pod",
        "--name", name,
        "--imageName", IMAGE,
        "--gpuType", GPU_IDS[gpu],
        "--gpuCount", "1",
        "--secureCloud",                       # datacenter pods are secure cloud
        "--dataCenterId", datacenter,          # must match the network volume's DC
        "--networkVolumeId", volume_id,
        "--volumePath", VOLUME_MOUNT,
        "--cost", str(cost if cost is not None else COST_CEILING[gpu]),
        "--containerDiskSize", "30",
        "--ports", "22/tcp",
        "--startSSH",
    ]
    for k, v in full_env.items():
        cmd += ["--env", f"{k}={v}"]
    return _parse_pod_id(_sh(cmd))


def _parse_pod_id(create_stdout: str) -> str:
    """Pull the pod id out of `runpodctl create pod` output (text or json)."""
    import re

    s = create_stdout.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and obj.get("id"):
            return str(obj["id"])
    except json.JSONDecodeError:
        pass
    m = re.search(r'pod\s+"([^"]+)"', s)  # 'pod "<id>" created for $X.XX / hr'
    if m:
        return m.group(1)
    raise RuntimeError(f"could not parse pod id from: {s!r}")


def destroy_pod(pod_id: str | None) -> None:
    """Terminate a pod and VERIFY it is gone — a lingering GPU pod burns money.

    `runpodctl pod delete` wants interactive confirmation and no-ops in a
    subprocess, so we use the (deprecated but non-interactive) `remove pod`,
    then poll `pod list` to confirm the id disappeared. If it will not die we
    shout — a false "[gone]" is worse than noise.
    """
    if not pod_id:
        return
    for attempt in range(4):
        _sh(["runpodctl", "remove", "pod", pod_id], check=False)
        time.sleep(3)
        listing = _sh(["runpodctl", "pod", "list", "-o", "json"], check=False)
        if pod_id not in listing:
            print(f"  [gone] {pod_id}")
            return
        print(f"  [retry teardown {attempt + 1}/4] {pod_id} still alive")
    print(f"  !!! FAILED TO TERMINATE {pod_id} — KILL MANUALLY: "
          f"runpodctl remove pod {pod_id}")


def reap_all_pods() -> None:
    """Kill switch — terminate EVERY pod on the account and confirm none remain.

    The one guarantee independent of any run state: call this any time (even
    after a crash) to be certain nothing is billing. Idempotent. Note: pods that
    have already self-terminated are gone; this catches any that hung or leaked.
    """
    listing = _sh(["runpodctl", "pod", "list", "-o", "json"], check=False)
    try:
        pods = json.loads(listing) or []
    except json.JSONDecodeError:
        pods = []
    if not pods:
        print("no pods on the account — nothing to reap.")
        return
    print(f"reaping {len(pods)} pod(s)...")
    for p in pods:
        destroy_pod(str(p.get("id")))
    remaining = _sh(["runpodctl", "pod", "list", "-o", "json"], check=False)
    if remaining.strip() in ("[]", ""):
        print("✓ all pods terminated — account is clean, nothing billing.")
    else:
        print(f"!!! pods still present after reap:\n{remaining}")


# ── Fire the fleet (no babysitting) ──────────────────────────────────

def fire(pods: list[dict], *, gpu: str, datacenter: str,
         volume_id: str) -> list[dict]:
    """Create every pod (retrying create on transient no-stock), then return.

    pods: [{"name": str, "env": {"CELLS": "a b c", ...}}, ...]. Does NOT wait —
    each pod self-runs and self-terminates. A pod that can't be created after
    PROVISION_ATTEMPTS is recorded with id=None so the caller can report it.
    Returns [{"name", "id"}, ...].
    """
    created = []
    for spec in pods:
        pod_id = None
        for attempt in range(1, PROVISION_ATTEMPTS + 1):
            try:
                pod_id = create_pod(spec["name"], gpu, datacenter=datacenter,
                                    volume_id=volume_id, env=spec["env"])
                print(f"  [fired] {spec['name']} → {pod_id} ({gpu})")
                break
            except Exception as e:  # noqa: BLE001 — retry transient create failures
                print(f"  [create {attempt}/{PROVISION_ATTEMPTS}] {spec['name']} "
                      f"failed: {str(e).splitlines()[-1][:120]}")
                time.sleep(PROVISION_BACKOFF_S)
        if pod_id is None:
            print(f"  [could not fire] {spec['name']}")
        created.append({"name": spec["name"], "id": pod_id})
    return created


def running_pods() -> list[dict]:
    """Current pods on the account (name/id/status) — for monitoring a fan-out."""
    listing = _sh(["runpodctl", "pod", "list", "-o", "json"], check=False)
    try:
        return json.loads(listing) or []
    except json.JSONDecodeError:
        return []


# ── SSH (used only by collect) ───────────────────────────────────────

def _parse_ssh_info(ssh_info_json: str) -> tuple[str, int] | None:
    try:
        obj = json.loads(ssh_info_json)
    except json.JSONDecodeError:
        return None
    if obj.get("error") or not obj.get("ip") or not obj.get("port"):
        return None
    return str(obj["ip"]), int(obj["port"])


def wait_for_ssh(pod_id: str, timeout: float = SSH_ATTEMPT_TIMEOUT) -> tuple[str, int]:
    """Poll `runpodctl ssh info <id>` until the SSH endpoint (ip, port) is up."""
    deadline = time.monotonic() + timeout
    polls = 0
    while time.monotonic() < deadline:
        out = _sh(["runpodctl", "ssh", "info", pod_id], check=False)
        hp = _parse_ssh_info(out)
        if hp:
            return hp
        polls += 1
        if polls % 5 == 1:
            print(f"  [ssh-wait] {pod_id} poll {polls}: {out.strip()[:120]}")
        time.sleep(8)
    raise TimeoutError(f"pod {pod_id} SSH not ready within {timeout:.0f}s")


def _ssh_base(host: str, port: int) -> list[str]:
    return [
        "ssh", "-i", str(PRIVKEY_PATH), "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=15",
        f"root@{host}",
    ]


def run_on_pod(host: str, port: int, script: str, timeout: float) -> str:
    """Run a script on the pod over SSH (script passed as one argv element, so a
    multi-word `cd X && …` stays intact rather than leaking past a shell -c)."""
    return _sh(_ssh_base(host, port) + [script], timeout=timeout)


def sync_dir(host: str, port: int, remote_dir: str, local_dir: str) -> None:
    """rsync a directory from the pod back to the local machine."""
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    ssh_e = (f"ssh -i {PRIVKEY_PATH} -p {port} "
             f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null")
    _sh(["rsync", "-az", "-e", ssh_e, f"root@{host}:{remote_dir}", local_dir],
        timeout=1800)


# ── Collect artifacts off the volume ─────────────────────────────────

def collect(*, datacenter: str, volume_id: str, local_root: str,
            gpu: str = "4090") -> None:
    """Retrieve <mount>/training from the volume to local_root via a throwaway pod.

    Spins one cheap pod (no CELLS → sshd only) with the volume mounted, rsyncs
    the training root down, and tears the pod down. This is the only step that
    needs the laptop after firing — run it once the fleet has self-terminated.
    """
    pod_id = None
    try:
        for attempt in range(1, PROVISION_ATTEMPTS + 1):
            try:
                pod_id = create_pod("collector", gpu, datacenter=datacenter,
                                    volume_id=volume_id, env={})
                host, port = wait_for_ssh(pod_id)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  [collector {attempt}/{PROVISION_ATTEMPTS}] failed: "
                      f"{str(e).splitlines()[-1][:120]}")
                destroy_pod(pod_id)
                pod_id = None
                time.sleep(PROVISION_BACKOFF_S)
        else:
            raise RuntimeError("could not provision a collector pod")
        print(f"  [collect] rsync {VOLUME_MOUNT}/training → {local_root}")
        sync_dir(host, port, f"{VOLUME_MOUNT}/training/", f"{local_root}/")
        print("  [collect] done")
    finally:
        destroy_pod(pod_id)
