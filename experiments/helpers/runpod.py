"""The one RunPod fan-out system every experiment uses — fire-and-forget over a
network volume.

The dispatch backend for cloud runs (Modal, the earlier SSH-driven backend, was
retired). Where that design babysat every pod from the laptop, this is
laptop-independent: pods are created with their work assignment as env vars,
self-run against a shared RunPod network volume, and self-terminate when done.
The laptop is only needed to *fire* the pods (seconds) and, once they are gone,
to *collect* the volume's contents (minutes). If the laptop sleeps or dies
mid-run, the pods carry on and the artifacts stay safe on the volume.

Two layers, both here:
  • Pod lifecycle (low level): create (pinned to the volume's datacenter), fire
    the fleet, collect a volume subpath via a throwaway pod, teardown that
    VERIFIES a pod is gone (destroy_pod), and a kill switch (reap_all_pods).
    pod_self_terminate() runs ON a pod to remove itself at the end of its work.
  • Fan-out orchestration (high level): `dispatch()` is the single --runpod entry
    every runner calls — it dry-runs / collects / fires from a list of pod
    buckets. `pod_run_loop()` is the single --pod-run entry every runner calls on
    the pod — it iterates the CELLS assignment, skips done jobs, and always
    self-terminates. `chunk_buckets()` packs jobs into pods.

The runner owns only the *what*: it enumerates its job ids (cell names for
training, infer-job ids for analysis), maps a job id to an `is_done` check and a
`run_job` action, and picks where output lands on the volume. Everything about
talking to RunPod lives here, so exp022 (training), exp037 and exp042 (inference)
all share one code path.

The pod's start script (see the repo Dockerfile) self-runs when the CELLS env var
is set: it checks out PIN_SHA, runs experiments/${PINGLAB_POD_RUNNER}.py --pod-run
(the runner `dispatch()` recorded), then self-terminates. With no CELLS it just
runs sshd — the mode collect_subpath() and debugging use.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from .dotenv import load_dotenv
from .paths import REPO

# ── Laptop-side .env autoload ────────────────────────────────────────
# The dispatch side reads its credentials straight from os.environ
# (RUNPOD_CONTAINER_REGISTRY_AUTH_ID for authenticated GHCR pulls, RUNPOD_API_KEY,
# the PINGLAB_* path overrides). Auto-load the repo-root .env at import so the
# human need not `source` it before firing — env still wins over the file
# (override=False). Gated on RUNPOD_POD_ID: a pod gets its env injected by
# dispatch() and has no .env, so --pod-run execution is untouched.
if not os.environ.get("RUNPOD_POD_ID"):
    load_dotenv()

# ── Fleet configuration ──────────────────────────────────────────────
IMAGE = "ghcr.io/eoinmurray/pinglab:cu128"
GPU_IDS = {"4090": "NVIDIA GeForce RTX 4090", "5090": "NVIDIA GeForce RTX 5090"}
GPU_CHOICES = tuple(GPU_IDS)
# $/hr ceilings (measured EU-RO-1 secure 2026-07-05: 4090 $0.69, 5090 $0.99),
# with a little margin so create never rejects on a small price wobble.
COST_CEILING = {"4090": 0.79, "5090": 1.09}
POD_REPO = "/workspace/pinglab"            # where the image cloned the code
UV = "/root/.local/bin/uv"
VOLUME_MOUNT = "/shared"                    # network volume mount point on pods
PUBKEY_PATH = Path.home() / ".ssh" / "id_ed25519.pub"
PRIVKEY_PATH = Path.home() / ".ssh" / "id_ed25519"

# Account-level anchors (created 2026-07-05): the shared network volume and the
# datacenter it lives in. EU-RO-1 carries High 4090 AND 5090 stock, so pods and
# the volume co-locate and provisioning is reliable. Every runner fans out here.
DATACENTER = "EU-RO-1"
VOLUME_ID = "3t2fhu0bzr"

# The volume layout pods read/write (all under VOLUME_MOUNT):
#   training/          — the exp022 weight bank every analysis runner loads
#   artifacts/<slug>/  — per-experiment infer scratch an analysis fan-out writes
TRAINING_SUBDIR = "training"
ARTIFACTS_SUBDIR = "artifacts"

# Timeouts (seconds) — used only for the collect pod, which we DO wait on.
SSH_ATTEMPT_TIMEOUT = 600      # readiness spans 40s→>490s; wait out slow pods
PROVISION_ATTEMPTS = 3         # fresh-pod retries before giving up
PROVISION_BACKOFF_S = 20       # pause between attempts (let stock recover)

# Default pod wall-clock backstops (seconds): the image force-removes a pod after
# MAX_RUNTIME so a hung job can never bill forever. 15 hr real / 40 min smoke.
MAX_RUNTIME_S = 54000
PLUMBING_RUNTIME_S = 2400


# ── Local ↔ volume path contract ─────────────────────────────────────
# Both resolved from env on a pod (set by dispatch) and default to local scratch
# on a laptop, so a runner writes to the same relative place in both worlds.

def training_root() -> Path:
    """The exp022 weight bank: PINGLAB_TRAINING_ROOT on a pod (/shared/training),
    else the local scratch default. cell_dir / load_cell read through this."""
    return Path(os.environ.get(
        "PINGLAB_TRAINING_ROOT",
        str(REPO / "temp" / "experiments" / "exp022"),
    ))


def artifacts_scratch(slug: str) -> Path:
    """A runner's infer scratch: PINGLAB_ARTIFACTS_ROOT on a pod
    (/shared/artifacts/<slug>), else local temp/experiments/<slug>."""
    override = os.environ.get("PINGLAB_ARTIFACTS_ROOT")
    return Path(override) if override else REPO / "temp" / "experiments" / slug


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


# ── Container-registry auth (authenticated GHCR pulls) ───────────────
# A fan-out cold-pulls IMAGE on N pods at once. Anonymous GHCR pulls share a
# tight rate limit, so under load those pulls throttle and pods sit "RUNNING but
# not ready" for 30+ min. Attaching a RunPod "Container Registry Auth" credential
# at pod creation authenticates every pull and restores ~2–4 min cold starts.
# The credential is created ONCE on the RunPod side and its id exported into the
# env — nothing secret ever lives in the repo. See the one-time setup below.
#
#   One-time (RunPod side; needs a GitHub PAT with read:packages):
#     runpodctl registry create --name ghcr \
#         --username eoinmurray --password <PAT>   # → prints the auth id
#     export RUNPOD_CONTAINER_REGISTRY_AUTH_ID=<that id>
#
# Absent ⇒ pods fall back to anonymous pulls (a one-time warning), so nothing
# breaks for anyone without the credential set.

RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"
_ANON_PULL_WARNED = False


def registry_auth_id() -> str | None:
    """The RunPod container-registry auth id for authenticated image pulls, from
    RUNPOD_CONTAINER_REGISTRY_AUTH_ID (mirrors how api_key() reads the env). None
    ⇒ anonymous pulls (throttle-prone under a large fan-out)."""
    return os.environ.get("RUNPOD_CONTAINER_REGISTRY_AUTH_ID") or None


def _warn_anonymous_pull_once() -> None:
    """Warn (once per process) that pods will pull IMAGE unauthenticated — a big
    fan-out may hit the GHCR rate limit and stall on cold start."""
    global _ANON_PULL_WARNED
    if _ANON_PULL_WARNED:
        return
    _ANON_PULL_WARNED = True
    print("  [registry-auth] RUNPOD_CONTAINER_REGISTRY_AUTH_ID unset — pods will "
          f"pull {IMAGE} anonymously; a large fan-out may hit the GHCR rate limit "
          "and stall on cold start (set the auth id — see runpod.py).")


# ── Pod lifecycle (laptop side) ──────────────────────────────────────

def _create_pod_graphql(name: str, gpu: str, *, datacenter: str, volume_id: str,
                        full_env: dict[str, str], cost: float,
                        auth_id: str) -> str:
    """Create one pod via the RunPod GraphQL podFindAndDeployOnDemand mutation,
    attaching *auth_id* so the GHCR image pull is authenticated.

    This mirrors the exact payload the installed `runpodctl create pod` sends
    (secure cloud, price ceiling → deployCost, gpu, DC-pinned network volume,
    30 GB container disk, ssh, env), adding the containerRegistryAuthId field —
    which this runpodctl version's `create pod` cannot pass as a flag. Used only
    when the auth id is set; otherwise create_pod keeps the proven CLI path.
    """
    import urllib.request

    pod_input = {
        "cloudType": "SECURE",                 # --secureCloud
        "name": name,
        "imageName": IMAGE,
        "gpuTypeId": GPU_IDS[gpu],
        "gpuCount": 1,
        "dataCenterId": datacenter,            # must match the network volume's DC
        "networkVolumeId": volume_id,
        "volumeMountPath": VOLUME_MOUNT,
        "volumeInGb": 1,                       # runpodctl --volumeSize default
        "containerDiskInGb": 30,               # --containerDiskSize
        "minMemoryInGb": 20,                   # runpodctl --mem default
        "minVcpuCount": 1,                     # runpodctl --vcpu default
        "deployCost": cost,                    # --cost $/hr price ceiling
        "ports": "22/tcp",
        "startSsh": True,
        "containerRegistryAuthId": auth_id,    # ← the authenticated-pull credential
        "env": [{"key": k, "value": v} for k, v in full_env.items()],
    }
    query = ("mutation createPod($input: PodFindAndDeployOnDemandInput!) {"
             " podFindAndDeployOnDemand(input: $input) { id } }")
    body = json.dumps({"query": query, "variables": {"input": pod_input}}).encode()
    # A User-Agent is required: api.runpod.io sits behind Cloudflare, which
    # 1010-bans urllib's default "Python-urllib/x.y" signature (→ HTTP 403
    # "error code: 1010"). runpodctl sends a normal UA and is unaffected; this
    # raw urllib POST must set one explicitly or every create is Forbidden.
    req = urllib.request.Request(
        f"{RUNPOD_GRAPHQL_URL}?api_key={api_key()}",
        data=body,
        headers={"Content-Type": "application/json",
                 "User-Agent": "pinglab-runpod/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        payload = json.loads(r.read())
    errs = payload.get("errors")
    if errs:
        raise RuntimeError(f"podFindAndDeployOnDemand failed: {errs[0].get('message')}")
    pod = (payload.get("data") or {}).get("podFindAndDeployOnDemand") or {}
    pod_id = pod.get("id")
    if not pod_id:
        raise RuntimeError(f"no pod id in GraphQL response: {payload}")
    return str(pod_id)


def create_pod(name: str, gpu: str, *, datacenter: str, volume_id: str,
               env: dict[str, str], cost: float | None = None) -> str:
    """Create one pod pinned to the volume's datacenter, return its id.

    env is injected as --env KEY=VALUE. When it contains CELLS the image's start
    script self-runs the assignment; without CELLS the pod just runs sshd (the
    collect / debug mode). PUBLIC_KEY is always injected so sshd accepts our key.
    Raises on no-stock / API error.

    When RUNPOD_CONTAINER_REGISTRY_AUTH_ID is set, creation routes through the
    GraphQL path so the image pull is authenticated (see _create_pod_graphql);
    otherwise it uses the proven runpodctl CLI path with an anonymous pull.
    """
    full_env = {"PUBLIC_KEY": PUBKEY_PATH.read_text().strip(), **env}
    cost_val = cost if cost is not None else COST_CEILING[gpu]

    auth_id = registry_auth_id()
    if auth_id:
        return _create_pod_graphql(
            name, gpu, datacenter=datacenter, volume_id=volume_id,
            full_env=full_env, cost=cost_val, auth_id=auth_id,
        )

    _warn_anonymous_pull_once()
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
        "--cost", str(cost_val),
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

def _collect_via_pod(*, remote_dir: str, local_dir: str, datacenter: str,
                     volume_id: str, gpu: str = "4090") -> None:
    """Rsync *remote_dir* on the volume (absolute pod path) → *local_dir*."""
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
        print(f"  [collect] rsync {remote_dir} → {local_dir}")
        sync_dir(host, port, f"{remote_dir.rstrip('/')}/", f"{local_dir}/")
        print("  [collect] done")
    finally:
        destroy_pod(pod_id)


def collect_subpath(*, subpath: str, local_dir: str, datacenter: str,
                    volume_id: str, gpu: str = "4090") -> None:
    """Retrieve an arbitrary subpath under the volume mount (e.g. training or
    artifacts/exp037) to local_dir via a throwaway pod.

    Spins one cheap pod (no CELLS → sshd only) with the volume mounted, rsyncs
    the subpath down, and tears the pod down. This is the only step that needs
    the laptop after firing — run it once the fleet has self-terminated. The
    --collect path in dispatch() is the single caller."""
    remote = f"{VOLUME_MOUNT}/{subpath.strip('/')}"
    _collect_via_pod(
        remote_dir=remote,
        local_dir=local_dir,
        datacenter=datacenter,
        volume_id=volume_id,
        gpu=gpu,
    )


# ── Provenance + auth (shared by every fan-out) ──────────────────────

def head_sha() -> str:
    """The current HEAD sha (pods check this exact commit out)."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO,
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def sha_is_pushed(sha: str) -> bool:
    """True if the sha is on a remote branch (pods fetch it from GitHub)."""
    out = subprocess.run(
        ["git", "branch", "-r", "--contains", sha], cwd=REPO,
        capture_output=True, text=True,
    ).stdout.strip()
    return bool(out)


def api_key() -> str:
    """The RunPod API key, needed on pods for self-termination. From the env or
    runpodctl's config."""
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key
    cfg = Path.home() / ".runpod" / "config.toml"
    if cfg.exists():
        for line in cfg.read_text().splitlines():
            if line.strip().lower().startswith("apikey"):
                return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("no RunPod API key (set RUNPOD_API_KEY or run `runpodctl config`)")


# ── Job → pod bucketing ──────────────────────────────────────────────

def chunk_buckets(job_ids: list[str], per_pod: int, *, prefix: str = "job") -> list[dict]:
    """Pack job ids into pod assignments [{name, cells}, …], `per_pod` each.

    `cells` is the historical key name the CELLS env is built from — a "cell" is
    just a job id (a training cell name, or an infer-job id). A runner that needs
    a smarter split (e.g. exp022's canonical-per-pod) builds its own buckets and
    passes them straight to dispatch()."""
    buckets = []
    for i in range(0, len(job_ids), per_pod):
        buckets.append({
            "name": f"{prefix}-{i // per_pod:02d}",
            "cells": job_ids[i:i + per_pod],
        })
    return buckets


# ── Fan-out orchestration (the one --runpod entry) ───────────────────

def dispatch(
    *,
    slug: str,
    runner: str,
    buckets: list[dict],
    gpu: str = "4090",
    live: bool = False,
    plumbing: bool = False,
    collect: bool = False,
    collect_subdir: str,
    local_collect_dir: str,
    extra_env: dict[str, str] | None = None,
    plumbing_env: dict[str, str] | None = None,
    max_runtime_s: int = MAX_RUNTIME_S,
    plumbing_runtime_s: int = PLUMBING_RUNTIME_S,
) -> None:
    """The single laptop-side --runpod path for every runner.

    Three modes, chosen by the flags:
      • collect=True  → rsync <mount>/<collect_subdir> down to local_collect_dir.
      • live=False    → dry-run: print the fleet plan, create nothing.
      • live=True     → pin+verify the sha, then fire the fleet fire-and-forget.

    `buckets` is [{"name", "cells": [job-id, …]}]; the runner builds them (via
    chunk_buckets or its own logic). `runner` is the experiments/expNNN stem the
    pod re-invokes as `--pod-run` (recorded in PINGLAB_POD_RUNNER). `extra_env` /
    `plumbing_env` inject runner-specific env (e.g. PINGLAB_ARTIFACTS_ROOT)."""
    if collect:
        print(f"collecting {VOLUME_MOUNT}/{collect_subdir} @ {DATACENTER} "
              f"→ {local_collect_dir}")
        collect_subpath(
            subpath=collect_subdir, local_dir=str(local_collect_dir),
            datacenter=DATACENTER, volume_id=VOLUME_ID, gpu=gpu,
        )
        print(f"→ build figures with: uv run python experiments/{runner}.py --skip-training")
        return

    n_jobs = sum(len(b["cells"]) for b in buckets)
    dry = not live
    scale = "plumbing" if plumbing else "standard"
    print(f"{'DRY-RUN' if dry else 'LIVE'}  runner={runner}  gpu={gpu}  scale={scale}")
    print(f"fleet: {len(buckets)} pods · {n_jobs} jobs "
          f"(pods skip jobs already done on the volume)")

    if dry:
        for b in buckets:
            print(f"\n▸ POD {b['name']} [{gpu} @ {DATACENTER}] "
                  f"— CELLS={' '.join(b['cells'])}")
        print("\n(dry-run — nothing created. Re-run with --live to spend.)")
        return

    # ── Provenance: pin an exact, pushed commit + the image digest ──
    sha = head_sha()
    if not sha_is_pushed(sha):
        raise SystemExit(f"HEAD {sha[:12]} is not pushed to origin — pods fetch "
                         "from GitHub. Commit + push, then re-run.")
    digest = resolve_image_digest() or "(digest unresolved)"
    print(f"pinned sha : {sha}")
    print(f"image      : {digest}")

    common = {
        "PIN_SHA": sha,
        "RUNPOD_API_KEY": api_key(),
        "PINGLAB_TRAINING_ROOT": f"{VOLUME_MOUNT}/{TRAINING_SUBDIR}",
        "PINGLAB_POD_RUNNER": runner,
        "MAX_RUNTIME": str(plumbing_runtime_s if plumbing else max_runtime_s),
    }
    if extra_env:
        common.update(extra_env)
    if plumbing and plumbing_env:
        common.update(plumbing_env)

    pods = [{"name": b["name"], "env": {**common, "CELLS": " ".join(b["cells"])}}
            for b in buckets]

    fired = fire(pods, gpu=gpu, datacenter=DATACENTER, volume_id=VOLUME_ID)
    ok = [p for p in fired if p["id"]]
    print(f"\n=== fired {len(ok)}/{len(pods)} pods ({gpu}) ===")
    print("pods self-run + self-terminate; monitor with `runpodctl pod list`.")
    print(f"when the list is empty, collect: "
          f"uv run python experiments/{runner}.py --runpod --collect"
          + (" --plumbing" if plumbing else ""))


def pod_run_loop(*, job_ids: list[str], is_done, run_job, label: str = "pod-run") -> None:
    """The single pod-side --pod-run path for every runner.

    Runs each job named in the CELLS env var, skipping any already done on the
    volume (scale/existence check is the runner's `is_done`), then ALWAYS
    self-terminates (finally) so the pod stops billing even if a job errors.
    `run_job(job_id)` does the one job; `job_ids` is the full valid set (an
    unknown CELLS token is skipped, not run)."""
    names = os.environ.get("CELLS", "").split()
    valid = set(job_ids)
    print(f"[{label}] jobs={names}")
    try:
        for name in names:
            if name not in valid:
                print(f"[{label}] unknown job {name!r} — skipping")
                continue
            if is_done(name):
                print(f"[skip] {name} already done on the volume")
                continue
            try:
                run_job(name)
            except Exception as e:  # noqa: BLE001 — isolate one job's failure
                print(f"[FAIL] {name}: {e}")
    finally:
        pod_self_terminate()
