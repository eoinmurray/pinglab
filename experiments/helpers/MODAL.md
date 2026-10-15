# Durable Modal compute

`modal_compute.py` submits one immutable compute reservation using Modal
`Function.spawn()`, saves the call ID, and returns. Use `modal run --detach`;
the entrypoint refuses a non-detached invocation. Once dispatch is acknowledged,
the remote job continues independently of the Mac, terminal or local process.
The Mac must be online for submission and later checks/downloads, not during
training. This follows Modal's documented [asynchronous invocation durability](https://modal.com/docs/guide/function-invocation-methods#durability).

## Dispatch

Use the experiment's authorized resource and scientific settings, for example:

```sh
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML \
  modal run --detach experiments/helpers/modal_compute.py \
  --experiment exp124 --models ping --variant calibrated --full-dataset \
  --epochs 50 --batch-size 256 --gpu H100 --cpu 4
```

This is paid training, not a status command. Do not dispatch again just because a
terminal closed or a check timed out. The CLI prints the reserved run ID and call
ID. The pending run's `.scratch/modal-dispatch.json` retains app/call IDs, volume
location and resource metadata. A receipt is written before spawning; if the
acknowledgement is lost, its null call ID means submission outcome is unknown.
The checker reports that state rather than automatically submitting a duplicate.

An optional `PINGLAB_MODAL_IMAGE_ID` reuses an explicitly selected built image.
The worker verifies installed SNNLab against the submitted dependency lock.
Sources are uploaded anew; do not edit mounted sources during upload.

## Periodic checks and collection

Use the exact run ID printed by dispatch:

```sh
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML \
  python experiments/helpers/modal_jobs.py --run-id <compute-run-id>

uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML \
  python experiments/helpers/modal_jobs.py --run-id <compute-run-id> --collect
```

1. `pending` means the call has not returned. Checks use `get(timeout=0)` and
   exit without waiting, building an image or starting a worker.
2. `completed` means results are available; `--collect` downloads them under the
   existing reservation and reports `collected` after validation and atomic
   completion. A partially interrupted download can be collected again without
   new training. A completed local run is validated and returned without network
   access. Concurrent collectors for one run are rejected.
3. `submission_unknown` means no call ID was acknowledged and no remote
   completion marker exists yet. Investigate the recorded app and dispatch
   rather than resubmitting automatically. Remote exceptions propagate as errors;
   checks never restart failed or cancelled jobs.

A monitor can invoke the `--collect` form every five minutes. Missing checks
while the Mac sleeps have no effect on remote execution; the next check can
recover the finished result. For epoch progress, fetch the recorded app's logs
with `modal app logs <app-id>`; no continuously connected log process is needed.
The checker only collects compute: analyse and present remain separate,
explicit commands.

The worker validates the completed v4 run, writes it to the durable result
volume, adds a completion marker outside the run, and commits the volume before
returning. The collector prefers that marker, so collection still works after
Modal's seven-day call-result retention window. It validates the downloaded
identity, origin, scope, layout and payload digest before exposing the run.
Submission metadata is retained in completed `run.json` under `execution.modal`;
no dispatch receipt is added to scientific exports or the completed run root.

This change addresses caller-disconnection cancellation. It does not add
intermediate training checkpoints: current exp124 still saves a native final
checkpoint only. Worker failures or an explicit cancellation can still lose
unfinished training. Historical synchronous reservations cannot be retroactively
converted into durable calls.

## Verification

Run `python -m pytest experiments/helpers/test_modal_jobs.py -q` for isolated
submission/collection tests. A separate 30-second CPU-only Modal probe also
verified that the submitting process exits while the remote call is pending,
and a fresh process can obtain its result afterward. No SHD training or GPU
was used for that live durability check.

## Supported experiments and source inputs

The shared worker calls existing compute entrypoints for exp047, exp080, exp081
and exp124. Authenticate once with the provider's own flow:

```sh
uvx --from 'modal>=1.1,<2' modal setup
```

The default GPU is A10G; `--gpu A100` or `--gpu H100` overrides it. The worker
requests four CPUs, 16 GiB RAM and a 24-hour timeout by default. `--smoke` selects
an experiment's existing reduced profile. All dispatches start paid compute:

```sh
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run --detach experiments/helpers/modal_compute.py --experiment exp047
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run --detach experiments/helpers/modal_compute.py --experiment exp081
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run --detach experiments/helpers/modal_compute.py --experiment exp080
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run --detach experiments/helpers/modal_compute.py --experiment exp124 --smoke
```

The image uses `uv.lock`, retains the current Git HEAD as a baseline, and uploads
current experiments/tools/article metadata/dependency files including uncommitted
edits. HEAD must be available from the public Pinglab GitHub remote; committing
local code edits is not required. Local datasets, credentials and Pingstore are
not uploaded. The exp080 worker downloads MNIST inside its container.

The exp124 reduced defaults are 200 training / 100 validation utterances and
five epochs for both models. The current default is the calibrated SHD recipe;
`--variant paper` remains an explicit alternative. COBA has two 128-neuron
layers; PING has two 128 E/32 I layers. Only the official training archive is
downloaded and authenticated. Count/epoch overrides are supported only for
exp124, cannot accompany `--smoke`, and counts must be multiples of 20.

For CPU benchmarking, `--gpu none` disables the GPU and `--cpu` matches requested
cores to PyTorch's thread count. Additional cores do not guarantee a speedup;
requested cores, actual threads, host and exposed CPU model are recorded:

```sh
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run --detach experiments/helpers/modal_compute.py --experiment exp124 --smoke --gpu none --cpu 4
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run --detach experiments/helpers/modal_compute.py --experiment exp124 --smoke --gpu none --cpu 8
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run --detach experiments/helpers/modal_compute.py --experiment exp124 --gpu none --cpu 8 --train-examples 1000 --validation-examples 200 --epochs 20
```

`--full-dataset --validation-fraction 0.1` retains all official training
utterances: 7,340 train / 816 validation in the seeded per-class holdout. It
cannot accompany `--smoke` or explicit counts. The historical CPU command with
`--variant paper --gpu none --cpu 8 --full-dataset --epochs 20` produced the
cancelled reservation exp124-r066-compute: the author stopped it during the
first COBA epoch, before PING started; it is not a completed result. See the
[exp124 README](../exp124/README.md) for later native training, source IDs,
measured results and scientific adaptations. Analyse/present remain explicit
local stages; neither helper publishes results.
