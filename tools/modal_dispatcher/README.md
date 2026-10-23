# Modal dispatcher

Run an experiment's compute stage on Modal, then check and collect its immutable
Pingstore result. Run these commands from the repository root. Dispatch launches
paid work; use only with authorization for the experiment, resources and inputs.

## Commands

Authenticate once:

```sh
uvx --from 'modal>=1.1,<2' modal setup
```

Dispatch, then use the exact printed compute ID for collection:

```sh
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML \
  python -m tools.modal_dispatcher dispatch --experiment exp141 --gpu H100

uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML \
  python -m tools.modal_dispatcher check --run-id COMPUTE_ID --collect
```

`dispatch --help` lists resource options. Defaults are one A10G, four CPUs,
16384 MiB RAM and a 24-hour timeout. Use `--gpu none` for CPU-only work, or a Modal
GPU count such as `--gpu A100-80GB:2`. The caller selects resources appropriate to
the recipe. The dispatcher sets native
`SNNLAB_DEVICE`, existing legacy device variables and CPU thread environment
variables. Experiments may impose their own resource/thread settings.

Arguments after `--` go directly to the experiment's compute CLI, without a shell.
Scientific configuration stays in the recipe; this tool adds no training knobs.
`--smoke` sets `PINGLAB_SMOKE` for experiments declaring that profile in their
compute entrypoint or recipe; others reject the request before dispatch.
Check the experiment's help for its actual protocol names and supported arguments.

## Source runs

For dependent experiments, declare each direct input as `--input ROLE=RUN_ID`
and pass its run ID through the experiment's own CLI. Roles must exactly match
those recorded by the experiment's `stage_run(inputs=...)` call:

```sh
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML \
  python -m tools.modal_dispatcher dispatch --experiment exp134 \
  --gpu none --cpu 1 --timeout 21600 \
  --input trained_network=exp022-r001-compute \
  --input images=exp123-r068-compute \
  -- --source exp022-r001-compute --image-source exp123-r068-compute
```

Explicit inputs and their complete ancestry are validated, transferred using
checksum-addressed volume paths, and restored before compute. Pending input pins
protect local source runs from pruning. Collection rejects changed local inputs
or a worker result with different roles or checksums. `--root PATH` selects a
local Pingstore for both commands; the worker uses its isolated `.pingstore`.

## Experiment contract and sources

There is no experiment allowlist. The tool discovers
`experiments/expXXX/compute.py`, checks its declared CLI flags without importing
it, and supplies one reserved `--run-id`. The entrypoint must accept that option
and complete that same Pingstore v4 compute run. Arguments cannot override the
reserved identity. Analysis and presentation remain independently invoked stages.
An existing compute entrypoint is not a guarantee of Linux, CUDA, dependency or
dataset compatibility: inspect the recipe and its input requirements first.

The image installs the submitted `uv.lock` and retains Git HEAD as its baseline.
Current `experiments/`, `tools/`, `pyproject.toml` and `uv.lock`, including local
edits and deletions, replace that baseline on the worker. Local datasets,
credentials and Pingstore are not uploaded wholesale; only explicit input runs
are transferred. Other data must be acquired by the experiment itself.

HEAD normally must be fetchable from the public repository. An explicit
`PINGLAB_MODAL_GIT_BUNDLE` containing the exact checkout HEAD can supply local
commits over `PINGLAB_MODAL_BASE_COMMIT`, a public full commit hash. Optional
`PINGLAB_MODAL_IMAGE_ID` reuses a chosen built image; the worker still checks the
installed SNNLab version against the submitted lock. Do not edit mounted sources
during upload.

## Durability and collection

Dispatch invokes `modal run --detach` and `Function.spawn()`. The worker continues
independently after acknowledged submission, following Modal's
[asynchronous invocation durability](https://modal.com/docs/guide/function-invocation-methods#durability).
The local process saves the receipt before spawning, then saves the acknowledged
call ID. The receipt retains the exact command, environment, resources and input
pins. Existing receipts and completed runs remain compatible with `check` after
the move from experiment helpers.

1. `pending`: the worker has not returned; checks use `get(timeout=0)`.
2. `completed`: remote results are ready; `--collect` downloads and validates them.
3. `collected`: the local completed run validates; repeat checks need no network.
4. `submission_unknown`: no call ID was acknowledged and no completion marker
   exists. Investigate the saved app/dispatch rather than submitting again.

Checks never load the training app, build images or restart work. Interrupted
collection can be retried under the same reservation; concurrent collectors are
rejected. The worker commits validated results and a completion marker to the
existing `pinglab-compute-results` volume before returning. Collection prefers
that durable marker, including after call-result retention expires. Execution
metadata is retained under `execution.modal` in the completed `run.json`.
Remote errors propagate. There is no automatic retry or training resumption.

The Mac needs connectivity for dispatch and collection, not throughout training.
For progress use `modal app logs APP_ID` with the saved app ID. Missing a check
while the Mac sleeps has no effect on remote execution.

## Implementation and verification

1. `__main__.py`: common dispatch/check CLI.
2. `request.py`: experiment validation and frozen execution request.
3. `app.py`: Modal image, detached submission and remote worker.
4. `jobs.py`: durable reservations, receipts, status and collection.
5. `inputs.py`: validated source-run transfer and restoration.

Run `uv run python -m pytest tools/modal_dispatcher/tests -q` for local tests.
These tests use mocks and a tiny local subprocess fixture, with no Modal calls,
scientific execution, data downloads or paid resources.
