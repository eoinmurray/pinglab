# Modal compute

One shared wrapper calls the existing compute entry points for exp047, exp080,
exp081 and exp124. From the repository root, authenticate once:

```sh
uvx --from 'modal>=1.1,<2' modal setup
```

Then run any one experiment:

```sh
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run experiments/helpers/modal_compute.py --experiment exp047
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run experiments/helpers/modal_compute.py --experiment exp081
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run experiments/helpers/modal_compute.py --experiment exp080
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run experiments/helpers/modal_compute.py --experiment exp124 --smoke
```

Add `--smoke` for the existing reduced compute profile. The default GPU is A10G;
use `--gpu A100` to override it. The function requests four CPUs, 16 GiB RAM and
a 24-hour timeout. These commands start paid Modal compute.

For exp124 CPU benchmarks, disable the GPU and match the requested physical
cores to PyTorch's thread count:

```sh
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run experiments/helpers/modal_compute.py --experiment exp124 --smoke --gpu none --cpu 4
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run experiments/helpers/modal_compute.py --experiment exp124 --smoke --gpu none --cpu 8
```

Both runs use the same scientific recipe. The requested cores, actual PyTorch
threads, host and CPU model (when exposed by the host) are recorded in execution metadata. Allocating more
cores does not guarantee a speedup.

The wrapper reserves a fresh local run identity before dispatch. It builds from
the committed `uv.lock`, retains the current Git HEAD as the source baseline and
uploads the current `experiments/`, `tools/`, article metadata and dependency
files, including uncommitted changes. HEAD must be available from the public
Pinglab GitHub remote; there is no need to commit local code edits first. Local
datasets, credentials and Pingstore are not uploaded. exp080 downloads MNIST
inside its container.

exp124 defaults to its class-balanced SHD smoke: 200
training and 100 validation examples, five epochs for both models, using the new
`paper` profile: two 128-neuron COBA layers and two 128 E/32 I PING layers. It downloads and authenticates only the official SHD training
archive inside the container. For the larger two-layer run, omit `--smoke` and
pass explicit counts and epochs:

```sh
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run --detach experiments/helpers/modal_compute.py --experiment exp124 --gpu none --cpu 8 --train-examples 1000 --validation-examples 200 --epochs 20
```

Count/epoch overrides are supported only for exp124 and cannot be combined with
`--smoke`. Example counts must be multiples of 20.

A successful compute run is retained on the `pinglab-compute-results` Modal
Volume, then downloaded and validated before becoming visible in local
`.pingstore/runs/`. The printed Volume path identifies the retained copy if the
local download is interrupted. Failed dispatches leave their local reservation
hidden; rerunning the command allocates a fresh identity and does not retry the
old compute. A disconnect can leave remote work running; check the Modal app
before starting another expensive run. Remote incomplete computation is not a
completed scientific result and is not recovered by this wrapper.

For the prepared full-data SHD profile, use the explicit full-dataset option:

```sh
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run --detach experiments/helpers/modal_compute.py --experiment exp124 --gpu none --cpu 8 --variant paper --full-dataset --validation-fraction 0.1 --epochs 20
```

This consumes every official training utterance: 7,340 for training and 816 for
validation, using a seeded stratified 10% holdout. It cannot be combined with
`--smoke` or explicit example counts. The wrapper forwards the profile and split
settings unchanged to exp124 compute. This full-data command was submitted as `exp124-r066-compute` and cancelled by the author during the first COBA epoch; PING did not start. No completed result is available. The [exp124 README](../exp124/README.md) records the scientific
settings and adaptations from the original paper.

Analyse and present remain explicit local commands using the returned compute
run ID. The wrapper does not launch them or publish.
