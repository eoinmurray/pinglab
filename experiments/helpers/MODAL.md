# Modal compute

One shared wrapper calls the existing compute entry points for exp047, exp080,
exp081, exp124 and exp126. From the repository root, authenticate once:

```sh
uvx --from 'modal>=1.1,<2' modal setup
```

Then run any one experiment:

```sh
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run experiments/helpers/modal_compute.py --experiment exp047
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run experiments/helpers/modal_compute.py --experiment exp081
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run experiments/helpers/modal_compute.py --experiment exp080
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run experiments/helpers/modal_compute.py --experiment exp124 --smoke
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run experiments/helpers/modal_compute.py --experiment exp126 --gpu A10 --cpu 4
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
inside its container. exp126 downloads MNIST in its container too, and runs its
fixed 6,000-training / 1,000-validation / 10,000-test recipe. Its `--smoke` flag
does not reduce this recipe. It measures optimization and evaluation separately,
then performs five warmed full-test inference timing passes. See
[exp126 instructions](../exp126/README.md) for timing boundaries and separate
analyse/present commands.

exp124 defaults to its class-balanced SHD smoke: 200
training and 100 validation examples, two hidden layers of 128 E/32 I neurons, five epochs for both
COBA and PING. It downloads and authenticates only the official SHD training
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

Analyse and present remain explicit local commands using the returned compute
run ID. The wrapper does not launch them or publish.

For the equation-matched exp126 snnTorch/snnlab comparison, use the experiment's
parallel launcher rather than the ordinary single-condition wrapper:

```sh
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML modal run experiments/exp126/modal_compare.py::compare --implementation both --seeds 42 --gpu A10 --cpu 4
```

It spawns both calls before waiting, with one GPU per container and at most two
active containers. Both implementations also support independent dispatch via
`--implementation snntorch` or `--implementation snnlab`. See
[exp126 comparison instructions](../exp126/README.md) for optional paired seeds,
explicit phase assembly into one full compute result before analysis, agreement
gates and timing boundaries. This comparison
has been prepared but not executed.
