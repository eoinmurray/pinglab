# exp112 — COBA/PING voltage-gradient damping

This experiment is a paired 2×2 comparison of COBA and PING training with
voltage-gradient damping divisors 1 and 1000. Each condition trains for 50
epochs on the same deterministic 2% subset of the official MNIST training
partition and is evaluated on the complete official MNIST test partition.

The recipe is self-contained. Its fixed numerical values were transcribed from
the exp022 unpenalised activity-frontier condition at commit `be48653b`, but no
exp112 module imports exp022, reads an exp022 run, or follows future exp022
changes. The two experimental factors are recurrent E/I-loop engagement and
voltage-gradient damping; every other committed setting is shared.

## Conditions

| Array index | Condition | E/I loop | Damping divisor |
| ---: | --- | --- | ---: |
| 0 | `coba-d1` | disabled | 1 |
| 1 | `coba-d1000` | disabled | 1000 |
| 2 | `ping-d1` | enabled | 1 |
| 3 | `ping-d1000` | enabled | 1000 |

COBA uses the same PING-shaped simulator model with E/I coupling strength zero.
This preserves identical tensor shapes and random-number consumption while
removing the active E→I→E loop.

## Paired data contract

All conditions use seed 42. The simulator deterministically samples 1,200
images without replacement from the 60,000-image official MNIST training
partition, then makes the same stratified 1,080/120 optimizer/validation split.
Model shape, initialization order, minibatch order and Poisson encoding streams
are shared. Each compute run records digests of the source arrays and the exact
selected, optimizer and validation index sequences. Analysis refuses runs whose
dataset identities differ. The epoch-50 checkpoint from every condition is
evaluated on all 10,000 official test images; validation-selected metrics remain
available separately in the training history. Each final checkpoint also records
E/I spikes for digit 0, sample 0 within that class, from the official test
partition. The data identity includes its raw test index and image digest, so all
four illustrative rasters must use the same image and deterministic encoding.

## Local stage commands

Each compute invocation produces one independent v4 run:

```sh
uv run python experiments/exp112/compute.py --shard-index 0
uv run python experiments/exp112/compute.py --shard-index 1
uv run python experiments/exp112/compute.py --shard-index 2
uv run python experiments/exp112/compute.py --shard-index 3
```

After all four complete, pass their identities explicitly:

```sh
uv run python experiments/exp112/analyse.py \
  --source <coba-d1-run> --source <coba-d1000-run> \
  --source <ping-d1-run> --source <ping-d1000-run>
uv run python experiments/exp112/present.py --source <analysis-run>
```

Stages never select a latest run or launch another stage.

## Wilkes3 execution

Use a clean committed checkout, a frozen environment and a prepopulated MNIST
cache. The standard HPC adapter first freezes the four complete cell
configurations, one-cell-per-task allocation, source identity and scheduler
resources in a reviewable plan. Review submits one four-element Slurm array;
it does not submit four separate jobs. Each array task receives one GPU and
executes one condition concurrently, subject to scheduler availability. A
receipt is written before Slurm is contacted, preventing blind resubmission
after an ambiguous response.

```sh
uv run python -m experiments.exp112.hpc prepare \
  --plan .scratch/exp112-hpc/production.json \
  --account <project> \
  --mnist-cache <persistent-directory-containing-MNIST> \
  --walltime 01:00:00
uv run python -m experiments.exp112.hpc review \
  .scratch/exp112-hpc/production.json
uv run python -m experiments.exp112.hpc review \
  .scratch/exp112-hpc/production.json --test-only
uv run python -m experiments.exp112.hpc review \
  .scratch/exp112-hpc/production.json --live
```

Separate submissions must remain at least 120 seconds apart. Queue time is not
part of the estimated 14–20 minute training time per condition.

Presentation produces training and test-accuracy figures plus a four-panel
`final-epoch-rasters.png`. Each raster shows all 1,024 E neurons followed by all
256 I neurons over the 200 ms presentation. It is an illustrative paired trial,
not an estimate of typical activity across inputs.

No compute, analysis or presentation has yet been run for this experiment.
