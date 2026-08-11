# exp022 on Wilkes3

The scientific registry and all training parameters remain in
`experiments/exp022.py`. These scripts only map its resource tiers onto Slurm.

## One-time setup

Build the locked environment on an Ampere compute node and populate the MNIST
cache before submitting arrays. Confirm the GPU project name with `mybalance`;
the Slurm account is normally a project ending in `-GPU`, not the service-level
label `SL2`.

## Canary

Submit representative cells directly before choosing final wall times:

```bash
sbatch --account=<PROJECT-GPU> --partition=ampere --nodes=1 --gres=gpu:1 \
  --time=04:00:00 --wrap="cd $PWD && module purge && module load rhel8/default-amp && \
  uv run python experiments/exp022.py --train-cell ping__variable_rate__seed42"
```

Recommended canaries are `ping__off__seed42`, `ping__dt0p05__seed42`,
`ping__canonical__seed42`, and `ping__variable_rate__seed42`.

## Arrays

After adjusting the provisional wall times in `submit-tier.sh` from the canary
measurements, submit each tier:

```bash
export EXP022_SLURM_ACCOUNT=<PROJECT-GPU>
bash hpc/exp022/submit-tier.sh standard
bash hpc/exp022/submit-tier.sh fine_dt
bash hpc/exp022/submit-tier.sh canonical_coba
bash hpc/exp022/submit-tier.sh canonical_ping
bash hpc/exp022/submit-tier.sh variable_rate
```

Each array task trains exactly one cell into
`temp/experiments/exp022/<cell-name>/`. Once all cells validate, aggregate the
bank and render its figures once:

```bash
uv run python experiments/exp022.py --skip-training
```
