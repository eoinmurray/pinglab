"""The preserved production recipe; imports never execute an experiment."""

SLUG = "exp048"
CHECKPOINT_ROLE = "best_validation"
CHECKPOINT_POLICY = {"purpose": "deployment_performance", "role": CHECKPOINT_ROLE}

SEEDS: list[int] = [42, 43, 44]

N_E: int = 1024

N_I: int = 256

N_IN: int = 784

N_CLASSES: int = 10

DT: float = 0.1  # ms

TRAINED_T_MS: float = 200.0  # trained trial duration

INPUT_RATE_HZ: float = 25.0  # canonical Poisson input rate

TAU_HEADLINE_MS: float = 50.0  # digit duration in headline figure

N_DIGITS_HEADLINE: int = 5  # number of digits in the headline stream

N_STREAMS: int = 20

N_PER_STREAM: int = 10

N_GRID_STREAMS: int = 40

TAU_SWEEP_MS: list[float] = [25.0, 50.0, 100.0, 200.0]

TAU_GRID_MS: list[float] = [10.0, 15.0, 25.0, 40.0, 50.0, 75.0, 100.0, 200.0]

RATE_GRID_HZ: list[float] = [5.0, 10.0, 25.0, 50.0, 100.0, 200.0]

LOW_RATE_HZ: list[float] = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0]

LOW_RATE_STREAMS: int = 10

LOW_RATE_DIGITS_PER_STREAM: int = 10

VARYING_HEADLINE: list[tuple[float, float]] = [
    (200.0, 10.0),  # long, weak
    (50.0, 100.0),  # short, strong
    (100.0, 25.0),  # medium, canonical
    (25.0, 200.0),  # very short, very strong
    (75.0, 15.0),  # intermediate, weak-ish
]

RASTER_N_E_PLOT: int = 200

RASTER_N_I_PLOT: int = 64

SEED: int = 42


def configuration():
    return {
        "schema": "exp048.recipe/v1",
        "checkpoint_policy": CHECKPOINT_POLICY,
        "n_e": N_E,
        "n_i": N_I,
        "n_in": N_IN,
        "n_classes": N_CLASSES,
        "dt": DT,
        "trained_t_ms": TRAINED_T_MS,
        "tau_headline_ms": TAU_HEADLINE_MS,
        "n_digits_headline": N_DIGITS_HEADLINE,
        "tau_sweep_ms": TAU_SWEEP_MS,
        "tau_grid_ms": TAU_GRID_MS,
        "rate_grid_hz": RATE_GRID_HZ,
        "input_rate_hz": INPUT_RATE_HZ,
        "n_streams": N_STREAMS,
        "n_grid_streams": N_GRID_STREAMS,
        "n_per_stream": N_PER_STREAM,
        "train_seeds": SEEDS,
        "seed": SEED,
        "low_rates_hz": LOW_RATE_HZ,
        "low_rate_streams": LOW_RATE_STREAMS,
        "low_rate_digits": LOW_RATE_DIGITS_PER_STREAM,
        "varying_segments": [list(s) for s in VARYING_HEADLINE],
        "evaluation_partition": "official_mnist_test",
    }


def cell_name(seed):
    return f"ping__off__seed{seed}"


def jobs():
    """Preserve legacy loop order and RNG reset boundaries, including paired sweeps."""
    result = []

    def add(kind, seed, segments, count, group, sample_seed, poisson_seed, **extra):
        result.append(
            {
                "id": f"job-{len(result):03d}",
                "kind": kind,
                "seed": seed,
                "segments": [list(s) for s in segments],
                "streams": count,
                "sample_group": group,
                "sample_seed": sample_seed,
                "poisson_seed": poisson_seed,
                **extra,
            }
        )

    add(
        "headline",
        SEEDS[0],
        [(TAU_HEADLINE_MS, INPUT_RATE_HZ)] * N_DIGITS_HEADLINE,
        1,
        "headline",
        SEED,
        SEED + 1,
    )
    add("varying", SEEDS[0], VARYING_HEADLINE, 1, "varying", SEED + 7, SEED + 9)
    for seed in SEEDS:
        for compensate in (False, True):
            for tau in TAU_SWEEP_MS:
                rate = (
                    INPUT_RATE_HZ * TRAINED_T_MS / tau if compensate else INPUT_RATE_HZ
                )
                add(
                    "tau",
                    seed,
                    [(tau, rate)] * N_PER_STREAM,
                    N_STREAMS,
                    f"tau-{seed}-{compensate}",
                    SEED + 100 + seed,
                    SEED + 1000 + 100 * seed,
                    rate_compensate=compensate,
                )
        for tau in TAU_GRID_MS:
            for rate in RATE_GRID_HZ:
                add(
                    "grid",
                    seed,
                    [(tau, rate)] * N_PER_STREAM,
                    N_GRID_STREAMS,
                    f"grid-{seed}",
                    SEED + 555 + seed,
                    SEED + 2000 + 100 * seed,
                )
    for rate in LOW_RATE_HZ:
        for seed in SEEDS:
            add(
                "low",
                seed,
                [(TRAINED_T_MS, rate)] * LOW_RATE_DIGITS_PER_STREAM,
                LOW_RATE_STREAMS,
                f"low-{seed}-{rate}",
                65000 + seed,
                65000 + 100 * seed,
            )
    return result


FIGURES = (
    "headline_stream.png",
    "headline_stream.pdf",
    "varying_headline_stream.png",
    "varying_headline_stream.pdf",
    "acc_vs_tau.svg",
    "acc_vs_tau.pdf",
    "acc_grid_tau_rate.png",
    "acc_grid_tau_rate.pdf",
)
