from pathlib import Path

from experiments.exp112 import hpc, recipe


def test_complete_factorial_case_order():
    assert [case["id"] for case in recipe.CASES] == [
        "coba-d1",
        "coba-d1000",
        "ping-d1",
        "ping-d1000",
    ]
    assert {(case["architecture"], case["v_grad_dampen"]) for case in recipe.CASES} == {
        ("coba", 1.0),
        ("coba", 1000.0),
        ("ping", 1.0),
        ("ping", 1000.0),
    }


def test_two_percent_mnist_pairing_contract():
    assert recipe.TRAINING_POOL_SAMPLES == 1200
    assert recipe.OPTIMIZER_TRAIN_SAMPLES == 1080
    assert recipe.VALIDATION_SAMPLES == 120
    assert recipe.MNIST_TEST_SAMPLES == 10000
    assert recipe.MNIST_SUBSET_SEED == recipe.MNIST_SPLIT_SEED == recipe.SEED == 42


def test_recipe_is_self_contained_and_changes_only_factors():
    configs = [recipe.configuration(case) for case in recipe.CASES]
    assert all(not cfg["parameter_origin"]["runtime_dependency"] for cfg in configs)
    stripped = []
    for cfg in configs:
        value = dict(cfg)
        value.pop("condition")
        topology = dict(value["topology"])
        topology.pop("ei_strength")
        topology.pop("ei_loop_enabled")
        value["topology"] = topology
        training = dict(value["training"])
        training.pop("voltage_gradient_damping_divisor")
        value["training"] = training
        stripped.append(value)
    assert stripped.count(stripped[0]) == len(stripped)


def test_training_command_carries_fixed_recipe(tmp_path: Path):
    for case in recipe.CASES:
        args = recipe.training_args(case, tmp_path / case["id"])
        assert args[args.index("--max-samples") + 1] == "1200"
        assert args[args.index("--epochs") + 1] == "50"
        assert args[args.index("--ei-strength") + 1] == str(case["ei_strength"])
        assert args[args.index("--v-grad-dampen") + 1] == str(case["v_grad_dampen"])
        assert "exp022" not in " ".join(args)


def test_final_epoch_raster_command_is_paired_digit_zero(tmp_path: Path):
    args = recipe.raster_args(tmp_path / "training", tmp_path / "raster")
    assert args[args.index("--digit") + 1] == "0"
    assert args[args.index("--sample") + 1] == "0"
    assert args[args.index("--recording-mode") + 1] == "spikes"
    assert args[args.index("--output-fields") + 1 : args.index("--out-dir")] == [
        "spk_e",
        "spk_i",
    ]
    assert Path(args[args.index("--load-weights") + 1]).name == "weights_final.pth"


def test_hpc_array_uses_one_frozen_condition_per_task(tmp_path: Path):
    plan = {
        "account": "project",
        "partition": "ampere",
        "walltime": "01:00:00",
        "cpus": 4,
        "memory_gb": 32,
        "gpus": 1,
        "concurrency": 4,
    }
    command = hpc.command(plan, tmp_path / "plan.json")
    assert "--array=0-3%4" in command
    assert "--gres=gpu:1" in command
    assert command[-2:] == [str(tmp_path / "plan.json"), str(hpc.REPO)]
