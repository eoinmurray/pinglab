from experiments import (
    exp025,
    exp037,
    exp038,
    exp041,
    exp042,
    exp044,
    exp046,
    exp049,
    exp082,
)


def test_collection_checkpoint_roles_are_explicit() -> None:
    assert {
        exp025.CHECKPOINT_ROLE,
        exp041.CHECKPOINT_ROLE,
        exp042.CHECKPOINT_ROLE,
        exp044.CHECKPOINT_ROLE,
        exp046.CHECKPOINT_ROLE,
        exp049.CHECKPOINT_ROLE,
    } == {"final_epoch"}
    assert {
        exp037.CHECKPOINT_ROLE,
        exp038.CHECKPOINT_ROLE,
        exp082.CHECKPOINT_ROLE,
    } == {"best_validation"}


def test_collection_runners_derive_one_role_from_their_analysis_purpose() -> None:
    endpoint = (exp025, exp041, exp042, exp044, exp046, exp049)
    deployment = (exp037, exp038, exp082)
    for module in endpoint:
        assert module.ANALYSIS_PURPOSE == "endpoint_dynamics"
        assert module.CHECKPOINT_POLICY == {
            "purpose": module.ANALYSIS_PURPOSE,
            "role": "final_epoch",
        }
        assert module.CHECKPOINT_ROLE == module.CHECKPOINT_POLICY["role"]
    for module in deployment:
        assert module.ANALYSIS_PURPOSE == "deployment_performance"
        assert module.CHECKPOINT_POLICY == {
            "purpose": module.ANALYSIS_PURPOSE,
            "role": "best_validation",
        }
        assert module.CHECKPOINT_ROLE == module.CHECKPOINT_POLICY["role"]
