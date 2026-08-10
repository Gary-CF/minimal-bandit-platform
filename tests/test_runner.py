import pytest

from run import run_single_experiment


@pytest.mark.parametrize(
    "algorithm_name",
    [
        "ucb1",
        "ucb_v",
    ],
)
def test_ucb_algorithms_support_horizon_smaller_than_num_arms(
    algorithm_name: str,
) -> None:
    algorithm_config = {
        "name": algorithm_name,
        "parameters": {},
    }

    if algorithm_name == "ucb_v":
        algorithm_config["parameters"] = {
            "reward_range": 1.0
        }

    records = run_single_experiment(
        seed=0,
        environment_config={
            "name": "bernoulli",
            "arm_means": [
                0.2,
                0.5,
                0.8,
            ],
        },
        horizon=2,
        algorithm_config=(
            algorithm_config
        ),
    )

    assert len(records) == 2

    assert [
        record["action"]
        for record in records
    ] == [0, 1]