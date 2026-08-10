import numpy as np
import pytest

from run import (
    create_environment,
    validate_algorithm_environment,
)


@pytest.mark.parametrize(
    (
        "environment_name",
        "algorithm_name",
        "should_pass",
    ),
    [
        ("bernoulli", "random", True),
        ("bernoulli", "ucb1", True),
        ("bernoulli", "ucb_v", True),
        (
            "bernoulli",
            "thompson_sampling",
            True,
        ),
        ("gaussian", "random", True),
        ("gaussian", "ucb1", False),
        ("gaussian", "ucb_v", False),
        (
            "gaussian",
            "thompson_sampling",
            False,
        ),
    ],
)
def test_algorithm_environment_compatibility(
    environment_name: str,
    algorithm_name: str,
    should_pass: bool,
) -> None:
    if environment_name == "bernoulli":
        environment_config = {
            "name": "bernoulli",
            "arm_means": [
                0.2,
                0.5,
                0.7,
            ],
        }
    else:
        environment_config = {
            "name": "gaussian",
            "arm_means": [
                0.2,
                0.5,
                0.7,
            ],
            "arm_stds": [
                0.1,
                0.1,
                0.1,
            ],
        }

    env = create_environment(
        environment_config=(
            environment_config
        ),
        rng=np.random.default_rng(0),
    )

    if should_pass:
        validate_algorithm_environment(
            algorithm_name=algorithm_name,
            env=env,
        )
    else:
        with pytest.raises(ValueError):
            validate_algorithm_environment(
                algorithm_name=(
                    algorithm_name
                ),
                env=env,
            )