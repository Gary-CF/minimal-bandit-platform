from __future__ import annotations

import numpy as np

from run import (
    create_environment,
    validate_algorithm_environment,
)


def main() -> None:
    test_cases = [
    ("bernoulli", "random", True),
    ("bernoulli", "ucb1", True),
    ("bernoulli", "ucb_v", True),
    ("bernoulli", "thompson_sampling", True),

    ("gaussian", "random", True),
    ("gaussian", "ucb1", False),
    ("gaussian", "ucb_v", False),
    ("gaussian", "thompson_sampling", False),
]

    for environment_name, algorithm_name, should_pass in test_cases:
        if environment_name == "bernoulli":
            environment_config = {
                "name": "bernoulli",
                "arm_means": [0.2, 0.5, 0.7],
            }
        else:
            environment_config = {
                "name": "gaussian",
                "arm_means": [0.2, 0.5, 0.7],
                "arm_stds": [0.1, 0.1, 0.1],
            }

        env = create_environment(
            environment_config=environment_config,
            rng=np.random.default_rng(0),
        )

        try:
            validate_algorithm_environment(
                algorithm_name=algorithm_name,
                env=env,
            )

        except ValueError as error:
            assert not should_pass, (
                f"{environment_name} + {algorithm_name} "
                "was unexpectedly rejected"
            )

            print(
                f"REJECTED: "
                f"{environment_name} + {algorithm_name}"
            )
            print(f"  reason: {error}")

        else:
            assert should_pass, (
                f"{environment_name} + {algorithm_name} "
                "was unexpectedly accepted"
            )

            print(
                f"ALLOWED:  "
                f"{environment_name} + {algorithm_name}"
            )


if __name__ == "__main__":
    main()