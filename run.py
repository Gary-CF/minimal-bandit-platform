from __future__ import annotations

import numpy as np


def main() -> None:
    # 1、实验参数

    seed = 42
    rng = np.random.default_rng(seed)

    arm_means = np.array([0.2, 0.5, 0.7], dtype=float)

    horizon = 1000

    num_arms = len(arm_means)

    best_mean = float(np.max(arm_means))
    best_arm = int(np.argmax(arm_means))

    # 2、实验状态

    total_reward = 0
    cumulative_regret = 0.0

    records: list[dict[str, int | float]] = []

    # 3、Bandit实验主循环

    for t in range(1, horizon + 1):
        action = int(rng.integers(num_arms))

        reward = int(
            rng.binomial(
                n=1,
                p=arm_means[action],
            )
        )

        instant_regret = best_mean - float(arm_means[action])

        total_reward += reward
        cumulative_regret += instant_regret

        record = {
            "step": t,
            "action": action,
            "reward": reward,
            "instant_regret": instant_regret,
            "cumulative_regret": cumulative_regret,
        }
        records.append(record)
    # 4、自动调试
    assert len(records) == horizon

    assert all(record["reward"] in (0, 1) for record in records)

    assert all(0 <= record["action"] < num_arms for record in records)

    assert all(record["instant_regret"] >= 0 for record in records)

    cumulative_regrets = [float(record["cumulative_regret"]) for record in records]

    assert all(
        current >= previous
        for previous, current in zip(
            cumulative_regrets,
            cumulative_regrets[1:],
        )
    )

    # 5、输出结果
    print(f"number of arms:{num_arms}")
    print(f"best arm:{best_arm}")
    print(f"best mean:{best_mean:.2f}")
    print(f"total reward:{total_reward}")
    print(f"cumulative regret:{cumulative_regret:.2f}")

if __name__ == "__main__":
    main()
