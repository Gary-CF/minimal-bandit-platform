import numpy as np

from algorithms.ucb_v import UCBV


def main() -> None:
    algorithm = UCBV(
        num_arms=2,
        reward_range=1.0,
    )

    # 初始化时应首先选择第 0 个臂。
    assert algorithm.select_action() == 0

    algorithm.update(
        action=0,
        reward=0.0,
    )

    # 第 0 个臂已尝试，第 1 个还未尝试。
    assert algorithm.select_action() == 1

    algorithm.update(
        action=1,
        reward=1.0,
    )

    # 额外向第 0 个臂加入奖励 1 和 0。
    # 第 0 个臂的奖励序列现在是 [0, 1, 0]。
    algorithm.update(
        action=0,
        reward=1.0,
    )

    algorithm.update(
        action=0,
        reward=0.0,
    )

    print("counts:")
    print(algorithm.counts)

    print("reward sums:")
    print(algorithm.reward_sums)

    print("reward square sums:")
    print(algorithm.reward_square_sums)

    print("estimated means:")
    print(algorithm.estimated_means)

    print("empirical variances:")
    print(algorithm.empirical_variances)

    assert np.array_equal(
        algorithm.counts,
        np.array([3, 1]),
    )

    assert np.allclose(
        algorithm.reward_sums,
        np.array([1.0, 1.0]),
    )

    assert np.allclose(
        algorithm.reward_square_sums,
        np.array([1.0, 1.0]),
    )

    assert np.allclose(
        algorithm.estimated_means,
        np.array([1.0 / 3.0, 1.0]),
    )

    assert np.allclose(
        algorithm.empirical_variances,
        np.array([2.0 / 9.0, 0.0]),
    )

    action = algorithm.select_action()

    assert 0 <= action < 2

    print("all UCB-V checks passed")


if __name__ == "__main__":
    main()