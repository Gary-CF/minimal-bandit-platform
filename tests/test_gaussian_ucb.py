import numpy as np
import pytest

from algorithms.gaussian_ucb import GaussianUCB


# ============================================================
# 1. Constructor
# ============================================================

def test_gaussian_ucb_rejects_invalid_num_arms():
    with pytest.raises(ValueError):
        GaussianUCB(
            num_arms=0,
            known_std=1.0,
        )

    with pytest.raises(ValueError):
        GaussianUCB(
            num_arms=-1,
            known_std=1.0,
        )

    with pytest.raises(ValueError):
        GaussianUCB(
            num_arms=True,
            known_std=1.0,
        )


@pytest.mark.parametrize(
    "known_std",
    [
        0.0,
        -0.1,
        -1.0,
        np.inf,
        np.nan,
    ],
)
def test_gaussian_ucb_rejects_invalid_known_std(
    known_std,
):
    with pytest.raises(ValueError):
        GaussianUCB(
            num_arms=3,
            known_std=known_std,
        )


# ============================================================
# 2. Initialization
# ============================================================

def test_gaussian_ucb_initializes_arms_in_order():
    """
    GaussianUCB 在正式计算 index 前，
    每个 arm 必须先被访问一次。
    """
    algorithm = GaussianUCB(
        num_arms=4,
        known_std=1.0,
    )

    actions = []

    for reward in [
        0.2,
        -0.1,
        1.5,
        0.7,
    ]:
        action = algorithm.select_action()
        actions.append(action)
        algorithm.update(action, reward)

    assert actions == [0, 1, 2, 3]


# ============================================================
# 3. Update
# ============================================================

def test_gaussian_ucb_update_tracks_empirical_mean():
    """
    Gaussian reward 可以是任意实数，
    因此这里故意同时使用正负奖励。
    """
    algorithm = GaussianUCB(
        num_arms=2,
        known_std=1.0,
    )

    rewards = [
        1.5,
        -0.5,
        2.0,
    ]

    for reward in rewards:
        algorithm.update(
            action=0,
            reward=reward,
        )

    assert algorithm.counts[0] == 3

    assert algorithm.reward_sums[0] == pytest.approx(
        sum(rewards)
    )

    assert algorithm.estimated_mean[0] == pytest.approx(
        np.mean(rewards)
    )


def test_gaussian_ucb_update_rejects_invalid_action():
    algorithm = GaussianUCB(
        num_arms=3,
        known_std=1.0,
    )

    with pytest.raises(IndexError):
        algorithm.update(
            action=-1,
            reward=0.0,
        )

    with pytest.raises(IndexError):
        algorithm.update(
            action=3,
            reward=0.0,
        )


# ============================================================
# 4. UCB index
# ============================================================

def test_gaussian_ucb_select_action_matches_manual_index():
    """
    手动设置状态，并按公式：

        mean_i
        + sigma * sqrt(
            2 log(t) / N_i
        )

    独立计算 expected action。
    """
    algorithm = GaussianUCB(
        num_arms=3,
        known_std=0.8,
    )

    algorithm.counts[:] = np.array(
        [10, 4, 20],
        dtype=int,
    )

    algorithm.estimated_mean[:] = np.array(
        [0.5, 0.3, 0.65],
        dtype=float,
    )

    algorithm.reward_sums[:] = (
        algorithm.counts
        * algorithm.estimated_mean
    )

    current_round = (
        int(algorithm.counts.sum())
        + 1
    )

    expected_indices = (
        algorithm.estimated_mean
        + algorithm.known_std
        * np.sqrt(
            2
            * np.log(current_round)
            / algorithm.counts
        )
    )

    expected_action = int(
        np.argmax(expected_indices)
    )

    actual_action = (
        algorithm.select_action()
    )

    assert actual_action == expected_action


def test_larger_known_std_encourages_more_exploration():
    """
    构造两个 arm：

    arm 0:
        均值较高，但已经采样很多次

    arm 1:
        均值较低，但只采样了一次

    sigma 很小时选择 arm 0；
    sigma 很大时，不确定性 bonus 会让 arm 1 胜出。
    """

    small_std_algorithm = GaussianUCB(
        num_arms=2,
        known_std=0.05,
    )

    large_std_algorithm = GaussianUCB(
        num_arms=2,
        known_std=1.0,
    )

    counts = np.array(
        [100, 1],
        dtype=int,
    )

    means = np.array(
        [0.8, 0.0],
        dtype=float,
    )

    for algorithm in [
        small_std_algorithm,
        large_std_algorithm,
    ]:
        algorithm.counts[:] = counts
        algorithm.estimated_mean[:] = means
        algorithm.reward_sums[:] = (
            counts * means
        )

    small_std_action = (
        small_std_algorithm.select_action()
    )

    large_std_action = (
        large_std_algorithm.select_action()
    )

    assert small_std_action == 0
    assert large_std_action == 1