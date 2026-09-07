from __future__ import annotations

import numpy as np
import pytest

from algorithms.gaussian_thompson_sampling import GaussianThompsonSampling


def make_algorithm(
    *,
    num_arms: int = 3,
    prior_mean: float = 0.0,
    prior_variance: float = 1.0,
    noise_variance: float = 2.0,
    seed: int = 42,
) -> GaussianThompsonSampling:
    return GaussianThompsonSampling(
        num_arms=num_arms,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        noise_variance=noise_variance,
        rng=np.random.default_rng(seed),
    )


@pytest.mark.parametrize(
    "prior_variance",
    [0.0, -1.0],
)
def test_prior_variance_must_be_positive(
    prior_variance: float,
) -> None:
    with pytest.raises(ValueError):
        make_algorithm(
            prior_variance=prior_variance,
        )


@pytest.mark.parametrize(
    "noise_variance",
    [0.0, -1.0],
)
def test_noise_variance_must_be_positive(
    noise_variance: float,
) -> None:
    with pytest.raises(ValueError):
        make_algorithm(
            noise_variance=noise_variance,
        )


def test_initial_posterior_equals_prior() -> None:
    algorithm = make_algorithm(
        num_arms=4,
        prior_mean=1.5,
        prior_variance=3.0,
    )

    assert np.array_equal(
        algorithm.counts,
        np.zeros(4, dtype=int),
    )
    assert np.allclose(
        algorithm.reward_sums,
        np.zeros(4),
    )
    assert np.allclose(
        algorithm.posterior_means,
        np.full(4, 1.5),
    )
    assert np.allclose(
        algorithm.posterior_variances,
        np.full(4, 3.0),
    )


def test_one_update_matches_hand_calculation() -> None:
    """
    prior:
        m0 = 0
        v0 = 1

    observation noise:
        sigma^2 = 2

    one reward:
        x = 2

    expected posterior:
        v = (1/1 + 1/2)^(-1) = 2/3
        m = v * (0/1 + 2/2) = 2/3
    """
    algorithm = make_algorithm(
        num_arms=2,
        prior_mean=0.0,
        prior_variance=1.0,
        noise_variance=2.0,
    )

    algorithm.update(
        action=0,
        reward=2.0,
    )

    assert algorithm.counts[0] == 1
    assert np.isclose(
        algorithm.reward_sums[0],
        2.0,
    )
    assert np.isclose(
        algorithm.posterior_variances[0],
        2.0 / 3.0,
    )
    assert np.isclose(
        algorithm.posterior_means[0],
        2.0 / 3.0,
    )

    # 未被选择的 arm 不应发生变化。
    assert np.isclose(
        algorithm.posterior_variances[1],
        1.0,
    )
    assert np.isclose(
        algorithm.posterior_means[1],
        0.0,
    )


def test_posterior_variance_never_increases_with_more_observations() -> None:
    algorithm = make_algorithm(
        num_arms=2,
        prior_variance=2.0,
        noise_variance=1.5,
    )

    variances = [
        algorithm.posterior_variances[0]
    ]

    for reward in [0.2, -1.0, 0.7, 1.5, 0.3]:
        algorithm.update(
            action=0,
            reward=reward,
        )
        variances.append(
            algorithm.posterior_variances[0]
        )

    assert all(
        current <= previous
        for previous, current in zip(
            variances,
            variances[1:],
        )
    )


def test_same_rng_seed_gives_same_action_sequence() -> None:
    algorithm_1 = make_algorithm(
        seed=12345,
    )
    algorithm_2 = make_algorithm(
        seed=12345,
    )

    actions_1 = [
        algorithm_1.select_action()
        for _ in range(30)
    ]
    actions_2 = [
        algorithm_2.select_action()
        for _ in range(30)
    ]

    assert actions_1 == actions_2


def test_selected_action_is_always_valid() -> None:
    algorithm = make_algorithm(
        num_arms=5,
        seed=7,
    )

    for _ in range(200):
        action = algorithm.select_action()

        assert 0 <= action < algorithm.num_arms


def test_update_changes_only_selected_arm() -> None:
    algorithm = make_algorithm(
        num_arms=3,
        prior_mean=0.5,
        prior_variance=2.0,
        noise_variance=1.0,
    )

    old_means = algorithm.posterior_means.copy()
    old_variances = (
        algorithm.posterior_variances.copy()
    )

    algorithm.update(
        action=1,
        reward=1.2,
    )

    assert np.isclose(
        algorithm.posterior_means[0],
        old_means[0],
    )
    assert np.isclose(
        algorithm.posterior_means[2],
        old_means[2],
    )
    assert np.isclose(
        algorithm.posterior_variances[0],
        old_variances[0],
    )
    assert np.isclose(
        algorithm.posterior_variances[2],
        old_variances[2],
    )

    assert algorithm.counts[1] == 1
    assert algorithm.counts[0] == 0
    assert algorithm.counts[2] == 0


@pytest.mark.parametrize(
    "reward",
    [np.inf, -np.inf, np.nan],
)
def test_reward_must_be_finite(
    reward: float,
) -> None:
    algorithm = make_algorithm()

    with pytest.raises(ValueError):
        algorithm.update(
            action=0,
            reward=reward,
        )


@pytest.mark.parametrize(
    "action",
    [-1, 3],
)
def test_invalid_action_is_rejected(
    action: int,
) -> None:
    algorithm = make_algorithm(
        num_arms=3,
    )

    with pytest.raises(ValueError):
        algorithm.update(
            action=action,
            reward=0.5,
        )