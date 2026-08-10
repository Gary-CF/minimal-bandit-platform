import numpy as np
import pytest

from algorithms.ucb_v import UCBV


def test_ucb_v_forces_initial_exploration() -> None:
    algorithm = UCBV(
        num_arms=2,
        reward_range=1.0,
    )

    assert algorithm.select_action() == 0

    algorithm.update(
        action=0,
        reward=0.0,
    )

    assert algorithm.select_action() == 1


def test_ucb_v_tracks_statistics_correctly() -> None:
    algorithm = UCBV(
        num_arms=2,
        reward_range=1.0,
    )

    algorithm.update(0, 0.0)
    algorithm.update(1, 1.0)
    algorithm.update(0, 1.0)
    algorithm.update(0, 0.0)

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
        np.array([
            1.0 / 3.0,
            1.0,
        ]),
    )

    assert np.allclose(
        algorithm.empirical_variances,
        np.array([
            2.0 / 9.0,
            0.0,
        ]),
    )


def test_ucb_v_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        UCBV(
            num_arms=0,
            reward_range=1.0,
        )

    with pytest.raises(ValueError):
        UCBV(
            num_arms=2,
            reward_range=0.0,
        )

    algorithm = UCBV(
        num_arms=2,
        reward_range=1.0,
    )

    with pytest.raises(ValueError):
        algorithm.update(
            action=-1,
            reward=1.0,
        )

    with pytest.raises(ValueError):
        algorithm.update(
            action=2,
            reward=1.0,
        )

    with pytest.raises(ValueError):
        algorithm.update(
            action=0,
            reward=np.nan,
        )