from __future__ import annotations

import numpy as np
import pytest

from algorithms.moss import MOSS


def _update_many(algorithm: MOSS, action: int, rewards: list[float]) -> None:
    for reward in rewards:
        algorithm.update(action, reward)


def test_initialization_selects_every_arm_in_order() -> None:
    algorithm = MOSS(num_arms=3, horizon=100)
    actions: list[int] = []

    for reward in [0.0, 1.0, 0.0]:
        action = algorithm.select_action()
        actions.append(action)
        algorithm.update(action, reward)

    assert actions == [0, 1, 2]
    assert algorithm.counts.tolist() == [1, 1, 1]


def test_selects_arm_with_largest_hand_calculated_index() -> None:
    algorithm = MOSS(num_arms=2, horizon=100)

    # N_0 = 1, mean_0 = 0.0
    _update_many(algorithm, action=0, rewards=[0.0])
    # N_1 = 4, mean_1 = 0.5
    _update_many(algorithm, action=1, rewards=[1.0, 1.0, 0.0, 0.0])

    index_0 = 0.0 + np.sqrt(np.log(100 / (2 * 1)) / 1)
    index_1 = 0.5 + np.sqrt(np.log(100 / (2 * 4)) / 4)

    assert index_0 == pytest.approx(1.977883466088977)
    assert index_1 == pytest.approx(1.2946270578561139)
    assert index_0 > index_1
    assert algorithm.select_action() == 0


def test_nonpositive_log_term_is_clipped_to_zero() -> None:
    algorithm = MOSS(num_arms=2, horizon=4)

    # T / (K N_0) = 1; empirical mean = 1/2.
    _update_many(algorithm, action=0, rewards=[1.0, 0.0])
    # T / (K N_1) = 2/3; empirical mean = 2/3.
    _update_many(algorithm, action=1, rewards=[1.0, 1.0, 0.0])

    # Raise instead of silently accepting invalid square roots or divisions.
    with np.errstate(invalid="raise", divide="raise"):
        action = algorithm.select_action()

    assert action == 1


def test_short_horizon_smaller_than_num_arms_is_legal() -> None:
    algorithm = MOSS(num_arms=5, horizon=2)
    actions: list[int] = []

    for reward in [0.0, 1.0]:
        action = algorithm.select_action()
        actions.append(action)
        algorithm.update(action, reward)

    assert actions == [0, 1]
    assert algorithm.counts.tolist() == [1, 1, 0, 0, 0]
    assert int(algorithm.counts.sum()) == 2


def test_update_tracks_counts_reward_sums_and_means() -> None:
    algorithm = MOSS(num_arms=2, horizon=10)

    algorithm.update(0, 0.0)
    algorithm.update(0, 1.0)
    algorithm.update(1, 1.0)

    assert algorithm.counts.tolist() == [2, 1]
    assert algorithm.reward_sums.tolist() == pytest.approx([1.0, 1.0])
    assert algorithm.estimated_means.tolist() == pytest.approx([0.5, 1.0])
    assert int(algorithm.counts.sum()) == 3


@pytest.mark.parametrize("invalid_num_arms", [0, -1, 1.5, True, False, "3", None])
def test_rejects_invalid_num_arms(invalid_num_arms: object) -> None:
    with pytest.raises(ValueError):
        MOSS(
            num_arms=invalid_num_arms,  # type: ignore[arg-type]
            horizon=100,
        )


@pytest.mark.parametrize("invalid_horizon", [0, -1, 1.5, True, False, "100", None])
def test_rejects_invalid_horizon(invalid_horizon: object) -> None:
    with pytest.raises(ValueError):
        MOSS(
            num_arms=3,
            horizon=invalid_horizon,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_action", [-1, 3])
def test_update_rejects_out_of_range_action(invalid_action: int) -> None:
    algorithm = MOSS(num_arms=3, horizon=100)

    with pytest.raises(IndexError):
        algorithm.update(invalid_action, 1.0)
