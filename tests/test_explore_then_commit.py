from __future__ import annotations

import pytest

from algorithms.explore_then_commit import ETC


def _run_steps(
    algorithm: ETC,
    rewards_by_arm: dict[int, float],
    num_steps: int,
) -> list[int]:
    """Run the standard select -> reward -> update loop deterministically."""
    actions: list[int] = []
    for _ in range(num_steps):
        action = algorithm.select_action()
        actions.append(action)
        algorithm.update(action, rewards_by_arm[action])
    return actions


def test_exploration_actions_follow_round_robin_order() -> None:
    algorithm = ETC(num_arms=3, exploration_rounds_per_arm=2)

    actions = _run_steps(
        algorithm,
        rewards_by_arm={0: 0.1, 1: 0.9, 2: 0.4},
        num_steps=6,
    )

    assert actions == [0, 1, 2, 0, 1, 2]
    assert algorithm.counts.tolist() == [2, 2, 2]
    assert int(algorithm.counts.sum()) == 6
    assert algorithm.committed_arm is None


def test_commits_to_arm_with_largest_exploration_mean() -> None:
    algorithm = ETC(num_arms=3, exploration_rounds_per_arm=2)
    _run_steps(
        algorithm,
        rewards_by_arm={0: 0.1, 1: 0.9, 2: 0.4},
        num_steps=6,
    )

    action = algorithm.select_action()

    assert action == 1
    assert algorithm.committed_arm == 1


def test_never_switches_after_commit_but_keeps_updating_statistics() -> None:
    algorithm = ETC(num_arms=3, exploration_rounds_per_arm=2)
    _run_steps(
        algorithm,
        rewards_by_arm={0: 0.1, 1: 0.9, 2: 0.4},
        num_steps=6,
    )

    committed_action = algorithm.select_action()
    assert committed_action == 1

    # This reward makes arm 1's updated empirical mean worse than arm 2's.
    algorithm.update(committed_action, -100.0)

    assert algorithm.estimated_mean[1] < algorithm.estimated_mean[2]
    assert algorithm.select_action() == 1
    assert algorithm.committed_arm == 1
    assert algorithm.counts.tolist() == [2, 3, 2]
    assert int(algorithm.counts.sum()) == 7


def test_short_horizon_can_end_during_exploration() -> None:
    algorithm = ETC(num_arms=3, exploration_rounds_per_arm=4)

    actions = _run_steps(
        algorithm,
        rewards_by_arm={0: 0.0, 1: 1.0, 2: 0.5},
        num_steps=5,
    )

    assert actions == [0, 1, 2, 0, 1]
    assert algorithm.counts.tolist() == [2, 2, 1]
    assert int(algorithm.counts.sum()) == 5
    assert algorithm.committed_arm is None


def test_update_tracks_counts_reward_sums_and_means() -> None:
    algorithm = ETC(num_arms=2, exploration_rounds_per_arm=1)

    algorithm.update(0, 0.25)
    algorithm.update(0, 0.75)
    algorithm.update(1, -1.0)

    assert algorithm.counts.tolist() == [2, 1]
    assert algorithm.reward_sums.tolist() == pytest.approx([1.0, -1.0])
    assert algorithm.estimated_mean.tolist() == pytest.approx([0.5, -1.0])
    assert int(algorithm.counts.sum()) == 3


@pytest.mark.parametrize("invalid_num_arms", [0, -1, 1.5, True, False, "3", None])
def test_rejects_invalid_num_arms(invalid_num_arms: object) -> None:
    with pytest.raises(ValueError):
        ETC(
            num_arms=invalid_num_arms,  # type: ignore[arg-type]
            exploration_rounds_per_arm=2,
        )


@pytest.mark.parametrize(
    "invalid_rounds",
    [0, -1, 1.5, True, False, "2", None],
)
def test_rejects_invalid_exploration_rounds(invalid_rounds: object) -> None:
    with pytest.raises(ValueError):
        ETC(
            num_arms=3,
            exploration_rounds_per_arm=invalid_rounds,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_action", [-1, 3])
def test_update_rejects_out_of_range_action(invalid_action: int) -> None:
    algorithm = ETC(num_arms=3, exploration_rounds_per_arm=1)

    with pytest.raises(IndexError):
        algorithm.update(invalid_action, 1.0)
