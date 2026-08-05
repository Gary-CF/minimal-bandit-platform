import numpy as np

from algorithms.ucb1 import UCB1

def test_ucb_initializes_each_arm_once()->None:
    algorithm=UCB1(num_arms=3)

    actions=[]

    for _ in range(3):
        action=algorithm.select_action()
        actions.append(action)

        algorithm.update(
            action=action,
            reward=0
        )

    assert actions == [0,1,2]
    assert np.array_equal(
        algorithm.counts,
        np.array([1,1,1])
    )

def test_ucb_update_tracks_observed_rewards()->None:
    algorithm=UCB1(num_arms=3)

    algorithm.update(
        action=1,
        reward=1
    )

    assert np.array_equal(
        algorithm.counts,
        np.array([0,1,0])
    )

    assert np.allclose(
        algorithm.reward_sums,
        np.array([0.0,1.0,0.0])
    )

    assert np.allclose(
        algorithm.estimated_means,
        np.array([0.0,1.0,0.0])
    )

def test_ucb_estimated_mean_matchs_empirical_mean()->None:
    algorithm= UCB1 (num_arms=2)

    rewards=[1,0,1,1]

    for reward in rewards:
        algorithm.update(
            action=0,
            reward=reward,
        )

    assert algorithm.counts[0]==4
    assert algorithm.reward_sums[0]==3.0
    assert np.isclose(
        algorithm.estimated_means[0],
        3.0/4.0,
    )

    assert algorithm.counts[1]==0
    assert algorithm.reward_sums[1]==0.0
    assert algorithm.estimated_means[1]==0.0

def test_ucb_accepts_reward_inside_unit_interval() -> None:
    algorithm = UCB1(num_arms=2)

    algorithm.update(
        action=0,
        reward=0.5,
    )

    assert algorithm.counts[0] == 1
    assert np.isclose(
        algorithm.reward_sums[0],
        0.5,
    )
    assert np.isclose(
        algorithm.estimated_means[0],
        0.5,
    )