from algorithms.epsilon_greedy import EpsilonGreedy
import pytest
import numpy as np

def test_invalid_num_arms():
    with pytest.raises(ValueError):
        EpsilonGreedy(num_arms=-1,epsilon=0.1, rng=np.random.default_rng(42),)
def test_invalid_epsilon():
    with pytest.raises(ValueError):
        EpsilonGreedy(num_arms=3,epsilon=-0.1, rng=np.random.default_rng(42),)
    with pytest.raises(ValueError):
        EpsilonGreedy(num_arms=3,epsilon=1.1, rng=np.random.default_rng(42),)

def test_initial_state():
    algorithm = EpsilonGreedy(
        num_arms=3,
        epsilon=0.1,
         rng=np.random.default_rng(42),
    )

    assert algorithm.counts.shape==(3,)
    assert algorithm.reward_sum.shape==(3,)
    assert algorithm.estimated_reward.shape==(3,)

    np.testing.assert_array_equal(
        algorithm.counts,np.array([0,0,0])
    )

    np.testing.assert_allclose(
        algorithm.reward_sum,np.array([0.0,0.0,0.0])
    )

    np.testing.assert_allclose(
        algorithm.estimated_reward,np.array([0.0,0.0,0.0])
    )

def test_update_statistics():
    algorithm=EpsilonGreedy(
        num_arms=3,
        epsilon=0.1,
         rng=np.random.default_rng(42),
    )

    algorithm.update(action=1,reward=1.0)
    algorithm.update(action=1,reward=0.5)

    np.testing.assert_array_equal(algorithm.counts,np.array([0,2,0]))
    np.testing.assert_allclose(algorithm.reward_sum,np.array([0.0,1.5,0.0]))
    np.testing.assert_allclose(algorithm.estimated_reward,np.array([0.0,0.75,0.0]))

    algorithm.update(action=2,reward=-0.4)

    np.testing.assert_array_equal(
        algorithm.counts,np.array([0,2,1])
    )

    np.testing.assert_allclose(
        algorithm.reward_sum,np.array([0.0,1.5,-0.4])
    )

    np.testing.assert_allclose(
        algorithm.estimated_reward,np.array([0.0,0.75,-0.4])
    )

def test_epsilon_zero_selects_best_empical_arm():
    algorithm=EpsilonGreedy(
        num_arms=3,
        epsilon=0.0,
         rng=np.random.default_rng(42),
    )

    algorithm.update(action=0,reward=0.2)
    algorithm.update(action=1,reward=0.9)
    algorithm.update(action=2,reward=0.5)

    for _ in range(20):
        action=algorithm.select_action()
        assert action==1

def test_epsilon_one_returns_valid_actions():
    algorithm=EpsilonGreedy(
        num_arms=3,
        epsilon=1.0,
         rng=np.random.default_rng(42)
    )

    actions=[
        algorithm.select_action()
        for _ in range(100)
    ]

    assert all(
        0<=action<3 for action in actions
    )
    assert all(
        isinstance(action,(int,np.integer)) for action in actions
    )

def test_same_seed_produces_same_action_sequence():
    algorithm_1=EpsilonGreedy(
        num_arms=3,
        epsilon=0.4,
        rng=np.random.default_rng(42),
    )

    algorithm_2=EpsilonGreedy(
        num_arms=3,
        epsilon=0.4,
        rng=np.random.default_rng(42),
    )

    action_1=[
        algorithm_1.select_action() for _ in range(100)
    ]

    action_2=[
        algorithm_2.select_action() for _ in range(100)
    ]

    assert action_1 == action_2