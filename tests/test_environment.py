import numpy as np

from envs.bernoulli_bandit import BernoulliBandit

import pytest

def test_deterministic_arms_return_expected_rewards()->None:
    rng=np.random.default_rng(0)

    env=BernoulliBandit(
        arm_means=[0.0,1.0],
        rng=rng
    )

    assert env.step(0)==0
    assert env.step(1)==1

def test_pseudo_regret_matches_definition()->None:
    rng=np.random.default_rng(0)

    env=BernoulliBandit(
        arm_means=[0.2,0.5,0.9],rng=rng
    )

    expected_regrets=[0.7,0.4,0.0]

    for action,expected_regret in enumerate(expected_regrets):
        actual_regret=env.pseudo_regret(action)

        assert np.isclose(
            actual_regret,
            expected_regret,
        )

def test_pseudo_regret_is_nonnegative_and_zero_for_best_arm()->None:
    rng=np.random.default_rng(0)

    env=BernoulliBandit(
        arm_means=[0.15,0.6,0.35,0.8],
        rng=rng
    )

    regrets=[
        env.pseudo_regret(action)
        for action in range(len(env.arm_means))
    ]

    assert all(regret>=0.0 for regret in regrets)
    assert any(np.isclose(regret,0.0) for regret in regrets)

def test_same_seed_reproduces_reward_sequence()->None:
    arm_means=[0.2,0.5,0.8]
    actions=[0,1,2,2,1,0]*20

    env_one= BernoulliBandit(
        arm_means=arm_means,
        rng=np.random.default_rng(42)
    )

    env_two= BernoulliBandit(
            arm_means=arm_means,
            rng=np.random.default_rng(42)
        )

    rewards_one=[
        env_one.step(action)
        for action in actions
    ]

    rewards_two=[
        env_two.step(action)
        for action in actions
    ]

    assert rewards_one == rewards_two


def test_bernoulli_rejects_invalid_arm_means() -> None:
    rng = np.random.default_rng(0)

    invalid_arm_means = [
        [],
        [-0.1, 0.5],
        [0.5, 1.1],
        [0.5, np.nan],
        [[0.2, 0.8]],
    ]

    for arm_means in invalid_arm_means:
        with pytest.raises(ValueError):
            BernoulliBandit(
                arm_means=arm_means,
                rng=rng,
            )


def test_bernoulli_rejects_invalid_actions() -> None:
    env = BernoulliBandit(
        arm_means=[0.2, 0.8],
        rng=np.random.default_rng(0),
    )

    with pytest.raises(ValueError):
        env.step(-1)

    with pytest.raises(ValueError):
        env.step(len(env.arm_means))

    with pytest.raises(ValueError):
        env.pseudo_regret(-1)

    with pytest.raises(ValueError):
        env.pseudo_regret(
            len(env.arm_means)
        )

