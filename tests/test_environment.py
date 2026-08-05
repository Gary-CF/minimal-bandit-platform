import numpy as np

from envs.bernoulli_bandit import BernoulliBandit

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



