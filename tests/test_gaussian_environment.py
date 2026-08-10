import numpy as np
import pytest

from envs.gaussian_bandit import GaussianBandit

def make_gaussian_bandit() -> GaussianBandit:
    return GaussianBandit(
        arm_means=[0.2, 0.8],
        arm_stds=[1.0, 1.0],
        rng=np.random.default_rng(0),
    )

def test_step_rejects_negative_action() -> None:
    env = make_gaussian_bandit()

    with pytest.raises(ValueError):
        env.step(-1)

def test_step_rejects_action_equal_to_num_arms() -> None:
    env = make_gaussian_bandit()

    with pytest.raises(ValueError):
        env.step(len(env.arm_means))

def test_step_accepts_last_valid_action() -> None:
    env = make_gaussian_bandit()

    reward = env.step(
        len(env.arm_means) - 1
    )

    assert np.isfinite(reward)     

def test_pseudo_regret_rejects_negative_action() -> None:
    env = make_gaussian_bandit()

    with pytest.raises(ValueError):
        env.pseudo_regret(-1)

def test_pseudo_regret_rejects_action_equal_to_num_arms() -> None:
    env = make_gaussian_bandit()

    with pytest.raises(ValueError):
        env.pseudo_regret(
            len(env.arm_means)
        )