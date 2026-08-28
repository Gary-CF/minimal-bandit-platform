from __future__ import annotations

import numpy as np
import pytest

from algorithms.gaussian_thompson_sampling import GaussianThompsonSampling
from envs.bernoulli_bandit import BernoulliBandit
from envs.gaussian_bandit import GaussianBandit
from run import create_algorithm, validate_algorithm_environment


def test_gaussian_thompson_sampling_accepts_gaussian_environment() -> None:
    env = GaussianBandit(
        arm_means=[0.0, 0.5, 1.0],
        arm_stds=[1.0, 1.0, 1.0],
        rng=np.random.default_rng(0),
    )

    validate_algorithm_environment(
        algorithm_name="gaussian_thompson_sampling",
        env=env,
    )


def test_gaussian_thompson_sampling_rejects_bernoulli_environment() -> None:
    env = BernoulliBandit(
        arm_means=[0.2, 0.5, 0.8],
        rng=np.random.default_rng(0),
    )

    with pytest.raises(ValueError):
        validate_algorithm_environment(
        algorithm_name="gaussian_thompson_sampling",
        env=env,
    )


def test_create_gaussian_thompson_sampling() -> None:
    algorithm = create_algorithm(
        algorithm_config={
            "name": "gaussian_thompson_sampling",
            "parameters": {
                "prior_mean": 0.0,
                "prior_variance": 2.0,
                "noise_variance": 1.5,
            },
        },
        num_arms=3,
        rng=np.random.default_rng(123),
        horizon=100,
    )

    assert isinstance(
        algorithm,
        GaussianThompsonSampling,
    )
    assert algorithm.num_arms == 3
    assert np.isclose(
        algorithm.prior_mean,
        0.0,
    )
    assert np.isclose(
        algorithm.prior_variance,
        2.0,
    )
    assert np.isclose(
        algorithm.noise_variance,
        1.5,
    )