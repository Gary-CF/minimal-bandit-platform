"""Additional contract tests for the Day 3 Bernoulli environment.

Save as tests/test_nonstationary_bernoulli_contract.py in the project.
Keep the original environment tests unchanged. This file contains tests only.

Contract: Python/NumPy integer actions accepted; bool/noninteger/out-of-range
transactions raise ValueError; invalid rounds retain trajectory exceptions;
rewards are Python int; sampling uses the supplied RNG and current means;
evaluation does not advance the RNG.
"""

from copy import deepcopy
import numpy as np
import pytest
from envs.mean_trajectories import StationaryMeans, PiecewiseConstantMeans
from envs.nonstationary_bernoulli_bandit import NonStationaryBernoulliBandit

class RecordingRNG:
    """只用于测试：记录参数，固定返回一个 NumPy 整数。"""
    def __init__(self):
        self.calls = []

    def binomial(self, n, p):
        self.calls.append((n, p))
        return np.int64(1)


def test_environment_accepts_numpy_integer_actions():
    env = NonStationaryBernoulliBandit(
        StationaryMeans([0.0, 1.0]), np.random.default_rng(0)
    )
    for integer_type in (int, np.int32, np.int64):
        assert env.step(integer_type(0), 1) == 0
        assert env.step(integer_type(1), 1) == 1
        assert env.pseudo_regret(integer_type(0), 1) == 1.0
        assert env.pseudo_regret(integer_type(1), 1) == 0.0


def test_environment_rejects_invalid_actions_without_sampling():
    rng = RecordingRNG()
    env = NonStationaryBernoulliBandit(StationaryMeans([0.2, 0.8]), rng)
    invalid_actions = (
        True, False, np.bool_(True), np.bool_(False),
        0.0, 1.5, "0", None, -1, 2, np.int64(-1), np.int64(2),
    )
    for action in invalid_actions:
        with pytest.raises(ValueError):
            env.step(action, 1)
        with pytest.raises(ValueError):
            env.pseudo_regret(action, 1)
    assert rng.calls == []


def test_environment_delegates_round_validation():
    rng = RecordingRNG()
    env = NonStationaryBernoulliBandit(StationaryMeans([0.2, 0.8]), rng)
    cases = (
        (True, TypeError), (1.0, TypeError), ("1", TypeError),
        (None, TypeError), (0, ValueError), (-1, ValueError),
    )
    for t, error in cases:
        with pytest.raises(error):
            env.step(0, t)
        with pytest.raises(error):
            env.pseudo_regret(0, t)
    assert rng.calls == []


def test_step_passes_current_selected_probability_to_rng():
    rng = RecordingRNG()
    env = NonStationaryBernoulliBandit(
        PiecewiseConstantMeans([1, 4], [[0.2, 0.8], [0.9, 0.1]]), rng
    )
    for action, t in ((0, 3), (1, 3), (0, 4), (1, 4), (0, 3)):
        reward = env.step(action, t)
        assert type(reward) is int
        assert reward == 1
    assert [n for n, p in rng.calls] == [1, 1, 1, 1, 1]
    assert [p for n, p in rng.calls] == pytest.approx([0.2, 0.8, 0.9, 0.1, 0.2])


def test_pseudo_regret_does_not_change_rng_state():
    rng = np.random.Generator(np.random.PCG64(123))
    env = NonStationaryBernoulliBandit(
        PiecewiseConstantMeans([1, 4], [[0.2, 0.8], [0.9, 0.1]]), rng
    )
    before = deepcopy(rng.bit_generator.state)
    for t in (3, 4, 1, 4, 100):
        env.pseudo_regret(0, t)
        env.pseudo_regret(1, t)
    assert rng.bit_generator.state == before


def test_step_uses_supplied_rng_without_reseeding():
    rng = np.random.Generator(np.random.PCG64(2026))
    reference = np.random.Generator(np.random.PCG64(2026))
    env = NonStationaryBernoulliBandit(
        PiecewiseConstantMeans([1, 4], [[0.2, 0.8], [0.9, 0.1]]), rng
    )
    actions = (0, 1, 0, 1, 0, 1)
    probabilities = (0.2, 0.8, 0.2, 0.1, 0.9, 0.1)
    expected = [int(reference.binomial(n=1, p=p)) for p in probabilities]
    actual = [env.step(action, t) for t, action in enumerate(actions, start=1)]
    assert actual == expected
    assert rng.bit_generator.state == reference.bit_generator.state
