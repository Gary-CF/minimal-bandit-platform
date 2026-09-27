"""Persistent E-D4/E-D5 checks against independent history-based references.

Place in tests/ and run: python -m pytest -q
This file does not modify implementations or create experiment output.
"""
import numpy as np
import pytest

from algorithms.sliding_window_ucb import SlidingWindowUCB
from algorithms.discounted_ucb import DiscountedUCB
from algorithms.ucb1 import UCB1
from run import create_algorithm


@pytest.mark.parametrize("num_arms,window", [
    (0, 3), (-1, 3), (True, 3), (2.5, 3),
    (3, 0), (3, -1), (3, True), (3, 2.5),
])
def test_sw_rejects_invalid_parameters(num_arms, window):
    with pytest.raises(ValueError):
        SlidingWindowUCB(num_arms, window)


@pytest.mark.parametrize("gamma", [0.0, 1.0, -0.1, 1.1, True, float("nan"), float("inf")])
def test_du_rejects_invalid_gamma(gamma):
    with pytest.raises(ValueError):
        DiscountedUCB(3, gamma)


@pytest.mark.parametrize("num_arms", [0, -1, True, 2.5])
def test_du_rejects_invalid_arm_count(num_arms):
    with pytest.raises(ValueError):
        DiscountedUCB(num_arms, 0.9)


@pytest.fixture(params=["sw", "du"])
def policy(request):
    if request.param == "sw":
        return SlidingWindowUCB(3, 5)
    return DiscountedUCB(3, 0.9)


def test_initialization_and_selection_are_read_only(policy):
    for arm in range(3):
        counts = policy.counts.copy()
        sums = policy.reward_sums.copy()
        steps = policy.completed_steps
        history = list(policy.history) if hasattr(policy, "history") else None
        for _ in range(2):
            chosen = policy.select_action()
            assert type(chosen) is int
            assert chosen == arm
        np.testing.assert_array_equal(policy.counts, counts)
        np.testing.assert_array_equal(policy.reward_sums, sums)
        assert policy.completed_steps == steps
        if history is not None:
            assert list(policy.history) == history
        policy.update(arm, 0.5)


@pytest.mark.parametrize("action,reward", [
    (-1, 0.5), (3, 0.5), (True, 0.5), (1.5, 0.5),
    (0, -0.1), (0, 1.1), (0, float("nan")), (0, float("inf")),
])
def test_invalid_feedback_leaves_state_unchanged(policy, action, reward):
    policy.update(0, 0.25)
    counts, sums = policy.counts.copy(), policy.reward_sums.copy()
    steps = policy.completed_steps
    history = list(policy.history) if hasattr(policy, "history") else None
    with pytest.raises((ValueError, IndexError)):
        policy.update(action, reward)
    np.testing.assert_array_equal(policy.counts, counts)
    np.testing.assert_array_equal(policy.reward_sums, sums)
    assert policy.completed_steps == steps
    if history is not None:
        assert list(policy.history) == history


@pytest.mark.parametrize("window_size", [1, 3, 200])
def test_sw_statistics_equal_direct_window_recomputation(window_size):
    agent = SlidingWindowUCB(3, window_size)
    rng = np.random.default_rng(7)
    feedback = [(int(rng.integers(3)), float(rng.random())) for _ in range(100)]
    for completed, (action, reward) in enumerate(feedback, 1):
        agent.update(action, reward)
        window = feedback[max(0, completed - window_size):completed]
        expected_counts = [sum(a == arm for a, r in window) for arm in range(3)]
        expected_sums = [sum(r for a, r in window if a == arm) for arm in range(3)]
        assert list(agent.history) == window
        np.testing.assert_array_equal(agent.counts, expected_counts)
        np.testing.assert_allclose(agent.reward_sums, expected_sums, rtol=0, atol=1e-12)
        assert agent.completed_steps == completed
        chosen = agent.select_action()
        assert type(chosen) is int and 0 <= chosen < 3


def test_sw_explores_arm_whose_last_sample_expired():
    agent = SlidingWindowUCB(3, 3)
    for action, reward in [(0, 1), (1, 0), (0, 0), (2, 1), (1, 1), (1, 0)]:
        agent.update(action, reward)
    for _ in range(2):
        action = agent.select_action()
        assert type(action) is int and action == 0
    np.testing.assert_array_equal(agent.counts, [0, 2, 1])
    np.testing.assert_allclose(agent.reward_sums, [0, 1, 1])
    assert agent.completed_steps == 6
    assert list(agent.history) == [(2, 1), (1, 1), (1, 0)]


def test_sw_bonus_can_outweigh_larger_empirical_mean():
    agent = SlidingWindowUCB(2, 5)
    for action, reward in [(0, 0.6)] * 4 + [(1, 0.4)]:
        agent.update(action, reward)
    assert agent.select_action() == 1


def test_sw_without_expiration_matches_existing_ucb1():
    sw, ucb = SlidingWindowUCB(3, 200), UCB1(3)
    rewards = np.random.default_rng(19).integers(0, 2, size=(80, 3))
    for t in range(80):
        action = sw.select_action()
        assert action == ucb.select_action()
        reward = float(rewards[t, action])
        sw.update(action, reward)
        ucb.update(action, reward)


@pytest.mark.parametrize("gamma", [0.5, 0.9, 0.99])
def test_du_statistics_equal_explicit_weighted_history(gamma):
    agent = DiscountedUCB(3, gamma)
    rng = np.random.default_rng(7)
    feedback = [(int(rng.integers(3)), float(rng.random())) for _ in range(60)]
    for completed, (action, reward) in enumerate(feedback, 1):
        agent.update(action, reward)
        counts, sums = np.zeros(3), np.zeros(3)
        for s, (old_action, old_reward) in enumerate(feedback[:completed], 1):
            weight = gamma ** (completed - s)
            counts[old_action] += weight
            sums[old_action] += weight * old_reward
        np.testing.assert_allclose(agent.counts, counts, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(agent.reward_sums, sums, rtol=1e-12, atol=1e-12)
        assert agent.completed_steps == completed


def test_du_revisits_stale_arm_without_mutating_statistics():
    agent = DiscountedUCB(2, 0.9)
    agent.update(0, 0.0)
    for _ in range(20):
        agent.update(1, 1.0)
    counts, sums = agent.counts.copy(), agent.reward_sums.copy()
    for _ in range(2):
        action = agent.select_action()
        assert type(action) is int and action == 0
    np.testing.assert_array_equal(agent.counts, counts)
    np.testing.assert_array_equal(agent.reward_sums, sums)
    assert agent.completed_steps == 21


@pytest.mark.parametrize("extra_updates", [1074, 1100])
def test_du_tiny_positive_and_zero_counts_are_safe(extra_updates):
    agent = DiscountedUCB(2, 0.5)
    agent.update(0, 0.0)
    for _ in range(extra_updates):
        agent.update(1, 1.0)
    if extra_updates == 1074:
        assert agent.counts[0] > 0.0
    else:
        assert agent.counts[0] == 0.0
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        action = agent.select_action()
    assert type(action) is int and action == 0


@pytest.mark.parametrize("config,expected_type,attribute,value", [
    ({"name": "sliding_window_ucb", "parameters": {"window_size": 30}}, SlidingWindowUCB, "window_size", 30),
    ({"name": "discounted_ucb", "parameters": {"gamma": 0.9}}, DiscountedUCB, "gamma", 0.9),
])
def test_factory_preserves_policy_parameters(config, expected_type, attribute, value):
    agent = create_algorithm(config, 3, np.random.default_rng(0), 120)
    assert isinstance(agent, expected_type)
    assert getattr(agent, attribute) == value
