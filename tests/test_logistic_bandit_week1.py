"""D7 regression tests. Save as tests/test_logistic_bandit_week1.py.

Run from the repository root:
    python -m pytest -q -s tests/test_logistic_bandit_week1.py

Episodes use K=5,d=2,T=200 and random/UCB1 baselines. These baselines do
not estimate theta or implement a feature-based Logistic Bandit policy.
Pytest temporary directories retain the per-episode CSVs and configuration.
"""
from __future__ import annotations

import copy
import csv
import json

import numpy as np
import pytest

from envs.logistic_bandit import (
    LogisticBernoulliBandit,
    logistic_gradient,
    logistic_hessian,
    logistic_loss,
)


def inputs():
    features = np.array([
        [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
        [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
    ])
    return features, np.array([1.0, -0.5])


def make_env(seed=0):
    features, theta = inputs()
    return LogisticBernoulliBandit(features, theta, np.random.default_rng(seed))


def test_hand_calculated_example():
    theta, x = np.zeros(2), np.array([1.0, -2.0])
    assert isinstance(logistic_loss(theta, x, 1.0), float)
    np.testing.assert_allclose(logistic_loss(theta, x, 1.0), np.log(2.0))
    np.testing.assert_allclose(logistic_gradient(theta, x, 1.0), [-0.5, 1.0])
    np.testing.assert_allclose(logistic_hessian(theta, x), [[0.25, -0.5], [-0.5, 1.0]])


@pytest.mark.parametrize("theta,x", [
    (np.array([0.3, -0.7]), np.array([1.2, -0.4])),
    (np.array([-1.1, 0.8]), np.array([0.5, 1.3])),
    (np.array([0.2, -0.4, 0.6]), np.array([-0.7, 0.9, 1.1])),
])
@pytest.mark.parametrize("y", [0.0, 1.0])
def test_derivatives(theta, x, y):
    before = theta.copy(), x.copy()
    h = 1e-5
    basis = np.eye(theta.size)
    numerical_gradient = np.array([
        (logistic_loss(theta + h * e, x, y) - logistic_loss(theta - h * e, x, y))
        / (2 * h) for e in basis
    ])
    numerical_hessian = np.column_stack([
        (logistic_gradient(theta + h * e, x, y)
         - logistic_gradient(theta - h * e, x, y)) / (2 * h) for e in basis
    ])
    gradient = logistic_gradient(theta, x, y)
    hessian = logistic_hessian(theta, x)
    assert gradient.shape == theta.shape
    assert hessian.shape == (theta.size, theta.size)
    np.testing.assert_allclose(gradient, numerical_gradient, atol=1e-7, rtol=1e-5)
    np.testing.assert_allclose(hessian, numerical_hessian, atol=1e-7, rtol=1e-5)
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-12)
    assert np.linalg.eigvalsh(hessian).min() >= -1e-12
    np.testing.assert_array_equal(theta, before[0])
    np.testing.assert_array_equal(x, before[1])


@pytest.mark.parametrize("z", [-1000.0, 1000.0])
@pytest.mark.parametrize("y", [0.0, 1.0])
def test_extreme_scores(z, y):
    theta, x = np.array([z]), np.ones(1)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        loss = logistic_loss(theta, x, y)
        gradient = logistic_gradient(theta, x, y)
        hessian = logistic_hessian(theta, x)
    assert np.isfinite(loss) and loss >= 0
    assert np.isfinite(gradient).all() and np.isfinite(hessian).all()
    np.testing.assert_allclose(loss, max(z, 0.0) - y * z, atol=1e-10)
    np.testing.assert_allclose(gradient, [(1.0 if z > 0 else 0.0) - y], atol=1e-12)


@pytest.mark.parametrize("bad", [
    [1.0, 2.0], np.array(1.0), np.ones((1, 2)), np.array([]),
    np.array([1, 2]), np.array([1.0 + 0j, 2.0 + 0j]),
    np.array([np.nan, 1.0]), np.array([np.inf, 1.0]), np.ones(3),
])
def test_invalid_vectors(bad):
    good = np.ones(2)
    for theta, x in ((bad, good), (good, bad)):
        with pytest.raises(ValueError):
            logistic_loss(theta, x, 1.0)
        with pytest.raises(ValueError):
            logistic_gradient(theta, x, 1.0)
        with pytest.raises(ValueError):
            logistic_hessian(theta, x)


@pytest.mark.parametrize("bad", [
    True, np.bool_(False), "1", np.array([1.0]), 1.0 + 0j,
    np.nan, np.inf, -1.0, 0.5, 2.0,
])
def test_invalid_labels(bad):
    for function in (logistic_loss, logistic_gradient):
        with pytest.raises(ValueError):
            function(np.ones(2), np.ones(2), bad)


@pytest.mark.parametrize("label", [0, 1, 0.0, 1.0, np.int64(1), np.float64(0)])
def test_valid_labels(label):
    loss = logistic_loss(np.zeros(2), np.ones(2), label)
    gradient = logistic_gradient(np.zeros(2), np.ones(2), label)
    np.testing.assert_allclose(loss, np.log(2.0))
    np.testing.assert_allclose(gradient, np.full(2, 0.5 - label))


def test_environment_means_and_input_copies():
    features, theta = inputs()
    original_features, original_theta = features.copy(), theta.copy()
    # Independent probability formula on moderate, nonsaturating scores.
    expected = 1.0 / (1.0 + np.exp(-(features @ theta)))
    env = LogisticBernoulliBandit(features, theta, np.random.default_rng(7))
    assert env.num_arms == 5 and env.dimension == 2
    assert env.arm_means.shape == (5,)
    np.testing.assert_allclose(env.arm_means, expected, atol=1e-14, rtol=0)
    assert int(np.argmax(env.arm_means)) == 0
    features[:] = 99.0
    theta[:] = -99.0
    np.testing.assert_array_equal(env.features, original_features)
    np.testing.assert_array_equal(env.theta_star, original_theta)
    np.testing.assert_allclose(env.arm_means, expected, atol=1e-14, rtol=0)


@pytest.mark.parametrize("field,bad", [
    ("features", [[1.0, 0.0]]), ("features", np.ones(2)),
    ("features", np.empty((0, 2))), ("features", np.empty((5, 0))),
    ("features", np.ones((5, 2), dtype=int)),
    ("features", np.full((5, 2), np.nan)),
    ("theta", [1.0, -0.5]), ("theta", np.ones((2, 1))),
    ("theta", np.ones(3)), ("theta", np.array([1.0, np.inf])),
    ("rng", 0),
])
def test_invalid_environment_inputs(field, bad):
    features, theta = inputs()
    kwargs = dict(features=features, theta_star=theta, rng=np.random.default_rng(0))
    kwargs[{"features": "features", "theta": "theta_star", "rng": "rng"}[field]] = bad
    with pytest.raises(ValueError):
        LogisticBernoulliBandit(**kwargs)


@pytest.mark.parametrize("bad,error", [
    (True, ValueError), (np.bool_(True), ValueError),
    (1.0, ValueError), ("1", ValueError), (-1, IndexError), (5, IndexError),
])
def test_invalid_actions_preserve_rng(bad, error):
    env = make_env()
    state = copy.deepcopy(env.rng.bit_generator.state)
    means = env.arm_means.copy()
    for method in (env.step, env.pseudo_regret):
        with pytest.raises(error):
            method(bad)
        assert env.rng.bit_generator.state == state
        np.testing.assert_array_equal(env.arm_means, means)


def test_regret_is_read_only_and_numpy_integer_actions_work():
    env = make_env(9)
    state = copy.deepcopy(env.rng.bit_generator.state)
    for action in range(5):
        value = env.pseudo_regret(np.int64(action))
        assert isinstance(value, float)
        np.testing.assert_allclose(value, max(env.arm_means) - env.arm_means[action])
    assert env.rng.bit_generator.state == state
    reward = env.step(np.int64(0))
    assert isinstance(reward, int) and reward in (0, 1)


def test_saturated_reward_boundaries():
    env = LogisticBernoulliBandit(
        np.array([[1.0], [-1.0]]), np.array([1000.0]), np.random.default_rng(0)
    )
    np.testing.assert_array_equal(env.arm_means, [1.0, 0.0])
    for _ in range(10):
        assert env.step(0) == 1 and env.step(1) == 0
    assert env.pseudo_regret(0) == 0.0 and env.pseudo_regret(1) == 1.0


def episode(policy, seed):
    from run import create_algorithm

    features, theta = inputs()
    means = 1.0 / (1.0 + np.exp(-(features @ theta)))
    env_seed, policy_seed = np.random.SeedSequence(seed).spawn(2)
    env = LogisticBernoulliBandit(features, theta, np.random.default_rng(env_seed))
    reference_rng = np.random.default_rng(env_seed)
    # Only the number of arms and a separate RNG are passed to the policy.
    algorithm = create_algorithm(
        algorithm_config={"name": policy, "parameters": {}},
        num_arms=5, rng=np.random.default_rng(policy_seed), horizon=200,
    )
    records = []
    cumulative = 0.0
    for step in range(1, 201):
        action = algorithm.select_action()
        assert isinstance(action, (int, np.integer)) and not isinstance(action, (bool, np.bool_))
        assert 0 <= action < 5
        reward = env.step(action)
        expected_reward = int(reference_rng.binomial(1, means[action]))
        assert isinstance(reward, int) and reward in (0, 1)
        assert reward == expected_reward
        algorithm.update(action, reward)
        regret = env.pseudo_regret(action)
        expected_regret = float(means.max() - means[action])
        np.testing.assert_allclose(regret, expected_regret, atol=1e-14, rtol=0)
        cumulative += regret
        records.append((step, int(action), reward, regret, cumulative))
    return records


@pytest.mark.parametrize("policy", ["random", "ucb1"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_200_round_episode_and_exact_replay(policy, seed, tmp_path):
    first, second = episode(policy, seed), episode(policy, seed)
    assert first == second and len(first) == 200
    assert np.isfinite(np.array(first, dtype=float)).all()
    assert np.all(np.diff([row[4] for row in first]) >= -1e-12)
    np.testing.assert_allclose(first[-1][4], sum(row[3] for row in first), atol=1e-12)
    paths = [tmp_path / f"{policy}_seed{seed}_repeat{i}.csv" for i in (1, 2)]
    for path, rows in zip(paths, (first, second)):
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["step", "action", "reward", "instant_regret", "cumulative_regret"])
            writer.writerows(rows)
    assert paths[0].read_bytes() == paths[1].read_bytes()
    features, theta = inputs()
    (tmp_path / "config.json").write_text(json.dumps({
        "features": features.tolist(), "theta_star": theta.tolist(),
        "policy": policy, "seed": seed, "horizon": 200,
        "scope": "Environment integration only; policy does not estimate theta",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n{policy}, seed={seed}: K=5,d=2,T=200, reward/regret/replay/CSV passed; "
          f"final regret={first[-1][4]:.6f}; evidence={tmp_path}")
