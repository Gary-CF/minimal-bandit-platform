import numpy as np
import pytest
from scipy.optimize import brentq, minimize
from scipy.special import expit

import estimators.discounted_logistic_mle as module
from estimators.discounted_logistic_mle import DiscountedLogisticMLE, _terms


@pytest.mark.parametrize("gamma", [0.8, 0.99, 1.0])
def test_matches_independent_batch_solver(gamma):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 3))
    y = rng.binomial(1, expit(X @ np.array([0.8, -0.5, 0.3])))
    ridge = 0.7
    model = DiscountedLogisticMLE(3, gamma, ridge)
    for n, (x, reward) in enumerate(zip(X, y), 1):
        model.update(x, int(reward))
        if n not in (1, 10, 30):
            continue
        weights = np.array([gamma ** (n - 1 - s) for s in range(n)])

        def objective(theta):
            z = X[:n] @ theta
            return (weights @ (np.logaddexp(0, z) - y[:n] * z)
                    + ridge / 2 * (theta @ theta))

        def jacobian(theta):
            return (X[:n].T @ (weights * (expit(X[:n] @ theta) - y[:n]))
                    + ridge * theta)

        result = minimize(
            objective, np.zeros(3), jac=jacobian,
            method="BFGS", options={"gtol": 1e-8}
        )
        assert np.linalg.norm(jacobian(result.x), ord=np.inf) < 1e-6
        np.testing.assert_allclose(model.theta, result.x, atol=2e-6, rtol=0)
        np.testing.assert_allclose(model.weights, weights, atol=1e-15, rtol=0)


def test_derivatives_by_finite_differences():
    rng = np.random.default_rng(7)
    X, y = rng.normal(size=(9, 3)), rng.integers(2, size=9)
    theta, weights = rng.normal(size=3), 0.9 ** np.arange(8, -1, -1)
    _, gradient, H = _terms(theta, X, y, weights, 0.4)
    eps = 1e-5
    for i in range(3):
        offset = eps * np.eye(3)[i]
        plus = _terms(theta + offset, X, y, weights, 0.4)
        minus = _terms(theta - offset, X, y, weights, 0.4)
        np.testing.assert_allclose(
            gradient[i], (plus[0] - minus[0]) / (2 * eps), atol=1e-8
        )
        np.testing.assert_allclose(
            H[:, i], (plus[1] - minus[1]) / (2 * eps), atol=1e-8
        )


def test_old_data_discounted_but_regularizer_preserved():
    model = DiscountedLogisticMLE(1, 0.5, 0.3)
    model.update(np.array([1.0]), 1)
    model.update(np.array([0.0]), 0)
    expected = brentq(lambda z: 0.3 * z + 0.5 * (expit(z) - 1), 0, 10)
    np.testing.assert_allclose(model.theta, [expected], atol=1e-8)
    np.testing.assert_array_equal(model.weights, [0.5, 1.0])


def test_estimator_is_unconstrained():
    model = DiscountedLogisticMLE(1, 1.0, 0.001)
    model.update(np.array([1.0]), 1)
    assert model.theta[0] > 2.0
    assert model.last_diagnostics["grad_inf"] <= model.tol


def test_extreme_logits_are_finite():
    value, gradient, H = _terms(
        np.array([1000.0]), np.array([[1.0], [-1.0]]),
        np.array([0, 1]), np.ones(2), 0.1
    )
    assert np.isfinite(value)
    assert np.isfinite(gradient).all() and np.isfinite(H).all()
    np.testing.assert_allclose(gradient, [102.0])
    np.testing.assert_allclose(H, [[0.1]])


def test_failure_does_not_commit_and_input_is_copied(monkeypatch):
    model = DiscountedLogisticMLE(2, 0.99, 1.0)
    x = np.array([1.0, 0.0])
    model.update(x, 1)
    x[:] = 99
    np.testing.assert_array_equal(model.X, [[1.0, 0.0]])
    saved = [a.copy() for a in (model.X, model.y, model.theta, model.H)]
    old_info = model.last_diagnostics.copy()

    with pytest.raises(ValueError):
        model.update(np.array([1.0, 0.0]), 2)

    def fail(*args):
        raise RuntimeError("injected solver failure")

    monkeypatch.setattr(module, "_solve", fail)
    with pytest.raises(RuntimeError):
        model.update(np.array([0.0, 1.0]), 0)
    for before, after in zip(saved, (model.X, model.y, model.theta, model.H)):
        np.testing.assert_array_equal(before, after)
    assert model.completed_steps == 1
    assert model.last_diagnostics == old_info


@pytest.mark.parametrize("kwargs", [
    dict(gamma=0), dict(gamma=1.1), dict(gamma=True),
    dict(ridge=0), dict(ridge=float("nan")),
    dict(dimension=True), dict(max_iter=0), dict(tol=0),
])
def test_invalid_configuration(kwargs):
    config = dict(dimension=2, gamma=0.99, ridge=1.0)
    config.update(kwargs)
    with pytest.raises(ValueError):
        DiscountedLogisticMLE(**config)
