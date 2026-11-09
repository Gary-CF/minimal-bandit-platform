import numpy as np
import pytest
from scipy.optimize import minimize

import estimators.domd_logistic as domd_module
from estimators.domd_logistic import OnlineLogisticDOMD
from estimators.omd_logistic import OnlineLogisticOMD
from algorithms.logistic_domd import LogisticDOMDPolicy


def reference_step(theta, H, x, y, gamma, eta, l2, radius):
    A = gamma * H + (1 - gamma) * l2 * np.eye(len(theta))
    p = 1 / (1 + np.exp(-(x @ theta)))
    g, G = (p - y) * x, p * (1 - p) * np.outer(x, x)
    def objective(v):
        z = radius * v - theta
        return g @ z + 0.5 * z @ G @ z + z @ A @ z / (2 * eta)
    def gradient(v):
        z = radius * v - theta
        return radius * (g + G @ z + A @ z / eta)
    result = minimize(objective, theta / radius, jac=gradient, method="SLSQP",
        constraints=[{"type": "ineq", "fun": lambda v: 1-v@v,
                      "jac": lambda v: -2*v}], options={"ftol": 1e-10, "maxiter": 500})
    assert result.success, result.message
    candidate = radius * result.x
    p_new = 1 / (1 + np.exp(-(x @ candidate)))
    return candidate, A + p_new * (1-p_new) * np.outer(x, x)


@pytest.mark.parametrize("radius", [0.2, 2.0])
@pytest.mark.parametrize("gamma", [0.8, 1.0])
def test_independent_constrained_reference(radius, gamma):
    model = OnlineLogisticDOMD(2, gamma, eta=2., l2=0.1, radius=radius)
    model.theta = np.array([0.1, -0.1])
    model.H = np.array([[2., .3], [.3, 1.]])
    x = np.array([.8, -.6])
    theta, H = reference_step(model.theta, model.H, x, 1., gamma, 2., .1, radius)
    info = model.update(x, 1.)
    np.testing.assert_allclose(model.theta, theta, atol=2e-6, rtol=0)
    np.testing.assert_allclose(model.H, H, atol=2e-6, rtol=0)
    assert (info["rho"] > 0) == (radius == .2)


def test_short_stream_against_independent_optimizer():
    model = OnlineLogisticDOMD(2, .9, eta=2., l2=.2, radius=.4)
    theta, H = model.theta.copy(), model.H.copy()
    rng = np.random.default_rng(51)
    for _ in range(20):
        x = rng.normal(size=2)
        x /= np.linalg.norm(x)
        y = int(rng.integers(2))
        theta, H = reference_step(theta, H, x, y, .9, 2., .2, .4)
        model.update(x, y)
        np.testing.assert_allclose(model.theta, theta, atol=1e-5, rtol=0)
        np.testing.assert_allclose(model.H, H, atol=1e-5, rtol=0)


def test_gamma_one_matches_old_estimator():
    old = OnlineLogisticOMD(3, eta=3., l2=126., radius=2.)
    new = OnlineLogisticDOMD(3, 1., eta=3., l2=126., radius=2.)
    rng = np.random.default_rng(52)
    for _ in range(30):
        x = rng.normal(size=3)
        x /= np.linalg.norm(x)
        y = int(rng.integers(2))
        old.update(x, y)
        new.update(x, y)
        np.testing.assert_array_equal(old.theta, new.theta)
        np.testing.assert_array_equal(old.H, new.H)


def test_explicit_discounted_curvature_and_floor():
    model = OnlineLogisticDOMD(2, .8, eta=2., l2=.5, radius=2.)
    history = []
    for i in range(15):
        x = np.eye(2)[i % 2]
        model.update(x, i % 2)
        p = 1 / (1 + np.exp(-(x @ model.theta)))
        history.append(p * (1-p) * np.outer(x, x))
        expected = .5 * np.eye(2) + sum(.8**(i-j) * G for j, G in enumerate(history))
        np.testing.assert_allclose(model.H, expected, atol=1e-12, rtol=0)
    H = model.H.copy()
    theta = model.theta.copy()
    model.update(np.zeros(2), 0)
    np.testing.assert_allclose(model.theta, theta, rtol=0, atol=2e-10)
    np.testing.assert_allclose(model.H-.5*np.eye(2), .8*(H-.5*np.eye(2)))


def test_failed_update_preserves_state(monkeypatch):
    model = OnlineLogisticDOMD(2, .9, eta=3., l2=1.)
    model.update(np.array([1., 0.]), 1)
    theta, H, n = model.theta.copy(), model.H.copy(), model.completed_steps
    with pytest.raises(ValueError):
        model.update(np.array([1., 0.]), 2)
    def fail(*args, **kwargs):
        raise RuntimeError("injected solver failure")
    monkeypatch.setattr(domd_module, "omd_step", fail)
    with pytest.raises(RuntimeError, match="injected"):
        model.update(np.array([1., 0.]), 0)
    np.testing.assert_array_equal(model.theta, theta)
    np.testing.assert_array_equal(model.H, H)
    assert model.completed_steps == n


@pytest.mark.parametrize("gamma", [0., -0.1, 1.1, np.nan, True])
def test_invalid_gamma(gamma):
    with pytest.raises(ValueError):
        OnlineLogisticDOMD(2, gamma, eta=3., l2=1.)


def test_paper_policy_radius_and_selection():
    policy = LogisticDOMDPolicy(3, gamma=.98)
    model = policy.estimator
    assert model.eta == 3.
    assert model.l2 == pytest.approx(48 * 3 * 3 / 7)
    features = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    for n in range(10):
        effective = sum(.98**j for j in range(n))
        beta2 = (4*model.l2*4 + 30*np.log(np.pi**2*(n+1)**2/(3*.05))
                 + 2*3*9.5*3*np.log(1+.25*effective/(3*model.l2)))
        assert policy.confidence_radius()**2 == pytest.approx(beta2)
        scores = [x @ model.theta + np.sqrt(beta2 * (x @ np.linalg.solve(model.H, x))) for x in features]
        assert policy.select_action(features) == int(np.argmax(scores))
        policy.update(features[n % 3], n % 2)
    with pytest.raises(ValueError):
        LogisticDOMDPolicy(3, gamma=1.)
    with pytest.raises(ValueError):
        LogisticDOMDPolicy(3, l2=1.)
