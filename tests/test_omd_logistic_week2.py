import numpy as np
import pytest
from scipy.optimize import minimize

from estimators.omd_logistic import omd_step


@pytest.mark.parametrize("radius", [2.0, 0.2])
def test_omd_step_against_independent_solver(radius):
    theta = np.array([0.1, -0.1])
    H = np.array([[2.0, 0.3], [0.3, 1.0]])
    x = np.array([1.0, -0.5])
    y, eta = 1.0, 2.0

    theta_before, H_before = theta.copy(), H.copy()

    # Build the original quadratic objective independently.
    p = 1.0 / (1.0 + np.exp(-(x @ theta)))
    g = (p - y) * x
    G = p * (1.0 - p) * np.outer(x, x)

    def objective(candidate):
        delta = candidate - theta
        return (
            g @ delta
            + 0.5 * delta @ G @ delta
            + delta @ H @ delta / (2.0 * eta)
        )

    def gradient(candidate):
        delta = candidate - theta
        return g + G @ delta + H @ delta / eta

    reference = minimize(
        objective,
        theta.copy(),
        jac=gradient,
        method="SLSQP",
        constraints=[{
            "type": "ineq",
            "fun": lambda candidate: radius**2 - candidate @ candidate,
            "jac": lambda candidate: -2.0 * candidate,
        }],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    assert reference.success, reference.message
    assert np.linalg.norm(reference.x) <= radius + 1e-7

    theta_next, H_next, info = omd_step(
        theta, H, x, y, eta, radius
    )

    np.testing.assert_allclose(
        theta_next, reference.x, atol=2e-6, rtol=0
    )
    assert abs(objective(theta_next) - reference.fun) <= 1e-8

    # Check the projection KKT conditions.
    assert info["rho"] >= 0
    assert info["constraint_violation"] <= 1e-7
    assert info["kkt_residual"] <= 1e-8
    assert abs(
        info["rho"] * (theta_next @ theta_next - radius**2)
    ) <= 1e-8

    if radius == 2.0:
        assert info["rho"] == 0.0
        assert np.linalg.norm(theta_next) < radius
    else:
        assert info["rho"] > 0.0
        assert abs(np.linalg.norm(theta_next) - radius) <= 1e-9

    # Check curvature at the NEW parameter, without an eta multiplier.
    p_next = 1.0 / (1.0 + np.exp(-(x @ theta_next)))
    expected_H = H + p_next * (1.0 - p_next) * np.outer(x, x)
    np.testing.assert_allclose(H_next, expected_H, atol=1e-12, rtol=0)
    np.testing.assert_allclose(H_next, H_next.T, atol=1e-12, rtol=0)
    assert np.linalg.eigvalsh(H_next).min() > 0

    np.testing.assert_array_equal(theta, theta_before)
    np.testing.assert_array_equal(H, H_before)


def test_online_state_and_fixed_storage():
    from estimators.omd_logistic import OnlineLogisticOMD

    model = OnlineLogisticOMD(dimension=1, eta=2.0)
    state_keys = set(vars(model))

    model.update(np.array([1.0]), 1.0)

    # Independent first-step answer.
    p = 1.0 / (1.0 + np.exp(-2.0 / 3.0))
    np.testing.assert_allclose(model.theta, [2.0 / 3.0])
    np.testing.assert_allclose(model.H, [[1.0 + p * (1.0 - p)]])
    assert model.completed_steps == 1

    for i in range(20):
        model.update(np.array([1.0]), float(i % 2))

    assert model.completed_steps == 21
    assert model.theta.shape == (1,)
    assert model.H.shape == (1, 1)
    assert set(vars(model)) == state_keys
    assert all(
        isinstance(value, (int, float, np.ndarray))
        for value in vars(model).values()
    )
    arrays = [
        value for value in vars(model).values()
        if isinstance(value, np.ndarray)
    ]
    assert len(arrays) == 2


def test_failed_update_preserves_state(monkeypatch):
    import estimators.omd_logistic as module

    model = module.OnlineLogisticOMD(dimension=1, eta=2.0)
    model.update(np.array([1.0]), 1.0)

    theta = model.theta.copy()
    H = model.H.copy()
    steps = model.completed_steps

    # Invalid observed label.
    with pytest.raises(ValueError):
        model.update(np.array([1.0]), 2.0)

    np.testing.assert_array_equal(model.theta, theta)
    np.testing.assert_array_equal(model.H, H)
    assert model.completed_steps == steps

    # A numerical solver failure must also preserve the old state.
    def fail_projection(*args, **kwargs):
        raise RuntimeError("simulated projection failure")

    monkeypatch.setattr(module, "project_metric_ball", fail_projection)
    with pytest.raises(RuntimeError, match="simulated projection failure"):
        model.update(np.array([1.0]), 0.0)

    np.testing.assert_array_equal(model.theta, theta)
    np.testing.assert_array_equal(model.H, H)
    assert model.completed_steps == steps
