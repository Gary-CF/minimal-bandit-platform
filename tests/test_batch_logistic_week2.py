"""E8 regression checks: independent numerical references and failure paths."""
import numpy as np
import pytest
from scipy.optimize import OptimizeResult, brentq

from estimators.batch_logistic import (
    batch_logistic_objective as objective,
    batch_logistic_gradient as gradient,
    batch_logistic_hessian as hessian,
    fit_batch_logistic,
)
from estimators.batch_logistic_checks import attach_fit_diagnostics, validate_fit_inputs


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nonzero_finite_differences(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(7, 3))
    y = rng.integers(0, 2, size=7)
    theta = rng.normal(size=3)
    l2, eps = 0.7, 1e-5
    basis = np.eye(3)
    numerical_g = np.array([
        (objective(theta + eps * e, X, y, l2)
         - objective(theta - eps * e, X, y, l2)) / (2 * eps)
        for e in basis
    ])
    numerical_H = np.column_stack([
        (gradient(theta + eps * e, X, y, l2)
         - gradient(theta - eps * e, X, y, l2)) / (2 * eps)
        for e in basis
    ])
    np.testing.assert_allclose(gradient(theta, X, y, l2), numerical_g, atol=1e-8)
    H = hessian(theta, X, l2)
    np.testing.assert_allclose(H, numerical_H, atol=1e-8)
    np.testing.assert_allclose(H, H.T, atol=1e-14)
    assert np.linalg.eigvalsh(H).min() >= l2 - 1e-12


def test_sum_and_regularization_once():
    X = np.array([[0.4, -0.2], [1.0, 0.5]])
    y, theta, l2 = np.array([1, 0]), np.array([0.3, -0.6]), 0.8
    XX, yy = np.tile(X, (2, 1)), np.tile(y, 2)
    np.testing.assert_allclose(objective(theta, XX, yy, l2),
                               2 * objective(theta, X, y, l2) - l2 / 2 * (theta @ theta))
    np.testing.assert_allclose(gradient(theta, XX, yy, l2),
                               2 * gradient(theta, X, y, l2) - l2 * theta)
    np.testing.assert_allclose(hessian(theta, XX, l2),
                               2 * hessian(theta, X, l2) - l2 * np.eye(2))


@pytest.mark.parametrize("label", [0, 1])
def test_extreme_logits_finite(label):
    X, theta, y = np.array([[1000.0], [-1000.0]]), np.ones(1), np.full(2, label)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert objective(theta, X, y, 1.0) == pytest.approx(1000.5)
        np.testing.assert_allclose(gradient(theta, X, y, 1.0), [1001.0])
        np.testing.assert_allclose(hessian(theta, X, 1.0), [[1.0]])


def test_hand_calculation():
    X, y, theta = np.eye(2), np.array([1, 0]), np.zeros(2)
    assert objective(theta, X, y, 1.0) == pytest.approx(2 * np.log(2))
    np.testing.assert_allclose(gradient(theta, X, y, 1.0), [-0.5, 0.5])
    np.testing.assert_allclose(hessian(theta, X, 1.0), 1.25 * np.eye(2))


def test_interior_independent_root():
    result = fit_batch_logistic(np.eye(2), np.array([1, 0]), ftol=1e-12)
    # For this separable objective, each positive coordinate solves a=1/(1+exp(a)).
    a = brentq(lambda v: v - 1 / (1 + np.exp(v)), 0, 1, xtol=1e-14)
    assert result.success and result.accepted
    np.testing.assert_allclose(result.x, [a, -a], atol=1e-6, rtol=0)
    assert result.solver_config["loss_reduction"] == "sum"


def test_boundary_analytic_solution_and_status():
    result = fit_batch_logistic(np.eye(2), np.array([1, 0]), radius=0.2, ftol=1e-12)
    expected = np.array([1, -1]) * 0.2 / np.sqrt(2)
    if result.success:
        assert result.accepted, result.diagnostics
        np.testing.assert_allclose(result.x, expected, atol=1e-7, rtol=0)
        assert result.diagnostics["gradient_norm"] > 0.1
        assert result.diagnostics["projected_gradient_residual"] <= 1e-5
    else:
        # SLSQP can report a failure near a boundary on some versions.
        # This is a verified rejection path, NOT acceptance of that fitted point.
        assert not result.accepted
        assert "solver_reported_failure" in result.diagnostics["failure_reasons"]


def test_iteration_limit_is_rejected():
    result = fit_batch_logistic(np.eye(2), np.array([1, 0]), maxiter=1)
    assert not result.success and not result.accepted
    assert "solver_reported_failure" in result.diagnostics["failure_reasons"]


@pytest.mark.parametrize("kind,reason", [
    ("nonoptimal", "first_order_residual_too_large"),
    ("infeasible", "constraint_violation"),
    ("wrong_objective", "objective_value_mismatch"),
    ("nan", "invalid_numerical_result"),
    ("raw_failure", "solver_reported_failure"),
])
def test_independent_diagnostics_reject_and_preserve_raw(kind, reason):
    X, y = np.eye(2), np.array([1, 0])
    a = brentq(lambda v: v - 1 / (1 + np.exp(v)), 0, 1)
    theta = np.array([a, -a])
    if kind == "nonoptimal":
        theta = np.zeros(2)
    if kind == "infeasible":
        theta = np.array([3.0, 0.0])
    if kind == "nan":
        theta = np.array([np.nan, 0.0])
    value = objective(theta, X, y, 1.0) if kind != "nan" else 0.0
    if kind == "wrong_objective":
        value += 1.0
    result = OptimizeResult(x=theta.copy(), fun=value,
                            success=kind != "raw_failure", status=7, message="sentinel")
    raw_success = result.success
    attach_fit_diagnostics(result, X, y, 1.0, 2.0, objective, gradient)
    assert not result.accepted
    assert reason in result.diagnostics["failure_reasons"]
    np.testing.assert_equal(result.x, theta)
    assert result.fun == value
    assert result.success == raw_success and result.status == 7 and result.message == "sentinel"


@pytest.mark.parametrize("kind", [
    "list_X", "integer_X", "empty_X", "nan_X", "complex_X",
    "column_y", "short_y", "bool_y", "nonbinary_y", "infinite_y",
])
def test_invalid_arrays(kind):
    X, y = np.eye(2), np.array([1, 0])
    if kind == "list_X": X = X.tolist()
    elif kind == "integer_X": X = X.astype(int)
    elif kind == "empty_X": X, y = np.empty((0, 2)), np.empty(0)
    elif kind == "nan_X": X[0, 0] = np.nan
    elif kind == "complex_X": X = X.astype(complex)
    elif kind == "column_y": y = y[:, None]
    elif kind == "short_y": y = y[:1]
    elif kind == "bool_y": y = y.astype(bool)
    elif kind == "nonbinary_y": y = np.array([1, 2])
    elif kind == "infinite_y": y = np.array([1.0, np.inf])
    with pytest.raises(ValueError):
        fit_batch_logistic(X, y)


@pytest.mark.parametrize("name,value", [
    ("l2", 0), ("l2", True), ("radius", -1), ("radius", np.inf),
    ("ftol", 0), ("ftol", np.nan), ("maxiter", 0), ("maxiter", 1.5),
])
def test_invalid_fit_options(name, value):
    with pytest.raises(ValueError):
        fit_batch_logistic(np.eye(2), np.array([1, 0]), **{name: value})


def test_validation_copies_to_float64():
    X, y = np.eye(2, dtype=np.float32), np.array([1, 0], dtype=np.int32)
    XX, yy = validate_fit_inputs(X, y, 1.0, 2.0, 1e-10, 500)
    assert XX.dtype == yy.dtype == np.dtype("float64")
    assert not np.shares_memory(X, XX) and not np.shares_memory(y, yy)
    XX[0, 0], yy[0] = 9, 9
    assert X[0, 0] == y[0] == 1


def test_readonly_inputs_are_unchanged():
    X, y = np.eye(2), np.array([1.0, 0.0])
    X.setflags(write=False)
    y.setflags(write=False)
    result = fit_batch_logistic(X, y, ftol=1e-12)
    assert result.accepted
    np.testing.assert_array_equal(X, np.eye(2))
    np.testing.assert_array_equal(y, [1, 0])
