"""Validation and independent diagnostics for the E-D8 batch reference.

Save as estimators/batch_logistic_checks.py. This module does not solve the
optimization problem and never changes a solver's x/success/status/message.
"""
from __future__ import annotations

import numpy as np


def _positive_real(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))):
        raise ValueError(f"{name} must be a positive finite real scalar")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must fit in float64") from exc
    if not np.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be a positive finite real scalar")
    return converted


def validate_fit_inputs(X, y, l2, radius, ftol, maxiter):
    """Return independent float64 arrays after validating the public fit input.

    X must have a real floating dtype. Labels may have integer or floating
    dtype, but not bool, complex, or object dtype. Inputs are never mutated.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.size == 0:
        raise ValueError("X must be a nonempty 2D numpy array")
    if not np.issubdtype(X.dtype, np.floating):
        raise ValueError("X must have a real floating dtype")
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be a 1D numpy array with one label per row of X")
    if not (np.issubdtype(y.dtype, np.floating) or np.issubdtype(y.dtype, np.integer)):
        raise ValueError("y must have an integer or real floating dtype")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must contain only finite values")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be 0 or 1")
    _positive_real(l2, "l2")
    radius = _positive_real(radius, "radius")
    _positive_real(ftol, "ftol")
    # The caller represents the ball by radius**2 - theta @ theta.
    if radius > np.sqrt(np.finfo(float).max):
        raise ValueError("radius**2 must fit in float64")
    if radius * radius == 0:
        raise ValueError("radius**2 must be representable above zero")
    if (isinstance(maxiter, (bool, np.bool_))
            or not isinstance(maxiter, (int, np.integer)) or maxiter <= 0):
        raise ValueError("maxiter must be a positive integer")
    with np.errstate(over="ignore", invalid="ignore"):
        X64, y64 = X.astype(np.float64, copy=True), y.astype(np.float64, copy=True)
    if not np.isfinite(X64).all() or not np.isfinite(y64).all():
        raise ValueError("X and y must be representable as finite float64 values")
    return X64, y64


def _project_ball(vector, radius):
    """Euclidean projection with scaling to avoid squaring large entries."""
    scale = float(np.max(np.abs(vector)))
    if scale == 0:
        return vector.copy()
    unit = vector / scale
    length = float(np.linalg.norm(unit))
    if scale <= radius / length:
        return vector.copy()
    return (radius / length) * unit


def attach_fit_diagnostics(
    result, X, y, l2, radius, objective, gradient,
    feasibility_tol=1e-7, optimality_tol=1e-5,
):
    """Attach accepted/diagnostics while preserving the raw solver result.

    For a differentiable convex objective on a Euclidean ball, exact
    first-order optimality is theta = projection(theta - gradient(theta)).
    We check its absolute L2 residual with a fixed step of 1. Numerical
    thresholds are engineering checks, not a bound on parameter error.
    """
    feasibility_tol = _positive_real(feasibility_tol, "feasibility_tol")
    optimality_tol = _positive_real(optimality_tol, "optimality_tol")
    reasons = []
    if not bool(result.get("success", False)):
        reasons.append("solver_reported_failure")
    checks = {
        "feasibility_tol": feasibility_tol,
        "optimality_tol": optimality_tol,
        "projected_gradient_step": 1.0,
        "theta_norm": None,
        "constraint_violation": None,
        "gradient_norm": None,
        "projected_gradient_residual": None,
        "objective_recomputed": None,
        "objective_consistency_error": None,
        "objective_consistency_tolerance": None,
        "diagnostic_error": None,
    }
    try:
        theta = np.asarray(result["x"])
        if (theta.shape != (X.shape[1],) or not np.isrealobj(theta)
                or not np.isfinite(theta).all()):
            raise ValueError("solver returned an invalid parameter vector")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            theta = theta.astype(float, copy=True)
            value = float(objective(theta, X, y, l2))
            reported_value = float(result["fun"])
            grad = np.asarray(gradient(theta, X, y, l2))
            if (grad.shape != theta.shape or not np.isrealobj(grad)
                    or not np.isfinite(grad).all()):
                raise ValueError("objective gradient is invalid")
            theta_norm = float(np.linalg.norm(theta))
            grad_norm = float(np.linalg.norm(grad))
            residual = float(np.linalg.norm(theta - _project_ball(theta - grad, radius)))
            numbers = (value, reported_value, theta_norm, grad_norm, residual)
            if not all(np.isfinite(v) for v in numbers):
                raise ValueError("nonfinite objective or diagnostic value")
            violation = max(0.0, theta_norm - radius)
            consistency_error = abs(value - reported_value)
            consistency_tol = 1e-10 * max(1.0, abs(value))
        checks.update(
            theta_norm=theta_norm, constraint_violation=violation,
            gradient_norm=grad_norm, projected_gradient_residual=residual,
            objective_recomputed=value, objective_consistency_error=consistency_error,
            objective_consistency_tolerance=consistency_tol,
        )
        if violation > feasibility_tol:
            reasons.append("constraint_violation")
        if residual > optimality_tol:
            reasons.append("first_order_residual_too_large")
        if consistency_error > consistency_tol:
            reasons.append("objective_value_mismatch")
    except (ValueError, TypeError, KeyError, FloatingPointError, OverflowError) as exc:
        checks["diagnostic_error"] = f"{type(exc).__name__}: {exc}"
        reasons.append("invalid_numerical_result")
    checks["failure_reasons"] = reasons
    result["diagnostics"] = checks
    result["accepted"] = not reasons
    return result
