"""Unconstrained discounted MLE, Wang et al. (AISTATS 2023), Eq. (8).

ridge denotes lambda * c_mu. This is an estimator, not the SCB policy.
"""
import numpy as np


def _terms(theta, X, y, weights, ridge):
    z = X @ theta
    p = np.exp(-np.logaddexp(0.0, -z))
    q = np.exp(-np.logaddexp(0.0, z))
    loss = np.logaddexp(0.0, (1.0 - 2.0 * y) * z)
    value = float(weights @ loss + 0.5 * ridge * (theta @ theta))
    residual = np.where(y == 1, -q, p)
    gradient = X.T @ (weights * residual) + ridge * theta
    hessian = X.T @ ((weights * p * q)[:, None] * X)
    hessian += ridge * np.eye(theta.size)
    return value, gradient, hessian


def _solve(X, y, weights, ridge, start, tol, max_iter):
    theta = start.copy()
    for iteration in range(max_iter + 1):
        value, gradient, hessian = _terms(theta, X, y, weights, ridge)
        if not (np.isfinite(value) and np.isfinite(gradient).all()
                and np.isfinite(hessian).all()):
            raise RuntimeError("Nonfinite MLE objective or derivatives")
        grad_inf = float(np.linalg.norm(gradient, ord=np.inf))
        if grad_inf <= tol:
            np.linalg.cholesky(hessian)
            return theta, hessian, dict(
                objective=value, grad_inf=grad_inf, newton_steps=iteration,
                parameter_error_bound=float(np.linalg.norm(gradient) / ridge))
        if iteration == max_iter:
            break
        direction = np.linalg.solve(hessian, gradient)
        decrement = float(gradient @ direction)
        if not np.isfinite(decrement) or decrement <= 0:
            raise RuntimeError("Invalid Newton direction")
        step = 1.0
        for _ in range(60):
            candidate = theta - step * direction
            new_value, new_gradient, _ = _terms(candidate, X, y, weights, ridge)
            if (np.isfinite(new_value) and np.isfinite(new_gradient).all()
                    and (new_value <= value - 1e-4 * step * decrement
                         or np.linalg.norm(new_gradient, ord=np.inf) <= tol)):
                theta = candidate
                break
            step *= 0.5
        else:
            raise RuntimeError("MLE line search failed")
    raise RuntimeError(f"MLE did not converge: gradient infinity norm={grad_inf:g}")


class DiscountedLogisticMLE:
    def __init__(self, dimension, gamma, ridge, *, tol=1e-9, max_iter=100):
        for name, value in (("dimension", dimension), ("max_iter", max_iter)):
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer)) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("gamma", gamma), ("ridge", ridge), ("tol", tol)):
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a positive finite scalar")
        if gamma > 1:
            raise ValueError("gamma must be in (0, 1]")
        self.gamma, self.ridge, self.tol = float(gamma), float(ridge), float(tol)
        self.max_iter = int(max_iter)
        self.theta = np.zeros(dimension)
        self.H = self.ridge * np.eye(dimension)
        self.X = np.empty((0, dimension))
        self.y = np.empty(0)
        self.completed_steps = 0
        self.last_diagnostics = None

    @property
    def weights(self):
        return self.gamma ** np.arange(self.completed_steps - 1, -1, -1)

    def update(self, x, y):
        x = np.asarray(x)
        if (x.shape != self.theta.shape or x.dtype.kind not in "fiu"
                or not np.isfinite(x).all()):
            raise ValueError("x must be a finite real vector of the declared dimension")
        if (isinstance(y, (bool, np.bool_))
                or not isinstance(y, (int, float, np.integer, np.floating))
                or y not in (0, 1)):
            raise ValueError("y must be 0 or 1")
        X_next = np.vstack((self.X, x.astype(float)))
        y_next = np.append(self.y, float(y))
        weights = self.gamma ** np.arange(len(y_next) - 1, -1, -1)
        theta, H, info = _solve(X_next, y_next, weights, self.ridge,
                               self.theta, self.tol, self.max_iter)
        # Commit only after a successful solve; retain all historical samples.
        self.X, self.y, self.theta, self.H = X_next, y_next, theta, H
        self.completed_steps += 1
        self.last_diagnostics = info
        return info
