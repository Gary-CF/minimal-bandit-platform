"""Wang et al., AISTATS 2023, Eqs. (8), (14)-(17).

Nonconvex projection: deterministic multistart SLSQP, local KKT checks.
This numerical implementation does NOT certify a global projection optimum.
"""
from copy import copy

import numpy as np
from scipy.optimize import minimize

from algorithms.logistic_adapters import LogisticOMDAdapter
from estimators.discounted_logistic_mle import DiscountedLogisticMLE, _terms


def _projection_terms(theta, model, gradient_at_hat):
    _, gradient, H = _terms(theta, model.X, model.y, model.weights, model.ridge)
    residual = gradient - gradient_at_hat  # g(theta) - g(theta_hat)
    v = np.linalg.solve(H, residual)
    z = model.X @ theta
    p = np.exp(-np.logaddexp(0.0, -z))
    q = np.exp(-np.logaddexp(0.0, z))
    correction = model.weights * p * q * (q - p) * (model.X @ v) ** 2
    jacobian = residual - 0.5 * model.X.T @ correction
    return float(0.5 * residual @ v), jacobian


def project_scb(model, radius):
    hat = model.theta
    if np.linalg.norm(hat) <= radius:
        return hat.copy(), dict(active=0, objective=0.0, kkt=0.0, accepted=0)

    _, gradient_at_hat, _ = _terms(
        hat, model.X, model.y, model.weights, model.ridge
    )
    radial = hat / np.linalg.norm(hat)
    radial_value = _projection_terms(
        radius * radial, model, gradient_at_hat
    )[0]
    scale = max(1.0, radial_value)

    def objective(u):
        value, jacobian = _projection_terms(
            radius * u, model, gradient_at_hat
        )
        return value / scale, radius * jacobian / scale

    # Radial rescaling is only an initial point for the original objective.
    starts = [
        radial, np.zeros_like(hat),
        *np.eye(hat.size), *(-np.eye(hat.size)),
    ]
    constraint = dict(
        type="ineq",
        fun=lambda u: 1.0 - u @ u,
        jac=lambda u: -2.0 * u,
    )
    accepted = []
    for start in starts:
        result = minimize(
            objective, start, jac=True, method="SLSQP",
            constraints=[constraint],
            options={"ftol": 1e-12, "maxiter": 500},
        )
        u = result.x
        if (not result.success or not np.isfinite(u).all()
                or np.linalg.norm(u) > 1 + 1e-7):
            continue

        # Remove only floating-point feasibility overshoot, then recheck KKT.
        u = u / max(1.0, float(np.linalg.norm(u)))
        theta = radius * u
        value, gradient = _projection_terms(theta, model, gradient_at_hat)
        rho = (
            max(0.0, float(-theta @ gradient / (radius * radius)))
            if np.linalg.norm(u) >= 1 - 1e-6 else 0.0
        )
        kkt = float(
            np.linalg.norm(gradient + rho * theta, ord=np.inf)
            / (1.0 + np.linalg.norm(gradient, ord=np.inf))
        )
        complementarity = (
            abs(rho * (theta @ theta - radius * radius)) / (1 + abs(value))
        )
        if (np.isfinite([value, kkt, complementarity]).all()
                and kkt <= 1e-6 and complementarity <= 1e-6
                and value <= radial_value + 1e-8 * scale):
            accepted.append((value, theta, kkt))

    if not accepted:
        raise RuntimeError("SCB projection failed local KKT/feasibility checks")
    value, theta, kkt = min(accepted, key=lambda item: item[0])
    return theta, dict(
        active=1, objective=value, kkt=kkt, accepted=len(accepted)
    )


class SCBWeightUCBPolicy:
    def __init__(
        self, dimension, horizon, gamma=0.98,
        radius=2.0, delta=0.05, l2=None,
    ):
        for name, value in (("dimension", dimension), ("horizon", horizon)):
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer)) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if horizon < 2:
            raise ValueError("horizon must be >= 2")
        for name, value in (("radius", radius), ("delta", delta)):
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"invalid {name}")
        if delta >= 1:
            raise ValueError("delta must be in (0, 1)")

        q = np.exp(-float(radius))
        self.c_mu = q / (1 + q) ** 2
        if self.c_mu == 0:
            raise ValueError("radius is too large for finite curvature constants")
        l2 = dimension * np.log(horizon) / self.c_mu if l2 is None else l2
        if (isinstance(l2, (bool, np.bool_))
                or not isinstance(l2, (int, float, np.integer, np.floating))
                or not np.isfinite(l2) or l2 <= 0):
            raise ValueError("l2 must be positive and finite")

        self.estimator = DiscountedLogisticMLE(
            dimension, gamma, float(l2) * self.c_mu
        )
        if self.estimator.gamma == 1:
            raise ValueError("SCB policy requires gamma < 1")
        self.l2 = float(l2)
        self.radius, self.delta = float(radius), float(delta)
        self.V = self.l2 * np.eye(dimension)
        self.theta = np.zeros(dimension)  # Projected estimate; MLE is separate.
        self.projection_info = dict(
            active=0, objective=0.0, kkt=0.0, accepted=0
        )

    def confidence_radius(self):
        model = self.estimator
        n, d = model.completed_steps, model.theta.size
        # AISTATS 2023 Eq. (16): squared discounts.
        effective_n = (
            -np.expm1(2 * n * np.log(model.gamma))
            / (1 - model.gamma ** 2)
        )
        root = np.sqrt(model.ridge)
        return float(
            root / 2
            + 2 / root * (np.log(1 / self.delta) + d * np.log(2))
            + d / root * np.log1p(
                0.25 * effective_n / (model.ridge * d)
            )
            + root * self.radius
        )

    def scores(self, features):
        X = np.asarray(features)
        if (X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != self.theta.size
                or X.dtype.kind not in "fiu" or not np.isfinite(X).all()
                or np.any(np.linalg.norm(X, axis=1) > 1 + 1e-10)):
            raise ValueError(
                "features must be finite K-by-d vectors with norms <= 1"
            )
        widths2 = np.sum(X * np.linalg.solve(self.V, X.T).T, axis=1)
        if np.any(widths2 < -1e-12):
            raise RuntimeError("Negative uncertainty quadratic form")
        prediction = np.exp(-np.logaddexp(0.0, -(X @ self.theta)))
        bonus = 2 * np.sqrt(1 + 2 * self.radius) * 0.25 / np.sqrt(self.c_mu)
        scores = (
            prediction
            + bonus * self.confidence_radius()
            * np.sqrt(np.maximum(0, widths2))
        )
        if not np.isfinite(scores).all():
            raise RuntimeError("Nonfinite SCB scores")
        return scores

    def select_action(self, features):
        return int(np.argmax(self.scores(features)))

    def update(self, x, y):
        x = np.asarray(x)
        if (x.shape != self.theta.shape or x.dtype.kind not in "fiu"
                or not np.isfinite(x).all() or np.linalg.norm(x) > 1 + 1e-10):
            raise ValueError("x must be a finite feature vector with norm <= 1")

        # MLE replaces arrays, so a candidate copy keeps this update atomic.
        candidate = copy(self.estimator)
        candidate.update(x, y)
        theta, info = project_scb(candidate, self.radius)
        V = (
            candidate.gamma * self.V + np.outer(x, x)
            + (1 - candidate.gamma) * self.l2 * np.eye(x.size)
        )
        np.linalg.cholesky(V)
        self.estimator = candidate
        self.theta, self.V, self.projection_info = theta, V, info
        return info

    def audit(self):
        info, mle = self.projection_info, self.estimator
        return dict(
            scb_projection_active_post=info["active"],
            scb_projection_objective_post=info["objective"],
            scb_projection_kkt_post=info["kkt"],
            scb_projection_accepted_post=info["accepted"],
            scb_mle_grad_inf_post=mle.last_diagnostics["grad_inf"],
            scb_mle_norm_post=float(np.linalg.norm(mle.theta)),
            scb_projected_norm_post=float(np.linalg.norm(self.theta)),
            scb_lambda=self.l2,
        )


class SCBWeightUCBAdapter(LogisticOMDAdapter):
    def predict_means(self):
        return np.exp(
            -np.logaddexp(0.0, -(self.features @ self.policy.theta))
        )
