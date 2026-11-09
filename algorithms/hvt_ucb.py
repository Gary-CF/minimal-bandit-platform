"""Hvt-UCB, Wang et al. ICML 2025 / arXiv:2503.00419v2, Alg. 1.

Literal constants from Lemma 1 / Theorem 1, not a tuned Huber heuristic.
Known central moment scale nu = E|noise|**(1+epsilon)**(1/(1+epsilon)).
The squared variant is an explicitly labelled geometry-matched ablation.
"""
import numpy as np

from estimators.omd_logistic import project_metric_ball


class HvtUCB:
    def __init__(self, dimension, horizon, nu, epsilon=1., radius=2., squared=False):
        if (isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer))
                or dimension < 1 or isinstance(horizon, bool)
                or not isinstance(horizon, (int, np.integer)) or horizon < 2):
            raise ValueError("positive integer dimension and horizon >= 2 required")
        if not np.isfinite([nu, epsilon, radius]).all() or nu <= 0 or not 0 < epsilon <= 1 or radius <= 0:
            raise ValueError("invalid moment scale, epsilon or radius")
        self.d, self.T = int(dimension), int(horizon)
        self.nu, self.epsilon, self.radius = float(nu), float(epsilon), float(radius)
        self.squared = bool(squared)
        self.alpha, self.l2 = 4., float(dimension)
        self.delta, self.sigma_min = 1 / (8 * horizon), 1 / np.sqrt(horizon)
        self.power = (1 - epsilon) / (2 * (1 + epsilon))
        self.log_confidence = np.log(2 * horizon ** 2 / self.delta)
        kappa = dimension * np.log1p(horizon / (4 * self.sigma_min ** 2 * self.l2 * dimension))
        self.tau0 = (np.sqrt(2 * kappa) * np.log(3 * horizon) ** self.power
                     / self.log_confidence ** (1 / (1 + epsilon)))
        self.theta = np.zeros(dimension)
        self.V = self.l2 * np.eye(dimension)
        self.completed_steps = 0
        self.last_diagnostics = None

    def confidence_radius(self, n=None):
        n = self.completed_steps if n is None else n
        # At epsilon=1 the time factor is identically one, including n=0.
        # For epsilon<1 use 0**positive = 0 in the literal Eq. (10).
        factor = 1. if self.power == 0 else float(n) ** self.power
        return float(107 * self.log_confidence * self.tau0 * factor
                     + np.sqrt(self.l2 * (2 + 4 * self.radius ** 2)))

    def scores(self, X):
        X = np.asarray(X, dtype=float)
        if (X.ndim != 2 or X.shape[1] != self.d or not len(X)
                or not np.isfinite(X).all() or np.any(np.linalg.norm(X, axis=1) > 1 + 1e-10)):
            raise ValueError("finite features with norm <= 1 required")
        widths = np.sqrt(np.maximum(0., np.sum(X * np.linalg.solve(self.V, X.T).T, axis=1)))
        return X @ self.theta + self.confidence_radius() * widths

    def select_action(self, X):
        return int(np.argmax(self.scores(X)))

    def update(self, x, reward):
        x = np.asarray(x, dtype=float)
        if (x.shape != (self.d,) or not np.isfinite(x).all()
                or np.linalg.norm(x) > 1 + 1e-10 or not np.isfinite(reward)):
            raise ValueError("invalid feature/reward")
        t = self.completed_steps + 1
        if t > self.T:
            raise ValueError("declared horizon exhausted")
        beta = self.confidence_radius()
        width = np.sqrt(max(0., float(x @ np.linalg.solve(self.V, x))))
        sigma = max(self.nu, self.sigma_min,
                    np.sqrt(2 * beta / (self.tau0 * np.sqrt(self.alpha) * t ** self.power)) * width)
        w = width / (np.sqrt(self.alpha) * sigma)
        # Zero feature: loss has zero gradient; tau=infinity is its limiting value.
        tau = np.inf if w == 0 else self.tau0 * np.hypot(1., w) / w * t ** self.power
        z = (float(reward) - x @ self.theta) / sigma
        psi = z if self.squared else np.clip(z, -tau, tau)
        gradient = -psi * x / sigma
        V = self.V + np.outer(x, x) / (self.alpha * sigma ** 2)
        unconstrained = self.theta - np.linalg.solve(V, gradient)
        theta, rho = project_metric_ball(unconstrained, V, self.radius)
        residual = float(np.linalg.norm(V @ (theta - unconstrained) + rho * theta))
        if (not np.isfinite(theta).all() or np.linalg.norm(theta) > self.radius + 1e-7
                or residual > 1e-7 or abs(rho * (theta @ theta - self.radius ** 2)) > 1e-7):
            raise RuntimeError("Hvt metric projection failed KKT/feasibility")
        np.linalg.cholesky(V)
        self.theta, self.V, self.completed_steps = theta, V, t
        self.last_diagnostics = dict(sigma=float(sigma), tau=None if w == 0 else float(tau),
                                     beta_pre=beta, normalized_residual=float(z),
                                     clipped=int(not self.squared and abs(z) > tau),
                                     projection_active=int(rho > 0), projection_kkt=residual)
        return self.last_diagnostics


class RidgeLinearPolicy:
    """Recursive ridge / LinUCB-style comparator; no heavy-tail guarantee claimed."""
    def __init__(self, dimension, radius=2., noise_scale=.5, greedy=False, delta=.05):
        self.V = np.eye(dimension)
        self.b = np.zeros(dimension)
        self.theta = np.zeros(dimension)
        self.radius, self.noise_scale, self.delta = radius, noise_scale, delta
        self.greedy, self.completed_steps = greedy, 0

    def select_action(self, X):
        if self.greedy:
            return int(np.argmax(X @ self.theta))
        # Gaussian-calibrated baseline; the same fixed nominal scale is used on t noise.
        beta = self.radius + self.noise_scale * np.sqrt(
            np.linalg.slogdet(self.V)[1] + 2 * np.log(1 / self.delta))
        widths = np.sqrt(np.maximum(0., np.sum(X * np.linalg.solve(self.V, X.T).T, axis=1)))
        return int(np.argmax(X @ self.theta + beta * widths))

    def update(self, x, reward):
        self.V = self.V + np.outer(x, x)
        self.b = self.b + reward * x
        self.theta = np.linalg.solve(self.V, self.b)
        self.completed_steps += 1
