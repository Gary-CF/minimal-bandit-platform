import numpy as np

from envs.logistic_bandit import (
    logistic_gradient,
    logistic_hessian,
)


def omd_local_model(theta, H, x, y, eta):
    g = logistic_gradient(theta, x, y)
    G = logistic_hessian(theta, x)
    A = H + eta * G

    # Solve A @ direction = g without forming the inverse.
    direction = np.linalg.solve(A, g)
    u = theta - eta * direction

    return g, G, A, u

def metric_ball_candidate(u, A, rho):
    identity = np.eye(u.size)
    matrix = A + rho * identity
    rhs = A @ u
    return np.linalg.solve(matrix, rhs)

def project_metric_ball(u, A, radius, tol=1e-10, maxiter=100):
    if np.linalg.norm(u) <= radius:
        return u.copy(), 0.0

    # Find an interval containing the multiplier.
    low, high = 0.0, 1.0
    for _ in range(maxiter):
        if np.linalg.norm(metric_ball_candidate(u, A, high)) <= radius:
            break
        high *= 2.0
    else:
        raise RuntimeError("Failed to bracket the projection multiplier")

    # Solve ||theta(rho)|| = radius by bisection.
    for _ in range(maxiter):
        rho = (low + high) / 2.0
        theta = metric_ball_candidate(u, A, rho)
        gap = np.linalg.norm(theta) - radius

        if abs(gap) <= tol:
            return theta, rho

        if gap > 0:
            low = rho
        else:
            high = rho

    raise RuntimeError("Metric projection did not converge")

def omd_step(theta, H, x, y, eta, radius):
    g, G, A, u = omd_local_model(theta, H, x, y, eta)
    theta_next, rho = project_metric_ball(u, A, radius)

    # Accumulate curvature evaluated at the updated parameter.
    G_next = logistic_hessian(theta_next, x)
    H_next = H + G_next

    diagnostics = {
        "rho": rho,
        "theta_norm": float(np.linalg.norm(theta_next)),
        "constraint_violation": max(
            0.0, float(np.linalg.norm(theta_next)) - radius
        ),
        "kkt_residual": float(
            np.linalg.norm(A @ (theta_next - u) + rho * theta_next)
        ),
    }
    return theta_next, H_next, diagnostics

class OnlineLogisticOMD:
    def __init__(self, dimension, eta, l2=1.0, radius=2.0):
        if (
            isinstance(dimension, (bool, np.bool_))
            or not isinstance(dimension, (int, np.integer))
            or dimension <= 0
        ):
            raise ValueError("dimension must be a positive integer")

        parameters = {}
        for name, value in (("eta", eta), ("l2", l2), ("radius", radius)):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite scalar")
            parameters[name] = float(value)

        self.eta = parameters["eta"]
        self.l2 = parameters["l2"]
        self.radius = parameters["radius"]

        self.theta = np.zeros(dimension)
        self.H = self.l2 * np.eye(dimension)
        self.completed_steps = 0

    def update(self, x, y):
        theta_next, H_next, info = omd_step(
            self.theta, self.H, x, y, self.eta, self.radius
        )

        if (
            not np.isfinite(theta_next).all()
            or not np.isfinite(H_next).all()
        ):
            raise RuntimeError("OMD produced a nonfinite state")

        rho = info["rho"]
        residual = info["kkt_residual"]
        norm = float(np.linalg.norm(theta_next))

        if not np.isfinite([rho, residual, norm]).all():
            raise RuntimeError("OMD produced invalid diagnostics")

        if (
            rho < 0
            or norm > self.radius + 1e-7
            or residual > 1e-7
            or abs(rho * (norm**2 - self.radius**2)) > 1e-7
        ):
            raise RuntimeError("OMD update failed the KKT checks")

        if not np.allclose(H_next, H_next.T, atol=1e-10, rtol=0):
            raise RuntimeError("OMD curvature matrix is not symmetric")

        # Cholesky fails if the candidate matrix is not positive definite.
        np.linalg.cholesky(H_next)

        # Commit the new state only after all checks pass.
        self.theta = theta_next
        self.H = H_next
        self.completed_steps += 1

        return info