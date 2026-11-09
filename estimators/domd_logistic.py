"""Discounted Logistic OMD, arXiv:2605.25590v1, Section 4.1."""
import numpy as np

from estimators.omd_logistic import OnlineLogisticOMD, omd_step


class OnlineLogisticDOMD(OnlineLogisticOMD):
    def __init__(self, dimension, gamma, eta, l2, radius=2.0):
        super().__init__(dimension=dimension, eta=eta, l2=l2, radius=radius)
        if (isinstance(gamma, (bool, np.bool_))
                or not isinstance(gamma, (int, float, np.integer, np.floating))
                or not np.isfinite(gamma) or not 0 < gamma <= 1):
            raise ValueError("gamma must be in (0, 1]; 1 is an estimator diagnostic")
        self.gamma = float(gamma)

    def update(self, x, y):
        # Forget historical curvature while preserving the lambda I floor.
        A = self.gamma * self.H + (1 - self.gamma) * self.l2 * np.eye(self.theta.size)

        # Existing solver forms B=A+eta*G_old, projects in the B metric,
        # then returns H_next=A+G_new (no eta multiplier on G_new).
        theta_next, H_next, info = omd_step(
            self.theta, A, x, y, self.eta, self.radius
        )
        rho, residual = info["rho"], info["kkt_residual"]
        norm = float(np.linalg.norm(theta_next))
        if (not np.isfinite(theta_next).all()
                or not np.isfinite(H_next).all()
                or not np.isfinite([rho, residual, norm]).all()):
            raise RuntimeError("DOMD produced a nonfinite state")
        if (rho < 0 or norm > self.radius + 1e-7 or residual > 1e-7
                or abs(rho * (norm**2 - self.radius**2)) > 1e-7):
            raise RuntimeError("DOMD update failed KKT checks")
        if not np.allclose(H_next, H_next.T, atol=1e-10, rtol=0):
            raise RuntimeError("DOMD curvature matrix is not symmetric")
        np.linalg.cholesky(H_next)

        # Failed validation leaves BOTH theta and H unchanged.
        self.theta, self.H = theta_next, H_next
        self.completed_steps += 1
        return info
