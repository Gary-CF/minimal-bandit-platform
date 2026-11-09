"""Logistic DOMD-GLB: paper confidence radius and action rule."""
import numpy as np

from algorithms.logistic_omd import LogisticOMDPolicy
from estimators.domd_logistic import OnlineLogisticDOMD


class LogisticDOMDPolicy(LogisticOMDPolicy):
    def __init__(self, dimension, gamma=0.99, radius=2.0, delta=0.05, l2=None):
        # Reuse validated feature selection/update interfaces from OMD.
        super().__init__(dimension, radius=radius, delta=delta)
        eta = self.estimator.eta
        radius = self.estimator.radius
        q = np.exp(-radius)
        c_mu = q / (1 + q)**2
        lambda_min = max(1.5 * eta * radius, 48 * eta * dimension / 7, c_mu)
        self.estimator = OnlineLogisticDOMD(
            dimension, gamma, eta, lambda_min if l2 is None else l2, radius
        )
        if self.estimator.gamma == 1:
            raise ValueError("paper policy requires gamma < 1")
        if self.estimator.l2 < lambda_min:
            raise ValueError("l2 is below the paper's required bound")
        self.beta_mode = "domd_paper_v1"

    def confidence_radius(self):
        model = self.estimator
        n = model.completed_steps
        t = n + 1
        d = model.theta.size
        effective_n = -np.expm1(n * np.log(model.gamma)) / (1 - model.gamma)
        # R=1, g(tau)=1, k_mu=1/4. At t=1 use the same numeric expression;
        # the paper's confidence statement itself begins at t>=2.
        beta_squared = (
            4 * model.l2 * model.radius**2
            + 10 * model.eta * (2 * np.log(np.pi) + 2 * np.log(t) - np.log(3 * self.delta))
            + 2 * model.eta * (3 * model.eta + 0.5) * d
            * np.log1p(0.25 * effective_n / (model.l2 * d))
        )
        return float(np.sqrt(beta_squared))
