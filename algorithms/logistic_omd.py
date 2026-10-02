import numpy as np

from estimators.omd_logistic import OnlineLogisticOMD

def optimistic_scores(features, theta, H, beta):
    estimates = features @ theta

    # Solve all arm directions together: H @ directions = features.T.
    directions = np.linalg.solve(H, features.T)
    squared_widths = np.sum(features * directions.T, axis=1)

    if np.any(squared_widths < -1e-12):
        raise ValueError("Negative uncertainty quadratic form")

    widths = np.sqrt(np.maximum(squared_widths, 0.0))
    return estimates + beta * widths

class LogisticOMDPolicy:
    def __init__(
        self,
        dimension,
        radius=2.0,
        delta=0.05,
        beta_mode="paper_v2_literal",
    ):
        if not np.isfinite(delta) or not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        if beta_mode not in ("paper_v2_literal", "appendix_time"):
            raise ValueError("Unknown beta mode")

        eta = 1.0 + radius
        l2 = max(14.0 * dimension * eta, 1.5 * eta * radius)

        self.estimator = OnlineLogisticOMD(
            dimension=dimension,
            eta=eta,
            l2=l2,
            radius=radius,
        )
        self.delta = float(delta)
        self.beta_mode = beta_mode

    def confidence_radius(self):
        model = self.estimator
        d = model.theta.size

        # Preserve the printed v2 formula; label the alternative explicitly.
        time_factor = (
            1
            if self.beta_mode == "paper_v2_literal"
            else model.completed_steps
        )

        beta_squared = (
            4.0 * model.l2 * model.radius**2
            + 2.0 * model.eta * np.log(1.0 / self.delta)
            + d * (6.0 * model.eta**2 + model.eta)
            * np.log1p(0.25 * time_factor / model.l2)
        )
        return float(np.sqrt(beta_squared))

    def select_action(self, features):
        d = self.estimator.theta.size
        if (
            not isinstance(features, np.ndarray)
            or features.ndim != 2
            or features.shape[0] == 0
            or features.shape[1] != d
            or not np.issubdtype(features.dtype, np.floating)
            or not np.isfinite(features).all()
        ):
            raise ValueError("features must be a finite floating K-by-d array")

        if np.any(np.linalg.norm(features, axis=1) > 1.0 + 1e-10):
            raise ValueError("Arm feature norms must not exceed 1")

        scores = optimistic_scores(
            features,
            self.estimator.theta,
            self.estimator.H,
            self.confidence_radius(),
        )
        if not np.isfinite(scores).all():
            raise RuntimeError("Nonfinite optimistic scores")

        # Exact ties select the first arm deterministically.
        return int(np.argmax(scores))

    def update(self, x, y):
        return self.estimator.update(x, y)