"""Fixed-arm adapters for the unified Logistic benchmark."""
import numpy as np


class MABAdapter:
    def __init__(self, policy):
        self.policy = policy

    def select_action(self):
        return self.policy.select_action()

    def update(self, action, reward):
        return self.policy.update(action, reward)

    def predict_means(self):
        # No comparable prediction interface is exposed for these baselines yet.
        return None


class LogisticOMDAdapter:
    def __init__(self, policy, features):
        features = np.array(features, dtype=float, copy=True)
        d = policy.estimator.theta.size
        if (
            features.ndim != 2
            or features.shape[0] == 0
            or features.shape[1] != d
            or not np.isfinite(features).all()
        ):
            raise ValueError("features must be a finite nonempty K-by-d matrix")
        if np.any(np.linalg.norm(features, axis=1) > 1.0 + 1e-10):
            raise ValueError("feature norms must not exceed 1")
        features.setflags(write=False)
        self.features = features
        self.policy = policy

    def select_action(self):
        return self.policy.select_action(self.features)

    def update(self, action, reward):
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise ValueError("action must be an integer")
        if not 0 <= action < len(self.features):
            raise IndexError("action is outside the arm set")
        return self.policy.update(self.features[action], reward)

    def predict_means(self):
        logits = self.features @ self.policy.estimator.theta
        return np.exp(-np.logaddexp(0.0, -logits))
