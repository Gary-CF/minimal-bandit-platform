import numpy as np


def _validate_vectors(
    theta: np.ndarray,
    x: np.ndarray,
) -> None:
    for name, array in (("theta", theta), ("x", x)):
        if not isinstance(array, np.ndarray):
            raise ValueError(f"{name} must be a numpy.ndarray")

        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"{name} must be a nonempty 1D array")

        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"{name} must have a real floating dtype")

        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")

    if theta.shape != x.shape:
        raise ValueError("theta and x must have the same shape")


def _validate_label(y: float) -> None:
    if (
        isinstance(y, (bool, np.bool_))
        or not isinstance(y, (int, float, np.integer, np.floating))
    ):
        raise ValueError("y must be a real numeric scalar")

    if not np.isfinite(y) or y not in (0, 1):
        raise ValueError("y must be 0 or 1")

def _validate_environment_inputs(
    features: np.ndarray,
    theta_star: np.ndarray,
    rng: np.random.Generator,
) -> None:
    for name, array, ndim in (
        ("features", features, 2),
        ("theta_star", theta_star, 1),
    ):
        if not isinstance(array, np.ndarray):
            raise ValueError(f"{name} must be a numpy.ndarray")

        if array.ndim != ndim or array.size == 0:
            raise ValueError(
                f"{name} must be a nonempty {ndim}D array"
            )

        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"{name} must have a real floating dtype")

        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")

    if features.shape[1] != theta_star.shape[0]:
        raise ValueError(
            "features columns must match theta_star length"
        )

    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")


def logistic_loss(
    theta: np.ndarray,
    x: np.ndarray,
    y: float,
) -> float:
    _validate_vectors(theta, x)
    _validate_label(y)

    z=x@theta
    return float(np.logaddexp(0.0,z)-y*z)

def logistic_gradient(
    theta: np.ndarray,
    x: np.ndarray,
    y: float,
) -> np.ndarray:
    _validate_vectors(theta, x)
    _validate_label(y)

    z=x.T@theta
    p = np.exp(-np.logaddexp(0.0, -z))
    return (p-y)*x


def logistic_hessian(
    theta: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    _validate_vectors(theta, x)

    z=x.T@theta
    p = np.exp(-np.logaddexp(0.0, -z))
    return p*(1-p)*np.outer(x,x)

class LogisticBernoulliBandit:
    def __init__(
        self,
        features: np.ndarray,
        theta_star: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        _validate_environment_inputs(features, theta_star, rng)

        self.features=features.copy()
        self.theta_star=theta_star.copy()
        self.rng=rng
        self.num_arms=features.shape[0]
        self.dimension=features.shape[1]

        z=self.features@self.theta_star

        if not np.isfinite(z).all():
            raise ValueError("features @ theta_star must be finite")

        self.arm_means = np.exp(-np.logaddexp(0.0, -z))

    def step(self,action:int) ->int:
        self._validate_action(action)

        reward=self.rng.binomial(
            n=1,
            p=self.arm_means[action]
        )
        return int(reward)

    def pseudo_regret(self, action: int) -> float:
        self._validate_action(action)

        best_mean=float(np.max(self.arm_means))
        regret=best_mean-self.arm_means[action]
        return float(regret)

    def _validate_action(self, action: int) -> None:
        if (
        isinstance(action, (bool, np.bool_))
        or not isinstance(action, (int, np.integer))
    ):
            raise ValueError("action must be an integer")

        if not 0 <= action < self.num_arms:
            raise IndexError(
            f"action {action} is outside [0, {self.num_arms})"
        )
