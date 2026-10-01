import numpy as np

from scipy.optimize import OptimizeResult, minimize

from estimators.batch_logistic_checks import (
    validate_fit_inputs,
    attach_fit_diagnostics,
)

def batch_logistic_objective(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    l2: float,
) -> float:
    z = X @ theta

    # y=0: log(1+exp(z))
    # y=1: log(1+exp(-z))
    losses = np.logaddexp(0.0, (1.0 - 2.0 * y) * z)
    regularization = 0.5 * l2 * (theta @ theta)

    return float(losses.sum() + regularization)


def batch_logistic_gradient(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    l2: float,
) -> np.ndarray:
    z = X @ theta

    # 等价于 sigmoid(z)-y，避免 y=1、z 很大时的相减消去。
    signs = 1.0 - 2.0 * y
    residual = signs * np.exp(
        -np.logaddexp(0.0, -signs * z)
    )

    return X.T @ residual + l2 * theta


def batch_logistic_hessian(
    theta: np.ndarray,
    X: np.ndarray,
    l2: float,
) -> np.ndarray:
    z = X @ theta

    # sigmoid(z)(1-sigmoid(z)) 关于 z 对称。
    # 用较小的一侧概率计算，避免大正数时 sigmoid(z) 舍入为 1。
    q = np.exp(-np.logaddexp(0.0, np.abs(z)))
    weights = q * (1.0 - q)

    dimension = theta.size
    return (
        X.T @ (weights[:, None] * X)
        + l2 * np.eye(dimension)
    )


def fit_batch_logistic(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1.0,
    radius: float = 2.0,
    ftol: float = 1e-10,
    maxiter: int = 500,
) -> OptimizeResult:
    X, y = validate_fit_inputs(
    X, y, l2, radius, ftol, maxiter
    )

    dimension = X.shape[1]
    initial_theta = np.zeros(dimension, dtype=float)

    constraint = {
        "type": "ineq",
        "fun": lambda theta: radius**2-theta @ theta,  # c(theta)
        "jac": lambda theta: -2.0 * theta,  # 梯度：形状 (d,)
    }

    result = minimize(
        fun=batch_logistic_objective,
        x0=initial_theta,
        args=(X, y, l2),
        jac=batch_logistic_gradient,                  # 传入梯度函数本身，不在这里调用
        method="SLSQP",
        constraints=[constraint],
        options={
            "ftol": ftol,
            "maxiter": maxiter,
            "disp": False,
        },
    )

    # 把实际使用的配置随求解结果保留。
    result["solver_config"] = {
        "method": "SLSQP",
        "ftol": ftol,
        "maxiter": maxiter,
        "l2": l2,
        "radius": radius,
        "loss_reduction": "sum",
    }

    return attach_fit_diagnostics(
    result=result,
    X=X,
    y=y,
    l2=l2,
    radius=radius,
    objective=batch_logistic_objective,
    gradient=batch_logistic_gradient,
)