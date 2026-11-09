"""Predeclared supplemental tasks; no changes to the original frozen benchmark."""
import math

import numpy as np
from scipy.special import gammaln


def features(seed, k, d=3):
    x = np.random.default_rng(seed).normal(size=(k, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def parameter_path(scenario, horizon, angle=math.pi):
    """Rotation in a deterministic orthonormal plane, never normalize a zero chord."""
    if horizon < 2 or not np.isfinite(angle):
        raise ValueError("horizon >= 2 and finite angle required")
    before = np.array([1., -.5, .75])
    radius = np.linalg.norm(before)
    u = before / radius
    axis = np.eye(3)[int(np.argmin(np.abs(u)))]
    v = axis - (axis @ u) * u
    v /= np.linalg.norm(v)
    after = radius * (np.cos(angle) * u + np.sin(angle) * v)
    if scenario == "stationary":
        path = np.repeat(before[None, :], horizon, axis=0)
    elif scenario == "abrupt":
        path = np.repeat(before[None, :], horizon, axis=0)
        path[horizon // 2:] = after
    elif scenario == "drift":
        w = np.linspace(0, 1, horizon)[:, None]
        path = (1 - w) * before + w * after
    elif scenario == "rotation":
        phase = np.linspace(0, angle, horizon)[:, None]
        path = radius * (np.cos(phase) * u + np.sin(phase) * v)
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return path


def noise_spec(name):
    # Gaussian and t3 have the same population standard deviation, 0.5.
    # t1.5 has infinite variance: retain raw scale 0.5, moment order 1.25.
    if name == "gaussian":
        return dict(name=name, scale=.5, df=None, epsilon=1., nu=.5)
    if name == "student3":
        df, scale, epsilon = 3., .5 / np.sqrt(3.), 1.
    elif name == "student1_5":
        df, scale, epsilon = 1.5, .5, .25
    else:
        raise ValueError(f"unknown noise: {name}")
    p = 1 + epsilon
    log_moment = (p * np.log(scale) + p / 2 * np.log(df)
                  + gammaln((p + 1) / 2) + gammaln((df - p) / 2)
                  - .5 * np.log(np.pi) - gammaln(df / 2))
    return dict(name=name, scale=float(scale), df=df, epsilon=epsilon,
                nu=float(np.exp(log_moment / p)))


def noise_stream(name, seed, horizon):
    spec = noise_spec(name)
    reward_seed = np.random.SeedSequence(seed).spawn(2)[0]
    rng = np.random.default_rng(reward_seed)
    values = (rng.normal(size=horizon) if spec["df"] is None
              else rng.standard_t(spec["df"], size=horizon))
    return values * spec["scale"]
