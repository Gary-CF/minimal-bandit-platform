"""Unified Logistic episode; policies receive no environment truth."""
import numpy as np
from time import perf_counter

from algorithms.base import RandomPolicy
from algorithms.ucb1 import UCB1
from algorithms.sliding_window_ucb import SlidingWindowUCB
from algorithms.discounted_ucb import DiscountedUCB
from algorithms.logistic_omd import LogisticOMDPolicy
from algorithms.logistic_domd import LogisticDOMDPolicy
from algorithms.scb_weight_ucb import SCBWeightUCBPolicy, SCBWeightUCBAdapter
from algorithms.logistic_adapters import MABAdapter, LogisticOMDAdapter
from envs.logistic_bandit import LogisticBernoulliBandit
from scripts.d12_logistic_stress import parameter_path


ALGORITHMS = ("random", "ucb1", "sw_ucb", "discounted_ucb", "omd_paper_v2_literal", "domd_paper_v1", "scb_weight_ucb_numeric_v1")


def make_agent(name, features, rng, *, window_size=100, gamma=0.99,
               radius=2.0, delta=0.05, horizon=None):
    k, d = features.shape
    if name == "random":
        return MABAdapter(RandomPolicy(k, rng))
    if name == "ucb1":
        return MABAdapter(UCB1(k))
    if name == "sw_ucb":
        return MABAdapter(SlidingWindowUCB(k, window_size))
    if name == "discounted_ucb":
        return MABAdapter(DiscountedUCB(k, gamma))
    if name == "omd_paper_v2_literal":
        policy = LogisticOMDPolicy(d, radius=radius, delta=delta,
                                   beta_mode="paper_v2_literal")
        return LogisticOMDAdapter(policy, features)
    if name == "domd_paper_v1":
        policy = LogisticDOMDPolicy(d, gamma=gamma, radius=radius, delta=delta)
        return LogisticOMDAdapter(policy, features)
    if name == "scb_weight_ucb_numeric_v1":
        policy = SCBWeightUCBPolicy(d, horizon, gamma=gamma, radius=radius, delta=delta)
        return SCBWeightUCBAdapter(policy, features)
    raise ValueError(f"unknown algorithm: {name}")


def make_path(scenario, horizon, before, after, change_step=None):
    if scenario == "stationary":
        return parameter_path("drift", horizon, before, before, None)
    if scenario == "abrupt" and change_step is None:
        change_step = horizon // 2 + 1
    return parameter_path(scenario, horizon, before, after, change_step)


def run_episode(name, features, path, seed, *, window_size=100,
                gamma=0.99, radius=2.0, delta=0.05,
                on_step=None, collect_rows=True):
    features = np.array(features, dtype=float, copy=True)
    path = np.array(path, dtype=float, copy=True)
    if (features.ndim != 2 or features.size == 0
            or path.ndim != 2 or path.size == 0
            or features.shape[1] != path.shape[1]
            or not np.isfinite(features).all() or not np.isfinite(path).all()):
        raise ValueError("features and path must be finite nonempty K-by-d / T-by-d arrays")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    if np.any(np.linalg.norm(features, axis=1) > 1 + 1e-10):
        raise ValueError("feature norms must not exceed 1")
    if np.any(np.linalg.norm(path, axis=1) > radius + 1e-10):
        raise ValueError("true parameter norms exceed the declared radius")

    env_seed, policy_seed = np.random.SeedSequence(seed).spawn(2)
    reward_rng = np.random.default_rng(env_seed)
    agent = make_agent(name, features, np.random.default_rng(policy_seed),
                       window_size=window_size, gamma=gamma, radius=radius, delta=delta,
                       horizon=len(path))
    rows, cumulative = [], 0.0
    for step, theta_true in enumerate(path, start=1):
        start = perf_counter()
        env = LogisticBernoulliBandit(features, theta_true, reward_rng)
        environment_seconds = perf_counter() - start
        start = perf_counter()
        predicted = agent.predict_means()
        diagnostics_seconds = perf_counter() - start
        start = perf_counter()
        action = agent.select_action()
        select_seconds = perf_counter() - start
        start = perf_counter()
        reward = env.step(action)
        environment_seconds += perf_counter() - start
        start = perf_counter()
        regret = env.pseudo_regret(action)
        rmse = (None if predicted is None else
                float(np.sqrt(np.mean((predicted - env.arm_means) ** 2))))
        diagnostics_seconds += perf_counter() - start
        start = perf_counter()
        agent.update(action, reward)
        update_seconds = perf_counter() - start
        start = perf_counter()
        cumulative += regret
        row = {"algorithm": name, "seed": seed, "step": step,
               "action": int(action), "reward": reward,
               "instant_regret": regret, "cumulative_regret": cumulative,
               "prediction_rmse_pre": rmse}
        row.update({f"predicted_mean_{a}": None if predicted is None else float(predicted[a])
                    for a in range(len(features))})
        if name == "scb_weight_ucb_numeric_v1":
            row.update(agent.policy.audit())
        diagnostics_seconds += perf_counter() - start
        timing = {"step": step, "select_seconds": select_seconds,
                  "update_seconds": update_seconds,
                  "diagnostics_seconds": diagnostics_seconds,
                  "environment_seconds": environment_seconds}
        if on_step is not None:
            on_step(row, timing)
        if collect_rows:
            rows.append(row)
    return rows
