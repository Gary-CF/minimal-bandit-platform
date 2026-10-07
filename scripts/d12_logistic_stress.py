"""Experiment-only time-varying truth; reuse the accepted stationary policy."""
import csv

import numpy as np

from scripts.d8_batch_reference import write_json
from algorithms.base import RandomPolicy
from algorithms.logistic_omd import LogisticOMDPolicy
from envs.logistic_bandit import LogisticBernoulliBandit


def parameter_path(scenario, horizon, before, after, change_step):
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 2:
        raise ValueError("horizon must be an integer >= 2")
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if before.ndim != 1 or before.size == 0 or before.shape != after.shape:
        raise ValueError("parameter endpoints must be matching nonempty vectors")
    if not np.isfinite([before, after]).all():
        raise ValueError("parameter endpoints must be finite")
    if scenario == "abrupt":
        if isinstance(change_step, bool) or not isinstance(change_step, int) or not 2 <= change_step <= horizon:
            raise ValueError("change_step must be an integer in [2, horizon]")
        weight = (np.arange(1, horizon + 1) >= change_step).astype(float)
    elif scenario == "drift":
        weight = np.linspace(0.0, 1.0, horizon)
    else:
        raise ValueError("unknown scenario")
    return before + weight[:, None] * (after - before)


def run_episode(output, scenario, algorithm, seed, features, path, config, status):
    if algorithm not in ("random", "omd_paper_v2_literal"):
        raise ValueError("unknown algorithm")
    env_seed, policy_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(env_seed)
    is_omd = algorithm == "omd_paper_v2_literal"
    policy = (LogisticOMDPolicy(features.shape[1], radius=config["radius"],
                               delta=config["delta"], beta_mode=config["beta_mode"])
              if is_omd else RandomPolicy(len(features), np.random.default_rng(policy_seed)))
    records, cumulative = [], 0.0
    label = f"{scenario}_{algorithm}_seed{seed}"
    destination = output / f"{label}.csv"
    fields = ["scenario", "algorithm", "seed", "step", "action", "reward", "instant_regret",
              "cumulative_regret", "prediction_rmse_pre", "theta_norm", "kkt_residual", "rho"]
    fields += [f"predicted_mean_{a}" for a in range(len(features))]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step, theta_true in enumerate(path, 1):
            status["current"] = {"episode": label, "step": step}
            # Each round uses the existing validated environment with that round's truth.
            # The same RNG persists across rounds; the policy never sees this environment.
            env = LogisticBernoulliBandit(features, theta_true, rng)
            predicted = (np.exp(-np.logaddexp(0.0, -(features @ policy.estimator.theta)))
                         if is_omd else None)
            action = policy.select_action(features) if is_omd else policy.select_action()
            reward = env.step(action)
            regret = env.pseudo_regret(action)
            try:
                info = policy.update(features[action], reward) if is_omd else policy.update(action, reward)
            except Exception:
                status["solver_failures"] += 1
                raise
            cumulative += regret
            row = dict(zip(fields[:12], [scenario, algorithm, seed, step, action, reward, regret,
                cumulative, float(np.sqrt(np.mean((predicted - env.arm_means)**2))) if is_omd else "",
                info["theta_norm"] if is_omd else "", info["kkt_residual"] if is_omd else "",
                info["rho"] if is_omd else ""]))
            row.update({f"predicted_mean_{a}": float(predicted[a]) if is_omd else ""
                        for a in range(len(features))})
            writer.writerow(row)
            records.append(row)
    if is_omd:
        model = policy.estimator
        write_json(output / f"{label}_state.json", {
            "theta": model.theta, "H": model.H, "completed_steps": model.completed_steps,
            "array_payload_bytes": model.theta.nbytes + model.H.nbytes,
            "state_keys": sorted(vars(model)), "eta": model.eta, "lambda": model.l2,
            "beta_mode": policy.beta_mode,
        })
    return records


def validate_episode(rows, features, path, scenario, algorithm, seed):
    if len(rows) != len(path):
        raise ValueError("incomplete stress trajectory")
    means = 1 / (1 + np.exp(-(path @ features.T)))
    cumulative = 0.0
    for step, (row, truth) in enumerate(zip(rows, means), 1):
        if (row["scenario"], row["algorithm"], int(row["seed"]), int(row["step"])) != (scenario, algorithm, seed, step):
            raise ValueError("trajectory identity/time mismatch")
        action = int(row["action"])
        if not 0 <= action < len(features) or float(row["reward"]) not in (0, 1):
            raise ValueError("invalid action/reward")
        regret = float(truth.max() - truth[action])
        cumulative += regret
        if not np.allclose([float(row["instant_regret"]), float(row["cumulative_regret"])],
                           [regret, cumulative], atol=1e-9, rtol=0):
            raise ValueError("dynamic pseudo-regret mismatch")
        if algorithm != "random":
            predicted = np.array([float(row[f"predicted_mean_{a}"]) for a in range(len(features))])
            if not np.isfinite(predicted).all() or np.any((predicted < 0) | (predicted > 1)):
                raise ValueError("invalid probability prediction")
            if not np.isclose(float(row["prediction_rmse_pre"]), np.sqrt(np.mean((predicted - truth)**2)), atol=1e-12, rtol=0):
                raise ValueError("pre-update prediction RMSE mismatch")
            if not (0 <= float(row["theta_norm"]) <= 2 + 1e-7 and
                    0 <= float(row["kkt_residual"]) <= 1e-7 and float(row["rho"]) >= 0):
                raise ValueError("invalid update diagnostics")
