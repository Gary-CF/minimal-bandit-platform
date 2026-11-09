"""Development diagnostic: identical observations, only gamma changes."""
import argparse
import csv
import json
import shutil
import traceback
from pathlib import Path

import numpy as np

from estimators.domd_logistic import OnlineLogisticDOMD
from estimators.omd_logistic import OnlineLogisticOMD


def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -z))


def write_csv(path, rows):
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    status = out / "status.json"
    status.write_text('{"state": "running"}\n')
    try:
        T, K, d = 1000, 10, 3
        gammas = (1.0, 0.98, 0.99, 0.995, 0.998, 0.999)
        seeds = list(range(5))
        config = dict(horizon=T, num_arms=K, dimension=d,
                      feature_seed=20261001, seeds=seeds, gammas=gammas,
                      eta=3.0, l2=126.0, radius=2.0, change_step=501,
                      theta_before=[1.0, -0.5, 0.75],
                      theta_after=[-1.0, 0.5, -0.75],
                      action_rule="fixed uniform random",
                      metric="pre-update full-arm probability RMSE",
                      numpy_version=np.__version__)
        (out / "config.json").write_text(json.dumps(config, indent=2))
        root = Path(__file__).resolve().parents[1]
        for name in ("scripts/diagnose_domd_forgetting.py",
                     "estimators/domd_logistic.py",
                     "estimators/omd_logistic.py", "envs/logistic_bandit.py"):
            target = out / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, target)
        X = np.random.default_rng(config["feature_seed"]).normal(size=(K, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        before = np.array(config["theta_before"])
        after = np.array(config["theta_after"])
        per_seed = []
        for scenario in ("stationary", "abrupt", "drift"):
            weight = (np.zeros(T) if scenario == "stationary" else
                      (np.arange(T) >= 500).astype(float) if scenario == "abrupt"
                      else np.linspace(0.0, 1.0, T))
            truth = before + weight[:, None] * (after - before)
            probabilities = sigmoid(truth @ X.T)
            for seed in seeds:
                action_seed, reward_seed = np.random.SeedSequence(seed).spawn(2)
                actions = np.random.default_rng(action_seed).integers(K, size=T)
                rewards = np.random.default_rng(reward_seed).binomial(
                    1, probabilities[np.arange(T), actions])
                np.savez_compressed(out / f"inputs_{scenario}_{seed}.npz",
                                    X=X, truth=truth, actions=actions, rewards=rewards)
                models = [OnlineLogisticDOMD(d, g, 3.0, 126.0, 2.0) for g in gammas]
                reference = OnlineLogisticOMD(d, 3.0, 126.0, 2.0)
                errors = np.empty((T, len(gammas)))
                rows = []
                for t, (action, reward) in enumerate(zip(actions, rewards)):
                    x, y = X[action], int(reward)
                    for j, model in enumerate(models):
                        error = float(np.sqrt(np.mean(
                            (sigmoid(X @ model.theta) - probabilities[t]) ** 2)))
                        errors[t, j] = error
                        rows.append(dict(step=t + 1, gamma=gammas[j], rmse_pre=error))
                        model.update(x, y)
                    reference.update(x, y)
                    np.testing.assert_array_equal(models[0].theta, reference.theta)
                    np.testing.assert_array_equal(models[0].H, reference.H)
                write_csv(out / f"metrics_{scenario}_{seed}.csv", rows)
                windows = {"all": slice(0, T)}
                if scenario == "abrupt":
                    for width in (50, 100, 200):
                        windows.update({f"pre{width}": slice(500 - width, 500),
                                        f"post{width}": slice(500, 500 + width),
                                        f"late{width}": slice(T - width, T)})
                else:
                    windows["late200"] = slice(T - 200, T)
                for window, indices in windows.items():
                    means = errors[indices].mean(axis=0)
                    for j, gamma in enumerate(gammas):
                        per_seed.append(dict(scenario=scenario, seed=seed,
                                             gamma=gamma, window=window,
                                             rmse=float(means[j]),
                                             diff_omd=float(means[j] - means[0])))
                print(f"STREAM PASS: {scenario} seed={seed}; gamma=1 matches OMD", flush=True)
        write_csv(out / "per_seed.csv", per_seed)
        summary = []
        keys = dict.fromkeys((r["scenario"], r["window"], r["gamma"]) for r in per_seed)
        for scenario, window, gamma in keys:
            group = [r for r in per_seed if
                     (r["scenario"], r["window"], r["gamma"]) == (scenario, window, gamma)]
            row = dict(scenario=scenario, window=window, gamma=gamma, n=len(group))
            for metric in ("rmse", "diff_omd"):
                values = np.array([r[metric] for r in group])
                row[f"mean_{metric}"] = float(values.mean())
                row[f"sem_{metric}"] = float(values.std(ddof=1) / np.sqrt(len(values)))
            summary.append(row)
        write_csv(out / "summary.csv", summary)
        print("scenario    window    gamma    RMSE       paired diff +/- SEM")
        for r in summary:
            if r["window"] in ("all", "pre100", "post100", "late100"):
                print(f"{r['scenario']:11} {r['window']:9} {r['gamma']:.3f}  "
                      f"{r['mean_rmse']:.6f}  "
                      f"{r['mean_diff_omd']:+.6f} +/- {r['sem_diff_omd']:.6f}")
        status.write_text(json.dumps(dict(state="complete", fixed_streams=15,
                                         estimator_runs=90, equality_checks=15000)))
        print(f"DIAGNOSTIC PASS: 15 fixed streams; 90 estimator runs | {out}")
    except Exception:
        status.write_text('{"state": "failed"}\n')
        (out / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
