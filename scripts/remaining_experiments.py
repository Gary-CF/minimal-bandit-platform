"""Archive, verification and workers for the v2.2 supplemental experiment suite."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback
import uuid

import numpy as np

from algorithms.hvt_ucb import HvtUCB, RidgeLinearPolicy
from envs.remaining_bandit_tasks import features, noise_spec, noise_stream, parameter_path
from scripts.logistic_benchmark import ALGORITHMS, make_agent

ROOT = Path(__file__).resolve().parents[1]
FROZEN = {"random": {}, "ucb1": {}, "sw_ucb": {"window_size": 1000},
          "discounted_ucb": {"gamma": .999}, "omd_paper_v2_literal": {},
          "domd_paper_v1": {"gamma": .98}, "scb_weight_ucb_numeric_v1": {"gamma": .98}}
HEAVY = ("random", "ridge_greedy", "linucb_nominal", "hvt_ucb_literal")
FIXED = ("ridge_greedy", "hvt_ucb_literal", "hvt_squared_ablation")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    paths = [p for folder in ("algorithms", "estimators", "envs", "scripts", "tests")
             for p in (ROOT / folder).rglob("*.py")]
    paths += list((ROOT / "configs/v2_2").glob("*.json"))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def task_id(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()[:20]


def tasks_for(profile):
    smoke = profile == "smoke"
    fs = [20261130] if smoke else [20261130, 20261201, 20261202]
    seeds = [300] if smoke else [300, 301, 302]
    horizon = 60 if smoke else 1000
    settings = [(10, "stationary", 1.), (10, "abrupt", 1.), (10, "rotation", 1.)]
    if not smoke:
        settings += [(k, s, 1.) for k in (30, 100) for s in ("stationary", "abrupt", "rotation")]
        settings += [(10, "drift", 1.), (10, "abrupt", .25), (10, "abrupt", .5),
                     (10, "rotation", .5), (10, "rotation", 2.)]
    tasks = []
    def add(stage, alg, f, seed, T, k=10, scenario="stationary", angle_pi=1., noise="bernoulli"):
        tasks.append(dict(stage=stage, algorithm=alg, feature_seed=f, seed=seed,
                          horizon=T, num_arms=k, dimension=3, scenario=scenario,
                          angle_pi=angle_pi, noise=noise))
    for k, scenario, angle in settings:
        for f in fs:
            for seed in seeds:
                for alg in ALGORITHMS:
                    add("logistic", alg, f, seed, horizon, k, scenario, angle)
    for noise in ("gaussian", "student3", "student1_5"):
        for f in fs:
            for seed in seeds:
                for alg in FIXED:
                    add("heavy_fixed", alg, f, seed, horizon, noise=noise)
                for scenario in ("stationary", "abrupt", "rotation"):
                    for alg in HEAVY:
                        add("heavy_closed", alg, f, seed, horizon, scenario=scenario, noise=noise)
    for T in ([60] if smoke else [200, 1000, 5000]):
        for seed in ([310] if smoke else [310, 311, 312]):
            for alg in ("omd_paper_v2_literal", "domd_paper_v1", "scb_weight_ucb_numeric_v1",
                        "linucb_nominal", "hvt_ucb_literal"):
                add("cost", alg, fs[0], seed, T,
                    noise="bernoulli" if alg in ALGORITHMS else "student3")
    if len({task_id(t) for t in tasks}) != len(tasks):
        raise ValueError("duplicate task")
    return tasks


def make_policy(task, X):
    alg = task["algorithm"]
    if alg in ALGORITHMS and task["noise"] == "bernoulli":
        policy_seed = np.random.SeedSequence(task["seed"]).spawn(2)[1]
        return make_agent(alg, X, np.random.default_rng(policy_seed),
                          horizon=task["horizon"], radius=2., delta=.05, **FROZEN[alg])
    if alg in ("hvt_ucb_literal", "hvt_squared_ablation"):
        spec = noise_spec(task["noise"])
        return HvtUCB(X.shape[1], task["horizon"], spec["nu"], spec["epsilon"],
                      squared=alg == "hvt_squared_ablation")
    if alg in ("ridge_greedy", "linucb_nominal"):
        return RidgeLinearPolicy(X.shape[1], greedy=alg == "ridge_greedy")
    if alg == "random":
        return None
    raise ValueError(alg)


def memory_bytes(policy, X):
    """Unique NumPy backing allocations, excluding caller-owned static arm features.

    History arrays are reported separately. Python object overhead and temporary
    solver allocations are not included here; fresh-process RSS covers those.
    """
    seen, arrays = set(), {}
    def visit(obj, prefix):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            base = obj
            while isinstance(base.base, np.ndarray):
                base = base.base
            if not np.shares_memory(base, X):
                previous = arrays.get(id(base))
                kind = "history" if prefix.endswith((".X", ".y")) else "state"
                if previous is None or kind == "history":
                    arrays[id(base)] = (base.nbytes, kind)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                visit(v, f"{prefix}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                visit(v, f"{prefix}.{i}")
        elif hasattr(obj, "__dict__"):
            for k, v in vars(obj).items():
                # Logistic adapters copy static X; account it as arm storage, not state.
                if k != "features":
                    visit(v, f"{prefix}.{k}")
    visit(policy, "policy")
    return dict(state_array_bytes=int(sum(n for n, kind in arrays.values() if kind == "state")),
                retained_history_bytes=int(sum(n for n, kind in arrays.values() if kind == "history")),
                static_arm_bytes=int(X.nbytes))


@contextlib.contextmanager
def operation_counters():
    """Run ONLY in untimed replay, restore all wrappers even on failure."""
    import algorithms.scb_weight_ucb as scb
    import estimators.discounted_logistic_mle as mle
    import estimators.omd_logistic as omd
    counts = dict(loss_terms_calls=0, loss_terms_rows=0, projection_solver_calls=0,
                  projection_solver_iterations=0, metric_projection_linear_solves=0)
    originals = [(mle, "_terms", mle._terms), (scb, "_terms", scb._terms),
                 (scb, "minimize", scb.minimize),
                 (omd, "metric_ball_candidate", omd.metric_ball_candidate)]
    terms, solver, candidate = mle._terms, scb.minimize, omd.metric_ball_candidate
    def counted_terms(theta, X, *args):
        counts["loss_terms_calls"] += 1
        counts["loss_terms_rows"] += len(X)
        return terms(theta, X, *args)
    def counted_solver(*args, **kwargs):
        result = solver(*args, **kwargs)
        counts["projection_solver_calls"] += 1
        counts["projection_solver_iterations"] += int(result.nit)
        return result
    def counted_candidate(*args):
        counts["metric_projection_linear_solves"] += 1
        return candidate(*args)
    mle._terms = scb._terms = counted_terms
    scb.minimize, omd.metric_ball_candidate = counted_solver, counted_candidate
    try:
        yield counts
    finally:
        for module, name, original in originals:
            setattr(module, name, original)


def audit_cost(task, X, data):
    policy = make_policy(task, X)
    mle_iterations = 0
    with operation_counters() as counters:
        for i, (action, reward) in enumerate(zip(data["actions"], data["rewards"])):
            if task["algorithm"] in ALGORITHMS:
                chosen = policy.select_action()
                if chosen != action:
                    raise ValueError("audit replay action differs")
                policy.update(int(action), int(reward))
                estimator = policy.policy.estimator
                if hasattr(estimator, "last_diagnostics"):
                    mle_iterations += estimator.last_diagnostics["newton_steps"]
            else:
                if policy.select_action(X) != action:
                    raise ValueError("audit replay action differs")
                policy.update(X[action], reward)
    return dict(counters, mle_newton_iterations=mle_iterations, **memory_bytes(policy, X))


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def simulate(task):
    T, k = task["horizon"], task["num_arms"]
    X = features(task["feature_seed"], k)
    path = parameter_path(task["scenario"], T, np.pi * task["angle_pi"])
    linear_means = path @ X.T
    logistic = task["noise"] == "bernoulli"
    means = np.exp(-np.logaddexp(0., -linear_means)) if logistic else linear_means
    reward_seed, action_seed = np.random.SeedSequence(task["seed"]).spawn(2)
    rng = np.random.default_rng(reward_seed)
    innovations = rng.random(T) if logistic else noise_stream(task["noise"], task["seed"], T)
    action_rng = np.random.default_rng(action_seed)
    fixed = task["stage"] == "heavy_fixed"
    fixed_actions = action_rng.integers(k, size=T) if fixed else None
    policy = make_policy(task, X)
    data = dict(features=X, parameter_path=path, actions=np.zeros(T, dtype=np.int64),
                rewards=np.zeros(T), regret=np.zeros(T),
                predictions=np.full((T, k), np.nan),
                select_seconds=np.zeros(T), update_seconds=np.zeros(T),
                theta_norm_post=np.full(T, np.nan),
                sigma=np.full(T, np.nan), tau=np.full(T, np.nan), beta_pre=np.full(T, np.nan),
                clipped=np.zeros(T, dtype=np.int64), projection_active=np.zeros(T, dtype=np.int64))
    peak_before = rss_bytes()
    for i in range(T):
        # Diagnostic predictions are excluded from select/update timers.
        if task["algorithm"] in ALGORITHMS and logistic:
            pred = policy.predict_means()
        elif policy is not None and hasattr(policy, "theta"):
            pred = X @ policy.theta
        else:
            pred = None
        if pred is not None:
            data["predictions"][i] = pred
        start = time.perf_counter()
        if fixed:
            action = int(fixed_actions[i])
        elif task["algorithm"] == "random" and not logistic:
            action = int(action_rng.integers(k))
        elif logistic:
            action = policy.select_action()
        else:
            action = policy.select_action(X)
        data["select_seconds"][i] = time.perf_counter() - start
        reward = int(innovations[i] < means[i, action]) if logistic else means[i, action] + innovations[i]
        start = time.perf_counter()
        if logistic:
            policy.update(action, reward)
        elif policy is not None and hasattr(policy, "update"):
            policy.update(X[action], reward)
        data["update_seconds"][i] = time.perf_counter() - start
        data["actions"][i], data["rewards"][i] = action, reward
        data["regret"][i] = means[i].max() - means[i, action]
        if isinstance(policy, HvtUCB):
            for key in ("sigma", "tau", "beta_pre", "clipped", "projection_active"):
                val = policy.last_diagnostics[key]
                data[key][i] = np.nan if val is None else val
        if logistic and hasattr(policy.policy, "estimator"):
            theta = getattr(policy.policy, "theta", policy.policy.estimator.theta)
            data["theta_norm_post"][i] = np.linalg.norm(theta)
            if task["algorithm"] == "scb_weight_ucb_numeric_v1":
                data["projection_active"][i] = policy.policy.projection_info["active"]
        elif hasattr(policy, "theta"):
            data["theta_norm_post"][i] = np.linalg.norm(policy.theta)
    peak_after = rss_bytes()
    summary = summarize(task, data)
    summary.update(memory_bytes(policy, X))
    summary.update(process_peak_rss_bytes=peak_after,
                   peak_rss_increase_bytes=max(0, peak_after - peak_before),
                   rss_scope="fresh episode process" if task["stage"] == "cost" else "shared worker lifetime",
                   timing_scope="serial isolated cost worker" if task["stage"] == "cost" else "parallel throughput run")
    if task["stage"] == "cost":
        summary["operation_audit"] = audit_cost(task, X, data)
    return data, summary


def summarize(task, data):
    linear = data["parameter_path"] @ data["features"].T
    means = np.exp(-np.logaddexp(0., -linear)) if task["noise"] == "bernoulli" else linear
    pred = data["predictions"]
    have_prediction = np.isfinite(pred).all()
    T = task["horizon"]
    width = min(100, T // 3)
    windows = {"early": slice(0, width), "middle": slice((T - width) // 2, (T + width) // 2),
               "late": slice(T - width, T)}
    result = dict(task, task_id=task_id(task), final_regret=float(data["regret"].sum()),
                  prediction_rmse=float(np.sqrt(np.mean((pred - means) ** 2))) if have_prediction else None,
                  select_us=float(data["select_seconds"].mean() * 1e6),
                  update_us=float(data["update_seconds"].mean() * 1e6),
                  clipped_updates=int(data["clipped"].sum()),
                  projection_updates=int(data["projection_active"].sum()),
                  parameter_path_length=float(np.linalg.norm(np.diff(data["parameter_path"], axis=0), axis=1).sum()))
    for label, interval in windows.items():
        result[f"{label}_update_us"] = float(data["update_seconds"][interval].mean() * 1e6)
        result[f"{label}_rmse"] = float(np.sqrt(np.mean((pred[interval] - means[interval]) ** 2))) if have_prediction else None
    if task["scenario"] == "abrupt":
        change = T // 2
        w = min(100, change, T - change)
        for label, interval in (("pre", slice(change - w, change)), ("post", slice(change, change + w)),
                                ("late", slice(T - w, T))):
            result[f"abrupt_{label}_regret"] = float(data["regret"][interval].mean())
    return result


def validate_data(task, data):
    T, k = task["horizon"], task["num_arms"]
    X = features(task["feature_seed"], k)
    path = parameter_path(task["scenario"], T, task["angle_pi"] * np.pi)
    np.testing.assert_allclose(data["features"], X, rtol=0, atol=0)
    np.testing.assert_allclose(data["parameter_path"], path, rtol=0, atol=0)
    expected_shapes = {"features": (k, 3), "parameter_path": (T, 3), "predictions": (T, k)}
    for name, value in data.items():
        if value.shape != expected_shapes.get(name, (T,)):
            raise ValueError(f"invalid array shape: {name}")
        if name not in ("predictions", "theta_norm_post", "sigma", "tau", "beta_pre") and not np.isfinite(value).all():
            raise ValueError(f"nonfinite required array: {name}")
    actions = data["actions"]
    if actions.dtype.kind not in "iu" or np.any(actions < 0) or np.any(actions >= k):
        raise ValueError("invalid actions")
    means = path @ X.T
    if task["noise"] == "bernoulli":
        means = np.exp(-np.logaddexp(0., -means))
        seed = np.random.SeedSequence(task["seed"]).spawn(2)[0]
        rewards = (np.random.default_rng(seed).random(T) < means[np.arange(T), actions]).astype(float)
    else:
        rewards = means[np.arange(T), actions] + noise_stream(task["noise"], task["seed"], T)
    np.testing.assert_allclose(data["rewards"], rewards, rtol=0, atol=1e-12)
    regret = means.max(axis=1) - means[np.arange(T), actions]
    np.testing.assert_allclose(data["regret"], regret, rtol=0, atol=1e-12)
    if task["stage"] == "heavy_fixed":
        seed = np.random.SeedSequence(task["seed"]).spawn(2)[1]
        np.testing.assert_array_equal(actions, np.random.default_rng(seed).integers(k, size=T))
    prediction_expected = task["algorithm"] not in ("random", "ucb1", "sw_ucb", "discounted_ucb")
    if prediction_expected != bool(np.isfinite(data["predictions"]).all()):
        raise ValueError("missing or unexpected predictions")
    for name in ("select_seconds", "update_seconds"):
        if np.any(data[name] < 0):
            raise ValueError("negative time")
    if task["algorithm"] in ("hvt_ucb_literal", "hvt_squared_ablation", "omd_paper_v2_literal", "domd_paper_v1", "scb_weight_ucb_numeric_v1"):
        if not np.isfinite(data["theta_norm_post"]).all() or np.any(data["theta_norm_post"] > 2 + 1e-7):
            raise ValueError("invalid parameter norm")


def verify_task(directory, expected=None):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if set(manifest) != {"task.json", "summary.json", "trajectory.npz"}:
        raise ValueError("unexpected episode manifest")
    if {p.name for p in directory.iterdir()} != set(manifest) | {"manifest.json"}:
        raise ValueError("unexpected episode files")
    for name, sha in manifest.items():
        if digest(directory / name) != sha:
            raise ValueError(f"checksum mismatch: {directory / name}")
    task = read_json(directory / "task.json")
    if expected is not None and task != expected:
        raise ValueError("episode task changed")
    with np.load(directory / "trajectory.npz", allow_pickle=False) as archive:
        data = dict(archive)
    required = {"features", "parameter_path", "actions", "rewards", "regret", "predictions",
                "select_seconds", "update_seconds", "theta_norm_post", "sigma", "tau", "beta_pre",
                "clipped", "projection_active"}
    if set(data) != required:
        raise ValueError("unexpected trajectory fields")
    validate_data(task, data)
    saved = read_json(directory / "summary.json")
    for key, value in summarize(task, data).items():
        if saved[key] != value:
            raise ValueError(f"summary mismatch: {key}")
    if saved["external_trajectory_bytes"] != (directory / "trajectory.npz").stat().st_size:
        raise ValueError("external log size mismatch")
    return saved


def execute_task(task, root):
    root = Path(root)
    destination = root / "episodes" / task_id(task)
    if destination.exists():
        return verify_task(destination, task)
    temporary = root / "attempts" / f"{task_id(task)}-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        write_json(temporary / "task.json", task)
        data, summary = simulate(task)
        validate_data(task, data)
        np.savez_compressed(temporary / "trajectory.npz", **data)
        summary["external_trajectory_bytes"] = (temporary / "trajectory.npz").stat().st_size
        write_json(temporary / "summary.json", summary)
        write_json(temporary / "manifest.json", {name: digest(temporary / name)
                   for name in ("task.json", "trajectory.npz", "summary.json")})
        verify_task(temporary, task)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(destination)
        return summary
    except BaseException:
        (temporary / "failure.txt").write_text(traceback.format_exc())
        raise
