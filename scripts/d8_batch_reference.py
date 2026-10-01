"""Run E8 once, retain failures, and archive the exact stream for later OMD."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
import traceback
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, value):
    path.write_text(json.dumps(value, default=json_value, indent=2,
                               ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def command(args, timeout=30):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return {"command": args, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def snapshot(output):
    # Include untracked source too: Git HEAD alone does not describe this experiment.
    paths = {ROOT / name for name in ("run.py", "requirements.txt", "requirements-lock.txt",
                                     "requirements-d8.txt") if (ROOT / name).is_file()}
    for directory in ("algorithms", "envs", "estimators", "scripts", "tests"):
        paths.update((ROOT / directory).rglob("*.py"))
    paths.update((ROOT / "scripts").glob("*d8*.md"))
    hashes = {}
    with ZipFile(output / "source_snapshot.zip", "w", ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            archive.writestr(relative, data)
            hashes[relative] = hashlib.sha256(data).hexdigest()
    write_json(output / "source_sha256.json", hashes)


def run_experiment(output, config):
    import numpy as np
    from envs.logistic_bandit import LogisticBernoulliBandit
    from estimators.batch_logistic import fit_batch_logistic

    features = np.asarray(config["features"], dtype=float)
    truth = np.asarray(config["theta_star_for_generation_and_evaluation_only"], dtype=float)
    env_seed, action_seed = np.random.SeedSequence(config["seed"]).spawn(2)
    environment = LogisticBernoulliBandit(features, truth, np.random.default_rng(env_seed))
    action_rng = np.random.default_rng(action_seed)
    actions, rewards = [], []
    for _ in range(config["horizon"]):
        action = int(action_rng.integers(features.shape[0]))
        actions.append(action)
        rewards.append(environment.step(action))
    actions, rewards = np.asarray(actions), np.asarray(rewards, dtype=float)
    X = features[actions].copy()
    np.savez(output / "stream.npz", X=X, y=rewards, actions=actions)
    write_csv(output / "stream.csv", [
        {"step": i + 1, "action": int(actions[i]), "x0": X[i, 0],
         "x1": X[i, 1], "reward": int(rewards[i])}
        for i in range(len(rewards))
    ])
    angles = 2 * np.pi * (np.arange(config["eval_count"]) + 0.5) / config["eval_count"]
    eval_X = np.column_stack((np.cos(angles), np.sin(angles)))
    np.save(output / "eval_features.npy", eval_X)
    probability = lambda z: np.exp(-np.logaddexp(0.0, -z))
    eval_truth = probability(eval_X @ truth)
    np.savez(output / "evaluation_truth.npz", theta_star=truth, probabilities=eval_truth)

    # Fitting receives only saved observed covariates/labels and solver constants.
    with np.load(output / "stream.npz", allow_pickle=False) as saved:
        fit_X, fit_y = saved["X"].copy(), saved["y"].copy()
    eval_X = np.load(output / "eval_features.npy", allow_pickle=False)
    baseline = float(np.sqrt(np.mean((0.5 - eval_truth) ** 2)))
    summaries, predictions = [], []
    (output / "fits").mkdir()
    for n in config["prefixes"]:
        start = perf_counter()
        result = fit_batch_logistic(fit_X[:n], fit_y[:n], **config["solver"])
        elapsed = perf_counter() - start
        # Preserve the raw solver object even if a rejected result contains NaN.
        # Finite JSON summaries are used for normal results; repr is retained always.
        (output / "fits" / f"n{n}_raw.txt").write_text(repr(result) + "\n", encoding="utf-8")
        try:
            write_json(output / "fits" / f"n{n}.json", dict(result))
        except (ValueError, TypeError):
            if result.accepted:
                raise
            write_json(output / "fits" / f"n{n}.json", {
                "success": bool(result.success), "accepted": False,
                "status": int(result.status), "message": str(result.message),
                "serialization_note": "See raw text; solver returned a non-JSON numerical value.",
            })
        prediction = probability(eval_X @ result.x) if np.isfinite(result.x).all() else None
        rmse = float(np.sqrt(np.mean((prediction - eval_truth) ** 2))) if prediction is not None else None
        parameter_error = float(np.linalg.norm(result.x - truth)) if prediction is not None else None
        row = {"n": n, "success": bool(result.success), "accepted": bool(result.accepted),
               "status": int(result.status), "nit": int(result.get("nit", 0)),
               "prediction_rmse": rmse, "zero_prediction_rmse": baseline,
               "parameter_error": parameter_error, "fit_seconds": elapsed,
               "projected_gradient_residual": result.diagnostics["projected_gradient_residual"],
               "constraint_violation": result.diagnostics["constraint_violation"],
               "failure_reasons": ";".join(result.diagnostics["failure_reasons"])}
        summaries.append(row)
        write_csv(output / "summary.csv", summaries)
        if prediction is not None:
            predictions.extend({"n": n, "eval_index": j, "x0": eval_X[j, 0],
                                "x1": eval_X[j, 1], "true_probability": eval_truth[j],
                                "predicted_probability": prediction[j]} for j in range(len(eval_X)))
            write_csv(output / "predictions.csv", predictions)
        print(f"n={n}: success={result.success}, accepted={result.accepted}, "
              f"RMSE={rmse}, PG={row['projected_gradient_residual']}", flush=True)

    lines = ["# E8 固定数据流批量参照", "",
             "目标：Logistic 损失求和 + (l2/2)||theta||²；欧氏球约束；无额外截距。",
             "每个前缀从零初始化；估计器不接收真参数。", "",
             "| n | solver success | accepted | 概率RMSE | 零参数RMSE | 参数误差 | 耗时(s) |",
             "|---|---|---|---|---|---|---|"]
    for row in summaries:
        lines.append("| " + " | ".join(str(row[k]) for k in (
            "n", "success", "accepted", "prediction_rmse", "zero_prediction_rmse",
            "parameter_error", "fit_seconds")) + " |")
    lines.extend(["", "解释口径：",
                  "- 评估的是固定32点上的真实概率预测误差，不是闭环决策regret。",
                  "- 单条随机数据流的误差不必逐前缀下降；参数误差与预测误差不是同一个指标。",
                  "- 单次耗时包含输入检查、求解与独立诊断，只作描述，不能据此证明复杂度。",
                  "- 常数l2配合损失求和，正则相对于数据项的权重会随样本量变化。",
                  "- 后续OMD直接读取stream.npz与eval_features.npy；相同seed不足以保证同流。",
                  "- 若有拒绝项，保留原始结果，不自动重试或放宽容差。",
                  "", "待用户解释：比较n=10与n=200的预测误差；若中间反弹，说明为什么不直接判错。"])
    (output / "observations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return all(row["accepted"] for row in summaries)


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = ROOT / "results" / f"d8_batch_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    print(f"RESULTS: {output}", flush=True)
    accepted = False
    stage = "metadata"
    acceptance = {"accepted": False, "scope": "E8 automated tests and fixed-stream fits only",
                  "user_interpretation_and_git_sync": "pending"}
    try:
        import math
        config = {
            "seed": 20260928, "dimension": 2, "num_arms": 5, "horizon": 200,
            "features": [[1, 0], [0, 1], [-1, 0], [0, -1], [1 / math.sqrt(2), 1 / math.sqrt(2)]],
            "theta_star_for_generation_and_evaluation_only": [1.0, -0.5],
            "prefixes": [10, 25, 50, 100, 200], "eval_count": 32,
            "eval_angles": "2*pi*(j+0.5)/32, j=0..31",
            "rng": "SeedSequence(seed).spawn(2): environment, uniform actions",
            "solver": {"l2": 1.0, "radius": 2.0, "ftol": 1e-12, "maxiter": 500},
            "loss_reduction": "sum", "initialization": "zeros for each prefix",
            "feasibility_tol": 1e-7, "optimality_tol": 1e-5, "projected_gradient_step": 1.0,
        }
        write_json(output / "config.json", config)
        versions = {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}
        for package in ("numpy", "scipy", "pytest", "matplotlib"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "not installed"
        write_json(output / "versions.json", versions)
        git_info = {name: command(["git", *args]) for name, args in {
            "head": ["rev-parse", "HEAD"], "branch": ["branch", "--show-current"],
            "status": ["status", "--short", "--branch"], "diff_stat": ["diff", "--stat"],
        }.items()}
        write_json(output / "git.json", git_info)
        snapshot(output)
        stage = "pytest"
        test_command = [sys.executable, "-m", "pytest", "-q"]
        print("Running the full pytest suite once...", flush=True)
        with (output / "pytest.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(test_command, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=180)
        acceptance["pytest_command"] = test_command
        acceptance["pytest_returncode"] = completed.returncode
        tail = (output / "pytest.log").read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
        print("\n".join(tail), flush=True)
        if completed.returncode:
            raise RuntimeError("pytest failed; fixed-stream experiment was not started")
        stage = "fixed_stream_fits"
        accepted = run_experiment(output, config)
        acceptance["accepted"] = accepted
        acceptance["all_five_prefixes_accepted"] = accepted
        if not accepted:
            acceptance["failure"] = "At least one prefix was rejected; inspect summary.csv and fits/"
    except Exception as exc:
        acceptance.update(accepted=False, failure=f"{type(exc).__name__}: {exc}", failure_stage=stage)
        (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"FAILED at {stage}: {exc}", flush=True)
    finally:
        write_json(output / "acceptance.json", acceptance)
        hashes = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(output.rglob("*")) if p.is_file()}
        write_json(output / "manifest_sha256.json", hashes)
        archive = ROOT / "reports" / "d8" / stamp
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(output, archive)
        zip_path = output.parent / f"{output.name}_evidence.zip"
        with ZipFile(zip_path, "w", ZIP_DEFLATED) as zipped:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    zipped.write(path, arcname=f"{output.name}/{path.relative_to(output).as_posix()}")
        print(f"ARCHIVE: {archive}", flush=True)
        print(f"EVIDENCE_ZIP: {zip_path}", flush=True)
        print(f"E8 automated acceptance: {'PASS' if acceptance['accepted'] else 'FAIL'}", flush=True)
        print("Git commit/push and user interpretation are not performed by this script.", flush=True)
    return 0 if acceptance["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
