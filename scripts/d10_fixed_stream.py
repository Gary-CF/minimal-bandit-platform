"""E9 regression gate and E10 estimator comparison on the archived E8 stream."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import traceback
from zipfile import ZipFile, ZIP_DEFLATED

from d8_batch_reference import ROOT, command, snapshot, write_csv, write_json


def probability(z):
    import numpy as np
    return np.exp(-np.logaddexp(0.0, -z))


def verify_input(source):
    manifest = json.loads((source / "manifest_sha256.json").read_text())
    for name, expected in manifest.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source):
            raise ValueError("Invalid path in E8 manifest")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"E8 hash mismatch: {name}")
    acceptance = json.loads((source / "acceptance.json").read_text())
    if not acceptance["accepted"] or not acceptance["all_five_prefixes_accepted"]:
        raise ValueError("E8 acceptance is not complete")
    return manifest


def compare(output, source, config, status):
    import numpy as np
    from estimators.batch_logistic import fit_batch_logistic
    from estimators.omd_logistic import OnlineLogisticOMD

    with np.load(source / "stream.npz", allow_pickle=False) as saved:
        X, y = saved["X"].copy(), saved["y"].copy()
    eval_X = np.load(source / "eval_features.npy", allow_pickle=False)
    with np.load(source / "evaluation_truth.npz", allow_pickle=False) as truth:
        theta_star = truth["theta_star"].copy()
        true_probability = truth["probabilities"].copy()
    X.setflags(write=False)
    y.setflags(write=False)
    model = OnlineLogisticOMD(dimension=X.shape[1], **config["omd"])
    status["completed_updates"] = 0
    summaries, updates, predictions, states = [], [], [], {}
    online_time, batch_time = 0.0, 0.0
    baseline = float(np.sqrt(np.mean((0.5 - true_probability) ** 2)))
    (output / "batch_fits").mkdir()

    def evaluate(algorithm, n, theta, current_seconds, cumulative_seconds, array_bytes, **extra):
        pred = probability(eval_X @ theta)
        row = {
            "algorithm": algorithm, "n": n,
            "prediction_rmse": float(np.sqrt(np.mean((pred - true_probability) ** 2))),
            "zero_prediction_rmse": baseline,
            "parameter_error": float(np.linalg.norm(theta - theta_star)),
            "last_call_seconds": current_seconds,
            "cumulative_call_seconds": cumulative_seconds,
            "array_payload_bytes": array_bytes,
            "batch_success": extra.get("batch_success", ""),
            "batch_accepted": extra.get("batch_accepted", ""),
            "e8_theta_max_abs_difference": extra.get("e8_difference", ""),
        }
        summaries.append(row)
        predictions.extend({"algorithm": algorithm, "n": n, "eval_index": j,
                            "predicted_probability": pred[j], "true_probability": true_probability[j]}
                           for j in range(len(eval_X)))
        write_csv(output / "summary.csv", summaries)
        write_csv(output / "predictions.csv", predictions)
        print(f"n={n:3d} {algorithm:5s}: RMSE={row['prediction_rmse']:.8f}, "
              f"cumulative_call_seconds={cumulative_seconds:.6f}", flush=True)

    for n, (x, reward) in enumerate(zip(X, y), start=1):
        status["stage"] = f"omd_update_{n}"
        start = perf_counter()
        try:
            info = model.update(x, float(reward))
        except Exception as exc:
            status["solver_failures"] += 1
            write_json(output / "failed_update.json", {
                "step": n, "exception": f"{type(exc).__name__}: {exc}",
                "theta_before": model.theta, "H_before": model.H,
                "completed_steps": model.completed_steps,
            })
            raise
        elapsed = perf_counter() - start
        online_time += elapsed
        status["completed_updates"] = model.completed_steps
        arrays = [value for value in vars(model).values() if isinstance(value, np.ndarray)]
        if len(arrays) != 2 or model.theta.shape != (X.shape[1],) or model.H.shape != (X.shape[1], X.shape[1]):
            raise RuntimeError("Unexpected estimator state layout")
        array_bytes = sum(array.nbytes for array in arrays)
        updates.append({"step": n, "seconds": elapsed, "array_payload_bytes": array_bytes, **info})
        # Logging is outside the timed update and outside the estimator state.
        write_csv(output / "updates.csv", updates)
        if n not in config["prefixes"]:
            continue
        states[f"theta_{n}"] = model.theta.copy()
        states[f"H_{n}"] = model.H.copy()
        np.savez(output / "omd_states.npz", **states)
        evaluate("omd", n, model.theta, elapsed, online_time, array_bytes)

        status["stage"] = f"batch_fit_{n}"
        start = perf_counter()
        result = fit_batch_logistic(X[:n], y[:n], **config["batch"])
        elapsed = perf_counter() - start
        batch_time += elapsed
        (output / "batch_fits" / f"n{n}_raw.txt").write_text(repr(result) + "\n")
        if not result.accepted:
            status["solver_failures"] += 1
            write_json(output / "batch_fits" / f"n{n}_failure.json", {
                "success": bool(result.success), "accepted": False,
                "status": int(result.status), "message": str(result.message),
            })
            raise RuntimeError(f"Batch fit rejected at n={n}; no automatic retry")
        write_json(output / "batch_fits" / f"n{n}.json", dict(result))
        archived = json.loads((source / "fits" / f"n{n}.json").read_text())
        difference = float(np.max(np.abs(result.x - np.asarray(archived["x"]))))
        evaluate("batch", n, result.x, elapsed, batch_time, X[:n].nbytes + y[:n].nbytes,
                 batch_success=bool(result.success), batch_accepted=bool(result.accepted),
                 e8_difference=difference)

    if status["completed_updates"] != len(y) or len(summaries) != 2 * len(config["prefixes"]):
        raise RuntimeError("Incomplete fixed-stream comparison")
    comparison = ["# E10 OMD / Batch 固定数据流对照", "",
                  "OMD: eta=2, H1=I, radius=2；这是工程对照参数，不主张满足论文定理的参数条件。",
                  "Batch: 沿用E8损失求和、l2=1、radius=2、ftol=1e-12。",
                  "在线H1与批量L2系数作用不同；相同数值不意味着求解同一个优化目标。", "",
                  "| n | OMD概率RMSE | Batch概率RMSE | OMD累计更新(s) | Batch该次重拟合(s) |",
                  "|---|---|---|---|---|"]
    for n in config["prefixes"]:
        a, b = [row for row in summaries if row["n"] == n]
        comparison.append(f"| {n} | {a['prediction_rmse']:.8f} | {b['prediction_rmse']:.8f} | "
                          f"{a['cumulative_call_seconds']:.6f} | {b['last_call_seconds']:.6f} |")
    comparison.extend(["", "解读边界：",
        "- 使用E8原始X/y与固定评估点，不重新生成数据；处理n个样本后比较在线theta_(n+1)与批量n样本解。",
        "- OMD接收每个样本一次；Batch仅在五个指定前缀从零重拟合，不是每轮调用。",
        "- OMD累计耗时包括逐样本检查、投影、曲率更新和诊断；Batch耗时包括输入检查、SLSQP和诊断。",
        "- 计时不包含外部CSV日志、预测评估和源码归档。单次小规模计时不能证明复杂度或普遍速度排名。",
        "- OMD数组载荷只统计theta和H的nbytes；Batch统计输入前缀X/y载荷，不含临时副本、求解器工作区或Python开销。两者都不是峰值内存。",
        "- 估计器不持有历史列表；实验驱动程序独立保存流和日志。",
        "- 不要求两种参数相等，不要求某种算法每个前缀都更准，接受标准与算法排名分开。",
        "- 本实验不评价选臂或regret。E11还需乐观选臂、闭环smoke、正式实验与规模测量。",
        "", "自动验收通过不代表已提交推送或完成E11。"])
    (output / "observations.md").write_text("\n".join(comparison) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d8", required=True, type=Path)
    args = parser.parse_args()
    source = args.d8.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = ROOT / "results" / f"d10_fixed_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    status = {"accepted": False, "solver_failures": 0, "stage": "input_verification",
              "scope": "E9 regression gate and E10 fixed-stream numerical comparison",
              "git_sync_and_interpretation": "pending"}
    print(f"RESULTS: {output}", flush=True)
    try:
        manifest = verify_input(source)
        e8_config = json.loads((source / "config.json").read_text())
        shutil.copytree(source, output / "input_d8")
        source = output / "input_d8"
        config = {"source_d8": str(args.d8.resolve()), "input_hashes": manifest,
                  "prefixes": e8_config["prefixes"], "batch": e8_config["solver"],
                  "omd": {"eta": 2.0, "l2": 1.0, "radius": e8_config["solver"]["radius"]},
                  "parameter_mode": "engineering_comparison_not_theorem_configuration",
                  "timing": "single run; update/fit calls include validation and diagnostics; exclude evaluation and logging"}
        write_json(output / "config.json", config)
        write_json(output / "versions.json", {
            "python": sys.version, **{name: importlib.metadata.version(name)
                                     for name in ("numpy", "scipy", "pytest")}})
        write_json(output / "git.json", {
            "head": command(["git", "rev-parse", "HEAD"]),
            "status": command(["git", "status", "--short", "--branch"]),
        })
        snapshot(output)
        status["stage"] = "regression_tests"
        if not (ROOT / "tests/test_omd_logistic_week2.py").is_file():
            raise FileNotFoundError("The E9 persistent test file must be present")
        print("Running the full pytest suite once...", flush=True)
        try:
            result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=180)
        except subprocess.TimeoutExpired as exc:
            partial = exc.stdout or b""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            (output / "pytest.log").write_text(partial, encoding="utf-8")
            raise
        (output / "pytest.log").write_text(result.stdout, encoding="utf-8")
        status["pytest_returncode"] = result.returncode
        print("\n".join((output / "pytest.log").read_text(errors="replace").splitlines()[-8:]), flush=True)
        if result.returncode:
            raise RuntimeError("Regression tests failed; comparison not started")
        compare(output, source, config, status)
        status.update(accepted=True, stage="complete")
    except Exception as exc:
        status["failure"] = f"{type(exc).__name__}: {exc}"
        (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"FAIL: {status['failure']}", flush=True)
    finally:
        write_json(output / "acceptance.json", status)
        write_json(output / "manifest_sha256.json", {
            path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.rglob("*")) if path.is_file()})
        archive = ROOT / "reports" / "d10" / stamp
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(output, archive)
        package = output.parent / f"{output.name}_evidence.zip"
        with ZipFile(package, "w", ZIP_DEFLATED) as zipped:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    zipped.write(path, f"{output.name}/{path.relative_to(output).as_posix()}")
        print(f"ARCHIVE: {archive}", flush=True)
        print(f"EVIDENCE_ZIP: {package}", flush=True)
        print(f"E10 automated acceptance: {'PASS' if status['accepted'] else 'FAIL'}", flush=True)
    return 0 if status["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
