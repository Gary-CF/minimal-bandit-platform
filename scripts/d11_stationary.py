"""Fixed E11 protocol: smoke, stationary comparison, and instrumented cost measurements."""
from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from time import perf_counter
import traceback
from unittest.mock import patch
from zipfile import ZipFile, ZIP_DEFLATED

from d8_batch_reference import ROOT, command, snapshot, write_csv, write_json
import numpy as np
from algorithms.base import RandomPolicy
from algorithms.logistic_omd import LogisticOMDPolicy
from envs.logistic_bandit import LogisticBernoulliBandit
import estimators.omd_logistic as omd_module


class Meter:
    def __init__(self):
        self.phase = "idle"
        self.solve = 0.0
        self.projection = 0.0


@contextmanager
def instrument(meter):
    solve, project = np.linalg.solve, omd_module.project_metric_ball
    def timed_solve(*args, **kwargs):
        start = perf_counter()
        try:
            return solve(*args, **kwargs)
        finally:
            if meter.phase == "update":
                meter.solve += perf_counter() - start
    def timed_project(*args, **kwargs):
        start = perf_counter()
        try:
            return project(*args, **kwargs)
        finally:
            meter.projection += perf_counter() - start
    with patch.object(np.linalg, "solve", timed_solve), patch.object(omd_module, "project_metric_ball", timed_project):
        yield


def episode(output, label, algorithm, seed, horizon, features, theta_star, status, meter=None):
    env_seed, policy_seed = np.random.SeedSequence(seed).spawn(2)
    env = LogisticBernoulliBandit(features, theta_star, np.random.default_rng(env_seed))
    is_omd = algorithm == "omd_paper_v2_literal"
    policy = (LogisticOMDPolicy(features.shape[1], radius=2.0, delta=0.05,
                               beta_mode="paper_v2_literal") if is_omd else
              RandomPolicy(len(features), np.random.default_rng(policy_seed)))
    rows, times = [], []
    cumulative = 0.0
    path = output / "episodes" / f"{label}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["environment", "algorithm", "algorithm_id", "horizon", "seed", "step", "action",
              "reward", "instant_regret", "cumulative_regret"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in range(1, horizon + 1):
            status["current"] = {"episode": label, "step": step, "phase": "selection"}
            if meter:
                meter.phase = "selection"
            start = perf_counter()
            action = policy.select_action(features) if is_omd else policy.select_action()
            select_seconds = perf_counter() - start
            if not isinstance(action, (int, np.integer)) or not 0 <= action < len(features):
                raise RuntimeError("Invalid selected action")
            reward = env.step(action)
            regret = env.pseudo_regret(action)
            expected_regret = float(np.max(env.arm_means) - env.arm_means[action])
            if reward not in (0, 1) or not np.isclose(regret, expected_regret, atol=1e-14, rtol=0):
                raise RuntimeError("Invalid reward/regret")

            status["current"]["phase"] = "update"
            old_solve = meter.solve if meter else 0.0
            old_projection = meter.projection if meter else 0.0
            if meter:
                meter.phase = "update"
            start = perf_counter()
            try:
                info = policy.update(features[action], reward) if is_omd else policy.update(action, reward)
            except Exception:
                status["solver_failures"] += 1
                raise
            update_seconds = perf_counter() - start
            if meter:
                meter.phase = "idle"
            if is_omd:
                model = policy.estimator
                if model.completed_steps != step or info["constraint_violation"] > 1e-7:
                    raise RuntimeError("Invalid post-update state")
                array_bytes = model.theta.nbytes + model.H.nbytes
                if array_bytes != 8 * (features.shape[1] + features.shape[1] ** 2):
                    raise RuntimeError("Estimator array payload changed")
            else:
                array_bytes = 0
            cumulative += regret
            row = dict(zip(fields, ["logistic_bernoulli", algorithm, algorithm, horizon, seed, step,
                                    action, reward, regret, cumulative]))
            start = perf_counter()
            writer.writerow(row)
            logging_seconds = perf_counter() - start
            rows.append(row)
            solve_seconds = meter.solve - old_solve if meter else 0.0
            times.append({"step": step, "select_seconds": select_seconds,
                          "update_seconds": update_seconds, "logging_seconds": logging_seconds,
                          "update_linear_solve_seconds": solve_seconds,
                          "update_other_seconds": update_seconds - solve_seconds,
                          "projection_inclusive_seconds": meter.projection - old_projection if meter else 0.0,
                          "array_payload_bytes": array_bytes,
                          "projection_active": bool(info["rho"] > 0) if is_omd else False})
    write_csv(output / "episodes" / f"{label}_timing.csv", times)
    if is_omd:
        write_json(output / "episodes" / f"{label}_final.json", {
            "theta": policy.estimator.theta, "H": policy.estimator.H,
            "completed_steps": policy.estimator.completed_steps,
            "eta": policy.estimator.eta, "lambda": policy.estimator.l2,
            "beta_mode": policy.beta_mode, "beta": policy.confidence_radius(),
            "array_payload_bytes": array_bytes})
    return rows, times


def figures(output, curves, scaling):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    for name, runs in curves.items():
        values = np.asarray(runs)
        mean = values.mean(axis=0)
        sem = values.std(axis=0, ddof=1) / np.sqrt(len(values))
        t = np.arange(1, len(mean)+1)
        ax.plot(t, mean, label=name)
        ax.fill_between(t, mean-sem, mean+sem, alpha=.15)
    ax.set(xlabel="Round", ylabel="Cumulative pseudo-regret", ylim=(0, None),
           title="Stationary Logistic bandit: mean +/- SEM over 5 seeds")
    ax.legend(frameon=False)
    fig.savefig(output / "stationary_regret.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    horizons = sorted({row["horizon"] for row in scaling})
    for key, name in [("update_us_per_step", "Update (total)"),
                      ("select_us_per_step", "Selection"),
                      ("linear_solve_us_per_step", "Update linear solves (subset)"),
                      ("log_us_per_step", "CSV row write")]:
        values = [np.median([row[key] for row in scaling if row["horizon"] == h]) for h in horizons]
        axes[0].plot(horizons, values, marker="o", label=name)
    axes[0].set(xlabel="Horizon", ylabel="Microseconds / step", ylim=(0, None),
                title="Instrumented costs: median of 3 runs")
    axes[0].legend(frameon=False, fontsize=8)
    payloads = [max(row["array_payload_bytes"] for row in scaling if row["horizon"] == h) for h in horizons]
    axes[1].plot(horizons, payloads, marker="o")
    axes[1].set(xlabel="Horizon", ylabel="Bytes (theta + H only)", ylim=(0, max(payloads)*1.3),
                title="Estimator array payload, fixed d=3")
    fig.savefig(output / "cost_and_state.png", dpi=160)
    plt.close(fig)


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = ROOT / "results" / f"d11_stationary_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    status = {"accepted": False, "solver_failures": 0, "scope": "E11 stationary experiment and cost measurements",
              "beta_mode": "paper_v2_literal", "stage": "setup", "git_sync_and_interpretation": "pending"}
    print(f"RESULTS: {output}", flush=True)
    try:
        rng = np.random.default_rng(20261001)
        features = rng.normal(size=(10, 3))
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        theta_star = np.array([1.0, -0.5, 0.75])
        config = {"d": 3, "K": 10, "T": 1000, "seeds": list(range(5)),
                  "feature_seed": 20261001, "features": features,
                  "theta_star_environment_only": theta_star,
                  "radius": 2.0, "eta": 3.0, "lambda": 126.0, "delta": 0.05,
                  "algorithms": ["random", "omd_paper_v2_literal"],
                  "beta_mode": "paper_v2_literal", "paper": "arXiv:2507.11847v2",
                  "beta_formula": "sqrt(4*lambda*S^2 + 2*eta*log(1/delta) + d*(6*eta^2+eta)*log1p(0.25/lambda))",
                  "paper_discrepancy": "PDF p6 literal beta omits time; Appendix A.1 p14 includes time. Literal reproduction, not a resolution of theoretical inconsistency.",
                  "smoke": {"seed": 20261011, "horizon": 50, "repeats": 2},
                  "profiling": {"horizons": [200, 1000, 5000], "seeds": [100, 101, 102],
                                "note": "fixed d,K; timings include instrumentation overhead; no retuning"},
                  "rng": "SeedSequence(seed).spawn(2): environment, policy. Same uniforms across algorithms do not imply identical observed rewards.",
                  "failure_policy": "stop, retain partial episode, no retries or fallback action"}
        write_json(output / "config.json", config)
        np.save(output / "features.npy", features)
        write_json(output / "versions.json", {"python": sys.version, **{
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pytest", "matplotlib")}})
        write_json(output / "git.json", {"head": command(["git", "rev-parse", "HEAD"]),
                                         "status": command(["git", "status", "--short", "--branch"])})
        snapshot(output)
        status["stage"] = "pytest"
        print("Running full pytest once...", flush=True)
        try:
            tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
        except subprocess.TimeoutExpired as exc:
            partial = exc.stdout or b""
            (output / "pytest.log").write_text(partial.decode(errors="replace") if isinstance(partial, bytes) else partial)
            raise
        (output / "pytest.log").write_text(tests.stdout, encoding="utf-8")
        status["pytest_returncode"] = tests.returncode
        print("\n".join(tests.stdout.splitlines()[-8:]), flush=True)
        if tests.returncode:
            raise RuntimeError("Tests failed; experiments not started")
        status["stage"] = "smoke"
        for algorithm in config["algorithms"]:
            first, _ = episode(output, f"smoke_{algorithm}_1", algorithm, 20261011, 50, features, theta_star, status)
            second, _ = episode(output, f"smoke_{algorithm}_2", algorithm, 20261011, 50, features, theta_star, status)
            if first != second:
                raise RuntimeError(f"Smoke replay mismatch: {algorithm}")
        status["smoke_replay_passed"] = True
        print("Smoke passed: both algorithms, exact trajectory replay; timings excluded.", flush=True)
        status["stage"] = "formal"
        curves, summaries = {}, []
        for algorithm in config["algorithms"]:
            curves[algorithm] = []
            for seed in config["seeds"]:
                rows, times = episode(output, f"formal_{algorithm}_seed{seed}", algorithm, seed, 1000,
                                      features, theta_star, status)
                curves[algorithm].append([row["cumulative_regret"] for row in rows])
                summaries.append({"algorithm": algorithm, "seed": seed, "horizon": 1000,
                                  "reward": sum(row["reward"] for row in rows),
                                  "final_regret": rows[-1]["cumulative_regret"],
                                  "select_seconds": sum(row["select_seconds"] for row in times),
                                  "update_seconds": sum(row["update_seconds"] for row in times),
                                  "array_payload_bytes": times[-1]["array_payload_bytes"],
                                  "constraint_active_count": sum(row["projection_active"] for row in times)})
                write_csv(output / "formal_summary.csv", summaries)
                print(f"{algorithm} seed={seed}: regret={rows[-1]['cumulative_regret']:.6f}", flush=True)
        status["formal_episodes_completed"] = len(summaries)
        status["stage"] = "profiling"
        scaling = []
        for horizon in config["profiling"]["horizons"]:
            for seed in config["profiling"]["seeds"]:
                meter = Meter()
                with instrument(meter):
                    _, times = episode(output, f"profile_T{horizon}_seed{seed}", "omd_paper_v2_literal", seed,
                                       horizon, features, theta_star, status, meter)
                sums = {key: sum(row[key] for row in times) for key in (
                    "select_seconds", "update_seconds", "logging_seconds", "update_linear_solve_seconds",
                    "update_other_seconds", "projection_inclusive_seconds")}
                scaling.append({"horizon": horizon, "seed": seed,
                                "update_us_per_step": sums["update_seconds"] / horizon * 1e6,
                                "select_us_per_step": sums["select_seconds"] / horizon * 1e6,
                                "log_us_per_step": sums["logging_seconds"] / horizon * 1e6,
                                "linear_solve_us_per_step": sums["update_linear_solve_seconds"] / horizon * 1e6,
                                "update_other_us_per_step": sums["update_other_seconds"] / horizon * 1e6,
                                "projection_inclusive_us_per_step": sums["projection_inclusive_seconds"] / horizon * 1e6,
                                "array_payload_bytes": times[-1]["array_payload_bytes"]})
                write_csv(output / "scaling_summary.csv", scaling)
            print(f"Profile T={horizon}: 3 repeats complete, array payload={scaling[-1]['array_payload_bytes']} bytes", flush=True)
        status["profile_runs_completed"] = len(scaling)
        status["stage"] = "report"
        figures(output, curves, scaling)
        lines = ["# E11 平稳闭环与计算记录", "", "配置：d=3，K=10，T=1000，5 seeds；S=2，eta=3，lambda=126，delta=0.05。",
                 "使用paper_v2_literal原式版本，未调整置信系数。PDF正文p6的beta缺少附录A.1 p14出现的时间因子，保持差异记录；不声称解决了理论歧义。", "",
                 "| 算法 | 最终伪遗憾均值 | SEM |", "|---|---|---|"]
        for name, values in curves.items():
            final = np.asarray(values)[:, -1]
            lines.append(f"| {name} | {final.mean():.6f} | {final.std(ddof=1)/np.sqrt(len(final)):.6f} |")
        lines.extend(["", "- 阴影为5 seeds的SEM，不是95%置信区间；单个固定动作集不能支持普遍排名。",
            "- 两策略的环境/动作随机源分离；相同种子不等于收到同一观测流，因为选臂不同。",
            "- Smoke重复用于轨迹一致性；正式10个episode没有逐一重跑。",
            "- 平稳闭环只验证当前固定参数环境，不延伸为非平稳保证。",
            "- Profiling固定d/K，三个长度各3次；包装函数本身有计时开销，不作为严格复杂度证明。",
            "- update_total含梯度、曲率、线性求解、投影、验证与状态赋值；linear_solve为其子集；update_other=total-linear_solve。",
            "- projection_inclusive包含投影内部线性求解，与linear_solve有重叠，不能相加。",
            "- selection含beta、输入检查、评分和argmax；日志计时为CSV行格式化和缓冲写入，不含最终flush/close与归档。",
            "- 状态数组theta+H为96字节，不含Python对象、计数器、临时求解工作区、环境与外部实验日志，不是峰值内存。",
            "- Random的0字节表示没有估计器数组，并非策略实际内存为零。",
            "- 所有更新异常停止并保留现场，不重试、不改参数、不静默回退。",
            "", "自动计算验收已完成；用户解读与最终commit/push另行记录。"])
        (output / "observations.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        status.update(accepted=True, stage="complete")
    except Exception as exc:
        status["failure"] = f"{type(exc).__name__}: {exc}"
        (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"FAIL: {status['failure']}", flush=True)
    finally:
        write_json(output / "acceptance.json", status)
        write_json(output / "manifest_sha256.json", {
            p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.rglob("*")) if p.is_file()})
        archive = ROOT / "reports" / "d11" / stamp
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(output, archive)
        package = output.parent / f"{output.name}_evidence.zip"
        with ZipFile(package, "w", ZIP_DEFLATED) as zipped:
            for p in sorted(output.rglob("*")):
                if p.is_file():
                    zipped.write(p, f"{output.name}/{p.relative_to(output).as_posix()}")
        print(f"ARCHIVE: {archive}\nEVIDENCE_ZIP: {package}", flush=True)
        print(f"E11 automated acceptance: {'PASS' if status['accepted'] else 'FAIL'}", flush=True)
    return 0 if status["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
