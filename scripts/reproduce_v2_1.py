"""Fresh-output E10--E14 acceptance; --verify-only never runs experiments."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from zipfile import ZipFile, ZIP_DEFLATED

from d8_batch_reference import ROOT, command, write_csv, write_json
from d10_fixed_stream import compare, verify_input
from d11_stationary import Meter, episode, figures, instrument
from scripts.d12_logistic_stress import parameter_path, run_episode, validate_episode
import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def persist_complete_rows(path, rows):
    # Keep streaming partial logs on failure; atomically publish the complete returned trajectory.
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    temporary = path.with_suffix(path.suffix + ".complete")
    temporary.write_text(buffer.getvalue(), encoding="utf-8", newline="")
    temporary.replace(path)
    if len(read_csv(path)) != len(rows):
        raise RuntimeError(f"Incomplete CSV write: {path}")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files():
    paths = set()
    for folder in ("algorithms", "envs", "estimators", "scripts", "tests", "configs", "docs"):
        paths.update(p for p in (ROOT / folder).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".json", ".md", ".sh"))
    paths.update(ROOT.glob("requirements*.txt"))
    paths.update(ROOT.glob("*.md"))
    paths.update(ROOT / name for name in ("run.py", "reproduce_v2_1.sh"))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def fresh_output(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Output must be absent or empty; choose a new directory")
    path.mkdir(parents=True, exist_ok=True)


def checked_manifest(directory):
    manifest = read_json(directory / "manifest_sha256.json")
    for name, expected in manifest.items():
        p = (directory / name).resolve()
        if not p.is_relative_to(directory.resolve()) or not p.is_file() or digest(p) != expected:
            raise ValueError(f"Archive hash mismatch: {name}")
    return manifest


def verify(output, check_source=True):
    manifest = read_json(output / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Suite is not complete")
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Output file set mismatch")
    for name, expected in manifest["files"].items():
        if digest(output / name) != expected:
            raise ValueError(f"Output hash mismatch: {name}")
    if check_source and source_files() != read_json(output / "source_sha256.json"):
        raise ValueError("Current source differs from accepted source")
    with ZipFile(output / "source_snapshot.zip") as archive:
        hashes = read_json(output / "source_sha256.json")
        if set(archive.namelist()) != set(hashes):
            raise ValueError("Source snapshot file set mismatch")
        if any(hashlib.sha256(archive.read(k)).hexdigest() != v for k, v in hashes.items()):
            raise ValueError("Source snapshot hash mismatch")
    config = read_json(output / "config.json")
    features = np.load(output / "features.npy", allow_pickle=False)
    stress_count = 0
    for scenario in config["scenarios"]:
        path = parameter_path(scenario, config["horizon"], config["theta_before"],
                              config["theta_after"], config["change_step"])
        np.testing.assert_allclose(np.load(output / "stress" / f"{scenario}_theta.npy"), path, atol=0, rtol=0)
        for algorithm in config["algorithms"]:
            for seed in config["seeds"]:
                label = f"{scenario}_{algorithm}_seed{seed}"
                rows = read_csv(output / "stress" / f"{label}.csv")
                validate_episode(rows, features, path, scenario, algorithm, seed)
                stress_count += 1
                if algorithm != "random":
                    state = read_json(output / "stress" / f"{label}_state.json")
                    if state["completed_steps"] != len(path) or state["array_payload_bytes"] != 96:
                        raise ValueError("Stress state mismatch")
                    np.linalg.cholesky(np.asarray(state["H"]))
    means = 1 / (1 + np.exp(-(features @ np.asarray(config["theta_before"]))))
    stationary_count = 0
    for algorithm in config["algorithms"]:
        for seed in config["seeds"]:
            rows = read_csv(output / "stationary/episodes" / f"formal_{algorithm}_seed{seed}.csv")
            if len(rows) != config["horizon"]:
                raise ValueError("Incomplete stationary trajectory")
            cumulative = 0.0
            for step, row in enumerate(rows, 1):
                if (int(row["step"]), row["algorithm"], int(row["seed"])) != (step, algorithm, seed):
                    raise ValueError("Stationary identity mismatch")
                action = int(row["action"])
                if not 0 <= action < len(features) or int(row["reward"]) not in (0, 1):
                    raise ValueError("Invalid stationary action/reward")
                regret = means.max() - means[action]
                cumulative += regret
                np.testing.assert_allclose([float(row["instant_regret"]), float(row["cumulative_regret"])],
                                           [regret, cumulative], atol=1e-9, rtol=0)
            stationary_count += 1
    for horizon in config["profile_horizons"]:
        for seed in config["profile_seeds"]:
            label = f"profile_T{horizon}_seed{seed}"
            base = output / "stationary/episodes"
            rows = read_csv(base / f"{label}.csv")
            times = read_csv(base / f"{label}_timing.csv")
            state = read_json(base / f"{label}_final.json")
            if len(rows) != horizon or len(times) != horizon or state["completed_steps"] != horizon:
                raise ValueError("Incomplete profile run")
            if state["array_payload_bytes"] != 96:
                raise ValueError("Profile state mismatch")
            for step, row in enumerate(times, 1):
                if int(row["step"]) != step or int(row["array_payload_bytes"]) != 96:
                    raise ValueError("Profile timing identity mismatch")
                if any(not np.isfinite(float(v)) or float(v) < 0 for k, v in row.items() if k.endswith("seconds")):
                    raise ValueError("Invalid profile timing")
    checked_manifest(output / "input_d8")
    fits = list((output / "fixed_stream/batch_fits").glob("n*.json"))
    if len(fits) != 5 or any(not read_json(p)["accepted"] for p in fits):
        raise ValueError("Batch acceptance mismatch")
    if len(read_csv(output / "fixed_stream/updates.csv")) != 200:
        raise ValueError("Incomplete fixed-stream updates")
    if len(list((output / "figures").glob("*.png"))) != 5:
        raise ValueError("Expected five key figures")
    if manifest["solver_failures"] or stress_count != 20 or stationary_count != 10:
        raise ValueError("Acceptance count/failure mismatch")
    return manifest


def run_stationary(output, config, features, status):
    output.mkdir()
    truth = np.asarray(config["theta_before"])
    curves, summary, scaling = {}, [], []
    for algorithm in config["algorithms"]:
        curves[algorithm] = []
        for seed in config["seeds"]:
            rows, times = episode(output, f"formal_{algorithm}_seed{seed}", algorithm, seed,
                                  config["horizon"], features, truth, status)
            persist_complete_rows(output / "episodes" / f"formal_{algorithm}_seed{seed}.csv", rows)
            curves[algorithm].append([r["cumulative_regret"] for r in rows])
            summary.append({"algorithm": algorithm, "seed": seed, "final_regret": rows[-1]["cumulative_regret"],
                            "reward": sum(r["reward"] for r in rows)})
    write_csv(output / "summary.csv", summary)
    for horizon in config["profile_horizons"]:
        for seed in config["profile_seeds"]:
            meter = Meter()
            with instrument(meter):
                profile_rows, times = episode(output, f"profile_T{horizon}_seed{seed}", "omd_paper_v2_literal",
                                   seed, horizon, features, truth, status, meter)
            persist_complete_rows(output / "episodes" / f"profile_T{horizon}_seed{seed}.csv", profile_rows)
            scaling.append({"horizon": horizon, "seed": seed,
                **{target: sum(r[source] for r in times) / horizon * 1e6 for target, source in [
                    ("update_us_per_step", "update_seconds"), ("select_us_per_step", "select_seconds"),
                    ("linear_solve_us_per_step", "update_linear_solve_seconds"),
                    ("update_other_us_per_step", "update_other_seconds"),
                    ("projection_inclusive_us_per_step", "projection_inclusive_seconds"),
                    ("log_us_per_step", "logging_seconds")]}, "array_payload_bytes": times[-1]["array_payload_bytes"]})
    write_csv(output / "scaling_summary.csv", scaling)
    figures(output, curves, scaling)
    for name in ("stationary_regret.png", "cost_and_state.png"):
        shutil.move(output / name, output.parent / "figures" / name)
    return curves, scaling


def run_stress(output, config, features, status):
    output.mkdir()
    summary, curves = [], {}
    for scenario in config["scenarios"]:
        path = parameter_path(scenario, config["horizon"], config["theta_before"],
                              config["theta_after"], config["change_step"])
        np.save(output / f"{scenario}_theta.npy", path)
        means = np.exp(-np.logaddexp(0, -(path @ features.T)))
        np.save(output / f"{scenario}_means.npy", means)
        for algorithm in config["algorithms"]:
            curves[scenario, algorithm] = []
            for seed in config["seeds"]:
                rows = run_episode(output, scenario, algorithm, seed, features, path, config, status)
                persist_complete_rows(output / f"{scenario}_{algorithm}_seed{seed}.csv", rows)
                validate_episode(rows, features, path, scenario, algorithm, seed)
                curves[scenario, algorithm].append(rows)
                row = {"scenario": scenario, "algorithm": algorithm, "seed": seed,
                       "final_dynamic_pseudo_regret": rows[-1]["cumulative_regret"],
                       "reward": sum(r["reward"] for r in rows)}
                for label, start, stop in (("first_100", 0, 100), ("prechange_100", 400, 500),
                                            ("postchange_100", 500, 600), ("last_100", 900, 1000)):
                    part = rows[start:stop]
                    row[f"{label}_regret_per_step"] = float(np.mean([r["instant_regret"] for r in part]))
                    row[f"{label}_prediction_rmse"] = (float(np.mean([r["prediction_rmse_pre"] for r in part]))
                                                      if algorithm != "random" else "")
                summary.append(row)
                write_csv(output / "summary.csv", summary)
                print(f"{scenario} {algorithm} seed={seed}: dynamic pseudo-regret={row['final_dynamic_pseudo_regret']:.6f}", flush=True)
    return curves, summary


def make_report(output, config, stationary, scaling, curves, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fixed = read_csv(output / "fixed_stream/summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for algorithm in ("omd", "batch"):
        rows = [r for r in fixed if r["algorithm"] == algorithm]
        n = [int(r["n"]) for r in rows]
        axes[0].plot(n, [float(r["prediction_rmse"]) for r in rows], "o-", label=algorithm)
        axes[1].plot(n, [float(r["cumulative_call_seconds"]) * 1000 for r in rows], "o-",
                     label="OMD: n updates" if algorithm == "omd" else "Batch: up to 5 fits")
    axes[0].set(xlabel="Observed samples", ylabel="Probability RMSE", title="Same fixed stream and evaluation points")
    axes[1].set(xlabel="Observed samples", ylabel="Cumulative call time (ms)", title="Different workloads; not a speed ranking")
    for ax in axes:
        ax.legend(frameon=False)
    fig.savefig(output / "figures/fixed_stream.png", dpi=160)
    plt.close(fig)
    for scenario in config["scenarios"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        t = np.arange(1, config["horizon"] + 1)
        for algorithm in config["algorithms"]:
            values = np.array([[r["cumulative_regret"] for r in rows] for rows in curves[scenario, algorithm]])
            avg, sem = values.mean(0), values.std(0, ddof=1) / np.sqrt(len(values))
            axes[0].plot(t, avg, label=algorithm)
            axes[0].fill_between(t, avg-sem, avg+sem, alpha=.15)
        prediction = np.array([[r["prediction_rmse_pre"] for r in rows]
                               for rows in curves[scenario, "omd_paper_v2_literal"]])
        avg, sem = prediction.mean(0), prediction.std(0, ddof=1) / np.sqrt(len(prediction))
        axes[1].plot(t, avg, label="OMD pre-update probability RMSE")
        axes[1].fill_between(t, avg-sem, avg+sem, alpha=.15)
        for ax in axes:
            if scenario == "abrupt":
                ax.axvline(config["change_step"], color="black", linestyle="--", alpha=.5)
            ax.set_xlabel("Round")
            ax.legend(fontsize=8, frameon=False)
        axes[0].set_ylabel("Cumulative dynamic pseudo-regret")
        axes[1].set_ylabel("RMSE across all 10 arms")
        fig.suptitle(f"{scenario.capitalize()} stress: mean +/- SEM over 5 seeds")
        fig.savefig(output / f"figures/stress_{scenario}.png", dpi=160)
        plt.close(fig)
    lines = ["# Bandit v2.1 阶段报告", "", "本报告由本次原始日志生成；时间数据属于本次机器及运行。", "",
             "## 范围与配置", "", "E8/E9 既有验收由归档和本次回归测试覆盖；本次重新执行 E10、E11、E12。",
             "固定 d=3、K=10、T=1000、seeds=0..4；eta=3、lambda=126、S=2、delta=0.05。",
             "正文原式 paper_v2_literal；保留论文正文与附录的时间因子差异，不主张已经消除理论歧义。",
             "OMD 核心没有加入遗忘、重启、变化检测或调参。非平稳实验属于压力测试，不是 DOMD-GLB。", "",
             "## 固定数据流与闭环的区别", "",
             "E10 使用归档 E8 的同一 X/y 前缀及固定评估点；OMD 做 200 次更新，Batch 仅在 10/25/50/100/200 五个前缀从零重拟合。",
             "E10 使用 eta=2、H1=I 的工程参数，Batch 使用损失求和+l2=1 的正则；两者不要求估计参数相等。",
             "E11/E12 中选臂改变下一条观测。种子相同不代表两策略收到同一观测流。真参数仅用于环境与外部评估。",
             "![Fixed stream](figures/fixed_stream.png)", "", "## 平稳闭环", "",
             "| 算法 | 最终累计伪遗憾均值 | SEM |", "|---|---:|---:|"]
    for name, runs in stationary.items():
        final = np.asarray(runs)[:, -1]
        lines.append(f"| {name} | {final.mean():.6f} | {final.std(ddof=1)/np.sqrt(len(final)):.6f} |")
    lines.extend(["", "![Stationary](figures/stationary_regret.png)", "", "## 计算与状态", "",
                  "| T | 更新 μs/步（3 次中位数） | 选择 μs/步 | theta+H 字节 |", "|---|---:|---:|---:|"])
    for horizon in config["profile_horizons"]:
        rr = [r for r in scaling if r["horizon"] == horizon]
        lines.append(f"| {horizon} | {np.median([r['update_us_per_step'] for r in rr]):.3f} | {np.median([r['select_us_per_step'] for r in rr]):.3f} | 96 |")
    lines.extend(["", "计时包含 instrumentation 开销；更新包含求解、投影、曲率及验证。线性求解是更新的子集；投影整体时间与线性求解有重叠，不能相加。",
        "日志仅计 CSV 行格式化及缓冲写入，不含最终 flush/close。状态数组载荷不是峰值内存；实验日志存储随 T 增长。",
        "固定 d 的状态数组大小由代码和记录共同支持；这些时间点不能证明渐近复杂度或普遍速度优势。",
        "![Costs](figures/cost_and_state.png)", "", "## 非平稳压力测试", "",
        "突变：第1–500轮 theta=(1,-0.5,0.75)，第501轮起 theta=(-1,0.5,-0.75)。",
        "漂移：alpha=(t-1)/999，在同一端点之间线性插值。采样、预测评估和 regret 均使用本轮 theta_t。",
        "动态伪遗憾为 sum_t [max_a sigmoid(x_a^T theta_t)-sigmoid(x_A_t^T theta_t)]，比较点是每轮最优臂。",
        "预测 RMSE 在本轮更新前对全部10个臂计算，避免用刚到的标签评估同一次预测。", "",
        "| 场景 | 算法 | 最终动态伪遗憾均值 | SEM |", "|---|---|---:|---:|"])
    for scenario in config["scenarios"]:
        for alg in config["algorithms"]:
            vals = np.array([r["final_dynamic_pseudo_regret"] for r in summary if r["scenario"] == scenario and r["algorithm"] == alg])
            lines.append(f"| {scenario} | {alg} | {vals.mean():.6f} | {vals.std(ddof=1)/np.sqrt(len(vals)):.6f} |")
    lines.extend(["", "具体观察（同一场景跨5 seeds、每个窗口100轮均值）：", ""])
    for scenario in config["scenarios"]:
        rr = [r for r in summary if r["scenario"] == scenario and r["algorithm"] != "random"]
        avg = lambda field: np.mean([r[field] for r in rr])
        lines.append(f"- {scenario}：401–500轮预测 RMSE={avg('prechange_100_prediction_rmse'):.6f}，501–600轮={avg('postchange_100_prediction_rmse'):.6f}，901–1000轮={avg('last_100_prediction_rmse'):.6f}；对应每轮动态伪遗憾为 {avg('prechange_100_regret_per_step'):.6f}、{avg('postchange_100_regret_per_step'):.6f}、{avg('last_100_regret_per_step'):.6f}。")
    lines.extend(["", "解释候选：累积 H 没有遗忘旧曲率，更新幅度会受历史状态影响；旧参数也可能在突变后失配。",
        "替代解释：有限预算、奖励噪声、特征几何及较保守的固定置信半径也会影响预测与选臂；当前闭环无法把这些因素单独归因。",
        "后续可用完全相同的固定数据流比较保留状态与已知变化点重置；这是独立诊断，尚未执行，不纳入本阶段成果。",
        "渐变穿过零参数附近时各臂均值接近，regret 较小可能来自问题暂时容易，不能仅据此断言跟踪更好。",
        "![Abrupt](figures/stress_abrupt.png)", "", "![Drift](figures/stress_drift.png)", "",
        "## 正确性、失败与复现", "",
        "全套 pytest 日志见 pytest.log；受约束更新的独立 SLSQP 对照覆盖约束活跃与不活跃情况。",
        "新增测试覆盖突变边界、漂移端点、退化平稳一致性、重复轨迹与错误指标拒绝；已有极端输入测试覆盖稳定 Logistic 计算。",
        "全部求解异常计数并停止；不静默跳过、回退或自动重试。本次 solver_failures=0。",
        "源代码快照、Git基线、依赖版本及原始CSV均保留；manifest验证文件全集、哈希、源码和逐轮指标。",
        "从全新目录执行 python scripts/reproduce_v2_1.py --output-root artifacts/v2_1_local；随后对同目录加 --verify-only。",
        "历史 E11 轨迹对照见 historical_comparison.json；若跨版本数值不一致则显式披露，不改 seed 或阈值追求旧数字。", "",
        "## 人工验收与冻结", "",
        "计算门槛通过不自动代表独立讲解已完成。请按 docs/v2_1_walkthrough.md 做约10分钟讲解，在阶段收尾记录中写明结论。",
        "本报告覆盖工程线；不代替论文最小验证A、数学闭卷复盘或理论线验收。版本提交与标签以 Git 中的实际状态为准。"])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/v2_1_local")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if args.verify_only:
        verify(output)
        print("v2.1 verification: PASS (read-only)")
        return
    archive = ROOT / "reports/v2_1" / output.name
    package = output.with_name(output.name + "_evidence.zip")
    if archive.exists() or package.exists():
        raise ValueError("Archive or evidence ZIP already exists; choose a new output-root name")
    fresh_output(output)
    status = {"status": "running", "stage": "setup", "solver_failures": 0,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "human_walkthrough": "pending", "git_freeze": "pending"}
    try:
        config = read_json(ROOT / "configs/v2_1.json")
        write_json(output / "config.json", config)
        write_json(output / "git.json", {"head": command(["git", "rev-parse", "HEAD"]),
                                         "status": command(["git", "status", "--short", "--branch"])})
        write_json(output / "versions.json", {"python": sys.version, **{name: importlib.metadata.version(name)
                    for name in ("numpy", "scipy", "matplotlib", "pytest")}})
        hashes = source_files()
        write_json(output / "source_sha256.json", hashes)
        with ZipFile(output / "source_snapshot.zip", "w", ZIP_DEFLATED) as archive:
            for name in hashes:
                archive.write(ROOT / name, name)
        source = ROOT / config["d8_archive"]
        verify_input(source.resolve())
        shutil.copytree(source, output / "input_d8")
        (output / "figures").mkdir()
        status["stage"] = "pytest"
        print("Running full pytest once...", flush=True)
        result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
                                capture_output=True, text=True, timeout=180)
        (output / "pytest.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        print(result.stdout, end="", flush=True)
        if result.returncode:
            raise RuntimeError("pytest failed; experiments not started")
        status["stage"] = "fixed_stream"
        fixed = output / "fixed_stream"
        fixed.mkdir()
        e8 = read_json(source / "config.json")
        fixed_config = {"prefixes": e8["prefixes"], "batch": e8["solver"],
                        "omd": {"eta": 2.0, "l2": 1.0, "radius": 2.0}}
        write_json(fixed / "config.json", fixed_config)
        compare(fixed, output / "input_d8", fixed_config, status)
        rng = np.random.default_rng(config["feature_seed"])
        features = rng.normal(size=(config["num_arms"], config["dimension"]))
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        np.save(output / "features.npy", features)
        status["stage"] = "stationary_and_profile"
        stationary, scaling = run_stationary(output / "stationary", config, features, status)
        historical = ROOT / config["d11_archive"]
        checked_manifest(historical)
        if not read_json(historical / "acceptance.json")["accepted"]:
            raise ValueError("Historical E11 archive was not accepted")
        comparison = {}
        for name in (output / "stationary/episodes").glob("formal_*.csv"):
            if not name.stem.endswith("_timing"):
                comparison[name.name] = name.read_bytes() == (historical / "episodes" / name.name).read_bytes()
        write_json(output / "historical_comparison.json", {"reference": config["d11_archive"],
                    "reference_manifest_sha256": digest(historical / "manifest_sha256.json"),
                    "exact_trajectory_matches": comparison, "timings_compared": False})
        print(f"Historical E11 exact trajectory matches: {sum(comparison.values())}/10", flush=True)
        status["stage"] = "stress"
        curves, summary = run_stress(output / "stress", config, features, status)
        status["stage"] = "report"
        make_report(output, config, stationary, scaling, curves, summary)
        status.update(status="complete", stage="complete", formal_stationary_episodes=10,
                      stress_episodes=20, profile_runs=9, figures=5)
        status["files"] = {p.relative_to(output).as_posix(): digest(p)
                           for p in sorted(output.rglob("*")) if p.is_file()}
        write_json(output / "manifest.json", status)
        verify(output)
        archive = ROOT / "reports/v2_1" / output.name
        if archive.exists():
            raise ValueError("Archive name already exists; choose a new output-root name")
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(output, archive)
        package = output.with_name(output.name + "_evidence.zip")
        with ZipFile(package, "w", ZIP_DEFLATED) as zipped:
            for p in sorted(output.rglob("*")):
                if p.is_file():
                    zipped.write(p, f"{output.name}/{p.relative_to(output)}")
        print(f"ARCHIVE: {archive}\nEVIDENCE_ZIP: {package}\nv2.1 automated acceptance: PASS", flush=True)
    except Exception as exc:
        status.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(output / "manifest.json", status)
        print(f"FAILED; retained evidence: {output}", flush=True)
        raise


if __name__ == "__main__":
    main()
