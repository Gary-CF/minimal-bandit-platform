"""Build a report from frozen test artifacts; never run or tune a policy."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.run_logistic_benchmark import digest, read_csv, write_json

LABELS = {"random": "Random", "ucb1": "UCB1", "sw_ucb": "SW-UCB",
          "discounted_ucb": "Discounted-UCB", "omd_paper_v2_literal": "OMD",
          "domd_paper_v1": "DOMD", "scb_weight_ucb_numeric_v1": "SCB numeric"}


def read_json(path):
    return json.loads(path.read_text())


def check_hashes(folder, top=True):
    manifest = read_json(folder / "manifest.json")
    if not top:
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
        if actual != set(manifest) | {"manifest.json"}:
            raise ValueError(f"File set mismatch: {folder}")
    for name, expected in manifest.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or digest(path) != expected:
            raise ValueError(f"Archive hash mismatch: {path}")
        if top and path.name == "manifest.json":
            check_hashes(path.parent, top=False)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "| " + " | ".join(["---"] * len(headers)) + " |",
                       *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def main(run_dir, output):
    check_hashes(run_dir)
    protocol = read_json(run_dir / "protocol.json")
    summary = read_json(run_dir / "test_aggregate.json")
    grouped = read_json(run_dir / "test_by_feature.json")
    audit = read_json(run_dir / "test_audit.json")
    by_name = {r["algorithm"]: r for r in summary}
    algorithms = protocol["algorithms"]
    scenarios = protocol["base_config"]["scenarios"]
    features = protocol["test_tasks"]["feature_seeds"]
    seeds = protocol["base_config"]["seeds"]
    T = protocol["base_config"]["horizon"]
    change = protocol["base_config"]["change_step"] - 1
    focus = ("ucb1", "omd_paper_v2_literal", "domd_paper_v1", "scb_weight_ucb_numeric_v1")
    curves = {(a, s): [] for a in algorithms for s in scenarios}
    instant, rmse = {a: [] for a in focus}, {a: [] for a in focus}
    early, late = {a: [] for a in algorithms}, {a: [] for a in algorithms}
    for feature in features:
        for alg in algorithms:
            bundle = run_dir / f"feature{feature}_{alg}"
            for scenario in scenarios:
                for seed in seeds:
                    name = f"{scenario}__{alg}__seed{seed}.csv"
                    rows = read_csv(bundle / "episodes" / name)
                    timing = read_csv(bundle / "timing" / name)
                    if len(rows) != T or len(timing) != T:
                        raise ValueError("Incomplete trajectory")
                    curves[alg, scenario].append([float(r["cumulative_regret"]) for r in rows])
                    early[alg].append(np.mean([float(r["update_seconds"]) for r in timing[:100]]) * 1e6)
                    late[alg].append(np.mean([float(r["update_seconds"]) for r in timing[-100:]]) * 1e6)
                    if scenario == "abrupt" and alg in focus:
                        instant[alg].append([float(r["instant_regret"]) for r in rows])
                        if rows[0]["prediction_rmse_pre"] != "":
                            rmse[alg].append([float(r["prediction_rmse_pre"]) for r in rows])
    for alg in algorithms:
        scenario_means = []
        for scenario in scenarios:
            value = float(np.mean(np.asarray(curves[alg, scenario])[:, -1]))
            np.testing.assert_allclose(value, by_name[alg][scenario], rtol=0, atol=1e-9)
            scenario_means.append(value)
        np.testing.assert_allclose(np.mean(scenario_means), by_name[alg]["regret"], rtol=0, atol=1e-9)

    output.mkdir(parents=True, exist_ok=False)
    colors = dict(zip(algorithms, plt.get_cmap("tab10").colors))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(15, 4), layout="constrained")
    for ax, scenario in zip(np.atleast_1d(axes), scenarios):
        for alg in algorithms:
            ax.plot(np.arange(1, T + 1), np.mean(curves[alg, scenario], axis=0),
                    label=LABELS[alg], color=colors[alg], linestyle="--" if alg == "sw_ucb" else "-")
        if scenario == "abrupt":
            ax.axvline(change + 1, color="gray", linestyle=":")
        ax.set(title=scenario, xlabel="Round", ylabel="Mean cumulative pseudo-regret")
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8)
    fig.savefig(output / "regret.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, source, ylabel in zip(axes, (instant, rmse), ("Instant pseudo-regret", "Pre-update probability RMSE")):
        for alg in focus:
            if not source[alg]:
                continue
            mean = np.mean(source[alg], axis=0)
            ax.plot(np.arange(50, T + 1), np.convolve(mean, np.ones(50) / 50, mode="valid"),
                    label=LABELS[alg], color=colors[alg])
        ax.axvline(change + 1, color="gray", linestyle=":")
        ax.set(title="Abrupt: 50-round trailing mean", xlabel="Round", ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(output / "abrupt_diagnostics.png", dpi=160)
    plt.close(fig)

    models = list(focus[1:])
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    x = np.arange(len(models))
    for offset, source, label in ((-.18, early, "First 100 rounds"), (.18, late, "Last 100 rounds")):
        ax.bar(x + offset, [np.mean(source[a]) for a in models], width=.36, label=label)
    ax.set_xticks(x, [LABELS[a] for a in models])
    ax.set(ylabel="Mean update time (microseconds)", title="Observed update cost on this machine")
    ax.legend()
    fig.savefig(output / "update_cost.png", dpi=160)
    plt.close(fig)

    windows = {"pre": slice(change - 100, change), "post": slice(change, change + 100), "late": slice(T - 100, T)}
    window_rows = []
    for alg in focus:
        for metric, source in (("instant_regret", instant), ("prediction_rmse_pre", rmse)):
            if source[alg]:
                mean = np.mean(source[alg], axis=0)
                window_rows.append(dict(algorithm=alg, metric=metric,
                                        **{k: float(np.mean(mean[v])) for k, v in windows.items()}))
    write_json(output / "abrupt_windows.json", window_rows)
    domd = by_name["domd_paper_v1"]["regret"]
    reduction = {a: 100 * (1 - domd / by_name[a]["regret"]) for a in focus if a != "domd_paper_v1"}
    contents = ["# Bandit v2.2：冻结 Logistic 对照实验", "",
        f"固定配置后完成 {len(algorithms) * len(features) * len(scenarios) * len(seeds)} 局测试；每局 {T} 轮。测试结果未用于重新选择参数。",
        "", "## 实验口径", "",
        f"特征种子：{features}；奖励种子：{seeds}。每个算法 45 个任务，每个场景 15 个任务。",
        "总体结果对三个场景等权。配对基于相同特征、场景和奖励种子，策略可能接收不同观测。特征组只有三组，不将全部任务视为独立环境样本计算总体 SEM。",
        "", "冻结设置：", "", "```json", json.dumps(protocol["settings"], ensure_ascii=False, indent=2), "```", "",
        "## 测试结果", "",
        table(["算法", "总体", "平稳", "突变", "漂移", "减去 UCB1"],
              [[r["algorithm"], *[f"{r[k]:.4f}" for k in ("regret", "stationary", "abrupt", "drift", "diff_ucb1")]] for r in summary]),
        "", "![平均累计遗憾](regret.png)", "",
        f"DOMD 平均遗憾为 {domd:.4f}，相对 UCB1、OMD、SCB 分别降低 {reduction['ucb1']:.1f}%、{reduction['omd_paper_v2_literal']:.1f}%、{reduction['scb_weight_ucb_numeric_v1']:.1f}%。",
        "这些是当前任务族和冻结完整策略配置的比较；估计器、正则及探索规则并未统一，不能把全部差异归因于折扣机制。",
        "", "## 按特征组检查", "",
        table(["特征种子", *[LABELS[a] for a in focus]],
              [[f, *[f"{next(r['regret'] for r in grouped if r['feature_seed'] == f and r['algorithm'] == a):.4f}" for a in focus]] for f in features]),
        "", "## 突变后的行为", "",
        "下表分别为突变前 100 轮、突变后 100 轮和最后 100 轮的平均指标。图中使用 50 轮向后滑动平均，不用于调参。", "",
        table(["算法", "指标", "pre100", "post100", "late100"],
              [[r["algorithm"], r["metric"], *[f"{r[k]:.6f}" for k in ("pre", "post", "late")]] for r in window_rows]),
        "", "![突变诊断](abrupt_diagnostics.png)", "",
        "总遗憾应与分段指标一起解释：分别评价突变前积累的优势、突变后的短期代价和后期恢复，不能用总遗憾排名替代瞬时适应表现。",
        "## 计算开销与投影审查", "",
        table(["算法", "选臂 µs", "更新 µs", "前100轮更新 µs", "末100轮更新 µs"],
              [[a, f"{by_name[a]['select_us']:.2f}", f"{by_name[a]['update_us']:.2f}", f"{np.mean(early[a]):.2f}", f"{np.mean(late[a]):.2f}"] for a in algorithms]),
        "", "![更新开销](update_cost.png)", "",
        f"SCB 共 {audit['scb_updates']} 次更新，触发非凸投影 {audit['scb_projection_steps']} 次。",
        "耗时来自本次运行所在机器，包含现有计时与记录方式的影响；前后两段耗时不能单独证明渐近复杂度。",
        "", "## 结论范围", "",
        "SW-UCB 的窗口等于测试时长，因此本轮决策没有遗忘历史。SCB 使用原文置信半径及正则公式；未遍历其全部超参数。结论限定于当前任务族、轮数、候选参数和代码版本。",
        "", "## 复现记录", "",
        f"输入目录：`{run_dir.resolve()}`。输入根清单及各子实验清单的哈希检查通过；报告曲线终值与冻结汇总交叉检查通过。",
        "模型源码快照和逐轮数据保留在测试目录；冻结协议含验证依据。报告生成器只读取记录，不运行算法或调参。", ""]
    (output / "report.md").write_text("\n".join(contents), encoding="utf-8")
    write_json(output / "provenance.json", dict(test_manifest_sha256=digest(run_dir / "manifest.json"),
               protocol_sha256=digest(run_dir / "protocol.json"), report_script_sha256=digest(Path(__file__)),
               input_directory=str(run_dir.resolve())))
    write_json(output / "manifest.json", {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    print(f"REPORT PASS: {output.resolve() / 'report.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.run, args.output)
