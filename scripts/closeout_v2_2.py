"""Read-only experiment verification, final tables, and an explicit Git path list."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "reports/v2_2/release"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def release_paths(root=ROOT):
    values = load(root / "configs/v2_2/release_files.json")
    if not isinstance(values, list) or len(values) != len(set(values)):
        raise ValueError("invalid/duplicate release file list")
    for value in values:
        rel = Path(value)
        path = (root / rel).resolve()
        if (rel.is_absolute() or ".." in rel.parts or "\n" in value
                or not path.is_relative_to(root.resolve()) or not path.is_file()
                or value.startswith(("artifacts/", "reports/v2_1/"))):
            raise ValueError(f"invalid release path: {value}")
    return values


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *("| " + " | ".join(str(x) for x in row) + " |" for row in rows)])


def check_budget(frozen, smoke, full):
    paired = load(frozen / "test_paired.json")
    if len(paired) != 315:
        raise ValueError("Expected the original 315-episode test")
    if len(load(smoke / "protocol.json")["tasks"]) != 71:
        raise ValueError("Expected 71 smoke tasks")
    counts = Counter(t["stage"] for t in load(full / "protocol.json")["tasks"])
    if counts != Counter(logistic=882, heavy_fixed=81, heavy_closed=324, cost=45):
        raise ValueError(f"Unexpected supplemental task budget: {dict(counts)}")
    return dict(counts)


def report_text(frozen_rows, rows, counts, source_label):
    short = {"ucb1": "UCB1", "omd_paper_v2_literal": "OMD", "domd_paper_v1": "DOMD",
             "scb_weight_ucb_numeric_v1": "SCB numerical", "random": "Random", "sw_ucb": "SW-UCB",
             "discounted_ucb": "Discounted UCB", "ridge_greedy": "Ridge greedy",
             "linucb_nominal": "LinUCB nominal", "hvt_ucb_literal": "Hvt literal",
             "hvt_squared_ablation": "Hvt squared ablation"}
    fmt = lambda x: f"{x:.2f}"
    parts = ["# Bandit v2.2 最终实验报告", "", f"数据来源：{source_label}。所有表格由该运行归档生成；耗时属于生成报告的归档所记录机器。", "",
             "正式原冻结测试 315 局；补充正式任务 1332 项；另有 71 项 smoke。原测试使用验证阶段冻结的参数，补充实验不重新选参。", "",
             "## 1. 原冻结测试", "", "T=1000、d=3、K=10，每个算法 45 个任务。累计伪遗憾越低越好。", "",
             table(["算法", "总体", "平稳", "突变", "线性漂移"],
                   [[short[r["algorithm"]], *[fmt(r[k]) for k in ("regret", "stationary", "abrupt", "drift")]] for r in frozen_rows])]
    ref = {r["algorithm"]: r for r in frozen_rows}
    reduction_ucb = 100 * (1 - ref["domd_paper_v1"]["regret"] / ref["ucb1"]["regret"])
    reduction_omd = 100 * (1 - ref["domd_paper_v1"]["regret"] / ref["omd_paper_v2_literal"]["regret"])
    parts += ["", f"DOMD 总体遗憾相对 UCB1 降低 {reduction_ucb:.1f}%，相对 OMD 降低 {reduction_omd:.1f}%。这是完整策略配置比较，不能将全部提升归因于折扣。SW W=1000 与 T 相同，因此与 UCB1 一致。", "",
              "## 2. Logistic 的 K 扩展", "", "每个设置 3 特征组 × 3 奖励种子。表中为均值；较小差距不代表统计显著。转角为 π，d=3、T=1000。", ""]
    names = ["ucb1", "omd_paper_v2_literal", "domd_paper_v1", "scb_weight_ucb_numeric_v1"]
    def select(**kwargs):
        return [r for r in rows if all(r.get(k) == v for k, v in kwargs.items())]
    body = []
    for scenario in ("stationary", "abrupt", "rotation"):
        for k in (10, 30, 100):
            part = {r["algorithm"]: r for r in select(stage="logistic", scenario=scenario, num_arms=k, angle_pi=1.)}
            body.append([scenario, k, *[fmt(part[n]["mean"]) for n in names]])
    parts += [table(["场景", "K", *[short[n] for n in names]], body), "",
              "DOMD 在平稳和突变任务中表现较好，但旋转场景存在持平或其他方法均值较低的情况。扩大 K 同时改变动作集合和最优奖励，不能将结果变化全部归因于探索预算。", "",
              "## 3. 变化幅度", ""]
    body = []
    for scenario, angles in (("abrupt", (.25, .5, 1.)), ("rotation", (.5, 1., 2.))):
        for angle in angles:
            part = {r["algorithm"]: r for r in select(stage="logistic", scenario=scenario, num_arms=10, angle_pi=angle)}
            body.append([scenario, f"{angle:g}π", *[fmt(part[n]["mean"]) for n in names]])
    parts += [table(["场景", "角度", *[short[n] for n in names]], body), "",
              "旋转保持参数范数、不保证奖励差恒定；原直线漂移经过零附近。固定 γ 的适用性依赖变化方式与速度。", "",
              "## 4. 重尾线性 Bandit", "", "以下为 stationary 任务，T=1000；奖励尺度不同，不与 Logistic 表跨表比较。", ""]
    heavy = ["random", "ridge_greedy", "linucb_nominal", "hvt_ucb_literal"]
    noises = ["gaussian", "student3", "student1_5"]
    body = [[short[a], *[fmt(select(stage="heavy_closed", scenario="stationary", noise=n, algorithm=a)[0]["mean"]) for n in noises]] for a in heavy]
    parts += [table(["算法", "高斯", "t(3)", "t(1.5)"], body), "",
              "高斯与 t(3) 标准差均为 .5；t(1.5) 保留无限方差、原始尺度 .5，ε=.25，对应 1.25 阶矩。Hvt 使用已知矩尺度及论文原常数，当前时域未展示性能优势。非平稳重尾任务仅作压力测试。", "",
              "固定流估计采用完全相同的外生动作和奖励。", "",
              table(["噪声", "估计器", "更新前 RMSE", "平均裁剪次数"],
                    [[r["noise"], short[r["algorithm"]], f"{r['mean']:.6f}", fmt(r["clipped_updates"])]
                     for r in rows if r["stage"] == "heavy_fixed"]), "",
              "裁剪计数为零的实验没有触发 Huber 的异常值裁剪机制。Hvt 与去裁剪版本相同不意味着一般情况下裁剪无用；估计不佳也不能仅归因于闭环选臂。", "",
              "## 5. 计算和存储", "", "单步更新平均耗时，单位 μs；独立进程、BLAS 单线程，各 T 重复 3 次。", ""]
    cost_names = [*names[1:], "linucb_nominal", "hvt_ucb_literal"]
    body = []
    for name in cost_names:
        selected = [select(stage="cost", algorithm=name, horizon=t)[0] for t in (200, 1000, 5000)]
        body.append([short[name], *[f"{r['update_us']:.1f}" for r in selected],
                     *[int(r["retained_history_bytes"]) for r in selected]])
    parts += [table(["算法", "T200 μs", "T1000 μs", "T5000 μs", "历史 B@200", "历史 B@1000", "历史 B@5000"], body), "",
              "one-pass 方法不保存历史样本，固定 d 时更新时间随 T 较稳定。当前 SCB 保留历史并重复求解 MLE，成本随历史增长。历史数组字节不包含状态、Python 对象、临时数组、日志或进程 RSS。", "",
              "## 6. 可复核性与交付", "",
              "- 本报告生成前运行全套测试及三个归档的只读复核；结果和日志哈希见 `verification.json`。",
              "- `frozen_test_aggregate.json`、`supplement_aggregate.json`、`supplement_by_feature.csv` 保留精确数据。",
              "- `source_index.json` 保存源归档 manifest 与 protocol 哈希、原运行源码哈希。原始逐局数据和源码快照留在本机 artifacts。",
              "- 当前发布代码文件哈希见 `release_source_sha256.json`；这是收尾版本，不伪称与此前实验快照逐文件完全相同。",
              "- 校验和及指标重算不等于所有策略动作的完整重演；开销审计额外回放了动作。",
              "- 不把只有 3 个特征组的样本均值推广为普遍算法排名；不将数值 SCB 投影称为全局最优。", ""]
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-test-run", type=Path, default=ROOT / "artifacts/v2_2_frozen_test_r1")
    parser.add_argument("--remaining-run", type=Path, default=ROOT / "artifacts/v2_2_remaining_r1")
    parser.add_argument("--source-label", default="本机运行归档")
    args = parser.parse_args()
    frozen, remaining = args.frozen_test_run.resolve(), args.remaining_run.resolve()
    smoke, full = remaining / "smoke", remaining / "full"
    paths = release_paths()
    counts = check_budget(frozen, smoke, full)
    work = ROOT / "artifacts/v2_2_closeout"
    work.mkdir(parents=True, exist_ok=True)
    source_before = {p: sha(ROOT / p) for p in paths}
    results = []
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    commands = [("pytest", [sys.executable, "-m", "pytest", "-q", "tests"]),
                ("frozen_verify", [sys.executable, "-m", "scripts.run_frozen_logistic_test", "--output", str(frozen), "--verify-only"]),
                ("smoke_verify", [sys.executable, "-m", "scripts.run_remaining_v2_2", "--output", str(smoke), "--verify-only"]),
                ("full_verify", [sys.executable, "-m", "scripts.run_remaining_v2_2", "--output", str(full), "--verify-only"])]
    for label, command in commands:
        print(f"CHECK: {label}", flush=True)
        log = work / f"{label}.log"
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text()[-6000:])
            raise SystemExit(f"FAILED {label}: see {log}")
        print(log.read_text().strip().splitlines()[-1], flush=True)
        results.append(dict(check=label, status="passed", log_sha256=sha(log)))
    if source_before != {p: sha(ROOT / p) for p in paths}:
        raise RuntimeError("Source changed during checks")
    output = ROOT / RELEASE
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="v22-report-", dir=work))
    copied = {"frozen_test_aggregate.json": frozen / "test_aggregate.json",
              "frozen_test_by_feature.json": frozen / "test_by_feature.json",
              "frozen_test_audit.json": frozen / "test_audit.json",
              "supplement_aggregate.json": full / "aggregate.json",
              "supplement_aggregate.csv": full / "aggregate.csv",
              "supplement_by_feature.csv": full / "by_feature.csv",
              "supplement_environment.json": full / "environment.json"}
    for name, source in copied.items():
        shutil.copy2(source, staging / name)
    index = {}
    for label, folder in (("frozen", frozen), ("smoke", smoke), ("full", full)):
        protocol = load(folder / "protocol.json")
        index[label] = dict(manifest_sha256=sha(folder / "manifest.json"),
                            protocol_sha256=sha(folder / "protocol.json"),
                            run_source_sha256=protocol.get("source_sha256", {}))
    save(staging / "source_index.json", index)
    save(staging / "release_source_sha256.json", source_before)
    pytest_match = re.search(r"(\d+) passed", (work / "pytest.log").read_text())
    verification = dict(status="passed", source_label=args.source_label,
                        verified_at_utc=datetime.now(timezone.utc).isoformat(),
                        tests_passed=int(pytest_match.group(1)) if pytest_match else None,
                        frozen_episodes=315, smoke_tasks=71, supplemental_tasks=sum(counts.values()),
                        counts=counts, checks=results,
                        source_snapshot_note="Original run sources remain in source archives; release code includes closeout additions")
    save(staging / "verification.json", verification)
    (staging / "report.md").write_text(report_text(load(frozen / "test_aggregate.json"), load(full / "aggregate.json"), counts, args.source_label))
    save(staging / "manifest.json", {p.name: sha(p) for p in sorted(staging.iterdir())})
    if output.exists():
        backup = work / ("previous_report_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f"))
        output.rename(backup)
    staging.rename(output)
    files = sorted(paths + [f"{RELEASE}/{p.name}" for p in output.iterdir()])
    (work / "git-paths.txt").write_text("\n".join(files) + "\n")
    print(f"CLOSEOUT PASS: {output / 'report.md'}; {len(files)} explicit Git paths", flush=True)


if __name__ == "__main__":
    main()
