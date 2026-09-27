"""Archive existing D7 evidence without rerunning tests or modifying core code.

Run from the repository root:
python scripts/archive_d7.py --pytest-root /tmp/pytest-of-gary/pytest-1

Does not commit, push, create tags, or mark the OMD exercise as completed.
The test counts are explicitly recorded as user-reported results.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import zipfile


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n", encoding="utf-8")


def git(repo, *args):
    result = subprocess.run(["git", *args], cwd=repo, check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


def check_episode(folder, config):
    policy, seed = config["policy"], config["seed"]
    require(config["horizon"] == 200, f"Wrong horizon: {folder}")
    expected_features = [[1., 0.], [0., 1.], [-1., 0.], [0., -1.],
                         [1. / math.sqrt(2.), 1. / math.sqrt(2.)]]
    require(config["theta_star"] == [1., -0.5], f"Wrong theta: {folder}")
    require(len(config["features"]) == 5, f"Wrong number of arms: {folder}")
    for actual, expected in zip(config["features"], expected_features):
        require(len(actual) == 2 and all(math.isclose(a, b, rel_tol=0, abs_tol=1e-14)
                    for a, b in zip(actual, expected)), f"Wrong features: {folder}")
    means = [1. / (1. + math.exp(-(x[0] - 0.5 * x[1]))) for x in expected_features]
    first, second = [folder / f"{policy}_seed{seed}_repeat{i}.csv" for i in (1, 2)]
    require(first.is_file() and second.is_file(), f"Missing CSV: {folder}")
    require(first.read_bytes() == second.read_bytes(), f"Replay mismatch: {folder}")
    with first.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        require(reader.fieldnames == ["step", "action", "reward", "instant_regret", "cumulative_regret"],
                f"Wrong CSV fields: {first}")
        rows = list(reader)
    require(len(rows) == 200, f"Wrong row count: {first}")
    cumulative = 0.
    for step, row in enumerate(rows, 1):
        action = int(row["action"])
        require(int(row["step"]) == step and 0 <= action < 5, f"Wrong step/action: {first}")
        require(float(row["reward"]) in (0., 1.), f"Wrong reward: {first}")
        expected = max(means) - means[action]
        cumulative += expected
        require(math.isclose(float(row["instant_regret"]), expected, rel_tol=0, abs_tol=1e-12),
                f"Wrong instantaneous regret: {first}")
        require(math.isclose(float(row["cumulative_regret"]), cumulative, rel_tol=0, abs_tol=1e-10),
                f"Wrong cumulative regret: {first}")
    return first, second, cumulative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytest-root", required=True, type=Path)
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    required = [repo / "run.py", repo / "envs/logistic_bandit.py",
                repo / "tests/test_logistic_bandit_week1.py"]
    for path in required:
        require(path.is_file(), f"Run from the project root; missing {path}")
    require(args.pytest_root.is_dir(),
            "Pytest evidence directory is missing. Recreate only the episode evidence with: "
            "python -m pytest -q -s tests/test_logistic_bandit_week1.py "
            "-k test_200_round_episode_and_exact_replay ; then pass its pytest-N directory.")
    expected = {(policy, seed) for policy in ("random", "ucb1") for seed in range(3)}
    found = {}
    for folder in sorted(args.pytest_root.iterdir()):
        if folder.is_symlink() or not folder.is_dir():
            continue
        path = folder / "config.json"
        if not path.is_file():
            continue
        config = json.loads(path.read_text(encoding="utf-8"))
        key = config.get("policy"), config.get("seed")
        if key not in expected:
            continue
        require(key not in found, f"Duplicate episode configuration: {key}")
        first, second, regret = check_episode(folder, config)
        found[key] = path, first, second, regret
    require(set(found) == expected, f"Missing episodes in {args.pytest_root}: {expected - set(found)}")

    source_paths = [repo / "run.py", repo / "tests/test_logistic_bandit_week1.py"]
    for name in ("envs", "algorithms"):
        source_paths.extend(sorted((repo / name).rglob("*.py")))
    source_hashes = {str(p.relative_to(repo)): sha256(p) for p in source_paths}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = repo / "reports" / "d7" / stamp
    out.mkdir(parents=True, exist_ok=False)
    metadata = {
        "archived_at_utc": stamp,
        "git_commit_before_archive_commit": git(repo, "rev-parse", "HEAD"),
        "git_branch": git(repo, "branch", "--show-current"),
        "git_status_at_archive": git(repo, "status", "--short"),
        "source_sha256_at_archive": source_hashes,
        "test_results_reported_by_user": {
            "specialized": "62 passed in 0.35s",
            "full_suite": "449 passed in 1.49s",
            "reported_local_time": "2026-09-27T18:04:49+08:00",
        },
        "provenance_note": "Test counts came from the user's terminal output, not a rerun. "
            "Source hashes describe files at archive time; pytest did not record source hashes.",
        "archive_verification": "6 episode pairs: config, 200 rows, regret, exact CSV replay verified",
        "omd_prereading": "Explained; final independent exercise pending",
        "omd_implementation": "Not part of E-D7 scope",
    }
    write_json(out / "metadata.json", metadata)
    table = []
    for (policy, seed), (config, first, second, regret) in sorted(found.items()):
        target = out / "episodes" / f"{policy}_seed{seed}"
        target.mkdir(parents=True)
        for path in (config, first, second):
            shutil.copyfile(path, target / path.name)
        table.append(f"| {policy} | {seed} | {regret:.6f} |")
    with zipfile.ZipFile(out / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths:
            archive.write(path, str(path.relative_to(repo)))
        for pattern in ("requirements*.txt", "pyproject.toml", "uv.lock", "poetry.lock"):
            for path in repo.glob(pattern):
                archive.write(path, str(path.relative_to(repo)))
    require(source_hashes == {str(p.relative_to(repo)): sha256(p) for p in source_paths},
            "Source changed while archiving; do not commit this archive.")
    note = ["# E-D7 代码与测试归档", "",
            "- E-D1—D6 完成；v2.0 已发布，本次不移动该标签。",
            "- Logistic 损失、梯度、Hessian 与有限动作奖励环境已实现。",
            "- 用户报告：专项测试 62 passed in 0.35s；全量测试 449 passed in 1.49s。",
            "- 本次归档重新核对 6 组实验配置、每组 200 轮的奖励取值和遗憾、重复 CSV 字节一致性。",
            "- 原始 pytest 控制台日志未保存；以上通过数量来自用户提供的终端输出。",
            "- OMD 状态、约束更新和曲率累积顺序已讲解；最后独立手算题待验收。",
            "- 未实现完整 Logistic 策略或 OMD；属于后续工程任务。", "",
            "## 小实验", "", "d=2，K=5，T=200。Random/UCB1 不使用特征估计参数。", "",
            "| 算法 | 种子 | 最终动态伪遗憾 |", "|---|---:|---:|", *table, "",
            "共 6 条独立配置轨迹，每条重复运行两次；12 个 CSV、2400 行。", "",
            "## OMD 待验收题", "",
            "theta_t=0，H_t=1，x_t=1，y_t=1，eta=2，S=1。",
            "计算 g_t、G_t、A_t=H_t+eta G_t、无约束候选 u，判断是否需投影；",
            "写出 H_(t+1)，解释为什么一般不等于 A_t。", "",
            "源码快照和哈希记录的是归档时的工作区，Git 提交号是本次归档提交之前的 HEAD。",
            "归档完成不自动代表尚待验收的学习题已通过。", ""]
    (out / "README.md").write_text("\n".join(note), encoding="utf-8")
    shutil.copyfile(Path(__file__).resolve(), out / "archive_d7.py")
    write_json(out / "manifest_sha256.json", {
        str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()
    })
    print(f"D7 evidence archive verified: {out}")
    print("6 episode pairs / 12 CSVs / 2400 rows preserved; no tests rerun.")
    print("Code acceptance passed; final OMD exercise remains pending.")


if __name__ == "__main__":
    main()
