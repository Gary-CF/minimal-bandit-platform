"""D3 integration checks; run from the Bandit repository root.

Usage: python scripts/d3_cli_acceptance.py
Creates a fresh results/d3_acceptance_<UTC timestamp>/ directory.
Runs the actual run.py CLI twice per scenario in separate directories.
Does not edit source, commit, push, or overwrite existing experiment results.
Requires the project's numpy and matplotlib installations.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np


HORIZON = 120
SEEDS = [0, 1]
INSTANCES = [
    {"id": "random", "name": "random", "parameters": {}},
    {"id": "ucb1", "name": "ucb1", "parameters": {}},
    {"id": "eps005", "name": "epsilon_greedy", "parameters": {"epsilon": 0.05}},
    {"id": "eps020", "name": "epsilon_greedy", "parameters": {"epsilon": 0.20}},
]
SCENARIOS = {
    "stationary": {"name": "stationary", "means": [0.2, 0.7, 0.4]},
    "piecewise": {
        "name": "piecewise_constant", "starts": [1, 61],
        "levels": [[0.2, 0.7, 0.4], [0.8, 0.2, 0.4]],
    },
    "drift": {
        "name": "linear_drift", "start_means": [0.2, 0.7, 0.4],
        "end_means": [0.8, 0.2, 0.4], "start_t": 1, "end_t": HORIZON,
    },
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git_output(repo, *args):
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def source_hashes(repo):
    paths = [repo / "run.py"]
    for directory in [repo / "algorithms", repo / "envs"]:
        paths.extend(sorted(directory.rglob("*.py")))
    return {str(p.relative_to(repo)): digest(p) for p in paths}


def expected_means(scenario):
    # Independent reference for these three FIXED acceptance scenarios.
    start = np.array([0.2, 0.7, 0.4])
    end = np.array([0.8, 0.2, 0.4])
    if scenario == "stationary":
        return np.tile(start, (HORIZON, 1))
    if scenario == "piecewise":
        return np.vstack([np.tile(start, (60, 1)), np.tile(end, (60, 1))])
    if scenario == "drift":
        weight = np.arange(HORIZON)[:, None] / (HORIZON - 1)
        return start + weight * (end - start)
    raise ValueError(scenario)


def check_csv(path, entry, seed, means):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    require(len(rows) == HORIZON, f"Wrong row count: {path}")
    cumulative = 0.0
    for t, row in enumerate(rows, 1):
        require(int(row["step"]) == t, f"Wrong step: {path}, {t}")
        require(row["algorithm"] == entry["name"], f"Wrong algorithm: {path}")
        require(row["algorithm_id"] == entry["id"], f"Wrong instance ID: {path}")
        require(row["environment"] == "nonstationary_bernoulli", f"Wrong environment: {path}")
        require(int(row["seed"]) == seed, f"Wrong seed: {path}")
        require(int(row["horizon"]) == HORIZON, f"Wrong horizon: {path}")
        action = int(row["action"])
        require(0 <= action < 3, f"Invalid action: {path}, {t}")
        require(float(row["reward"]) in (0.0, 1.0), f"Invalid reward: {path}, {t}")
        regret = float(means[t - 1].max() - means[t - 1, action])
        cumulative += regret
        require(np.isclose(float(row["instant_regret"]), regret, rtol=0, atol=1e-10),
                f"Wrong instantaneous regret: {path}, {t}")
        require(np.isclose(float(row["cumulative_regret"]), cumulative, rtol=0, atol=1e-9),
                f"Wrong cumulative regret: {path}, {t}")
    return rows


def plot_scenario(path, scenario, means, records, plt):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    steps = np.arange(1, HORIZON + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True, sharex=True)
    for arm in range(3):
        axes[0].plot(steps, means[:, arm], label=f"arm {arm}")
    axes[0].set(ylabel="True mean", ylim=(-0.02, 1.02), title=f"D3 smoke: {scenario}")
    axes[0].legend(ncol=3)
    cmap = ListedColormap(["#2675b9", "#e88726", "#37976b"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
    for index, entry in enumerate(INSTANCES):
        actions = [int(r["action"]) for r in records[(entry["id"], 0)]]
        dots = axes[1].scatter(steps, np.full(HORIZON, index), c=actions,
                               cmap=cmap, norm=norm, s=18, marker="s")
        curves = [[float(r["cumulative_regret"]) for r in records[(entry["id"], seed)]]
                  for seed in SEEDS]
        axes[2].plot(steps, np.mean(curves, axis=0), label=entry["id"])
    axes[1].set_yticks(range(len(INSTANCES)), [e["id"] for e in INSTANCES])
    axes[1].set(ylabel="Instance", title="Actions, seed=0")
    fig.colorbar(dots, ax=axes[1], ticks=[0, 1, 2], label="Selected arm")
    axes[2].set(xlabel="Round", ylabel="Cumulative dynamic regret",
                title="Mean of 2 seeds; smoke check only")
    axes[2].legend(ncol=4)
    for ax in axes:
        ax.grid(alpha=0.18)
        if scenario == "piecewise":
            ax.axvline(61, color="black", linestyle="--", alpha=0.5)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    repo = Path.cwd().resolve()
    require((repo / "run.py").is_file() and (repo / "envs").is_dir(),
            "Run this script from the Bandit repository root.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(repo))
    from run import create_mean_trajectory

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = repo / "results" / f"d3_acceptance_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    print(f"Evidence directory: {output}", flush=True)
    code_hashes = source_hashes(repo)
    metadata = {
        "utc_time": stamp, "git_commit": git_output(repo, "rev-parse", "HEAD"),
        "git_branch": git_output(repo, "branch", "--show-current"),
        "git_status": git_output(repo, "status", "--short"),
        "python": sys.version, "python_executable": sys.executable,
        "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
        "source_sha256": code_hashes, "checker_sha256": digest(Path(__file__)),
        "horizon": HORIZON, "seeds": SEEDS, "scope": "D3 smoke, not D6 performance evidence",
    }
    write_json(output / "metadata.json", metadata)
    summary = []
    for scenario, trajectory_config in SCENARIOS.items():
        means = expected_means(scenario)
        trajectory = create_mean_trajectory(trajectory_config)
        actual_means = np.array([trajectory.at(t) for t in range(1, HORIZON + 1)])
        require(actual_means.shape == means.shape and np.allclose(actual_means, means, rtol=0, atol=1e-12),
                f"Trajectory differs from independent reference: {scenario}")
        case = output / scenario
        case.mkdir()
        with (case / "mean_trajectory.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["step", "mean_0", "mean_1", "mean_2"])
            writer.writerows((t, *m) for t, m in enumerate(actual_means, 1))
        raw = {
            "experiment_name": f"d3_{scenario}",
            "environment": {"name": "nonstationary_bernoulli", "trajectory": trajectory_config},
            "algorithms": INSTANCES, "horizon": HORIZON, "seeds": SEEDS,
        }
        config_path = case / "config.json"
        write_json(config_path, raw)
        expected_snapshot = dict(raw)
        expected_snapshot.pop("horizon")
        expected_snapshot["horizons"] = [HORIZON]
        expected_files = {
            f"nonstationary_bernoulli_{e['id']}_T{HORIZON}_seed{s}.csv"
            for e in INSTANCES for s in SEEDS
        }
        results = []
        for repeat in [1, 2]:
            work = case / f"repeat_{repeat}"
            work.mkdir()
            command = [sys.executable, str(repo / "run.py"), "--config", str(config_path)]
            with (case / f"cli_{repeat}.log").open("w", encoding="utf-8") as log:
                log.write("Command: " + json.dumps(command) + "\n")
                log.flush()
                result = subprocess.run(command, cwd=work, stdout=log, stderr=subprocess.STDOUT)
            require(result.returncode == 0, f"CLI failed; see {case / f'cli_{repeat}.log'}")
            directory = work / "results" / raw["experiment_name"]
            require({p.name for p in directory.glob("*.csv")} == expected_files,
                    f"Missing, overwritten, or incorrectly named CSVs: {directory}")
            snapshot = json.loads((directory / "config_snapshot.json").read_text(encoding="utf-8"))
            require(snapshot == expected_snapshot, f"Wrong configuration snapshot: {directory}")
            results.append(directory)
        records = {}
        for entry in INSTANCES:
            for seed in SEEDS:
                filename = f"nonstationary_bernoulli_{entry['id']}_T{HORIZON}_seed{seed}.csv"
                first, second = [directory / filename for directory in results]
                require(first.read_bytes() == second.read_bytes(), f"Replay mismatch: {scenario}/{filename}")
                records[(entry["id"], seed)] = check_csv(first, entry, seed, means)
        plot_scenario(case / "overview.png", scenario, means, records, plt)
        summary.append(f"- {scenario}: 8 CSVs per run; exact replay, IDs, config, means and regret passed.")
        print(f"{scenario}: passed (8 CSVs x 2 repeats, trajectory, regret, plot)", flush=True)
    require(source_hashes(repo) == code_hashes, "Source files changed during acceptance; rerun after edits finish.")
    (output / "SUMMARY.md").write_text(
        "# D3 CLI acceptance\n\n" + "\n".join(summary)
        + "\n\nK=3, T=120, seeds=[0,1]. 24 unique episodes; each repeated once.\n"
        + "\nThis is integration evidence, not a general performance claim or D6 completion.\n"
        + "Full repository regression tests and manual plot inspection are recorded separately.\n",
        encoding="utf-8",
    )
    print("D3 CLI acceptance passed: 3 scenarios, 24 unique episodes, exact replay.")
    print(f"Evidence directory: {output}")


if __name__ == "__main__":
    main()
