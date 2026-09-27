"""Run from repository root: python scripts/d6_benchmark.py

Default: smoke (K=5,T=200,seeds=0,1, exact replay), then full
(K=5,T=5000,seeds=0..9). --smoke-only stops after smoke.
Requires numpy and matplotlib. Calls the existing run.py; edits no core code.
Each invocation writes a new results/d6_<UTC>/ evidence directory.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np

INSTANCES = [
    {"id": "random", "name": "random", "parameters": {}},
    {"id": "ucb1", "name": "ucb1", "parameters": {}},
    {"id": "sw_w200", "name": "sliding_window_ucb", "parameters": {"window_size": 200}},
    {"id": "du_g0995", "name": "discounted_ucb", "parameters": {"gamma": 0.995}},
]
FIELDS = ["environment", "algorithm", "algorithm_id", "horizon", "seed",
          "step", "action", "reward", "instant_regret", "cumulative_regret"]
SCENARIOS = ("stationary", "piecewise", "drift")
LOCAL_WINDOW = 200


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_csv(path, rows):
    require(bool(rows), f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_output(repo, *args):
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout.strip()


def source_files(repo):
    paths = [repo / "run.py"]
    for directory in ("algorithms", "envs"):
        paths.extend(sorted((repo / directory).rglob("*.py")))
    return paths


def source_hashes(repo):
    return {str(p.relative_to(repo)): digest(p) for p in source_files(repo)}


def specification(scenario, horizon):
    """Fixed inputs plus a reference independent of project trajectory code."""
    before = np.array([0.6, 0.5, 0.4, 0.3, 0.1])
    if scenario == "stationary":
        config = {"name": "stationary", "means": before.tolist()}
        means = np.tile(before, (horizon, 1))
    elif scenario == "piecewise":
        after = np.array([0.6, 0.5, 0.4, 0.3, 0.9])
        change = horizon // 2 + 1
        config = {"name": "piecewise_constant", "starts": [1, change],
                  "levels": [before.tolist(), after.tolist()]}
        means = np.vstack([np.tile(before, (change - 1, 1)),
                           np.tile(after, (horizon - change + 1, 1))])
    elif scenario == "drift":
        start = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        end = start[::-1].copy()
        config = {"name": "linear_drift", "start_means": start.tolist(),
                  "end_means": end.tolist(), "start_t": 1, "end_t": horizon}
        means = start + np.arange(horizon)[:, None] / (horizon - 1) * (end - start)
    else:
        raise ValueError(scenario)
    return config, means


def verify_csv(path, entry, seed, means):
    horizon, arms = means.shape
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        require(reader.fieldnames == FIELDS, f"Wrong CSV columns: {path}")
        rows = list(reader)
    require(len(rows) == horizon, f"Wrong row count: {path}")
    for t, row in enumerate(rows, 1):
        require(None not in row and all(v is not None for v in row.values()),
                f"Malformed row: {path}:{t}")
        require(row["environment"] == "nonstationary_bernoulli"
                and row["algorithm"] == entry["name"]
                and row["algorithm_id"] == entry["id"]
                and int(row["horizon"]) == horizon
                and int(row["seed"]) == seed and int(row["step"]) == t,
                f"Wrong metadata/step: {path}:{t}")
    actions = np.array([int(r["action"]) for r in rows], dtype=int)
    rewards = np.array([float(r["reward"]) for r in rows])
    instant = np.array([float(r["instant_regret"]) for r in rows])
    cumulative = np.array([float(r["cumulative_regret"]) for r in rows])
    require(np.all((actions >= 0) & (actions < arms)), f"Invalid action: {path}")
    require(np.all((rewards == 0) | (rewards == 1)), f"Non-Bernoulli reward: {path}")
    expected = means.max(axis=1) - means[np.arange(horizon), actions]
    require(np.all(np.isfinite(instant)) and np.all(np.isfinite(cumulative)),
            f"Nonfinite regret: {path}")
    require(np.allclose(instant, expected, atol=1e-12, rtol=0), f"Wrong regret: {path}")
    require(np.allclose(cumulative, np.cumsum(expected), atol=1e-8, rtol=0),
            f"Wrong cumulative regret: {path}")
    return {"actions": actions, "instant": instant, "cumulative": cumulative}


def rolling_mean(values, window=LOCAL_WINDOW):
    prefix = np.r_[0.0, np.cumsum(values)]
    stop = np.arange(1, len(values) + 1)
    start = np.maximum(0, stop - window)
    return (prefix[stop] - prefix[start]) / (stop - start)


def switch_metrics(actions, instant):
    """Arm 4 becomes uniquely optimal at c; delays use zero-based waiting time.

    If chosen at c, delay=0. Empty delay means not revisited before horizon.
    pre_gap is the number of consecutive pre-change rounds without arm 4.
    A missing pre-change selection is separately marked (left censored).
    """
    horizon = len(actions)
    split = horizon // 2
    prior = np.flatnonzero(actions[:split] == 4) + 1
    post = np.flatnonzero(actions[split:] == 4)
    last = int(prior[-1]) if len(prior) else None
    n = min(LOCAL_WINDOW, horizon - split)
    return {
        "last_target_step_before_change": last,
        "pre_gap_rounds": split - last if last is not None else split,
        "pre_gap_left_censored": last is None,
        "target_pulls_last_200_pre": int(np.sum(actions[max(0, split - 200):split] == 4)),
        "first_target_delay_after_change": int(post[0]) if len(post) else None,
        "post_delay_right_censored": not bool(len(post)),
        "post_observation_rounds": horizon - split,
        "post_window_rounds": n,
        "target_pulls_first_200_post": int(np.sum(actions[split:split + n] == 4)),
        "mean_regret_first_200_post": float(np.mean(instant[split:split + n])),
    }


def plot_case(path, scenario, means, records, seeds, plt):
    horizon = len(means)
    steps = np.arange(1, horizon + 1)
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True, constrained_layout=True)
    for arm in range(5):
        axes[0].plot(steps, means[:, arm], label=f"arm {arm}")
    axes[0].set(ylabel="True mean", ylim=(0, 1),
                title=f"D6 {scenario}: K=5, T={horizon}, seeds={len(seeds)}")
    axes[0].legend(ncol=5)
    action_image = np.stack([records[(a["id"], seeds[0])]["actions"] for a in INSTANCES])
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap(["#3677b5", "#ed8d26", "#3d9969", "#a569ad", "#db5265"])
    pixels = axes[1].imshow(action_image, origin="lower", aspect="auto", interpolation="nearest",
                            extent=(0.5, horizon + 0.5, -0.5, 3.5), cmap=cmap,
                            norm=BoundaryNorm(np.arange(-0.5, 5.5), 5))
    axes[1].set_yticks(range(4), [a["id"] for a in INSTANCES])
    axes[1].set_title("Actions for seed 0 (see raw CSV for individual rounds)")
    fig.colorbar(pixels, ax=axes[1], ticks=range(5), label="Arm")
    for entry in INSTANCES:
        runs = [records[(entry["id"], seed)] for seed in seeds]
        cumulative = np.stack([r["cumulative"] for r in runs])
        average = cumulative.mean(axis=0)
        sem = cumulative.std(axis=0, ddof=1) / np.sqrt(len(seeds))
        line, = axes[2].plot(steps, average, label=entry["id"])
        axes[2].fill_between(steps, average - sem, average + sem, color=line.get_color(), alpha=0.15)
        axes[3].plot(steps, np.mean([rolling_mean(r["instant"]) for r in runs], axis=0),
                     label=entry["id"], color=line.get_color())
    axes[2].set(ylabel="Cumulative dynamic regret", title="Mean +/- 1 SEM; not a 95% confidence interval")
    axes[2].legend(ncol=4)
    axes[3].set(ylabel="Local mean regret", xlabel="Round", title="Trailing 200 rounds (shorter at startup)")
    for ax in axes:
        if scenario == "piecewise":
            ax.axvline(horizon // 2 + 1, color="black", linestyle="--", alpha=0.5)
        ax.grid(alpha=0.15)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_cli(repo, case, config_path, raw, horizon, seeds, repeat):
    work = case / f"repeat_{repeat}"
    work.mkdir()
    command = [sys.executable, str(repo / "run.py"), "--config", str(config_path)]
    log_path = case / f"cli_{repeat}.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Command: " + json.dumps(command) + "\n")
        log.flush()
        process = subprocess.run(command, cwd=work, stdout=log, stderr=subprocess.STDOUT)
    require(process.returncode == 0, f"CLI failed. See {log_path}")
    folder = work / "results" / raw["experiment_name"]
    expected = {f"nonstationary_bernoulli_{a['id']}_T{horizon}_seed{s}.csv"
                for a in INSTANCES for s in seeds}
    require(folder.is_dir(), f"Missing results: {folder}")
    require({p.name for p in folder.iterdir()} == expected | {"config_snapshot.json"},
            f"Missing/extra output files: {folder}")
    snapshot = dict(raw)
    snapshot["horizons"] = [snapshot.pop("horizon")]
    require(json.loads((folder / "config_snapshot.json").read_text()) == snapshot,
            f"Wrong config snapshot: {folder}")
    return folder


def run_phase(repo, output, phase, horizon, seeds, create_trajectory, plt):
    phase_dir = output / phase
    phase_dir.mkdir()
    summary, episode_rows, diagnostic_rows = [], [], []
    for scenario in SCENARIOS:
        print(f"{phase}/{scenario}: running {len(INSTANCES) * len(seeds)} episodes...", flush=True)
        case = phase_dir / scenario
        case.mkdir()
        trajectory_config, means = specification(scenario, horizon)
        trajectory = create_trajectory(trajectory_config)
        actual = np.array([trajectory.at(t) for t in range(1, horizon + 1)])
        require(actual.shape == means.shape and np.allclose(actual, means, rtol=0, atol=1e-12),
                f"Trajectory mismatch: {phase}/{scenario}")
        raw = {"experiment_name": f"d6_{phase}_{scenario}",
               "environment": {"name": "nonstationary_bernoulli", "trajectory": trajectory_config},
               "algorithms": INSTANCES, "horizon": horizon, "seeds": seeds}
        config_path = case / "config.json"
        write_json(config_path, raw)
        write_csv(case / "mean_trajectory.csv",
                  [{"step": t, **{f"mean_{i}": float(m[i]) for i in range(5)}}
                   for t, m in enumerate(means, 1)])
        first = run_cli(repo, case, config_path, raw, horizon, seeds, 1)
        second = run_cli(repo, case, config_path, raw, horizon, seeds, 2) if phase == "smoke" else None
        records = {}
        for entry in INSTANCES:
            for seed in seeds:
                name = f"nonstationary_bernoulli_{entry['id']}_T{horizon}_seed{seed}.csv"
                record = verify_csv(first / name, entry, seed, means)
                if second is not None:
                    require(digest(first / name) == digest(second / name), f"Replay mismatch: {name}")
                records[(entry["id"], seed)] = record
                episode_rows.append({"scenario": scenario, "algorithm_id": entry["id"], "seed": seed,
                                     "final_dynamic_regret": float(record["cumulative"][-1])})
                if scenario == "piecewise":
                    diagnostic_rows.append({"algorithm_id": entry["id"], "seed": seed,
                                            **switch_metrics(record["actions"], record["instant"])})
            final = np.array([records[(entry["id"], s)]["cumulative"][-1] for s in seeds])
            summary.append({"scenario": scenario, "algorithm_id": entry["id"], "seeds": len(seeds),
                            "horizon": horizon, "mean_final_regret": float(final.mean()),
                            "std_final_regret": float(final.std(ddof=1)),
                            "sem_final_regret": float(final.std(ddof=1) / np.sqrt(len(seeds)))})
        plot_case(case / "overview.png", scenario, means, records, seeds, plt)
        print(f"{phase}/{scenario}: trajectory, CSV, regret, plot passed"
              + ("; exact replay passed" if second is not None else ""), flush=True)
    write_csv(phase_dir / "summary.csv", summary)
    write_csv(phase_dir / "episodes.csv", episode_rows)
    write_csv(phase_dir / "switch_diagnostics.csv", diagnostic_rows)
    lines = [f"# D6 {phase} observations", "",
             f"K=5, T={horizon}, seeds={seeds}. Fixed SW window=200; D-UCB gamma=0.995.", "",
             "| Scenario | Instance | Final dynamic regret (mean +/- SEM) |",
             "|---|---|---:|"]
    for row in summary:
        lines.append(f"| {row['scenario']} | {row['algorithm_id']} | "
                     f"{row['mean_final_regret']:.3f} +/- {row['sem_final_regret']:.3f} |")
    lines += ["", "## Switch diagnostics (arm 4; zero-based delay after change)", "",
              "| Instance | Median pre-change gap | Revisited / seeds | Median delay among revisits |",
              "|---|---:|---:|---:|"]
    for entry in INSTANCES:
        rows = [r for r in diagnostic_rows if r["algorithm_id"] == entry["id"]]
        delays = [r["first_target_delay_after_change"] for r in rows
                  if not r["post_delay_right_censored"]]
        median = f"{np.median(delays):.1f}" if delays else "NA"
        lines.append(f"| {entry['id']} | {np.median([r['pre_gap_rounds'] for r in rows]):.1f} | "
                     f"{len(delays)} / {len(seeds)} | {median} |")
    lines += ["", "A delay of 0 means arm 4 was selected on the change round. Missing delays are",
              "right-censored, not zero; the conditional median excludes those seeds.",
              "Pre-change gaps count consecutive rounds without arm 4 immediately before the change.",
              "If it was never selected, that gap is left-censored; see the per-seed CSV flags.", "",
              "## Interpretation to complete", "",
              "- Stationary: quantify the exploration cost of forgetting using the table and local curves.",
              "- Piecewise: compare pre-change gaps, first revisits and regret over the first 200 post-change rounds.",
              "  A first revisit alone does not establish sustained recovery; inspect actions and local regret.",
              "- Drift: compare regret around the crossing and toward the end.",
              "- Record any result contrary to expectations without retuning this run.", "",
              "## Definitions and limits", "",
              "Dynamic pseudo-regret at t is max_a mu[t,a] - mu[t,action[t]].",
              "Local regret is its trailing 200-round mean, using available rounds at startup.",
              "SEM uses the sample SD across seeds divided by sqrt(number of seeds); it is not a 95% CI.",
              "The same numeric seeds do not imply an arm-indexed common reward table across policies.",
              "Window size 200 and asymptotic discounted count 1/(1-0.995)=200 are only a scale match.",
              "The two policies also have different confidence constants; this is not a controlled ablation.",
              "Ten seeds and three prescribed environments do not establish universal algorithm rankings.",
              "Smoke runs establish correctness only. Full experiments are not replayed by this script.",
              "Passing this script does not automatically complete interpretation or tag a release."]
    (phase_dir / "observations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    require((repo / "run.py").is_file() and (repo / "algorithms").is_dir(),
            "Run from the Bandit repository root.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(repo))
    from run import create_mean_trajectory

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = repo / "results" / f"d6_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    print(f"Evidence directory: {output}", flush=True)
    hashes = source_hashes(repo)
    runner = Path(__file__).resolve()
    metadata = {"utc_time": stamp, "git_commit": git_output(repo, "rev-parse", "HEAD"),
                "git_status": git_output(repo, "status", "--short"),
                "python": sys.version, "python_executable": sys.executable,
                "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
                "source_sha256": hashes, "runner_sha256": digest(runner),
                "instances": INSTANCES, "local_window": LOCAL_WINDOW,
                "smoke": {"horizon": 200, "seeds": [0, 1], "repeats": 2},
                "full": {"horizon": 5000, "seeds": list(range(10)), "repeats": 1},
                "smoke_only": args.smoke_only}
    write_json(output / "metadata.json", metadata)
    shutil.copyfile(runner, output / "d6_benchmark.py")
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in source_files(repo):
            archive.write(path, str(path.relative_to(repo)))
        for pattern in ("requirements*.txt", "pyproject.toml", "uv.lock", "poetry.lock"):
            for path in repo.glob(pattern):
                archive.write(path, str(path.relative_to(repo)))
    try:
        run_phase(repo, output, "smoke", 200, [0, 1], create_mean_trajectory, plt)
        require(source_hashes(repo) == hashes, "Core source changed during smoke; stop before full run.")
        require(digest(runner) == metadata["runner_sha256"], "Runner changed during smoke.")
        print("D6 smoke passed: 24 unique episodes, two identical repeats.", flush=True)
        if not args.smoke_only:
            run_phase(repo, output, "full", 5000, list(range(10)), create_mean_trajectory, plt)
        require(source_hashes(repo) == hashes, "Core source changed during benchmark.")
        require(digest(runner) == metadata["runner_sha256"], "Runner changed during benchmark.")
        write_json(output / "acceptance.json", {"status": "passed", "smoke_unique_episodes": 24,
                    "smoke_executed_episodes": 48, "full_unique_episodes": 0 if args.smoke_only else 120,
                    "full_rows": 0 if args.smoke_only else 600000,
                    "interpretation_complete": False})
    except Exception as exc:
        write_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        print(f"FAILED; evidence retained: {output}", file=sys.stderr, flush=True)
        raise
    files = sorted(p for p in output.rglob("*") if p.is_file())
    write_json(output / "manifest_sha256.json", {str(p.relative_to(output)): digest(p) for p in files})
    if not args.smoke_only:
        print("D6 full benchmark passed: 120 episodes, 600000 rows.", flush=True)
        print((output / "full" / "observations.md").read_text(encoding="utf-8"), flush=True)
    print(f"Evidence directory: {output}", flush=True)


if __name__ == "__main__":
    main()
