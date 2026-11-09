"""Development benchmark: python -m scripts.run_logistic_benchmark --help."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import traceback
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from scripts.logistic_benchmark import ALGORITHMS, make_path, run_episode

ROOT = Path(__file__).resolve().parents[1]
TIMING_KEYS = ("select_seconds", "update_seconds", "diagnostics_seconds",
               "environment_seconds", "trajectory_write_seconds")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(c):
    for key in ("dimension", "num_arms", "horizon", "window_size", "log_every"):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if c["horizon"] < 2:
        raise ValueError("horizon must be >= 2")
    if type(c["feature_seed"]) is not int or c["feature_seed"] < 0:
        raise ValueError("feature_seed must be a nonnegative integer")
    for key, allowed in (("algorithms", ALGORITHMS), ("scenarios", ("stationary", "abrupt", "drift"))):
        if not c[key] or len(set(c[key])) != len(c[key]) or not set(c[key]) <= set(allowed):
            raise ValueError(f"invalid/duplicate {key}")
    if (not c["seeds"] or any(type(s) is not int or s < 0 for s in c["seeds"])
            or len(set(c["seeds"])) != len(c["seeds"])):
        raise ValueError("seeds must be distinct nonnegative integers")
    if type(c["gamma"]) is not float or not 0 < c["gamma"] < 1:
        raise ValueError("gamma must be a float in (0, 1)")
    if not np.isfinite(c["radius"]) or c["radius"] <= 0 or not 0 < c["delta"] < 1:
        raise ValueError("invalid radius/delta")
    for key in ("theta_before", "theta_after"):
        v = np.asarray(c[key], dtype=float)
        if v.shape != (c["dimension"],) or not np.isfinite(v).all() or np.linalg.norm(v) > c["radius"] + 1e-10:
            raise ValueError(f"invalid {key}")
    for scenario in c["scenarios"]:
        make_path(scenario, c["horizon"], c["theta_before"], c["theta_after"], c["change_step"])
    return c


def episode_label(scenario, name, seed):
    return f"{scenario}__{name}__seed{seed}"


def snapshot_sources(output):
    files = set()
    for folder in ("algorithms", "envs", "estimators", "scripts", "tests"):
        files.update((ROOT / folder).rglob("*.py"))
    files.update((ROOT / "configs/v2_2").glob("*.json"))
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files)}
    with ZipFile(output / "source_snapshot.zip", "w", ZIP_DEFLATED) as z:
        for name in hashes:
            z.write(ROOT / name, name)
    write_json(output / "source_sha256.json", hashes)


def aggregate(summary, c):
    result = []
    for scenario in c["scenarios"]:
        for name in c["algorithms"]:
            part = [r for r in summary if r["scenario"] == scenario and r["algorithm"] == name]
            values = np.array([float(r["final_regret"]) for r in part])
            result.append({"scenario": scenario, "algorithm": name, "n": len(values),
                           "mean_regret": float(values.mean()),
                           "sem_regret": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
                           "mean_select_us": float(np.mean([float(r["select_seconds"]) / c["horizon"] * 1e6 for r in part])),
                           "mean_update_us": float(np.mean([float(r["update_seconds"]) / c["horizon"] * 1e6 for r in part]))})
    return result


def plot_results(output, c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scenarios = c["scenarios"]
    figures = output / "figures"
    figures.mkdir()
    for metric, ylabel, filename in (("cumulative_regret", "Cumulative dynamic pseudo-regret", "regret.png"),
                                      ("prediction_rmse_pre", "Pre-update probability RMSE", "prediction_rmse.png")):
        fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 3.7), squeeze=False, layout="constrained")
        for ax, scenario in zip(axes[0], scenarios):
            for name in c["algorithms"]:
                arrays = []
                for seed in c["seeds"]:
                    rows = read_csv(output / "episodes" / f"{episode_label(scenario, name, seed)}.csv")
                    if rows[0][metric] != "":
                        arrays.append([float(r[metric]) for r in rows])
                if not arrays:
                    continue
                values = np.asarray(arrays)
                mean = values.mean(axis=0)
                x = np.arange(1, len(mean) + 1)
                line, = ax.plot(x, mean, label=name)
                if len(values) > 1:
                    sem = values.std(axis=0, ddof=1) / np.sqrt(len(values))
                    ax.fill_between(x, mean - sem, mean + sem, alpha=.16, color=line.get_color())
            if scenario == "abrupt":
                ax.axvline(c["change_step"], color="gray", ls="--", lw=1)
            ax.set(title=scenario, xlabel="Round", ylabel=ylabel)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7)
            ax.grid(alpha=.2)
        fig.suptitle(f"Development comparison | mean +/- SEM, {len(c['seeds'])} seeds")
        fig.savefig(figures / filename, dpi=160)
        plt.close(fig)
    summary = read_csv(output / "summary.csv")
    x = np.arange(len(c["algorithms"]))
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for offset, key, label in ((-.2, "select_seconds", "Select"), (.2, "update_seconds", "Update")):
        vals = [np.mean([float(r[key]) / c["horizon"] * 1e6 for r in summary if r["algorithm"] == name]) for name in c["algorithms"]]
        ax.bar(x + offset, vals, width=.38, label=label)
    ax.set_xticks(x, c["algorithms"], rotation=12)
    ax.set(ylabel="Microseconds / step", title="Instrumented means across this run's scenarios and seeds")
    ax.legend()
    fig.savefig(figures / "cost.png", dpi=160)
    plt.close(fig)


def verify(output):
    manifest = json.loads((output / "manifest.json").read_text())
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if actual != set(manifest) | {"manifest.json"}:
        raise ValueError("output file set mismatch")
    for name, expected in manifest.items():
        p = (output / name).resolve()
        if not p.is_relative_to(output.resolve()) or digest(p) != expected:
            raise ValueError(f"hash mismatch: {name}")
    state = json.loads((output / "status.json").read_text())
    if state["status"] != "complete":
        raise ValueError("run is incomplete")
    c = validate_config(json.loads((output / "config.json").read_text()))
    features = np.load(output / "features.npy", allow_pickle=False)
    if features.shape != (c["num_arms"], c["dimension"]) or not np.isfinite(features).all():
        raise ValueError("invalid saved features")
    source_hashes = json.loads((output / "source_sha256.json").read_text())
    with ZipFile(output / "source_snapshot.zip") as z:
        if set(z.namelist()) != set(source_hashes):
            raise ValueError("source snapshot file set mismatch")
        for name, expected in source_hashes.items():
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError("source snapshot hash mismatch")
    summaries = read_csv(output / "summary.csv")
    expected_count = len(c["scenarios"]) * len(c["algorithms"]) * len(c["seeds"])
    if len(summaries) != expected_count or state["completed_episodes"] != expected_count:
        raise ValueError("episode count mismatch")
    for scenario in c["scenarios"]:
        path = make_path(scenario, c["horizon"], c["theta_before"], c["theta_after"], c["change_step"])
        np.testing.assert_array_equal(np.load(output / f"truth/{scenario}_theta.npy", allow_pickle=False), path)
        means = np.exp(-np.logaddexp(0., -(path @ features.T)))
        for name in c["algorithms"]:
            for seed in c["seeds"]:
                label = episode_label(scenario, name, seed)
                rows = read_csv(output / f"episodes/{label}.csv")
                timing = read_csv(output / f"timing/{label}.csv")
                if len(rows) != len(path) or len(timing) != len(path):
                    raise ValueError("incomplete trajectory")
                cumulative = 0.
                for step, (row, tm, truth) in enumerate(zip(rows, timing, means), 1):
                    if (row["algorithm"], int(row["seed"]), int(row["step"]), int(tm["step"])) != (name, seed, step, step):
                        raise ValueError("row identity mismatch")
                    a = int(row["action"])
                    if not 0 <= a < len(features) or row["reward"] not in ("0", "1"):
                        raise ValueError("invalid action/reward")
                    regret = float(truth.max() - truth[a])
                    cumulative += regret
                    np.testing.assert_allclose([float(row["instant_regret"]), float(row["cumulative_regret"])], [regret, cumulative], rtol=0, atol=1e-9)
                    if name in ("omd_paper_v2_literal", "domd_paper_v1", "scb_weight_ucb_numeric_v1"):
                        predicted = np.array([float(row[f"predicted_mean_{a}"]) for a in range(len(features))])
                        if not np.isfinite(predicted).all() or np.any((predicted < 0) | (predicted > 1)):
                            raise ValueError("invalid prediction")
                        np.testing.assert_allclose(float(row["prediction_rmse_pre"]), np.sqrt(np.mean((predicted-truth)**2)), rtol=0, atol=1e-12)
                    elif row["prediction_rmse_pre"] != "" or any(row[f"predicted_mean_{a}"] != "" for a in range(len(features))):
                        raise ValueError("unexpected MAB predictions")
                    if name == "scb_weight_ucb_numeric_v1":
                        audit = {k: float(v) for k, v in row.items() if k.startswith("scb_")}
                        if not all(np.isfinite(v) and v >= 0 for v in audit.values()):
                            raise ValueError("invalid SCB diagnostics")
                        active = audit["scb_projection_active_post"]
                        accepted = audit["scb_projection_accepted_post"]
                        if (active not in (0, 1) or accepted != int(accepted)
                                or accepted > 2 * c["dimension"] + 2
                                or (active == 1 and accepted < 1)
                                or (active == 0 and accepted != 0)
                                or audit["scb_projection_kkt_post"] > 1e-6
                                or audit["scb_mle_grad_inf_post"] > 1e-9
                                or audit["scb_projected_norm_post"] > c["radius"] + 1e-7
                                or (active == 0 and audit["scb_mle_norm_post"] > c["radius"])
                                or (active == 1 and audit["scb_mle_norm_post"] <= c["radius"])):
                            raise ValueError("SCB numerical acceptance failed")
                        q = np.exp(-c["radius"])
                        expected_lambda = c["dimension"] * np.log(c["horizon"]) / (q / (1 + q)**2)
                        np.testing.assert_allclose(audit["scb_lambda"], expected_lambda, rtol=1e-12)
                    if any(not np.isfinite(float(tm[k])) or float(tm[k]) < 0 for k in TIMING_KEYS):
                        raise ValueError("invalid timing")
                match = [r for r in summaries if (r["scenario"], r["algorithm"], int(r["seed"])) == (scenario, name, seed)]
                if len(match) != 1:
                    raise ValueError("summary identity mismatch")
                np.testing.assert_allclose(float(match[0]["final_regret"]), cumulative, rtol=0, atol=1e-9)
                if int(match[0]["total_reward"]) != sum(int(r["reward"]) for r in rows):
                    raise ValueError("summary reward mismatch")
                for k in TIMING_KEYS:
                    np.testing.assert_allclose(float(match[0][k]), sum(float(tm[k]) for tm in timing), rtol=1e-12, atol=1e-12)
    saved_aggregate = read_csv(output / "aggregate.csv")
    expected_aggregate = aggregate(summaries, c)
    if len(saved_aggregate) != len(expected_aggregate):
        raise ValueError("aggregate count mismatch")
    for actual_row, expected_row in zip(saved_aggregate, expected_aggregate):
        for key, value in expected_row.items():
            if value is None or isinstance(value, str):
                if actual_row[key] != ("" if value is None else value):
                    raise ValueError("aggregate identity/missing value mismatch")
            else:
                np.testing.assert_allclose(float(actual_row[key]), value, rtol=1e-12, atol=1e-12)
    return expected_count


def run(c, output):
    validate_config(c)
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "completed_episodes": 0, "current": None}
    write_json(output / "status.json", state)
    try:
        write_json(output / "config.json", c)
        write_json(output / "versions.json", {"python": platform.python_version(), "platform": platform.platform(),
            **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib", "pytest")}})
        def git(*args):
            p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
            return p.stdout.strip() if p.returncode == 0 else p.stderr.strip()
        write_json(output / "git.json", {"head": git("rev-parse", "HEAD"), "status": git("status", "--short", "--branch")})
        snapshot_sources(output)
        for folder in ("episodes", "timing", "truth"):
            (output / folder).mkdir()
        rng = np.random.default_rng(c["feature_seed"])
        features = rng.normal(size=(c["num_arms"], c["dimension"]))
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        np.save(output / "features.npy", features)
        summary, tasks = [], []
        for scenario in c["scenarios"]:
            path = make_path(scenario, c["horizon"], c["theta_before"], c["theta_after"], c["change_step"])
            np.save(output / f"truth/{scenario}_theta.npy", path)
            means = np.exp(-np.logaddexp(0., -(path @ features.T)))
            sorted_means = np.sort(means, axis=1)
            tasks.append({"scenario": scenario, "parameter_path_length": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
                          "best_arm_changes": int(np.count_nonzero(np.diff(means.argmax(axis=1)))),
                          "mean_top_two_gap": float(np.mean(sorted_means[:, -1] - sorted_means[:, -2])) if len(features) > 1 else None})
            for name in c["algorithms"]:
                for seed in c["seeds"]:
                    label = episode_label(scenario, name, seed)
                    state["current"] = {"episode": label, "completed_steps": 0}
                    write_json(output / "status.json", state)
                    totals = dict.fromkeys(TIMING_KEYS, 0.)
                    total_reward, last = 0, None
                    begin = perf_counter()
                    with (output / f"episodes/{label}.csv").open("x", newline="", encoding="utf-8") as data_file, (output / f"timing/{label}.csv").open("x", newline="", encoding="utf-8") as time_file:
                        data_writer = None
                        time_writer = csv.DictWriter(time_file, fieldnames=["step", *TIMING_KEYS])
                        time_writer.writeheader()
                        def on_step(row, timing):
                            nonlocal data_writer, total_reward, last
                            if data_writer is None:
                                data_writer = csv.DictWriter(data_file, fieldnames=list(row))
                                data_writer.writeheader()
                            start = perf_counter()
                            data_writer.writerow(row)
                            timing = dict(timing, trajectory_write_seconds=perf_counter() - start)
                            time_writer.writerow(timing)
                            for key in TIMING_KEYS:
                                totals[key] += timing[key]
                            total_reward += row["reward"]
                            last = row
                            state["current"]["completed_steps"] = row["step"]
                            if row["step"] % c["log_every"] == 0 or row["step"] == c["horizon"]:
                                print(f"{label} {row['step']}/{c['horizon']} regret={row['cumulative_regret']:.6f}", flush=True)
                        run_episode(name, features, path, seed, window_size=c["window_size"], gamma=c["gamma"],
                                    radius=c["radius"], delta=c["delta"], on_step=on_step, collect_rows=False)
                    summary.append({"scenario": scenario, "algorithm": name, "seed": seed,
                                    "final_regret": last["cumulative_regret"], "total_reward": total_reward,
                                    **totals, "episode_wall_seconds": perf_counter() - begin})
                    state["completed_episodes"] += 1
                    write_json(output / "status.json", state)
        write_csv(output / "summary.csv", summary)
        write_csv(output / "aggregate.csv", aggregate(summary, c))
        write_json(output / "task_diagnostics.json", tasks)
        plot_results(output, c)
        (output / "README.txt").write_text(
            "Development baseline, not tuned test results.\n"
            "Figures: mean +/- SEM across reward seeds for ONE feature set, not 95% CI.\n"
            "RMSE: before current reward update; MAB predictions are unavailable (blank).\n"
            "update_seconds includes solver/projection, not separate additive costs.\n"
            "trajectory_write_seconds times buffered trajectory row writes only.\n"
            "Timing CSV writes, progress prints, flush/close and setup are outside those components;\n"
            "episode_wall_seconds includes them, but excludes plotting and source archival.\n"
            "Times include instrumentation overhead and are specific to this machine/run.\n"
            "Seed pairing does not imply identical observations across policies.\n"
            "SCB numeric v1 uses AISTATS 2023 Eqs. (8), (14)-(17); lambda=d*log(T)/c_mu.\n"
            "SCB projection uses deterministic multistart SLSQP with local KKT checks, not a global certificate.\n"
            "SCB post-update diagnostics are saved per row; pre-update predictions use the projected estimate.\n"
            "Source snapshot captures uncommitted code; verification checks the saved snapshot, not current working tree.\n",
            encoding="utf-8")
        state["status"] = "complete"
        write_json(output / "status.json", state)
        write_json(output / "manifest.json", {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob("*")) if p.is_file()})
        count = verify(output)
        print(f"BENCHMARK PASS: {count} episodes | {output.resolve()}", flush=True)
    except Exception:
        state["status"] = "failed"
        state["failure"] = traceback.format_exc()
        write_json(output / "status.json", state)
        (output / "failure.txt").write_text(state["failure"], encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        count = verify(args.output)
        print(f"VERIFY PASS: {count} episodes (read-only)")
    else:
        if args.config is None:
            parser.error("--config is required unless --verify-only")
        run(json.loads(args.config.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
