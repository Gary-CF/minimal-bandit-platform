"""One command: frozen supplemental Logistic, heavy-tail and isolated cost runs.

Resume with the identical command. --verify-only never runs an agent or writes.
Threads are capped before importing NumPy; performance experiments run serially
in fresh subprocesses, after all parallel quality experiments have finished.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import fcntl
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from scripts.remaining_experiments import (ROOT, FROZEN, digest, execute_task, read_json,
    source_hashes, task_id, tasks_for, verify_task, write_json)
from envs.remaining_bandit_tasks import noise_spec


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def protocol_for(profile):
    return dict(schema="bandit-v2.2-remaining-v1", profile=profile, tasks=tasks_for(profile),
                frozen_logistic_parameters=FROZEN, source_sha256=source_hashes(),
                noises={n: noise_spec(n) for n in ("gaussian", "student3", "student1_5")},
                selection="No selection or tuning on these tasks; all parameters predeclared",
                original_frozen_test="Independent supplement; original 315 episodes are untouched",
                hvt="arXiv:2503.00419v2 Algorithm 1 / Eqs 8-10 / Theorem 1 literal constants; known moment input",
                hvt_beta_zero="epsilon=1: time factor 1 including t=0; epsilon<1: 0**positive=0",
                linucb="nominal Gaussian noise scale 0.5, ridge 1, radius 2, delta .05; no heavy-tail guarantee",
                squared_ablation="same Hvt normalization, metric, projection; replace clipped residual by raw residual; fixed streams only",
                uncertainty="Report feature-level means and SEM across 3 feature groups, not iid episode SEM",
                costs="serial fresh subprocess per episode; replay operation counters excluded from timing and RSS samples",
                data_check="Recompute truth, reward RNG, regrets and summaries; not a full policy replay (cost replay does check actions)")


def initialize(root, profile, jobs):
    protocol = protocol_for(profile)
    if (root / "protocol.json").exists():
        if read_json(root / "protocol.json") != protocol:
            raise ValueError("Protocol/source changed: resume from archived source or choose a new output directory")
        for rel, sha in protocol["source_sha256"].items():
            if digest(root / "source" / rel) != sha:
                raise ValueError(f"archived source changed: {rel}")
    else:
        if any(p.name != ".run.lock" for p in root.iterdir()):
            raise ValueError("Output directory is nonempty but has no protocol")
        for rel, sha in protocol["source_sha256"].items():
            target = root / "source" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / rel).read_bytes())
        write_json(root / "protocol.json", protocol)
        import scipy
        import matplotlib
        metadata = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                        scipy=scipy.__version__, matplotlib=matplotlib.__version__,
                        jobs=jobs, blas_threads=1, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        for name, cmd in (("git_head", ["git", "rev-parse", "HEAD"]), ("git_status", ["git", "status", "--short"])):
            result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            metadata[name] = result.stdout.strip() if result.returncode == 0 else None
        write_json(root / "environment.json", metadata)
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
        (root / "requirements-freeze.txt").write_text(result.stdout if result.returncode == 0 else result.stderr)
    return protocol


def aggregate(rows):
    dimensions = ("stage", "scenario", "noise", "num_arms", "angle_pi", "horizon", "algorithm")
    groups = defaultdict(list)
    lookup = {tuple(r[k] for k in ("stage", "scenario", "noise", "num_arms", "angle_pi", "horizon", "feature_seed", "seed", "algorithm")): r for r in rows}
    for row in rows:
        groups[tuple(row[k] for k in dimensions)].append(row)
    result, feature_rows = [], []
    for key, part in sorted(groups.items()):
        base = dict(zip(dimensions, key))
        by_feature = defaultdict(list)
        for r in part:
            by_feature[r["feature_seed"]].append(r)
        metric = "prediction_rmse" if base["stage"] == "heavy_fixed" else "final_regret"
        feature_values = []
        for seed, group in sorted(by_feature.items()):
            val = float(np.mean([r[metric] for r in group]))
            feature_values.append(val)
            feature_rows.append(dict(base, feature_seed=seed, n=len(group), metric=metric, mean=val))
        n = len(feature_values)
        row = dict(base, n_episodes=len(part), n_feature_groups=n, metric=metric,
                   mean=float(np.mean(feature_values)),
                   feature_sem=float(np.std(feature_values, ddof=1) / np.sqrt(n)) if n > 1 else None)
        if base["stage"] != "cost":
            comparator = ("ucb1" if base["stage"] == "logistic" else
                          "ridge_greedy" if base["stage"] == "heavy_fixed" else "linucb_nominal")
            differences = []
            for group in by_feature.values():
                diffs = []
                for r in group:
                    match = tuple(r[k] for k in ("stage", "scenario", "noise", "num_arms", "angle_pi", "horizon", "feature_seed", "seed")) + (comparator,)
                    diffs.append(r[metric] - lookup[match][metric])
                differences.append(float(np.mean(diffs)))
            row.update(comparator=comparator, paired_difference=float(np.mean(differences)),
                       paired_feature_sem=float(np.std(differences, ddof=1) / np.sqrt(n)) if n > 1 else None)
        for name in ("select_us", "update_us", "early_update_us", "middle_update_us", "late_update_us",
                     "state_array_bytes", "retained_history_bytes", "external_trajectory_bytes",
                     "process_peak_rss_bytes", "clipped_updates", "projection_updates"):
            row[name] = float(np.mean([r[name] for r in part]))
        if base["stage"] == "cost":
            row["repeat_update_sem_us"] = (float(np.std([r["update_us"] for r in part], ddof=1) / np.sqrt(len(part)))
                                           if len(part) > 1 else None)
            for name in part[0]["operation_audit"]:
                row[f"audit_{name}"] = float(np.mean([r["operation_audit"][name] for r in part]))
        result.append(row)
    return result, feature_rows


def make_report(root, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    aggregate_rows, feature_rows = aggregate(rows)
    clean_rows = [{k: v for k, v in r.items() if k != "operation_audit"} for r in rows]
    csv_write(root / "episodes.csv", clean_rows)
    csv_write(root / "aggregate.csv", aggregate_rows)
    csv_write(root / "by_feature.csv", feature_rows)
    write_json(root / "aggregate.json", aggregate_rows)
    cost = [r for r in aggregate_rows if r["stage"] == "cost"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name in sorted({r["algorithm"] for r in cost}):
        part = sorted((r for r in cost if r["algorithm"] == name), key=lambda r: r["horizon"])
        for ax, metric, ylabel in zip(axes, ("update_us", "retained_history_bytes", "state_array_bytes"),
                                     ("Mean update (us)", "Retained history (bytes)", "State arrays (bytes)")):
            ax.plot([r["horizon"] for r in part], [r[metric] for r in part], marker="o", label=name)
            ax.set(xlabel="T", ylabel=ylabel)
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "cost_scaling.png", dpi=150)
    plt.close(fig)
    closed = [r for r in aggregate_rows if r["stage"] == "heavy_closed" and r["scenario"] == "stationary"]
    fig, ax = plt.subplots(figsize=(10, 4))
    noises = ["gaussian", "student3", "student1_5"]
    algorithms = sorted({r["algorithm"] for r in closed})
    for i, alg in enumerate(algorithms):
        part = [next(r for r in closed if r["noise"] == noise and r["algorithm"] == alg) for noise in noises]
        ax.bar(np.arange(3) + (i - 1.5) * .18, [r["mean"] for r in part], .18,
               yerr=[r["feature_sem"] or 0 for r in part], label=alg)
    ax.set(xticks=np.arange(3), xticklabels=noises, ylabel="Cumulative pseudo-regret",
           title="Stationary heavy-tail tasks (feature-group SEM)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "heavy_tail.png", dpi=150)
    plt.close(fig)
    logistic = [r for r in aggregate_rows if r["stage"] == "logistic"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, scenario in zip(axes, ("stationary", "abrupt", "rotation")):
        for alg in FROZEN:
            part = sorted((r for r in logistic if r["scenario"] == scenario
                           and r["angle_pi"] == 1. and r["algorithm"] == alg), key=lambda r: r["num_arms"])
            ax.plot([r["num_arms"] for r in part], [r["mean"] for r in part], marker="o",
                    linestyle="--" if alg == "sw_ucb" else "-", label=alg)
        ax.set(xlabel="Number of arms K (d=3)", ylabel="Cumulative pseudo-regret", title=scenario)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(root / "logistic_k_scaling.png", dpi=150)
    plt.close(fig)
    counts = Counter(r["stage"] for r in rows)
    lines = ["# Bandit v2.2 supplemental experiments", "",
             f"Completed tasks: {len(rows)}. Counts: `{dict(counts)}`.", "",
             "All settings were frozen before execution. The original 315-episode test is unchanged.",
             "New feature seeds are 20261130/20261201/20261202; reward seeds 300–302; timing seeds 310–312.",
             "Smoke uses a strict subset and shorter horizons; smoke results are not formal conclusions.", "",
             "## Interpretation and scope", "",
             "- Rotation preserves parameter norm, not arm gaps; chord drift can shrink the parameter norm.",
             "- Keep K, angle, scenario and noise separate. Do not interpret their arbitrary pooled mean as a main result.",
             "- Existing Logistic configurations stay frozen, including SW window 1000 and DOMD/SCB gamma 0.98.",
             "- Hvt uses literal theoretical constants (including 107), not performance-tuned constants. A poor finite-horizon result is retained.",
             "- Its known noise moment is an explicit input. Gaussian and t3 have standard deviation 0.5; t1.5 uses raw scale 0.5 and epsilon 0.25, not unit-variance normalization.",
             "- Hvt's guarantee is stationary. Abrupt/rotation heavy-tail results are out-of-theorem stress tests.",
             "- Fixed-stream Hvt-squared is a normalization/geometry-matched clipping ablation; it has no claimed Hvt guarantee.",
             "- Feature SEM uses 3 feature-level means, each averaging 3 reward seeds. These are small-sample descriptive uncertainties.",
             "- Cost measurements run serially in fresh processes with BLAS threads=1. Parallel quality-run timings are not used for speed comparisons.",
             "- State/history bytes count NumPy backing allocations, excluding static arms. Python objects and temporary allocations contribute to process RSS, not those array counts.",
             "- Process RSS includes imports, environment truth and trajectory buffers. It is not policy-only memory.",
             "- Operation audit is an untimed replay. loss_terms_rows counts rows entering MLE/projection loss-gradient-Hessian evaluations, not every arithmetic/memory pass.",
             "- Read-only verification recomputes generated truth, reward RNG, regrets and summaries. It does not certify policy optimality; cost audit additionally replays actions.", "",
             "## Stationary heavy-tail results", "",
             "| Noise | Algorithm | Mean regret | Feature SEM |", "|---|---|---:|---:|"]
    for r in closed:
        sem = "NA" if r["feature_sem"] is None else f"{r['feature_sem']:.4f}"
        lines.append(f"| {r['noise']} | {r['algorithm']} | {r['mean']:.4f} | {sem} |")
    lines += ["", "## Logistic supplement: K=10, angle=pi", "",
              "| Scenario | Algorithm | Mean regret | Paired difference vs UCB1 | Feature SEM of difference |",
              "|---|---|---:|---:|---:|"]
    for r in logistic:
        if r["num_arms"] == 10 and r["angle_pi"] == 1.:
            sem = "NA" if r["paired_feature_sem"] is None else f"{r['paired_feature_sem']:.4f}"
            lines.append(f"| {r['scenario']} | {r['algorithm']} | {r['mean']:.4f} | {r['paired_difference']:.4f} | {sem} |")
    lines += ["", "## Fixed streams: stationary estimator comparison", "",
              "| Noise | Estimator | Pre-update prediction RMSE | Mean clipped updates |",
              "|---|---|---:|---:|"]
    for r in aggregate_rows:
        if r["stage"] == "heavy_fixed":
            lines.append(f"| {r['noise']} | {r['algorithm']} | {r['mean']:.6f} | {r['clipped_updates']:.2f} |")
    failures = list((root / "attempts").glob("*/failure.txt"))
    lines += ["", f"Retained failed attempts: {len(failures)}. Any retries preserve their original failure records.", ""]
    lines += ["", "## Outputs", "", "- `aggregate.csv`: all settings and paired differences.",
              "- `by_feature.csv`: feature-level means.", "- `episodes.csv`: task-level metrics and cost windows.",
              "- `episodes/*/trajectory.npz`: raw trajectories; `summary.json`: operation counters on cost runs.",
              "- `protocol.json`, `source/`, `environment.json`, `requirements-freeze.txt`: provenance.",
              "", "![Logistic K scaling](logistic_k_scaling.png)", "", "![Cost scaling](cost_scaling.png)", "", "![Heavy-tail comparison](heavy_tail.png)", "",
              "Source: https://arxiv.org/html/2503.00419v2 (Algorithm 1, Lemma 1, Theorem 1).", ""]
    (root / "report.md").write_text("\n".join(lines))


def final_manifest(root):
    files = [p for p in root.rglob("*") if p.is_file()
             and p.relative_to(root).parts[0] not in ("attempts", "worker_logs")
             and p.name != ".run.lock" and p != root / "manifest.json"
             and "__pycache__" not in p.parts]
    write_json(root / "manifest.json", {str(p.relative_to(root)): digest(p) for p in sorted(files)})


def verify_all(root):
    protocol = read_json(root / "protocol.json")
    manifest = read_json(root / "manifest.json")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
              and p.relative_to(root).parts[0] not in ("attempts", "worker_logs")
              and p.name != ".run.lock" and p != root / "manifest.json" and "__pycache__" not in p.parts}
    if actual != set(manifest):
        raise ValueError("root archive file set differs")
    for rel, sha in manifest.items():
        if digest(root / rel) != sha:
            raise ValueError(f"root checksum mismatch: {rel}")
    for rel, sha in protocol["source_sha256"].items():
        if digest(root / "source" / rel) != sha:
            raise ValueError("source snapshot mismatch")
    expected = {task_id(t) for t in protocol["tasks"]}
    if {p.name for p in (root / "episodes").iterdir()} != expected:
        raise ValueError("episode coverage differs")
    rows = [verify_task(root / "episodes" / task_id(t), t) for t in protocol["tasks"]]
    if aggregate(rows)[0] != read_json(root / "aggregate.json"):
        raise ValueError("aggregate mismatch")
    print(f"REMAINING VERIFY PASS (read-only): {len(rows)} tasks", flush=True)
    return rows


def run(root, profile, jobs):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another runner owns this output directory") from exc
        protocol = initialize(root, profile, jobs)
        if (root / "manifest.json").exists():
            return verify_all(root)
        tasks = protocol["tasks"]
        rows, pending = [], []
        for task in tasks:
            path = root / "episodes" / task_id(task)
            if path.exists():
                rows.append(verify_task(path, task))
            else:
                pending.append(task)
        print(f"FROZEN {profile}: {dict(Counter(t['stage'] for t in tasks))}; verified reuse={len(rows)}; pending={len(pending)}", flush=True)
        quality = [t for t in pending if t["stage"] != "cost"]
        start = time.perf_counter()
        if quality:
            with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {pool.submit(execute_task, task, str(root)): task for task in quality}
                for future in as_completed(futures):
                    row = future.result()  # Failed tasks are preserved; never silently omitted.
                    rows.append(row)
                    if len(rows) % 25 == 0 or len(rows) == len(tasks):
                        print(f"PROGRESS {len(rows)}/{len(tasks)} elapsed={time.perf_counter()-start:.1f}s", flush=True)
        # No parallel quality workers remain. Each cost task starts a fresh interpreter.
        for task in pending:
            if task["stage"] != "cost":
                continue
            logs = root / "worker_logs"
            logs.mkdir(exist_ok=True)
            with (logs / f"{task_id(task)}.log").open("w") as log:
                subprocess.run([sys.executable, "-m", "scripts.run_remaining_v2_2", "--output", str(root),
                                "--worker", task_id(task)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            rows.append(verify_task(root / "episodes" / task_id(task), task))
            print(f"COST {len(rows)}/{len(tasks)} {task['algorithm']} T={task['horizon']} seed={task['seed']}", flush=True)
        rows = [verify_task(root / "episodes" / task_id(t), t) for t in tasks]
        make_report(root, rows)
        final_manifest(root)
        verify_all(root)
        print(f"REMAINING PASS: {root / 'report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.verify_only:
        verify_all(root)
    elif args.worker:
        protocol = read_json(root / "protocol.json")
        task = next(t for t in protocol["tasks"] if task_id(t) == args.worker)
        execute_task(task, root)
    else:
        if args.jobs < 1:
            parser.error("--jobs must be positive")
        run(root, args.profile, args.jobs)


if __name__ == "__main__":
    main()
