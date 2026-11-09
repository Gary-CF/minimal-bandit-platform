"""Freeze a validation protocol, compare memory scales, reserve test tasks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.run_logistic_benchmark import ROOT, read_csv, run, verify, write_json

WINDOWS = (50, 100, 200, 500, 1000)
FEATURE_SEEDS = (20261110, 20261111, 20261112)
REWARD_SEEDS = (100, 101, 102)


def source_hashes():
    files = [p for folder in ("algorithms", "envs", "estimators", "scripts")
             for p in (ROOT / folder).rglob("*.py")]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def run_validation(output):
    base = json.loads((ROOT / "configs/v2_2/development.json").read_text())
    base.update(stage="memory_validation", seeds=list(REWARD_SEEDS),
                scenarios=["stationary", "abrupt", "drift"])
    protocol = {"base_config": base, "feature_seeds": list(FEATURE_SEEDS),
                "windows": list(WINDOWS), "source_sha256": source_hashes(),
                "selection": "one memory setting per algorithm; minimize mean total regret across all validation tasks; ties choose smaller W",
                "reserved_test": {"feature_seeds": [20261120, 20261121, 20261122],
                                  "reward_seeds": [200, 201, 202, 203, 204]}}
    if output.exists():
        saved = json.loads((output / "protocol.json").read_text())
        if saved != protocol:
            raise ValueError("Protocol/source changed; use a new output directory")
    else:
        output.mkdir(parents=True)
        write_json(output / "protocol.json", protocol)
    baseline, candidates = {}, []
    for feature_seed in FEATURE_SEEDS:
        jobs = [("baseline", ["random", "ucb1", "omd_paper_v2_literal"], 100)]
        jobs += [(f"W{w}", ["sw_ucb", "discounted_ucb"], w) for w in WINDOWS]
        for label, algorithms, window in jobs:
            config = dict(base, feature_seed=feature_seed, algorithms=algorithms,
                          window_size=window, gamma=float(1 - 1 / window))
            destination = output / f"feature{feature_seed}_{label}"
            if destination.exists():
                if json.loads((destination / "config.json").read_text()) != config:
                    raise ValueError(f"Configuration differs: {destination}")
                verify(destination)
                print(f"REUSE VERIFIED: {destination.name}", flush=True)
            else:
                run(config, destination)
            for row in read_csv(destination / "summary.csv"):
                key = (feature_seed, row["scenario"], int(row["seed"]))
                value = float(row["final_regret"])
                if label == "baseline":
                    baseline[row["algorithm"], key] = value
                else:
                    candidates.append((row["algorithm"], window, key, value))
    n_tasks = len(FEATURE_SEEDS) * len(base["scenarios"]) * len(REWARD_SEEDS)
    scores, selected = [], {}
    for name in ("sw_ucb", "discounted_ucb"):
        for window in WINDOWS:
            part = [(key, value) for alg, w, key, value in candidates
                    if alg == name and w == window]
            if len(part) != n_tasks or len({key for key, _ in part}) != n_tasks:
                raise ValueError("Incomplete or duplicate validation tasks")
            score = {"algorithm": name, "window": window, "gamma": 1 - 1 / window,
                     "n_tasks": n_tasks,
                     "mean_regret": float(np.mean([v for _, v in part])),
                     "mean_diff_to_ucb1": float(np.mean([v - baseline["ucb1", k] for k, v in part]))}
            for scenario in base["scenarios"]:
                score[scenario] = float(np.mean([v for k, v in part if k[1] == scenario]))
            scores.append(score)
        best = min((r for r in scores if r["algorithm"] == name),
                   key=lambda r: (r["mean_regret"], r["window"]))
        selected[name] = ({"window_size": best["window"]} if name == "sw_ucb"
                          else {"gamma": best["gamma"]})
    write_json(output / "validation_scores.json", scores)
    write_json(output / "selected_memory.json", selected)
    print("\nVALIDATION BASELINES (mean over the same tasks)")
    for name in ("random", "ucb1", "omd_paper_v2_literal"):
        values = [v for (alg, _), v in baseline.items() if alg == name]
        print(f"{name:24s} {np.mean(values):.6f}")
    print("\nalgorithm          W    overall  stationary   abrupt    drift   diff-UCB1")
    for r in scores:
        print(f"{r['algorithm']:17s} {r['window']:4d} {r['mean_regret']:10.4f}"
              f" {r['stationary']:10.4f} {r['abrupt']:9.4f} {r['drift']:9.4f}"
              f" {r['mean_diff_to_ucb1']:11.4f}")
    print("\nSELECTED:", json.dumps(selected))
    print(f"VALIDATION PASS: {n_tasks * (3 + 2 * len(WINDOWS))} episodes; reserved test tasks were not run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run_validation(parser.parse_args().output)
