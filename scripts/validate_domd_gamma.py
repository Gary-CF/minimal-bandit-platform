"""Select one DOMD gamma using closed-loop validation regret."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.run_logistic_benchmark import ROOT, read_csv, run, verify, write_json
from scripts.validate_memory_scale import source_hashes


def main(output):
    gammas = (0.98, 0.99, 0.995, 0.998, 0.999)
    features = (20261110, 20261111, 20261112)
    seeds = (100, 101, 102)
    scenarios = ("stationary", "abrupt", "drift")
    base = json.loads((ROOT / "configs/v2_2/development.json").read_text())
    base.update(stage="domd_gamma_validation", horizon=1000, change_step=501,
                seeds=list(seeds), scenarios=list(scenarios), log_every=1000)
    protocol = dict(base_config=base, gammas=list(gammas),
                    feature_seeds=list(features), source_sha256=source_hashes(),
                    selection="minimum mean total regret over 27 tasks; exact ties choose smaller gamma",
                    reserved_test=dict(feature_seeds=[20261120, 20261121, 20261122],
                                       reward_seeds=[200, 201, 202, 203, 204]))
    if output.exists():
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Protocol/source changed; use a new output directory")
    else:
        output.mkdir(parents=True)
        write_json(output / "protocol.json", protocol)

    baseline, candidates = {}, {}
    for feature_seed in features:
        jobs = [("baseline", ["ucb1", "omd_paper_v2_literal"], 0.99)]
        jobs += [(f"gamma{g:g}", ["domd_paper_v1"], g) for g in gammas]
        for label, algorithms, gamma in jobs:
            config = dict(base, feature_seed=feature_seed,
                          algorithms=algorithms, gamma=gamma)
            destination = output / f"feature{feature_seed}_{label}"
            if destination.exists():
                if json.loads((destination / "config.json").read_text()) != config:
                    raise ValueError(f"Configuration differs: {destination}")
                verify(destination)
                print(f"REUSE VERIFIED: {destination.name}", flush=True)
            else:
                run(config, destination)  # Includes verification before PASS.
            for row in read_csv(destination / "summary.csv"):
                task = (feature_seed, row["scenario"], int(row["seed"]))
                table = baseline if label == "baseline" else candidates
                key = (row["algorithm"], task) if label == "baseline" else (gamma, task)
                if key in table:
                    raise ValueError(f"Duplicate result: {key}")
                table[key] = float(row["final_regret"])

    tasks = {(f, s, r) for f in features for s in scenarios for r in seeds}
    expected_baseline = {(a, t) for a in ("ucb1", "omd_paper_v2_literal") for t in tasks}
    if set(baseline) != expected_baseline or set(candidates) != {(g, t) for g in gammas for t in tasks}:
        raise ValueError("Validation task coverage mismatch")
    scores, paired = [], []
    for gamma in gammas:
        score = dict(gamma=gamma, n_tasks=len(tasks),
                     mean_regret=float(np.mean([candidates[gamma, t] for t in sorted(tasks)])))
        for scenario in scenarios:
            score[scenario] = float(np.mean([
                candidates[gamma, t] for t in sorted(tasks) if t[1] == scenario]))
        for name in ("ucb1", "omd_paper_v2_literal"):
            score[f"diff_{name}"] = float(np.mean([
                candidates[gamma, t] - baseline[name, t] for t in sorted(tasks)]))
        scores.append(score)
        for task in sorted(tasks):
            paired.append(dict(gamma=gamma, feature_seed=task[0], scenario=task[1],
                               seed=task[2], regret=candidates[gamma, task],
                               diff_omd=candidates[gamma, task] - baseline["omd_paper_v2_literal", task],
                               diff_ucb1=candidates[gamma, task] - baseline["ucb1", task]))
    best = min(scores, key=lambda r: (r["mean_regret"], r["gamma"]))
    selected = {"domd_paper_v1": {"gamma": best["gamma"]}}
    write_json(output / "validation_scores.json", scores)
    write_json(output / "paired_results.json", paired)
    write_json(output / "selected_domd.json", selected)
    print("\nVALIDATION BASELINES (same 27 tasks)")
    for name in ("ucb1", "omd_paper_v2_literal"):
        print(f"{name:24s} {np.mean([baseline[name, t] for t in sorted(tasks)]):.6f}")
    print("\ngamma   overall  stationary    abrupt     drift   diff-OMD  diff-UCB1")
    for r in scores:
        print(f"{r['gamma']:.3f} {r['mean_regret']:9.4f} {r['stationary']:11.4f}"
              f" {r['abrupt']:9.4f} {r['drift']:9.4f}"
              f" {r['diff_omd_paper_v2_literal']:10.4f} {r['diff_ucb1']:10.4f}")
    print("SELECTED:", json.dumps(selected))
    print("VALIDATION PASS: 189 episodes; reserved test tasks were not run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
