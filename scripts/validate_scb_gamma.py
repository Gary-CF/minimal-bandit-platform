"""Validate SCB gamma against UCB1, OMD and the frozen DOMD configuration."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.run_logistic_benchmark import ROOT, read_csv, run, verify, write_json
from scripts.validate_memory_scale import source_hashes


def main(output, domd_selection):
    selected_domd = json.loads(domd_selection.read_text())
    domd_gamma = selected_domd["domd_paper_v1"]["gamma"]
    if type(domd_gamma) not in (int, float) or not 0 < domd_gamma < 1:
        raise ValueError("Invalid frozen DOMD gamma")
    gammas = (0.98, 0.99, 0.995, 0.998, 0.999)
    features, seeds = (20261110, 20261111, 20261112), (100, 101, 102)
    scenarios = ("stationary", "abrupt", "drift")
    baselines = ("ucb1", "omd_paper_v2_literal", "domd_paper_v1")
    name = "scb_weight_ucb_numeric_v1"
    base = json.loads((ROOT / "configs/v2_2/development.json").read_text())
    base.update(stage="scb_gamma_validation", horizon=1000, change_step=501,
                seeds=list(seeds), scenarios=list(scenarios), log_every=1000)
    protocol = dict(base_config=base, gammas=list(gammas), feature_seeds=list(features),
                    frozen_domd=selected_domd, source_sha256=source_hashes(),
                    scb_regularizer="lambda=d*log(T)/c_mu; fixed formula, not tuned",
                    projection="deterministic multistart SLSQP; local KKT, no global certificate",
                    selection="minimum mean regret over 27 tasks; exact ties choose smaller gamma",
                    reserved_test=dict(feature_seeds=[20261120, 20261121, 20261122],
                                       reward_seeds=[200, 201, 202, 203, 204]))
    if output.exists():
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Protocol/source changed; use a new output directory")
    else:
        output.mkdir(parents=True)
        write_json(output / "protocol.json", protocol)
    reference, candidates, audits = {}, {}, {}
    for feature_seed in features:
        jobs = [("baseline", list(baselines), float(domd_gamma))]
        jobs += [(f"gamma{g:g}", [name], g) for g in gammas]
        for label, algorithms, gamma in jobs:
            config = dict(base, feature_seed=feature_seed, algorithms=algorithms, gamma=gamma)
            destination = output / f"feature{feature_seed}_{label}"
            if destination.exists():
                if json.loads((destination / "config.json").read_text()) != config:
                    raise ValueError(f"Configuration differs: {destination}")
                verify(destination)
                print(f"REUSE VERIFIED: {destination.name}", flush=True)
            else:
                run(config, destination)  # Includes full saved-run verification.
            for row in read_csv(destination / "summary.csv"):
                task = (feature_seed, row["scenario"], int(row["seed"]))
                table = reference if label == "baseline" else candidates
                key = (row["algorithm"], task) if label == "baseline" else (gamma, task)
                if key in table:
                    raise ValueError(f"Duplicate task: {key}")
                table[key] = float(row["final_regret"])
                if label != "baseline":
                    episode = f"{row['scenario']}__{name}__seed{row['seed']}.csv"
                    trajectory = read_csv(destination / "episodes" / episode)
                    timing = read_csv(destination / "timing" / episode)
                    audits[key] = dict(
                        projection_steps=sum(int(r["scb_projection_active_post"]) for r in trajectory),
                        max_mle_norm=max(float(r["scb_mle_norm_post"]) for r in trajectory),
                        max_projection_kkt=max(float(r["scb_projection_kkt_post"]) for r in trajectory),
                        mean_update_us=float(row["update_seconds"]) / base["horizon"] * 1e6,
                        early_update_us=float(np.mean([float(r["update_seconds"]) for r in timing[:100]])) * 1e6,
                        late_update_us=float(np.mean([float(r["update_seconds"]) for r in timing[-100:]])) * 1e6)
    tasks = sorted((f, s, r) for f in features for s in scenarios for r in seeds)
    if (set(reference) != {(a, t) for a in baselines for t in tasks}
            or set(candidates) != {(g, t) for g in gammas for t in tasks}
            or set(audits) != set(candidates)):
        raise ValueError("Incomplete validation task coverage")
    scores, paired = [], []
    for gamma in gammas:
        score = dict(gamma=gamma, n_tasks=len(tasks),
                     mean_regret=float(np.mean([candidates[gamma, t] for t in tasks])),
                     projection_steps=sum(audits[gamma, t]["projection_steps"] for t in tasks),
                     mean_update_us=float(np.mean([audits[gamma, t]["mean_update_us"] for t in tasks])))
        for scenario in scenarios:
            score[scenario] = float(np.mean([candidates[gamma, t] for t in tasks if t[1] == scenario]))
        for alg in baselines:
            score[f"diff_{alg}"] = float(np.mean([candidates[gamma, t] - reference[alg, t] for t in tasks]))
        scores.append(score)
        for task in tasks:
            paired.append(dict(gamma=gamma, feature_seed=task[0], scenario=task[1], seed=task[2],
                               regret=candidates[gamma, task], **audits[gamma, task],
                               **{f"diff_{a}": candidates[gamma, task] - reference[a, task] for a in baselines}))
    best = min(scores, key=lambda r: (r["mean_regret"], r["gamma"]))
    selected = {name: {"gamma": best["gamma"]}}
    write_json(output / "validation_scores.json", scores)
    write_json(output / "paired_results.json", paired)
    write_json(output / "selected_scb.json", selected)
    print(f"\nFROZEN DOMD gamma={domd_gamma}")
    for alg in baselines:
        print(f"{alg:24s} {np.mean([reference[alg, t] for t in tasks]):.6f}")
    print("\ngamma   overall  stationary    abrupt     drift  diff-DOMD  proj_steps  update_us")
    for r in scores:
        print(f"{r['gamma']:.3f} {r['mean_regret']:9.4f} {r['stationary']:11.4f}"
              f" {r['abrupt']:9.4f} {r['drift']:9.4f} {r['diff_domd_paper_v1']:10.4f}"
              f" {r['projection_steps']:11d} {r['mean_update_us']:10.2f}")
    print("SELECTED:", json.dumps(selected))
    print("VALIDATION PASS: 216 episodes; reserved test tasks were not run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domd-selection", type=Path,
                        default=ROOT / "artifacts/v2_2_domd_validation_r1/selected_domd.json")
    args = parser.parse_args()
    main(args.output, args.domd_selection)
