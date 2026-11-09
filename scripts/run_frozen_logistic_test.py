"""Freeze validation-selected settings, then evaluate the reserved test tasks."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.run_logistic_benchmark import ROOT, digest, read_csv, run, verify, write_json
from scripts.validate_memory_scale import source_hashes

ALGORITHMS = ("random", "ucb1", "sw_ucb", "discounted_ucb", "omd_paper_v2_literal",
              "domd_paper_v1", "scb_weight_ucb_numeric_v1")
RESULTS = ("test_aggregate.json", "test_paired.json", "test_by_feature.json", "test_audit.json")
RESERVED = dict(feature_seeds=[20261120, 20261121, 20261122],
                reward_seeds=[200, 201, 202, 203, 204])


def build_protocol(args):
    evidence, settings = {}, {a: {} for a in ALGORITHMS}
    for label, folder, filename in (
        ("memory", args.memory_validation, "selected_memory.json"),
        ("domd", args.domd_validation, "selected_domd.json"),
        ("scb", args.scb_validation, "selected_scb.json"),
    ):
        saved = {key: json.loads((folder / file).read_text()) for key, file in
                 (("protocol", "protocol.json"), ("scores", "validation_scores.json"), ("selected", filename))}
        if saved["protocol"]["reserved_test"] != RESERVED:
            raise ValueError(f"Reserved test tasks differ: {label}")
        if label == "memory":
            expected = {}
            for alg in ("sw_ucb", "discounted_ucb"):
                best = min((r for r in saved["scores"] if r["algorithm"] == alg),
                           key=lambda r: (r["mean_regret"], r["window"]))
                expected[alg] = ({"window_size": best["window"]} if alg == "sw_ucb"
                                 else {"gamma": best["gamma"]})
        else:
            alg = "domd_paper_v1" if label == "domd" else "scb_weight_ucb_numeric_v1"
            best = min(saved["scores"], key=lambda r: (r["mean_regret"], r["gamma"]))
            expected = {alg: {"gamma": best["gamma"]}}
        if saved["selected"] != expected:
            raise ValueError(f"Selection does not match validation scores: {label}")
        settings.update(expected)
        evidence[label] = saved
    base = dict(evidence["memory"]["protocol"]["base_config"])
    for label in ("domd", "scb"):
        other = evidence[label]["protocol"]["base_config"]
        for key in ("dimension", "num_arms", "horizon", "change_step", "radius", "delta",
                    "theta_before", "theta_after", "scenarios"):
            if base[key] != other[key]:
                raise ValueError(f"Validation environments differ: {label}/{key}")
    if evidence["scb"]["protocol"]["frozen_domd"] != evidence["domd"]["selected"]:
        raise ValueError("SCB validation used a different DOMD configuration")
    base.update(stage="frozen_reserved_test", seeds=RESERVED["reward_seeds"], log_every=1000)
    return dict(base_config=base, test_tasks=RESERVED, settings=settings,
                algorithms=list(ALGORITHMS), source_sha256=source_hashes(),
                validation_evidence=evidence,
                selection_rule="Frozen before test; no selection using test results",
                uncertainty_note="Report paired task differences and feature-group means; no pooled iid SEM")


def collect(output, protocol, readonly):
    base, rows, child_manifests = protocol["base_config"], [], []
    audit = dict(scb_updates=0, scb_projection_steps=0, max_scb_projection_kkt=0.0)
    for feature_seed in protocol["test_tasks"]["feature_seeds"]:
        for alg in protocol["algorithms"]:
            config = dict(base, feature_seed=feature_seed, algorithms=[alg])
            config.update(protocol["settings"][alg])
            destination = output / f"feature{feature_seed}_{alg}"
            if destination.exists():
                if json.loads((destination / "config.json").read_text()) != config:
                    raise ValueError(f"Configuration differs: {destination}")
                verify(destination)
                print(f"VERIFIED: {destination.name}", flush=True)
            elif readonly:
                raise ValueError(f"Missing test bundle: {destination}")
            else:
                run(config, destination)
            child_manifests.append(destination / "manifest.json")
            for row in read_csv(destination / "summary.csv"):
                rows.append(dict(feature_seed=feature_seed, scenario=row["scenario"],
                                 seed=int(row["seed"]), algorithm=row["algorithm"],
                                 regret=float(row["final_regret"]),
                                 select_us=float(row["select_seconds"]) / base["horizon"] * 1e6,
                                 update_us=float(row["update_seconds"]) / base["horizon"] * 1e6))
                if alg == "scb_weight_ucb_numeric_v1":
                    episode = destination / "episodes" / f"{row['scenario']}__{alg}__seed{row['seed']}.csv"
                    for r in read_csv(episode):
                        audit["scb_updates"] += 1
                        audit["scb_projection_steps"] += int(r["scb_projection_active_post"])
                        audit["max_scb_projection_kkt"] = max(
                            audit["max_scb_projection_kkt"], float(r["scb_projection_kkt_post"]))
    tasks = [(f, s, r) for f in protocol["test_tasks"]["feature_seeds"]
             for s in base["scenarios"] for r in base["seeds"]]
    key = lambda r: (r["feature_seed"], r["scenario"], r["seed"])
    lookup = {(r["algorithm"], key(r)): r["regret"] for r in rows}
    expected = {(a, t) for a in protocol["algorithms"] for t in tasks}
    if len(rows) != len(expected) or set(lookup) != expected:
        raise ValueError("Incomplete or duplicate test task coverage")
    for row in rows:
        row["diff_ucb1"] = row["regret"] - lookup["ucb1", key(row)]
        row["diff_domd"] = row["regret"] - lookup["domd_paper_v1", key(row)]
    aggregate, by_feature = [], []
    metrics = ("regret", "diff_ucb1", "diff_domd", "select_us", "update_us")
    for alg in protocol["algorithms"]:
        part = [r for r in rows if r["algorithm"] == alg]
        result = dict(algorithm=alg, n_tasks=len(part))
        result.update({m: float(np.mean([r[m] for r in part])) for m in metrics})
        for scenario in base["scenarios"]:
            result[scenario] = float(np.mean([r["regret"] for r in part if r["scenario"] == scenario]))
        aggregate.append(result)
        for feature_seed in protocol["test_tasks"]["feature_seeds"]:
            group = [r for r in part if r["feature_seed"] == feature_seed]
            by_feature.append(dict(algorithm=alg, feature_seed=feature_seed, n_tasks=len(group),
                                   **{m: float(np.mean([r[m] for r in group])) for m in metrics}))
    return dict(zip(RESULTS, (aggregate, rows, by_feature, audit))), child_manifests


def main(args):
    out = args.output
    if args.verify_only:
        protocol = json.loads((out / "protocol.json").read_text())
        manifest = json.loads((out / "manifest.json").read_text())
        for name, expected_hash in manifest.items():
            path = (out / name).resolve()
            if not path.is_relative_to(out.resolve()) or digest(path) != expected_hash:
                raise ValueError(f"Test manifest mismatch: {name}")
    else:
        protocol = build_protocol(args)
        if out.exists():
            if json.loads((out / "protocol.json").read_text()) != protocol:
                raise ValueError("Frozen protocol/source differs; do not silently change a test run")
        else:
            out.mkdir(parents=True)
            write_json(out / "protocol.json", protocol)  # Written before any test episode.
    results, child_manifests = collect(out, protocol, args.verify_only)
    files = [out / "protocol.json", *(out / name for name in RESULTS), *child_manifests]
    if args.verify_only:
        if set(manifest) != {p.relative_to(out).as_posix() for p in files}:
            raise ValueError("Test manifest file set mismatch")
        for name, value in results.items():
            if json.loads((out / name).read_text()) != value:
                raise ValueError(f"Recomputed test summary mismatch: {name}")
    else:
        for name, value in results.items():
            write_json(out / name, value)
        write_json(out / "manifest.json", {p.relative_to(out).as_posix(): digest(p) for p in files})
    print("\nFROZEN TEST SETTINGS:", json.dumps(protocol["settings"]))
    print("\nalgorithm                    overall stationary    abrupt     drift  diff-UCB1  diff-DOMD")
    for r in results["test_aggregate.json"]:
        print(f"{r['algorithm']:28s} {r['regret']:8.4f} {r['stationary']:10.4f}"
              f" {r['abrupt']:9.4f} {r['drift']:9.4f} {r['diff_ucb1']:10.4f} {r['diff_domd']:10.4f}")
    print("SCB AUDIT:", json.dumps(results["test_audit.json"]))
    label = "TEST VERIFY PASS (read-only)" if args.verify_only else "FROZEN TEST PASS"
    print(f"{label}: {len(results['test_paired.json'])} episodes; no test-time tuning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    for label in ("memory", "domd", "scb"):
        parser.add_argument(f"--{label}-validation", type=Path,
                            default=ROOT / f"artifacts/v2_2_{label}_validation_r1")
    main(parser.parse_args())
