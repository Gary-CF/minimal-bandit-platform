"""Read existing benchmark outputs and print baseline/window diagnostics."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_sem(values):
    if not values:
        return "N/A"
    values = np.asarray(values, dtype=float)
    sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else None
    return f"{values.mean():.6f} +/- {sem:.6f}" if sem is not None else f"{values.mean():.6f} (n=1)"


def window_average(rows, field, start, end):
    selected = [r for r in rows if start <= int(r["step"]) <= end]
    if start < 1 or len(selected) != end - start + 1:
        return None
    if any(r[field] == "" for r in selected):
        return None
    return float(np.mean([float(r[field]) for r in selected]))


def analyze(output):
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    summary = read_csv(output / "summary.csv")
    print("\nFINAL REGRET: mean +/- SEM across reward seeds")
    for scenario in config["scenarios"]:
        print(f"\n[{scenario}]")
        for name in config["algorithms"]:
            values = [float(r["final_regret"]) for r in summary
                      if r["scenario"] == scenario and r["algorithm"] == name]
            print(f"{name:24s} {mean_sem(values)}")
        paired = {}
        for name in ("ucb1", "omd_paper_v2_literal"):
            paired[name] = {int(r["seed"]): float(r["final_regret"]) for r in summary
                            if r["scenario"] == scenario and r["algorithm"] == name}
        if all(set(paired[name]) == set(config["seeds"]) for name in paired):
            diffs = [paired["omd_paper_v2_literal"][s] - paired["ucb1"][s]
                     for s in config["seeds"]]
            print("paired OMD - UCB1:", mean_sem(diffs), "(negative favors OMD)")

    if "abrupt" not in config["scenarios"]:
        return
    change, horizon = config["change_step"], config["horizon"]
    for width in (50, 100, 200):
        windows = {"pre": (change - width, change - 1),
                   "post": (change, change + width - 1),
                   "late": (horizon - width + 1, horizon)}
        print(f"\nABRUPT W={width}: {windows}")
        for name in config["algorithms"]:
            episodes = [read_csv(output / "episodes" / f"abrupt__{name}__seed{s}.csv")
                        for s in config["seeds"]]
            for field in ("instant_regret", "prediction_rmse_pre"):
                stats = []
                for label, (start, end) in windows.items():
                    values = [window_average(rows, field, start, end) for rows in episodes]
                    text = "N/A" if any(v is None for v in values) else mean_sem(values)
                    stats.append(f"{label}={text}")
                if field == "instant_regret" or any(r[field] != "" for r in episodes[0]):
                    print(f"{name:24s} {field:20s} " + " | ".join(stats))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    analyze(parser.parse_args().output)
