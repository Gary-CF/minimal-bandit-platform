from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPERIMENTS = (
    "easy_gap",
    "hard_gap",
    "different_horizon",
)

REPORTS_DIR = Path("reports")
RESULTS_DIR = Path("results")


def load_final_regrets(
    experiment_name: str,
) -> dict[tuple[str, int], list[float]]:
    experiment_dir = (
        RESULTS_DIR
        / experiment_name
    )

    csv_paths = sorted(
        experiment_dir.glob("*.csv")
    )

    if not csv_paths:
        raise FileNotFoundError(
            f"no CSV files found in "
            f"{experiment_dir}"
        )

    grouped: dict[
        tuple[str, int],
        list[float],
    ] = defaultdict(list)

    for csv_path in csv_paths:
        with csv_path.open(
            mode="r",
            encoding="utf-8",
            newline="",
        ) as file:
            rows = list(
                csv.DictReader(file)
            )

        if not rows:
            raise ValueError(
                f"empty CSV: {csv_path}"
            )

        final_row = rows[-1]

        algorithm = (
            final_row["algorithm"]
        )
        horizon = int(
            final_row["horizon"]
        )
        final_regret = float(
            final_row[
                "cumulative_regret"
            ]
        )

        grouped[
            (algorithm, horizon)
        ].append(
            final_regret
        )

    return grouped


def summarize(
    values: list[float],
) -> tuple[int, float, float, float]:
    array = np.asarray(
        values,
        dtype=float,
    )

    n = len(array)

    mean = float(
        np.mean(array)
    )

    if n > 1:
        std = float(
            np.std(
                array,
                ddof=1,
            )
        )
    else:
        std = 0.0

    sem = (
        std / math.sqrt(n)
        if n > 0
        else float("nan")
    )

    return (
        n,
        mean,
        std,
        sem,
    )


def main() -> None:
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_csv = (
        REPORTS_DIR
        / "benchmark_summary.csv"
    )

    rows: list[
        dict[str, object]
    ] = []

    for experiment_name in EXPERIMENTS:
        grouped = load_final_regrets(
            experiment_name
        )

        for (
            algorithm,
            horizon,
        ), values in sorted(
            grouped.items(),
            key=lambda item: (
                item[0][1],
                item[0][0],
            ),
        ):
            (
                n,
                mean,
                std,
                sem,
            ) = summarize(values)

            rows.append(
                {
                    "experiment": (
                        experiment_name
                    ),
                    "algorithm": algorithm,
                    "horizon": horizon,
                    "num_seeds": n,
                    "mean_final_regret": mean,
                    "std_final_regret": std,
                    "sem_final_regret": sem,
                }
            )

    with output_csv.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as file:
        fieldnames = [
            "experiment",
            "algorithm",
            "horizon",
            "num_seeds",
            "mean_final_regret",
            "std_final_regret",
            "sem_final_regret",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(
        "experiment, algorithm, horizon, "
        "n, mean, std, sem"
    )

    for row in rows:
        print(
            f"{row['experiment']}, "
            f"{row['algorithm']}, "
            f"{row['horizon']}, "
            f"{row['num_seeds']}, "
            f"{row['mean_final_regret']:.3f}, "
            f"{row['std_final_regret']:.3f}, "
            f"{row['sem_final_regret']:.3f}"
        )

    print()
    print(
        f"summary written to "
        f"{output_csv}"
    )


if __name__ == "__main__":
    main()