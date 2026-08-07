from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


RunData = dict[str, Any]


def load_runs(
    experiment_name: str,
) -> list[RunData]:
    results_dir = (
        Path("results")
        / experiment_name
    )

    csv_paths = sorted(
        results_dir.glob("*.csv")
    )

    if len(csv_paths) == 0:
        raise FileNotFoundError(
            f"no CSV files found in {results_dir}"
        )

    runs: list[RunData] = []

    for csv_path in csv_paths:
        with csv_path.open(
            mode="r",
            encoding="utf-8",
            newline="",
        ) as file:
            rows = list(
                csv.DictReader(file)
            )

        if len(rows) == 0:
            raise ValueError(
                f"empty CSV: {csv_path}"
            )

        algorithm = rows[0]["algorithm"]
        horizon = int(rows[0]["horizon"])
        seed = int(rows[0]["seed"])

        steps = np.asarray(
            [
                int(row["step"])
                for row in rows
            ],
            dtype=int,
        )

        actions = np.asarray(
            [
                int(row["action"])
                for row in rows
            ],
            dtype=int,
        )

        cumulative_regrets = np.asarray(
            [
                float(
                    row["cumulative_regret"]
                )
                for row in rows
            ],
            dtype=float,
        )

        assert len(rows) == horizon
        assert np.array_equal(
            steps,
            np.arange(
                1,
                horizon + 1,
            ),
        )

        runs.append(
            {
                "algorithm": algorithm,
                "horizon": horizon,
                "seed": seed,
                "actions": actions,
                "cumulative_regrets": (
                    cumulative_regrets
                ),
            }
        )

    return runs


def plot_mean_regret(
    runs: list[RunData],
    output_path: Path,
    title: str,
) -> None:
    grouped: dict[
        str,
        list[np.ndarray],
    ] = defaultdict(list)

    for run in runs:
        grouped[
            str(run["algorithm"])
        ].append(
            run["cumulative_regrets"]
        )

    plt.figure(
        figsize=(9, 6)
    )

    for algorithm in sorted(grouped):
        regret_matrix = np.stack(
            grouped[algorithm],
            axis=0,
        )

        mean_regret = np.mean(
            regret_matrix,
            axis=0,
        )

        steps = np.arange(
            1,
            len(mean_regret) + 1,
        )

        plt.plot(
            steps,
            mean_regret,
            label=algorithm,
        )

    plt.xlabel("Step")
    plt.ylabel(
        "Mean cumulative pseudo-regret"
    )
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def plot_horizon_comparison(
    runs: list[RunData],
    output_path: Path,
) -> None:
    grouped: dict[
        tuple[str, int],
        list[float],
    ] = defaultdict(list)

    for run in runs:
        algorithm = str(
            run["algorithm"]
        )

        horizon = int(
            run["horizon"]
        )

        final_regret = float(
            run["cumulative_regrets"][-1]
        )

        grouped[
            (algorithm, horizon)
        ].append(final_regret)

    algorithms = sorted(
        {
            algorithm
            for algorithm, _ in grouped
        }
    )

    plt.figure(
        figsize=(9, 6)
    )

    for algorithm in algorithms:
        horizons = sorted(
            {
                horizon
                for name, horizon in grouped
                if name == algorithm
            }
        )

        mean_final_regrets = [
            float(
                np.mean(
                    grouped[
                        (algorithm, horizon)
                    ]
                )
            )
            for horizon in horizons
        ]

        plt.plot(
            horizons,
            mean_final_regrets,
            marker="o",
            label=algorithm,
        )

    plt.xlabel("Horizon")
    plt.ylabel(
        "Mean final cumulative pseudo-regret"
    )
    plt.title(
        "Final regret across horizons"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def plot_action_frequency(
    runs: list[RunData],
    output_path: Path,
) -> None:
    num_arms = (
        max(
            int(np.max(run["actions"]))
            for run in runs
        )
        + 1
    )

    grouped: dict[
        str,
        list[np.ndarray],
    ] = defaultdict(list)

    for run in runs:
        actions = run["actions"]

        frequency = np.bincount(
            actions,
            minlength=num_arms,
        ).astype(float)

        frequency /= len(actions)

        grouped[
            str(run["algorithm"])
        ].append(frequency)

    algorithms = sorted(grouped)

    x_positions = np.arange(
        num_arms
    )

    bar_width = (
        0.8
        / len(algorithms)
    )

    plt.figure(
        figsize=(9, 6)
    )

    for index, algorithm in enumerate(
        algorithms
    ):
        frequency_matrix = np.stack(
            grouped[algorithm],
            axis=0,
        )

        mean_frequency = np.mean(
            frequency_matrix,
            axis=0,
        )

        offset = (
            index
            - (len(algorithms) - 1) / 2
        ) * bar_width

        plt.bar(
            x_positions + offset,
            mean_frequency,
            width=bar_width,
            label=algorithm,
        )

    plt.xlabel("Arm")
    plt.ylabel(
        "Mean action frequency"
    )
    plt.title(
        "Action frequency in hard-gap experiment"
    )
    plt.xticks(
        x_positions,
        [
            str(arm)
            for arm in range(num_arms)
        ],
    )
    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.3,
    )
    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def main() -> None:
    figures_dir = Path("figures")

    easy_gap_runs = load_runs(
        "easy_gap"
    )

    hard_gap_runs = load_runs(
        "hard_gap"
    )

    horizon_runs = load_runs(
        "different_horizon"
    )

    plot_mean_regret(
        runs=easy_gap_runs,
        output_path=(
            figures_dir
            / "easy_gap_regret.png"
        ),
        title=(
            "Easy-gap mean cumulative regret"
        ),
    )

    plot_mean_regret(
        runs=hard_gap_runs,
        output_path=(
            figures_dir
            / "hard_gap_regret.png"
        ),
        title=(
            "Hard-gap mean cumulative regret"
        ),
    )

    plot_horizon_comparison(
        runs=horizon_runs,
        output_path=(
            figures_dir
            / "horizon_comparison.png"
        ),
    )

    plot_action_frequency(
        runs=hard_gap_runs,
        output_path=(
            figures_dir
            / "action_frequency.png"
        ),
    )

    print(
        "benchmark figures saved to figures/"
    )


if __name__ == "__main__":
    main()