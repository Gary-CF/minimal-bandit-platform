from __future__ import annotations

import matplotlib.pyplot as plt
import csv
from pathlib import Path
import numpy as np

def load_experiment(
        csv_path:Path,
)->tuple[str,int,list[int],list[float]]:

    algorithm_name:str | None =None
    seed:int |None =None
    
    steps:list[int]=[]
    cumulative_regrets:list[float]=[]

    with csv_path.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader=csv.DictReader(file)

        for row in reader:
            current_algorithm=row["algorithm"]
            current_seed=int(row["seed"])

            if algorithm_name is None:
                algorithm_name=current_algorithm
                seed=current_seed
            else:
                assert(
                    current_algorithm==algorithm_name
                )
                assert current_seed==seed

            step = int(row["step"])
            cumulative_regret=float(row["cumulative_regret"])

            steps.append(step)
            cumulative_regrets.append(cumulative_regret)

    assert len(steps) == len(cumulative_regrets)
    

    assert algorithm_name is not None
    assert seed is not None


    assert len(steps) > 0

    assert len(steps)==len(cumulative_regrets)

    assert steps == list(
        range(1, len(steps) + 1)
    )

    assert all(
        previous <= current
        for previous, current in zip(
            cumulative_regrets,
            cumulative_regrets[1:],
        )
    )

    return (algorithm_name,seed,steps, cumulative_regrets)

def load_all_curves(results_dir:Path)->tuple[dict[str,list[int]],dict[str,list[list[float]]]]:
    steps_by_algorithm:dict[str,list[int]]={}

    curves_by_algorithm:dict[str,list[list[float]]]={}

    csv_paths=sorted(results_dir.glob("*.csv"))

    if not csv_paths:
        raise FileNotFoundError(
            f"no CSV files found in {results_dir}"
        )
    for csv_path in csv_paths:
        (
            algorithm_name,
            seed,
            steps,
            cumulative_regrets,
        )=load_experiment(csv_path)

        if(
            algorithm_name not in curves_by_algorithm
        ):
            curves_by_algorithm[algorithm_name]=[]
            steps_by_algorithm[algorithm_name]=steps
        else:
            assert(
                steps==steps_by_algorithm[algorithm_name]
            )
        curves_by_algorithm[algorithm_name].append(cumulative_regrets)

        print(
            f"loaded algorithm= {algorithm_name}"
            f"seed={seed}"
            f"records={len(steps)}"
        )

    return(
            steps_by_algorithm,
            curves_by_algorithm
        )


def plot_mean_curves(
    steps_by_algorithm: dict[
        str,
        list[int],
    ],
    curves_by_algorithm: dict[
        str,
        list[list[float]],
    ],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(8, 5)
    )

    for algorithm_name in sorted(
        curves_by_algorithm
    ):
        curves = curves_by_algorithm[
            algorithm_name
        ]

        regret_matrix = np.asarray(
            curves,
            dtype=float,
        )

        assert regret_matrix.ndim == 2

        mean_regrets = np.mean(
            regret_matrix,
            axis=0,
        )

        steps = steps_by_algorithm[
            algorithm_name
        ]

        assert len(mean_regrets) == len(
            steps
        )

        plt.plot(
            steps,
            mean_regrets,
            label=algorithm_name,
        )

        print(
            f"{algorithm_name}: "
            f"{regret_matrix.shape[0]} seeds, "
            f"final mean regret="
            f"{mean_regrets[-1]:.4f}"
        )

    plt.xlabel("Step")
    plt.ylabel("Mean cumulative regret")
    plt.title(
        "Mean cumulative regret "
        "across seeds"
    )

    plt.legend()

    plt.grid(
        visible=True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()





def main()->None:
    results_dir=Path(
        "results"
    )

    output_path=Path(
        "figures/regret_curve.png"
    )

    (steps_by_algorithm,curves_by_algorithm)=load_all_curves(
        results_dir=results_dir
    )

    print()

    for algorithm_name in sorted(
        curves_by_algorithm
    ):
        print(
            f"algorithm={algorithm_name}"
            f"number of curves="
            f"{len(curves_by_algorithm[algorithm_name])}"
        )

    print()

    plot_mean_curves(
            steps_by_algorithm=(steps_by_algorithm),
            curves_by_algorithm=(curves_by_algorithm),
            output_path=output_path,
        )

    print()
    print(
            f"saved figure to:{output_path}"
        )

if __name__=="__main__":
    main()