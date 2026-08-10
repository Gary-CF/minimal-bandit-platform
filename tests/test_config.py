import json

import pytest

from run import (
    normalize_config,
    save_config_snapshot,
    validate_results_directory,
)

def test_normalize_config_converts_single_horizon() -> None:
    raw_config = {
        "experiment_name": "test",
        "environment": {
            "name": "bernoulli",
            "arm_means": [0.2, 0.8],
        },
        "algorithms": [
            "ucb1",
        ],
        "horizon": 100,
        "seeds": [0],
    }

    config = normalize_config(
        raw_config
    )

    assert config["horizons"] == [100]

    assert config["algorithms"] == [
        {
            "name": "ucb1",
            "parameters": {},
        }
    ]

@pytest.mark.parametrize(
    "field,value",
    [
        ("algorithms", []),
        ("seeds", []),
        ("horizon", 0),
        ("horizon", -1),
    ],
)
def test_normalize_config_rejects_invalid_values(
    field: str,
    value: object,
) -> None:
    raw_config = {
        "experiment_name": "test",
        "environment": {
            "name": "bernoulli",
            "arm_means": [0.2, 0.8],
        },
        "algorithms": ["ucb1"],
        "horizon": 10,
        "seeds": [0],
    }

    raw_config[field] = value

    with pytest.raises(ValueError):
        normalize_config(
            raw_config
        )

def test_results_directory_accepts_same_config(
    tmp_path,
) -> None:
    results_dir = (
        tmp_path
        / "experiment"
    )

    config = {
        "experiment_name": "experiment",
        "environment": {
            "name": "bernoulli",
            "arm_means": [0.2, 0.8],
        },
        "algorithms": [
            {
                "name": "ucb1",
                "parameters": {},
            }
        ],
        "horizons": [100],
        "seeds": [0],
    }

    save_config_snapshot(
        config=config,
        output_path=(
            results_dir
            / "config_snapshot.json"
        ),
    )

    validate_results_directory(
        results_dir=results_dir,
        config=config,
    )

def test_results_directory_rejects_different_config(
    tmp_path,
) -> None:
    results_dir = (
        tmp_path
        / "experiment"
    )

    existing_config = {
        "experiment_name": "experiment",
        "environment": {
            "name": "bernoulli",
            "arm_means": [0.2, 0.8],
        },
        "algorithms": [],
        "horizons": [100],
        "seeds": [0],
    }

    new_config = dict(
        existing_config
    )
    new_config["seeds"] = [1]

    save_config_snapshot(
        config=existing_config,
        output_path=(
            results_dir
            / "config_snapshot.json"
        ),
    )

    with pytest.raises(RuntimeError):
        validate_results_directory(
            results_dir=results_dir,
            config=new_config,
        )
def test_results_directory_rejects_unknown_provenance(
    tmp_path,
) -> None:
    results_dir = (
        tmp_path
        / "experiment"
    )

    results_dir.mkdir()

    (
        results_dir
        / "old_result.csv"
    ).write_text(
        "stale result",
        encoding="utf-8",
    )

    config = {
        "experiment_name": "experiment",
        "environment": {},
        "algorithms": [],
        "horizons": [10],
        "seeds": [0],
    }

    with pytest.raises(RuntimeError):
        validate_results_directory(
            results_dir=results_dir,
            config=config,
        )