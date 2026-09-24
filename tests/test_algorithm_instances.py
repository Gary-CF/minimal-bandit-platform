"""Tests for the new algorithm-instance configuration layer.

Place this file in tests/test_algorithm_instances.py.
Run from the repository root:
    python -m pytest -q tests/test_algorithm_instances.py

Required new public function in run.py:
    normalize_algorithm_instances(raw_algorithms)

This file contains tests only. It does not implement the normalizer or alter
legacy normalize_config/create_algorithm behavior.
"""

from copy import deepcopy
import json

import numpy as np
import pytest

from run import create_algorithm, normalize_algorithm_instances


def example_algorithms():
    return [
        "random",
        {
            "id": "eps_005",
            "name": "epsilon_greedy",
            "parameters": {"epsilon": 0.05},
        },
        {"name": "ucb1"},
        {
            "id": "eps_020",
            "name": "epsilon_greedy",
            "parameters": {"epsilon": 0.20},
        },
    ]


def test_instances_expand_defaults_preserve_order_and_parameters():
    actual = normalize_algorithm_instances(example_algorithms())
    assert actual == [
        {"id": "random", "name": "random", "parameters": {}},
        {
            "id": "eps_005",
            "name": "epsilon_greedy",
            "parameters": {"epsilon": 0.05},
        },
        {"id": "ucb1", "name": "ucb1", "parameters": {}},
        {
            "id": "eps_020",
            "name": "epsilon_greedy",
            "parameters": {"epsilon": 0.20},
        },
    ]
    assert all(set(item) == {"id", "name", "parameters"} for item in actual)


def test_instances_allow_repeated_names_with_distinct_ids():
    # The uniqueness rule concerns IDs, not algorithm names or parameter values.
    actual = normalize_algorithm_instances([
        {"id": "ucb_a", "name": "ucb1"},
        {"id": "ucb_b", "name": "ucb1", "parameters": {}},
    ])
    assert [item["id"] for item in actual] == ["ucb_a", "ucb_b"]
    assert [item["name"] for item in actual] == ["ucb1", "ucb1"]


def test_instances_reject_duplicate_explicit_and_default_ids():
    cases = [
        ["ucb1", "ucb1"],
        [
            {"name": "epsilon_greedy", "parameters": {"epsilon": 0.05}},
            {"name": "epsilon_greedy", "parameters": {"epsilon": 0.20}},
        ],
        [
            {"id": "same", "name": "random"},
            {"id": "same", "name": "ucb1"},
        ],
        ["random", {"id": "random", "name": "ucb1"}],
    ]
    for raw in cases:
        with pytest.raises(ValueError):
            normalize_algorithm_instances(raw)


def test_instances_require_a_nonempty_list_and_valid_entry_types():
    for raw in (None, [], "random", {}, ("random",), 3, True):
        with pytest.raises(ValueError):
            normalize_algorithm_instances(raw)

    for bad_entry in (None, 3, True, [], ("random",)):
        with pytest.raises(ValueError):
            normalize_algorithm_instances([bad_entry])

