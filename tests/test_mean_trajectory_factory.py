"""配置 -> 均值轨迹的契约测试；不包含被测实现。

在项目根目录运行：
    python -m pytest -q tests/test_mean_trajectory_factory.py

被测接口：run.create_mean_trajectory(trajectory_config)
配置结构/字段错误用 ValueError；构造器的参数异常原样保留。
"""

from copy import deepcopy
import json

import pytest

from envs.mean_trajectories import (
    LinearDriftMeans,
    PiecewiseConstantMeans,
    StationaryMeans,
)
from run import create_mean_trajectory


def example_configs():
    """每次返回一套全新的配置，避免测试间共享可变输入。"""
    return [
        {"name": "stationary", "means": [0.2, 0.8]},
        {
            "name": "piecewise_constant",
            "starts": [1, 4],
            "levels": [[0.2, 0.8], [0.9, 0.1]],
        },
        {
            "name": "linear_drift",
            "start_means": [0.2, 0.8],
            "end_means": [0.8, 0.2],
            "start_t": 3,
            "end_t": 7,
        },
    ]


def test_factory_builds_stationary_means():
    trajectory = create_mean_trajectory(example_configs()[0])
    assert isinstance(trajectory, StationaryMeans)
    for t in (1, 4, 100):
        actual = trajectory.at(t)
        assert isinstance(actual, tuple)
        assert actual == pytest.approx((0.2, 0.8))


def test_factory_builds_piecewise_constant_means():
    trajectory = create_mean_trajectory(example_configs()[1])
    assert isinstance(trajectory, PiecewiseConstantMeans)
    for t, expected in (
        (1, (0.2, 0.8)),
        (3, (0.2, 0.8)),
        (4, (0.9, 0.1)),
        (100, (0.9, 0.1)),
    ):
        actual = trajectory.at(t)
        assert isinstance(actual, tuple)
        assert actual == pytest.approx(expected)


def test_factory_builds_linear_drift_means():
    trajectory = create_mean_trajectory(example_configs()[2])
    assert isinstance(trajectory, LinearDriftMeans)
    for t, expected in (
        (1, (0.2, 0.8)),
        (3, (0.2, 0.8)),
        (4, (0.35, 0.65)),
        (5, (0.5, 0.5)),
        (6, (0.65, 0.35)),
        (7, (0.8, 0.2)),
        (100, (0.8, 0.2)),
    ):
        actual = trajectory.at(t)
        assert isinstance(actual, tuple)
        assert actual == pytest.approx(expected)


def test_factory_accepts_json_decoded_configuration():
    """这里只测解析后的字典，不要求完整 CLI 已经接入。"""
    for config in example_configs():
        decoded = json.loads(json.dumps(config))
        trajectory = create_mean_trajectory(decoded)
        direct = create_mean_trajectory(config)
        for t in (1, 4, 5, 100):
            assert trajectory.at(t) == pytest.approx(direct.at(t))


def test_factory_rejects_non_dictionary_configuration():
    for invalid in (None, [], "stationary", 1, True):
        with pytest.raises(ValueError):
            create_mean_trajectory(invalid)
