import numpy as np
import pytest

from envs.mean_trajectories import (
    StationaryMeans,
    PiecewiseConstantMeans,
)
from envs.nonstationary_bernoulli_bandit import (
    NonStationaryBernoulliBandit,
)


def test_stationary_trajectory_can_drive_environment():
    """同一个环境类也能使用平稳轨迹。"""
    trajectory = StationaryMeans([0.0, 1.0])
    env = NonStationaryBernoulliBandit(
        trajectory=trajectory,
        rng=np.random.default_rng(0),
    )

    assert env.num_arms == 2

    for t in (1, 4, 100):
        assert env.step(0, t) == 0
        assert env.step(1, t) == 1

        assert env.pseudo_regret(0, t) == 1.0
        assert env.pseudo_regret(1, t) == 0.0


def test_change_point_applies_to_reward_and_regret_in_same_round():
    """第4轮开始，采样分布和评价基准必须一起变化。"""
    trajectory = PiecewiseConstantMeans(
        starts=[1, 4],
        levels=[
            [0.0, 1.0],
            [1.0, 0.0],
        ],
    )
    env = NonStationaryBernoulliBandit(
        trajectory=trajectory,
        rng=np.random.default_rng(0),
    )

    # 第3轮：臂0奖励必为0，且不是最优臂。
    assert env.step(0, 3) == 0
    assert env.pseudo_regret(0, 3) == 1.0

    # 第4轮：臂0奖励必为1，且已经成为最优臂。
    assert env.step(0, 4) == 1
    assert env.pseudo_regret(0, 4) == 0.0

    # 再查询第3轮，评价仍应使用第3轮均值。
    assert env.pseudo_regret(0, 3) == 1.0


def test_regret_uses_current_means_not_initial_best_mean():
    """非端点概率下，手工核对各轮的均值差。"""
    trajectory = PiecewiseConstantMeans(
        starts=[1, 4],
        levels=[
            [0.2, 0.8],
            [0.9, 0.1],
        ],
    )
    env = NonStationaryBernoulliBandit(
        trajectory=trajectory,
        rng=np.random.default_rng(0),
    )

    assert env.pseudo_regret(0, 3) == pytest.approx(0.6)
    assert env.pseudo_regret(0, 4) == pytest.approx(0.0)
    assert env.pseudo_regret(1, 4) == pytest.approx(0.8)