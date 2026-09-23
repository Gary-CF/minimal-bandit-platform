"""Tests for the first, in-memory nonstationary interaction loop.

Expected entry point in run.py:
    run_nonstationary_episode(env, algorithm, horizon)

Each returned row must contain step, action, reward, instant_regret,
and cumulative_regret. Extra metadata fields are allowed for later work.
This file contains tests only, not the runner implementation.
"""

import numpy as np
import pytest

from algorithms.base import BanditAlgorithm, RandomPolicy
from algorithms.ucb1 import UCB1
from envs.mean_trajectories import PiecewiseConstantMeans, StationaryMeans
from envs.nonstationary_bernoulli_bandit import NonStationaryBernoulliBandit
from run import run_nonstationary_episode


class ScriptedPolicy(BanditAlgorithm):
    """Test-only policy: fixed actions, with strict select/update sequencing."""

    def __init__(self, actions):
        self.actions = tuple(actions)
        self.num_arms = 2
        self.selections = 0
        self.observations = []

    def select_action(self):
        # The preceding reward must be delivered before the next selection.
        assert len(self.observations) == self.selections
        assert self.selections < len(self.actions)
        action = self.actions[self.selections]
        self.selections += 1
        return action

    def update(self, action, reward):
        # Exactly one update per selection, with the selected action.
        assert self.selections == len(self.observations) + 1
        assert action == self.actions[len(self.observations)]
        self.observations.append((action, reward))


def test_episode_change_point_feedback_and_cumulative_regret():
    trajectory = PiecewiseConstantMeans(
        starts=[1, 4],
        levels=[[0.0, 1.0], [1.0, 0.0]],
    )
    env = NonStationaryBernoulliBandit(
        trajectory, np.random.default_rng(0)
    )
    actions = [0, 1, 0, 1, 0, 1]
    policy = ScriptedPolicy(actions)

    rows = run_nonstationary_episode(env, policy, horizon=6)

    # Independent hand calculations: no call to env.pseudo_regret here.
    rewards = [0, 1, 0, 0, 1, 0]
    gaps = [1, 0, 1, 1, 0, 1]
    cumulative = [1, 1, 2, 3, 3, 4]
    assert isinstance(rows, list)
    assert len(rows) == 6
    # Appending the same mutable dictionary repeatedly is not a valid log.
    assert len({id(row) for row in rows}) == 6
    assert [row['step'] for row in rows] == [1, 2, 3, 4, 5, 6]
    assert [row['action'] for row in rows] == actions
    assert [row['reward'] for row in rows] == rewards
    assert [row['instant_regret'] for row in rows] == pytest.approx(gaps)
    assert [row['cumulative_regret'] for row in rows] == pytest.approx(cumulative)
    assert policy.selections == 6
    # Check actual sampled rewards were delivered, not the regret values.
    assert policy.observations == list(zip(actions, rewards))


@pytest.mark.parametrize('algorithm_name', ['random', 'ucb1'])
@pytest.mark.parametrize('trajectory_kind', ['stationary', 'piecewise'])
def test_existing_policies_run_and_reproduce(algorithm_name, trajectory_kind):
    horizon = 30

    def fresh_run():
        # Recreate both objects and both RNGs for each independent experiment.
        env_seed, policy_seed = np.random.SeedSequence(19).spawn(2)
        env_rng = np.random.default_rng(env_seed)
        policy_rng = np.random.default_rng(policy_seed)
        if trajectory_kind == 'stationary':
            trajectory = StationaryMeans([0.2, 0.8])
        else:
            trajectory = PiecewiseConstantMeans(
                starts=[1, 12],
                levels=[[0.2, 0.8], [0.9, 0.1]],
            )
        env = NonStationaryBernoulliBandit(trajectory, env_rng)
        if algorithm_name == 'random':
            policy = RandomPolicy(num_arms=env.num_arms, rng=policy_rng)
        else:
            policy = UCB1(num_arms=env.num_arms)
        rows = run_nonstationary_episode(env, policy, horizon=horizon)
        return rows, policy

    first, policy = fresh_run()
    second, _ = fresh_run()
    assert first == second
    assert len(first) == horizon
    assert len({id(row) for row in first}) == horizon

    cumulative = 0.0
    for t, row in enumerate(first, start=1):
        assert row['step'] == t
        assert type(row['action']) is int
        action = row['action']
        assert action in (0, 1)
        assert row['reward'] in (0, 1)
        # Independent reference for this small, deterministic mean schedule.
        means = (0.9, 0.1) if trajectory_kind == 'piecewise' and t >= 12 else (0.2, 0.8)
        gap = max(means) - means[action]
        cumulative += gap
        assert row['instant_regret'] == pytest.approx(gap)
        assert row['cumulative_regret'] == pytest.approx(cumulative)

    if algorithm_name == 'ucb1':
        actions = [row['action'] for row in first]
        rewards = [row['reward'] for row in first]
        assert actions[:2] == [0, 1]
        np.testing.assert_array_equal(
            policy.counts, np.bincount(actions, minlength=2)
        )
        np.testing.assert_allclose(
            policy.reward_sums,
            np.bincount(actions, weights=rewards, minlength=2),
        )


def test_episode_one_round_returns_one_record():
    env = NonStationaryBernoulliBandit(
        StationaryMeans([0.0, 1.0]), np.random.default_rng(0)
    )
    policy = ScriptedPolicy([0])
    rows = run_nonstationary_episode(env, policy, horizon=1)
    assert len(rows) == 1
    assert rows[0]['step'] == 1
    assert rows[0]['cumulative_regret'] == pytest.approx(1.0)
    assert policy.observations == [(0, 0)]


@pytest.mark.parametrize('horizon', [True, False, 0, -1, 2.5, '3', None])
def test_episode_rejects_bad_horizon_before_interaction(horizon):
    env = NonStationaryBernoulliBandit(
        StationaryMeans([0.0, 1.0]),
        np.random.Generator(np.random.PCG64(0)),
    )
    policy = ScriptedPolicy([0])
    state_before = env.rng.bit_generator.state
    with pytest.raises(ValueError):
        run_nonstationary_episode(env, policy, horizon=horizon)
    assert policy.selections == 0
    assert policy.observations == []
    assert env.rng.bit_generator.state == state_before
