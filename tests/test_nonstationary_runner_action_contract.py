"""Runner action validation: validate the policy output before coercing it.

These are additional tests; keep the original runner tests.
Run from the project root with:
    python -m pytest -q tests/test_nonstationary_runner_action_contract.py
"""

import numpy as np
import pytest

from envs.mean_trajectories import StationaryMeans
from envs.nonstationary_bernoulli_bandit import NonStationaryBernoulliBandit
from run import run_nonstationary_episode


class RecordingRNG:
    """Test double: record sampling calls without using random outcomes."""

    def __init__(self):
        self.calls = []

    def binomial(self, n, p):
        self.calls.append((n, p))
        return np.int64(0)


class FixedOutputPolicy:
    def __init__(self, action):
        self.action = action
        self.feedback = []

    def select_action(self):
        return self.action

    def update(self, action, reward):
        self.feedback.append((action, reward))


def test_runner_rejects_invalid_actions_before_casting_or_sampling():
    # int(...) would silently turn every value below into a valid arm number.
    for bad_action in (True, False, 0.9, "0", np.bool_(True), np.float64(0.9)):
        rng = RecordingRNG()
        env = NonStationaryBernoulliBandit(StationaryMeans([0.2, 0.8]), rng)
        policy = FixedOutputPolicy(bad_action)

        with pytest.raises(ValueError):
            run_nonstationary_episode(env, policy, horizon=1)

        assert rng.calls == [], f"Invalid action consumed a sample: {bad_action!r}"
        assert policy.feedback == [], "Invalid action reached policy.update"


def test_runner_keeps_support_for_valid_numpy_integer_actions():
    for valid_action in (0, np.int64(0), np.int32(1)):
        rng = RecordingRNG()
        env = NonStationaryBernoulliBandit(StationaryMeans([0.2, 0.8]), rng)
        policy = FixedOutputPolicy(valid_action)

        records = run_nonstationary_episode(env, policy, horizon=1)

        assert len(records) == 1
        assert type(records[0]["action"]) is int
        assert records[0]["action"] == int(valid_action)
        assert len(rng.calls) == 1
        assert len(policy.feedback) == 1
