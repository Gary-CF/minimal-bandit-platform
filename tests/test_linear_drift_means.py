"""Tests for the Day-3 linear drift trajectory and its existing episode loop.

Interface under test:
    LinearDriftMeans(start_means, end_means, start_t, end_t)
    trajectory.at(t) -> tuple of means

Contract:
- start_t and end_t are Python integers, excluding bool;
  1 <= start_t < end_t.
- start/end vectors have the same positive length and finite probabilities.
- t is a Python integer excluding bool, and t >= 1.
- At/before start_t use the start means; at/after end_t use the end means.
- Strictly between the endpoints, interpolate linearly.
- Queries do not depend on earlier calls, and the input lists are copied.

This file contains tests only, not the trajectory implementation.
"""

import pytest

from envs.mean_trajectories import LinearDriftMeans


def example_trajectory():
    return LinearDriftMeans(
        start_means=[0.2, 0.8],
        end_means=[0.8, 0.2],
        start_t=3,
        end_t=7,
    )


def test_linear_drift_query_matches_hand_calculated_table():
    trajectory = example_trajectory()
    expected = [
        (1, (0.2, 0.8)),
        (2, (0.2, 0.8)),
        (3, (0.2, 0.8)),
        (4, (0.35, 0.65)),
        (5, (0.5, 0.5)),
        (6, (0.65, 0.35)),
        (7, (0.8, 0.2)),
        (8, (0.8, 0.2)),
        (100, (0.8, 0.2)),
    ]
    for t, means in expected:
        actual = trajectory.at(t)
        assert isinstance(actual, tuple)
        assert actual == pytest.approx(means, rel=1e-12, abs=1e-12), f"t={t}"


def test_linear_drift_query_is_independent_of_query_order():
    trajectory = example_trajectory()
    for t, expected in [
        (100, (0.8, 0.2)),
        (1, (0.2, 0.8)),
        (6, (0.65, 0.35)),
        (4, (0.35, 0.65)),
        (4, (0.35, 0.65)),
        (3, (0.2, 0.8)),
    ]:
        assert trajectory.at(t) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_linear_drift_copies_both_input_vectors():
    start_means = [0.2, 0.8]
    end_means = [0.8, 0.2]
    trajectory = LinearDriftMeans(start_means, end_means, 3, 7)

    start_means[0] = 0.9
    end_means[1] = 0.9
    start_means.append(0.5)

    assert trajectory.at(1) == (0.2, 0.8)
    assert trajectory.at(5) == pytest.approx((0.5, 0.5))
    assert trajectory.at(100) == (0.8, 0.2)


def test_linear_drift_accepts_one_arm_and_tuple_inputs():
    trajectory = LinearDriftMeans((0.2,), (0.8,), 3, 7)
    assert trajectory.at(4) == pytest.approx((0.35,))
    assert trajectory.at(7) == (0.8,)


def test_linear_drift_same_endpoints_can_be_stationary_without_normalizing():
    # These are per-arm Bernoulli probabilities, not one categorical distribution.
    trajectory = LinearDriftMeans([0.8, 0.9], [0.8, 0.9], 3, 7)
    for t in (1, 3, 4, 5, 6, 7, 100):
        assert trajectory.at(t) == pytest.approx((0.8, 0.9))


def test_linear_drift_accepts_probability_endpoints_and_one_time_interval():
    trajectory = LinearDriftMeans([0.0, 1.0], [1.0, 0.0], 3, 4)
    assert trajectory.at(2) == (0.0, 1.0)
    assert trajectory.at(3) == (0.0, 1.0)
    assert trajectory.at(4) == (1.0, 0.0)
    assert trajectory.at(5) == (1.0, 0.0)


def test_linear_drift_rejects_invalid_time_parameter_types():
    for name in ("start_t", "end_t"):
        for invalid in (True, False, 3.0, "3", None):
            times = {"start_t": 3, "end_t": 7}
            times[name] = invalid
            with pytest.raises(TypeError):
                LinearDriftMeans([0.2, 0.8], [0.8, 0.2], **times)


def test_linear_drift_rejects_nonpositive_or_unordered_time_endpoints():
    for start_t, end_t in ((0, 7), (-1, 7), (3, 0), (3, -1), (3, 3), (7, 3)):
        with pytest.raises(ValueError):
            LinearDriftMeans([0.2, 0.8], [0.8, 0.2], start_t, end_t)


def test_linear_drift_rejects_invalid_query_time_types():
    trajectory = example_trajectory()
    for invalid in (True, False, 1.0, "3", None):
        with pytest.raises(TypeError):
            trajectory.at(invalid)


def test_linear_drift_rejects_nonpositive_query_times():
    trajectory = example_trajectory()
    for invalid in (0, -1):
        with pytest.raises(ValueError):
            trajectory.at(invalid)


def test_linear_drift_rejects_empty_or_mismatched_mean_vectors():
    invalid_pairs = [
        ([], []),
        ([], [0.2]),
        ([0.2], []),
        ([0.2, 0.8], [0.9]),
        ([0.2], [0.8, 0.2]),
    ]
    for start_means, end_means in invalid_pairs:
        with pytest.raises(ValueError):
            LinearDriftMeans(start_means, end_means, 3, 7)


def test_linear_drift_rejects_invalid_probabilities_at_either_endpoint():
    for invalid in (-0.1, 1.2, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            LinearDriftMeans([0.2, invalid], [0.8, 0.2], 3, 7)
        with pytest.raises(ValueError):
            LinearDriftMeans([0.2, 0.8], [0.8, invalid], 3, 7)


def test_linear_drift_stays_between_each_arms_endpoints():
    start_means = [0.0, 1.0, 0.3]
    end_means = [1.0, 0.0, 0.7]
    trajectory = LinearDriftMeans(start_means, end_means, 2, 9)
    for t in range(1, 15):
        means = trajectory.at(t)
        assert len(means) == 3
        for actual, start, end in zip(means, start_means, end_means):
            assert min(start, end) - 1e-12 <= actual <= max(start, end) + 1e-12


def test_linear_drift_integrates_with_existing_environment_and_episode():
    # Local imports keep the trajectory-only tests independent of runner imports.
    from envs.nonstationary_bernoulli_bandit import NonStationaryBernoulliBandit
    from run import run_nonstationary_episode

    class RecordingRNG:
        """Test double: return a fixed reward, record the actual sampling probability."""

        def __init__(self):
            self.calls = []

        def binomial(self, n, p):
            self.calls.append((n, p))
            return 1

    class AlwaysChooseZero:
        def __init__(self):
            self.feedback = []

        def select_action(self):
            return 0

        def update(self, action, reward):
            self.feedback.append((action, reward))

    rng = RecordingRNG()
    env = NonStationaryBernoulliBandit(example_trajectory(), rng)
    policy = AlwaysChooseZero()
    records = run_nonstationary_episode(env, policy, horizon=8)

    expected_probabilities = [0.2, 0.2, 0.2, 0.35, 0.5, 0.65, 0.8, 0.8]
    expected_regrets = [0.6, 0.6, 0.6, 0.3, 0.0, 0.0, 0.0, 0.0]
    expected_cumulative = [0.6, 1.2, 1.8, 2.1, 2.1, 2.1, 2.1, 2.1]

    assert len(records) == 8
    assert [row["step"] for row in records] == list(range(1, 9))
    assert [n for n, _ in rng.calls] == [1] * 8
    assert [p for _, p in rng.calls] == pytest.approx(expected_probabilities)
    assert [row["instant_regret"] for row in records] == pytest.approx(expected_regrets)
    assert [row["cumulative_regret"] for row in records] == pytest.approx(expected_cumulative)
    assert policy.feedback == [(0, 1.0)] * 8
