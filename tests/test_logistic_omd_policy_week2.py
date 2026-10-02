import numpy as np
import pytest

from algorithms.logistic_omd import LogisticOMDPolicy, optimistic_scores


def test_scores_match_hand_calculation():
    X, theta, H = np.eye(2), np.array([0.4, 0.2]), np.diag([4.0, 1.0])
    np.testing.assert_allclose(optimistic_scores(X, theta, H, 0.0), [0.4, 0.2])
    scores = optimistic_scores(X, theta, H, 1.0)
    np.testing.assert_allclose(scores, [0.9, 1.2])
    assert np.argmax(scores) == 1


def test_literal_parameters_and_radius():
    policy = LogisticOMDPolicy(3, radius=2.0, delta=0.05)
    assert policy.estimator.eta == 3.0
    assert policy.estimator.l2 == 126.0
    expected = np.sqrt(2016.0 + 6.0 * np.log(20.0) + 171.0 * np.log1p(1.0 / 504.0))
    assert policy.confidence_radius() == pytest.approx(expected)
    policy.update(np.array([1.0, 0.0, 0.0]), 1.0)
    assert policy.confidence_radius() == pytest.approx(expected)


def test_appendix_mode_uses_observed_sample_count():
    policy = LogisticOMDPolicy(3, beta_mode="appendix_time")
    initial = np.sqrt(2016.0 + 6.0 * np.log(20.0))
    assert policy.confidence_radius() == pytest.approx(initial)
    policy.update(np.array([1.0, 0.0, 0.0]), 1.0)
    assert policy.confidence_radius() > initial


def test_selection_does_not_update_and_ties_are_deterministic():
    policy = LogisticOMDPolicy(2)
    theta, H = policy.estimator.theta.copy(), policy.estimator.H.copy()
    assert policy.select_action(np.eye(2)) == 0
    assert policy.select_action(np.eye(2)) == 0
    assert policy.estimator.completed_steps == 0
    np.testing.assert_array_equal(policy.estimator.theta, theta)
    np.testing.assert_array_equal(policy.estimator.H, H)


@pytest.mark.parametrize("features", [np.array([[2.0, 0.0]]), np.array([[np.nan, 0.0]]), np.eye(3)])
def test_invalid_features_preserve_state(features):
    policy = LogisticOMDPolicy(2)
    with pytest.raises(ValueError):
        policy.select_action(features)
    assert policy.estimator.completed_steps == 0
    np.testing.assert_array_equal(policy.estimator.theta, np.zeros(2))
