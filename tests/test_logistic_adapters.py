import numpy as np
import pytest

from algorithms.base import RandomPolicy
from algorithms.ucb1 import UCB1
from algorithms.sliding_window_ucb import SlidingWindowUCB
from algorithms.discounted_ucb import DiscountedUCB
from algorithms.logistic_omd import LogisticOMDPolicy
from algorithms.logistic_adapters import MABAdapter, LogisticOMDAdapter


@pytest.mark.parametrize("factory", [
    lambda: RandomPolicy(4, np.random.default_rng(7)),
    lambda: UCB1(4),
    lambda: SlidingWindowUCB(4, window_size=10),
    lambda: DiscountedUCB(4, gamma=0.9),
])
def test_mab_adapter_preserves_behavior(factory):
    adapted, original = MABAdapter(factory()), factory()
    for reward in np.random.default_rng(11).integers(0, 2, size=40):
        action = adapted.select_action()
        assert action == original.select_action()
        adapted.update(action, int(reward))
        original.update(action, int(reward))
    assert adapted.predict_means() is None


def test_omd_adapter_preserves_behavior_and_copies_features():
    features = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    source = features.copy()
    adapted = LogisticOMDAdapter(LogisticOMDPolicy(2), source)
    original = LogisticOMDPolicy(2)
    source[:] = 0
    np.testing.assert_array_equal(adapted.predict_means(), np.full(4, 0.5))
    for reward in np.random.default_rng(11).integers(0, 2, size=40):
        action = adapted.select_action()
        assert action == original.select_action(features)
        adapted.update(action, int(reward))
        original.update(features[action], int(reward))
        np.testing.assert_array_equal(adapted.policy.estimator.theta, original.estimator.theta)
        np.testing.assert_array_equal(adapted.policy.estimator.H, original.estimator.H)
        expected = 1 / (1 + np.exp(-(features @ original.estimator.theta)))
        np.testing.assert_allclose(adapted.predict_means(), expected)


@pytest.mark.parametrize("action", [-1, 2, True, 0.5])
def test_omd_adapter_rejects_invalid_action_without_update(action):
    adapted = LogisticOMDAdapter(LogisticOMDPolicy(1), [[1.], [-1.]])
    with pytest.raises((ValueError, IndexError)):
        adapted.update(action, 1)
    assert adapted.policy.estimator.completed_steps == 0
