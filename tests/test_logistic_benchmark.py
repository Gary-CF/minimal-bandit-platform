import numpy as np
import pytest

from scripts.logistic_benchmark import ALGORITHMS, make_path, run_episode
from scripts.d12_logistic_stress import run_episode as old_episode


FEATURES = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
BEFORE, AFTER = [1., -0.5], [-1., 0.5]


@pytest.mark.parametrize("name", ALGORITHMS)
@pytest.mark.parametrize("scenario", ["stationary", "abrupt", "drift"])
def test_replay_and_independent_regret(name, scenario):
    path = make_path(scenario, 40, BEFORE, AFTER)
    rows = run_episode(name, FEATURES, path, 7, window_size=10, gamma=0.9)
    assert rows == run_episode(name, FEATURES, path, 7, window_size=10, gamma=0.9)
    means = 1 / (1 + np.exp(-(path @ FEATURES.T)))
    actions = np.array([row["action"] for row in rows])
    expected = means.max(axis=1) - means[np.arange(len(path)), actions]
    np.testing.assert_allclose([r["instant_regret"] for r in rows], expected, atol=1e-14)
    np.testing.assert_allclose([r["cumulative_regret"] for r in rows], expected.cumsum())
    assert [r["step"] for r in rows] == list(range(1, 41))
    assert all(r["reward"] in (0, 1) for r in rows)
    if name in ("random", "ucb1", "sw_ucb", "discounted_ucb"):
        assert all(r["prediction_rmse_pre"] is None for r in rows)


@pytest.mark.parametrize("name", ["random", "omd_paper_v2_literal"])
@pytest.mark.parametrize("scenario", ["abrupt", "drift"])
def test_matches_old_episode(name, scenario, tmp_path):
    path = make_path(scenario, 40, BEFORE, AFTER)
    actual = run_episode(name, FEATURES, path, 7)
    config = {"radius": 2., "delta": .05, "beta_mode": "paper_v2_literal"}
    expected = old_episode(tmp_path, scenario, name, 7, FEATURES, path,
                           config, {"solver_failures": 0})
    for a, b in zip(actual, expected):
        for key in ("action", "reward", "instant_regret", "cumulative_regret"):
            assert a[key] == b[key]
        if name != "random":
            assert a["prediction_rmse_pre"] == b["prediction_rmse_pre"]


def test_boundary_and_pre_update_prediction():
    path = make_path("abrupt", 6, BEFORE, AFTER, change_step=4)
    np.testing.assert_array_equal(path[:3], np.tile(BEFORE, (3, 1)))
    np.testing.assert_array_equal(path[3:], np.tile(AFTER, (3, 1)))
    rows = run_episode("omd_paper_v2_literal", FEATURES, path, 7)
    truth = 1 / (1 + np.exp(-(FEATURES @ np.array(BEFORE))))
    assert all(rows[0][f"predicted_mean_{a}"] == 0.5 for a in range(4))
    assert rows[0]["prediction_rmse_pre"] == pytest.approx(np.sqrt(np.mean((0.5 - truth) ** 2)))
