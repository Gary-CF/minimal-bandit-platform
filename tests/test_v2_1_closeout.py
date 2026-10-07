"""Time semantics, independent regret checks, replay, and output protection."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scripts.d12_logistic_stress import parameter_path, run_episode, validate_episode
from scripts.reproduce_v2_1 import fresh_output
from d11_stationary import episode


def test_abrupt_boundary_and_drift_endpoints():
    before, after = [1., -.5], [-1., .5]
    path = parameter_path("abrupt", 6, before, after, 4)
    np.testing.assert_array_equal(path[:3], np.tile(before, (3, 1)))
    np.testing.assert_array_equal(path[3:], np.tile(after, (3, 1)))
    drift = parameter_path("drift", 5, before, after, 4)
    np.testing.assert_array_equal(drift[0], before)
    np.testing.assert_array_equal(drift[-1], after)
    np.testing.assert_array_equal(drift[2], [0, 0])
    np.testing.assert_allclose(np.diff(drift, axis=0), np.tile([-.5, .25], (4, 1)))


@pytest.mark.parametrize("scenario,horizon,change", [("other", 6, 4), ("abrupt", 1, 4),
                                                      ("abrupt", 6, 1), ("abrupt", 6, 7)])
def test_invalid_time_protocol(scenario, horizon, change):
    with pytest.raises(ValueError):
        parameter_path(scenario, horizon, [1.], [-1.], change)


@pytest.mark.parametrize("algorithm", ["random", "omd_paper_v2_literal"])
def test_constant_path_matches_accepted_stationary_runner(algorithm, tmp_path):
    features = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    truth = np.array([1., -.5])
    config = {"radius": 2., "delta": .05, "beta_mode": "paper_v2_literal"}
    status = {"solver_failures": 0}
    path = parameter_path("abrupt", 20, truth, truth, 11)
    first = run_episode(tmp_path, "abrupt", algorithm, 7, features, path, config, status)
    second = run_episode(tmp_path, "abrupt", algorithm, 7, features, path, config, status)
    assert first == second
    reference, _ = episode(tmp_path, "stationary", algorithm, 7, 20, features, truth, status)
    for actual, expected in zip(first, reference):
        for key in ("step", "action", "reward", "instant_regret", "cumulative_regret"):
            assert actual[key] == expected[key]
    validate_episode(first, features, path, "abrupt", algorithm, 7)
    assert status["solver_failures"] == 0
    if algorithm != "random":
        np.testing.assert_allclose(first[0]["prediction_rmse_pre"], np.sqrt(np.mean((.5 - 1/(1+np.exp(-(features@truth))))**2)))


def test_regret_uses_current_round_and_rejects_corruption(tmp_path):
    features = np.array([[1.], [-1.]])
    path = parameter_path("abrupt", 4, [1.], [-1.], 3)
    config = {"radius": 2., "delta": .05, "beta_mode": "paper_v2_literal"}
    rows = run_episode(tmp_path, "abrupt", "random", 0, features, path, config, {"solver_failures": 0})
    validate_episode(rows, features, path, "abrupt", "random", 0)
    means = 1 / (1 + np.exp(-(features[:, 0])))
    bad = copy.deepcopy(rows)
    # Evaluate the first post-change action using stale pre-change means.
    bad[2]["instant_regret"] = float(means.max() - means[bad[2]["action"]])
    with pytest.raises(ValueError, match="regret"):
        validate_episode(bad, features, path, "abrupt", "random", 0)


def test_nonempty_output_is_never_removed(tmp_path):
    path = tmp_path / "run"
    fresh_output(path)
    marker = path / "keep.txt"
    marker.write_text("original")
    with pytest.raises(ValueError, match="absent or empty"):
        fresh_output(path)
    assert marker.read_text() == "original"
