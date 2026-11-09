import copy
import json

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import t as student_t

from algorithms.hvt_ucb import HvtUCB, RidgeLinearPolicy
from envs.remaining_bandit_tasks import noise_spec, parameter_path
from scripts.remaining_experiments import (audit_cost, execute_task, features, simulate,
                                          task_id, tasks_for, verify_task)
from scripts.run_remaining_v2_2 import aggregate


@pytest.mark.parametrize("angle", [.25 * np.pi, np.pi, 2 * np.pi])
def test_rotation_has_constant_norm_and_correct_arc_length(angle):
    path = parameter_path("rotation", 1000, angle)
    r = np.linalg.norm([1, -.5, .75])
    np.testing.assert_allclose(np.linalg.norm(path, axis=1), r, atol=1e-14)
    expected = 999 * 2 * r * np.sin(angle / (2 * 999))
    assert np.linalg.norm(np.diff(path, axis=0), axis=1).sum() == pytest.approx(expected)
    np.testing.assert_allclose(path[0], [1, -.5, .75], atol=1e-14)


def test_rotation_avoids_chord_norm_collapse_and_abrupt_boundary():
    chord = parameter_path("drift", 101)
    assert np.linalg.norm(chord[50]) < 1e-14
    assert np.linalg.norm(parameter_path("rotation", 101)[50]) > 1
    abrupt = parameter_path("abrupt", 1000)
    np.testing.assert_array_equal(abrupt[499], abrupt[0])
    np.testing.assert_allclose(abrupt[500], -abrupt[0], atol=1e-14)


@pytest.mark.parametrize("name", ["student3", "student1_5"])
def test_moment_formula_matches_density_integral(name):
    spec = noise_spec(name)
    p = 1 + spec["epsilon"]
    value, _ = quad(lambda z: 2 * (spec["scale"] * z) ** p * student_t.pdf(z, spec["df"]),
                    0, np.inf, epsabs=1e-8)
    assert spec["nu"] ** p == pytest.approx(value, rel=1e-7)
    assert p < spec["df"]


def test_hvt_single_step_matches_independent_constrained_objective():
    model = HvtUCB(2, 200, .5)
    model.theta = np.array([1.14, 1.52])
    x, y = np.array([.6, .8]), 1e8  # Force clipping and a nontrivial projection.
    initial_V, initial_theta = model.V.copy(), model.theta.copy()
    beta = model.confidence_radius()
    width = np.sqrt(x @ np.linalg.solve(initial_V, x))
    sigma = max(.5, 1 / np.sqrt(200), np.sqrt(2 * beta / (model.tau0 * 2)) * width)
    w = width / (2 * sigma)
    tau = model.tau0 * np.sqrt(1 + w * w) / w
    V = initial_V + np.outer(x, x) / (4 * sigma ** 2)
    g = -np.clip((y - x @ initial_theta) / sigma, -tau, tau) * x / sigma
    objective = lambda theta: g @ theta + .5 * (theta - initial_theta) @ V @ (theta - initial_theta)
    solution = minimize(objective, initial_theta, method="SLSQP",
                        jac=lambda theta: g + V @ (theta - initial_theta),
                        constraints=[dict(type="ineq", fun=lambda theta: 4 - theta @ theta,
                                          jac=lambda theta: -2 * theta)],
                        options={"ftol": 1e-10, "maxiter": 300})
    assert solution.success
    info = model.update(x, y)
    assert info["clipped"] == 1
    assert info["projection_active"] == 1
    np.testing.assert_allclose(model.V, V, atol=1e-14)
    np.testing.assert_allclose(model.theta, solution.x, atol=1e-6)
    assert info["sigma"] == pytest.approx(sigma)
    assert info["tau"] == pytest.approx(tau)


@pytest.mark.parametrize("epsilon", [1., .25])
def test_hvt_confidence_literal_and_accumulation(epsilon):
    model = HvtUCB(3, 100, .5, epsilon)
    p = (1 - epsilon) / (2 * (1 + epsilon))
    kappa = 3 * np.log(1 + 100 / (4 * .01 * 3 * 3))
    tau0 = np.sqrt(2 * kappa) * np.log(300) ** p / np.log(16000000) ** (1 / (1 + epsilon))
    assert model.tau0 == pytest.approx(tau0)
    for n in (0, 1, 10):
        factor = 1 if epsilon == 1 else n ** p
        assert model.confidence_radius(n) == pytest.approx(107 * np.log(16000000) * tau0 * factor + np.sqrt(54))
    V = 3 * np.eye(3)
    for x, reward in zip(np.eye(3), [.3, -.2, 8.]):
        info = model.update(x, reward)
        V += np.outer(x, x) / (4 * info["sigma"] ** 2)
    np.testing.assert_allclose(model.V, V)


def test_hvt_zero_feature_invalid_update_and_horizon():
    model = HvtUCB(2, 2, .5)
    before = copy.deepcopy(model.__dict__)
    with pytest.raises(ValueError):
        model.update([2., 0], 1.)
    np.testing.assert_array_equal(model.V, before["V"])
    model.update([0., 0], 5.)
    np.testing.assert_array_equal(model.theta, [0, 0])
    assert model.last_diagnostics["tau"] is None
    model.update([1., 0], .5)
    with pytest.raises(ValueError):
        model.update([1., 0], .5)


def test_ridge_matches_batch_solution():
    rng = np.random.default_rng(8)
    X, y = rng.normal(size=(30, 3)), rng.normal(size=30)
    model = RidgeLinearPolicy(3)
    for x, reward in zip(X, y):
        model.update(x, reward)
    np.testing.assert_allclose(model.theta, np.linalg.solve(np.eye(3) + X.T @ X, X.T @ y), atol=1e-14)


def test_task_budget_and_frozen_settings():
    from collections import Counter
    full = tasks_for("full")
    assert Counter(t["stage"] for t in full) == dict(logistic=882, heavy_fixed=81, heavy_closed=324, cost=45)
    assert len({task_id(t) for t in full}) == 1332
    assert len(tasks_for("smoke")) == 71
    assert not {t["feature_seed"] for t in full} & {20261120, 20261121, 20261122}


def test_fixed_stream_pairs_and_policy_seed_determinism():
    tasks = [t for t in tasks_for("smoke") if t["stage"] == "heavy_fixed" and t["noise"] == "student3"]
    datasets = [simulate(t)[0] for t in tasks]
    for data in datasets[1:]:
        np.testing.assert_array_equal(data["actions"], datasets[0]["actions"])
        np.testing.assert_array_equal(data["rewards"], datasets[0]["rewards"])


def test_archive_resume_and_corruption_rejection(tmp_path):
    task = next(t for t in tasks_for("smoke") if t["algorithm"] == "hvt_ucb_literal")
    first = execute_task(task, tmp_path)
    directory = tmp_path / "episodes" / task_id(task)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    assert execute_task(task, tmp_path) == first
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    (directory / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        verify_task(directory, task)


def test_cost_counters_separate_history_and_restore_wrappers():
    import estimators.discounted_logistic_mle as mle
    original = mle._terms
    task = next(t for t in tasks_for("smoke") if t["stage"] == "cost" and t["algorithm"] == "scb_weight_ucb_numeric_v1")
    data, summary = simulate(task)
    assert mle._terms is original
    audit = summary["operation_audit"]
    assert audit["retained_history_bytes"] == 60 * 4 * 8
    assert audit["loss_terms_rows"] >= 60 * 61 // 2
    assert audit["mle_newton_iterations"] > 0
    assert audit["projection_solver_iterations"] == 0
