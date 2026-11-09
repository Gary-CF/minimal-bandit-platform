from types import SimpleNamespace
import json

import numpy as np
import pytest
from scipy.optimize import differential_evolution
from scipy.special import expit

import algorithms.scb_weight_ucb as scb
from algorithms.scb_weight_ucb import SCBWeightUCBPolicy, SCBWeightUCBAdapter, project_scb
from estimators.discounted_logistic_mle import DiscountedLogisticMLE, _terms
import scripts.run_logistic_benchmark as runner


def outside_model():
    model = DiscountedLogisticMLE(2, 0.95, 0.05)
    for x in [[1., 0.]] * 8 + [[0., 1.]] * 2 + [[.6, -.8]] * 3:
        model.update(np.array(x), 1)
    return model


def independent_objective(theta, model):
    X, w, ridge, hat = model.X, model.weights, model.ridge, model.theta
    residual = ridge * (theta - hat) + X.T @ (w * (expit(X @ theta) - expit(X @ hat)))
    p = expit(X @ theta)
    H = ridge * np.eye(len(theta)) + X.T @ ((w * p * (1 - p))[:, None] * X)
    return 0.5 * residual @ np.linalg.solve(H, residual)


def test_projection_gradient_by_finite_differences():
    model = outside_model()
    theta = np.array([.2, -.1])
    g_hat = _terms(model.theta, model.X, model.y, model.weights, model.ridge)[1]
    value, gradient = scb._projection_terms(theta, model, g_hat)
    assert value == pytest.approx(independent_objective(theta, model))
    for j in range(2):
        h = 1e-5 * np.eye(2)[j]
        expected = (independent_objective(theta + h, model)
                    - independent_objective(theta - h, model)) / 2e-5
        assert gradient[j] == pytest.approx(expected, abs=1e-7)


def test_projection_against_independent_polar_search():
    model, radius = outside_model(), 0.6
    theta, info = project_scb(model, radius)
    def polar(z):
        return independent_objective(z[0] * np.array([np.cos(z[1]), np.sin(z[1])]), model)
    reference = differential_evolution(polar, [(0, radius), (0, 2 * np.pi)],
                                       seed=7, tol=1e-10, polish=True)
    assert info["active"] == 1 and info["accepted"] >= 1 and info["kkt"] <= 1e-6
    assert np.linalg.norm(theta) <= radius + 1e-12
    assert info["objective"] == pytest.approx(reference.fun, abs=1e-7)
    radial = radius * model.theta / np.linalg.norm(model.theta)
    assert independent_objective(theta, model) < independent_objective(radial, model) - 1e-4


def test_inside_projection_and_all_solver_starts_failed(monkeypatch):
    model = outside_model()
    theta, info = project_scb(model, 100.)
    np.testing.assert_array_equal(theta, model.theta)
    assert not np.shares_memory(theta, model.theta) and info["active"] == 0
    monkeypatch.setattr(scb, "minimize", lambda fun, x, **kw: SimpleNamespace(success=False, x=x))
    with pytest.raises(RuntimeError, match="projection failed"):
        project_scb(model, 0.6)


def test_confidence_design_matrix_and_action_formula():
    policy = SCBWeightUCBPolicy(2, 50, gamma=.8)
    X = np.array([[1., 0.], [0., 1.], [.6, .8]])
    assert policy.l2 == pytest.approx(2 * np.log(50) / policy.c_mu)
    for n in range(4):
        root = np.sqrt(policy.l2 * policy.c_mu)
        beta = (root / 2 + 2 / root * (np.log(20) + 2 * np.log(2))
                + 2 / root * np.log1p(.25 * sum(.8 ** (2 * j) for j in range(n)) / (2 * root**2))
                + 2 * root)
        assert policy.confidence_radius() == pytest.approx(beta)
        V = policy.l2 * np.eye(2)
        for j, x in enumerate(X[:n]):
            V += .8 ** (n - 1 - j) * np.outer(x, x)
        np.testing.assert_allclose(policy.V, V)
        width = np.array([np.sqrt(x @ np.linalg.solve(V, x)) for x in X])
        score = expit(X @ policy.theta) + 2 * np.sqrt(5) * .25 / np.sqrt(policy.c_mu) * beta * width
        np.testing.assert_allclose(policy.scores(X), score)
        assert policy.select_action(X) == int(np.argmax(score))
        if n < 3:
            policy.update(X[n], n % 2)


def test_projected_prediction_and_transactional_failure(monkeypatch):
    policy = SCBWeightUCBPolicy(1, 20, gamma=.9, radius=.2, l2=.01)
    adapter = SCBWeightUCBAdapter(policy, np.array([[1.], [-1.]]))
    adapter.update(0, 1)
    assert policy.projection_info["active"] == 1
    assert abs(policy.estimator.theta[0]) > .2
    np.testing.assert_allclose(adapter.predict_means(), expit(adapter.features @ policy.theta))
    saved = [a.copy() for a in (policy.theta, policy.V, policy.estimator.theta, policy.estimator.X)]
    def fail(*args):
        raise RuntimeError("injected projection failure")
    monkeypatch.setattr(scb, "project_scb", fail)
    with pytest.raises(RuntimeError):
        adapter.update(1, 0)
    for a, b in zip(saved, (policy.theta, policy.V, policy.estimator.theta, policy.estimator.X)):
        np.testing.assert_array_equal(a, b)
    assert policy.estimator.completed_steps == 1


@pytest.mark.parametrize("kwargs", [dict(horizon=1), dict(gamma=1.), dict(l2=0),
                                   dict(radius=0), dict(delta=True)])
def test_invalid_configuration(kwargs):
    config = dict(dimension=2, horizon=20)
    config.update(kwargs)
    with pytest.raises(ValueError):
        SCBWeightUCBPolicy(**config)


def test_saved_scb_run_and_corrupt_projection_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "plot_results", lambda *args: None)
    config = json.loads((runner.ROOT / "configs/v2_2/scb_smoke.json").read_text())
    config.update(horizon=6, change_step=4, seeds=[7], algorithms=["scb_weight_ucb_numeric_v1"])
    out = tmp_path / "scb"
    runner.run(config, out)
    assert runner.verify(out) == 3
    path = out / "episodes/stationary__scb_weight_ucb_numeric_v1__seed7.csv"
    rows = runner.read_csv(path)
    rows[0]["scb_projected_norm_post"] = "100"
    import csv
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest[path.relative_to(out).as_posix()] = runner.digest(path)
    runner.write_json(out / "manifest.json", manifest)
    with pytest.raises(ValueError, match="SCB numerical acceptance"):
        runner.verify(out)
