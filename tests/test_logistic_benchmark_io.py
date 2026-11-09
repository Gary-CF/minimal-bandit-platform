import copy
import csv
import json

import numpy as np
import pytest

from scripts.logistic_benchmark import run_episode
import scripts.run_logistic_benchmark as runner


def small_config():
    c = json.loads((runner.ROOT / 'configs/v2_2/smoke.json').read_text())
    c.update(horizon=6, change_step=4, seeds=[7], algorithms=['random', 'omd_paper_v2_literal'])
    return c


def test_streaming_matches_returned_rows():
    x = np.array([[1.], [-1.]])
    path = np.array([[1.], [1.], [-1.], [-1.]])
    baseline = run_episode('omd_paper_v2_literal', x, path, 7)
    received, times = [], []
    def collect(row, timing):
        received.append(row)
        times.append(timing)
    result = run_episode('omd_paper_v2_literal', x, path, 7, on_step=collect, collect_rows=False)
    assert result == []
    assert received == baseline
    assert len(times) == len(path)
    assert all(np.isfinite(v) and v >= 0 for t in times for k, v in t.items() if k.endswith('_seconds'))


def test_saved_run_and_recomputed_corrupt_metric(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'plot_results', lambda *args: None)
    out = tmp_path / 'run'
    runner.run(small_config(), out)
    assert runner.verify(out) == 6
    with pytest.raises(FileExistsError):
        runner.run(small_config(), out)
    path = out / 'episodes/abrupt__omd_paper_v2_literal__seed7.csv'
    rows = runner.read_csv(path)
    rows[3]['instant_regret'] = '123.0'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match='hash mismatch'):
        runner.verify(out)
    # Even after the hash is recomputed, independent metric checks must fail.
    manifest_path = out / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest[path.relative_to(out).as_posix()] = runner.digest(path)
    runner.write_json(manifest_path, manifest)
    with pytest.raises(AssertionError):
        runner.verify(out)


def test_failure_retains_completed_rows(tmp_path, monkeypatch):
    real = runner.run_episode
    def fail_after_two(*args, **kwargs):
        callback = kwargs['on_step']
        def fail(row, timing):
            if row['step'] == 3:
                raise RuntimeError('injected failure')
            callback(row, timing)
        kwargs['on_step'] = fail
        return real(*args, **kwargs)
    monkeypatch.setattr(runner, 'run_episode', fail_after_two)
    out = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='injected failure'):
        runner.run(small_config(), out)
    state = json.loads((out / 'status.json').read_text())
    assert state['status'] == 'failed'
    assert state['current']['completed_steps'] == 2
    assert len(runner.read_csv(out / 'episodes/stationary__random__seed7.csv')) == 2
    assert (out / 'failure.txt').exists()
    assert not (out / 'manifest.json').exists()


@pytest.mark.parametrize('key,value', [('seeds', [0, 0]), ('horizon', 1),
                                      ('algorithms', ['not_an_algorithm']), ('gamma', 1.0),
                                      ('theta_before', [100., 0., 0.]), ('change_step', 7)])
def test_invalid_config_creates_no_run(key, value, tmp_path):
    c = copy.deepcopy(small_config())
    c[key] = value
    with pytest.raises(ValueError):
        runner.run(c, tmp_path / 'run')
    assert not (tmp_path / 'run').exists()
