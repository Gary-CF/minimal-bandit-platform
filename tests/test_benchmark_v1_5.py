import csv
import json
from pathlib import Path
import pytest
from run import normalize_config, run_single_experiment, save_config_snapshot, save_records
from scripts.benchmark_v1_5 import verify_experiment

@pytest.fixture
def outputs(tmp_path):
    config=normalize_config(dict(experiment_name='tiny',environment=dict(name='bernoulli',arm_means=[0.,1.]),
                                 algorithms=['ucb1'],horizon=3,seeds=[0]))
    folder=tmp_path/'results'/'tiny';folder.mkdir(parents=True)
    save_config_snapshot(config,folder/'config_snapshot.json')
    path=folder/'bernoulli_ucb1_T3_seed0.csv'
    save_records(run_single_experiment(0,config['environment'],3,config['algorithms'][0]),path)
    return tmp_path,config,path

def test_valid_results(outputs):
    root,config,path=outputs
    assert len(verify_experiment(root,config))==1

@pytest.mark.parametrize('damage',['missing','extra','truncated','regret','seed','algorithm_id','snapshot'])
def test_reject_corrupt_or_incomplete_results(outputs,damage):
    root,config,path=outputs
    text=path.read_text()
    if damage=='missing':path.unlink()
    elif damage=='extra':path.with_name('extra.csv').write_text(text)
    elif damage=='truncated':path.write_text('\n'.join(text.splitlines()[:-1])+'\n')
    elif damage=='snapshot':path.with_name('config_snapshot.json').write_text('{}')
    else:
        with path.open(newline='') as file:
            reader = csv.DictReader(file)
            fields = reader.fieldnames
            rows = list(reader)
        field = 'cumulative_regret' if damage == 'regret' else damage
        rows[0][field] = 'wrong_instance' if damage == 'algorithm_id' else '99'
        with path.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with pytest.raises(ValueError):verify_experiment(root,config)


def test_genuine_legacy_results_remain_readable(outputs):
    # A legacy snapshot and its matching legacy CSV are accepted together.
    root, config, path = outputs
    config = json.loads(json.dumps(config))
    for entry in config['algorithms']:
        entry.pop('id')
    save_config_snapshot(config, path.with_name('config_snapshot.json'))
    with path.open(newline='') as file:
        reader = csv.DictReader(file)
        fields = [key for key in reader.fieldnames if key != 'algorithm_id']
        rows = list(reader)
    for row in rows:
        row.pop('algorithm_id')
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    runs = verify_experiment(root, config)
    assert len(runs) == 1
    assert runs[0]['algorithm_id'] == 'ucb1'


def test_new_snapshot_rejects_csv_missing_instance_column(outputs):
    root, config, path = outputs
    with path.open(newline='') as file:
        reader = csv.DictReader(file)
        fields = [key for key in reader.fieldnames if key != 'algorithm_id']
        rows = list(reader)
    for row in rows:
        row.pop('algorithm_id')
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match='wrong CSV fields'):
        verify_experiment(root, config)


def test_parameter_instances_are_verified_and_summarized_separately(tmp_path):
    from scripts.benchmark_v1_5 import analyze
    config = normalize_config(dict(
        experiment_name='instances',
        environment=dict(name='bernoulli', arm_means=[0., 1.]),
        algorithms=[
            'random',
            dict(id='eps005', name='epsilon_greedy', parameters=dict(epsilon=0.05)),
            dict(id='eps020', name='epsilon_greedy', parameters=dict(epsilon=0.20)),
        ], horizon=3, seeds=[0],
    ))
    folder = tmp_path / 'results' / config['experiment_name']
    folder.mkdir(parents=True)
    save_config_snapshot(config, folder / 'config_snapshot.json')
    for entry in config['algorithms']:
        records = run_single_experiment(0, config['environment'], 3, entry)
        save_records(records, folder / f"bernoulli_{entry['id']}_T3_seed0.csv")
    runs = verify_experiment(tmp_path, config)
    assert {run['algorithm_id'] for run in runs} == {'random', 'eps005', 'eps020'}
    assert len(runs) == 3
    analyze(tmp_path, [config])
    with (tmp_path / 'reports' / 'benchmark_summary_v1_5.csv').open(newline='') as file:
        summary = list(csv.DictReader(file))
    assert len(summary) == 3
    assert {row['algorithm_id'] for row in summary} == {'random', 'eps005', 'eps020'}
    assert all(int(row['num_seeds']) == 1 for row in summary)
    assert {row['algorithm'] for row in summary if row['algorithm_id'].startswith('eps')} == {'epsilon_greedy'}
