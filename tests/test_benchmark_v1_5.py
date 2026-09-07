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

@pytest.mark.parametrize('damage',['missing','extra','truncated','regret','seed','snapshot'])
def test_reject_corrupt_or_incomplete_results(outputs,damage):
    root,config,path=outputs
    text=path.read_text()
    if damage=='missing':path.unlink()
    elif damage=='extra':path.with_name('extra.csv').write_text(text)
    elif damage=='truncated':path.write_text('\n'.join(text.splitlines()[:-1])+'\n')
    elif damage=='snapshot':path.with_name('config_snapshot.json').write_text('{}')
    else:
        lines=text.splitlines();row=lines[1].split(',');row[8 if damage=='regret' else 3]='99';lines[1]=','.join(row)
        path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError):verify_experiment(root,config)
