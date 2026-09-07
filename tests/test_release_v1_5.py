import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from run import normalize_config, preflight_config, run_single_experiment, create_algorithm
from algorithms.kl_ucb import binary_kl, kl_ucb_bound
from algorithms.gaussian_thompson_sampling import GaussianThompsonSampling
from envs.gaussian_bandit import GaussianBandit

ROOT = Path(__file__).resolve().parents[1]
BASE = dict(experiment_name='test', environment=dict(name='bernoulli', arm_means=[.2,.8]),
            algorithms=['ucb1'], horizon=10, seeds=[0])

@pytest.mark.parametrize('change', [
    {'horizon':2.9}, {'horizon':True}, {'seeds':[True]}, {'seeds':[-1]},
    {'seeds':[0,0]}, {'horizons':[10,10]}, {'horizons':[10]},
    {'algorithms':['ucb1','ucb1']},
    {'algorithms':[{'name':'epsilon_greedy','parameters':{'epslion':1}}]},
    {'algorithms':[{'name':'etc','parameters':{}}]},
    {'experiment_name':'../escape'}, {'environment':{'name':'bernoulli','arm_means':[True]}},
    {'algorithms':[{'name':'epsilon_greedy','parameters':{'epsilon':True}}]},
])
def test_invalid_batch_config(change):
    with pytest.raises(ValueError):
        normalize_config(BASE | change)

@pytest.mark.parametrize('field', ['arm_means','arm_stds'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_gaussian_environment_finite(field,value):
    args=dict(arm_means=[0.],arm_stds=[1.],rng=np.random.default_rng(0))
    args[field]=[value]
    with pytest.raises(ValueError): GaussianBandit(**args)

@pytest.mark.parametrize('field', ['prior_mean','prior_variance','noise_variance'])
@pytest.mark.parametrize('value', [np.nan,np.inf,-np.inf])
def test_gaussian_ts_finite(field,value):
    args=dict(num_arms=2,prior_mean=0.,prior_variance=1.,noise_variance=1.,rng=np.random.default_rng(0))
    args[field]=value
    with pytest.raises(ValueError): GaussianThompsonSampling(**args)

@pytest.mark.parametrize('p,q,expected', [(0,0,0),(1,1,0),(0,1,np.inf),(1,0,np.inf),(.5,0,np.inf),(.5,1,np.inf)])
def test_kl_endpoints(p,q,expected):
    assert binary_kl(p,q)==expected

@pytest.mark.parametrize('budget',[np.nan,np.inf,-np.inf])
def test_kl_budget_finite(budget):
    with pytest.raises(ValueError): kl_ucb_bound(.5,budget)

@pytest.mark.parametrize('name,params,stds,ok',[
    ('gaussian_thompson_sampling',{'noise_variance':4.},[2.,2.],True),
    ('gaussian_thompson_sampling',{'noise_variance':2.},[2.,2.],False),
    ('gaussian_thompson_sampling',{'noise_variance':1.},[1.,2.],False),
    ('gaussian_ucb',{'known_std':2.},[1.,2.],True),
    ('gaussian_ucb',{'known_std':1.},[1.,2.],False),
])
def test_noise_contract(name,params,stds,ok):
    cfg=normalize_config(BASE | {'environment':dict(name='gaussian',arm_means=[0.,1.],arm_stds=stds),
                                'algorithms':[dict(name=name,parameters=params)]})
    if ok: preflight_config(cfg)
    else:
        with pytest.raises(ValueError): preflight_config(cfg)

@pytest.mark.parametrize('algorithms',[
    [{'name':'epsilon_greedy','parameters':{'epsilon':0}}, {'name':'epsilon_greedy','parameters':{'epsilon':1}}],
    ['random','gaussian_ucb'],
    ['random',{'name':'etc','parameters':{'exploration_rounds_per_arm':0}}],
])
def test_bad_batch_writes_nothing(tmp_path,algorithms):
    path=tmp_path/'input.json';path.write_text(json.dumps(BASE|{'algorithms':algorithms}))
    result=subprocess.run([sys.executable,str(ROOT/'run.py'),'--config',str(path)],cwd=tmp_path,capture_output=True)
    assert result.returncode!=0
    assert not (tmp_path/'results').exists()

PARAMS={'random':{},'ucb1':{},'ucb_v':{},'thompson_sampling':{},'epsilon_greedy':{'epsilon':.1},
        'etc':{'exploration_rounds_per_arm':2},'moss':{},'kl_ucb':{},
        'gaussian_ucb':{'known_std':1.},'gaussian_thompson_sampling':{'noise_variance':1.}}
COMBOS=[('bernoulli',n) for n in list(PARAMS)[:8]]+[
    ('gaussian',n) for n in ['random','epsilon_greedy','etc','gaussian_ucb','gaussian_thompson_sampling']]
@pytest.mark.parametrize('env,name',COMBOS)
@pytest.mark.parametrize('horizon',[2,6,7,30])
def test_all_valid_combinations_reproduce_and_regret(env,name,horizon):
    config=dict(name=env,arm_means=[.2,.5,.7])
    if env=='gaussian': config['arm_stds']=[1.,1.,1.]
    args=dict(seed=1,environment_config=config,horizon=horizon,algorithm_config=dict(name=name,parameters=PARAMS[name]))
    rows=run_single_experiment(**args)
    assert rows==run_single_experiment(**args)
    expected=np.cumsum([.7-config['arm_means'][row['action']] for row in rows])
    np.testing.assert_allclose(expected,[row['cumulative_regret'] for row in rows])

@pytest.mark.parametrize('name',list(PARAMS)[1:])
def test_invalid_update_is_rejected_without_mutation(name):
    algo=create_algorithm(dict(name=name,parameters=PARAMS[name]),3,np.random.default_rng(0),10)
    arrays={k:v.copy() for k,v in vars(algo).items() if isinstance(v,np.ndarray)}
    for action,reward in [(0,np.nan),(True,1.),(.5,1.)]:
        with pytest.raises(ValueError): algo.update(action,reward)
    for key,value in arrays.items(): np.testing.assert_array_equal(value,getattr(algo,key))
