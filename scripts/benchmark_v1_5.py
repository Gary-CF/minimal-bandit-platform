"""Reproduce and verify the v1.5 suite without deleting existing experiments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import normalize_config, preflight_config

SUITE = ('v1_5_bernoulli_easy', 'v1_5_bernoulli_hard', 'v1_5_gaussian')
FIELDS = ['environment','algorithm','horizon','seed','step','action','reward','instant_regret','cumulative_regret']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_identity():
    paths = [ROOT/'run.py', ROOT/'requirements.txt']
    for folder, suffix in [('algorithms','*.py'),('envs','*.py'),('scripts','*.py'),('tests','*.py'),('configs','*.json')]:
        paths.extend((ROOT/folder).glob(suffix))
    lock=ROOT/'requirements-lock.txt'
    if lock.exists(): paths.append(lock)
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(paths)}


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def load_configs():
    configs=[]
    for name in SUITE:
        cfg=normalize_config(json.loads((ROOT/'configs'/f'{name}.json').read_text()))
        preflight_config(cfg)
        configs.append(cfg)
    return configs


def verify_experiment(output_root, config):
    """Reject missing, extra, truncated, inconsistent or numerically invalid CSVs."""
    folder=Path(output_root)/'results'/config['experiment_name']
    snapshot=folder/'config_snapshot.json'
    if not snapshot.is_file() or json.loads(snapshot.read_text()) != config:
        raise ValueError(f'{folder}: missing or mismatched snapshot')
    env=config['environment']['name']
    expected={f'{env}_{a["name"]}_T{T}_seed{s}.csv':(a['name'],T,s)
              for a in config['algorithms'] for T in config['horizons'] for s in config['seeds']}
    actual={p.name for p in folder.iterdir()}
    if actual != set(expected)|{'config_snapshot.json'}:
        raise ValueError(f'{folder}: unexpected/missing files: {actual.symmetric_difference(set(expected)|{"config_snapshot.json"})}')
    means=np.asarray(config['environment']['arm_means'])
    runs=[]
    for filename,(algorithm,T,seed) in sorted(expected.items()):
        actions=[]; regrets=[]; total=0.; seen=0
        with (folder/filename).open(newline='') as f:
            reader=csv.DictReader(f)
            if reader.fieldnames != FIELDS: raise ValueError(f'{filename}: wrong CSV fields')
            for row in reader:
                seen+=1
                if (row['environment'],row['algorithm'],int(row['horizon']),int(row['seed']),int(row['step'])) != (env,algorithm,T,seed,seen):
                    raise ValueError(f'{filename}: inconsistent identity or steps')
                action=int(row['action']);reward=float(row['reward'])
                instant=float(row['instant_regret']);cum=float(row['cumulative_regret'])
                if not 0<=action<len(means) or not np.all(np.isfinite([reward,instant,cum])):
                    raise ValueError(f'{filename}: invalid numeric data')
                if env=='bernoulli' and reward not in (0,1): raise ValueError(f'{filename}: non-Bernoulli reward')
                gap=float(means.max()-means[action]);total+=gap
                if not np.isclose(instant,gap,rtol=0,atol=1e-12) or not np.isclose(cum,total,rtol=1e-12,atol=1e-9):
                    raise ValueError(f'{filename}: incorrect pseudo-regret')
                actions.append(action);regrets.append(cum)
        if seen!=T: raise ValueError(f'{filename}: expected {T} rows, got {seen}')
        runs.append(dict(experiment=config['experiment_name'],algorithm=algorithm,horizon=T,seed=seed,
                         actions=np.array(actions),regrets=np.array(regrets)))
    return runs


def analyze(output_root, configs):
    os.environ.setdefault('MPLBACKEND','Agg')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output_root=Path(output_root)
    reports=output_root/'reports';figures=output_root/'figures'
    reports.mkdir(exist_ok=True);figures.mkdir(exist_ok=True)
    summary=[]; total_rows=0;total_runs=0
    for cfg in configs:
        runs=verify_experiment(output_root,cfg);total_runs+=len(runs)
        total_rows+=sum(len(r['regrets']) for r in runs)
        for T in cfg['horizons']:
            fig,(ax,af)=plt.subplots(1,2,figsize=(14,5),layout='constrained')
            algorithms=[a['name'] for a in cfg['algorithms']]
            for index,algorithm in enumerate(algorithms):
                chosen=[r for r in runs if r['algorithm']==algorithm and r['horizon']==T]
                matrix=np.stack([r['regrets'] for r in chosen]);n=len(matrix)
                mean=matrix.mean(axis=0);std=matrix.std(axis=0,ddof=1) if n>1 else np.zeros(T)
                sem=std/np.sqrt(n);steps=np.arange(1,T+1)
                line,=ax.plot(steps,mean,label=algorithm,linewidth=1.5)
                ax.fill_between(steps,mean-sem,mean+sem,color=line.get_color(),alpha=.15)
                freq=np.mean([np.bincount(r['actions'],minlength=len(cfg['environment']['arm_means']))/T for r in chosen],axis=0)
                width=.8/len(algorithms)
                af.bar(np.arange(len(freq))+(index-(len(algorithms)-1)/2)*width,freq,width,label=algorithm,color=line.get_color())
                summary.append(dict(experiment=cfg['experiment_name'],algorithm=algorithm,horizon=T,num_seeds=n,
                                    mean_final_regret=float(mean[-1]),std_final_regret=float(std[-1]),sem_final_regret=float(sem[-1])))
            ax.set(xlabel='Step',ylabel='Cumulative pseudo-regret',title=f'{cfg["experiment_name"]}, T={T}\nMean ± SEM across 10 seeds')
            ax.grid(alpha=.2);ax.legend(fontsize=7)
            af.set(xlabel='Arm',ylabel='Mean selection frequency',title='Action frequencies (all configured arms)',xticks=range(len(cfg['environment']['arm_means'])),ylim=(0,1))
            af.grid(axis='y',alpha=.2)
            fig.savefig(figures/f'{cfg["experiment_name"]}_T{T}.png',dpi=140);plt.close(fig)
    with (reports/'benchmark_summary_v1_5.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');writer.writeheader();writer.writerows(summary)
    lines=['# Bandit v1.5 正式实验结果','',
           '配置预先固定为 3 个场景、T=1000/3000、seeds=0–9；共 420 次运行、840,000 行记录。',
           'ε=0.1；ETC 每臂探索 20 次；MOSS 使用当前 T；KL-UCB c=3；UCB-V 范围为 1。',
           'Gaussian 场景均值 [0,0.2,0.5]、所有臂 std=1；Gaussian TS 先验 N(0,1)、观测方差 1。',
           '图的阴影为均值 ± SEM，描述均值估计的抽样不确定性，既非单次运行波动区间，也非 95% 置信区间。','',
           '| 场景 | 算法 | T | seeds | 最终 regret 均值 | 样本标准差 | SEM |',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["experiment"]} | {r["algorithm"]} | {r["horizon"]} | {r["num_seeds"]} | {r["mean_final_regret"]:.3f} | {r["std_final_regret"]:.3f} | {r["sem_final_regret"]:.3f} |')
    lines+=['','## Random 解析核对','', '| 场景 | T | 理论期望 | 实验均值 |','|---|---:|---:|---:|']
    for cfg in configs:
        means=np.array(cfg['environment']['arm_means'])
        for T in cfg['horizons']:
            measured=next(r['mean_final_regret'] for r in summary if r['experiment']==cfg['experiment_name'] and r['horizon']==T and r['algorithm']=='random')
            lines.append(f'| {cfg["experiment_name"]} | {T} | {T*(means.max()-means.mean()):.3f} | {measured:.3f} |')
    lines+=['','## 解释边界','',
            '这些是固定参数下的有限预算比较。ETC 的 m 未按 gap 调优，ε-Greedy 的 ε 为常数；不应将它们的表现解释为算法家族的最优性能。',
            '小 gap 场景更难识别，但每次选错的损失更小，跨场景绝对 regret 不宜直接排名。',
            'MOSS 明确获知 T；其不同预算实验不能理解为同一条轨迹截断。Gaussian TS/ UCB 使用已知噪声，这是额外建模假设。',
            '10 seeds 和两个预算不支持普遍优劣、统计显著性结论或严格渐近速率证明；没有进行超参数搜索或配对显著性检验。',
            '源码与依赖、运行时长及输出 SHA256 记录在输出根目录 suite_manifest.json。','']
    (reports/'benchmark_v1_5.md').write_text('\n'.join(lines),encoding='utf-8')
    return {'runs':total_runs,'rows':total_rows,'summary_rows':len(summary),'figures':len(list(figures.glob('*.png')))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=ROOT/'artifacts'/'v1_5')
    parser.add_argument('--verify-only',action='store_true',help='Verify manifest and all CSVs without rerunning or modifying them')
    args=parser.parse_args();out=args.output_root.resolve();configs=load_configs()
    if args.verify_only:
        manifest=json.loads((out/'suite_manifest.json').read_text())
        if manifest.get('status')!='complete' or manifest['code_sha256']!=code_identity():
            raise ValueError('Incomplete suite or source differs from manifest')
        observed={str(p.relative_to(out/'results')) for p in (out/'results').rglob('*') if p.is_file()}
        if observed!=set(manifest['result_sha256']): raise ValueError('Result file set differs from manifest')
        for rel,digest in manifest['result_sha256'].items():
            if sha256(out/'results'/rel)!=digest: raise ValueError(f'Output checksum mismatch: {rel}')
        for rel,digest in manifest['report_sha256'].items():
            if sha256(out/rel)!=digest: raise ValueError(f'Report checksum mismatch: {rel}')
        for cfg in configs: verify_experiment(out,cfg)
        print('VERIFIED',manifest['validation']);return
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'{out} is not empty; choose a new --output-root (no files were deleted)')
    out.mkdir(parents=True,exist_ok=True)
    (out/'logs').mkdir();started=time.monotonic()
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    manifest=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(),platform=platform.platform(),
                  dependencies={n:importlib.metadata.version(n) for n in ['numpy','matplotlib','pytest']},
                  git_base_revision=revision.stdout.strip() if revision.returncode==0 else None,
                  code_sha256=code_identity(),configs=configs)
    write_json(out/'suite_manifest.json',manifest)
    try:
        with (out/'logs'/'pytest.log').open('w') as log:
            subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        for index,name in enumerate(SUITE,1):
            print(f'[{index}/3] Running {name}',flush=True)
            with (out/'logs'/f'{name}.log').open('w') as log:
                subprocess.run([sys.executable,str(ROOT/'run.py'),'--config',str(ROOT/'configs'/f'{name}.json')],
                               cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
        validation=analyze(out,configs)
        if validation!={'runs':420,'rows':840000,'summary_rows':42,'figures':6}:
            raise ValueError(f'Unexpected suite size: {validation}')
        if manifest['code_sha256']!=code_identity(): raise ValueError('Source changed during suite execution')
        manifest.update(status='complete',elapsed_seconds=round(time.monotonic()-started,3),validation=validation,
                        result_sha256={str(p.relative_to(out/'results')):sha256(p) for p in sorted((out/'results').rglob('*')) if p.is_file()},
                        report_sha256={str(p.relative_to(out)):sha256(p) for folder in ['reports','figures'] for p in sorted((out/folder).glob('*')) if p.is_file()})
        write_json(out/'suite_manifest.json',manifest)
        print('COMPLETE',validation,'seconds',manifest['elapsed_seconds'])
    except Exception as exc:
        manifest.update(status='failed',error=str(exc),elapsed_seconds=round(time.monotonic()-started,3))
        write_json(out/'suite_manifest.json',manifest)
        raise


if __name__=='__main__': main()
