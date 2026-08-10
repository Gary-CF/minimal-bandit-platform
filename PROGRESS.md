# Bandit Project Progress

> 本文件记录项目开发进度、验收状态和 release 剩余任务。
>
> 它主要回答：
>
> **项目已经完成什么、现在处于哪里、v1.0 还差什么。**

---

## 1. 当前概况

- **项目名称**：Minimal Bandit Platform
- **当前目标版本**：v1.0
- **当前状态**：release candidate
- **当前 release workflow branch**：`release/v1.0`
- **v0.1**：教学型最小闭环
- **研究对象**：Stochastic Multi-Armed Bandit

当前环境：

```text
BernoulliBandit
GaussianBandit
```

当前算法：

```text
RandomPolicy
UCB1
UCBV
Beta–Bernoulli ThompsonSampling
```

当前已完成：

```text
core implementation
+ correctness hardening
+ regression tests
+ compatibility checks
+ experiment provenance
+ canonical benchmarks
+ benchmark figures
+ benchmark summary
+ benchmark report
+ end-to-end reproduction
+ main documentation synchronization
```

当前主要剩余：

```text
repository hygiene
+ fresh-environment audit
+ final Git audit
+ merge
+ tag
+ GitHub Release
```

版本口径：

```text
v0.1
    教学型最小闭环

v1.0
    暑期正式交付

当前
    v1.0 release candidate
```

---

## 2. v0.1 已完成里程碑

v0.1 建立了最小实验闭环：

- Bernoulli Bandit；
- Random Policy；
- UCB1；
- Beta–Bernoulli Thompson Sampling；
- environment / algorithm 解耦；
- 显式 RNG；
- JSON config；
- multi-algorithm；
- multi-seed；
- step-level CSV；
- cumulative pseudo-regret；
- plotting；
- 初版 `reproduce.sh`；
- README；
- requirements；
- `.gitignore`。

v0.1 的主要意义是：

> 证明 stochastic bandit 的最小实验闭环能够从环境、算法一路运行到 regret curve。

---

## 3. v1.0 交付标准

v1.0 的目标：

```text
完整闭环
+ 正确性验证
+ 环境–算法兼容性
+ config
+ reproducible seeds
+ standard benchmarks
+ multi-seed statistics
+ result provenance
+ automated tests
+ one-command reproduction
+ benchmark analysis
+ documentation
+ Git release
```

v1.0 不以继续增加算法数量作为完成标准。

---

## 4. v1.0 核心实现进度

### 4.1 Environment

已完成：

```text
envs/bernoulli_bandit.py
envs/gaussian_bandit.py
```

Bernoulli environment 当前具备：

- mean-vector validation；
- finite-value validation；
- probability-range validation；
- action-boundary validation；
- reward generation；
- pseudo-regret。

Gaussian environment 当前具备：

- arm mean/std configuration；
- shape validation；
- positive-standard-deviation validation；
- action-boundary validation；
- continuous reward generation；
- pseudo-regret。

---

### 4.2 Algorithms

已完成：

```text
RandomPolicy
UCB1
UCBV
ThompsonSampling
```

UCB-V 已实现：

- counts；
- reward sums；
- reward-square sums；
- empirical means；
- empirical variances；
- initialization；
- empirical-Bernstein-style bonus；
- reward-range correction。

Thompson Sampling 当前明确为：

```text
Beta–Bernoulli
```

不将当前实现错误解释为 generic continuous-reward Thompson Sampling。

---

### 4.3 Environment–algorithm compatibility

当前 compatibility：

```text
                     Bernoulli    Gaussian

Random                  ✓            ✓
UCB1                     ✓            ✗
UCB-V                    ✓            ✗
Thompson Sampling        ✓            ✗
```

当前策略：

> unsupported combination 尽早 fail，
> 而不是在 update 或 reward processing 阶段产生隐蔽错误。

---

### 4.4 Config system

当前 config 支持：

```text
experiment_name
environment
algorithms
algorithm parameters
horizon
horizons
seeds
```

支持：

- algorithm string representation；
- algorithm dictionary representation；
- single horizon；
- multiple horizons；
- experiment-specific outputs；
- normalized config snapshot。

当前配置：

```text
basic.json
config_system_smoke.json
different_horizon.json
easy_gap.json
hard_gap.json
ucb_v_smoke.json
```

其中正式 canonical benchmark：

```text
easy_gap.json
hard_gap.json
different_horizon.json
```

---

### 4.5 Experiment provenance

已完成 stale-result protection。

过去存在的风险：

```text
两个不同 config
共享同一个 experiment_name
        ↓
写入同一个 result directory
        ↓
旧 seed CSV 与新 seed CSV 混合
        ↓
aggregation silently contaminated
```

当前机制：

```text
results/<experiment_name>/
        ↓
config_snapshot.json
        ↓
作为 configuration identity
```

规则：

- 无目录：允许；
- 空目录：允许；
- snapshot 与 current config 相同：允许；
- snapshot 与 current config 不同：拒绝；
- 目录非空但没有 snapshot：拒绝。

该问题已通过人工攻击验证并转为 automated regression tests。

---

### 4.6 Runner correctness hardening

已完成多个边界修复。

#### Gaussian action boundary

原边界条件曾允许：

```text
action == num_arms
```

当前统一要求：

```text
0 <= action < num_arms
```

并补充 regression test。

#### Bernoulli validation

已补充：

- empty arm means；
- invalid dimensions；
- non-finite means；
- probability outside `[0,1]`；
- invalid actions。

#### Short-horizon UCB

过去 runner 对：

```text
horizon < num_arms
```

错误要求所有 arm 均完成初始化。

当前行为改为：

```text
expected initial actions
=
range(min(horizon, num_arms))
```

因此短 horizon experiment 可以合法结束。

---

## 5. Automated tests

正式命令：

```bash
python -m pytest -q
```

当前 automated coverage：

```text
Bernoulli environment
Gaussian environment
invalid actions
UCB1
UCB-V
Thompson Sampling
short-horizon runner
config normalization
experiment provenance
environment–algorithm compatibility
```

当前测试状态：

```text
PASS
```

长期文档不固定记录具体 test 数量。

原因：

> test suite 会继续演化；
> canonical command 和“全部通过”比具体数字更稳定。

---

## 6. Canonical benchmark status

v1.0 当前固定三组正式 benchmark：

### Easy-gap

```text
arm_means = [0.20, 0.50, 0.70]
horizon = 5000
num_seeds = 10
```

### Hard-gap

```text
arm_means = [0.45, 0.48, 0.50]
horizon = 10000
num_seeds = 10
```

### Different-horizon

```text
arm_means = [0.45, 0.48, 0.50]
horizons = [1000, 5000, 20000]
num_seeds = 10
```

正式结果与统计不再复制维护于 `PROGRESS.md`。

精确结果统一见：

```text
reports/benchmark_summary.csv
reports/benchmark_v1_0.md
```

这样可以减少：

```text
config
result
benchmark report
project-management document
```

之间的数据漂移。

---

## 7. Benchmark conclusions

当前 canonical benchmark 定性支持：

1. Random cumulative pseudo-regret 近似线性增长；
2. small gap 显著增加统计识别难度；
3. UCB 的 early exploration 来自 uncertainty bonus；
4. UCB-V 的 variance information 在有限 horizon 下可能被 correction term 抵消；
5. Thompson Sampling 的 seed variability 具有环境依赖性；
6. learning algorithms 在当前 horizons 下的 empirical regret growth 明显慢于 Random；
7. 当前有限 experiment 不足以验证严格 asymptotic regret rate；
8. 当前有限 experiment 不足以建立 universal algorithm ranking。

Random analytical sanity checks 与 experiment means 保持一致。

---

## 8. Benchmark evidence pipeline

当前 canonical evidence pipeline：

```text
configs/*.json
        ↓
run.py
        ↓
raw CSV
        ↓
plots/plot_benchmarks.py
        ↓
figures/
```

同时：

```text
raw CSV
   ↓
scripts/summarize_benchmarks.py
   ↓
reports/benchmark_summary.csv
   ↓
reports/benchmark_v1_0.md
```

这意味着 benchmark report 不再主要依赖人工抄写数字。

---

## 9. End-to-end reproduction

正式入口：

```bash
bash reproduce.sh
```

当前流程：

```text
clean results/ and figures/
        ↓
automated tests
        ↓
easy_gap
        ↓
hard_gap
        ↓
different_horizon
        ↓
figures
        ↓
benchmark summary
        ↓
output validation
```

预期 canonical run counts：

```text
easy_gap           40
hard_gap           40
different_horizon 120
```

当前状态：

```text
Reproduction completed successfully.
```

因此此前存在的：

```text
plot_regret.py
→ no CSV files found in results
```

旧 reproduction blocker 已解决。

---

## 10. Obsolete development artifacts

以下开发期入口已由正式 pipeline / tests 替代：

```text
check_ucb_v.py
check_compatibility.py
plots/plot_regret.py
```

v1.0 repository hygiene 阶段应删除这些文件。

替代关系：

```text
check_ucb_v.py
    ↓
tests/test_ucb_v.py

check_compatibility.py
    ↓
tests/test_compatibility.py

plots/plot_regret.py
    ↓
plots/plot_benchmarks.py
```

同时清理：

```text
*:Zone.Identifier
__pycache__/
.pytest_cache/
```

生成目录：

```text
results/
figures/
```

继续由 `.gitignore` 排除。

---

## 11. 当前项目结构

release candidate 的目标结构：

```text
.
├── algorithms/
│   ├── base.py
│   ├── thompson_sampling.py
│   ├── ucb1.py
│   └── ucb_v.py
├── configs/
│   ├── basic.json
│   ├── config_system_smoke.json
│   ├── different_horizon.json
│   ├── easy_gap.json
│   ├── hard_gap.json
│   └── ucb_v_smoke.json
├── envs/
│   ├── bernoulli_bandit.py
│   └── gaussian_bandit.py
├── plots/
│   └── plot_benchmarks.py
├── reports/
│   ├── benchmark_summary.csv
│   └── benchmark_v1_0.md
├── scripts/
│   └── summarize_benchmarks.py
├── tests/
│   ├── test_compatibility.py
│   ├── test_config.py
│   ├── test_environment.py
│   ├── test_gaussian_environment.py
│   ├── test_runner.py
│   ├── test_thompson.py
│   ├── test_ucb.py
│   └── test_ucb_v.py
├── CONTEXT.md
├── PROGRESS.md
├── README.md
├── reproduce.sh
├── requirements.txt
└── run.py
```

---

## 12. Documentation status

当前主要文档：

```text
README.md
CONTEXT.md
PROGRESS.md
reports/benchmark_v1_0.md
reports/benchmark_summary.csv
```

职责已经重新分离：

```text
README
    如何使用项目

CONTEXT
    项目如何工作

PROGRESS
    项目走到哪里

benchmark_summary
    机器可读实验统计

benchmark report
    实验为什么得到这些结论
```

不再在 PROGRESS 中维护重复的完整 benchmark 表。

---

## 13. 当前 Git / release workflow

当前 release 工作使用：

```text
release/v1.0
```

作为收官分支。

已形成的 release-hardening 工作包括：

```text
stale-result protection
boundary correctness fixes
regression tests
reproduction pipeline
benchmark freezing
documentation synchronization
```

最终 release 流程：

```text
release/v1.0
        ↓
final audit
        ↓
main
        ↓
annotated v1.0 tag
        ↓
GitHub
        ↓
GitHub Release
```

---

## 14. v1.0 remaining work

### 已完成

- [x] core implementation；
- [x] Bernoulli validation；
- [x] Gaussian validation；
- [x] UCB-V；
- [x] compatibility checks；
- [x] config normalization；
- [x] stale-result protection；
- [x] regression tests；
- [x] short-horizon runner fix；
- [x] canonical benchmark configs；
- [x] clean generated-output reproduction；
- [x] benchmark figures；
- [x] automatic benchmark summary；
- [x] benchmark report synchronization；
- [x] README / CONTEXT / PROGRESS synchronization；
- [x] obsolete-script cleanup plan。

### Release 前剩余

- [ ] 完成 repository hygiene；
- [ ] 确认 obsolete files 已删除；
- [ ] 确认 `.gitignore`；
- [ ] fresh virtual environment；
- [ ] `pip install -r requirements.txt`；
- [ ] fresh-env `python -m pytest -q`；
- [ ] fresh-env `bash reproduce.sh`；
- [ ] `git diff --check`；
- [ ] working tree clean；
- [ ] merge `release/v1.0`；
- [ ] create annotated `v1.0` tag；
- [ ] configure / verify GitHub remote；
- [ ] push branch / main / tag；
- [ ] create GitHub Release v1.0。

---

## 15. 当前风险

核心功能已经不再是主要风险。

当前最大的 release 风险变为：

### 15.1 文档漂移

应继续坚持：

```text
config
↓
raw evidence
↓
summary
↓
report
```

而不是人工维护多个相同数字表。

### 15.2 环境漂移

本地 `.venv` 已经验证通过并不代表 fresh install 一定通过。

因此必须完成 fresh-environment audit。

### 15.3 Release 前继续扩范围

当前禁止为了“让 v1.0 看起来更丰富”继续加入：

```text
new algorithms
new bandit families
new experiment frameworks
```

这些应进入后续版本，而不是阻塞 v1.0。

---

## 16. 当前判断

项目已经从最初的教学 Demo：

```text
single environment
+ basic algorithms
+ simple runner
```

成长为：

```text
environment abstraction
+ multiple algorithms
+ compatibility boundaries
+ config system
+ experiment provenance
+ reproducible RNG
+ runner
+ logs
+ regression tests
+ canonical benchmarks
+ figures
+ automatic statistics
+ benchmark analysis
+ end-to-end reproduction
+ release documentation
```

当前已经具备作为 v1.0 暑期正式交付的主体条件。

剩余工作不是继续扩功能，而是：

> 用 fresh environment 和最终 Git/GitHub release 流程证明：
> 这个仓库不仅在当前开发目录中能运行，
> 而且作为一个独立发布版本也能被重新安装、测试和复现。

---

## 17. 下一步

下一关只做 release audit：

```text
repository hygiene
        ↓
fresh venv
        ↓
install requirements
        ↓
pytest
        ↓
reproduce
        ↓
Git status / diff audit
```

通过后进入最终：

```text
merge
↓
tag v1.0
↓
push GitHub
↓
GitHub Release
```

当前不再增加算法。