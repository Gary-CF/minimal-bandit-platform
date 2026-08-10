# Bandit Project Context

> 本文件是项目的当前状态快照、架构说明和长期协作入口。
>
> 它主要回答：
>
> **项目是什么、当前代码如何组织、继续开发时必须保持哪些接口与实验约定。**

---

## 1. 项目身份

- **项目名称**：Minimal Bandit Platform
- **当前目标版本**：Bandit Platform v1.0
- **当前状态**：v1.0 release candidate
- **v0.1**：已完成的教学型最小闭环
- **领域**：Stochastic Multi-Armed Bandit

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
ThompsonSampling
```

其中：

```text
ThompsonSampling
=
当前 Beta–Bernoulli 实现
```

主要指标：

```text
instantaneous pseudo-regret
cumulative pseudo-regret
```

项目性质：

- 学习型科研工程；
- 可复现的最小实验平台；
- Online Learning / Bandit / RL 工程训练的基础项目；
- 强调理论假设、实现边界、实验证据和版本管理之间的一致性。

版本口径：

```text
v0.1
    教学型最小闭环

v1.0 release candidate
    当前状态

v1.0
    只有 fresh-environment audit、
    final Git audit、tag 和 GitHub Release
    全部完成后才正式成立
```

当前阶段不再增加算法。

---

## 2. v1.0 总体目标

v1.0 的目标不是堆叠算法数量，而是完成：

```text
理论
  ↓
实现
  ↓
正确性测试
  ↓
实验配置
  ↓
可复现实验
  ↓
统计汇总
  ↓
结果分析
  ↓
文档
  ↓
release
```

完成标准：

```text
正确性
+ 兼容性边界清楚
+ 配置可复现
+ seed 可复现
+ 结果 provenance 清楚
+ canonical benchmark 完整
+ 一键 reproduction
+ 文档与真实代码一致
+ clean release
```

---

## 3. 当前范围

### 3.1 已进入 v1.0

环境：

- Bernoulli Bandit；
- Gaussian Bandit。

算法：

- Random Policy；
- UCB1；
- UCB-V；
- Beta–Bernoulli Thompson Sampling。

实验基础设施：

- 统一算法接口；
- 环境与算法解耦；
- JSON config；
- config normalization；
- single horizon；
- multiple horizons；
- algorithm parameters；
- algorithm × horizon × seed 调度；
- `SeedSequence` RNG 派生；
- env RNG 与 algorithm RNG 分离；
- experiment-specific result directory；
- config snapshot；
- configuration provenance validation；
- step-level CSV；
- benchmark plotting；
- benchmark summary generation；
- environment–algorithm compatibility；
- automated regression tests；
- smoke configs；
- canonical benchmarks；
- end-to-end `reproduce.sh`；
- README；
- CONTEXT；
- PROGRESS；
- benchmark report。

### 3.2 当前明确不做

- contextual bandit；
- linear bandit；
- adversarial bandit；
- non-stationary bandit；
- neural bandit；
- distributed experiments；
- database logging；
- MLflow / Weights & Biases；
- dashboard；
- checkpoint / failure recovery；
- 大规模超参数搜索；
- 复杂 registry / plugin framework；
- 完整论文级 benchmark。

原则：

> 不用功能数量掩盖正确性问题；
> 不为形式成熟引入当前没有真实需求的抽象。

---

## 4. 事实来源优先级

当文档、历史聊天和实现不一致时，优先级为：

1. 当前 Git revision 下的可运行代码；
2. `python -m pytest -q` 的真实结果；
3. `bash reproduce.sh` 的真实结果；
4. `configs/*.json`；
5. `results/*/config_snapshot.json`；
6. 原始 CSV；
7. `reports/benchmark_summary.csv`；
8. `reports/benchmark_v1_0.md`；
9. `CONTEXT.md`；
10. `PROGRESS.md`；
11. `README.md`；
12. 历史聊天和临时笔记。

发现冲突时：

- 不默默猜测；
- 明确指出差异；
- 以当前代码和运行结果为准；
- 修复后同步相关文档；
- benchmark 数字以自动 summary 和原始 CSV 为准。

---

## 5. 当前架构

```text
JSON Configuration
configs/*.json
        ↓
Config loading and normalization
run.py
        ↓
Experiment identity / provenance validation
        ↓
Environment creation
envs/
        ↓
Environment–algorithm compatibility validation
run.py
        ↓
Algorithm creation
algorithms/
        ↓
Single experiment loop
action → reward → update → pseudo-regret
        ↓
Step-level records
        ↓
CSV + config snapshot
results/<experiment_name>/
        ↓
        ├── plots/plot_benchmarks.py
        │           ↓
        │        figures/
        │
        └── scripts/summarize_benchmarks.py
                    ↓
             reports/benchmark_summary.csv
                    ↓
             reports/benchmark_v1_0.md
```

---

## 6. 模块职责

### 6.1 `envs/`

负责：

- 保存环境真实参数；
- 验证环境输入；
- 验证 action；
- 根据 action 产生 reward；
- 提供 `best_mean`；
- 计算 pseudo-regret。

当前环境：

```text
BernoulliBandit
GaussianBandit
```

不负责：

- 算法选择；
- 算法统计量更新；
- 多实验调度；
- CSV 保存；
- 绘图；
- 将真实 arm mean 暴露给算法。

---

### 6.2 `algorithms/`

统一交互接口：

```python
action = algorithm.select_action()
algorithm.update(action, reward)
```

负责：

- 维护算法内部状态；
- 根据可观察历史选择 action；
- 根据 action 和 reward 更新状态。

不负责：

- 读取真实 arm means；
- 生成 reward；
- 计算 pseudo-regret；
- 调度实验；
- 保存日志；
- 聚合结果。

当前 config 名称：

```text
random
ucb1
ucb_v
thompson_sampling
```

当前类：

```text
RandomPolicy
UCB1
UCBV
ThompsonSampling
```

---

### 6.3 `run.py`

负责：

- 解析 `--config`；
- 加载 JSON；
- normalize config；
- 创建 environment；
- 验证 environment–algorithm compatibility；
- 创建 algorithm；
- 创建 root seed；
- 派生 env RNG 和 algorithm RNG；
- 执行单次 experiment；
- 调度 algorithm × horizon × seed；
- 检查运行时不变量；
- 验证 result-directory provenance；
- 保存 config snapshot；
- 保存 step-level CSV；
- 输出必要进度和 warning。

关键要求：

> provenance validation 必须发生在任何 snapshot 或 CSV 写入之前。

---

### 6.4 `configs/`

当前配置：

```text
basic.json
config_system_smoke.json
different_horizon.json
easy_gap.json
hard_gap.json
ucb_v_smoke.json
```

职责：

- 声明实验；
- 不实现算法；
- 固定 environment；
- 固定 algorithm configuration；
- 固定 horizon；
- 固定 seeds；
- 为 smoke、开发和 canonical benchmark 提供稳定入口。

正式 v1.0 canonical configs：

```text
easy_gap.json
hard_gap.json
different_horizon.json
```

---

### 6.5 `plots/`

当前正式绘图入口：

```text
plots/plot_benchmarks.py
```

负责：

- 读取已经生成的 CSV；
- 检查必要字段；
- 跨 seed 聚合；
- 绘制 benchmark figures；
- 保存到 `figures/`。

不负责：

- 重新运行算法；
- 修改原始实验记录；
- 维护 benchmark 数字。

旧的 `plot_regret.py` 已不再属于 v1.0 正式 pipeline。

---

### 6.6 `scripts/`

当前：

```text
scripts/summarize_benchmarks.py
```

职责：

- 从 canonical CSV 读取 final cumulative regret；
- 按 experiment / algorithm / horizon 分组；
- 计算：
  - number of seeds；
  - mean；
  - sample standard deviation；
  - SEM；
- 输出：

```text
reports/benchmark_summary.csv
```

原则：

> benchmark 数字应从原始实验结果自动计算，
> 不再依赖人工抄写。

---

### 6.7 `tests/`

当前正式测试文件：

```text
test_compatibility.py
test_config.py
test_environment.py
test_gaussian_environment.py
test_runner.py
test_thompson.py
test_ucb.py
test_ucb_v.py
```

正式测试命令：

```bash
python -m pytest -q
```

当前 automated coverage 包括：

- Bernoulli environment validation；
- Gaussian environment validation；
- invalid-action regression；
- UCB1；
- UCB-V；
- Thompson Sampling；
- short-horizon runner behavior；
- config normalization；
- result-directory provenance；
- environment–algorithm compatibility。

开发期的：

```text
check_compatibility.py
check_ucb_v.py
```

已经由正式 pytest regression tests 替代，不再作为 release 入口。

---

### 6.8 文档职责

```text
README.md
    面向第一次访问项目的人：
    安装、运行、配置、输出、复现、项目边界

CONTEXT.md
    面向未来开发者和 AI：
    架构、接口、不变量、恢复协议

PROGRESS.md
    面向项目管理：
    做完了什么、当前在哪里、还剩什么

reports/benchmark_summary.csv
    面向机器与审计：
    canonical benchmark 自动统计

reports/benchmark_v1_0.md
    面向评估者：
    实验设置、统计结果、理论解释、异常审计、limitations
```

避免在多个文档中重复维护大块 benchmark 数字。

---

## 7. 当前项目结构

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

本地生成但不应进入 Git：

```text
.venv/
__pycache__/
.pytest_cache/
results/
figures/
```

以及：

```text
*:Zone.Identifier
```

之类的操作系统元数据。

---

## 8. Environment 接口约定

实验循环依赖：

```python
reward = env.step(action)
instant_regret = env.pseudo_regret(action)
```

并可用于评估：

```python
env.arm_means
env.best_mean
```

共同要求：

```text
0 <= action < num_arms
reward finite
pseudo-regret >= 0
```

算法不得读取：

```text
env.arm_means
env.best_mean
```

用于 action selection。

### Bernoulli

要求：

```text
arm_means:
    one-dimensional
    non-empty
    finite
    within [0,1]

reward:
    0 or 1
```

### Gaussian

要求：

```text
arm_means:
    valid one-dimensional means

arm_stds:
    compatible shape
    finite
    strictly positive

reward:
    finite float
    not required to lie in [0,1]
```

---

## 9. Algorithm 接口与不变量

### 9.1 公共接口

```python
select_action() -> int
update(action, reward) -> None
```

共同要求：

- `0 <= action < num_arms`；
- 每轮只 update 一次；
- 每个独立实验创建新 algorithm object；
- 不同 seed 不共享算法状态；
- 算法只使用已观察反馈。

---

### 9.2 UCB1

核心状态：

```text
counts
reward_sums
estimated_means
```

初始化策略：

> 当仍有未尝试 arm 时，按 arm index 顺序进行 initial exploration。

当：

```text
horizon < num_arms
```

时，只要求算法执行：

```text
0, 1, ..., horizon - 1
```

不能要求它访问根本不存在的后续时间步。

应保持：

```text
counts.sum() == horizon
counts == experiment-side action_counts
reward_sums == experiment-side reward_sums
```

---

### 9.3 UCB-V

核心状态：

```text
counts
reward_sums
reward_square_sums
empirical_variances
```

当前 index 为 empirical-Bernstein-style 形式，使用：

```text
reward_range
```

作为有界奖励参数。

应保持：

```text
counts.sum() == horizon
empirical variance >= 0
counts == experiment-side action_counts
reward_sums == experiment-side reward_sums
```

当前实现的理论和代码假设均不允许无检查地把它用于 Gaussian reward。

---

### 9.4 Thompson Sampling

当前实现：

```text
Beta–Bernoulli Thompson Sampling
```

状态含义：

```text
alpha_i - 1 = successes_i
beta_i - 1 = failures_i
alpha_i + beta_i - 2 = counts_i
```

并保持：

```text
alpha_i >= 1
beta_i >= 1
```

不能直接用于连续 Gaussian reward。

---

## 10. 环境–算法兼容性

兼容性描述的是：

> 当前代码实现的可用范围。

不是：

> 整个算法家族在理论上的全部范围。

当前 matrix：

```text
                     Bernoulli    Gaussian

Random                  ✓            ✓
UCB1                     ✓            ✗
UCB-V                    ✓            ✗
Thompson Sampling        ✓            ✗
```

必须坚持：

- 不把理论可扩展性写成当前实现已经支持；
- 不让 Beta–Bernoulli TS 接受 Gaussian reward；
- 不让 bounded UCB-V 无检查运行在 Gaussian 上；
- unsupported combination 尽早抛出明确异常；
- 新增算法/environment 后同步更新 tests 和 docs。

---

## 11. Config 约定

### 11.1 核心结构

典型 normalized config：

```json
{
  "experiment_name": "easy_gap",
  "environment": {
    "name": "bernoulli",
    "arm_means": [0.2, 0.5, 0.7]
  },
  "algorithms": [
    {
      "name": "random",
      "parameters": {}
    },
    {
      "name": "ucb1",
      "parameters": {}
    },
    {
      "name": "ucb_v",
      "parameters": {
        "reward_range": 1.0
      }
    },
    {
      "name": "thompson_sampling",
      "parameters": {}
    }
  ],
  "horizons": [5000],
  "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}
```

### 11.2 兼容输入

单 horizon：

```json
"horizon": 5000
```

多 horizon：

```json
"horizons": [1000, 5000, 20000]
```

算法字符串：

```json
"algorithms": [
  "ucb1",
  "thompson_sampling"
]
```

带参数结构：

```json
"algorithms": [
  {
    "name": "ucb_v",
    "parameters": {
      "reward_range": 1.0
    }
  }
]
```

normalize 后，runner 内部应将 algorithms 视为：

```python
list[dict[str, Any]]
```

禁止再次写成：

```python
algorithms = [config["algorithms"]]
```

因为这会产生 nested list。

---

## 12. Experiment identity 与 provenance

`experiment_name` 决定：

```text
results/<experiment_name>/
```

每个 result directory 使用：

```text
config_snapshot.json
```

声明 configuration identity。

运行前检查规则：

```text
目录不存在
    → allow

目录为空
    → allow

snapshot 存在
且 snapshot == current normalized config
    → allow

snapshot 存在
但 snapshot != current config
    → reject

目录非空
但 snapshot 不存在
    → reject
```

目的：

> 防止旧 CSV 与新 experiment definition 静默混在同一个目录中。

典型错误：

```text
旧实验 seed 5–9
+
新实验 seed 0–4
↓
聚合器误以为它们来自同一配置
```

当前 provenance 机制会在写入之前阻止这种情况。

当前机制保护：

```text
configuration provenance
```

暂不记录：

```text
full code provenance
dependency lock hash
container hash
```

正式 v1.0 通过：

```text
fixed Git revision
+
clean canonical reproduction
+
annotated release tag
```

建立代码与结果之间的发布对应关系。

---

## 13. 单次实验约定

概念接口：

```python
run_single_experiment(
    seed: int,
    environment_config: dict,
    horizon: int,
    algorithm_config: dict,
) -> list[ExperimentRecord]
```

每次 experiment 必须：

1. 接收 root seed；
2. 派生 env RNG；
3. 派生 algorithm RNG；
4. 创建新 environment；
5. 验证 compatibility；
6. 创建新 algorithm；
7. 从 step 1 运行到 horizon；
8. 执行 action；
9. 生成 reward；
10. update algorithm；
11. 计算 pseudo-regret；
12. 更新 cumulative regret；
13. 保存 record；
14. 执行必要的不变量检查；
15. 返回长度为 `horizon` 的 records。

禁止跨实验复用：

- environment；
- algorithm；
- records；
- action counts；
- reward sums；
- cumulative regret。

---

## 14. 随机性与复现约定

统一使用：

```python
np.random.default_rng(...)
```

root seed 派生：

```python
seed_sequence = np.random.SeedSequence(seed)

env_seed, algorithm_seed = (
    seed_sequence.spawn(2)
)

env_rng = np.random.default_rng(
    env_seed
)

algorithm_rng = np.random.default_rng(
    algorithm_seed
)
```

理由：

- 环境噪声和算法随机性解耦；
- 算法内部增加 random draw 不直接消费环境 RNG；
- 一个 root seed 可恢复两个随机流；
- multi-seed experiment 容易复现。

复现边界：

> 相同 seed 只有在代码、config、dependency behavior 和 random-call order 相同时，才预期产生完全相同的轨迹。

---

## 15. 日志约定

当前 step-level schema：

```text
environment
algorithm
horizon
seed
step
action
reward
instant_regret
cumulative_regret
```

输出布局：

```text
results/<experiment_name>/config_snapshot.json

results/<experiment_name>/
    <environment>_<algorithm>_T<horizon>_seed<seed>.csv
```

每一行必须能够回答：

- 哪个 environment；
- 哪个 algorithm；
- 哪个 horizon；
- 哪个 seed；
- 哪个 step；
- 选择哪个 action；
- 得到什么 reward；
- instantaneous regret；
- cumulative regret。

同一份 configuration 重跑允许覆盖对应文件。

不同 configuration 不允许静默共享相同 experiment directory。

---

## 16. 正确性不变量

公共：

```text
len(records) == horizon

sum(action_counts) == horizon

0 <= action < num_arms

reward finite

instant_regret >= 0

cumulative_regret monotonically non-decreasing

sum(reward_sums) == total_reward
```

Bernoulli：

```text
reward ∈ {0,1}
```

UCB initial exploration：

```text
expected initial actions
=
range(min(horizon, num_arms))
```

日志：

```text
CSV data rows == horizon
```

实验解释：

- Random 的解析期望应与实验均值接近；
- 学习算法在 canonical 环境中应体现学习行为；
- “最优 arm 被选得最多”属于有限时间诊断，不是任意短 horizon 的普遍硬断言；
- 有限 benchmark 不能证明渐近 regret bound；
- 10 seeds 不足以建立普遍算法排名。

---

## 17. Canonical v1.0 benchmarks

### 17.1 Easy-gap

```text
arm_means = [0.20, 0.50, 0.70]
horizon = 5000
seeds = 10
```

目的：

- 检查明显较优 arm 的识别；
- 检查学习算法 action concentration；
- 检查 Random theoretical regret。

### 17.2 Hard-gap

```text
arm_means = [0.45, 0.48, 0.50]
horizon = 10000
seeds = 10
```

目的：

- 展示 small gap 的统计识别困难；
- 观察持续探索；
- 观察 seed variability。

### 17.3 Different-horizon

```text
arm_means = [0.45, 0.48, 0.50]
horizons = [1000, 5000, 20000]
seeds = 10
```

目的：

- 比较 final regret 随 \(T\) 的变化；
- 检查 Random 近线性 empirical growth；
- 比较 learning algorithms 的有限时间增长；
- 不将少数 horizon 点解释为严格 asymptotic proof。

精确数值只维护于：

```text
reports/benchmark_summary.csv
reports/benchmark_v1_0.md
```

不在 CONTEXT 和 PROGRESS 中复制完整结果表。

---

## 18. Reproduction contract

正式入口：

```bash
bash reproduce.sh
```

该脚本负责：

```text
clean generated outputs
        ↓
python -m pytest -q
        ↓
easy_gap
        ↓
hard_gap
        ↓
different_horizon
        ↓
benchmark figures
        ↓
benchmark summary
        ↓
expected-output validation
```

当前 canonical expected run counts：

```text
easy_gap          40
hard_gap          40
different_horizon 120
```

正式 pipeline 输出：

```text
figures/easy_gap_regret.png
figures/hard_gap_regret.png
figures/horizon_comparison.png
figures/action_frequency.png

reports/benchmark_summary.csv
```

成功必须以：

```text
Reproduction completed successfully.
```

结束。

---

## 19. 当前 release 状态

已完成：

- [x] Bernoulli environment；
- [x] Gaussian environment；
- [x] Random；
- [x] UCB1；
- [x] UCB-V；
- [x] Beta–Bernoulli TS；
- [x] action/input boundary hardening；
- [x] compatibility validation；
- [x] config normalization；
- [x] experiment provenance protection；
- [x] automated regression tests；
- [x] canonical benchmarks；
- [x] automatic benchmark summary；
- [x] canonical figures；
- [x] benchmark report；
- [x] end-to-end reproduction；
- [x] main documentation synchronization；
- [x] obsolete development-script cleanup。
- [x] repository hygiene 最终检查；
- [x] fresh virtual environment；
- [x] `pip install -r requirements.txt`；
- [x] fresh-env `python -m pytest -q`；
- [x] fresh-env `bash reproduce.sh`；

v1.0 tag 前仍需：


- [ ] `git diff --check`；
- [ ] working tree clean；
- [ ] merge `release/v1.0`；
- [ ] annotated `v1.0` tag；
- [ ] push GitHub；
- [ ] GitHub Release。

### Fresh-environment release audit

v1.0 release candidate 已通过独立 Git worktree +
fresh virtual environment 验证。

验证环境：

```text
Python     3.12.3
NumPy      2.5.2
Matplotlib 3.11.1
pytest     9.1.1
```

验证结果：

```text
fresh dependency installation  PASS
fresh pytest                    PASS
fresh canonical reproduction   PASS
easy-gap runs                  40
hard-gap runs                  40
different-horizon runs        120
post-reproduction git diff      clean
```

---

## 20. 增量开发规则

今后继续开发时：

1. 先明确当前目标；
2. 阅读 `CONTEXT.md` 和 `PROGRESS.md`；
3. 只打开直接相关模块；
4. 解释修改动机；
5. 小步实现；
6. 增加或更新 regression test；
7. 执行最小测试；
8. 执行相关 integration test；
9. 同步文档；
10. 创建职责单一的 Git commit。

未经明确讨论，不应：

- 大规模重命名；
- 引入重型框架；
- 删除断言而无替代测试；
- 修改 CSV schema 而不更新 plotting / summary；
- 修改 seed propagation 而不重新检查 reproducibility；
- 允许算法读取真实 arm means；
- 将单一 benchmark 写成普遍算法结论；
- 修改 canonical config 后继续沿用旧 benchmark report；
- 为追求功能数量扩大 release scope。

---

## 21. 新窗口与 AI 协作恢复协议

状态恢复至少提供：

```text
CONTEXT.md
PROGRESS.md
```

修改 algorithm 建议提供：

```text
CONTEXT.md
PROGRESS.md
algorithms/base.py
相关 algorithm file
run.py
相关 tests
相关 configs
```

修改 environment 建议提供：

```text
CONTEXT.md
PROGRESS.md
相关 env file
run.py
相关 tests
相关 configs
```

修改 experiment / benchmark 建议提供：

```text
CONTEXT.md
PROGRESS.md
run.py
相关 configs
plots/
scripts/summarize_benchmarks.py
benchmark summary
```

版本发布审计优先提供完整 project archive，并按顺序读取：

1. `CONTEXT.md`
2. `PROGRESS.md`
3. `README.md`
4. `reports/benchmark_v1_0.md`
5. `reports/benchmark_summary.csv`
6. `run.py`
7. `algorithms/`
8. `envs/`
9. `configs/`
10. `tests/`
11. `reproduce.sh`

不得只凭历史聊天猜测当前接口。

---

## 22. 当前下一步

当前禁止继续扩算法范围。

release 主线：

```text
repository hygiene
        ↓
fresh virtual environment
        ↓
install requirements
        ↓
pytest
        ↓
full reproduce
        ↓
final Git audit
        ↓
merge release/v1.0
        ↓
annotated v1.0 tag
        ↓
push GitHub
        ↓
GitHub Release v1.0
```