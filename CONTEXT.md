# Bandit Project Context

> 本文件是项目的当前状态快照、架构说明和长期协作入口。
> 它主要回答：**项目是什么、当前代码怎样组织、继续开发时必须保持哪些约定。**

---

## 1. 项目身份

- **项目名称**：Minimal Bandit Platform
- **当前开发目标**：Bandit Platform v1.0
- **当前状态**：v1.0 development，进入结果分析、文档与最终复现阶段
- **v0.1 状态**：已于 8 月 1 日完成教学型最小闭环
- **领域**：Stochastic Multi-Armed Bandit
- **当前环境**：
  - Bernoulli Bandit
  - Gaussian Bandit
- **当前算法**：
  - Random Policy
  - UCB1
  - UCB-V
  - Thompson Sampling（当前实现为 Beta–Bernoulli）
- **主要指标**：
  - instantaneous pseudo-regret
  - cumulative pseudo-regret
- **项目性质**：
  - 学习型科研工程；
  - 可复现的最小实验平台；
  - 后续 Online Learning、Bandit、RL 与序贯决策工程的底座。

版本口径必须统一：

```text
v0.1
    8 月 1 日完成的教学型最小闭环

v1.0
    8 月 2 日—8 月 9 日的暑期正式交付目标

当前
    v1.0 development / final validation
```

当前阶段不使用 `v0.2` 作为版本名称，也不应在最终验收完成前宣称 v1.0 已正式发布。

---

## 2. v1.0 的总体目标

> 在保持代码透明、范围克制和可解释的前提下，完成“理论—实现—测试—实验—分析—复现—文档”的完整闭环。

v1.0 的完成标准不是算法数量，而是：

```text
正确性
+ 兼容性边界清楚
+ 多 seed 可复现
+ 标准实验完整
+ 曲线解释可信
+ 使用文档完整
+ 从零复现通过
```

---

## 3. 当前范围

### 3.1 已进入 v1.0 的内容

环境：

- Bernoulli Bandit；
- Gaussian Bandit。

算法：

- Random Policy；
- UCB1；
- UCB-V；
- Bernoulli Thompson Sampling。

基础设施：

- 统一算法接口；
- 环境与算法解耦；
- `SeedSequence` 派生环境 RNG 与算法 RNG；
- JSON config；
- config 归一化；
- 单 horizon 与多 horizon；
- 字符串和字典两种算法配置形式；
- algorithm × horizon × seed 调度；
- experiment-specific results directory；
- config snapshot；
- step-level CSV；
- plotting / benchmark 输出；
- compatibility checks；
- smoke configs；
- 最小自动测试；
- `reproduce.sh`；
- README、CONTEXT、PROGRESS、benchmark report。

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
- checkpoint 与失败恢复；
- 大规模超参数搜索；
- 复杂注册器或插件系统；
- 完整论文级 benchmark。

原则：

> 不用功能数量掩盖正确性问题；不为形式成熟引入当前没有真实需求的抽象。

---

## 4. 事实来源优先级

当文档、聊天记录和实现不一致时，按以下顺序判断：

1. 当前可运行代码；
2. `python -m pytest -q` 与 smoke test 的真实输出；
3. `configs/*.json` 的真实内容；
4. CSV、figure 和 config snapshot；
5. `CONTEXT.md`；
6. `PROGRESS.md`；
7. `README.md`；
8. 历史聊天和临时笔记。

发现冲突时：

- 不默默猜测；
- 明确指出差异；
- 以代码和运行结果为准；
- 随后同步三份主文档和实验报告。

---

## 5. 当前架构

```text
JSON Configuration
configs/*.json
        ↓
Config loading and normalization
run.py
        ↓
Environment creation ───── Compatibility validation
envs/                       check_compatibility.py / runner checks
        ↕
Algorithm creation
algorithms/
        ↓
Single experiment loop
action → reward → update → pseudo-regret
        ↓
Step-level records and config snapshot
results/<experiment_name>/
        ↓
Aggregation and plotting
plots/
        ↓
Figures and human analysis
figures/ + reports/benchmark_v1_0.md
```

---

## 6. 模块职责

### 6.1 `envs/`

负责：

- 保存环境真实参数；
- 验证 action；
- 根据 action 生成 reward；
- 提供最佳均值；
- 计算 pseudo-regret。

当前环境：

```text
BernoulliBandit
GaussianBandit
```

不负责：

- 算法选择；
- 算法统计量更新；
- CSV 保存；
- 多实验调度；
- 绘图；
- 为算法暴露真实 arm mean。

### 6.2 `algorithms/`

共同接口：

```python
action = algorithm.select_action()
algorithm.update(action, reward)
```

负责：

- 维护算法内部状态；
- 根据可观察历史选择动作；
- 根据 action 和 reward 更新。

不负责：

- 读取环境真实均值；
- 生成 reward；
- 计算真实 pseudo-regret；
- 调度实验；
- 保存 CSV；
- 聚合绘图。

当前配置名称：

```text
random
ucb1
ucb_v
thompson_sampling
```

当前类名：

```text
RandomPolicy
UCB1
UCBV
ThompsonSampling
```

### 6.3 `run.py`

负责：

- 解析 `--config`；
- 加载 JSON；
- 归一化旧式和新式 config；
- 创建环境与算法；
- 执行兼容性检查；
- 创建 root seed；
- 派生环境 RNG 与算法 RNG；
- 执行单次实验；
- 调度 algorithm × horizon × seed；
- 检查运行时不变量；
- 保存 config snapshot；
- 保存逐 step CSV；
- 输出必要进度与 warning。

### 6.4 `configs/`

当前可见配置：

```text
basic.json
config_system_smoke.json
different_horizon.json
easy_gap.json
hard_gap.json
ucb_v_smoke.json
```

职责：

- 声明实验，不实现实验；
- 固定 environment、algorithm、horizon、seed；
- 为 smoke、标准实验和多 horizon 实验提供可重复入口。

### 6.5 `plots/`

负责：

- 读取已经生成的实验结果；
- 检查字段和 step 对齐；
- 跨 seed 聚合；
- 绘制 regret 或动作频率；
- 保存 figure。

不负责重新实现算法或修改实验记录。

### 6.6 `tests/`

当前测试文件：

```text
test_environment.py
test_ucb.py
test_thompson.py
```

正式测试命令：

```bash
python -m pytest -q
```

当前状态：

- 使用该命令时，所有已收集测试通过；
- 直接运行 `pytest -q` 曾在 collection 阶段因项目根目录未进入 import path 而失败；
- 该问题属于启动方式和导入路径，不属于算法断言失败；
- 当前仓库以 `python -m pytest -q` 为 canonical command。

### 6.7 辅助检查脚本

```text
check_compatibility.py
check_ucb_v.py
```

作用：

- 快速验证当前环境–算法组合边界；
- 快速检查 UCB-V 的关键行为；
- 用于开发期诊断，不替代正式测试和最终复现。

### 6.8 文档

```text
README.md
    面向使用者：安装、运行、配置、输出、边界

CONTEXT.md
    面向未来开发者和 AI：架构、接口、约束、恢复协议

PROGRESS.md
    面向项目管理：完成了什么、验收到哪里、下一步是什么

reports/benchmark_v1_0.md
    面向评估者：实验设计、结果、异常排查与理论解释
```

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
├── figures/
├── plots/
├── reports/
│   └── benchmark_v1_0.md
├── results/
├── tests/
│   ├── test_environment.py
│   ├── test_thompson.py
│   └── test_ucb.py
├── check_compatibility.py
├── check_ucb_v.py
├── CONTEXT.md
├── PROGRESS.md
├── README.md
├── reproduce.sh
├── requirements.txt
└── run.py
```

本地但不应进入 Git 的内容包括：

```text
.venv/
__pycache__/
.pytest_cache/
results/
figures/
```

代表性 figure 若需要展示，应复制到受版本控制的 `assets/` 或 `reports/assets/`，而不是强行提交整个生成目录。

---

## 8. Environment 接口约定

实验循环依赖环境提供：

```python
reward = env.step(action)
instant_regret = env.pseudo_regret(action)
```

并使用：

```python
env.arm_means
env.best_mean
```

共同性质：

- action 合法；
- reward 有限；
- pseudo-regret 非负；
- 环境可读取真实均值用于反馈与评估；
- 算法不得读取真实均值。

Bernoulli 特有：

```text
reward ∈ {0, 1}
arm_means ∈ [0, 1]
```

Gaussian 特有：

```text
reward 为有限浮点数
每个臂具有 arm_mean 与 arm_std
奖励不保证位于 [0,1]
```

---

## 9. Algorithm 接口与不变量

### 9.1 公共约定

```python
select_action() -> int
update(action, reward) -> None
```

必须满足：

- `0 <= action < num_arms`；
- 每轮只更新一次；
- 新实验创建新算法对象；
- 不同 seed 不共享状态；
- 算法只使用已观测反馈。

### 9.2 UCB1

应保持：

```text
初始化阶段覆盖所有臂
counts.sum() == horizon
counts 与实验端 action_counts 一致
reward_sums 与实验端一致
estimated_means 与经验均值一致
```

### 9.3 UCB-V

应保持：

```text
初始化阶段覆盖所有臂
维护 counts
维护 reward_sums
维护 reward_square_sums 或等价方差统计量
经验方差非负
counts 与实验端 action_counts 一致
reward_sums 与实验端一致
```

当前实现按有界奖励公式使用 `reward_range`，不能因为算法家族理论上可扩展，就默认当前类支持 Gaussian。

### 9.4 Thompson Sampling

当前为 Beta–Bernoulli：

```text
alpha_i - 1 = successes_i
beta_i - 1 = failures_i
alpha_i + beta_i - 2 = counts_i
alpha_i >= 1
beta_i >= 1
```

不能直接用于 Gaussian reward。

---

## 10. 兼容性约定

兼容性检查描述的是当前代码实现。

必须坚持：

- 不把理论上可推广解释成当前代码已经支持；
- 不让 Beta–Bernoulli TS 接收连续 Gaussian reward；
- 不让依赖 `[0,1]` 有界性的 UCB-V 无检查地运行于 Gaussian；
- 不兼容组合应尽早抛出清楚的 `ValueError`；
- 新增环境或参数后同步更新检查脚本、测试和文档。

Bernoulli 标准实验使用：

```text
Random
UCB1
UCB-V
Thompson Sampling
```

Gaussian 实验只使用通过当前 compatibility check 的算法配置。

---

## 11. 配置约定

### 11.1 归一化后的核心结构

```json
{
  "experiment_name": "easy_gap",
  "environment": {
    "name": "bernoulli",
    "arm_means": [0.02, 0.05, 0.95]
  },
  "algorithms": [
    {"name": "random", "parameters": {}},
    {"name": "ucb1", "parameters": {}},
    {"name": "ucb_v", "parameters": {"reward_range": 1.0}},
    {"name": "thompson_sampling", "parameters": {}}
  ],
  "horizons": [5000],
  "seeds": [0, 1, 2, 3, 4]
}
```

### 11.2 兼容输入

单 horizon：

```json
"horizon": 5000
```

多 horizon：

```json
"horizons": [500, 1000, 2000, 5000, 10000]
```

旧式算法字符串：

```json
"algorithms": ["ucb1", "thompson_sampling"]
```

带参数字典：

```json
"algorithms": [
  {"name": "ucb_v", "parameters": {"reward_range": 1.0}}
]
```

归一化后，runner 内部应始终把 `algorithms` 当作：

```python
list[dict[str, Any]]
```

禁止再次写成：

```python
algorithms = [config["algorithms"]]
```

因为这会生成嵌套列表，并在 `algorithm_config["name"]` 处触发：

```text
TypeError: list indices must be integers or slices, not str
```

正确形态为：

```python
algorithms = list(config["algorithms"])
```

该问题已修复，并已通过 `config_system_smoke.json` 完成验证：8 个 algorithm × horizon × seed 实验全部运行完成。

---

## 12. 单次实验约定

概念接口：

```python
run_single_experiment(
    seed: int,
    environment_config: dict,
    horizon: int,
    algorithm_config: dict,
) -> list[ExperimentRecord]
```

每次实验必须：

1. 接收 root seed；
2. 派生 env RNG 与 algorithm RNG；
3. 创建新环境；
4. 验证兼容性；
5. 创建新算法；
6. 从 step 1 运行至 horizon；
7. 执行 action → reward → update → regret；
8. 保存逐 step record；
9. 执行公共和算法专属检查；
10. 返回长度为 horizon 的 records。

禁止跨实验复用：

- env；
- algorithm；
- records；
- counts；
- reward sums；
- cumulative regret。

---

## 13. 随机性与复现约定

统一使用：

```python
np.random.default_rng(...)
```

root seed 派生：

```python
seed_sequence = np.random.SeedSequence(seed)
env_seed, algorithm_seed = seed_sequence.spawn(2)

env_rng = np.random.default_rng(env_seed)
algorithm_rng = np.random.default_rng(algorithm_seed)
```

理由：

- 分离环境噪声和算法内部随机性；
- 算法增加一次内部采样不会直接消费环境 RNG；
- 一个 root seed 可恢复子随机状态；
- 便于复现实验。

复现边界：

> 相同 seed 只有在代码、配置、依赖版本和随机调用顺序相同时，才预期生成相同 CSV。

---

## 14. 日志约定

当前字段：

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

建议输出布局：

```text
results/<experiment_name>/config_snapshot.json
results/<experiment_name>/<environment>_<algorithm>_T<horizon>_seed<seed>.csv
```

每行必须回答：

- 在哪个环境；
- 使用哪个算法；
- horizon 是多少；
- 哪个 seed；
- 当前 step；
- 选择哪个 action；
- 获得什么 reward；
- 当前 instant regret；
- 当前 cumulative regret。

同一路径重跑时默认覆盖，不追加重复记录。

---

## 15. 正确性不变量

公共：

```text
len(records) == horizon
sum(action_counts) == horizon
0 <= action < num_arms
reward 为有限值
instant_regret >= 0
cumulative_regret 单调不减
sum(reward_sums) == total_reward
```

Bernoulli：

```text
reward ∈ {0,1}
```

日志：

```text
CSV 数据行数 == horizon
相同 seed 的重复运行可复现
不同 seed 通常产生不同轨迹
```

实验解释：

- Random 的解析期望应与实验均值接近；
- 学习算法应在合理配置下显著优于 Random；
- “最优臂被选得最多”是有限实验诊断，不是每个短 smoke run 都必须满足的硬断言；
- 有限实验不能证明渐近 regret bound。

---

## 16. 标准实验

### Easy-gap

目的：

- 验证算法能快速识别明显最优臂；
- 检查动作选择比例；
- 检查 Random regret 解析值。

### Hard-gap

目的：

- 展示小 gap 带来的识别困难；
- 观察多 seed 波动；
- 比较 UCB1、UCB-V 与 TS 的有限时间行为。

### Different-horizon

目的：

- 比较最终 regret 随 \(T\) 的增长；
- 区分 Random 的近线性趋势与学习算法的次线性趋势；
- 不把少量 horizon 点写成严格渐近证明。

### Smoke configs

目的：

- 快速检查 config schema；
- 快速检查多算法调度；
- 快速检查 UCB-V 的初始化与更新；
- 不承担最终 benchmark 结论。

---

## 17. 当前测试状态

已确认：

```bash
python -m pytest -q
```

能够完成测试收集并通过当前测试。

曾出现：

```bash
pytest -q
```

在 collection 阶段报：

```text
ModuleNotFoundError: No module named 'envs'
ModuleNotFoundError: No module named 'algorithms'
```

判断：

- 不是测试断言失败；
- 不是算法实现失败；
- 是 launcher 与 `sys.path` 的差异；
- README 和 release instructions 统一写 `python -m pytest -q`。

不要在每个测试文件里临时 `sys.path.append(...)` 来掩盖项目运行约定。

---

## 18. 当前已知未完成项

在创建 v1.0 tag 前仍需：

- [x] 用修正后的 runner 通过 `config_system_smoke.json`；
- [x] 再跑一次 `ucb_v_smoke.json`；
- [x] `python -m pytest -q` 通过；
- [ ] 确认 `pytest` 已写入 `requirements.txt`；
- [ ] 完成 README 与 benchmark report；
- [ ] 从空 `results/`、`figures/` 执行 `bash reproduce.sh`；
- [ ] 在干净虚拟环境安装和复现；
- [ ] 清理临时文件与误生成文件；
- [ ] `git diff --check` 通过；
- [ ] 创建清晰 Git commits；
- [ ] 工作区 clean 后创建 annotated `v1.0` tag。

---

## 19. 增量开发规则

每一步：

1. 先明确当前目标；
2. 读取 `CONTEXT.md` 和 `PROGRESS.md`；
3. 只打开直接相关代码；
4. 解释修改动机；
5. 用户亲手完成小步修改；
6. 运行最小测试；
7. 达到验收标准后再进入下一步；
8. 同步文档；
9. 创建职责单一的 Git commit。

未经明确讨论，不应：

- 大规模重命名；
- 引入重型框架；
- 删除断言而无替代测试；
- 修改 CSV schema 而不更新 plotting；
- 修改 seed 传播而不做复现；
- 让算法访问真实均值；
- 将一次实验写成普遍算法结论；
- 为追求功能数量扩大 v1.0 范围。

---

## 20. 新窗口与 AI 协作恢复协议

规划和状态恢复至少提供：

```text
CONTEXT.md
PROGRESS.md
```

修改算法至少提供：

```text
CONTEXT.md
PROGRESS.md
algorithms/base.py
相关算法文件
run.py
相关 config
相关 tests
```

修改环境至少提供：

```text
CONTEXT.md
PROGRESS.md
相关 env 文件
run.py
compatibility checks
相关 config
相关 tests
```

修改实验或绘图至少提供：

```text
CONTEXT.md
PROGRESS.md
run.py
plots/
相关 configs
一小组代表性 CSV 或 figure
```

版本发布优先提供完整项目压缩包，并按以下顺序读取：

1. `CONTEXT.md`
2. `PROGRESS.md`
3. `README.md`
4. `reports/benchmark_v1_0.md`
5. `run.py`
6. algorithms / envs
7. configs / tests / reproduce script

不得只凭历史聊天猜测当前接口。

---

## 21. 当前下一步

当前不再增加算法。

下一步顺序：

```text
修复后的 config smoke
        ↓
UCB-V smoke
        ↓
文档与实验报告
        ↓
完整 reproduce
        ↓
干净环境复现
        ↓
Git commit
        ↓
v1.0 release/tag
```
