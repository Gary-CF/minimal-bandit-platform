# Bandit Project Context

> 本文件是项目的当前状态快照、架构说明和长期协作入口。  
> 它主要回答：**项目是什么，为什么这样设计，当前接口怎样工作，继续开发时必须保持哪些约定。**

---

## 1. 项目身份

- **项目名称**：Minimal Bandit Platform
- **当前版本**：v0.1
- **领域**：Stochastic Multi-Armed Bandit
- **当前环境类型**：Bernoulli Bandit
- **当前已实现算法**：
  - Random Policy
  - UCB1
  - Thompson Sampling
- **当前主要指标**：
  - instantaneous pseudo-regret
  - cumulative pseudo-regret
- **项目性质**：
  - 学习型科研工程；
  - 最小可复现实验平台；
  - 后续在线学习、强化学习和序贯决策工程的起点。

---

## 2. 项目总体目标

项目的当前目标是：

> 构建一个结构清晰、随机性可控、实验可复现、结果可记录、算法可比较、后续可扩展的最小 Bandit 平台。

本项目不仅用于实现算法，也用于训练以下工程能力：

- 将算法与环境解耦；
- 设计统一接口；
- 显式管理随机性；
- 使用配置定义实验；
- 保存结构化日志；
- 聚合多 seed 结果；
- 分离实验运行与结果分析；
- 编写一键复现脚本；
- 使用 Git 管理版本；
- 为长期人机协作维护显式上下文。

---

## 3. 当前范围

### v0.1 包含

- Bernoulli Bandit；
- Random Policy；
- UCB1；
- Thompson Sampling；
- JSON configuration；
- multi-algorithm scheduling；
- multi-seed experiments；
- step-level CSV logging；
- mean cumulative regret aggregation；
- regret visualization；
- one-command reproduction；
- README、requirements 与 Git ignore。

### v0.1 不包含

- contextual bandit；
- non-stationary bandit；
- linear bandit；
- neural bandit；
- distributed experiments；
- database logging；
- MLflow 或 Weights & Biases；
- 完整 benchmark suite；
- 并行调度；
- checkpoint 和失败恢复；
- 完整自动化测试体系。

原则：

> 先保持最小、透明和可理解，再根据真实需求增加抽象与工具。

---

## 4. 事实来源优先级

当文档、聊天记录和实现不一致时，按以下顺序判断：

1. 当前可运行代码；
2. 自动测试和实际复现结果；
3. `configs/*.json` 中的真实配置；
4. `CONTEXT.md` 中的当前接口约定；
5. `PROGRESS.md` 中的进度记录；
6. `README.md` 中的对外说明；
7. 历史聊天或临时笔记。

如果发现冲突：

- 不应默默猜测；
- 应明确指出差异；
- 以代码和运行结果为准；
- 随后同步更新文档。

---

## 5. 当前架构

```text
Configuration
configs/basic.json
        ↓
Experiment orchestration
run.py
        ↓
Algorithm layer
algorithms/
        ↔
Environment layer
envs/
        ↓
Step-level records
results/*.csv
        ↓
Analysis layer
plots/plot_regret.py
        ↓
Visualization
figures/regret_curve.png
```

---

## 6. 模块职责

### 6.1 `envs/`

负责：

- 定义奖励环境；
- 保存真实环境参数；
- 根据 action 生成 reward；
- 提供评估所需的真实信息；
- 计算 pseudo-regret。

不负责：

- 算法决策；
- 算法内部更新；
- CSV 保存；
- 多 seed 调度；
- 绘图；
- README 或实验报告。

---

### 6.2 `algorithms/`

负责：

- 保存算法内部状态；
- 根据当前状态选择 action；
- 根据 action 和 reward 更新状态。

不负责：

- 读取真实 `arm_means`；
- 生成环境 reward；
- 计算真实 pseudo-regret；
- 保存 CSV；
- 调度多个实验；
- 聚合或绘图。

---

### 6.3 `run.py`

负责：

- 解析命令行参数；
- 读取 JSON config；
- 转换并提取实验参数；
- 创建根 seed；
- 派生环境 RNG 和算法 RNG；
- 创建环境与算法；
- 运行一个单次实验；
- 调度 algorithm × seed；
- 执行必要的一致性检查；
- 返回逐 step records；
- 保存 CSV；
- 输出必要进度信息。

---

### 6.4 `plots/plot_regret.py`

负责：

- 读取已经生成的 CSV；
- 验证日志结构；
- 按 algorithm 分组；
- 检查多 seed step 对齐；
- 组成 regret matrix；
- 跨 seed 求平均；
- 绘制和保存结果。

它不负责重新运行 Bandit 实验。

---

### 6.5 `reproduce.sh`

负责：

- 删除旧的可再生输出；
- 运行默认配置；
- 生成 CSV；
- 调用绘图脚本；
- 完成一键复现。

---

### 6.6 文档

```text
README.md
    面向项目使用者

CONTEXT.md
    面向未来开发者与 AI，保存当前架构和接口约定

PROGRESS.md
    保存项目演进、验收状态和下一步计划
```

三者不应重复承担同一职责。

---

## 7. 当前项目结构

```text
.
├── algorithms/
│   ├── base.py
│   ├── thompson_sampling.py
│   └── ucb1.py
├── configs/
│   └── basic.json
├── envs/
│   └── bernoulli_bandit.py
├── plots/
│   └── plot_regret.py
├── .gitignore
├── CONTEXT.md
├── PROGRESS.md
├── README.md
├── reproduce.sh
├── requirements.txt
└── run.py
```

可再生输出：

```text
results/
figures/
```

本地环境和缓存：

```text
.venv/
venv/
__pycache__/
```

---

## 8. Algorithm 接口约定

当前实验循环要求算法对象支持：

```python
action = algorithm.select_action()
algorithm.update(action, reward)
```

预期行为：

- `select_action()` 返回一个合法臂编号；
- `0 <= action < num_arms`；
- `update(action, reward)` 只使用可观察反馈；
- 算法不读取环境真实均值；
- 每次新实验创建新的算法对象；
- 不同实验之间不共享算法状态。

当前配置名称约定：

```text
random
ucb1
thompson_sampling
```

当前 Python 类名：

```text
RandomPolicy
UCB1
ThompsonSampling
```

配置名称、类名和文件名可以采用不同命名风格，例如：

```text
配置名称：thompson_sampling
类名：ThompsonSampling
文件名：thompson_sampling.py
```

新增算法时，需要同步检查：

- 新算法实现；
- 算法创建或注册逻辑；
- config；
- 测试；
- README；
- CONTEXT；
- PROGRESS。

---

## 9. Environment 接口约定

当前实验循环依赖环境提供类似以下接口：

```python
reward = env.step(action)
instant_regret = env.pseudo_regret(action)
```

并会使用环境属性：

```python
env.arm_means
env.best_mean
```

环境预期性质：

- action 必须合法；
- Bernoulli reward 只能是 `0` 或 `1`；
- pseudo-regret 非负；
- 环境真实参数只用于反馈和评估；
- 算法不应访问真实 arm mean。

---

## 10. 单次实验约定

当前单次实验概念接口为：

```python
run_single_experiment(
    seed: int,
    arm_means: list[float],
    horizon: int,
    algorithm_name: str,
) -> list[ExperimentRecord]
```

具体函数签名若与本地代码略有差异，以本地代码为准。

每次单次实验必须：

1. 接收一个根 seed；
2. 派生环境 RNG 和算法 RNG；
3. 创建新的环境对象；
4. 创建新的算法对象；
5. 从 step 1 运行到 horizon；
6. 每一步保存 record；
7. 执行必要的一致性检查；
8. 返回长度为 horizon 的 records。

禁止跨 seed 复用：

- `env`；
- `algorithm`；
- `records`；
- `action_counts`；
- `reward_sums`；
- cumulative regret 状态。

---

## 11. 随机性约定

当前统一使用：

```python
np.random.default_rng(...)
```

不应在项目内部随意混用全局随机 API：

```python
np.random.seed(...)
np.random.random(...)
```

当前随机状态派生方式：

```python
seed_sequence = np.random.SeedSequence(seed)
env_seed, algorithm_seed = seed_sequence.spawn(2)

env_rng = np.random.default_rng(env_seed)
algorithm_rng = np.random.default_rng(algorithm_seed)
```

设计理由：

- 环境随机性和算法随机性分离；
- 算法内部增加随机采样时，不直接消耗环境 RNG；
- 根 seed 足以恢复两个子随机状态；
- CSV 只需记录根 seed。

复现边界：

> 相同 seed 只有在代码、配置、依赖和随机数调用顺序一致时，才预期产生相同结果。

---

## 12. 配置约定

当前运行命令：

```bash
python run.py --config configs/basic.json
```

当前必需字段：

```json
{
  "arm_means": [0.3, 0.5, 0.7],
  "horizon": 5000,
  "algorithms": [
    "ucb1",
    "thompson_sampling"
  ],
  "seeds": [0, 1, 2, 3, 4]
}
```

字段含义：

- `arm_means`
  - Bernoulli arms 的真实奖励概率；
- `horizon`
  - 每个单次实验的交互步数；
- `algorithms`
  - 要运行的算法名称；
- `seeds`
  - 独立重复实验使用的根 seed。

默认实验数：

```text
len(algorithms) × len(seeds)
= 2 × 5
= 10
```

若未来修改 config schema，需要同步更新：

- JSON 文件；
- 配置读取逻辑；
- README；
- reproduce script；
- CONTEXT；
- PROGRESS；
- 自动测试。

---

## 13. 日志约定

一条实验记录概念上为：

```python
ExperimentRecord = dict[
    str,
    str | int | float,
]
```

当前字段顺序：

```text
algorithm
seed
step
action
reward
instant_regret
cumulative_regret
```

每行应能够独立回答：

- 哪个算法；
- 哪个 seed；
- 第几个 step；
- 选择哪个 action；
- 得到什么 reward；
- 当前即时 regret；
- 当前累计 regret。

当前文件命名：

```text
results/{algorithm_name}_seed_{seed}.csv
```

例如：

```text
results/ucb1_seed_0.csv
results/thompson_sampling_seed_4.csv
```

同一 algorithm 和 seed 重跑时，默认覆盖旧文件，不追加重复内容。

正式实验数据进入 CSV，终端 `print` 只用于进度与必要提示。

---

## 14. 绘图与聚合约定

当前绘图脚本假设：

- 每个 CSV 只包含一次实验；
- 同一个 CSV 内 algorithm 不变化；
- 同一个 CSV 内 seed 不变化；
- step 从 1 连续增长；
- cumulative regret 单调不减；
- 同一算法不同 seed 的 step 完全对齐。

对于一个算法，聚合矩阵形状为：

```text
(number_of_seeds, horizon)
```

当前平均方式：

```python
mean_regrets = np.mean(
    regret_matrix,
    axis=0,
)
```

解释：

- 跨 seed 求平均；
- 保留 step 维度。

当前输出：

```text
figures/regret_curve.png
```

当前结果仅代表特定配置下的经验比较，不应表述为算法的一般优劣证明。

---

## 15. 正确性不变量

任何后续重构都应保持以下不变量。

### 15.1 公共实验不变量

```text
len(records) == horizon
sum(action_counts) == horizon
0 <= action < num_arms
reward ∈ {0, 1}
instant_regret >= 0
cumulative_regret 单调不减
sum(reward_sums) == total_reward
```

### 15.2 UCB1 不变量

```text
初始化阶段依次选择所有臂
algorithm.counts 与 action_counts 一致
algorithm.reward_sums 与 reward_sums 一致
estimated_means 与经验均值一致
```

### 15.3 Thompson Sampling 不变量

```text
alpha - 1 = successes
beta - 1 = failures
alpha + beta - 2 = action_counts
alpha >= 1
beta >= 1
```

### 15.4 日志与复现不变量

```text
每个 CSV 行数 = horizon + 1
相同 seed 重跑应得到相同 CSV
不同 seed 通常产生不同 CSV
```

---

## 16. 当前运行工作流

运行默认实验：

```bash
python run.py --config configs/basic.json
```

单独绘图：

```bash
python plots/plot_regret.py
```

一键复现：

```bash
bash reproduce.sh
```

最小验收：

```bash
rm -rf results figures
bash reproduce.sh

find results -maxdepth 1 -name "*.csv" | wc -l
test -f figures/regret_curve.png
```

默认预期：

```text
10 个 CSV
figures/regret_curve.png 存在
```

---

## 17. 当前设计决策

### 17.1 使用 CSV，而不是数据库

理由：

- 当前数据规模较小；
- 文件结构透明；
- 人可以直接查看；
- Python、R、MATLAB 等工具都能读取；
- 有利于理解运行与分析解耦。

当前不引入数据库、MLflow 或 W&B。

### 17.2 保存逐 step 数据

理由：

- 可以画完整 regret curve；
- 可以分析 action、reward 和即时 regret；
- 后续增加分析指标时无需重新运行实验。

### 17.3 生成目录不提交 Git

当前：

```text
results/
figures/
```

是可再生输出，因此由 `.gitignore` 忽略。

未来若 GitHub README 需要展示图像，建议复制代表性图片到：

```text
assets/
```

并单独纳入版本控制。

### 17.4 终端输出保持简洁

终端主要显示：

- 总实验数量；
- 当前 algorithm 和 seed；
- 保存路径；
- warning；
- reproduce 阶段。

算法内部统计与正式数据进入 CSV。

### 17.5 暂不引入重量级框架

v0.1 不使用：

- pandas；
- Hydra；
- MLflow；
- Weights & Biases；
- 数据库；
- 分布式任务系统。

只有当项目规模出现真实需求时再升级。

---

## 18. 已知限制

当前限制包括：

- 仅支持 Bernoulli Bandit；
- config 没有正式 schema；
- 没有独立 `tests/`；
- 算法注册方式还比较直接；
- CSV 没有实验元数据 summary；
- 未记录 Git commit 和依赖版本；
- 未记录运行时间与硬件；
- 没有标准差或置信区间；
- 没有并行调度；
- 没有 checkpoint；
- 输入输出路径较固定；
- v0.x 阶段核心接口仍可能演化。

扩展时应优先解决真实瓶颈，而不是为了形式成熟而增加抽象。

---

## 19. 增量开发规则

普通功能开发时应：

1. 先读取 `CONTEXT.md`；
2. 再读取 `PROGRESS.md`；
3. 读取与当前任务直接相关的代码；
4. 保持已有接口，除非任务明确要求重构；
5. 每次修改一个清晰职责；
6. 修改后立即执行最小测试；
7. 完成后执行完整 reproduce；
8. 同步更新相关文档；
9. 创建清晰 Git commit。

未经明确讨论，不应：

- 一次性重命名所有核心文件；
- 用新框架替换整个 v0.1；
- 删除断言而不提供替代测试；
- 修改 CSV schema 而不更新绘图脚本；
- 修改 seed 传播方式而不做复现测试；
- 让算法访问真实 arm mean；
- 将有限实验结果写成普遍理论结论。

---

## 20. 新窗口与 AI 协作恢复协议

`CONTEXT.md` 能恢复高层状态，但不能替代真实代码。

### 20.1 规划或路线讨论

提供：

```text
CONTEXT.md
PROGRESS.md
```

通常足够。

### 20.2 新增或修改算法

至少提供：

```text
CONTEXT.md
PROGRESS.md
algorithms/base.py
相关算法文件
run.py
configs/basic.json
```

例如实现 KL-UCB：

```text
CONTEXT.md
PROGRESS.md
algorithms/base.py
algorithms/ucb1.py
run.py
configs/basic.json
```

### 20.3 修改环境

至少提供：

```text
CONTEXT.md
PROGRESS.md
envs/bernoulli_bandit.py
run.py
configs/basic.json
受影响的算法文件
```

### 20.4 修改日志或绘图

至少提供：

```text
CONTEXT.md
PROGRESS.md
run.py
plots/plot_regret.py
configs/basic.json
```

### 20.5 架构升级或版本发布

优先上传完整项目压缩包，并要求按以下顺序读取：

1. `CONTEXT.md`
2. `PROGRESS.md`
3. `README.md`
4. `run.py`
5. 相关算法和环境文件
6. config 和测试

不得只根据聊天记忆猜测当前接口。

---

## 21. 当前下一步

v0.1 完成后，推荐进入 v0.2：

1. 建立 `tests/`；
2. 把关键不变量转成自动化测试；
3. 实现一个新算法；
4. 加入标准差阴影；
5. 扩展 config 和文档；
6. 重新完成复现性验收；
7. 发布 v0.2。

候选算法优先级：

```text
Epsilon-Greedy
    简单基线，适合验证扩展接口

KL-UCB
    深入理解分布相关的置信上界

Bernstein-UCB
    连接方差自适应集中不等式
```

具体选择应服从下一阶段的理论学习目标。

---

## 22. 文档维护约定

每次重要修改后：

- `README.md`
  - 更新用户使用方式；
- `CONTEXT.md`
  - 更新当前接口和设计约束；
- `PROGRESS.md`
  - 更新完成情况和下一里程碑。

三份文档定位：

```text
README
    项目如何使用

CONTEXT
    项目现在如何设计和继续开发

PROGRESS
    项目已经走到哪里
```
