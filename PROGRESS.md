# Bandit Project Progress

> 本文件记录项目的开发进度、已完成里程碑、验收结果、已知限制和下一阶段计划。  
> 它主要回答：**项目已经完成了什么，目前处于什么状态，下一步准备推进什么。**

---

## 1. 项目概况

- **项目名称**：Minimal Bandit Platform
- **当前版本**：v0.1
- **项目类型**：学习型、研究型最小实验平台
- **当前研究对象**：随机多臂老虎机（Stochastic Multi-Armed Bandit）
- **当前环境**：Bernoulli Multi-Armed Bandit
- **主要评价指标**：
  - instantaneous pseudo-regret
  - cumulative pseudo-regret
- **当前默认实验规模**：
  - 2 个算法
  - 5 个随机 seed
  - 每次实验 5000 个交互 step
  - 共 10 次独立实验

---

## 2. v0.1 的目标

v0.1 的核心目标不是实现大量算法，而是完成第一个完整、可信、可重复执行的科研实验闭环：

```text
JSON configuration
        ↓
algorithm × seed experiment scheduling
        ↓
step-level CSV logging
        ↓
cross-seed aggregation
        ↓
cumulative regret visualization
        ↓
one-command reproduction
```

v0.1 的完成标准包括：

- 算法与环境职责分离；
- 至少实现两种有效的 Bandit 算法；
- 显式控制随机性；
- 支持多算法、多 seed 批量运行；
- 实验参数与运行逻辑分离；
- 保存逐 step 的结构化日志；
- 对多 seed 结果逐 step 求平均；
- 自动生成 regret 对比图；
- 删除输出目录后可以一键重新生成；
- README、依赖和 Git 规则完整；
- 可以为该版本创建 Git commit 和 tag。

---

## 3. 已完成内容

### 3.1 Bernoulli Bandit 环境

已实现 Bernoulli Bandit 环境，当前职责包括：

- 保存每个臂的真实奖励概率 `arm_means`；
- 根据算法选择的 action 生成 Bernoulli reward；
- 提供最佳臂均值；
- 计算所选动作对应的 pseudo-regret；
- 保持环境真实参数与算法内部决策隔离。

当前默认环境配置：

```json
{
  "arm_means": [0.3, 0.5, 0.7],
  "horizon": 5000
}
```

---

### 3.2 统一的算法交互方式

当前实验循环使用统一交互方式：

```python
action = algorithm.select_action()
reward = env.step(action)
algorithm.update(action, reward)
```

当前设计原则：

- 算法负责选择动作和更新内部状态；
- 环境负责产生奖励和提供评估所需的真实信息；
- 算法不直接读取环境中的真实 `arm_means`；
- 每次单独实验重新创建环境和算法对象；
- 不同 seed 之间不共享算法状态。

---

### 3.3 Random Policy

已实现随机策略基线，用于：

- 验证环境与算法之间的交互闭环；
- 验证随机数生成器的显式传递；
- 提供最基础的行为参考；
- 检查统一算法接口是否可用。

---

### 3.4 UCB1

已实现 UCB1，包括：

- 初始化阶段依次尝试所有臂；
- 维护各臂的选择次数；
- 维护各臂的累计奖励；
- 计算经验均值；
- 根据估计均值与置信上界选择动作；
- 根据 reward 更新内部统计量。

已加入的关键一致性检查包括：

```text
初始化动作依次覆盖所有臂
algorithm.counts.sum() == horizon
algorithm.counts 与 action_counts 一致
algorithm.reward_sums 与 reward_sums 一致
algorithm.estimated_means 与实验端经验均值一致
```

---

### 3.5 Thompson Sampling

已实现 Bernoulli Bandit 下的 Thompson Sampling，包括：

- 为每个臂维护 Beta 后验参数 `alpha` 和 `beta`；
- 从各臂后验分布中采样；
- 选择采样值最大的臂；
- 根据成功或失败更新 Beta 后验。

已加入的关键一致性检查包括：

```text
alpha_i - 1 = 第 i 个臂的成功次数
beta_i - 1 = 第 i 个臂的失败次数
alpha_i + beta_i - 2 = 第 i 个臂的选择次数
alpha_i >= 1
beta_i >= 1
```

---

### 3.6 单次实验封装

原先集中在 `main()` 中的一次实验逻辑，已经拆分为单次实验函数。

当前职责划分的核心思想是：

```text
main()
    读取配置并调度实验

run_single_experiment()
    执行一个 algorithm × seed 的完整实验

create_algorithm()
    根据算法名称创建算法对象

save_records()
    保存结构化 CSV 日志
```

单次实验会完成：

1. 根据根 seed 创建随机状态；
2. 派生环境 RNG 和算法 RNG；
3. 创建环境对象；
4. 创建算法对象；
5. 运行 `horizon` 个 step；
6. 记录 action、reward 和 regret；
7. 执行公共与算法专属检查；
8. 返回逐 step records。

---

### 3.7 随机性与可复现性

当前使用 NumPy Generator API：

```python
np.random.default_rng(...)
```

根 seed 通过 `SeedSequence` 派生环境和算法各自的随机状态：

```python
seed_sequence = np.random.SeedSequence(seed)
env_seed, algorithm_seed = seed_sequence.spawn(2)

env_rng = np.random.default_rng(env_seed)
algorithm_rng = np.random.default_rng(algorithm_seed)
```

这样做的目的包括：

- 环境随机性与算法随机性分离；
- 减少算法内部随机调用变化对环境随机序列的干扰；
- 只记录根 seed 即可恢复两个子 RNG；
- 相同代码、配置和 seed 下可以重放实验。

当前已采用同 seed 两次运行并比较 CSV 的方式检查复现性。

---

### 3.8 JSON 配置

已增加 JSON 配置和命令行参数。

当前运行方式：

```bash
python run.py --config configs/basic.json
```

当前默认配置：

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

当前配置已经将以下内容从代码逻辑中分离：

- 环境臂均值；
- 每次实验长度；
- 参与运行的算法；
- 重复实验使用的 seed。

---

### 3.9 多算法、多 seed 调度

当前 `main()` 支持双层循环：

```text
for each algorithm
    for each seed
        run one independent experiment
```

默认运行：

```text
2 algorithms × 5 seeds = 10 experiments
```

每次调用都会重新创建：

- environment；
- algorithm；
- records；
- action counts；
- reward sums；
- cumulative regret 状态。

因此不同实验之间不会继承状态。

---

### 3.10 CSV 日志

每个 algorithm × seed 组合保存一个独立 CSV 文件，例如：

```text
results/ucb1_seed_0.csv
results/thompson_sampling_seed_0.csv
```

当前 CSV 字段为：

```text
algorithm
seed
step
action
reward
instant_regret
cumulative_regret
```

默认每个 CSV 包含：

```text
1 行表头 + 5000 行实验数据 = 5001 行
```

日志采用覆盖写入，而不是追加写入。相同 algorithm 和 seed 重跑时会重新生成对应文件。

终端输出已经从详细调试信息缩减为：

- 当前运行的 algorithm 和 seed；
- 输出文件路径；
- 必要 warning；
- 脚本执行进度。

正式实验数据以 CSV 为准，不依赖终端输出保存。

---

### 3.11 多 seed 聚合与绘图

已实现：

```text
plots/plot_regret.py
```

当前功能包括：

- 自动发现 `results/` 下的 CSV；
- 读取 algorithm、seed、step 和 cumulative regret；
- 验证一个 CSV 中 algorithm 和 seed 保持不变；
- 验证 step 连续；
- 验证累计 regret 单调不减；
- 按算法分组；
- 检查同一算法不同 seed 的 step 对齐；
- 将多 seed 曲线转换成二维矩阵；
- 沿 seed 维度逐 step 求平均；
- 绘制多算法 mean cumulative regret；
- 保存图片。

当前正式输出：

```text
figures/regret_curve.png
```

当前只绘制平均曲线，标准差或置信区间阴影尚未实现。

---

### 3.12 一键复现

已实现：

```text
reproduce.sh
```

当前流程：

1. 删除旧的 `results/`；
2. 删除旧的 `figures/`；
3. 使用默认配置运行所有实验；
4. 生成全部 CSV；
5. 调用绘图脚本；
6. 生成平均累计 regret 图。

运行方式：

```bash
bash reproduce.sh
```

核心验收要求：

```text
删除 results/ 和 figures/
        ↓
不修改任何源码
        ↓
执行 bash reproduce.sh
        ↓
重新生成 10 个 CSV 和 regret_curve.png
```

---

### 3.13 README、依赖与 Git 配置

已加入：

- `README.md`
- `requirements.txt`
- `.gitignore`

当前第三方依赖：

```text
numpy
matplotlib
```

当前 Git 忽略项包括：

```text
.venv/
venv/
__pycache__/
*.py[cod]
results/
figures/
.vscode/
.DS_Store
```

README 已包含：

- 项目定位；
- 已实现算法；
- 环境和 regret 定义；
- 项目结构；
- 安装方法；
- 复现命令；
- 配置说明；
- 输出文件；
- 复现性检查；
- 初步结果解释；
- 后续扩展方向。

---

## 4. 当前项目结构

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

运行时生成但不进入版本控制：

```text
results/
figures/
__pycache__/
```

---

## 5. v0.1 验收清单

### 功能验收

- [x] Bernoulli Bandit 环境可运行；
- [x] Random Policy 可运行；
- [x] UCB1 可运行；
- [x] Thompson Sampling 可运行；
- [x] 算法与环境职责分离；
- [x] 显式使用 NumPy Generator；
- [x] 环境 RNG 与算法 RNG 分离；
- [x] 单次实验封装完成；
- [x] 配置通过 JSON 提供；
- [x] 支持多算法、多 seed；
- [x] 每次实验保存独立 CSV；
- [x] CSV 保存逐 step action、reward 和 regret；
- [x] 可自动读取全部 CSV；
- [x] 可按算法聚合多 seed 结果；
- [x] 可逐 step 计算 mean cumulative regret；
- [x] 可生成多算法 regret 对比图；
- [x] 可通过 Bash 脚本一键复现；
- [x] 已编写 README；
- [x] 已记录 Python 依赖；
- [x] 生成文件已加入 Git ignore。

### 发布前检查

正式创建或确认 `v0.1` tag 前，应运行：

```bash
python -m py_compile     run.py     plots/plot_regret.py     algorithms/base.py     algorithms/ucb1.py     algorithms/thompson_sampling.py     envs/bernoulli_bandit.py
```

检查 JSON：

```bash
python -m json.tool configs/basic.json
```

从空目录复现：

```bash
rm -rf results figures
bash reproduce.sh
```

检查 CSV 数量：

```bash
find results -maxdepth 1 -name "*.csv" | wc -l
```

预期：

```text
10
```

检查单个文件行数：

```bash
wc -l results/ucb1_seed_0.csv
```

预期：

```text
5001
```

检查图片：

```bash
test -f figures/regret_curve.png     && echo "regret figure exists"
```

检查同 seed 复现：

```bash
cp results/ucb1_seed_0.csv /tmp/ucb1_seed_0_first.csv

python run.py --config configs/basic.json

diff     results/ucb1_seed_0.csv     /tmp/ucb1_seed_0_first.csv
```

`diff` 无输出表示两次文件一致。

检查 Git：

```bash
git diff --check
git status
```

---

## 6. 当前已知限制

v0.1 有意保持最小化，目前存在以下限制：

1. 仅支持 Bernoulli reward；
2. 当前算法数量有限；
3. 算法注册仍可能依赖 `if` 分支；
4. JSON config 只有基础读取与类型转换，没有正式 schema；
5. 没有独立的 `tests/` 自动化测试目录；
6. 很多正确性检查仍在运行函数中以 assert 形式存在；
7. CSV 适合当前规模，但不适合大规模分布式实验；
8. 没有实验级 summary 文件；
9. 没有记录 Git commit、依赖版本、运行时间和硬件信息；
10. 绘图脚本默认使用固定输入和输出目录；
11. 只绘制 mean cumulative regret；
12. 没有标准差、标准误或置信区间；
13. 没有并行调度；
14. 没有 checkpoint、失败恢复或断点续跑；
15. 当前有限实验不能证明理论 regret bound；
16. 当前算法对比只对默认配置和所选 seeds 有效。

---

## 7. 下一里程碑建议

### v0.2：自动化测试与一个新算法

建议目标：

> 在不破坏 v0.1 核心接口的前提下，建立最小测试体系，并增加一个新的算法基线。

建议顺序：

1. 创建 `tests/`；
2. 为 BernoulliBandit 编写最小测试；
3. 为 UCB1 的初始化和状态更新编写测试；
4. 为 Thompson Sampling 的后验更新编写测试；
5. 将适合的运行时 assert 转化为独立测试；
6. 实现一个新算法；
7. 更新配置、绘图和文档；
8. 发布 v0.2。

候选新算法：

- `EpsilonGreedy`
  - 适合作为简单基线；
  - 容易验证统一接口；
- `KLUCB`
  - 适合深入学习分布相关的置信上界；
- `BernsteinUCB`
  - 适合连接方差自适应集中不等式。

### v0.5：增强实验管理

候选功能：

- config schema 校验；
- 多组环境配置；
- summary CSV；
- 标准差或标准误阴影；
- 命令行指定输出目录；
- 自动记录运行时间；
- 自动记录 Git commit；
- 更多评价指标；
- 更清晰的算法注册机制。

### v1.0：稳定的 Bandit 研究平台

v1.0 不应只由算法数量决定，建议满足：

- 核心接口相对稳定；
- 关键逻辑有自动测试；
- 支持多种算法和环境；
- 实验元数据记录完整；
- 结果可以稳定复现；
- README 能独立指导使用；
- 历史 tag 清晰；
- 至少有一组解释充分的 benchmark 实验；
- 外部用户可以在不阅读全部源码的情况下运行项目。

---

## 8. 文档更新规则

每完成一个可独立说明的功能后，应同步更新：

- `README.md`
  - 更新外部用户需要知道的安装、运行和输出方式；
- `CONTEXT.md`
  - 更新当前架构、接口、约束和设计决策；
- `PROGRESS.md`
  - 更新已完成内容、验收状态、限制和下一步。

维护原则：

- 不把计划写成已完成事实；
- 不把一次实验结果写成一般理论结论；
- 代码和测试与文档冲突时，以实际代码和运行结果为准；
- 发现冲突后应立即修正文档；
- 发布新 tag 时记录验收结果和版本范围。
