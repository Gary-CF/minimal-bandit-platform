# Bandit Project Progress

> 本文件记录项目的开发进度、验收结果、已知问题和下一步计划。
> 它主要回答：**项目已经走到哪里，目前哪些结论可信，v1.0 发布前还缺什么。**

---

## 1. 当前概况

- **项目名称**：Minimal Bandit Platform
- **当前目标版本**：v1.0
- **当前状态**：v1.0 development，功能主体完成，正在做结果分析与交付验收
- **v0.1**：8 月 1 日完成的教学型最小闭环
- **研究对象**：Stochastic Multi-Armed Bandit
- **当前环境**：
  - Bernoulli Bandit
  - Gaussian Bandit
- **当前算法**：
  - Random Policy
  - UCB1
  - UCB-V
  - Thompson Sampling（Beta–Bernoulli）
- **当前核心产出**：
  - config 驱动实验；
  - multi-seed 与 multi-horizon；
  - step-level CSV；
  - 标准 benchmark；
  - 最小测试；
  - 结果解释与异常审计；
  - v1.0 benchmark report。

版本说明：

```text
v0.1 = 教学型最小闭环
v1.0 = 暑期正式交付
```

本阶段不使用 `v0.2` 作为版本名称。

---

## 2. v0.1 已完成里程碑

v0.1 完成了：

- Bernoulli Bandit；
- Random Policy；
- UCB1；
- Bernoulli Thompson Sampling；
- 算法与环境解耦；
- 显式 RNG；
- JSON config；
- multi-algorithm / multi-seed；
- step-level CSV；
- mean cumulative regret；
- plotting；
- `reproduce.sh`；
- README、requirements、`.gitignore`。

它证明了最小实验闭环可以运行，但尚未构成暑期最终交付。

---

## 3. v1.0 目标

v1.0 的交付标准为：

```text
完整闭环
+ 正确性验证
+ 环境–算法兼容性
+ 标准实验
+ 多 seed 不确定性
+ 结果分析
+ 一键复现
+ 完整文档
+ Git release
```

Must-have：

- Bernoulli Bandit；
- Gaussian Bandit；
- Random；
- UCB1；
- UCB-V / Bernstein 型算法；
- Thompson Sampling；
- config；
- seed；
- logging；
- multi-seed；
- plotting；
- tests；
- reproduce；
- README；
- 实验报告；
- release/tag。

---

## 4. v1.0 已完成内容

### 4.1 Gaussian Bandit

已新增：

```text
envs/gaussian_bandit.py
```

当前意义：

- reward 不再被限制为 `0/1`；
- 可以讨论 bounded 与 sub-Gaussian 假设的差异；
- 可以测试 environment–algorithm compatibility；
- 为后续连续奖励 Bandit 奠定接口基础。

### 4.2 UCB-V

已新增：

```text
algorithms/ucb_v.py
```

当前实现关注：

- counts；
- reward sums；
- reward-square / empirical variance statistics；
- 早期初始化；
- reward range；
- empirical-Bernstein-style bonus；
- 与 UCB1 共享统一接口。

开发期辅助检查：

```text
check_ucb_v.py
configs/ucb_v_smoke.json
```

### 4.3 环境–算法兼容性

已加入 compatibility 检查思路和辅助脚本：

```text
check_compatibility.py
```

当前边界：

- Beta–Bernoulli TS 不直接支持 Gaussian；
- bounded UCB-V 不应无检查地用于 Gaussian；
- Gaussian 只运行当前实现明确接受的算法配置；
- 错误组合应尽早给出清晰错误，而不是在 update 阶段产生隐蔽错误。

### 4.4 Config 系统增强

当前 config 已从 v0.1 的平面形式扩展为：

```text
experiment_name
environment
algorithms
horizon 或 horizons
seeds
algorithm parameters
```

支持：

- string algorithm config；
- dictionary algorithm config；
- single horizon；
- multiple horizons；
- experiment-specific output；
- config snapshot；
- 不改源码切换实验。

当前配置文件包括：

```text
basic.json
config_system_smoke.json
different_horizon.json
easy_gap.json
hard_gap.json
ucb_v_smoke.json
```

### 4.5 日志增强

当前记录字段包括：

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

输出按 experiment 分目录，并在结果目录保存 config snapshot。

### 4.6 自动测试

已建立：

```text
tests/test_environment.py
tests/test_ucb.py
tests/test_thompson.py
```

正式命令：

```bash
python -m pytest -q
```

当前结果：

```text
所有被该命令收集的测试通过
```

曾直接运行：

```bash
pytest -q
```

在 collection 阶段出现：

```text
No module named 'envs'
No module named 'algorithms'
```

排查结论：

- 测试函数尚未运行；
- 不属于 assertion failure；
- 不说明算法错误；
- 原因是 launcher 的 import path 与 `python -m pytest` 不同；
- 当前统一使用 `python -m pytest -q`。

### 4.7 标准实验

已完成或已形成结果的核心实验：

- easy-gap；
- hard-gap；
- different-horizon；
- action frequency；
- Gaussian low-noise 诊断。

标准实验主要使用 10 个 seed，避免单 seed 排名。

### 4.8 实验报告

已创建：

```text
reports/benchmark_v1_0.md
```

当前正在从简短结论扩展为：

```text
Problem formulation
Platform design
Algorithms
Experimental setup
Results and analysis
Anomaly audit
Limitations
Release checklist
```

---

## 5. 当前 benchmark 结果

以下数值来自当前 benchmark 汇总，不应解释成普遍算法排名。

### 5.1 Easy-gap Bernoulli

配置：

```text
arm_means = [0.02, 0.05, 0.95]
horizon = 5000
num_seeds = 10
```

最终 cumulative pseudo-regret 均值：

| Algorithm | Mean | Std |
|---|---:|---:|
| Random | 3039.375 | 37.077 |
| Thompson Sampling | 4.932 | 0.641 |
| UCB1 | 32.016 | 0.955 |
| UCB-V | 58.824 | 6.223 |

观察：

- Random 近似线性；
- 三种学习算法几乎锁定 arm 2；
- TS 在该极易环境中有限时间表现最好；
- UCB-V 早期 range correction 带来额外探索成本。

### 5.2 Hard-gap Bernoulli

配置：

```text
arm_means = [0.45, 0.48, 0.50]
horizon = 10000
num_seeds = 10
```

最终 cumulative pseudo-regret 均值：

| Algorithm | Mean | Std |
|---|---:|---:|
| Random | 232.929 | 2.433 |
| Thompson Sampling | 54.209 | 23.182 |
| UCB1 | 139.053 | 21.085 |
| UCB-V | 123.682 | 21.154 |

观察：

- hard-gap 的困难在于识别，不代表绝对 regret 必然高；
- TS 的 seed 波动明显增大；
- UCB-V 均值略低于 UCB1；
- 只有 10 个 seed，尚不足以证明 UCB-V 稳定优于 UCB1。

### 5.3 Different-horizon

最终 regret：

| Horizon | Random | TS | UCB1 | UCB-V |
|---:|---:|---:|---:|---:|
| 500 | 114.40 | 10.44 | 32.46 | 42.81 |
| 1000 | 231.05 | 11.00 | 45.89 | 55.24 |
| 2000 | 465.58 | 11.72 | 63.14 | 69.92 |
| 5000 | 1161.03 | 13.76 | 82.25 | 85.17 |
| 10000 | 2329.29 | 15.17 | 96.63 | 95.57 |

观察：

- Random 随 \(T\) 近似成比例增长；
- 学习算法增长远慢于 Random；
- 结果与 sublinear regret 预期一致；
- 当前 horizon 点和 seed 数不能验证具体的 \(O(\log T)\) bound。

### 5.4 Gaussian low-noise

当前诊断结果中：

| Algorithm | Mean final regret | Std |
|---|---:|---:|
| Random | 663.29 | 12.059 |
| noise-scale-aware UCB configuration | 2.13 | 0.564 |

该实验用于展示已知低噪声尺度时更窄置信区间的效果，不是 UCB-V 优势实验，也不能用于评价 Beta–Bernoulli TS。

---

## 6. Sanity checks

### 6.1 Random easy-gap

理论每步 regret：

\[
0.95-\frac{0.02+0.05+0.95}{3}=0.61.
\]

理论最终 regret：

\[
5000\times0.61=3050.
\]

实验：

```text
3039.375
```

接近理论值。

### 6.2 Random hard-gap

理论每步 regret：

\[
0.50-\frac{0.45+0.48+0.50}{3}
\approx0.02333.
\]

理论最终 regret：

\[
10000\times0.02333\approx233.33.
\]

实验：

```text
232.929
```

接近理论值。

这为以下环节提供了重要证据：

- Random action；
- pseudo-regret；
- cumulative update；
- multi-seed aggregation；
- horizon 对齐。

---

## 7. 已发现和已排查问题

### 7.1 Config 算法列表嵌套 bug

曾出现：

```python
algorithms = [
    config["algorithms"]
]
```

归一化后的 `config["algorithms"]` 已经是 list，再套一层会导致：

```text
list[list[dict]]
```

随后：

```python
algorithm_config["name"]
```

报：

```text
TypeError: list indices must be integers or slices, not str
```

正确形态：

```python
algorithms = list(config["algorithms"])
```

当前状态：

```text
算法列表嵌套 bug 已修复；
config_system_smoke 已成功运行 8 个实验；
多算法、多 horizon、多 seed 调度验收通过。
```

### 7.2 `pytest -q` import-path issue

现象：

```text
ERROR during collection
ModuleNotFoundError
```

排查：

- `python -m pytest -q` 正常；
- 测试本身通过；
- 不是算法 bug；
- README 已统一 canonical command。

### 7.3 临时文件

误生成的：

```text
check_ucb_v.pyclear
```

已删除。

---

## 8. 当前尚未完成

### 8.1 今天的剩余验收

- [x] 运行 `config_system_smoke.json`；
- [x] 运行 `ucb_v_smoke.json`；
- [x] `python -m pytest -q` 通过；
- [ ] `git diff --check`；
- [ ] 检查 `git status`；
- [ ] 提交当前核心功能。

### 8.2 文档

- [x] 创建 `reports/benchmark_v1_0.md`；
- [x] 将主实验结论写入报告；
- [x] 同步 README、CONTEXT、PROGRESS；
- [ ] 对照当前代码做最终逐项校验；
- [ ] 确认 `pytest` 进入 requirements；
- [ ] 补全 README 的 clean reproduction 说明。

### 8.3 最终发布

- [ ] 删除生成输出；
- [ ] `bash reproduce.sh`；
- [ ] 检查 CSV、snapshot 与 figures；
- [ ] 新建干净 venv；
- [ ] `pip install -r requirements.txt`；
- [ ] `python -m pytest -q`；
- [ ] 再次 reproduce；
- [ ] 工作区 clean；
- [ ] annotated tag `v1.0`；
- [ ] GitHub release（若本次决定公开发布）。

---

## 9. 当前项目结构

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

---

## 10. Git 状态与提交计划

当前这一阶段尚未形成 Git commit，但已经达到需要保存开发快照的程度。

建议提交顺序：

### Commit 1：核心功能与测试

```bash
git add algorithms/ envs/ configs/ tests/
git add run.py check_compatibility.py check_ucb_v.py
git add CONTEXT.md PROGRESS.md README.md
git commit -m "feat: add UCB-V, Gaussian bandit, and experiment configs"
```

提交前必须确认：

```bash
python -m pytest -q
python run.py --config configs/config_system_smoke.json
python run.py --config configs/ucb_v_smoke.json
git diff --check
git status
```

### Commit 2：实验报告

```bash
git add reports/benchmark_v1_0.md
git commit -m "docs: add v1.0 benchmark analysis"
```

### Release commit

最终复现与文档完成后：

```bash
git add .
git commit -m "chore: prepare v1.0 release"
```

然后才创建：

```bash
git tag -a v1.0 -m "Bandit Platform v1.0"
```

---

## 11. 当前判断

目前项目已经不再是单文件教学 Demo。

它已经形成：

```text
environment
+ algorithms
+ compatibility
+ config
+ runner
+ logs
+ tests
+ standard experiments
+ analysis
+ documentation
```

尚不能称为正式发布的 v1.0，原因不是核心功能缺失，而是以下交付工作尚未闭环：

- smoke verification；
- clean reproduction；
- dependency check；
- Git history；
- release tag。

当前最大的风险已经从“功能做不完”转变为：

> 在最后阶段继续扩张范围，或者没有认真完成复现、文档和版本管理。

因此当前策略保持：

```text
不加算法
先验收
再提交
最后 release
```
