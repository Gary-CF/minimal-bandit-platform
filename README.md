# Minimal Bandit Platform — v2.1

用于学习与小规模研究的 Bandit 平台：经典随机多臂策略、非平稳 Bernoulli 环境与遗忘基线、Logistic 批量估计及 One-Pass OMD。实验配置、逐轮日志、数值失败、来源和解释均可追踪。

v2.1 的计算验收由复现入口生成；正式冻结状态以 `v2.1` Git tag 和对应 `reports/v2_1/closeout.md` 为准。

## 安装与本阶段复现

使用 Python 3.12；其他 Python/平台组合没有逐一认证。直接依赖包含 NumPy、SciPy、Matplotlib、pytest：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-d8.txt
python scripts/reproduce_v2_1.py --output-root artifacts/v2_1_local
python scripts/reproduce_v2_1.py --output-root artifacts/v2_1_local --verify-only
```

也可用 `bash reproduce_v2_1.sh --output-root artifacts/v2_1_local`。输出目录须不存在或为空；失败现场保留，不删除、不自动重试。再次执行需选择新名字。历史 `requirements-lock.txt` 属于 v1.5，缺少 SciPy；本阶段每次记录实际 Python 和四个直接依赖版本至 `versions.json`。

入口先执行一次全套 pytest，再完成：

| 实验 | 配置与数量 | 评估 |
|---|---|---|
| 固定流 OMD / Batch | 复用已归档 E8 的200条观测、5个前缀 | 相同固定评估点上的概率RMSE、调用时间 |
| 平稳 Logistic 闭环 | d=3、K=10、T=1000、5 seeds；Random / OMD，10条轨迹 | 累计伪遗憾 |
| 时间与状态 | T=200/1000/5000，各3次 | 更新/求解/选择/日志时间与数组载荷 |
| 非平稳压力测试 | 第501轮突变、全程线性漂移；同样两个策略与5 seeds，20条轨迹 | 动态伪遗憾、更新前全臂概率RMSE |

生成五张关键图、`report.md`、逐轮CSV、源码快照、来源哈希和依赖版本，并复制到 `reports/v2_1/<输出目录名>/`，另生成 evidence ZIP。`--verify-only` 只读检查状态、文件全集、哈希、当前源码和逐轮指标。冻结后可对对应归档目录再次验证。

E8 的已验收数据是固定输入，不重新抽样；此前批量和单步正确性验证由归档及回归测试保留。本入口复现本阶段 Logistic 实验；v1.5 全量420次基准和 D6 非平稳 Bernoulli 大实验仍使用各自入口与历史归档，没有为本次收尾重复运行。

## 当前能力与入口

| 内容 | 入口/位置 |
|---|---|
| 十个经典平稳策略，Bernoulli / Gaussian | `run.py`；[v1.5 使用说明](README-v1.5.md) 中的策略表和配置示例 |
| 分段突变/线性漂移 Bernoulli，UCB1/SW-UCB/Discounted-UCB | `run.py`，`configs/d4_sw_smoke.json`、`configs/d5_du_smoke.json` |
| Logistic 采样、稳定损失/梯度/Hessian | `envs/logistic_bandit.py` |
| 正则化、球约束批量 Logistic 估计 | `estimators/batch_logistic.py`；`scripts/d8_batch_reference.py` |
| OMD 受约束更新与乐观选臂 | `estimators/omd_logistic.py`；`algorithms/logistic_omd.py` |
| 固定流、平稳闭环、压力测试 | `scripts/reproduce_v2_1.py`；`configs/v2_1.json` |
| 公式对应、十分钟讲解 | [公式—代码表](docs/v2_1_formula_map.md)、[讲解提纲](docs/v2_1_walkthrough.md) |
| 验收与发布 | [进度](PROGRESS.md)、[冻结步骤](RELEASE.md) |

Logistic 使用独立实验入口，没有把特征策略硬接入 `run.py` 的经典 `select_action()` 接口。非平稳 Logistic 是实验驱动提供真参数路径、逐轮复用 Logistic 环境；不新增一个声称具备非平稳保证的 OMD 算法。

```bash
python run.py --config configs/d4_sw_smoke.json
python run.py --config configs/d5_du_smoke.json
```

## 实验口径和边界

E10 使用同一 X/y 固定流；闭环策略的选臂影响后续观测。真实参数/均值只由环境和评估器使用。动态伪遗憾每轮以当时的最优臂为参照：

$$R_T=\sum_{t=1}^T\left[\max_a\sigma(x_a^\top\theta_t^\star)-\sigma(x_{A_t}^\top\theta_t^\star)\right].$$

正文原式 `paper_v2_literal` 是正式版本；正文/附录时间因子差异保留在公式说明中。`appendix_time` 是单独标注的备选模式，本阶段正式实验没有使用。OMD 在变化环境中的表现属于压力测试，不将平稳理论保证延伸为动态保证，也没有实现 DOMD-GLB、变化检测、重置或折扣 OMD。

图中阴影为5 seeds的 SEM，不是95%置信区间。Batch仅五次重拟合与OMD逐步更新是不同工作量；96字节只统计 d=3 时 theta/H 数组。外部日志、Python对象及临时求解内存不在这个数字内。单个固定动作集和有限预算不支持普遍算法排名或复杂度证明。

保留 v1.5 与 v2.0 历史证据；旧 `reproduce.sh` 会清理整个 results/figures，应使用对应版本的新入口。许可证见 [LICENSE](LICENSE)。
