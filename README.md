# Minimal Bandit Platform — v2.2

用于学习与小规模研究的 Bandit 平台：经典多臂策略、非平稳 Bernoulli、Logistic OMD/DOMD/SCB，以及重尾线性 Bandit。提供验证集选参、冻结测试、固定数据流消融、计算与存储审计、来源快照和只读复核。

## 主要结果

原冻结测试：d=3、K=10、T=1000；3 个特征集 × 3 个场景 × 5 个奖励种子 × 7 算法，共 315 局。累计伪遗憾越低越好。

| 算法 | 总体 | 平稳 | 突变 | 线性漂移 |
|---|---:|---:|---:|---:|
| Random | 191.43 | 232.35 | 224.33 | 117.62 |
| UCB1 / SW-UCB | 112.35 | 120.28 | 133.08 | 83.67 |
| Discounted UCB | 156.12 | 183.69 | 181.53 | 103.14 |
| OMD | 85.76 | 58.72 | 123.92 | 74.65 |
| **DOMD** | **42.35** | **13.69** | **71.83** | **41.53** |
| SCB numerical | 129.08 | 150.43 | 147.78 | 89.03 |

DOMD 平均遗憾较 UCB1 低 62.3%、较 OMD 低 50.6%。这是完整策略配置的比较，正则化和置信半径也不同，不能将改善全部归因于折扣。

另完成 1332 项正式补充实验：Logistic 882、重尾固定流 81、重尾闭环 324、开销审计 45。K=10 的恒范数半圈旋转中，DOMD 与 OMD 的遗憾约为 76.04 与 76.06；Hvt 原论文常数版在当前时域表现接近随机策略，固定流裁剪次数为 0，未展示出鲁棒收益。保留负结果，不做测试后调参。

详见 [最终报告](reports/v2_2/release/report.md)、[公式映射](docs/v2_2_formula_map.md)、[实验说明](docs/v2_2_remaining_experiments.md)。

## 安装与测试

主要验收环境为 Python 3.12，归档记录实际依赖版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-v2_2-remaining.txt
python -m pytest -q tests
```

## 从头复现

```bash
python -m scripts.validate_memory_scale --output artifacts/v2_2_memory_validation_r1
python -m scripts.validate_domd_gamma --output artifacts/v2_2_domd_validation_r1
python -m scripts.validate_scb_gamma --output artifacts/v2_2_scb_validation_r1
python -m scripts.run_frozen_logistic_test --output artifacts/v2_2_frozen_test_r1
python -m scripts.reproduce_v2_2 --output artifacts/v2_2_remaining_r1 --jobs 4 --frozen-test-run artifacts/v2_2_frozen_test_r1
```

固定流 DOMD 诊断入口为 `python -m scripts.diagnose_domd_forgetting --help`。其他路径选项也可用对应入口的 `--help` 查询。补充实验中断后可执行相同命令继续；源码或协议变化时使用新目录，不能混用旧结果。原始运行源码保存在归档内。

已有实验只读复核并收尾：

```bash
python -m scripts.closeout_v2_2
```

该入口执行测试、复核既有 315 局、71 项 smoke 和 1332 项补充实验，生成最终报告、精简数据和 Git 路径清单；不重跑策略或重新选参。发布见 [RELEASE.md](RELEASE.md)。

## 实现与边界

| 内容 | 入口 |
|---|---|
| 经典 MAB / 非平稳 Bernoulli | `run.py`；[v1.5 文档](README-v1.5.md) |
| Logistic OMD / DOMD | `estimators/omd_logistic.py`、`estimators/domd_logistic.py` 及对应策略 |
| SCB 折扣 MLE 与数值投影 | `estimators/discounted_logistic_mle.py`、`algorithms/scb_weight_ucb.py` |
| Hvt、Ridge、LinUCB | `algorithms/hvt_ucb.py` |
| 旋转轨迹与解析噪声矩 | `envs/remaining_bandit_tasks.py` |
| 冻结测试 / 补充实验 | `scripts/run_frozen_logistic_test.py`、`scripts/run_remaining_v2_2.py` |

- 环境真值用于采样和评估；Hvt 的已知噪声矩尺度是显式输入。
- SCB 多起点 SLSQP 检查局部 KKT，不保证非凸投影的全局最优。
- 旋转保持参数范数，不保证臂间奖励差恒定。
- 高斯与 t(3) 同方差；t(1.5) 保留无限方差，使用有限 1.25 阶矩。
- Hvt 非平稳实验是平稳理论之外的压力测试；去裁剪版本是机制消融。
- 补充误差按 3 个特征组均值计算，SEM 不是 95% 置信区间。
- 开销实验使用独立进程和单线程 BLAS，区分状态数组、历史、外部日志与 RSS。

大体积逐局数据、源码快照、备份和运行日志保留在 `artifacts/`；Git 提交源码、配置、测试和精简报告。历史见 [v2.1 文档](README-v2.1.md)、[进度](PROGRESS.md) 和 tags。许可证见 [LICENSE](LICENSE)。
