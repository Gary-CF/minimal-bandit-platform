# Minimal Bandit Platform — v1.5

一个小而透明的平稳随机多臂老虎机实验平台，用于学习探索—利用权衡、比较算法行为，并练习可复现科研工程。

**v1.5：10 个策略、2 种环境、统一交互接口、严格配置校验及独立正式实验流水线。** 发布验收与实际实验结果见 [PROGRESS](PROGRESS.md) 和 [v1.5 实验报告](reports/v1_5/benchmark_v1_5.md)。

## 安装与快速开始

代码语法要求 Python 3.10+；本次发布实际验证 Python 3.12，锁定依赖请使用 `requirements-lock.txt`。其他 Python/平台组合尚未逐一认证。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pytest -q
python run.py --config configs/gaussian_algorithms_smoke.json
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活。`requirements.txt` 只列直接依赖，供开发时选择兼容版本；正式复现优先使用锁定版本。

## 十个策略

以下参数位于算法配置的 `parameters` 内。`K`、策略 RNG 和当前 horizon 由 runner 按需注入。

| 配置名 | Python 类 | 参数（省略时的默认值） | Bernoulli | Gaussian |
|---|---|---|:---:|:---:|
| `random` | `RandomPolicy` | 无 | ✓ | ✓ |
| `ucb1` | `UCB1` | 无 | ✓ | — |
| `ucb_v` | `UCBV` | `reward_range=1.0`，至少 1 | ✓ | — |
| `thompson_sampling` | `ThompsonSampling` | 固定 Beta(1,1) 先验 | ✓ | — |
| `epsilon_greedy` | `EpsilonGreedy` | `epsilon=0.1`，固定 ε | ✓ | ✓ |
| `etc` | `ETC` | 必填正整数 `exploration_rounds_per_arm` | ✓ | ✓ |
| `moss` | `MOSS` | 当前 horizon 自动注入 | ✓ | — |
| `kl_ucb` | `KLUCB` | `c=3.0`，c≥0 | ✓ | — |
| `gaussian_ucb` | `GaussianUCB` | 必填 `known_std` | — | ✓ |
| `gaussian_thompson_sampling` | `GaussianThompsonSampling` | 必填 `noise_variance`；`prior_mean=0.0`、`prior_variance=1.0` | — | ✓ |

总共 10 个不同策略实现（包含 Random 基线），不是每个环境都支持全部十个。Bernoulli 有 8 个可用策略，Gaussian 有 5 个，其中 3 个通用策略重叠。`GaussianBandit` 是环境，不计为算法。

Gaussian 环境接受逐臂标准差；Gaussian UCB 的标量 `known_std` 必须不小于每个臂的标准差，可以是保守共同上界。Gaussian TS 的公共 `noise_variance` 必须与每个臂的 `arm_stds**2` 匹配，不支持一般异方差模型。标准差 2 对应方差 4。

## 一轮实验与指标

```python
action = algorithm.select_action()
reward = env.step(action)
algorithm.update(action, reward)
instant_regret = env.pseudo_regret(action)
```

环境生成奖励，策略只接收动作与实际反馈，runner 负责评估和记录。真实均值不进入策略。

$$r_t=\mu^\star-\mu_{A_t},\qquad R_T=\sum_{t=1}^T r_t.$$

日志中的 regret 均为 **pseudo-regret**，不是用一次随机奖励计算的 realized regret。它逐步非负、累计不减。Gaussian reward 可为负，不影响这一性质。

## 配置约定

```json
{
  "experiment_name": "my_comparison",
  "environment": {"name": "bernoulli", "arm_means": [0.2, 0.5, 0.7]},
  "algorithms": [
    "random", "ucb1", "moss", "kl_ucb",
    {"name": "epsilon_greedy", "parameters": {"epsilon": 0.1}},
    {"name": "etc", "parameters": {"exploration_rounds_per_arm": 20}}
  ],
  "horizons": [1000, 3000],
  "seeds": [0, 1, 2]
}
```

保存为配置文件后执行 `python run.py --config <配置路径>`。单预算使用 `horizon`，与 `horizons` 二选一。所有算法 × horizon × seed 都重新实例化。

- T 必须是真正的正整数；seed 是非负整数；拒绝 bool、小数截断和重复项。
- 同一配置内不允许重复算法名。比较多个 ε 时，使用不同 `experiment_name` 的独立配置。
- 未知字段、拼错参数名、缺失必填参数、NaN/Inf、非法环境—算法组合在整个 batch 写入前拒绝。
- `experiment_name` 使用字母/数字开头及字母、数字、`_`、`.`、`-`，作为单个目录名，不接受路径。

普通运行结果在当前工作目录的 `results/<experiment_name>/`：一个规范化 `config_snapshot.json`，以及 `<environment>_<algorithm>_T<horizon>_seed<seed>.csv`。

CSV 字段为 `environment,algorithm,horizon,seed,step,action,reward,instant_regret,cumulative_regret`。step 从 1 开始，action 从 0 开始。已有快照不匹配时拒绝写入；相同配置允许重跑覆盖。普通 runner 尚不提供断点恢复、原子 batch 或完整源码溯源；正式 suite 提供额外校验与 manifest。

## 一键复现 v1.5

```bash
bash reproduce_v1_5.sh
```

等价的跨平台入口：

```bash
python scripts/benchmark_v1_5.py
```

默认生成到 `artifacts/v1_5/`。**输出目录必须不存在或为空；不会删除旧实验。** 再跑一次请换新目录：

```bash
python scripts/benchmark_v1_5.py --output-root artifacts/v1_5_repeat
python scripts/benchmark_v1_5.py --output-root artifacts/v1_5_repeat --verify-only
```

| 正式配置 | 环境 | 均值 | 算法数 | 预算 | seeds | CSV 数 |
|---|---|---|---:|---|---|---:|
| `v1_5_bernoulli_easy.json` | Bernoulli | [0.2,0.5,0.7] | 8 | 1000、3000 | 0–9 | 160 |
| `v1_5_bernoulli_hard.json` | Bernoulli | [0.45,0.48,0.5] | 8 | 1000、3000 | 0–9 | 160 |
| `v1_5_gaussian.json` | Gaussian，std 全为 1 | [0,0.2,0.5] | 5 | 1000、3000 | 0–9 | 100 |

验收预期：**420 次实验、840,000 行记录、42 行统计、6 张图**。

入口先运行 pytest，再运行全部配置；分析前严格检查预期文件集合、快照、CSV 字段、身份、步数、奖励及逐步 regret，拒绝缺失、混入或损坏结果。图显示跨 seed 均值 ± SEM，并展示所有配置臂的动作频率。SEM 不是单次运行的标准差或 95% 置信区间。

输出根目录包含 `results/`、`figures/`、`reports/`、`logs/` 和 `suite_manifest.json`。manifest 记录状态、源码 SHA256、配置、Python/依赖、耗时及结果/报告哈希。`--verify-only` 不运行算法、不修改文件，检查来源及产物完整性。若源码变化，应在新目录重跑；失败目录保留诊断信息，不会自动冒充完成。

已提交的精简实验证据位于 `reports/v1_5/`，原始大量 CSV 留在可再生输出目录。不同 NumPy 版本可能改变随机实现，数值复现应固定代码与锁定依赖。

## 工程结构与阅读顺序

| 位置 | 职责 |
|---|---|
| `algorithms/` | 选臂、状态更新；Random 在 base.py |
| `envs/` | 采样、环境参数和 pseudo-regret |
| `run.py` | 配置、工厂、全 batch 预校验、调度、日志和不变量 |
| `configs/` | 旧配置、新 v1.5 正式配置、预期失败 smoke |
| `tests/` | 算法、边界、兼容性、结果损坏回归 |
| `scripts/benchmark_v1_5.py` | 独立安全复现、严格验证、汇总、绘图和 manifest |
| `reports/v1_5/` | 冻结的正式汇总、图、运行元数据和发布验收 |
| `CONTEXT.md` / `PROGRESS.md` | 开发契约 / 验收状态 |
| `RELEASE.md` | 合入现有仓库及开源发布步骤 |
| `reports/engineering_walkthrough_v1_5.md` | 算法复习、接口细节和 15 分钟讲解 |

旧 `reproduce.sh`、`plots/plot_benchmarks.py`、`scripts/summarize_benchmarks.py` 保留为 v1.0 历史流程。**旧 reproduce.sh 会删除当前目录整个 results/figures，不要拿它执行 v1.5 复现。** 历史报告的旧发布清单不代表当前状态。

## 边界

平稳随机 bandit；无 contextual、linear、adversarial、non-stationary、neural bandit 或分布式功能。固定参数与有限预算实验不能证明普遍算法排名或严格渐近 regret 界。当前算法内部状态 O(K)，runner 每次将 T 行日志存入内存，为 O(T)。

许可证见 [LICENSE](LICENSE)。
