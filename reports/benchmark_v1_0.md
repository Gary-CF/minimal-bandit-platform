# Bandit Platform v1.0 Benchmark Report

> **Status:** canonical benchmark report for the v1.0 release candidate.
>
> 本报告对应 `configs/easy_gap.json`、
> `configs/hard_gap.json` 和
> `configs/different_horizon.json`
> 三组正式 benchmark，并由当前 release candidate
> 从空生成目录重新复现。

---

## 1. 实验目的

v1.0 benchmark 不追求堆叠算法，而是检查整个实验闭环是否可信。

核心问题：

1. Random regret 是否近似线性？
2. hard-gap 为什么更难？
3. UCB 为什么早期持续探索？
4. Thompson Sampling 为什么可能出现 seed 路径依赖？
5. UCB-V 在什么环境下可能有优势？
6. 当前结果是否支持理论预期？
7. 是否存在实现异常或实验管线 bug？

判断原则：

> 当曲线明显不符合预期时，先查环境、算法更新、regret、seed、聚合和 config，不用文字强行解释。

---

## 2. Problem formulation

共有 \(K\) 个臂。臂 \(i\) 的真实均值为 \(\mu_i\)，最优均值为：

\[
\mu^\star=\max_{i\in[K]}\mu_i.
\]

第 \(t\) 轮选择动作 \(A_t\) 后，即时 pseudo-regret 为：

\[
r_t=\mu^\star-\mu_{A_t}.
\]

累计 pseudo-regret 为：

\[
R_T=\sum_{t=1}^{T}r_t.
\]

本报告分析的是 pseudo-regret，而不是由单次随机 reward 直接计算的 realized regret。

---

## 3. Platform snapshot

当前环境：

- Bernoulli Bandit；
- Gaussian Bandit。

当前算法：

- Random Policy；
- UCB1；
- UCB-V；
- Beta–Bernoulli Thompson Sampling。

当前实验基础设施：

- JSON config；
- algorithm × horizon × seed 调度；
- 独立 env RNG 与 algorithm RNG；
- step-level CSV；
- config snapshot；
- multi-seed aggregation；
- figure；
- compatibility checks；
- automated tests；
- smoke configs。

当前正式测试命令：

```bash
python -m pytest -q
```

该命令下，当前收集到的测试通过。

---

## 4. 实验设置

### 4.1 Easy-gap Bernoulli

```text
arm_means = [0.20, 0.50, 0.70]
horizon = 5000
num_seeds = 10
```

作用：

- 最优臂与其他臂存在清晰但非极端的 gap；
- 检查算法是否能逐渐集中到最优臂；
- 检查动作选择比例；
- 检查 Random regret 的解析 sanity check。

### 4.2 Hard-gap Bernoulli

```text
arm_means = [0.45, 0.48, 0.50]
horizon = 10000
num_seeds = 10
```

作用：

- 最优臂与次优臂只差 \(0.02\)；
- Bernoulli 方差接近最大；
- 检查识别困难、持续探索和 seed 波动。

### 4.3 Different-horizon Bernoulli

```text
arm_means = [0.45, 0.48, 0.50]
horizons = [1000, 5000, 20000]
num_seeds = 10
```

作用：

- 在同一 hard-gap 环境下改变 horizon；
- 比较 final regret 随 T 的变化；
- 检查 Random 的近线性趋势；
- 比较学习算法有限时间 regret 的增长速度



## 5. 聚合方法

对于固定 experiment、algorithm 和 horizon，将不同 seed 的 cumulative regret 对齐成：

\[
\mathbf R\in\mathbb R^{S\times T},
\]

其中 \(S\) 为 seed 数。

逐 step 均值：

\[
\bar R_t=\frac{1}{S}\sum_{s=1}^{S}R_t^{(s)}.
\]

最终 regret 标准差：

\[
\operatorname{Std}(R_T)
=
\sqrt{
\frac{1}{S-1}
\sum_{s=1}^{S}
\left(R_T^{(s)}-\bar R_T\right)^2
}.
\]

当前曲线和表格只能说明所选环境、horizon、参数和 seeds 下的经验行为。

---

## 6. Sanity check：Random regret

Random Policy 每轮均匀选择臂，因此：

\[
\mathbb E[r_t]
=
\mu^\star-\frac{1}{K}\sum_{i=1}^{K}\mu_i.
\]

该量不随 \(t\) 改变，所以：

\[
\mathbb E[R_T]
=
T\left(
\mu^\star-\frac{1}{K}\sum_{i=1}^{K}\mu_i
\right).
\]

### 6.1 Easy-gap

\[
0.70-\frac{0.20+0.50+0.70}{3}
=
0.2333\ldots
\]

理论 final regret：

\[
5000\times0.2333\ldots
\approx1166.67.
\]

实验均值：

```text
1161.030
```

与解析期望接近。

### 6.2 Hard-gap

\[
0.50-\frac{0.45+0.48+0.50}{3}
\approx0.02333.
\]

理论 final regret：

\[
10000\times0.02333\approx233.33.
\]

实验均值：

```text
232.929
```

同样接近。

### 6.3 审计意义

这为以下模块提供了较强的整体 sanity check：

- Random action sampling；
- 环境 arm means；
- pseudo-regret；
- cumulative regret；
- horizon；
- CSV logging；
- multi-seed aggregation。

它不能单独证明所有学习算法实现正确，但说明实验主干没有出现明显量纲或累计错误。

---

## 7. 结果

### 7.1 Easy-gap final regret

| Algorithm | Mean | Std | SEM |
|---|---:|---:|---:|
| Random | 1161.030 | 19.974 | 6.316 |
| Thompson Sampling | 13.760 | 7.425 | 2.348 |
| UCB1 | 81.800 | 12.629 | 3.994 |
| UCB-V | 85.170 | 6.416 | 2.029 |

### 7.2 Hard-gap final regret

| Algorithm | Mean | Std | SEM |
|---|---:|---:|---:|
| Random | 232.929 | 2.433 | 0.769 |
| Thompson Sampling | 54.209 | 23.182 | 7.331 |
| UCB1 | 143.021 | 19.304 | 6.104 |
| UCB-V | 123.682 | 21.154 | 6.690 |

### 7.3 Different-horizon final regret

| Horizon | Random | Thompson Sampling | UCB1 | UCB-V |
|---:|---:|---:|---:|---:|
| 1000 | 23.105 | 15.417 | 19.454 | 19.391 |
| 5000 | 116.103 | 38.762 | 77.064 | 72.203 |
| 20000 | 466.339 | 73.858 | 236.351 | 181.978 |





## 8. 为什么 hard-gap 更难？

Easy-gap：

```text
[0.20, 0.50, 0.70]
```

最优臂与次优臂的 gap：

\[
0.70-0.50=0.20.
\]

相比 hard-gap 的 0.02，区分成本明显更低。

Hard-gap：

```text
[0.45, 0.48, 0.50]
```

最优臂与次优臂的 gap：

\[
0.50-0.48=0.02.
\]

Bernoulli 标准差在 \(\mu\approx0.5\) 时最大，而真实均值差只有 \(0.02\)。经验均值的普通随机波动很容易暂时颠倒臂排序。

识别样本复杂度通常包含类似：

\[
\frac{1}{\Delta_i^2}
\]

的依赖，因此小 gap 会显著增加区分成本。

需要区分：

```text
统计识别难
≠
每次选错损失大
```

hard-gap 中选错一次只损失 \(0.02\) 或 \(0.05\)，所以绝对 final regret 可以低于 easy-gap 的 Random regret。

---

## 9. UCB 为什么早期探索明显？

UCB1 的典型 index：

\[
\operatorname{UCB}_i(t)
=
\hat\mu_i(t)
+
\sqrt{\frac{2\log t}{N_i(t)}}.
\]

当 \(N_i(t)\) 较小时，bonus 较大。算法会把“样本不足”解释为仍然存在潜在乐观空间，因此主动回访该臂。

UCB 的探索不是额外加入一个随机动作，而是 index 中的 uncertainty bonus 产生的确定性探索。

初始化阶段通常还会显式覆盖所有臂，防止 \(N_i=0\) 和除零。

---

## 10. Thompson Sampling 与 seed 波动

Beta–Bernoulli TS：

\[
\theta_i^{(t)}
\sim
\operatorname{Beta}(\alpha_i,\beta_i),
\qquad
A_t=\arg\max_i\theta_i^{(t)}.
\]

即使后验参数相同，不同内部采样也可能产生不同 action。

早期成功和失败会影响后验，后验又影响后续采样和数据收集，因此具有路径依赖。

当前结果：

- easy-gap 中 TS 标准差仅约 \(7.43\)；
- hard-gap 中 TS 标准差约 \(23.18\)。

所以当前证据支持：

> TS 在难区分环境中可能出现较明显的 seed 波动。

不支持：

> TS 在所有环境里一定比 UCB 波动更大。

---

## 11. UCB-V 与 Bernstein 型优势

UCB-V 使用类似：

\[
\hat\mu_i
+
\sqrt{
\frac{2\hat V_i\log t}{N_i}
}
+
\frac{3b\log t}{N_i}
\]

的 index。

与只使用奖励范围的 Hoeffding 型 bonus 相比，经验方差 \(\hat V_i\) 较小时，第一项可能更紧。

潜在优势条件：

- reward 有界；
- 实际方差明显低于最坏情况；
- horizon 足够长；
- 经验方差估计已经稳定；
- \(3b\log t/N_i\) 不再主导。

### 11.1 为什么 easy-gap 中 UCB-V 反而更差？

在 easy-gap 中，UCB-V 的 mean final regret
略高于 UCB1：

```text
UCB1  = 81.800
UCB-V = 85.170
```

该差异本身不足以支持稳定优劣判断。
UCB-V 的 empirical-variance bonus 并不保证在所有有限 horizon
上都优于 Hoeffding-style UCB1。

### 11.2 Hard-gap 中是否证明 UCB-V 更好？

hard-gap 均值：

```text
UCB-V = 123.682
UCB1  = 143.021
```



UCB-V 在该配置下出现更低均值信号，但 10 seeds 不足以建立普遍算法排名。

---

## 12. Different-horizon 是否支持理论直觉？

实验保持同一个 hard-gap Bernoulli 环境：

```text
arm_means = [0.45, 0.48, 0.50]
```

只改变 horizon：

```text
1000 → 5000 → 20000
```

因此最大 horizon 是最小 horizon 的 20 倍。

### Random

Random 的 mean final regret 为：

\[
23.105
\rightarrow
116.103
\rightarrow
466.339.
\]

当 horizon 从 \(1000\) 增长到 \(20000\) 时，final regret 大约也增长 20 倍。

这与 Random policy 的理论直觉一致：由于它始终以固定概率选择各个 arm，每轮承担近似固定的期望 pseudo-regret，因此 cumulative regret 应近似随 \(T\) 线性增长。

### Thompson Sampling

Thompson Sampling 的 mean final regret 为：

\[
15.417
\rightarrow
38.762
\rightarrow
73.858.
\]

虽然 horizon 增长了 20 倍，但 final regret 的增长明显慢于 Random。

### UCB1

UCB1 的 mean final regret 为：

\[
19.454
\rightarrow
77.064
\rightarrow
236.351.
\]

其 final regret 同样随着 horizon 增长，但增长速度显著慢于 Random 的近线性增长。

### UCB-V

UCB-V 的 mean final regret 为：

\[
19.391
\rightarrow
72.203
\rightarrow
181.978.
\]

在当前三个 horizon 上，UCB-V 也表现出明显慢于 Random 的 empirical regret growth。

### 小结

因此，当前 different-horizon benchmark 定性支持以下理论直觉：

```text
Random:
approximately linear cumulative-regret growth

Learning algorithms:
substantially slower empirical regret growth
```

不过，当前实验**不能据此验证严格的渐近 regret rate**，例如：

\[
R_T = O(\log T).
\]

原因包括：

- 当前仅测试了 \(3\) 个 horizon；
- 每个 horizon 仅使用 \(10\) 个 random seeds；
- finite-time constants 仍然可能显著影响曲线；
- 理论 regret bound 通常是上界，而不是实际 regret 的精确表达式；
- 当前没有对 regret 与 \(\log T\)、\(\sqrt{T}\) 等候选增长形式进行 asymptotic fitting。

因此，这组实验的定位应当是：

> 验证实现结果是否与基本 regret 理论直觉相符，而不是通过有限规模实验“证明”某个渐近 regret bound。



## 13. 异常排查

### 13.1 确认存在的 runner bug

错误代码：

```python
algorithms = [
    config["algorithms"]
]
```

问题：

- normalized `config["algorithms"]` 已是 list；
- 再包装会形成 nested list；
- `algorithm_config["name"]` 对 list 使用字符串索引；
- 抛出 `TypeError`。

正确形态：

```python
algorithms = list(config["algorithms"])
```

状态：

```text
已修复并完成验证；
config_system_smoke 已成功运行 8 个实验；
多算法、多 horizon、多 seed 调度正常。
```

### 13.2 pytest collection error

直接运行：

```bash
pytest -q
```

曾报：

```text
No module named 'envs'
No module named 'algorithms'
```

改用：

```bash
python -m pytest -q
```

后测试正常收集并通过。

结论：

- 属于 import path / launcher 问题；
- 不属于算法测试失败；
- canonical test command 已固定。

### 13.3 UCB-V easy-gap 落后

当前可以由保守 finite-sample correction 解释，且其他 sanity checks 正常。因此暂不判为 bug。

若后续出现以下迹象，应重新审计：

- empirical variance 为负；
- counts 与 action counts 不一致；
- reward-square sum 更新遗漏；
- bonus 在 counts 增加时反而系统性增大；
- initialization 未覆盖所有臂；
- cumulative regret 减少。

### 13.4 TS 表现过强

easy-gap 中最优臂与次优臂具有较清晰的 gap，Beta posterior 可以相对较快地排除明显次优臂。Random 数值和 action frequency 均正常，暂时没有证据说明 TS 更新错误。

---

## 14. 当前可信结论

1. Random regret 的解析值与实验值高度一致。
2. easy-gap 中算法能快速锁定最优臂。
3. hard-gap 明显增加识别和持续探索成本。
4. UCB 的早期探索来自 confidence bonus。
5. TS 的 seed 波动具有环境依赖性。
6. UCB-V 的方差优势可能被早期修正项抵消。
7. 学习算法增长明显慢于 Random。
8. 当前有限实验不能证明具体 regret bound。
9. 当前有限实验不能给出普遍算法排名。
10. 尚未发现需要用文字掩盖的明确算法公式级异常。

---

## 15. Limitations

当前 v1.0 benchmark 仍有以下限制：

- 每组正式 benchmark 仅使用 \(10\) 个 random seeds；
- benchmark environment 数量仍然有限，主要集中于两个 stationary Bernoulli settings；
- different-horizon benchmark 只覆盖 \(3\) 个 horizon；
- 当前未进行 paired statistical significance test；
- 当前尚未系统扫描 reward variance 对算法表现的影响；
- `GaussianBandit` environment 已实现，但正式算法 compatibility 仍然有限，因此未纳入 v1.0 canonical benchmark；
- 当前 benchmark 的主要目标是验证定性理论预期，而不是拟合或实证验证严格的渐近 regret rate；
- 当前尚未记录完整的硬件环境和 benchmark wall-clock runtime 元数据；
- 最终的 fresh-environment release audit 尚未执行。

## 16. v1.0 发布前剩余动作

- [x] config system smoke validation；
- [x] UCB-V regression tests；
- [x] environment compatibility tests；
- [x] `python -m pytest -q` 全部通过；
- [x] stale-result protection；
- [x] canonical benchmark 从空输出目录重新运行；
- [x] CSV 数量与 figures 自动验收；
- [x] benchmark summary 自动生成；
- [x] benchmark report 与 canonical configs 对齐；
- [ ] README、CONTEXT、PROGRESS 最终同步；
- [ ] 删除 obsolete development scripts；
- [ ] repository hygiene audit；
- [x] fresh virtual environment reproduction；
- [x] `git diff --check`；
- [ ] working tree clean；
- [ ] merge release branch；
- [ ] annotated tag `v1.0`；
- [ ] GitHub Release。

## 17. 当前总结

当前结果整体支持最初的理论预期：

```text
Random 不学习
    → 每步期望 regret 固定
    → cumulative regret 近似线性

gap 变小
    → 经验均值更难区分
    → 探索成本与 seed 波动增加

UCB
    → uncertainty bonus 驱动探索

UCB-V
    → 利用方差，但 early correction 可能更保守

Thompson Sampling
    → posterior sampling，具有路径依赖

有限 benchmark
    → 支持定性预期
    ≠ 证明渐近 bound 或普遍排名
```

项目当前已完成“环境—算法—实验—结果—分析”的主体闭环，并已完成 canonical benchmark 的自动化复现。正式 v1.0 目前主要剩余文档同步、repository hygiene、fresh-environment release audit 与 GitHub release 管理。
