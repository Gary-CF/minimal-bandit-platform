# v2.2 剩余实验：一次执行

在项目根目录运行：

```bash
python -m pip install -r requirements-v2_2-remaining.txt
python -m scripts.reproduce_v2_2 --output artifacts/v2_2_remaining_r1 --jobs 4 --frozen-test-run artifacts/v2_2_frozen_test_r1
```

依次执行单元测试、旧 315 局只读复核、71 项 smoke、1332 项正式补充实验、汇总及旧测试报告。旧测试目录缺失时省略 `--frozen-test-run`。没有后台偷偷启动的任务；命令退出成功且打印 `ALL REMAINING EXPERIMENTS PASS` 才算完成。

中断后执行相同命令。已完成任务先核验再复用，未完成任务重跑，失败现场留在 `attempts/`。源码或协议改变会拒绝混用结果；使用新输出目录。程序拒绝两个进程同时写同一输出目录。

只读复核（不重新运行实验）：

```bash
python -m scripts.reproduce_v2_2 --output artifacts/v2_2_remaining_r1 --verify-only --frozen-test-run artifacts/v2_2_frozen_test_r1
```

正式结果在 `artifacts/v2_2_remaining_r1/full/`，其中 `report.md`、`aggregate.csv`、`by_feature.csv` 和三张图便于阅读；逐局原始数组在 `episodes/`。

## 预先固定的实验量

| 阶段 | 设计 | 次数 |
|---|---|---:|
| Logistic | 14 个 K/场景/变化幅度组合 × 3 特征集 × 3 奖励种子 × 7 算法，T=1000 | 882 |
| 重尾固定流 | 3 噪声 × 3 特征集 × 3 种子 × 3 估计器，T=1000 | 81 |
| 重尾闭环 | 3 噪声 × 3 特征集 × 3 种子 × 3 场景 × 4 策略，T=1000 | 324 |
| 开销审计 | 5 算法 × T∈{200,1000,5000} × 3 次独立重复 | 45 |
| 总计 | 不含 smoke 和旧 315 局 | 1332 |

Logistic：K=10/30/100，d=3；每个 K 有 stationary、abrupt、rotation（总转角 π）。另外 K=10 加入原弦插值 drift、abrupt 转角 π/4 与 π/2、rotation 总转角 π/2 与 2π。角度与 K 分组报告，不随意混成一个排名。新的特征种子 20261130、20261201、20261202；奖励种子 300–302；开销种子 310–312。

沿用冻结参数：SW W=1000、Discounted-UCB γ=.999、DOMD γ=.98、SCB γ=.98。补充结果不再用于调参。保留三特征集分别的结果，误差条按特征组均值计算，不能把全部局视作独立特征任务。

## 轨迹和重尾口径

固定 θ₀=(1,−.5,.75)。由 u=θ₀/‖θ₀‖ 与确定性正交单位向量 v 构造

θₜ=‖θ₀‖[cos(ωₜ)u+sin(ωₜ)v]，ωₜ 从 0 线性增长到预设角度。

这保持参数范数，避免 θ→−θ 的直线插值穿过零；它不保证各臂奖励间距恒定。abrupt 在第 T//2+1 步切换到对应圆弧终点。

线性奖励 r=xᵀθ+ξ，不做 Bernoulli 化，也不裁剪环境奖励。高斯标准差 .5；Student-t(3) 乘 .5/√3，方差同为 .25；Student-t(1.5) 乘 .5，保留无限方差，选 ε=.25 使用有限 1.25 阶矩。噪声的解析矩界作为 Hvt 已知输入公开在协议中。

固定流用相同外生动作和相同噪声比较 ridge、Hvt 和 Hvt-squared。后者保留 Hvt 的归一化、度量和投影，只去掉梯度裁剪，是机制消融，不享有 Hvt 保证。闭环比较 random、ridge-greedy、LinUCB nominal、Hvt literal。LinUCB 使用固定高斯标称噪声 .5，不声称重尾保证。

Hvt 的 stationary 实验符合该部分模型假设；abrupt/rotation 是超出定理的压力测试。采用论文理论常数，包括 107，有限时域表现不一定好；如实保留结果，不根据测试表现偷偷调小置信半径。

## 开销与可复核性

质量实验按 `--jobs` 并发加速；全部结束后，开销实验逐个在新 Python 进程中执行，BLAS 单线程。早期/中期/晚期各取 100 步（T=200 时取 66 步），计时不含预测诊断、磁盘写入或审计回放。

独立记录：策略 NumPy 状态字节、保留历史字节、静态臂特征字节、外部轨迹文件字节、进程峰值 RSS。RSS 包含导入模块、环境真值和结果缓存，不能解释为策略单独占用。

开销轨迹再进行不计时的确定性回放，核对动作，统计 MLE Newton 次数、损失/梯度/Hessian 历史行评估量、SCB SLSQP 调用及迭代数、度量球投影线性方程求解次数。历史行评估量仅统计指定计算入口，不等于所有矩阵乘法的数据扫描次数。

每项任务写入完整临时目录并核验后原子提交。每个归档包含源码快照、协议、版本、任务/轨迹/汇总校验和。验证器重建环境真值与奖励随机流，重算遗憾、RMSE 和汇总；它不声称对所有闭环策略做动作重演，只有开销审计做了该项检查。

从归档源码复核时，可在 `full/source/` 下运行 `python -m scripts.run_remaining_v2_2 --output <full目录绝对路径> --verify-only`。重做独立复现应使用新的输出目录，并使用记录的 Python/依赖版本。

## 实验完成后的收尾

本入口补齐实验，不自动替你宣称个人讲解验收已完成，也不创建 Git tag。最终仍需结合旧报告与补充报告写结论，检查 Git diff，保留用户自己的 v2.1 报告修改，完成独立复现和讲解后再发布 v2.2。
