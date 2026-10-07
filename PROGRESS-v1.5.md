# Bandit v1.5 进度与验收

更新：2026-09-06。**代码与正式实验已完成发布验收；用户本地合入、最终提交、v1.5 tag 和远端发布尚未执行。**

## 本次补齐

- [x] 拒绝同一配置重复算法名，避免不同参数静默覆盖 CSV；重复 seed/horizon 同时拒绝。
- [x] 严格整数、有限数、未知参数键和必填参数校验；合法旧配置保留。
- [x] 在任何输出写入前，对整个 batch 检查构造、兼容性与模型约定。
- [x] Gaussian 环境与 TS 的 NaN/Inf 校验；Gaussian TS 公共噪声匹配、Gaussian UCB 标准差上界、UCB-V 范围校验。
- [x] Bernoulli KL q=0/1 数学端点和有限 budget；学习策略非法动作/奖励在更新前拒绝。
- [x] 新 v1.5 Bernoulli easy/hard、Gaussian 同方差配置，覆盖十个不同策略。
- [x] 安全复现入口、完整 CSV 验证、统计、均值±SEM 图、源码/依赖/输出 manifest。
- [x] 在全新隔离 venv 中安装依赖，生成 requirements-lock.txt，pip check 通过。
- [x] 263 项测试全部通过。
- [x] 正式 420 runs、840000 rows、42 summary rows、6 figures 全部通过；本次流水线约 82.182 秒。
- [x] 两次完整运行的 420 个 CSV 与 3 个快照 SHA256 一致；旧 14 份 basic/smoke 配置按预期运行。
- [x] 图表检查与结果解读；README/CONTEXT/讲解稿同步；MIT LICENSE 与 RELEASE 操作说明。

## 实验设置与证据

| 场景 | 策略数 | T | seeds | runs |
|---|---:|---|---|---:|
| Bernoulli easy-gap | 8 | 1000、3000 | 0–9 | 160 |
| Bernoulli hard-gap | 8 | 1000、3000 | 0–9 | 160 |
| Gaussian std=1 | 5 | 1000、3000 | 0–9 | 100 |

固定 ε=0.1、ETC 每臂 m=20；没有事后超参数搜索。准确数值以 `reports/v1_5/benchmark_summary_v1_5.csv` 为准；报告、图及 manifest 位于同目录。旧版四策略的结果依然属于 v1.0 历史实验。

本次新隔离环境：Python 3.12.13，NumPy 2.5.2，Matplotlib 3.11.1，pytest 9.1.1。本地重新运行请安装锁定版本；不能把上一轮旧运行环境的版本混写进新实验。

## 仍需用户执行的发布动作

- [ ] 将补丁应用到自己的实际仓库，保留本地额外改动，检查 diff。
- [ ] 在本地运行测试/复现和 verify-only，确认实际机器可用。
- [ ] 提交、合并到目标分支，创建 v1.5 tag 并推送。
- [ ] 创建 GitHub Release；确认 MIT 许可证符合自己的开放方式。

具体命令见 RELEASE.md。本次未访问远端，也未替用户合并、推送或创建标签。

## 保留的限制

有限场景与 seed 不构成普遍排名或渐近证明；普通 runner 不提供原子 batch 或 checkpoint；正式 suite 遇中断会保留失败目录，需要新目录重跑；未实现同批次同名算法参数扫描、contextual 或 non-stationary bandit。

旧 reproduce.sh 有全目录清理行为，v1.5 使用新入口。下一阶段无需为了完成 v1.5 再加第 11 个算法。
