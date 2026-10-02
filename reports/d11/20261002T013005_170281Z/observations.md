# E11 平稳闭环与计算记录

配置：d=3，K=10，T=1000，5 seeds；S=2，eta=3，lambda=126，delta=0.05。
使用paper_v2_literal原式版本，未调整置信系数。PDF正文p6的beta缺少附录A.1 p14出现的时间因子，保持差异记录；不声称解决了理论歧义。

| 算法 | 最终伪遗憾均值 | SEM |
|---|---|---|
| random | 254.444563 | 1.361251 |
| omd_paper_v2_literal | 69.271133 | 1.114940 |

- 阴影为5 seeds的SEM，不是95%置信区间；单个固定动作集不能支持普遍排名。
- 两策略的环境/动作随机源分离；相同种子不等于收到同一观测流，因为选臂不同。
- Smoke重复用于轨迹一致性；正式10个episode没有逐一重跑。
- 平稳闭环只验证当前固定参数环境，不延伸为非平稳保证。
- Profiling固定d/K，三个长度各3次；包装函数本身有计时开销，不作为严格复杂度证明。
- update_total含梯度、曲率、线性求解、投影、验证与状态赋值；linear_solve为其子集；update_other=total-linear_solve。
- projection_inclusive包含投影内部线性求解，与linear_solve有重叠，不能相加。
- selection含beta、输入检查、评分和argmax；日志计时为CSV行格式化和缓冲写入，不含最终flush/close与归档。
- 状态数组theta+H为96字节，不含Python对象、计数器、临时求解工作区、环境与外部实验日志，不是峰值内存。
- Random的0字节表示没有估计器数组，并非策略实际内存为零。
- 所有更新异常停止并保留现场，不重试、不改参数、不静默回退。

自动计算验收已完成；用户解读与最终commit/push另行记录。
