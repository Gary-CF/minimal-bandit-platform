# E10 固定数据流对照

从项目根目录运行一次：

```bash
python scripts/d10_fixed_stream.py --d8 reports/d8/20261001T070031_381809Z
```

本包只新增实验驱动与说明，不覆盖估计器或测试文件。
依赖已有 `scripts/d8_batch_reference.py` 中的归档辅助函数。
先验证E8哈希和验收状态，保留输入副本，再运行一次全套pytest。
随后将原始X/y逐个交给OMD，在10/25/50/100/200样本处评估。
Batch在相同五个前缀各重新求解一次，以记录当前环境耗时；不重新生成数据。

OMD使用eta=2、H1=I、radius=2。这是固定工程配置，不声称具备论文定理保证。
Batch保留E8配置。两者的正则机制不同，不能要求参数估计相等。
真参数与真实概率只用于外部评估。

输出：summary.csv、predictions.csv、updates.csv、五份Batch原始结果、
OMD前缀状态、observations.md、测试日志、源码快照和哈希。
失败停止并保留证据，无自动重试或调参。结果复制至reports/d10，另提供证据zip。

时间口径：OMD每步update包含验证/投影/曲率/诊断；Batch每次fit包含验证/求解/诊断。
评估和CSV写入不计入二者。累计OMD成本和单次Batch成本必须分开解读。
Batch仅五次调用，不能描述为每轮重拟合。
数组载荷不是峰值内存；本小实验不能证明复杂度或普遍速度优势。

运行后返回控制台输出和EVIDENCE_ZIP。结果解读后再提交E9/E10源码、测试及证据。
随后进入E11乐观选臂与平稳闭环。
