# E8 固定数据流实验

适配用户上传的三个 estimators 源码文件；本补充包不覆盖它们。

运行环境：项目 .venv。依赖入口：`python -m pip install -r requirements-d8.txt`。
已安装的 SciPy 不强制升级；具体版本写入实验 versions.json。

在项目根目录只运行一次：

```bash
python scripts/d8_batch_reference.py
```

入口先运行一次全套 pytest（最多180秒），失败即停止拟合并归档失败信息。
新增35项测试覆盖有限差分、求和与正则、极端logit、独立内点参照、边界状态、
迭代失败、伪成功拒绝及输入隔离。测试数量以实际输出为准。
边界测试允许特定SciPy版本报告失败，但必须忠实拒绝；既有用户边界成功记录仍保留。
正式五个数据前缀必须全部 accepted=True。

实验参数：d=2、K=5、T=200，seed=20260928；均匀随机动作；
前缀10/25/50/100/200；32个固定单位圆评估点。
目标为损失求和，l2=1、radius=2、ftol=1e-12、maxiter=500。
函数默认ftol仍为1e-10；独立最优性容差保持1e-5。
每个前缀从零初始化，无自动重试或动态调参。

输出位于唯一 `results/d8_batch_<UTC>/`，完整复制到 `reports/d8/<UTC>/`；
后者可正常纳入Git。另生成 `results/d8_batch_<UTC>_evidence.zip` 供反馈。
成功和失败都会保留数据、原始结果、配置、源码快照、哈希及诊断信息。
`manifest_sha256.json` 覆盖输出文件（不包含清单自身）。

后续OMD必须直接读取stream.npz中的X/y和eval_features.npy。
真参数与真实预测概率只用于生成/评估，不能传入估计器。
误差不要求逐前缀下降；零参数基线不是必须超越的验收门槛。
单次耗时仅为描述，不作为复杂度证明。

PASS仅表示自动测试与固定流拟合通过；用户解读及提交推送需另行完成。
请先返回终端摘要和evidence.zip，避免在未读结果前重复运行。
