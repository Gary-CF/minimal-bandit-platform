# v2.1 验收与冻结

在 `dev/v2.1-bandit` 上完成本阶段。本次从已验收的 `2c652ec` 开始，复用核心算法，仅增加压力测试、复现验证及文档。历史v1.5发布材料保留在 RELEASE-v1.5.md。

## 计算验收

```bash
source .venv/bin/activate
python scripts/reproduce_v2_1.py --output-root artifacts/v2_1_local
python scripts/reproduce_v2_1.py --output-root artifacts/v2_1_local --verify-only
```

输出目录须全新。成功会归档到 reports/v2_1/v2_1_local，提供 evidence ZIP。失败保留目录与failure.txt，不调参或自动重试；先定位具体问题。

预期：全套pytest通过，固定流200次OMD更新/5次Batch拟合，平稳10条正式轨迹，profiling 9次，压力测试20条轨迹，五张关键图，求解失败0。新环境中完整轨迹与历史E11是否逐字节相同会明确记录；数值差异必须解释，不能修改旧证据。

报告只引用本次数据；终端自动PASS不等于完成手动讲解或远端同步。

## 人工验收

按 docs/v2_1_walkthrough.md 用约10分钟独立讲解，回查不清楚的部分。阅读 report.md 中两类压力测试的窗口统计，能够区分现象、可能机制和替代解释。创建 `reports/v2_1/closeout.md`，记录证据目录、实际讲解结果、遗留项（无则写无）、目标版本与分支。

## 提交与冻结

确认处在 dev/v2.1-bandit，核对diff后显式暂存本包文件与本次归档。提交本地运行生成的归档，不提交助手机器上的演示数据。`git diff --cached --check`通过后commit并推送。

在无遗留项的已验收提交上执行：

```bash
git tag -a v2.1 -m "Bandit Platform v2.1: Logistic OMD and reproducible stress tests"
git push origin dev/v2.1-bandit
git push origin v2.1
git status --short --branch
git log -1 --oneline --decorate
```

已有v2.1标签时停止核对，不覆盖或强推。Git标签是不可移动的验收点；是否同时合入main是独立分支管理动作，不影响这条冻结记录。

若决定将本人main推进到该已验收版本，在工作区干净且已同步远端后，可用 `git switch main`、`git pull --ff-only origin main`、`git merge --ff-only dev/v2.1-bandit`、`git push origin main`，然后切回开发分支。若不能fast-forward，先审查分叉，不强制重写历史。此操作只针对本人的origin仓库。
