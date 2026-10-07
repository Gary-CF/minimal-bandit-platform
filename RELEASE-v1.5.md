# v1.5 合入与发布

本次基于 `d02f8b5` 开发，保留原工厂、接口、算法主公式和旧版实验，新增发布边界修复与独立 v1.5 实验链。补丁包含新增文件、文档、测试、正式配置与冻结报告。

## 在你原来的仓库应用补丁

进入真实 bandit-project 仓库，先检查工作区并保留自己的未提交工作。不要用本次源码包覆盖你的 `.git`。

```bash
git status --short
git switch feat/v1.5-algorithm-expansion
git rev-parse --short HEAD
git switch -c release/v1.5
git apply --check /实际下载路径/bandit-v1.5.patch
git apply /实际下载路径/bandit-v1.5.patch
```

补丁以原始 `d02f8b5` 的三份文档为基线。若你已覆盖上一轮生成的 README/CONTEXT/PROGRESS，或有其他代码变更，`--check` 可能报告冲突；不要强行 `--reject` 或 reset。此时打开完整源码包对照相应文件，保留自己的额外修改再整合。本次完整源码包不含 Git 历史或虚拟环境，可作为参照。

## 本地最后验收

使用 Python 3.12，位于原项目根目录：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pytest -q
python scripts/benchmark_v1_5.py --output-root artifacts/v1_5_local
python scripts/benchmark_v1_5.py --output-root artifacts/v1_5_local --verify-only
git diff --check
git status --short
```

以上输出根目录已存在且非空时，换新的目录名；不要为重跑删除其他实验。

验收预期：263 tests；420 runs；840000 rows；42 summary rows；6 figures；manifest 为 complete。依赖安装耗时和实验耗时依机器而异。本地重跑若使用不同依赖，数值不一定逐字节相同；不要伪造旧 manifest 来通过检查。

源码包已附 `reports/v1_5/` 的验证结果。自行重跑后如要采用本地结果，整套同步 summary、报告、6 张图与 suite_manifest，而不是只替换其中一部分数字。日志和原始 CSV 默认留在 artifacts，不提交。

## 提交、合并与标签

确认 diff 中只包含预期文件后，显式暂存本次变更（不要无检查地把个人文件一起提交）：

```bash
git add .gitattributes .gitignore algorithms envs run.py tests configs scripts/benchmark_v1_5.py reproduce_v1_5.sh requirements-lock.txt README.md CONTEXT.md PROGRESS.md RELEASE.md LICENSE reports
git diff --cached --stat
git diff --cached --check
git commit -m "Release v1.5: harden validation and add reproducible ten-policy benchmarks"
git switch main
git merge --no-ff release/v1.5
git tag -a v1.5 -m "Bandit Platform v1.5"
git push origin main
git push origin v1.5
```

这些命令供你审阅后在本地执行；本次交付未替你 merge/tag/push。若你的 main 已有额外提交或采用 PR 流程，应先通过 PR 合并后在对应提交打 tag，避免在不确定的提交上标版本。

仓库补充了 MIT 许可证，版权署名 Gary Chen。发布前确认它符合你的开放方式及仓库中所有内容的授权来源；本次未添加第三方实现源码。

## Release 描述草稿

Minimal Bandit Platform v1.5 adds six policies on top of the v1.0 baseline, bringing the total to ten: Random, UCB1, UCB-V, Bernoulli Thompson Sampling, fixed epsilon-greedy, ETC, MOSS, KL-UCB, Gaussian UCB, and Gaussian Thompson Sampling.

This release prevents configuration collisions and silent parameter mistakes, validates Gaussian noise assumptions and numeric boundaries, and adds an isolated reproduction pipeline. The suite covers two Bernoulli settings and one Gaussian setting with ten seeds and two horizons: 420 runs, 840,000 records, 42 summary rows, and six mean ± SEM figures. Regression tests and output integrity checks are included.

Results describe these fixed finite-budget configurations; they do not establish universal algorithm rankings or asymptotic regret rates. See README and reports/v1_5 for reproduction and evidence.
