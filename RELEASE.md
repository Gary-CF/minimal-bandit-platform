# Bandit v2.2 发布

本版增加 DOMD、SCB numerical、Hvt-UCB、统一验证/冻结测试、旋转与动作集扩展、重尾消融和时间/存储审计。核心结论及负结果见 README 和最终报告。

## 本机一键收尾并发布

```bash
bash scripts/publish_v2_2.sh
```

默认读取 `artifacts/v2_2_frozen_test_r1` 和 `artifacts/v2_2_remaining_r1/{smoke,full}`。路径不同可传：

```bash
bash scripts/publish_v2_2.sh --frozen-test-run PATH_TO_FROZEN --remaining-run PATH_TO_REMAINING
```

脚本执行测试、只读复核并生成精简报告。只暂存 `configs/v2_2/release_files.json` 和生成报告清单内的路径，保留原 v2.1 报告等其他未列出的修改。已有暂存内容时停止，避免混入其他工作。

发布分支为 `dev/v2.2-bandit`，目标为 `Gary-CF/minimal-bandit-platform`。发布前检查远端 main 是当前 HEAD 的祖先；不能快进时停止，不重写历史。提交后创建 annotated `v2.2` 标签，原子推送 main、开发分支和标签，随后核对远端引用。

推送认证失败时，本地提交和标签保留。完成 GitHub 登录后执行：

```bash
git push --atomic origin HEAD:refs/heads/main HEAD:refs/heads/dev/v2.2-bandit refs/tags/v2.2
git ls-remote origin refs/heads/main refs/heads/dev/v2.2-bandit refs/tags/v2.2 'refs/tags/v2.2^{}'
```

main 受保护时，原子推送会拒绝整个更新；按仓库规则走 PR，不强推。已有标签指向不同提交时停止核对，不移动标签。

## 只准备、不提交

```bash
python -m scripts.closeout_v2_2
```

输出最终报告、数据表、来源索引、测试记录及 `artifacts/v2_2_closeout/git-paths.txt`。不运行新实验、不做 Git 操作。大体积轨迹和源码快照留在本机 `artifacts/`，不将助手机器预跑文件混入本机正式发布。

历史：[v2.1 发布说明](RELEASE-v2.1.md)、[v1.5 发布说明](RELEASE-v1.5.md)。
