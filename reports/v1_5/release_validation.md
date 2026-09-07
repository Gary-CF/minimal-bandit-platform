# v1.5 发布验收记录

基线：d02f8b5。最终源码身份详见 suite_manifest.json 的 code_sha256。

- 全新隔离 Python 3.12.13 venv 安装依赖，pip check 通过；锁定完整安装版本。
- 原有 150 项 + 新增 113 项，共 263 项测试通过。
- 14 份历史 basic/smoke 配置：11 份正常运行，3 份非法组合按预期拒绝。
- 最终完整 suite：420 runs / 840000 rows / 42 summary rows / 6 figures；82.182 秒。
- 两次独立完整运行的所有 420 个原始 CSV 和 3 个配置快照 SHA256 完全一致；第二次统一汇总表 LF 换行。
- 分析执行前严格验证预期集合、配置、身份、步数、数值和 pseudo-regret。缺失/多余/截断/伪造 regret/seed/snapshot 的回归测试均拒绝。
- 三种场景的 T=3000 图已视觉核查，六张图均生成成功。
- 发布包只含源码、配置、测试、文档与精简图表证据，不含 Git 历史、虚拟环境或大量原始 CSV；原始输出另附证据包。

用户实际仓库的提交、合并、标签、推送和 GitHub Release 尚未执行；详见 RELEASE.md。
