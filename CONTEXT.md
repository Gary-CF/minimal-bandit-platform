# Bandit v1.5 开发上下文

本项目服务于概率统计、在线学习、Bandit 和 RL 的理论与工程训练。目标是能解释假设、追踪数据流、验证实验，而不是搭建大型框架。用法见 README，执行证据见 PROGRESS。

## 版本和边界

v0.1 是教学最小闭环。v1.0 为 Random、UCB1、UCB-V、Bernoulli TS 四策略和两环境，形成实验发布流程。v1.5 在上传基线 `d02f8b5` 上完成六个新增策略与发布加固。

十个策略的名字、参数、兼容性矩阵以 README 和 run.py 为准。GaussianBandit 是环境。当前不包含非平稳或 contextual 模型，也不自动支持参数扫描。

## 核心接口

`BanditAlgorithm` 通过 `NotImplementedError` 表达接口，不是 ABC。接口：`select_action()->int`，`update(action,reward)->None`。每轮选一次、更新一次，每个独立实验新建对象，无 reset、checkpoint 或离线训练接口。

环境没有统一基类；runner 使用 BernoulliBandit/GaussianBandit union。接口是 `step(action)` 和 `pseudo_regret(action)`；`arm_means`/`best_mean` 只用于环境采样与评估，不传给算法。CSV step 从 1 开始，action 从 0 开始。

| 策略 | 关键状态 | 实现细节 |
|---|---|---|
| Random | RNG | update 不学习 |
| ε-Greedy | counts、reward_sum、estimated_reward | 固定 ε，零初始化，无强制逐臂覆盖 |
| ETC | counts、reward_sums、estimated_mean、committed_arm | 探索 Km 轮，下一次 select 才 commit；之后更新统计但不切臂 |
| UCB1 | counts、reward_sums、estimated_means | 初始化覆盖；log 使用 counts.sum()+1 |
| UCB-V | counts、reward_sums、reward_square_sums | 均值和经验方差为 property；方差截为非负；log 使用 max(counts.sum(),2) |
| MOSS | counts、reward_sums、estimated_means、horizon | 总预算进入公式，多 T 时每次重新注入 |
| KL-UCB | counts、reward_sums、estimated_mean、c | Bernoulli KL；40 次二分；端点明确处理 |
| Bernoulli TS | alpha、beta、RNG | 固定 Beta(1,1) 先验，不强制覆盖 |
| Gaussian UCB | counts、reward_sums、estimated_mean、known_std | std 为公共标准差上界 |
| Gaussian TS | counts、reward_sums、posterior_means/variances、RNG | 标量已知观测方差，要求环境同方差匹配 |

np.argmax 并列时取首个索引。多数 UCB 类的初始化依赖正常 select/update 配对，不保证任意乱序离线更新语义。ETC 当前每轮求 counts.sum()，即便已 commit，选择函数仍为 O(K)。

## 配置与工厂

`normalize_config` 规范化字符串算法、单 horizon，检查字段、真正整数、重复项和有限数；`preflight_config` 使用独立临时对象检查整个 batch 的构造、兼容性和模型参数，发生在任何快照/CSV 写入之前。

`create_algorithm` 接口仍为 `(algorithm_config,num_arms,rng,horizon)`。小型参数白名单用于防拼写错误，不扩展成插件框架。重复算法名直接拒绝，多参数比较分 experiment_name。

模型约定：UCB-V Bernoulli reward_range≥1；Gaussian UCB known_std≥每臂 std；Gaussian TS noise_variance 与全部 std² 在严格浮点容差内一致。无需自动从环境推断算法参数，这样已知噪声假设留在配置中可检查。

类入口补了 Gaussian 有限值、动作整数及学习策略奖励有限性校验；当前有界策略 reward 在 [0,1]。Random.update 本来无学习状态。运行时 assert 继续用于内部一致性检查，但 `python -O` 会移除 assert，因此不能代替入口异常校验。

## RNG 和评估

每次实验 `SeedSequence(seed).spawn(2)` 分离环境和策略 RNG。工厂预校验不消耗真实实验随机流。算法顺序不会影响各自 run；同 seed 不代表不同动作得到相同奖励。MOSS 依赖 T，不能要求多 horizon 的前缀一致。

pseudo-regret 用真实均值与所选动作计算，而不是用 realized reward。策略不能读取真实均值。runner 的“未最常选最优臂”是诊断 warning，短预算、Random 或并列最优都可能触发。

## 两层输出保证

普通 run.py：配置快照防不同配置混写；同配置允许重跑覆盖。仍无原子 batch、恢复标记、源码身份或完整集分析；中途异常可能留下部分 CSV。

正式 v1.5 suite：写入全新的输出根目录，先测试，后运行；manifest 状态 running/complete/failed；分析要求预期文件全集且所有记录一致，最终保存源码、配置、依赖、结果和报告哈希。只有所有步骤通过才 complete。非空目录不自动清理。

`--verify-only` 校验 manifest 完成状态、当前源码指纹、文件集合及结果/报告哈希，再检查 CSV 内容，不修改产物。源码身份覆盖算法、环境、runner、配置、scripts、tests、requirements 和 lock；文档文本不进入算法来源指纹。git_base_revision 是开发基线，准确发布源码以内容哈希为准。

图为均值±SEM，SEM=样本标准差/sqrt(n)，不是 95% CI。动作频率图从配置读 K，显示未访问臂。summary 数字必须自动计算，人工文字不作为权威数据。

## 如何继续改动

添加策略时改类、工厂/参数白名单、兼容性、必要的 runner 断言、测试、配置和文档。保持公共接口即可；无需引入 registry 或统一所有内部字段名。

改公式、默认参数、seed 调用顺序、tie-break 或正式配置后重跑 suite；不要只改报告。旧 v1.0 实验保留独立定义，新的版本使用独立输出路径。

后续非平稳扩展还需要环境时间演化、评估基准和动态 regret 设计，不能只修改 arm_means。优先保持当前可解释性。
