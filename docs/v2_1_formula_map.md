# Logistic OMD 公式与代码

论文依据：Generalized Linear Bandits: Almost Optimal Regret with One-Pass Update，arXiv:2507.11847v2，固定使用已核对版本。

| 公式/对象 | 代码位置 | 约定 |
|---|---|---|
| ell(theta)=log(1+exp(x^T theta))-y x^T theta | envs/logistic_bandit.py: logistic_loss | logaddexp 稳定计算；无额外截距 |
| g=(sigmoid(x^T theta)-y)x | logistic_gradient | 当前观测；旧 theta |
| G=sigmoid(z)(1-sigmoid(z))xx^T | logistic_hessian | 局部模型在旧 theta；累积在新 theta |
| A=H+eta G，u=theta-eta A^{-1}g | estimators/omd_logistic.py: omd_local_model | solve，不显式求逆 |
| min_{||v||<=S} (v-u)^T A(v-u) | project_metric_ball | 内点直接返回；边界求 rho>=0 |
| v(rho)=(A+rho I)^{-1}Au | metric_ball_candidate | 先括区间，再二分，使 ||v||=S |
| H_next=H+G(theta_next,x) | omd_step | 新 theta，无 eta 因子 |
| x_a^T theta + beta sqrt(x_a^T H^{-1}x_a) | algorithms/logistic_omd.py: optimistic_scores | 单调链接下按乐观 logit 排序；并列首臂 |
| eta=1+S；lambda=max(14 d eta,1.5 eta S) | LogisticOMDPolicy.__init__ | Logistic 的 R=1、L_mu=1/4、g=1 |
| beta²=4lambda S²+2eta log(1/delta)+d(6eta²+eta)log(1+0.25/lambda) | confidence_radius | paper_v2_literal 原式；不缩放 |

论文 PDF 第6页正文与第14页附录 A.1 的置信半径时间因子存在差异。默认模式原样保留正文；`appendix_time` 是单独标记的实现选项，本次正式实验没有使用它，也不宣称已解决理论歧义。

E10 的 eta=2、lambda=1 是固定工程对照；E11/E12 的 d=3、S=2 给出 eta=3、lambda=126。批量 L2 正则和在线 H 初始化不能视为同一目标。

E12 只在外部实验程序中生成真参数路径，逐轮复用已验证 LogisticBernoulliBandit 采样；策略只见 features 与所选臂 reward。动态伪遗憾比较每轮真均值最优臂，预测指标在更新前计算。
