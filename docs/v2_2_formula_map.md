# v2.2 核心公式与代码映射

## Logistic

| 算法 | 更新与选择 | 代码 |
|---|---|---|
| OMD | A=H+η∇²ℓ(θ)；u=θ−ηA⁻¹∇ℓ(θ)；A 度量球投影；H⁺=H+∇²ℓ(θ⁺) | `estimators/omd_logistic.py` |
| DOMD | 先令 H̃=γH+(1−γ)λI，再调用同一 OMD 步；选臂采用该策略自己的置信半径 | `estimators/domd_logistic.py`, `algorithms/logistic_domd.py` |
| SCB | 折扣 Logistic MLE、原论文非凸投影目标、折扣设计矩阵和 UCB | `estimators/discounted_logistic_mle.py`, `algorithms/scb_weight_ucb.py` |

OMD/DOMD 完整策略的 λ 和置信半径也不同，二者闭环差不能全部归因于遗忘。SCB 多起点 SLSQP 仅验证局部 KKT，不保证全局最优，因此名称保留 numeric。

## Hvt-UCB

依据 Wang et al., ICML 2025，arXiv:2503.00419v2 Algorithm 1、Lemma 1、Theorem 1：
https://arxiv.org/html/2503.00419v2

代码：`algorithms/hvt_ucb.py` 的 `HvtUCB`。

输入 T,d,S=2,L=1，以及 ε∈(0,1] 和噪声中心矩尺度 ν，满足 E|ξ|^(1+ε)≤ν^(1+ε)。设 α=4、λ=d、δ=1/(8T)、σ_min=1/√T。

令 p=(1−ε)/(2(1+ε))、κ=d log(1+L²T/(4σ_min²λd))：

\[
\tau_0=\frac{\sqrt{2\kappa}(\log 3T)^p}{(\log(2T^2/\delta))^{1/(1+\varepsilon)}},\qquad
\beta_n=107\log(2T^2/\delta)\tau_0 n^p+\sqrt{\lambda(2+4S^2)}.
\]

初始 θ₁=0,V₀=λI；ε=1 时 n^p 恒为 1（含 n=0），ε<1 时 0^p=0。第 t 步用 θₜ、Vₜ₋₁、βₜ₋₁ 选择 x，使 xᵀθₜ+βₜ₋₁√(xᵀVₜ₋₁⁻¹x) 最大。

\[
\sigma_t=\max\left\{\nu,\sigma_{\min},
 \sqrt{\frac{2\beta_{t-1}}{\tau_0\sqrt\alpha\,t^p}}\|x_t\|_{V_{t-1}^{-1}}\right\},\quad
w_t=\frac{\|x_t\|_{V_{t-1}^{-1}}}{\sqrt\alpha\sigma_t},\quad
\tau_t=\tau_0\frac{\sqrt{1+w_t^2}}{w_t}t^p.
\]

\[
z_t=(r_t-x_t^\top\theta_t)/\sigma_t,\quad
g_t=-\operatorname{clip}(z_t,-\tau_t,\tau_t)x_t/\sigma_t,
\quad V_t=V_{t-1}+\frac{x_tx_t^\top}{\alpha\sigma_t^2}.
\]

\[
u_t=\theta_t-V_t^{-1}g_t,\qquad
\theta_{t+1}=\arg\min_{\|\theta\|_2\le S}\|\theta-u_t\|_{V_t}^2.
\]

先更新 V，再求梯度步；不能误用旧 V。梯度按**旧参数**计算，裁剪的是归一化残差，不能直接裁剪奖励。投影调用现有 `project_metric_ball`，用 (V+ρI)θ=Vu 和球边界确定 ρ，包含可行性与 KKT 检查。

实现用线性方程求解，不显式缓存逆矩阵；固定 d 时仍不随历史长度增长，但不宣称实现了 Sherman–Morrison 的 d² 复杂度。Hvt 不保存历史样本。论文的自适应 σ、τ、置信半径不能替换成固定阈值 Huber-SGD 后仍称为同一算法。

## 检验对应

`tests/test_remaining_v2_2.py`：与独立受约束凸优化器比较单步 Hvt 更新和投影；直接计算 β、V 累积；数值积分核对 Student-t 中心矩；验证旋转范数、切换边界；核对固定流配对、断点续跑、损坏拒绝及历史评估计数。
