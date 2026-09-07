# Bandit v1.5 正式实验结果

配置预先固定为 3 个场景、T=1000/3000、seeds=0–9；共 420 次运行、840,000 行记录。
ε=0.1；ETC 每臂探索 20 次；MOSS 使用当前 T；KL-UCB c=3；UCB-V 范围为 1。
Gaussian 场景均值 [0,0.2,0.5]、所有臂 std=1；Gaussian TS 先验 N(0,1)、观测方差 1。
图的阴影为均值 ± SEM，描述均值估计的抽样不确定性，既非单次运行波动区间，也非 95% 置信区间。

| 场景 | 算法 | T | seeds | 最终 regret 均值 | 样本标准差 | SEM |
|---|---|---:|---:|---:|---:|---:|
| v1_5_bernoulli_easy | random | 1000 | 10 | 231.050 | 5.601 | 1.771 |
| v1_5_bernoulli_easy | epsilon_greedy | 1000 | 10 | 58.370 | 20.111 | 6.360 |
| v1_5_bernoulli_easy | etc | 1000 | 10 | 14.000 | 0.000 | 0.000 |
| v1_5_bernoulli_easy | ucb1 | 1000 | 10 | 45.830 | 8.161 | 2.581 |
| v1_5_bernoulli_easy | ucb_v | 1000 | 10 | 55.240 | 7.381 | 2.334 |
| v1_5_bernoulli_easy | moss | 1000 | 10 | 18.650 | 7.331 | 2.318 |
| v1_5_bernoulli_easy | kl_ucb | 1000 | 10 | 24.240 | 6.572 | 2.078 |
| v1_5_bernoulli_easy | thompson_sampling | 1000 | 10 | 11.000 | 8.378 | 2.649 |
| v1_5_bernoulli_easy | random | 3000 | 10 | 700.090 | 14.040 | 4.440 |
| v1_5_bernoulli_easy | epsilon_greedy | 3000 | 10 | 103.230 | 19.406 | 6.137 |
| v1_5_bernoulli_easy | etc | 3000 | 10 | 14.000 | 0.000 | 0.000 |
| v1_5_bernoulli_easy | ucb1 | 3000 | 10 | 70.390 | 11.334 | 3.584 |
| v1_5_bernoulli_easy | ucb_v | 3000 | 10 | 75.960 | 6.524 | 2.063 |
| v1_5_bernoulli_easy | moss | 3000 | 10 | 21.300 | 7.087 | 2.241 |
| v1_5_bernoulli_easy | kl_ucb | 3000 | 10 | 32.360 | 8.219 | 2.599 |
| v1_5_bernoulli_easy | thompson_sampling | 3000 | 10 | 12.540 | 7.897 | 2.497 |
| v1_5_bernoulli_hard | random | 1000 | 10 | 23.105 | 0.560 | 0.177 |
| v1_5_bernoulli_hard | epsilon_greedy | 1000 | 10 | 16.049 | 6.482 | 2.050 |
| v1_5_bernoulli_hard | etc | 1000 | 10 | 23.020 | 19.341 | 6.116 |
| v1_5_bernoulli_hard | ucb1 | 1000 | 10 | 19.454 | 2.577 | 0.815 |
| v1_5_bernoulli_hard | ucb_v | 1000 | 10 | 19.391 | 3.466 | 1.096 |
| v1_5_bernoulli_hard | moss | 1000 | 10 | 13.685 | 5.528 | 1.748 |
| v1_5_bernoulli_hard | kl_ucb | 1000 | 10 | 18.214 | 4.800 | 1.518 |
| v1_5_bernoulli_hard | thompson_sampling | 1000 | 10 | 15.417 | 7.581 | 2.397 |
| v1_5_bernoulli_hard | random | 3000 | 10 | 70.009 | 1.404 | 0.444 |
| v1_5_bernoulli_hard | epsilon_greedy | 3000 | 10 | 43.467 | 19.403 | 6.136 |
| v1_5_bernoulli_hard | etc | 3000 | 10 | 69.020 | 60.491 | 19.129 |
| v1_5_bernoulli_hard | ucb1 | 3000 | 10 | 50.894 | 7.520 | 2.378 |
| v1_5_bernoulli_hard | ucb_v | 3000 | 10 | 48.653 | 6.777 | 2.143 |
| v1_5_bernoulli_hard | moss | 3000 | 10 | 35.658 | 14.630 | 4.626 |
| v1_5_bernoulli_hard | kl_ucb | 3000 | 10 | 42.384 | 11.505 | 3.638 |
| v1_5_bernoulli_hard | thompson_sampling | 3000 | 10 | 30.879 | 15.660 | 4.952 |
| v1_5_gaussian | random | 1000 | 10 | 263.950 | 6.000 | 1.897 |
| v1_5_gaussian | epsilon_greedy | 1000 | 10 | 67.300 | 84.947 | 26.863 |
| v1_5_gaussian | etc | 1000 | 10 | 72.400 | 118.902 | 37.600 |
| v1_5_gaussian | gaussian_ucb | 1000 | 10 | 40.970 | 10.532 | 3.330 |
| v1_5_gaussian | gaussian_thompson_sampling | 1000 | 10 | 28.610 | 17.585 | 5.561 |
| v1_5_gaussian | random | 3000 | 10 | 799.810 | 12.664 | 4.005 |
| v1_5_gaussian | epsilon_greedy | 3000 | 10 | 144.110 | 161.523 | 51.078 |
| v1_5_gaussian | etc | 3000 | 10 | 192.400 | 371.884 | 117.600 |
| v1_5_gaussian | gaussian_ucb | 3000 | 10 | 56.010 | 14.041 | 4.440 |
| v1_5_gaussian | gaussian_thompson_sampling | 3000 | 10 | 42.220 | 23.032 | 7.283 |

## Random 解析核对

| 场景 | T | 理论期望 | 实验均值 |
|---|---:|---:|---:|
| v1_5_bernoulli_easy | 1000 | 233.333 | 231.050 |
| v1_5_bernoulli_easy | 3000 | 700.000 | 700.090 |
| v1_5_bernoulli_hard | 1000 | 23.333 | 23.105 |
| v1_5_bernoulli_hard | 3000 | 70.000 | 70.009 |
| v1_5_gaussian | 1000 | 266.667 | 263.950 |
| v1_5_gaussian | 3000 | 800.000 | 799.810 |

## 解释边界

这些是固定参数下的有限预算比较。ETC 的 m 未按 gap 调优，ε-Greedy 的 ε 为常数；不应将它们的表现解释为算法家族的最优性能。
小 gap 场景更难识别，但每次选错的损失更小，跨场景绝对 regret 不宜直接排名。
MOSS 明确获知 T；其不同预算实验不能理解为同一条轨迹截断。Gaussian TS/ UCB 使用已知噪声，这是额外建模假设。
10 seeds 和两个预算不支持普遍优劣、统计显著性结论或严格渐近速率证明；没有进行超参数搜索或配对显著性检验。
源码与依赖、运行时长及输出 SHA256 记录在输出根目录 suite_manifest.json。
