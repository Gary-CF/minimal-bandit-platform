from __future__ import annotations

import numpy as np

class GaussianBandit:
    """
    Gaussian 多臂老虎机环境。

    第i个臂的奖励满足：

        X_{t,i} ~ N(mu_i,sigma_i^2)

    其中：
    - arm_means[i] 是第i个臂的均值 mu_i
    - arm_stds[i] 是第i个臂的标准差 sigma_i
    """

    def __init__(
            self,
            arm_means:list[float],
            arm_stds:list[float],
            rng:np.random.Generator,
    )->None:
        self.arm_means = np.asarray(
            arm_means,
            dtype=float,
        )
        self.arm_stds = np.asarray(
            arm_stds,
            dtype=float,
        )
        self.rng=rng

        if self.arm_means.ndim !=1:
            raise ValueError(
                "arm_means must be one-dimensional"
            )
        if len(self.arm_means)==0:
            raise ValueError(
                "arm_means must not be empty"
            )

        if self.arm_stds.shape != self.arm_means.shape:
            raise ValueError(
                "arm_stds must have the same shape as arm_means"
            )
        if np.any(self.arm_stds <=0.0):
            raise ValueError(
                "all Gaussian standard deviations must be positive"
            )

        self.best_mean=float(
            np.max(self.arm_means)
        )

    def step(
            self,action:int,
    )->float:
        """
        执行动作，并返回一次Gaussian 随机奖励
        """

        if not 0<=action<len(self.arm_means):
            raise ValueError(
                f"invalid action: {action}"
            )
        reward = self.rng.normal(
            loc=self.arm_means[action],
            scale=self.arm_stds[action]
        )

        return float(reward)

    def pseudo_regret(
            self,
            action:int,
    )->float:
        """
        计算选择该动作造成的单步pseudo-regret
        """
        if not 0<=action<len(self.arm_means):
            raise ValueError(
                f"invalid action: {action}"
            )
        
        regret=(
            self.best_mean-self.arm_means[action]
        )

        return float(regret)