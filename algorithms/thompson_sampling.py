from __future__ import annotations

import numpy as np

from .base import BanditAlgorithm

class ThompsonSampling(BanditAlgorithm):
    """
    Beta-Bernoulli Thompson Sampling
    """

    def __init__(
            self,
            num_arms:int,
            rng:np.random.Generator,
    )->None:
        super().__init__()

        self.alpha = np.ones(num_arms,dtype=float)
        self.beta = np.ones(num_arms,dtype=float)
        self.num_arms=num_arms
        self.rng=rng

    def select_action(self)->int:
        """
        从每个臂的后验中采样，并选择采样值最大的臂
        """
        samples=self.rng.beta(
            self.alpha,
            self.beta,
        )

        return int(np.argmax(samples))

    def update(
            self,
            action:int,
            reward:float,
    )->None:
        """根据一次Bernoulli奖励更新被选择臂的后验。"""
        if not 0<=action<self.num_arms:
            raise ValueError(
                f"action must be in [0,{self.num_arms}, got {action}]"
            )

        if reward not in (0,1):
            raise ValueError(
                f"Thompson Sampling ecpects Bernoulli reward 0 or 1,"
                f"got {reward}"
            )

        self.alpha[action]+=reward
        self.beta[action]+=1-reward