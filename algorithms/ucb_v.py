from __future__ import annotations

import numpy as np

from algorithms.base import BanditAlgorithm

class UCBV(BanditAlgorithm):
    """
    Variance-aware Upper Confidence Bound algorithm.

    当前实现用于有界奖励环境。项目当前只允许它与
    BernoulliBandit配合使用，此时reward_range = 1。
    """

    def __init__(
            self,
            num_arms:int,
            reward_range:float = 1.0,
    )->None:
        if (
            not isinstance(num_arms,int) 
            or isinstance(num_arms,bool)
            or num_arms<=0
        ):
            raise ValueError("num_arms must be positive integer")
        self.num_arms=num_arms

        if (not isinstance(reward_range, (int, float)) or isinstance(reward_range, bool)
                or not np.isfinite(reward_range) or reward_range <= 0.0):
            raise ValueError(
                "reward_range must be positive float"
            )
        self.reward_range=float(reward_range)

        self.counts=np.zeros(num_arms,dtype=int)

        self.reward_sums=np.zeros(num_arms,dtype=float)

        self.reward_square_sums=np.zeros(num_arms,dtype=float)

    @property
    def estimated_means(
        self,
    )->np.ndarray:
        means=np.divide(
            self.reward_sums,
            self.counts,
            out=np.zeros(
                self.num_arms,
                dtype=float,
            ),
            where=self.counts>0
        )

        return means

    @property
    def empirical_variances(
        self,
    )->np.ndarray:
        means=self.estimated_means

        second_moments=np.divide(
            self.reward_square_sums,
            self.counts,
            out=np.zeros(
                self.num_arms,
                dtype=float,
                ),
                where=self.counts>0,
        )

        variances=(second_moments-means**2)

        # 应对两个数据潜在存在的浮点数误差导致平方差小于0的情况
        variances=np.maximum(
            variances,
            0.0,
        )
        return variances

    def select_action(
            self,
    )->int:
        untried_arms=np.flatnonzero(
            self.counts==0
        )

        if len(untried_arms) >0:
            return int(untried_arms[0])

        total_steps=int(
            self.counts.sum()
        )

        log_term=np.log(
            max(total_steps,2)
        )

        means=self.estimated_means
        variances=self.empirical_variances

        variance_bonus=np.sqrt(
            2.0*variances*log_term/self.counts
        )

        range_bonus=(
            3.0*self.reward_range*log_term/self.counts
        )

        ucb_values=(
            means+variance_bonus+range_bonus
        )

        action=int(np.argmax(ucb_values))

        return action

    def update(self,action:int, reward:float,)->None:
        if not 0.0 <= reward <= 1.0:
            raise ValueError("current bounded policy requires reward in [0, 1]")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise ValueError("action must be an integer")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        if not 0<=action<self.num_arms:
            raise ValueError( f"invalid action:{action}")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        self.counts[action]+=1

        self.reward_sums[action]+=reward

        self.reward_square_sums[action]+=(reward**2)