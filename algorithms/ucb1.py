from __future__ import annotations

import numpy as np

from algorithms.base import BanditAlgorithm

class UCB1(BanditAlgorithm):
    """
    经典UCB1算法：

    当前实现假设奖励大致位于[0,1]，
    与Bernoulli Bandit 环境相匹配
    """

    def __init__(self,num_arms:int)->None:
        if (
            not isinstance(num_arms,int) 
            or isinstance(num_arms,bool)
            or num_arms<=0
        ):
            raise ValueError("num_arms must be positive integer")
        self.num_arms=num_arms

        # counts[i]:
        # 第i个臂已经被选择了多少次
        self.counts = np.zeros(
            num_arms,
            dtype=int,
        )

        # reward_sums[i]:
        # 第i个臂目前累计获得的奖励
        self.reward_sums = np.zeros(
            num_arms,
            dtype=float,
        )

        # estimated_means[i]
        # 第i个臂当前的经验平均奖励
        self.estimated_means = np.zeros(
            num_arms,
            dtype=float,
        )

    def select_action(self) -> int:
        """
        根据当前信息选择一个臂。
        """
        completed_steps = int(self.counts.sum())

        # 前K轮依次选择0,1,...,K-1
        # 保证每个臂至少被访问一次
        # 这样后面计算UCB时counts中不会出现0
        # 避免出现0
        if completed_steps < self.num_arms:
            return completed_steps

        # selection_action 在当前轮奖励产生之前调用
        # 如果已经完成 completed_steps  轮
        # 当前正在选择的是第 completed_steps + 1轮
        current_round=completed_steps+1

        confidence_radios=np.sqrt(
            2.0*np.log(current_round)/self.counts
        )

        ucb_values=(
            self.estimated_means+confidence_radios
        )

        return int(np.argmax(ucb_values))

    def update(
            self,
            action:int,
            reward:float,
    )->None:
        """
        根据本轮观察到的奖励更新统计量
        """
        if not 0<=action<self.num_arms:
            raise IndexError(
                f"action {action} is outside"
                f"[0,{self.num_arms})"
            )

        self.counts[action]+=1
        self.reward_sums[action]+=reward

        self.estimated_means[action]=(
            self.reward_sums[action]/self.counts[action]
        )