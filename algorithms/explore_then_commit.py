from __future__ import annotations

import numpy as np

from algorithms.base import BanditAlgorithm

class ETC(BanditAlgorithm):
    def __init__(
            self,
            num_arms:int,
            exploration_rounds_per_arm:int,
    )->None:
        if (
            not isinstance(num_arms,int) 
            or isinstance(num_arms,bool)
            or num_arms<=0
        ):
            raise ValueError("num_arms must be positive integer")
        self.num_arms=num_arms

        if (
            not isinstance(exploration_rounds_per_arm,int) 
            or isinstance(exploration_rounds_per_arm,bool)
            or exploration_rounds_per_arm<=0
        ):
            raise ValueError("exploration_rounds_per_arm must be positive integer")
        self.exploration_rounds_per_arm=exploration_rounds_per_arm

        self.counts=np.zeros(
            num_arms,dtype=int,
        )

        self.reward_sums=np.zeros(
            num_arms,dtype=float,
        )

        self.estimated_mean=np.zeros(
            num_arms,dtype=float,
        )

        self.explore_bound=num_arms*exploration_rounds_per_arm

        self.committed_arm:int | None=None

    def select_action(self) -> int:
        completed_steps=int(self.counts.sum())

        if completed_steps < self.explore_bound:
            return completed_steps%(self.num_arms)
        if self.committed_arm is None:
            self.committed_arm=int(np.argmax(self.estimated_mean))
        return self.committed_arm

    def update(self,action:int,reward:float)->None:
        if not 0<=action<self.num_arms:
            raise IndexError(
                f"action {action} is outside"
                f"[0,{self.num_arms})"
            )
        self.counts[action]+=1
        self.reward_sums[action]+=reward
        self.estimated_mean[action]=(
            self.reward_sums[action]/self.counts[action]
        )