from __future__ import annotations

import numpy as np

from algorithms.base import BanditAlgorithm

class MOSS(BanditAlgorithm):
    def __init__(
            self,num_arms:int,horizon:int,
    ):
        if (
            not isinstance(num_arms,int) 
            or isinstance(num_arms,bool)
            or num_arms<=0
        ):
            raise ValueError("num_arms must be positive integer")
        self.num_arms=num_arms

        if (
            not isinstance(horizon,int) 
            or isinstance(horizon,bool)
            or horizon<=0
        ):
            raise ValueError("horizon must be positive integer")
        self.horizon=horizon

        self.counts=np.zeros(
            num_arms,dtype=int,
        )

        self.reward_sums=np.zeros(
            num_arms,dtype=float,
        )

        self.estimated_means=np.zeros(
            num_arms,dtype=float,
        )

    def select_action(self) -> int:
        completed_steps=int(self.counts.sum())

        if completed_steps<self.num_arms:
            return completed_steps

        confidence_radios=np.sqrt(
            np.maximum(np.log(self.horizon/(self.num_arms*self.counts)),0.0)/self.counts
        )

        MOSS_values=(
            self.estimated_means+confidence_radios
        )

        return int(np.argmax(MOSS_values))

    def update(
            self,action:int,reward:float,
    )->None:
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