from __future__ import annotations

import numpy as np

from algorithms.base import BanditAlgorithm

class GaussianUCB(BanditAlgorithm):
    def __init__(
            self,num_arms:int,known_std:float
    )->None:
        if (
            not isinstance(num_arms,int) 
            or isinstance(num_arms,bool)
            or num_arms<=0
        ):
            raise ValueError("num_arms must be positive integer")
        self.num_arms=num_arms

        if known_std<=0 or not np.isfinite(known_std):
            raise ValueError("std must be positive finite float")
        self.known_std=known_std

        self.counts=np.zeros(
            num_arms,dtype=int,
        )

        self.reward_sums=np.zeros(
            num_arms,dtype=float,
        )

        self.estimated_mean=np.zeros(
            num_arms,dtype=float,
        )
    def select_action(self) -> int:
        completed_steps=int(self.counts.sum())
        if completed_steps<self.num_arms:
            return completed_steps
        current_round=completed_steps+1
        confidence_radius=self.known_std*(np.sqrt(2*np.log(current_round)/self.counts))
        gaussian_value=self.estimated_mean+confidence_radius
        return int(np.argmax(gaussian_value))

    def update(self, action: int, reward: float) -> None:
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