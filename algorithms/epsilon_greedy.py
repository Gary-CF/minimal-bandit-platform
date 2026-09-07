from __future__ import annotations

import numpy as np

from algorithms.base import BanditAlgorithm

class EpsilonGreedy(BanditAlgorithm):
    def __init__(
        self,num_arms:int,epsilon:float,rng:np.random.Generator       
    )->None:
        if (
            not isinstance(num_arms,int) 
            or isinstance(num_arms,bool)
            or num_arms<=0
        ):
            raise ValueError("num_arms must be positive integer")
        self.num_arms=num_arms

        if not 0<=epsilon<=1:
            raise ValueError("epsilon must in [0,1]")
        self.epsilon=epsilon

        self.counts=np.zeros(
            num_arms,
            dtype=int,
        )

        self.reward_sum=np.zeros(
            num_arms,
            dtype=float,
        )

        self.estimated_reward=np.zeros(
            num_arms,
            dtype=float,
        ) 

        self.rng=rng

    def select_action(self) -> int:
        x=self.rng.random()
        if x<self.epsilon:
            action=int(self.rng.integers(self.num_arms))
            return action
        return int(np.argmax(self.estimated_reward))

    def update(self, action: int, reward: float) -> None:
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise ValueError("action must be an integer")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        if not 0<=action<self.num_arms:
            raise IndexError(f"action {action} is outside [0,{self.num_arms})")

        self.counts[action]+=1
        self.reward_sum[action]+=reward

        self.estimated_reward[action]=(
            self.reward_sum[action]/self.counts[action]
        )