from __future__ import annotations

import numpy as np

from algorithms.base import BanditAlgorithm

def binary_kl(p:float,q:float)->float:
    if not (0.0<=p<=1.0 and  0.0<=q<=1.0):
        raise ValueError("p and q should be in [0,1]") 
    if p == q:
        return 0.0
    if q in (0.0, 1.0):
        return float("inf")
    if p==0.0:
        return -np.log(1-q)
    if p==1.0:
        return -np.log(q)
    return p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))

def kl_ucb_bound(p:float,budget:float)->float:
    if not np.isfinite(budget) or budget<0:
        raise ValueError("budget must not be negative")
    if not 0.0<=p<=1.0:
        raise ValueError("p should be in [0,1]")
    if budget==0.0:
        return p

    left=p
    right=1.0

    max_iter=40

    for _ in range(max_iter):
        mid=(left+right)/2

        if binary_kl(p,mid)<=budget:
            left=mid
        else:
            right =mid
    return left

class KLUCB(BanditAlgorithm):
    def __init__(
            self,num_arms:int,c:float=3.0
    )->None:
        if (
            not isinstance(num_arms,int) 
            or isinstance(num_arms,bool)
            or num_arms<=0
        ):
            raise ValueError("num_arms must be positive integer")
        self.num_arms=num_arms

        if c<0 or not np.isfinite(c):
            raise ValueError("c must be a positive finite number")
        self.c=c

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
        
        T=completed_steps+1
        kl_budget=(np.log(T)+self.c*np.log(max(np.log(T),1)))/self.counts
        kl_bound=np.array([
            kl_ucb_bound(
                self.estimated_mean[arm],
                kl_budget[arm],
            )
            for arm in range(self.num_arms)
        ]
        )
        return int(np.argmax(kl_bound))

    def update(self, action: int, reward: float) -> None:
        if not 0.0 <= reward <= 1.0:
            raise ValueError("current bounded policy requires reward in [0, 1]")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise ValueError("action must be an integer")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
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



        