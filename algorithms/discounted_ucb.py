import numpy as np

from algorithms.base import BanditAlgorithm

class DiscountedUCB(BanditAlgorithm):
    def __init__(self,num_arms:int,gamma:float) ->None:
        if (
            not isinstance(num_arms,int) 
            or isinstance(num_arms,bool)
            or num_arms<=0
        ):
            raise ValueError("num_arms must be positive integer")

        if(
            not isinstance(gamma,float)
            or not 0<gamma<1
        ):
            raise ValueError("gamma must be a float with (0,1)")
        
        self.num_arms=num_arms
        self.gamma=gamma

        
        self.counts = np.zeros(
            num_arms,
            dtype=float,
        )

        
        self.reward_sums = np.zeros(
            num_arms,
            dtype=float,
        )

        self.completed_steps=0

    def update(
            self,
            action:int,
            reward:float,
    )->None:
        
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

        self.counts*=self.gamma
        self.reward_sums*=self.gamma
        self.counts[action]+=1
        self.reward_sums[action]+=reward
        self.completed_steps+=1

    def select_action(self) -> int:
        if self.completed_steps< self.num_arms:
            return self.completed_steps
        empty_arms = np.flatnonzero(self.counts==0)
        if len(empty_arms)>0:
            return int(empty_arms[0])

        n_eff=max(1.0,float(self.counts.sum()))

        means=self.reward_sums/self.counts
        confidence_radios = (
    2.0 * np.sqrt(2.0 * np.log(n_eff))
    / np.sqrt(self.counts)
)

        scores=means+confidence_radios

        return int(np.argmax(scores))
       

        