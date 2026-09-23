import numpy as np

class NonStationaryBernoulliBandit:
    def __init__(self,trajectory,rng):
        self.trajectory=trajectory
        self.rng=rng
        self.num_arms=len(trajectory.at(1))

    def step(self,action,t):
        means=self.trajectory.at(t)

        if isinstance(action,(bool,np.bool_)) or not  isinstance(action,(int,np.integer)):
            raise ValueError("action must be an integer")
        if not 0<=action<self.num_arms:
            raise ValueError("action must be in [0,num_arms)")

        mean=means[action]

        return int(self.rng.binomial(n=1,p=mean))

    def pseudo_regret(self,action,t):
        means=self.trajectory.at(t)

        if isinstance(action,(bool,np.bool_)) or not  isinstance(action,(int,np.integer)):
            raise ValueError("action must be an integer")
        if not 0<=action<self.num_arms:
            raise ValueError("action must be in [0,num_arms)")

        return float(max(means)-means[action])