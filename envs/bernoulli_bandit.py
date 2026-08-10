import numpy as np

class BernoulliBandit:
    """
    Bernoulli 多臂老虎机环境

    第i个臂在每次被选择时：
    - 以arm_means[i]的概率返回奖励1；
    - 否则返回奖励0

    """
    def __init__(
            self,
            arm_means:list[float],
            rng:np.random.Generator,
    )->None:
        self.arm_means=np.asarray(arm_means,dtype=float)

        if self.arm_means.ndim !=1:
            raise ValueError("arm_means must be one dimensional")

        if len(self.arm_means) ==0:
            raise ValueError("arm_means must not be empty")

        if not np.all(np.isfinite(self.arm_means)):
            raise ValueError("arm_means must contain only finite values")

        if not np.all(
            (0.0 <= self.arm_means) & (self.arm_means<=1.0)):
            raise ValueError("Bernoulli arm means lie in [0,1]")
        

        self.rng=rng

        self.best_mean=float(np.max(self.arm_means))

    def step(self,action:int)->int:
        """
        执行动作，并返回一次随机奖励
        """
        if not 0<=action<len(self.arm_means):
            raise ValueError(
                f"invalid action:{action}"
            )
        
        reward=self.rng.binomial(
            n=1,
            p=self.arm_means[action]
        )
        return int(reward)
    
    def pseudo_regret(self,action:int)->float:
        """
        计算选择该动作造成的单步pseudo-regret
        """
        if not 0<=action< len(self.arm_means):
            raise ValueError(
                f"invalid action: {action}"
            )
        
        regret=self.best_mean-self.arm_means[action]
        return float(regret)