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
        self.rng=rng
        self.best_mean=float(np.max(self.arm_means))

    def step(self,action:int)->int:
        """
        执行动作，并返回一次随机奖励
        """
        reward=self.rng.binomial(
            n=1,
            p=self.arm_means[action]
        )
        return int(reward)
    
    def pseudo_regret(self,action:int)->float:
        """
        计算选择该动作造成的单步pseudo-regret
        """
        regret=self.best_mean-self.arm_means[action]
        return float(regret)