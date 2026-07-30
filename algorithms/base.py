import numpy as np

class BanditAlgorithm:
    """
    所有Bandit算法都需要遵守的统一接口
    """

    def select_action(self)->int:
        """
        根据算法当前状态选择一个动作
        """
        raise NotImplementedError

    def update(self,action:int,reward:int)->None:
        """
        根据本轮动作和奖励更新算法状态
        """
        raise NotImplementedError

class RandomPolicy(BanditAlgorithm):
    """
    随机策略：以相同概率随机选择每一个臂

    它不会学习，只用于测试环境和算法接口是否能够正常配合
    """
    def __init__(
            self,
            num_arms:int,
            rng:np.random.Generator,
    )->None:
        self.num_arms=num_arms
        self.rng=rng

    def select_action(self)->int:
        action=int(self.rng.integers(self.num_arms))
        return action
    def update(self,action:int,reward:int)->None:
        pass