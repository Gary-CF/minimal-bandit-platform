from __future__ import annotations

import numpy as np

from algorithms.base import RandomPolicy
from envs.bernoulli_bandit import BernoulliBandit

def main() -> None:
    # ====================
    # 1、实验参数
    # 一般需要确认随机种子，bandit方法集，观察轮数，以此确定随机数生成器，bandit数
    # ====================

    seed=42
    arm_means=[0.2,0.5,0.7]
    horizon=1000

    rng=np.random.default_rng(seed)
    num_arms=len(arm_means)
    
   
    # ====================
    # 2、创建环境和算法
    # env表示的是本次bandit实验所处环境：一般会确定bandit背景，给出奖励和伪遗憾
    # algorithm表示的是当前采取的策略，负责动作选择和策略更新
    # 当前的env和algorithm都需要随机数生成器
    # ====================

    env=BernoulliBandit(
        arm_means=arm_means,
        rng=rng
    )

    algorithm=RandomPolicy(
        num_arms=num_arms,
        rng=rng
    )

    best_mean=env.best_mean
    best_arm=int(np.argmax(env.arm_means))


    # ====================
    # 3、实验状态
    # 实验中应当跟踪的状态一般有总奖励，累计遗憾和实验记录
    # 试验记录通常包括：当前轮数、采取行动、获得奖励、即时遗憾、累计遗憾
    # ====================
    
    total_reward=0
    cumulative_regret=0.0
    records=[]

    # ====================    
    # 4、主循环
    # 进行指定轮数的循环
    # 每一轮：动作选择->奖励反馈->策略更新->遗憾统计->状态跟踪
    # 每轮都会即时检查动作，奖励和伪遗憾是否合规

    for t in range(1,horizon+1):
        action=algorithm.select_action()

        reward=env.step(action)

        algorithm.update(action,reward)

        instant_regret=env.pseudo_regret(action)

        total_reward+=reward
        cumulative_regret+=instant_regret
        records.append(
            {
                "step":t,
                "action":action,
                "reward":reward,
                "instant_regret":instant_regret,
                "cumulative_regret":cumulative_regret,

            }
        )
        assert 0<=action<num_arms
        assert reward in (0,1)
        assert instant_regret>=0

    # ====================   
    # 5、整体测试
    # 一般会整体测试内部轮数是否符合设定，生成遗憾数据表并检查是否单调
    # ====================   

    assert len(records)==horizon

    cumulative_regrets=[float(record["cumulative_regret"])
                        for record in records]

    assert all(
            previous <= current
            for previous,current in zip(
                cumulative_regrets[:],
                cumulative_regrets[1:],
            )
        
    )

    # ====================   
    # 6、输出结果
    # 一般会输出bandit数目，最佳arm和最优方法，总奖励和累计遗憾，记录数，并标识实验全部通过
    # ====================   
    print(f"number of arms:{num_arms}")
    print(f"best arm:{best_arm}")
    print(f"best mean:{best_mean:.2f}")
    print(f"total reward:{total_reward}")
    print(f"cumulative regret:{cumulative_regret:.2f}")
    print(f"number of records:{len(records)}")
    print("all tests passed")

if __name__ == "__main__":
    main()
