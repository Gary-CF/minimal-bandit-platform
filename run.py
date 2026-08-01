from __future__ import annotations

import numpy as np

from algorithms.base import RandomPolicy
from algorithms.ucb1 import UCB1
from algorithms.thompson_sampling import ThompsonSampling
from envs.bernoulli_bandit import BernoulliBandit

def main() -> None:
    # ====================
    # 1、实验参数
    # 一般需要确认随机种子，bandit方法集，观察轮数，以此确定随机数生成器，bandit数
    # ====================

    seed=42
    arm_means=[0.3,0.5,0.7]
    horizon=5000

    num_arms=len(arm_means)

    algorithm_name="ucb1"

    seed_sequence=np.random.SeedSequence(seed)
    env_seed,algorithm_seed = seed_sequence.spawn(2)

    env_rng=np.random.default_rng(env_seed)
    algorithm_rng=np.random.default_rng(algorithm_seed)
   
    # ====================
    # 2、创建环境和算法
    # env表示的是本次bandit实验所处环境：一般会确定bandit背景，给出奖励和伪遗憾
    # algorithm表示的是当前采取的策略，负责动作选择和策略更新
    # 当前的env和algorithm都需要随机数生成器
    # ====================

    env=BernoulliBandit(
        arm_means=arm_means,
        rng=env_rng
    )

    if algorithm_name=="random":
        algorithm=RandomPolicy(
            num_arms=num_arms,
            rng=algorithm_rng,
        )
    elif algorithm_name=="ucb1":
        algorithm = UCB1(
            num_arms=num_arms
        )
    elif algorithm_name=="thompson":
        algorithm=ThompsonSampling(
            num_arms=num_arms,
            rng=algorithm_rng,
        )
    else:
        raise ValueError(
            f"unknown algorithm: {algorithm_name}"
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
    first_actions=[]

    action_counts=np.zeros(
        num_arms,
        dtype=int,
    )

    reward_sums=np.zeros(
        num_arms,
        dtype=float,
    )

    # ====================    
    # 4、主循环
    # 进行指定轮数的循环
    # 每一轮：动作选择->奖励反馈->策略更新->遗憾统计->状态跟踪
    # 每轮都会即时检查动作，奖励和伪遗憾是否合规

    for t in range(1,horizon+1):
        action=algorithm.select_action()

        if t<= num_arms:
            first_actions.append(action)

        reward=env.step(action)

        algorithm.update(action,reward)

        instant_regret=env.pseudo_regret(action)

        total_reward+=reward
        cumulative_regret+=instant_regret

        action_counts[action]+=1
        reward_sums[action]+=reward

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
    # 首先检查所有算法都应当满足的公共性质，
    # 然后分别检查不同算法自己的内部状态
    # ====================

    # ---------- 公共测试 ----------

    assert len(records) == horizon

    assert int(action_counts.sum()) == horizon

    assert np.all(action_counts >= 0)

    assert np.isclose(
        reward_sums.sum(),
        total_reward,
    )

    cumulative_regrets = [
        float(record["cumulative_regret"])
        for record in records
    ]

    assert all(
        previous <= current
        for previous, current in zip(
            cumulative_regrets,
            cumulative_regrets[1:],
        )
    )

    empirical_means = np.divide(
        reward_sums,
        action_counts,
        out=np.zeros(
            num_arms,
            dtype=float,
        ),
        where=action_counts > 0,
    )

    # ---------- UCB1 专属测试 ----------

    if algorithm_name == "ucb1":
        assert isinstance(algorithm ,UCB1)
        # UCB1 的初始化阶段会依次选择所有臂。
        assert first_actions == list(
            range(num_arms)
        )

        assert int(
            algorithm.counts.sum()
        ) == horizon

        assert np.all(
            algorithm.counts >= 1
        )

        assert np.array_equal(
            algorithm.counts,
            action_counts,
        )

        assert np.allclose(
            algorithm.reward_sums,
            reward_sums,
        )

        assert np.allclose(
            algorithm.estimated_means,
            empirical_means,
        )

    # ---------- Thompson Sampling 专属测试 ----------

    if algorithm_name == "thompson":
        assert isinstance(algorithm ,ThompsonSampling)
        # alpha_i - 1 应当等于第 i 个臂的成功次数。
        assert np.allclose(
            algorithm.alpha - 1,
            reward_sums,
        )

        # beta_i - 1 应当等于第 i 个臂的失败次数。
        assert np.allclose(
            algorithm.beta - 1,
            action_counts - reward_sums,
        )

        # 初始 alpha、beta 都是 1，
        # 更新后它们不应小于 1。
        assert np.all(
            algorithm.alpha >= 1
        )

        assert np.all(
            algorithm.beta >= 1
        )

        # 每选择一次臂，alpha 或 beta 总共增加一次。
        assert np.allclose(
            algorithm.alpha
            + algorithm.beta
            - 2,
            action_counts,
        )

   # ====================
    # 6、输出结果
    # 一般会输出：
    # 算法名称、bandit 数目、最佳 arm、
    # 总奖励、累计遗憾、选择次数和经验均值
    # ====================

    most_selected_arm = int(
        np.argmax(action_counts)
    )

    print(f"algorithm: {algorithm_name}")
    print(f"number of arms: {num_arms}")
    print(f"horizon: {horizon}")
    print(f"best arm: {best_arm}")
    print(f"best mean: {best_mean:.2f}")
    print()

    print(f"first actions: {first_actions}")
    print(f"arm counts: {action_counts}")
    print(
        "empirical means:",
        np.round(
            empirical_means,
            decimals=4,
        ),
    )

    if algorithm_name == "thompson":
        assert isinstance(algorithm ,ThompsonSampling)
        posterior_means = (
            algorithm.alpha
            / (
                algorithm.alpha
                + algorithm.beta
            )
        )

        print(
            "posterior means:",
            np.round(
                posterior_means,
                decimals=4,
            ),
        )
        print(f"alpha: {algorithm.alpha}")
        print(f"beta: {algorithm.beta}")

    print()

    print(
        f"most selected arm: "
        f"{most_selected_arm}"
    )
    print(f"total reward: {total_reward}")
    print(
        f"cumulative regret: "
        f"{cumulative_regret:.2f}"
    )

    if most_selected_arm != best_arm:
        print(
            "Warning: the most selected arm is not "
            "the true best arm in this run"
        )

    print("all checks passed")

if __name__ == "__main__":
    main()
