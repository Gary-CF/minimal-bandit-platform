from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from algorithms.base import RandomPolicy
from algorithms.ucb1 import UCB1
from algorithms.thompson_sampling import ThompsonSampling
from envs.bernoulli_bandit import BernoulliBandit

ExperimentRecord = dict[
     str,
     str | int | float,
]

def parse_args()->argparse.Namespace:
    parser=argparse.ArgumentParser(
            description="Run multi-armed bandit experiments"
            )
    parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="Path to the JSON experiment configuraion.",
            )
    return parser.parse_args()

def load_config(
          config_path:str,
)->dict[str,Any]:
    with open(
          config_path,
          mode="r",
          encoding="utf-8",
     ) as file:
          config=json.load(file)

    return config

def create_algorithm(
        algorithm_name:str,
        num_arms:int,
        rng:np.random.Generator,
)->RandomPolicy | UCB1 | ThompsonSampling:
    if algorithm_name=="random":
            return RandomPolicy(
                num_arms=num_arms,
                rng=rng,
            )
    if algorithm_name=="ucb1":
            return UCB1(
                num_arms=num_arms
            )
    if algorithm_name=="thompson_sampling":
            return ThompsonSampling(
                num_arms=num_arms,
                rng=rng,
            )
    raise ValueError(
                f"unknown algorithm: {algorithm_name}"
            )



def run_single_experiment(
        seed:int,
        arm_means:list[float],
        horizon:int,
        algorithm_name:str,
) -> list[ExperimentRecord]:
    # ====================
    # 1、实验参数
    # 一般需要确认随机种子，bandit方法集，观察轮数，以此确定随机数生成器，bandit数
    # ====================



    num_arms=len(arm_means)



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

    algorithm=create_algorithm(
         algorithm_name=algorithm_name,num_arms=num_arms,rng=algorithm_rng)

    best_mean=env.best_mean
    best_arm=int(np.argmax(env.arm_means))


    # ====================
    # 3、实验状态
    # 实验中应当跟踪的状态一般有总奖励，累计遗憾和实验记录
    # 试验记录通常包括：当前轮数、采取行动、获得奖励、即时遗憾、累计遗憾
    # ====================

    total_reward=0
    cumulative_regret=0.0
    records: list[ExperimentRecord]=[]
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
                "algorithm": algorithm_name,
                "seed":seed,
                "step":t,
                "action":int(action),
                "reward":int(reward),
                "instant_regret":float(instant_regret),
                "cumulative_regret":float(cumulative_regret),

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

    if algorithm_name == "thompson_sampling":
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
    # 6、检查实验结果
    # ====================

    most_selected_arm = int(
    np.argmax(action_counts)
)

    if most_selected_arm != best_arm:
        print(
        f"warning: {algorithm_name} "
        "did not select the optimal arm most often"
    )

    return records


def save_records(
          records:list[ExperimentRecord],
          output_path:Path,
)->None:
     fieldnames=[
        "algorithm",
        "seed",
        "step",
        "action",
        "reward",
        "instant_regret",
        "cumulative_regret",
     ]

     output_path.parent.mkdir(
          parents=True,exist_ok=True,
     )

     with output_path.open(
          mode="w",
          newline="",
          encoding="utf-8",
     ) as file:
          writer = csv.DictWriter(
               file,
               fieldnames=fieldnames,
          )

          writer.writeheader()
          writer.writerows(records)






def main()->None:
    args=parse_args()

    config=load_config(
         config_path=args.config
    )

    arm_means=[
         float(mean) for mean in config["arm_means"]
    ]

    horizon=int(config["horizon"])

    algorithms=[
         str(name) for name in config["algorithms"]
    ]

    seeds=[
         int(seed) for seed in config["seeds"]
    ]

    num_experiments=len(algorithms)*len(seeds)

    print(
         f"running {len(algorithms)} algorithms"
         f" x {len(seeds)} seeds"
         f" = {num_experiments} experiments"
    )
    print()

    results_dir=Path("results")

    for algorithm_name in algorithms:
        for seed in seeds:
            print("="*60)
            print(
              f"running algorithm={algorithm_name},"
              f"seed={seed}"
         )
            print("="*60)

            records=run_single_experiment(
            seed=seed,
            arm_means=arm_means,
            horizon=horizon,
            algorithm_name=algorithm_name
         )

            output_path=(
                 results_dir/f"{algorithm_name}_seed_{seed}.csv"
            )

            save_records(
                 records=records,
                 output_path=output_path
            )

            print(
                 f"saved results to:"
                 f"{output_path}"
            )

            print()


if __name__ == "__main__":
    main()
