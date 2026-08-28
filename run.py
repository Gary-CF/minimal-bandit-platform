from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from algorithms.base import (BanditAlgorithm,RandomPolicy)
from algorithms.ucb1 import UCB1
from algorithms.thompson_sampling import ThompsonSampling
from algorithms.ucb_v import UCBV
from algorithms.epsilon_greedy import EpsilonGreedy
from algorithms.explore_then_commit import ETC
from algorithms.moss import MOSS
from algorithms.kl_ucb import KLUCB
from algorithms.gaussian_ucb import GaussianUCB
from algorithms.gaussian_thompson_sampling import GaussianThompsonSampling

from envs.bernoulli_bandit import BernoulliBandit
from envs.gaussian_bandit import GaussianBandit

BanditEnvironment= GaussianBandit | BernoulliBandit

ExperimentRecord = dict[
     str,
     str | int | float,
]

def parse_args()->argparse.Namespace:
    '''
    解析命令行参数，要求用户通过 --config 指定实验配置文件。
    '''
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
    '''
    打开配置文件，加载配置信息
    '''
    with open(
          config_path,
          mode="r",
          encoding="utf-8",
     ) as file:
          config=json.load(file)

    return config

def normalize_config(raw_config:dict[str,Any],)->dict[str,Any]:
     '''
     规范化参数列表
     '''
     experiment_name=str(raw_config["experiment_name"])

     if not experiment_name:
          raise ValueError("experiment_name must not be empty")

     environment_config = dict(raw_config["environment"])

     raw_algorithms=list(raw_config["algorithms"])

     algorithms:list[dict[str,Any]]=[]

     for raw_algorithm in raw_algorithms:
          #兼容旧配置
          #“ucb1" -> {"name":"ucb1","parameters":{}}
          if isinstance(raw_algorithm,str):
               algorithms.append(
                    {
                         "name":raw_algorithm,
                         "parameters":{},
                    }
               )
               continue
          if not isinstance(raw_algorithm,dict,):
               raise ValueError(
                    "each algorithm config must be"
                    "a string or a dictionary"
               )

          algorithms.append(
               {
                    "name":str(raw_algorithm["name"]),
                    "parameters":dict(raw_algorithm.get("parameters",{},))
               }
          )

     if "horizons" in raw_config:
               horizons=[
                    int(horizon) for horizon in raw_config["horizons"]
               ]
     else:
               horizons=[
                    int(raw_config["horizon"])
               ]
     seeds=[int(seed) for seed in raw_config["seeds"]]

     if len(algorithms)==0:
               raise ValueError("algorithms must not be empty")
     if len(horizons)==0:
               raise ValueError("horizons must not be empty")
     if any(horizon <=0 for horizon in horizons):
               raise ValueError("all horizons must be positive")
     if len(seeds)==0:
               raise ValueError("seeds must not be empty")
     return{
               "experiment_name":experiment_name,
               "environment":environment_config,
               "algorithms":algorithms,
               "horizons":horizons,
               "seeds":seeds,
          }



def create_environment(
          environment_config:dict[str,Any],
          rng:np.random.Generator
)->BanditEnvironment:
     environment_name=str(
          environment_config["name"]
     )

     arm_means=[
          float(mean) for mean in environment_config["arm_means"]
     ]

     if environment_name=="bernoulli":
          return BernoulliBandit(
               arm_means=arm_means,
                rng=rng,
          )
     if environment_name=="gaussian":
          arm_stds=[
               float(std) for std in environment_config["arm_stds"]
          ]
          return GaussianBandit(
               arm_means=arm_means,
               arm_stds=arm_stds,
               rng=rng,
          )
     raise ValueError(
          f"unknown environment:{environment_name}"
     )


def validate_algorithm_environment(
          algorithm_name:str,
          env:BanditEnvironment,
)->None:
     """
     检查当前算法实现是否适用于当前环境。

     这里检查的是“当前代码实现”的适用范围，
     而不是算法家族在理论上的全部适用范围。
     """

     if (algorithm_name=="random" or algorithm_name=="epsilon_greedy"
         or algorithm_name=="etc") :
          return

     if algorithm_name=="ucb1":
          if isinstance(env,BernoulliBandit):
               return
          raise ValueError(
               "the current UCB1 implementation assumes"
               "rewards are bounded in [0,1]"
               f"it cannot be used with {type(env).__name__}"
          )
     if algorithm_name=="thompson_sampling":
          if isinstance(env,BernoulliBandit):
               return
          raise ValueError(
            "the current ThompsonSampling implementation "
            "uses a Beta-Bernoulli posterior; "
            f"it cannot be used with {type(env).__name__}"
        )
     if algorithm_name=="ucb_v":
          if isinstance(env,BernoulliBandit):
               return
          raise ValueError(
               "the current UCBV implementation assumes "
                "rewards are bounded in [0, 1]; "
                f"it cannot be used with {type(env).__name__}"
          )
     if algorithm_name=="moss":
           if isinstance(env,BernoulliBandit):
                 return
           raise ValueError(
                 "the current MOSS implementation assumes rewards are bounded in [0, 1], "
                 f"it cannot be used with {type(env).__name__}"
           )
     if algorithm_name=="kl_ucb":
          if isinstance(env,BernoulliBandit):
               return
          raise ValueError(
               "the current KLUCB implementation "
               "uses Bernoulli KL divergence; "
               f"it cannot be used with {type(env).__name__}"
          )
     if algorithm_name=="gaussian_ucb":
          if isinstance(env,GaussianBandit):
               return
          raise ValueError(
               "GaussianUCB requires a GaussianBandit environment; "
               f"it cannot be used with {type(env).__name__}"
          )
     if algorithm_name=="gaussian_thompson_sampling":
          if isinstance(env,GaussianBandit):
               return

          raise ValueError(
               "GaussianThompsonSamping requires a GaussianBandit environment; "
               f"if cannot be used with {type(env).__name__}"
          )
def create_algorithm(
        algorithm_config:dict[str,Any],
        num_arms:int,
        rng:np.random.Generator,
        horizon:int,
)->BanditAlgorithm:
    algorithm_name=str(algorithm_config["name"])
    parameters=dict(algorithm_config.get("parameters",{},))

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
    if algorithm_name=="ucb_v":
         reward_range=float(
              parameters.get("reward_range",1.0)
         )
         return UCBV(
              num_arms=num_arms,
              reward_range=reward_range,
         )
    if algorithm_name=="epsilon_greedy":
          epsilon=float(parameters.get("epsilon",0.1))
          return EpsilonGreedy(
                num_arms=num_arms,
                epsilon=epsilon,
                rng=rng
          )
    if algorithm_name=="etc":
          exploration_round_per_arm=parameters["exploration_rounds_per_arm"]
          return ETC(
                num_arms=num_arms,
                exploration_rounds_per_arm=exploration_round_per_arm
          )
    if algorithm_name=="moss":
          return MOSS(
                num_arms=num_arms,
                horizon=horizon,
          )
    if algorithm_name=="kl_ucb":
         c=float(
              parameters.get("c",3.0)
         )
         return KLUCB(
              num_arms=num_arms,
              c=c,
         )
    if algorithm_name=="gaussian_ucb":
         known_std=float(
              parameters["known_std"]
         )

         return GaussianUCB(
              num_arms=num_arms,
              known_std=known_std,
         )
    if algorithm_name=="gaussian_thompson_sampling":
         prior_mean=float(
              parameters.get("prior_mean",0.0)
         )

         prior_variance=float(
              parameters.get("prior_variance",1.0)
         )

         noise_variance=float(
              parameters["noise_variance"]
         )
         return GaussianThompsonSampling(
              num_arms=num_arms,
              prior_mean=prior_mean,
              prior_variance=prior_variance,
              noise_variance=noise_variance,
              rng=rng,
         )
    
    raise ValueError(
                f"unknown algorithm: {algorithm_name}"
            )
    

def run_single_experiment(
        seed:int,
        environment_config:dict[str,Any],
        horizon:int,
        algorithm_config:dict[str,Any]
) -> list[ExperimentRecord]:
    # ====================
    # 1、实验参数
    # 一般需要确认随机种子，bandit方法集，观察轮数，以此确定随机数生成器，bandit数
    # ====================

    algorithm_name=str(algorithm_config["name"])

    seed_sequence=np.random.SeedSequence(seed)
    env_seed,algorithm_seed = seed_sequence.spawn(2)

    env_rng=np.random.default_rng(env_seed)
    algorithm_rng=np.random.default_rng(algorithm_seed)

    env=create_environment(
         environment_config=environment_config,rng=env_rng,
    )

    validate_algorithm_environment(
         algorithm_name=algorithm_name,
         env=env,
    )

    num_arms=len(env.arm_means)

    # ====================
    # 2、创建环境和算法
    # env表示的是本次bandit实验所处环境：一般会确定bandit背景，给出奖励和伪遗憾
    # algorithm表示的是当前采取的策略，负责动作选择和策略更新
    # 当前的env和algorithm都需要随机数生成器
    # ====================


    algorithm=create_algorithm(
         algorithm_config=algorithm_config,num_arms=num_arms,rng=algorithm_rng,horizon=horizon)

    best_arm=int(np.argmax(env.arm_means))


    # ====================
    # 3、实验状态
    # 实验中应当跟踪的状态一般有总奖励，累计遗憾和实验记录
    # 试验记录通常包括：当前轮数、采取行动、获得奖励、即时遗憾、累计遗憾
    # ====================

    total_reward=0.0
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
    "environment": str(
        environment_config["name"]
    ),
    "algorithm": algorithm_name,
    "horizon": horizon,
    "seed": seed,
    "step": t,
    "action": int(action),
    "reward": float(reward),
    "instant_regret": float(
        instant_regret
    ),
    "cumulative_regret": float(
        cumulative_regret
    ),
}
        )
        assert 0<=action<num_arms
        assert np.isfinite(reward)
        assert instant_regret>=0

        if isinstance(env,BernoulliBandit):
             assert reward in (0.0,1.0)

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
            range(min(horizon,num_arms))
        )

        assert int(
            algorithm.counts.sum()
        ) == horizon

        if horizon >= num_arms:
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
    # ---------- UCB-V 专属测试 ----------
    if algorithm_name == "ucb_v":
        assert isinstance(
        algorithm,
        UCBV,
    )

    # UCB-V 的初始化阶段应依次选择所有臂。
        assert first_actions == list(
        range(min(horizon,num_arms))
    )

    # 算法总更新次数应等于实验 horizon。
        assert int(
        algorithm.counts.sum()
    ) == horizon

    # 初始化后，每个臂至少被选择一次。
        if horizon>=num_arms:
            assert np.all(
            algorithm.counts >= 1
            )

    # 算法内部计数应与实验端计数一致。
        assert np.array_equal(
        algorithm.counts,
        action_counts,
    )

    # 算法内部奖励和应与实验端一致。
        assert np.allclose(
        algorithm.reward_sums,
        reward_sums,
    )
    # ---------- Epsilon-Greedy 专属测试 ----------

    if algorithm_name == "epsilon_greedy":
        assert isinstance(
        algorithm,
        EpsilonGreedy,
    )

        assert int(
        algorithm.counts.sum()
    ) == horizon

        assert np.array_equal(
        algorithm.counts,
        action_counts,
    )

        assert np.allclose(
        algorithm.reward_sum,
        reward_sums,
    )

        assert np.allclose(
        algorithm.estimated_reward,
        empirical_means,
    )
    # ---------- ETC 专属测试 ----------

    if algorithm_name == "etc":
        assert isinstance(
            algorithm,
            ETC,
        )

        # 算法总更新次数应等于实验 horizon。
        assert int(
            algorithm.counts.sum()
        ) == horizon

        # 算法内部计数应与 runner 记录一致。
        assert np.array_equal(
            algorithm.counts,
            action_counts,
        )

        # 算法内部奖励和应与 runner 记录一致。
        assert np.allclose(
            algorithm.reward_sums,
            reward_sums,
        )

        # 算法内部经验均值应与 runner 计算结果一致。
        assert np.allclose(
            algorithm.estimated_mean,
            empirical_means,
        )

        # 在探索阶段，动作必须严格按照
        # 0, 1, ..., K-1, 0, 1, ... 的顺序进行。
        exploration_steps = min(
            horizon,
            algorithm.explore_bound,
        )

        exploration_actions = [
            int(record["action"])
            for record in records[:exploration_steps]
        ]

        expected_exploration_actions = [
            step % num_arms
            for step in range(exploration_steps)
        ]

        assert (
            exploration_actions
            == expected_exploration_actions
        )

        # 如果实验在探索阶段结束前或刚好结束时停止，
        # 就还没有下一次 select_action() 来触发 commit。
        if horizon <= algorithm.explore_bound:
            assert algorithm.committed_arm is None
        else:
            assert algorithm.committed_arm is not None
            assert (
                0
                <= algorithm.committed_arm
                < num_arms
            )

    # ---------- MOSS 专属测试 ----------

    if algorithm_name == "moss":
        assert isinstance(
            algorithm,
            MOSS,
        )

        # 初始化阶段依次选择所有尚未访问的臂。
        assert first_actions == list(
            range(min(horizon, num_arms))
        )

        # 算法总更新次数应等于实验 horizon。
        assert int(
            algorithm.counts.sum()
        ) == horizon

        # 如果 horizon 足够长，每个臂至少访问一次。
        if horizon >= num_arms:
            assert np.all(
                algorithm.counts >= 1
            )

        # 算法内部计数应与 runner 记录一致。
        assert np.array_equal(
            algorithm.counts,
            action_counts,
        )

        # 算法内部奖励和应与 runner 记录一致。
        assert np.allclose(
            algorithm.reward_sums,
            reward_sums,
        )

        # 算法内部经验均值应与 runner 计算结果一致。
        assert np.allclose(
            algorithm.estimated_means,
            empirical_means,
        )

        # 所有内部统计量都必须是有限数。
        assert np.all(
            np.isfinite(
                algorithm.estimated_means
            )
        )
    # ---------- KL-UCB 专属测试 ----------

    if algorithm_name == "kl_ucb":
        assert isinstance(
        algorithm,
        KLUCB,
    )

    # 初始化阶段依次访问所有臂。
        assert first_actions == list(
        range(
            min(
                horizon,
                num_arms,
            )
        )
    )

    # 总更新次数必须等于实验 horizon。
        assert int(
        algorithm.counts.sum()
    ) == horizon

    # horizon 足够时，每个臂至少被初始化一次。
        if horizon >= num_arms:
            assert np.all(
            algorithm.counts >= 1
        )

    # 算法内部计数与 runner 统计一致。
        assert np.array_equal(
        algorithm.counts,
        action_counts,
    )

    # 奖励和一致。
        assert np.allclose(
        algorithm.reward_sums,
        reward_sums,
    )

    # 经验均值一致。
        assert np.allclose(
        algorithm.estimated_mean,
        empirical_means,
    )

    # 内部统计不能出现 NaN / inf。
        assert np.all(
        np.isfinite(
            algorithm.estimated_mean
        )
    )

    # ---------- GaussianUCB 专属测试 ----------

    if algorithm_name == "gaussian_ucb":
        assert isinstance(
        algorithm,
        GaussianUCB,
    )

    # 初始化阶段依次访问所有臂。
        assert first_actions == list(
        range(
            min(
                horizon,
                num_arms,
            )
        )
    )

    # 总更新次数正确。
        assert int(
        algorithm.counts.sum()
    ) == horizon

    # 初始化完成后每个臂至少访问一次。
        if horizon >= num_arms:
            assert np.all(
            algorithm.counts >= 1
        )

    # 算法内部计数与 runner 一致。
        assert np.array_equal(
        algorithm.counts,
        action_counts,
    )

    # 奖励和一致。
        assert np.allclose(
        algorithm.reward_sums,
        reward_sums,
    )

    # 经验均值一致。
        assert np.allclose(
        algorithm.estimated_mean,
        empirical_means,
    )

    # Gaussian reward 可以任意实数，
    # 但统计量必须保持有限。
        assert np.all(
        np.isfinite(
            algorithm.estimated_mean
        )
    )

        assert (
        np.isfinite(
            algorithm.known_std
        )
    )

        assert algorithm.known_std > 0

    # ---------- Gaussian Thompson Sampling 专属测试 ----------

    if algorithm_name == "gaussian_thompson_sampling":
        assert isinstance(
        algorithm,
        GaussianThompsonSampling,
    )

    # 总更新次数正确
        assert int(
        algorithm.counts.sum()
    ) == horizon

    # sufficient statistics 与 runner 一致
        assert np.array_equal(
        algorithm.counts,
        action_counts,
    )

        assert np.allclose(
        algorithm.reward_sums,
        reward_sums,
    )

    # posterior 合法
        assert np.all(
        np.isfinite(
            algorithm.posterior_means
        )
    )

        assert np.all(
        np.isfinite(
            algorithm.posterior_variances
        )
    )

        assert np.all(
        algorithm.posterior_variances > 0
    )

        assert np.all(
        algorithm.posterior_variances
        <= algorithm.prior_variance
    )

    # 根据整个实验的 sufficient statistics
    # 独立重算 posterior
        expected_variances = 1.0 / (
        1.0 / algorithm.prior_variance
        + action_counts
        / algorithm.noise_variance
    )

        expected_means = (
        expected_variances
        * (
            algorithm.prior_mean
            / algorithm.prior_variance
            + reward_sums
            / algorithm.noise_variance
        )
    )

        assert np.allclose(
        algorithm.posterior_variances,
        expected_variances,
    )

        assert np.allclose(
        algorithm.posterior_means,
        expected_means,
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
     fieldnames = [
    "environment",
    "algorithm",
    "horizon",
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

def validate_results_directory(results_dir:Path,config:dict[str,Any],)->None:
      if not results_dir.exists():
            return
      snapshot_path=(results_dir/"config_snapshot.json")
      if not snapshot_path.exists():
            if any(results_dir.iterdir()):
                  raise RuntimeError(
                        f"results directory '{results_dir}'"
                        "already contains files but has no"
                        "config_snapshot.json; refusing to"
                        "overwrite potentially stale results"
                  )
            return
      existing_config=load_config(str(snapshot_path))
      if existing_config != config:
            raise RuntimeError(
                  f"results directory '{results_dir}'"
                  "belongs to a different configuration;"
                  "use a different experiment_name or"
                  "remove the old results explicitly"
            )

def save_config_snapshot(config:dict[str,Any],output_path:Path)->None:
     output_path.parent.mkdir(parents=True,exist_ok=True)

     with output_path.open(mode="w",encoding="utf-8",)as file:
          json.dump(
               config,file,indent=2,ensure_ascii=False,
          )



def main()->None:
    args=parse_args()

    raw_config=load_config(
         config_path=args.config
    )

    config=normalize_config(raw_config)

    experiment_name=str(
         config["experiment_name"]
    )

    environment_config=dict(
         config["environment"]
    )

    horizons=[int(horizon) for horizon in config["horizons"]]

    algorithms=config["algorithms"]


    seeds=[
         int(seed) for seed in config["seeds"]
    ]

    environment_name = str(
        environment_config["name"]
    )

    results_dir = (
        Path("results")
        / experiment_name
    )
    validate_results_directory(results_dir=results_dir,config=config)

    save_config_snapshot(
        config=config,
        output_path=(
            results_dir
            / "config_snapshot.json"
        ),
    )

    num_experiments=len(algorithms)*len(seeds)*len(horizons)

    print(
        f"running {num_experiments} experiments"
    )



    for algorithm_config in algorithms:
        algorithm_name=str(algorithm_config["name"])

        for horizon in horizons:
             for seed in seeds:
                print(
                    f"running "
                    f"experiment={experiment_name}, "
                    f"environment={environment_name}, "
                    f"algorithm={algorithm_name}, "
                    f"horizon={horizon}, "
                    f"seed={seed}"
                )

                records = run_single_experiment(
                    seed=seed,
                    environment_config=(
                        environment_config
                    ),
                    horizon=horizon,
                    algorithm_config=(
                        algorithm_config
                    ),
                )

                filename = (
                    f"{environment_name}_"
                    f"{algorithm_name}_"
                    f"T{horizon}_"
                    f"seed{seed}.csv"
                )

                output_path = (
                    results_dir
                    / filename
                )

                save_records(
                    records=records,
                    output_path=output_path,
                )



if __name__ == "__main__":
    main()



