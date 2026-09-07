from __future__ import annotations

import numpy as np

from .base import BanditAlgorithm

class GaussianThompsonSampling(BanditAlgorithm):
    def __init__(
            self,num_arms:int,
            prior_mean:float,
            prior_variance:float,
            noise_variance:float,
            rng:np.random.Generator,
    )->None:
        super().__init__()

        if (
            not isinstance(num_arms, int)
            or isinstance(num_arms, bool)
            or num_arms <= 0
        ):
            raise ValueError(
                "num_arms must be a positive integer"
            )

        if not np.isfinite(prior_mean):
            raise ValueError("prior_mean must be finite")

        if not np.isfinite(prior_variance) or prior_variance <= 0:
            raise ValueError(
                "prior_variance must be positive"
            )

        if not np.isfinite(noise_variance) or noise_variance <= 0:
            raise ValueError(
                "noise_variance must be positive"
            )

        self.num_arms=num_arms

        self.prior_mean=float(prior_mean)
        self.prior_variance=float(prior_variance)
        self.noise_variance=float(noise_variance)

        self.rng=rng

        self.counts=np.zeros(
            num_arms,dtype=int,
        )

        self.reward_sums=np.zeros(
            num_arms,
            dtype=float,
        )

        self.posterior_means=np.full(
            num_arms,
            self.prior_mean,
            dtype=float,
        )

        self.posterior_variances=np.full(
            num_arms,
            self.prior_variance,
            dtype=float,
        )

    def select_action(self) -> int:
        samples=self.rng.normal(
            loc=self.posterior_means,
            scale=np.sqrt(
                self.posterior_variances,
            ),
        )
        return int(np.argmax(samples))

    def update(
            self,action:int,reward:float,
    )->None:
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise ValueError("action must be an integer")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        if not 0 <= action < self.num_arms:
            raise ValueError(
                f"action must be in "
                f"[0, {self.num_arms}), "
                f"got {action}"
            )

        if not np.isfinite(reward):
            raise ValueError(
                "reward must be finite"
            )

        self.counts[action]+=1
        self.reward_sums[action]+=reward

        n_i=self.counts[action]
        S_i=self.reward_sums[action]

        self.posterior_variances[action]=1/(
            1/self.prior_variance+n_i/self.noise_variance
        )

        self.posterior_means[action]=(
            self.posterior_variances[action]*(
                self.prior_mean/self.prior_variance+
                S_i/self.noise_variance
            )
        )