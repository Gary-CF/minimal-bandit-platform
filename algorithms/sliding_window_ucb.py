from collections import deque

import numpy as np

from algorithms.base import BanditAlgorithm


class SlidingWindowUCB(BanditAlgorithm):
    def __init__(self, num_arms: int, window_size: int) -> None:
        if not isinstance(num_arms, int) or isinstance(num_arms, bool) or num_arms <= 0:
            raise ValueError("num_arms must be a positive integer")

        if (
            not isinstance(window_size, int)
            or isinstance(window_size, bool)
            or window_size <= 0
        ):
            raise ValueError("window_size must be a positive integer")

        self.num_arms = num_arms
        self.window_size = window_size
        self.history = deque()
        self.counts = np.zeros(num_arms, dtype=int)
        self.reward_sums = np.zeros(num_arms, dtype=float)
        self.completed_steps = 0

    def update(self, action: int, reward: float) -> None:
        # TODO：先检查输入，再更新窗口和统计量。
        if not 0.0 <= reward <= 1.0:
            raise ValueError("current bounded policy requires reward in [0, 1]")
        if isinstance(action, (bool, np.bool_)) or not isinstance(
            action, (int, np.integer)
        ):
            raise ValueError("action must be an integer")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        if not 0 <= action < self.num_arms:
            raise IndexError(f"action {action} is outside[0,{self.num_arms})")

        self.history.append((action, reward))
        self.counts[action] += 1
        self.reward_sums[action] += reward

        if len(self.history) > self.window_size:
            action_t, reward_t = self.history.popleft()
            self.counts[action_t] -= 1
            self.reward_sums[action_t] -= reward_t

        self.completed_steps += 1

    def select_action(self) -> int:
        if self.completed_steps< self.num_arms:
            return self.completed_steps
        empty_arms = np.flatnonzero(self.counts==0)
        if len(empty_arms)>0:
            return int(empty_arms[0])
        t=self.completed_steps+1
        m_t=min(t,self.window_size)

        means=self.reward_sums/self.counts
        confidence_radios=np.sqrt(2*np.log(m_t)/self.counts)
        scores=means+confidence_radios
        return int(np.argmax(scores))
