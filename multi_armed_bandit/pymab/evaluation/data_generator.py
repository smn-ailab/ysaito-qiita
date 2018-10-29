"""Generate data with bandit feedback."""
import copy
from typing import List, Optional, Tuple, Union

import numpy as np
from pandas import DataFrame

from pymab.bandit import BaseBandit
from pymab.policy import BaseContextualPolicy, BasePolicy, BaseThompsonSampling


class DataGenerator():
    """Data generator.

    Parameters
    ----------
    n_arms: int
        The number of given bandit arms.

    n_features: int
        The dimention of context vectors.

    noise: int, optional(default=None)
        The variance of gaussian noise on linear model of contextual rewards.

    n_rounds: int
        The number of rounds in a simulation.

    randomized: bool, optional(default=False)
        Whether the past policy is randomized or not.

    """

    def __init__(self,
                 policy: Union[BasePolicy, BaseThompsonSampling, BaseContextualPolicy],
                 bandit: BaseBandit,
                 n_rounds: int,
                 randomized: bool=False) -> None:
        """Initialize Class."""
        self.policy = policy
        self.bandit = bandit
        self.n_arms = self.bandit.n_arms
        self.n_features = self.bandit.n_features
        self.n_rounds = n_rounds
        self.randomized = randomized

    def generate_data(self) -> Tuple:
        """Generate data."""
        data = np.concatenate([np.expand_dims(np.random.randint(2, size=self.n_features), axis=1)
                               for i in np.arange(self.n_rounds)], axis=1)

        chosen_arms = np.zeros(self.n_rounds, dtype=int)
        rewards = np.zeros(self.n_rounds)
        p, b = copy.deepcopy(self.policy), copy.deepcopy(self.bandit)

        for t in np.arange(self.n_rounds):
            x = np.copy(data[:, t])

            if self.randomized:
                chosen_arm = np.random.randint(self.n_arms)
            else:
                chosen_arm = p.select_arm(x) if p._policy_type == "contextual" else p.select_arm()

            reward = b.pull(x=x, chosen_arm=chosen_arm)
            chosen_arms[t], rewards[t] = chosen_arm, reward
            p.update(x, chosen_arm, reward) if p._policy_type == "contextual" else p.update(chosen_arm, reward)

        return data.T, chosen_arms, rewards
