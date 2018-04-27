"""This Module contains basic Multi-Armed Bandit Algorithms."""
import random
from abc import ABC, abstractmethod

import numpy as np


class MABInterface(ABC):
    """Abstract base class for various Multi-Armed Bandit Algorithms."""

    @abstractmethod
    def select_arm(self) -> None:
        """Decide which arm should be selected."""
        pass

    @abstractmethod
    def update(self) -> None:
        """Update the information about the arms."""
        pass

    @abstractmethod
    def batch_update(self) -> None:
        """Update the information about the arms."""
        pass


class EpsilonGreedy(MABInterface):
    """Epsilon Greedy Algorithm for Multi-Armed Bandit problems."""

    def __init__(self, epsilon: float, n_arms: int, batch_size: int=None) -> None:
        """Initialize class.

        :param epsilon:  the hyper-parameter which represents how often the algorithm explore.
        :param n_arms: the number of given arms.
        """
        self.epsilon = epsilon
        self.n_arms = n_arms
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)

        self._values = np.zeros(self.n_arms)
        self.batch_size = batch_size
        self.data_size = 0

    def select_arm(self) -> int:
        """Decide which arm should be selected.

        :return: index of the selected arm.
        """
        result = random.randrange(self.values.shape[0])
        if np.random.rand() > self.epsilon:
            result = np.argmax(self.values)
        return result

    def update(self, chosen_arm: int, reward: float) -> None:
        """Update the information about the arms.

        :param chosen_arm: index of the chosen arm.
        :param reward: reward from the chosen arm.
        """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = (value * (n - 1) / n) + reward / n
        self.values[chosen_arm] = new_value

    def batch_update(self, chosen_arm: int, reward: float) -> None:
        self.data_size += 1
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self._values[chosen_arm]
        new_value = (value * (n - 1) / n) + reward / n
        self._values[chosen_arm] = new_value

        if self.data_size % self.batch_size == 0:
            self.values = self._values
        else:
            pass


class SoftMax(MABInterface):
    """SoftMax Algorithm for Multi-Armed Bandit problems."""

    def __init__(self, temperature: float, n_arms: int, batch_size: int=None) -> None:
        """Initialize class.

        :param temperature:  the hyper-parameter which represents how much the algorithm uses explored information about arms.
        :param n_arms: the number of given arms.
        """
        self.temperature = temperature
        self.n_arms = n_arms
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)

        self._values = np.zeros(self.n_arms)
        self.batch_size = batch_size
        self.data_size = 0

    def select_arm(self) -> int:
        """Decide which arm should be selected.

        :return: index of the selected arm.
        """
        z = np.sum(np.exp(self.values) / self.temperature)
        probs = (np.exp(self.values) / self.temperature) / z
        return np.random.choice(self.counts.shape[0], p=probs)

    def update(self, chosen_arm: int, reward: float) -> None:
        """Update the information about the arms.

        :param chosen_arm: index of the chosen arm.
        :param reward: reward from the chosen arm.
        """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]

        new_value = ((n - 1) / n) * self.values[chosen_arm] + (1 / n) * reward
        self.values[chosen_arm] = new_value

    def batch_update(self, chosen_arm: int, reward: float) -> None:
        self.data_size += 1
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self._values[chosen_arm]
        new_value = (value * (n - 1) / n) + reward / n
        self._values[chosen_arm] = new_value

        if self.data_size % self.batch_size == 0:
            self.values = self._values
        else:
            pass


class UCB1(MABInterface):
    """Upper Confidence Bound1 Algorithm for Multi-Armed Bandit problems."""

    def __init__(self, n_arms: int, batch_size: int=None) -> None:
        """Initialize class.

        :param n_arms: the number of given arms.
        """
        self.n_arms = n_arms
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)

        self._values = np.zeros(self.n_arms)
        self.batch_size = batch_size
        self.data_size = 0

    def select_arm(self) -> int:
        """Decide which arm should be selected.

        :return: index of the selected arm.
        """
        if 0 in self.counts:
            result = np.where(self.counts == 0)[0][0]
        else:
            ucb_values = np.zeros(self.n_arms)
            total_counts = sum(self.counts)
            bounds = np.sqrt(2 * np.log(total_counts) / self.counts)
            ucb_values = self.values + bounds
            result = np.argmax(ucb_values)

        return result

    def update(self, chosen_arm: int, reward: float) -> None:
        """Update the information about the arms.

        :param chosen_arm: index of the chosen arm.
        :param reward: reward from the chosen arm.
        """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]

        new_value = ((n - 1) / n) * self.values[chosen_arm] + (1 / n) * reward
        self.values[chosen_arm] = new_value

    def batch_update(self, chosen_arm: int, reward: float) -> None:
        self.data_size += 1
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self._values[chosen_arm]
        new_value = (value * (n - 1) / n) + reward / n
        self._values[chosen_arm] = new_value

        if self.data_size % self.batch_size == 0:
            self.values = self._values
        else:
            pass


class ThompsonSampling(MABInterface):
    """Thompson Sampling Algorithm for Multi-Armed Bandit problems."""

    def __init__(self, n_arms: int, alpha=1.0, beta=1.0, batch_size: int=None) -> None:
        """Initialize class.

        :param n_arms: the number of given arms.
        :param alpha: a hyper-parameter which represents alpha of the prior beta distribution. (default=1.0)
        :param beta: a hyper-parameter which represents beta of the prior beta distribution. (default=1.0)
        """
        self.alpha = alpha
        self.beta = beta
        self.n_arms = n_arms
        self.counts = np.zeros(self.n_arms)
        self.counts_alpha = np.zeros(self.n_arms)
        self.counts_beta = np.zeros(self.n_arms)

        self._counts_alpha = np.zeros(self.n_arms)
        self._counts_beta = np.zeros(self.n_arms)
        self.batch_size = batch_size
        self.data_size = 0

    def select_arm(self) -> int:
        """Decide which arm should be selected.

        :return: index of the selected arm.
        """
        theta = np.random.beta(a=self.counts_alpha + self.alpha, b=self.counts_beta + self.beta)
        return np.argmax(theta)

    def update(self, chosen_arm: int, reward: float) -> None:
        """Update the information about the arms.

        :param chosen_arm: index of the chosen arm.
        :param reward: reward from the chosen arm.
        """
        self.counts[chosen_arm] += 1
        if reward == 1:
            self.counts_alpha[chosen_arm] += 1
        else:
            self.counts_beta[chosen_arm] += 1

    def batch_update(self, chosen_arm: int, reward: float) -> None:
        self.data_size += 1
        self.counts[chosen_arm] += 1
        if reward == 1:
            self._counts_alpha[chosen_arm] += 1
        else:
            self._counts_beta[chosen_arm] += 1

        if self.data_size % self.batch_size == 0:
            self.counts_alpha = self._counts_alpha
            self.counts_beta = self._counts_beta
        else:
            pass
