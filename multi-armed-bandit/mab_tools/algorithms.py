import math
import random
from abc import ABC, abstractmethod

import numpy as np


class MABInterface(ABC):

    @abstractmethod
    def select_arm(self) -> None:
        pass

    @abstractmethod
    def update(self) -> None:
        pass


class EpsilonGreedy(MABInterface):
    def __init__(self, epsilon: float, n_arms: int) -> None:
        self.epsilon = epsilon
        self.n_arms = n_arms
        self.counts = [0] * self.n_arms
        self.values = [0.0] * self.n_arms

    def select_arm(self) -> int:
        result = random.randrange(len(self.values))
        if random.random() > self.epsilon:
            result = np.argmax(self.values)
        return result

    def update(self, chosen_arm: int, reward: float) -> None:
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = (value * (n - 1) / n) + reward / n
        self.values[chosen_arm] = new_value


class SoftMax(MABInterface):
    def __init__(self, temperature: float, n_arms: int) -> None:
        self.temperature = temperature
        self.n_arms = n_arms
        self.counts = [0] * self.n_arms
        self.values = [0.0] * self.n_arms

    def select_arm(self) -> int:
        z = sum([math.exp(v / self.temperature) for v in self.values])
        probs = [math.exp(v / self.temperature) / z for v in self.values]
        return np.random.choice(len(self.counts), p=probs)

    def update(self, chosen_arm: int, reward: float) -> None:
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]

        new_value = ((n - 1) / n) * self.values[chosen_arm] + (1 / n) * reward
        self.values[chosen_arm] = new_value


class UCB1(MABInterface):
    def __init__(self, n_arms: int) -> None:
        self.n_arms = n_arms
        self.counts = [0] * self.n_arms
        self.values = [0.0] * self.n_arms

    def select_arm(self) -> int:
        if 0 in self.counts:
            result = self.counts.index(0)
        else:
            ucb_values = [0.0] * self.n_arms
            total_counts = sum(self.counts)
            for arm in range(self.n_arms):
                bonus = math.sqrt(2 * math.log(total_counts) / self.counts[arm])
                ucb_values[arm] = self.values[arm] + bonus
            result = np.argmax(ucb_values)

        return result

    def update(self, chosen_arm: int, reward: float) -> None:
        self.counts[chosen_arm] += 1
        n = sum(self.counts)

        new_value = ((n - 1) / n) * self.values[chosen_arm] + (1 / n) * reward
        self.values[chosen_arm] = new_value


class TompsonSampling(MABInterface):
    def __init__(self, n_arms: int, alpha=1.0, beta=1.0) -> None:
        self.alpha = alpha
        self.beta = beta
        self.n_arms = n_arms
        self.counts_alpha = [0] * self.n_arms
        self.counts_beta = [0] * self.n_arms
        self.values = [0.0] * self.n_arms

    def select_arm(self) -> int:
        theta = [random.betavariate(self.counts_alpha[arm] + self.alpha,
                                    self.counts_beta[arm] + self.beta)
                 for arm in range(len(self.counts_alpha))]
        return np.argmax(theta)

    def update(self, chosen_arm: int, reward: float) -> None:
        if reward == 1:
            self.counts_alpha[chosen_arm] += 1
        else:
            self.counts_beta[chosen_arm] += 1

        n = sum(self.counts_alpha) + sum(self.counts_beta)

        new_value = ((n - 1) / n) * self.values[chosen_arm] + (1 / n) * reward
        self.values[chosen_arm] = new_value
