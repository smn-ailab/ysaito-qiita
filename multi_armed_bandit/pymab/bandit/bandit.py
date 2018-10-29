"""This module contains bandit classes."""
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np

from pymab.utils import _check_x_input, sigmoid


class BaseBandit():
    """Base class for all bandits in pymab.

    Parameters
    ----------
    n_arms: int
        The number of given bandit arms.

    n_features: int
        The dimention of context vectors.

    noise: int, optional(default=None)
        The variance of gaussian noise on linear model of contextual rewards.

    contextual: bool, optional(default=False)
        Whether rewards are models contextual or not.

    """

    def __init__(self, n_arms: int, n_features: Optional[int]=None, scale: float=0.1,
                 noise: float=0.1, contextual: bool=False) -> None:
        """Initialize Class."""
        self.rewards = 0.0
        self.regrets = 0.0
        self.n_arms = n_arms
        self.contextual = contextual
        if self.contextual:
            self.scale = scale
            self.noise = noise
            self.n_features = n_features
            self.params = np.random.multivariate_normal(np.zeros(self.n_features),
                                                        self.scale * np.identity(self.n_features), size=self.n_arms).T
        else:
            self.mu = np.random.uniform(low=0.01, high=0.1, size=n_arms)
            self.mu_max, self.best_arm = np.max(self.mu), np.argmax(self.mu)

    @abstractmethod
    def pull(self, chosen_arm: int, x: Optional[np.ndarray]=None) -> Union[int, float]:
        """Pull arms.

        chosen_arm: int
            The chosen arm.

        x : array-like, shape = (n_features, ), optional(default=None)
            A test sample.

        """
        pass


class BernoulliBandit(BaseBandit):
    """Bernoulli Bandit.

    Parameters
    ----------
    n_arms: int
        The number of given bandit arms.

    n_features: int
        The dimention of context vectors.

    noise: float, optional(default=0.1)
        The variance of gaussian noise on linear model of contextual rewards.

    contextual: bool, optional(default=False)
        Whether rewards are models contextual or not.

    """

    def __init__(self, n_arms: int, n_features: Optional[int]=None, scale: float=0.1,
                 noise: float=0.1, contextual: bool=False) -> None:
        """Initialize Class."""
        super().__init__(n_arms=n_arms, n_features=n_features, scale=scale, noise=noise, contextual=contextual)

    def pull(self, chosen_arm: int, x: Optional[np.ndarray]=None) -> Union[int, float]:
        """Pull arms.

        chosen_arm: int
            The chosen arm.

        x : array-like, shape = (n_features, ), optional(default=None)
            A test sample.

        """
        if self.contextual:
            x, e = _check_x_input(x), np.random.normal(loc=0, scale=self.noise)
            mu = np.ravel(x.T @ self.params)
            reward, regret, self.best_arm = \
                np.random.binomial(n=1, p=sigmoid(mu[chosen_arm] + e)), \
                np.max(mu) - mu[chosen_arm], np.argmax(mu)
        else:
            reward, regret = \
                np.random.binomial(n=1, p=self.mu[chosen_arm]), self.mu_max - self.mu[chosen_arm]

        self.rewards += reward
        self.regrets += regret

        return reward


class GaussianBandit(BaseBandit):
    """Gaussian Bandit.

    Parameters
    ----------
    n_arms: int
        The number of given bandit arms.

    n_features: int
        The dimention of context vectors.

    noise: int, optional(default=None)
        The variance of gaussian noise on linear model of contextual rewards.

    contextual: bool, optional(default=False)
        Whether rewards are modeled as contextual or not.

    """

    def __init__(self, n_arms: int, n_features: Optional[int]=None, scale: float=0.1,
                 noise: float=0.1, contextual: bool=False) -> None:
        """Initialize Class."""
        super().__init__(n_arms=n_arms, n_features=n_features, scale=scale, noise=noise, contextual=contextual)

    def pull(self, chosen_arm: int, x: Optional[np.ndarray]=None) -> Union[int, float]:
        """Pull arms.

        Parameters
        ----------
        chosen_arm: int
            The chosen arm.

        x : array-like, shape = (n_features, ), optional(default=None)
            A test sample.

        """
        if self.contextual:
            x, e = _check_x_input(x), np.random.normal(loc=0, scale=self.noise)
            mu = np.ravel(x.T @ self.params)
            reward, regret, self.best_arm = \
                np.random.normal(loc=mu[chosen_arm] + e, scale=self.scale), \
                np.max(mu) - mu[chosen_arm], np.argmax(mu)
        else:
            reward, regret = \
                np.random.normal(loc=self.mu[chosen_arm], scale=self.scale), self.mu_max - self.mu[chosen_arm]

        self.rewards += reward
        self.regrets += regret

        return reward
