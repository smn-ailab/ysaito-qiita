"""Base classes for all policies."""
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

from pymab.utils import (_check_contextual_input, _check_stochastic_input,
                         _check_update_input, _check_x_input)


class PolicyInterface(ABC):
    """Abstract Base class for all stochastic policies in pymab."""

    @abstractmethod
    def select_arm(self) -> int:
        """Select arms according to the policy for new data.

        Returns
        -------
        result: int
            The selected arm.

        """
        pass

    @abstractmethod
    def update(self, chosen_arm: int, reward: Union[int, float]) -> None:
        """Update the reward information about earch arm.

        Parameters
        ----------
        chosen_arm: int
            The chosen arm.

        reward: int, float
            The observed reward value from the chosen arm.

        """
        pass


class ContextualPolicyInterface(ABC):
    """Abstract base class for all contextual policies in pymab."""

    @abstractmethod
    def select_arm(self, x: np.ndarray) -> int:
        """Select arms according to the policy for new data.

        Parameters
        ----------
        x : array-like, shape = (n_features, )
            A test sample.

        Returns
        -------
        result: int
            The selected arm.

        """
        pass

    @abstractmethod
    def update(self, x: np.ndarray, chosen_arm: int, reward: Union[int, float]) -> None:
        """Update the reward and parameter information about earch arm.

        Parameters
        ----------
        x : array-like, shape = (n_features, )
            A test sample.

        chosen_arm: int
            The chosen arm.

        reward: int, float
            The observed reward value from the chosen arm.

        """
        pass


class BasePolicy(PolicyInterface):
    """Base class for basic stochastic policies in pymab.

    Parameters
    ----------
    n_arms: int
        The number of given bandit arms.

    batch_size: int, optional (default=1)
        The number of data given in each batch.

    """

    _policy_type = "stochastic"

    def __init__(self, n_arms: int, batch_size: int=1) -> None:
        """Initialize class."""
        _check_stochastic_input(n_arms, batch_size)

        self.n_arms = n_arms
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.values = np.zeros(self.n_arms)
        self._values = np.zeros(self.n_arms)
        self.batch_size = batch_size
        self.data_size = 0

    def update(self, chosen_arm: int, reward: Union[int, float]) -> None:
        """Update the reward information about earch arm.

        Parameters
        ----------
        chosen_arm: int
            The chosen arm.

        reward: int, float
            The observed reward value from the chosen arm.

        """
        _check_update_input(chosen_arm, reward)

        self.data_size += 1
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self._values[chosen_arm]
        new_value = (self.values[chosen_arm] * (n - 1) / n) + (reward / n)
        self._values[chosen_arm] = new_value

        if self.data_size % self.batch_size == 0:
            self.values = np.copy(self._values)


class BaseThompsonSampling(PolicyInterface):
    """Base class for stochastic Thompson Sampling policies."""

    _policy_type = "stochastic"

    def __init__(self, n_arms: int, batch_size: int=1) -> None:
        """Initialize class.

        Parameters
        ----------
        n_arms: int
            The number of given bandit arms.

        batch_size: int (default=1)
            The number of data given in each batch.

        """
        _check_stochastic_input(n_arms, batch_size)

        self.n_arms = n_arms
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.values = np.zeros(self.n_arms)
        self._values = np.zeros(self.n_arms)
        self.batch_size = batch_size
        self.data_size = 0

    def update(self, chosen_arm: int, reward: Union[int, float]) -> None:
        """Update the reward information about earch arm.

        Parameters
        ----------
        chosen_arm: int
            The chosen arm.

        reward: int, float
            The observed reward value from the chosen arm.

        """
        _check_update_input(chosen_arm, reward)

        self.data_size += 1
        self.counts[chosen_arm] += 1


class BaseContextualPolicy(ContextualPolicyInterface):
    """Base class for all contextual bandit algorithms.

    Parameters
    ----------
    n_arms: int
        The number of given bandit arms.

    n_features: int
        The dimention of context vectors.

    warmup: int, optional(default=1)
        The minimum number of pull of earch arm.

    batch_size: int, optional (default=1)
        The number of data given in each batch.

    """

    _policy_type = "contextual"

    def __init__(self, n_arms: int, n_features: int, warmup: int=1, batch_size: int=1) ->None:
        """Initialize class."""
        _check_contextual_input(n_arms, n_features, warmup, batch_size)

        self.n_arms = n_arms
        self.n_features = n_features
        self.warmup = warmup

        self.rewards = 0.0
        self.counts = np.zeros(self.n_arms, dtype=int)

        self.batch_size = batch_size
        self.data_size = 0
