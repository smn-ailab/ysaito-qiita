"""This Module contains Stochastic Bandit Policies."""
from typing import Union

import numpy as np

from .base import BasePolicy, BaseThompsonSampling


class EpsilonGreedy(BasePolicy):
    """Epsilon Greedy.

    Parameters
    ----------
    n_arms: int
        The number of given bandit arms.

    epsilon: float
        The hyper-parameter which represents how often the algorithm explore.

    batch_size: int, optional (default=1)
        The number of data given in each batch.

    """

    def __init__(self, n_arms: int, epsilon: float, batch_size: int=1) -> None:
        """Initialize class."""
        super().__init__(n_arms, batch_size)
        if not isinstance(epsilon, float):
            raise TypeError("The hyper-parameter 'epsilon' must be a float value.")
        assert (epsilon <= 1.0) and (epsilon >= 0.0), "The hyper-parameter 'epsilon' must be between 0 and 1."

        self.epsilon = epsilon
        self.name = f"EpsilonGreedy(ε={self.epsilon})"

    def select_arm(self) -> int:
        """Select arms according to the policy for new data.

        Returns
        -------
        result: int
            The selected arm.

        """
        result = np.random.randint(self.values.shape[0])
        if np.random.rand() > self.epsilon:
            result = np.argmax(self.values)
        return result


class SoftMax(BasePolicy):
    """SoftMax.

    Parameters
    ----------
    n_arms: int
        The number of given bandit arms.

    tau: float
        The hyper-parameter which represents how often the algorithm explores.

    batch_size: int, optional (default=1)
        The number of data given in each batch.

    """

    def __init__(self, n_arms: int, tau: float, batch_size: int=1) -> None:
        """Initialize class."""
        super().__init__(n_arms, batch_size)
        if not isinstance(tau, float):
            raise TypeError("The hyper-parameter 'tau' must be a float.")
        assert (tau <= 1) and (tau >= 0), "The hyper-parameter 'tau' must be between 0 and 1."

        self.tau = tau
        self.name = f"SoftMax(Ï„={self.tau})"

    def select_arm(self) -> int:
        """Select arms according to the policy for new data.

        Returns
        -------
        result: int
            The selected arm.

        """
        z = np.sum(np.exp(self.values) / self.tau)
        probs = (np.exp(self.values) / self.tau) / z
        return np.random.choice(self.counts.shape[0], p=probs)


class UCB1(BasePolicy):
    """Upper Confidence Bound.

    Parameters
    ----------
    n_arms: int
        The number of given bandit arms.

    batch_size: int, optional (default=1)
        The number of data given in each batch.

    """

    name = "UCB1"

    def __init__(self, n_arms: int, batch_size: int=1) -> None:
        """Initialize class."""
        super().__init__(n_arms, batch_size)

    def select_arm(self) -> int:
        """Select arms according to the policy for new data.

        Returns
        -------
        result: int
            The selected arm.

        """
        if 0 in self.counts:
            result = np.argmin(self.counts)
        else:
            ucb_values, total_counts = np.zeros(self.n_arms), np.sum(self.counts)
            bounds = np.sqrt(2 * np.log(total_counts) / self.counts)
            result = np.argmax(self.values + bounds)

        return result


class UCBTuned(BasePolicy):
    """Uppler Confidence Bound Tuned.

    Parameters
    ----------
    n_arms: int
        The number of given bandit arms.

    batch_size: int, optional (default=1)
        The number of data given in each batch.

    """

    name = "UCBTuned"

    def __init__(self, n_arms: int, batch_size: int=1) -> None:
        """Initialize class."""
        super().__init__(n_arms, batch_size)

        self.sigma = np.zeros(self.n_arms, dtype=float)
        self._sigma = np.zeros(self.n_arms, dtype=float)

    def select_arm(self) -> int:
        """Select arms according to the policy for new data.

        Returns
        -------
        result: int
            The selected arm.

        """
        if 0 in self.counts:
            result = np.argmin(self.counts)
        else:
            ucb_values, total_counts = np.zeros(self.n_arms), np.sum(self.counts)
            bounds1 = np.log(total_counts) / self.counts
            bounds2 = np.minimum(1 / 4, self.sigma + 2 * np.log(total_counts) / self.counts)
            result = np.argmax(self.values + np.sqrt(bounds1 * bounds2))

        return result

    def update(self, chosen_arm: int, reward: Union[int, float]) -> None:
        """Update the reward information about earch arm.

        Parameters
        ----------
        chosen_arm: int
            The chosen arm.

        reward: int, float
            The observed reward value from the chosen arm.

        """
        if not isinstance(chosen_arm, int):
            raise TypeError("chosen_arm must be a float.")
        if not isinstance(reward, (int, float)):
            raise TypeError("reward must be an integer or float.")

        self.data_size += 1
        self.counts[chosen_arm] += 1

        n = self.counts[chosen_arm]
        new_value = (self.values[chosen_arm] * (n - 1) / n) + (reward / n)
        self._values[chosen_arm] = new_value
        new_sigma = ((n * ((self._sigma[chosen_arm] ** 2) + (self._values[chosen_arm] ** 2)) + reward ** 2) / (n + 1)) - new_value ** 2
        self._sigma[chosen_arm] = new_sigma

        if self.data_size % self.batch_size == 0:
            self.values = np.copy(self._values)
            self.sigma = np.copy(self._sigma)


class ThompsonSampling(BaseThompsonSampling):
    """Bernoulli Thompson Sampling."""

    name = "ThompsonSampling"

    def __init__(self, n_arms: int, alpha: float=1.0, beta: float=1.0, batch_size: int=1) -> None:
        """Initialize class.

        Parameters
        ----------
        n_arms: int
            The number of given bandit arms.

        alpha: float (default=1.0)
            Hyperparameter alpha for beta distribution.

        beta: float (default=1.0)
            Hyperparameter beta for beta distribution.

        batch_size: int, optional (default=1)
            The number of data given in each batch.

        """
        super().__init__(n_arms, batch_size)

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float.")
        if not isinstance(beta, float):
            raise TypeError("beta must be a float.")
        assert alpha >= 0, "alpha must be a non-negative value"
        assert beta >= 0, "beta must be a non-negative value"

        self.alpha = alpha
        self.beta = beta
        self.counts_alpha = np.zeros(self.n_arms, dtype=int)
        self.counts_beta = np.zeros(self.n_arms, dtype=int)
        self._counts_alpha = np.zeros(self.n_arms, dtype=int)
        self._counts_beta = np.zeros(self.n_arms, dtype=int)

    def select_arm(self) -> int:
        """Select arms according to the policy for new data.

        Returns
        -------
        result: int
            The selected arm.

        """
        theta = np.random.beta(a=self.counts_alpha + self.alpha, b=self.counts_beta + self.beta)
        result = np.argmax(theta)
        return result

    def update(self, chosen_arm: int, reward: Union[int, float]) -> None:
        """Update the reward information about earch arm.

        Parameters
        ----------
        chosen_arm: int
            The chosen arm.

        reward: int, float
            The observed reward value from the chosen arm.

        """
        super().update(chosen_arm, reward)
        if reward == 1:
            self._counts_alpha[chosen_arm] += 1
        else:
            self._counts_beta[chosen_arm] += 1

        if self.data_size % self.batch_size == 0:
            self.counts_alpha = np.copy(self._counts_alpha)
            self.counts_beta = np.copy(self._counts_beta)


class GaussianThompsonSampling(BaseThompsonSampling):
    """Gaussian Thompson Sampling.

    Parameters
    ----------
    n_arms: int
        The number of given bandit arms.

    mu_prior: float (default=1.0)
        The hyperparameter mu for prior gaussian distribution.

    lam_likelihood: float (default=1.0)
        The hyperparameter lamda for likelihood gaussian distribution.

    lam_prior: float (defaut=1.0)
        The hyperparameter lamda for prior gaussian distribution.

    batch_size: int, optional (default=1)
        The number of data given in each batch.

    """

    name = "GaussianThompsonSampling"

    def __init__(self, n_arms: int, mu_prior: float=0.0, lam_likelihood: float=1.0, lam_prior: float=1.0, batch_size: int=1) -> None:
        """Initialize class."""
        super().__init__(n_arms, batch_size)
        if not isinstance(mu_prior, float):
            raise TypeError("mu_prior must be a float.")
        if not isinstance(lam_likelihood, float):
            raise TypeError("lam_likelihood must be a float.")
        if not isinstance(lam_prior, float):
            raise TypeError("lam_prior must be a float.")
        assert lam_likelihood >= 0, "lam_likelihood must be a non-negative value"
        assert lam_prior >= 0, "lam_prior must be a non-negative value"

        self.mu_prior = mu_prior
        self.lam_prior = lam_prior
        self.lam_likelihood = lam_likelihood

        self.counts = np.zeros(self.n_arms, dtype=int)
        self.values = np.zeros(self.n_arms, dtype=float)
        self.mu = np.zeros(self.n_arms, dtype=float)
        self.lam = np.ones(self.n_arms, dtype=float) * self.lam_prior

    def select_arm(self) -> int:
        """Select arms according to the policy for new data.

        Returns
        -------
        result: int
            The selected arm.

        """
        theta = np.random.normal(loc=self.mu, scale=(1.0 / self.lam))
        result = np.argmax(theta)
        return result

    def update(self, chosen_arm: int, reward: Union[int, float]) -> None:
        """Update the reward information about earch arm.

        Parameters
        ----------
        chosen_arm: int
            The chosen arm.

        reward: int, float
            The observed reward value from the chosen arm.

        """
        super().update(chosen_arm, reward)
        self.values[chosen_arm] += reward

        if self.data_size % self.batch_size == 0:
            self.lam = self.counts * self.lam_likelihood + self.lam_prior
            self.mu = (self.lam_likelihood * self.values + self.lam_prior * self.mu_prior) / self.lam
