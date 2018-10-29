"""This Module contains Contextual Bandit Policies."""
import copy
import math
import random
from typing import Optional, Tuple, Union

import numpy as np
from scipy.stats import norm

from pymab.utils import _check_x_input

from .base import BaseContextualPolicy


class LinUCB(BaseContextualPolicy):
    """Linear Upper Confidence Bound.

    Parameters
    ----------
    n_arms: int
        The number of given bandit arms.

    n_features: int
        The dimention of context vectors.

    alpha: float, optional(default=1.0)
        The hyper-parameter which represents how often the algorithm explores.

    warmup: int, optional(default=1)
        The minimum number of pull of earch arm.

    batch_size: int, optional (default=1)
        The number of data given in each batch.

    References
    -------
    [1] L. Li, W. Chu, J. Langford, and E. Schapire.
        A contextual-bandit approach to personalized news article recommendation.
        In Proceedings of the 19th International Conference on World Wide Web, pp. 661–670. ACM, 2010.

    """

    def __init__(self, n_arms: int, n_features: int, alpha: float=1.0, warmup: int=1, batch_size: int=1) -> None:
        """Initialize class."""
        super().__init__(n_arms, n_features, warmup, batch_size)

        self.alpha = alpha
        self.name = f"LinUCB(α={self.alpha})"

        self.theta_hat = np.zeros((self.n_features, self.n_arms))  # d * k
        self.A_inv = np.concatenate([np.identity(self.n_features)
                                     for i in np.arange(self.n_arms)]).reshape(self.n_arms, self.n_features, self.n_features)  # k * d * d
        self.b = np.zeros((self.n_features, self.n_arms))  # d * k

        self._A_inv = np.concatenate([np.identity(self.n_features)
                                      for i in np.arange(self.n_arms)]).reshape(self.n_arms, self.n_features, self.n_features)
        self._b = np.zeros((self.n_features, self.n_arms))

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
        if True in (self.counts < self.warmup):
            result = np.argmax(np.array(self.counts < self.warmup, dtype=int))
        else:
            x = _check_x_input(x)
            self.theta_hat = np.concatenate([self.A_inv[i] @ np.expand_dims(self.b[:, i], axis=1)
                                             for i in np.arange(self.n_arms)], axis=1)  # user_dim * n_arms
            sigma_hat = np.concatenate([np.sqrt(x.T @ self.A_inv[i] @ x) for i in np.arange(self.n_arms)], axis=1)  # 1 * n_arms
            result = np.argmax(x.T @ self.theta_hat + self.alpha * sigma_hat)
        return result

    def update(self, x: np.matrix, chosen_arm: int, reward: Union[int, float]) -> None:
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
        x = _check_x_input(x)
        self.data_size += 1
        self.counts[chosen_arm] += 1
        self.rewards += reward
        self._A_inv[chosen_arm] -= \
            self._A_inv[chosen_arm] @ x @ x.T @ self._A_inv[chosen_arm] / (1 + x.T @ self._A_inv[chosen_arm] @ x)  # d * d
        self._b[:, chosen_arm] += np.ravel(x) * reward  # d * 1
        if self.data_size % self.batch_size == 0:
            self.A_inv, self.b = np.copy(self._A_inv), np.copy(self._b)  # d * d,  d * 1


class HybridLinUCB(BaseContextualPolicy):
    """Hybrid Linear Upper Confidence Bound.

    Parameters
    ----------
    n_arms: int
        The number of given bandit arms.

    z_dim: int,
        The dimensions of context vectors which are common to all arms.

    x_dim:, int
        The dimentions of context vectors which are unique to earch arm.

    alpha: float, optional(default=1.0)
        The hyper-parameter which represents how often the algorithm explores.

    warmup: int, optional(default=1)
        The minimum number of pull of earch arm.

    batch_size: int, optional (default=1)
        The number of data given in each batch.

    References
    -------
    [1] L. Li, W. Chu, J. Langford, and E. Schapire.
        A contextual-bandit approach to personalized news article recommendation.
        In Proceedings of the 19th International Conference on World Wide Web, pp. 661–670. ACM, 2010.

    """

    def __init__(self, n_arms: int, z_dim: int, x_dim: int, alpha: float=1.0, warmup: int=1, batch_size: int=1) -> None:
        """Initialize class."""
        super().__init__(n_arms, z_dim + x_dim, warmup, batch_size)

        self.z_dim = z_dim  # k
        self.x_dim = x_dim  # d
        self.alpha = alpha
        self.name = f"HybridLinUCB(α={self.alpha})"

        self.beta = np.zeros(self.z_dim)
        self.theta_hat = np.zeros((self.x_dim, self.n_arms))  # d * k

        # matrices which are common to all context
        self.A_zero, self.b_zero = np.identity(self.z_dim), np.zeros((self.z_dim, 1))  # k * k, k * 1
        self.A_inv = np.concatenate([np.identity(self.x_dim)
                                     for i in np.arange(self.n_arms)]).reshape(self.n_arms, self.x_dim, self.x_dim)  # k * d * d
        self.B = np.concatenate([np.zeros((self.x_dim, self.z_dim))
                                 for i in np.arange(self.n_arms)]).reshape(self.n_arms, self.x_dim, self.z_dim)
        self.b = np.zeros((self.x_dim, self.n_arms))

        self._A_zero, self._b_zero = np.identity(self.z_dim), np.zeros((self.z_dim, 1))
        self._A_inv = np.concatenate([np.identity(self.x_dim)
                                      for i in np.arange(self.n_arms)]).reshape(self.n_arms, self.x_dim, self.x_dim)  # k * d * d
        self._B = np.concatenate([np.zeros((self.x_dim, self.z_dim))
                                  for i in np.arange(self.n_arms)]).reshape(self.n_arms, self.x_dim, self.z_dim)
        self._b = np.zeros((self.x_dim, self.n_arms))

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
        if True in (self.counts < self.warmup):
            result = np.argmax(np.array(self.counts < self.warmup, dtype=int))
        else:
            z, x = _check_x_input(x[:self.z_dim]), _check_x_input(x[self.z_dim:])
            self.beta = np.linalg.inv(self.A_zero) @ self.b_zero  # k * 1
            self.theta_hat = np.concatenate([(self.A_inv[i] @ (np.expand_dims(self.b[:, i], axis=1) - self.B[i] @ self.beta))
                                             for i in np.arange(self.n_arms)], axis=1)
            s1 = z.T @ np.linalg.inv(self.A_zero) @ z
            s2 = - 2 * np.concatenate([z.T @ np.linalg.inv(self.A_zero) @ self.B[i].T @ self.A_inv[i] @ x
                                       for i in np.arange(self.n_arms)], axis=1)
            s3 = np.concatenate([x.T @ self.A_inv[i] @ x for i in np.arange(self.n_arms)], axis=1)
            s4 = np.concatenate([x.T @ self.A_inv[i] @ self.B[i] @ np.linalg.inv(self.A_zero) @ self.B[i].T @ self.A_inv[i] @ x
                                 for i in np.arange(self.n_arms)], axis=1)
            sigma_hat = s1 + s2 + s3 + s4
            result = np.argmax(z.T @ self.beta + x.T @ self.theta_hat + self.alpha * sigma_hat)
        return result

    def update(self, x: np.ndarray, chosen_arm: int, reward: float) -> None:
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
        z, x = _check_x_input(x[:self.z_dim]), _check_x_input(x[self.z_dim:])

        self.data_size += 1
        self.counts[chosen_arm] += 1
        self.rewards += reward
        self._A_zero += self._B[chosen_arm].T @ self._A_inv[chosen_arm] @ self._B[chosen_arm]
        self._b_zero += self._B[chosen_arm].T @ self._A_inv[chosen_arm] @ self._b[chosen_arm]
        self._A_inv[chosen_arm] -= self._A_inv[chosen_arm] @ x @ x.T @ self._A_inv[chosen_arm] / (1 + x.T @ self._A_inv[chosen_arm] @ x)
        self._B[chosen_arm] += x @ z.T
        self._b[:, chosen_arm] += np.ravel(x) * reward
        self._A_zero += z @ z.T - self._B[chosen_arm].T @ self._A_inv[chosen_arm] @ self._B[chosen_arm]
        self._b_zero += z * reward - self._B[chosen_arm].T @ self._A_inv[chosen_arm] @ np.expand_dims(self._b[:, chosen_arm], axis=1)

        if self.data_size % self.batch_size == 0:
            self.A_zero, self.b_zero = np.copy(self._A_zero), np.copy(self._b_zero)
            self.A_inv, self.B, self.b = np.copy(self._A_inv), np.copy(self._B), np.copy(self._b)


class LinTS(BaseContextualPolicy):
    """Linear Thompson Sampling.

    Parameters
    ----------
    n_arms: int
        The number of given bandit arms.

    n_features: int
        The dimention of context vectors.

    sigma: float, optional(default=1.0)
        The variance of prior gaussian distribution.

    warmup: int, optional(default=1)
        The minimum number of pull of earch arm.

    sample_batch: int, optional (default=1)
        How often the policy sample new parameters.

    batch_size: int, optional (default=1)
        The number of data given in each batch.

    References
    -------
    [1] 本多淳也, 中村篤祥. バンディット問題の理論とアルゴリズム. 講談社 機械学習プロフェッショナルシリーズ. 2016.

    """

    def __init__(self, n_arms: int, n_features: int, sigma: float=1.0,
                 warmup: int=1, sample_batch: int=1, batch_size: int=1) -> None:
        """Initialize class."""
        super().__init__(n_arms, n_features, warmup, batch_size)

        self.sigma = sigma
        self.sample_batch = sample_batch
        self.name = f"LinTS(σ={self.sigma})"

        self.theta_hat, self.theta_tilde = np.zeros((self.n_features, self.n_arms)), np.zeros((self.n_features, self.n_arms))
        self.A_inv = np.concatenate([np.identity(self.n_features)
                                     for i in np.arange(self.n_arms)]).reshape(self.n_arms, self.n_features, self.n_features)  # k * d * d
        self.b = np.zeros((self.n_features, self.n_arms))  # d * k

        self._A_inv = np.concatenate([np.identity(self.n_features)
                                      for i in np.arange(self.n_arms)]).reshape(self.n_arms, self.n_features, self.n_features)
        self._b = np.zeros((self.n_features, self.n_arms))

    def select_arm(self, x: np.matrix) -> int:
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
        if True in (self.counts < self.warmup):
            result = np.argmax(np.array(self.counts < self.warmup, dtype=int))
        else:
            x = _check_x_input(x)
            if self.data_size % self.sample_batch == 0:
                self.theta_hat = np.concatenate([self.A_inv[i] @ np.expand_dims(self.b[:, i], axis=1)
                                                 for i in np.arange(self.n_arms)], axis=1)
                self.theta_tilde = np.concatenate([np.expand_dims(np.random.multivariate_normal(self.theta_hat[:, i], self.A_inv[i]), axis=1)
                                                   for i in np.arange(self.n_arms)], axis=1)
            result = np.argmax(x.T @ self.theta_tilde)

        return result

    def update(self, x: np.matrix, chosen_arm: int, reward: float) -> None:
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
        x = _check_x_input(x)
        self.data_size += 1
        self.counts[chosen_arm] += 1
        self.rewards += reward
        self._A_inv[chosen_arm] -= \
            self._A_inv[chosen_arm] @ x @ x.T @ self._A_inv[chosen_arm] / (1 + x.T @ self._A_inv[chosen_arm] @ x)  # d * d
        self._b[:, chosen_arm] += np.ravel(x) * reward  # d * 1
        if self.data_size % self.batch_size == 0:
            self.A_inv, self.b = np.copy(self._A_inv), np.copy(self._b)  # d * d,  d * 1


class LogisticTS(BaseContextualPolicy):
    """Logistic Thompson Sampling.

    Parameters
    ----------
    n_arms: int
        The number of given bandit arms.

    n_features: int
        The dimention of context vectors.

    sigma: float, optional(default=1.0)
        The variance of prior gaussian distribution.

    n_iter: int, optional(default=1)
        The num of iteration of newton method in each parameter update.

    sample_batch: int, optional (default=1)
        How often the policy sample new parameters.

    warmup: int, optional(default=1)
        The minimum number of pull of earch arm.

    batch_size: int, optional (default=1)
        The number of data given in each batch.

    References
    -------
    [1] 本多淳也, 中村篤祥. バンディット問題の理論とアルゴリズム. 講談社 機械学習プロフェッショナルシリーズ, 2016.

    [2] O. Chapelle, L. Li. An Empirical Evaluation of Thompson Sampling. In NIPS, pp. 2249–2257, 2011.

    """

    def __init__(self, n_arms: int, n_features: int, sigma: float=0.1,
                 n_iter: int=1, warmup: int=1, sample_batch: int=1,  batch_size: int=1) -> None:
        """Initialize Class."""
        super().__init__(n_arms, n_features, warmup, batch_size)

        self.sigma = sigma
        self.n_iter = n_iter
        self.sample_batch = sample_batch
        self.name = f"LogisticTS(σ={self.sigma})"

        self.data_stock: list = [[] for i in np.arange(self.n_arms)]
        self.reward_stock: list = [[] for i in np.arange(self.n_arms)]

        # array - (n_arms * user_dim),
        self.theta_hat, self.theta_tilde = np.zeros((self.n_features, self.n_arms)), np.zeros((self.n_features, self.n_arms))
        self.hessian_inv = np.concatenate([np.identity(self.n_features)
                                           for i in np.arange(self.n_arms)]).reshape(self.n_arms, self.n_features, self.n_features)

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
        if True in (self.counts < self.warmup):
            result = np.argmax(np.array(self.counts < self.warmup, dtype=int))
        else:
            x = _check_x_input(x)
            if self.data_size % self.sample_batch == 0:
                self.theta_tilde = np.concatenate([np.expand_dims(np.random.multivariate_normal(self.theta_hat[:, i], self.hessian_inv[i]), axis=1)
                                                   for i in np.arange(self.n_arms)], axis=1)
            result = np.argmax(x.T @ self.theta_tilde)
        return result

    def update(self, x: np.ndarray, chosen_arm: int, reward: float) -> None:
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
        x = _check_x_input(x)
        self.counts[chosen_arm] += 1
        self.rewards += reward
        self.data_stock[chosen_arm].append(x)  # (user_dim + arm_dim) * 1
        self.reward_stock[chosen_arm].append(reward)
        self.data_size += 1

        if self.data_size % self.batch_size == 0:
            for i in np.arange(self.n_iter):
                self.theta_hat[:, chosen_arm], self.hessian_inv[chosen_arm] = \
                    self._update_theta_hat(chosen_arm, self.theta_hat[:, chosen_arm])

    def _calc_gradient(self, chosen_arm: int, theta_hat: np.ndarray) -> np.ndarray:
        _hat = np.expand_dims(theta_hat, axis=1)
        _gradient = _hat / self.sigma
        _data = np.concatenate(self.data_stock[chosen_arm], axis=1)  # arm_dim * n_user
        _gradient += np.expand_dims(np.sum(_data * (np.exp(_hat.T @ _data) / (1 + np.exp(_hat.T @ _data))), axis=1), axis=1)
        _gradient -= np.expand_dims(np.sum(_data[:, np.array(self.reward_stock[chosen_arm]) == 1], axis=1), axis=1)
        return _gradient

    def _calc_hessian(self, chosen_arm: int, theta_hat: np.ndarray) -> np.ndarray:
        _hat = np.expand_dims(theta_hat, axis=1)
        _hessian = np.identity(self.n_features) / self.sigma
        _data = np.concatenate(self.data_stock[chosen_arm], axis=1)
        mat = [np.expand_dims(_data[:, i], axis=1) @ np.expand_dims(_data[:, i], axis=1).T
               for i in np.arange(self.counts[chosen_arm])]
        weight = np.ravel(np.exp(_hat.T @ _data) / (1 + np.exp(_hat.T @ _data)) ** 2)  # 1 * data_size
        _hessian += np.sum(
            np.concatenate([_mat * w for _mat, w in zip(mat, weight)], axis=0).reshape(self.counts[chosen_arm],
                                                                                       self.n_features,
                                                                                       self.n_features),
            axis=0)

        return _hessian

    def _update_theta_hat(self, chosen_arm: int, theta_hat: np.ndarray) -> np.ndarray:
        _theta_hat = np.expand_dims(theta_hat, axis=1)  # (user_dim * arm_dim) * 1
        _gradient = self._calc_gradient(chosen_arm, theta_hat)
        _hessian_inv = np.linalg.inv(self._calc_hessian(chosen_arm, theta_hat))
        _theta_hat -= _hessian_inv @ _gradient
        return np.ravel(_theta_hat), _hessian_inv


class ACTS(BaseContextualPolicy):
    """Action Centered Thompson Sampling Algorithm for Contextual Multi-Armed Bandit Problem.

    References
    -------
    [1] K. Greenewald, Ambuj Tewari, S. Murphy, and P. Klasnja. Action centered contextual bandits. In NIPS, 2017.

    """

    def __init__(self, n_arms: int, n_features: int, v: float = 1.0,
                 pi_min: float = 0.1, pi_max: float = 0.9, warmup: int = 10,
                 batch_size: int = 100, sample_batch_size: int = 20) -> None:
        """Initialize class."""
        self.n_arms = n_arms
        self.n_features = n_features  # n_arms * user_dim
        self.warmup = warmup
        self.sigma = v ** 2  # v ** 2 ?
        self.pi_min = pi_min
        self.pi_max = pi_max
        self.a_bar = 0
        self.pi_t = pi_max
        self.sample_batch_size = sample_batch_size

        self.B_inv = [np.copy(np.matrix(np.identity(self.n_features))) for i in np.arange(self.n_arms)]
        self.b = [np.copy(np.matrix(np.zeros(self.n_features)).T) for i in np.arange(self.n_arms)]
        self.theta = [np.copy(np.zeros(self.n_features)) for i in np.arange(self.n_arms)]
        self.theta_tilde = np.matrix(np.zeros(shape=(self.n_features, self.n_arms)))

        self.data_size = 0
        self.batch_size = batch_size
        self._B_inv = [np.copy(np.matrix(np.identity(self.n_features))) for i in np.arange(self.n_arms)] * 1
        self._b = [np.copy(np.matrix(np.zeros(self.n_features)).T) for i in np.arange(self.n_arms)]
        self._theta = [np.copy(np.zeros(self.n_features)) for i in np.arange(self.n_arms)]

        self.counts_warmup = np.zeros(n_arms, dtype=int)
        self.counts = np.zeros(n_arms + 1, dtype=int)
        self.rewards = 0.0

    def select_arm(self, x: np.matrix) -> int:
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
        if True in (self.counts_warmup < self.warmup):
            self.a_bar = np.where(self.counts_warmup < self.warmup)[0][0]
            self.counts_warmup[self.a_bar] += 1
            result = self.a_bar + 1
        else:
            values = np.zeros(self.n_arms)

            if self.data_size % self.sample_batch_size == 0:
                self.theta_tilde = np.concatenate([np.matrix(np.random.multivariate_normal(mean=self.theta[i], cov=self.sigma * self.B_inv[i])).T
                                                   for i in np.arange(self.n_arms)], axis=1)

            values = self.theta_tilde.T @ x
            self.a_bar = np.argmax(values)
            mu_bar = self.theta_tilde[:, self.a_bar].T @ x
            sigma_bar = self.sigma * (x.T @ self.B_inv[self.a_bar] @ x).A[0]
            self.pi_t = 1.0 - np.clip(a=norm.cdf(x=0, loc=mu_bar, scale=sigma_bar), a_min=self.pi_min, a_max=self.pi_max)[0][0]

            result = np.random.choice([0, self.a_bar + 1], p=[1 - self.pi_t, self.pi_t])
        return result

    def update(self, x: np.matrix, chosen_arm: int, reward: float) -> None:
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
        self.data_size += 1
        self.counts[chosen_arm] += 1
        self.rewards += reward
        _x = (1 - self.pi_t) * self.pi_t * x
        self._B_inv[self.a_bar] -= self._B_inv[self.a_bar] @ _x @ _x.T @ self._B_inv[self.a_bar] / (1 + _x.T @ self._B_inv[self.a_bar] @ _x)
        self._b[self.a_bar] += x * reward * (np.sign([chosen_arm]) - self.pi_t)
        self._theta[self.a_bar] = (self._B_inv[self.a_bar] @ self._b[self.a_bar]).A.reshape(self.n_features)

        if self.data_size % self.batch_size == 0:
            self.B_inv = np.copy(self._B_inv)  # d * d
            self.b = np.copy(self._b)  # d * 1
            self.theta = np.copy(self._theta)
