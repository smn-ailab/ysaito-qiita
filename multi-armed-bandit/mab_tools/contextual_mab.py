"""This Module contains basic Contextual Multi-Armed Bandit Algorithms."""
import copy
import math
import random
from abc import ABC, abstractmethod

import numpy as np
from pandas import DataFrame, Series
from scipy.stats import norm


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


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


class LinUCB(MABInterface):
    """Linear Upper Confidence Bound Algorithm for Contextual Multi-Armed Bandit Problem.

    References
    -------
    [1] Li, Lihong, Chu, Wei, Langford, John, and Schapire, Robert E.: A contextual-bandit approach to personalized news article recommendation.
    In Proceedings of the 19th International Conference on World Wide Web, pp. 661–670. ACM, 2010.
    """

    def __init__(self, n_arms: int, feature_dim: int, alpha: float =1.0, warmup: int =15, batch_size: int=0) -> None:
        """Initialize class.

        :param n_arms: the number of given arms.
        :param feature_dim: dimentions of context matrix.
        :param alpha:  the hyper-parameter which represents how often the algorithm explore.
        :param warmup: how many times the algorithms randomly explore arms at first.
        :param batch_size: the size of information about rewards given in a update.
        """

        self.n_arms = n_arms
        self.feature_dim = feature_dim
        self.warmup = warmup
        self.alpha = alpha

        self.theta = [copy.deepcopy(np.zeros(self.feature_dim)) for i in np.arange(n_arms)]  # d * 1
        self.A_inv = [copy.deepcopy(np.matrix(np.identity(self.feature_dim))) for i in np.arange(self.n_arms)]  # d * d
        self.b = [copy.deepcopy(np.matrix(np.zeros(self.feature_dim)).T) for i in np.arange(self.n_arms)]  # d * 1

        self.data_size = 0
        self.batch_size = batch_size
        self._A_inv = [copy.deepcopy(np.matrix(np.identity(self.feature_dim))) for i in np.arange(self.n_arms)]  # d * d
        self._b = [copy.deepcopy(np.matrix(np.zeros(self.feature_dim)).T) for i in np.arange(self.n_arms)]  # d * 1

        self.counts = np.zeros(self.n_arms, dtype=int)
        self.rewards = 0

    def select_arm(self, x: np.matrix) -> int:
        """Decide which arm should be selected.

        :param x: observed context matrix.

        :return: index of the selected arm.
        """
        if True in (self.counts < self.warmup):
            result = np.where(self.counts < self.warmup)[0][0]
        else:
            ucb_values = np.zeros(self.n_arms)
            self.theta = np.concatenate([self.A_inv[i].dot(self.b[i]) for i in np.arange(self.n_arms)], axis=1)  # user_dim * n_arms
            mu_hat = self.theta.T.dot(x)  # n_arms * 1
            sigma_hat = self.alpha * np.concatenate([np.sqrt(x.T.dot(self.A_inv[i].dot(x))) for i in np.arange(self.n_arms)], axis=0)  # n_arms * 1
            result = np.argmax(mu_hat + sigma_hat)
        return result

    def update(self, x: np.matrix, chosen_arm: int, reward: float) -> None:
        """Update the information about the arms.

        :param x: observed context matrix.
        :param chosen_arm: index of the chosen arm.
        :param reward: reward from the chosen arm.
        """
        self.counts[chosen_arm] += 1
        self.rewards += reward
        self.A_inv[chosen_arm] -= self.A_inv[chosen_arm].dot(x.dot(x.T.dot(self.A_inv[chosen_arm]))) / (1 + x.T.dot(self.A_inv[chosen_arm].dot(x)))
        self.b[chosen_arm] += x * reward  # d * 1

    def batch_update(self, x: np.matrix, chosen_arm: int, reward: float) -> None:
        """Update the information about the arms with a new batch of data.

        :param x: observed context matrix.
        :param chosen_arm: index of the chosen arm.
        :param reward: reward from the chosen arm.
        """
        self.data_size += 1
        self.counts[chosen_arm] += 1
        self.rewards += reward
        self._A_inv[chosen_arm] -= self._A_inv[chosen_arm].dot(x.dot(x.T.dot(self._A_inv[chosen_arm]))) / (1 + x.T.dot(self._A_inv[chosen_arm].dot(x)))  # d * d
        self._b[chosen_arm] += x * reward  # d * 1
        if self.data_size % self.batch_size == 0:
            self.A_inv = copy.deepcopy(self._A_inv)  # d * d
            self.b = copy.deepcopy(self._b)  # d * 1
        else:
            pass


class HybridLinUCB(MABInterface):
    """Hybrid Linear Upper Confidence Bound Algorithm for Contextual Multi-Armed Bandit Problem.

    References
    -------
    [1] Li, Lihong, Chu, Wei, Langford, John, and Schapire, Robert E.: A contextual-bandit approach to personalized news article recommendation.
    In Proceedings of the 19th International Conference on World Wide Web, pp. 661–670. ACM, 2010.
    """

    def __init__(self, n_arms: int, z_dim: int, x_dim: int, alpha: float =1.0, warmup: int =15, batch_size: int=0) -> None:
        """Initialize class.

        :param n_arms: the number of given arms.
        :param z_dim: dimensions of context matrix which is common to all arms.
        :param x_dim: dimentions of context matrix which is unique to earch arm.
        :param alpha:  the hyper-parameter which represents how often the algorithm explore.
        :param warmup: how many times the algorithms randomly explore arms at first.
        :param batch_size: the size of information about rewards given in a update.
        """

        self.n_arms = n_arms
        self.z_dim = z_dim  # k
        self.x_dim = x_dim  # d
        self.warmup = warmup
        self.alpha = alpha

        self.beta = np.zeros(self.z_dim)
        self.theta = None  # d * 1

        # matrices which are common to all context
        self.A_zero = np.matrix(np.identity(self.z_dim))  # k * k
        self.b_zero = np.matrix(np.zeros(self.z_dim)).T  # k * 1
        # matrices which are different for each context
        self.A_inv = [copy.deepcopy(np.matrix(np.identity(self.x_dim))) for i in np.arange(self.n_arms)]
        self.B = [copy.deepcopy(np.matrix(np.zeros((self.x_dim, self.z_dim)))) for i in range(self.n_arms)]  # d * k
        self.b = [copy.deepcopy(np.matrix(np.zeros(self.x_dim)).T) for i in range(self.n_arms)]  # d * 1

        self.data_size = 0
        self.batch_size = batch_size
        self._A_zero = np.matrix(np.identity(self.z_dim))  # k * k
        self._b_zero = np.matrix(np.zeros(self.z_dim)).T  # k * 1
        self._A_inv = [copy.deepcopy(np.matrix(np.identity(self.x_dim))) for i in range(self.n_arms)]  # d * d
        self._B = [copy.deepcopy(np.matrix(np.zeros((self.x_dim, self.z_dim)))) for i in range(self.n_arms)]  # d * k
        self._b = [copy.deepcopy(np.matrix(np.zeros(self.x_dim)).T) for i in range(self.n_arms)]  # d * 1

        self.counts = np.zeros(self.n_arms, dtype=int)
        self.rewards = 0

    def select_arm(self, x: np.matrix) -> int:
        """Decide which arm should be selected.

        :param x: observed context matrix.

        :return: index of the selected arm.
        """

        z = x[:][:self.z_dim]
        x = x[:][self.z_dim:]

        if True in (self.counts < self.warmup):
            result = np.where(self.counts < self.warmup)[0][0]
        else:
            ucb_values = np.zeros(self.n_arms)
            self.beta = np.linalg.inv(self.A_zero).dot(self.b_zero)  # k * 1
            self.theta = [self.A_inv[i].dot(self.b[i] - self.B[i].dot(self.beta)).A.reshape(self.x_dim) for i in np.arange(self.n_arms)]  # d * 1
            mu_hat = [z.T.dot(self.beta) + x.T.dot(self.theta[i]) for i in np.arange(self.n_arms)]
            s1 = z.T.dot(np.linalg.inv(self.A_zero)).dot(z).A[0]
            s2 = - 2 * np.array([z.T.dot(np.linalg.inv(self.A_zero)).dot(self.B[i].T).dot(self.A_inv[i]).dot(x) for i in np.arange(self.n_arms)])
            s3 = np.array([x.T.dot(self.A_inv[i]).dot(x) for i in np.arange(self.n_arms)])
            s4 = np.array([x.T.dot(self.A_inv[i]).dot(self.B[i]).dot(np.linalg.inv(self.A_zero)).dot(
                self.B[i].T).dot(self.A_inv[i]).dot(x) for i in np.arange(self.n_arms)])
            sigma_hat = s1 + s2 + s3 + s4
            ucb_values = mu_hat + self.alpha * sigma_hat
            result = np.argmax(ucb_values)
        return result

    def update(self, x: np.matrix, chosen_arm: int, reward: float) -> None:
        """Update the information about the arms.

        :param x: observed context matrix.
        :param chosen_arm: index of the chosen arm.
        :param reward: reward from the chosen arm.
        """
        z = x[:][:self.z_dim]
        x = x[:][self.z_dim:]

        self.counts[chosen_arm] += 1
        self.rewards += reward
        self.A_zero += self.B[chosen_arm].T.dot(self.A_inv[chosen_arm]).dot(self.B[chosen_arm])
        self.b_zero += self.B[chosen_arm].T.dot(self.A_inv[chosen_arm]).dot(self.b[chosen_arm])
        self.A_inv[chosen_arm] -= self.A_inv[chosen_arm].dot(x.dot(x.T.dot(self.A_inv[chosen_arm]))) / (1 + x.T.dot(self.A_inv[chosen_arm].dot(x)))
        self.B[chosen_arm] += x.dot(z.T)
        self.b[chosen_arm] += x * reward
        self.A_zero += z.dot(z.T) - self.B[chosen_arm].T.dot(self.A_inv[chosen_arm]).dot(self.B[chosen_arm])
        self.b_zero += z * reward - self.B[chosen_arm].T.dot(self.A_inv[chosen_arm]).dot(self.b[chosen_arm])

    def batch_update(self, x: np.matrix, chosen_arm: int, reward: float) -> None:
        """Update the information about the arms with a new batch of data.

        :param x: observed context matrix.
        :param chosen_arm: index of the chosen arm.
        :param reward: reward from the chosen arm.
        """
        self.data_size += 1
        z = x[:][:self.z_dim]
        x = x[:][self.z_dim:]

        self.counts[chosen_arm] += 1
        self.rewards += reward
        self._A_zero += self._B[chosen_arm].T.dot(self._A_inv[chosen_arm]).dot(self._B[chosen_arm])
        self._b_zero += self._B[chosen_arm].T.dot(self._A_inv[chosen_arm]).dot(self._b[chosen_arm])
        self._A_inv[chosen_arm] -= self._A_inv[chosen_arm].dot(x.dot(x.T.dot(self._A_inv[chosen_arm]))) / (1 + x.T.dot(self._A_inv[chosen_arm].dot(x)))
        self._B[chosen_arm] += x.dot(z.T)
        self._b[chosen_arm] += x * reward
        self._A_zero += z.dot(z.T) - self._B[chosen_arm].T.dot(self._A_inv[chosen_arm]).dot(self._B[chosen_arm])
        self._b_zero += z * reward - self._B[chosen_arm].T.dot(self._A_inv[chosen_arm]).dot(self._b[chosen_arm])

        if self.data_size % self.batch_size == 0:
            self.A_zero = self._A_zero[:]
            self.b_zero = self._b_zero[:]
            self.A_inv = copy.deepcopy(self._A_inv)
            self.B = copy.deepcopy(self._B)
            self.b = copy.deepcopy(self._b)
        else:
            pass


class LinTS(MABInterface):
    """Linear Thompson Sampling Algorithm for Contextual Multi - Armed Bandit Problem

    References
    -------
    [1] 本多淳也, 中村篤祥. バンディット問題の理論とアルゴリズム. 講談社 機械学習プロフェッショナルシリーズ.
    """

    def __init__(self, n_arms: int, feature_dim: int, sigma: float=1.0, warmup: int=15,
                 sample_batch_size: int=20, batch_size: int=100) -> None:
        """Initialize class.

        :param n_arms: the number of given arms.
        :param feature_dim: dimensions of context matrix.
        :param sigma:  the hyper-parameter which adjust the variance of posterior gaussian distribution.
        :param warmup: how many times the algorithm randomly explore arms at first.
        :param sample_batch_size: how often the algorithm sample theta_tilde from posterior multivariate gaussian distribution.
        :param batch_size: the size of information about rewards given in a update.
        """
        self.n_arms = n_arms
        self.feature_dim = feature_dim
        self.warmup = warmup
        self.sample_batch_size = sample_batch_size
        self.sigma = sigma

        self.theta = None
        self.theta_tilde = np.matrix(np.zeros(shape=(self.feature_dim, self.n_arms)))

        self.A_inv = [copy.deepcopy(np.matrix(np.identity(self.feature_dim))) for i in np.arange(self.n_arms)]  # d * d
        self.b = [copy.deepcopy(np.matrix(np.zeros(self.feature_dim)).T) for i in np.arange(self.n_arms)]  # d * 1

        self.data_size = 0
        self.batch_size = batch_size
        self._A_inv = [copy.deepcopy(np.matrix(np.identity(self.feature_dim))) for i in np.arange(self.n_arms)]  # d * d
        self._b = [copy.deepcopy(np.matrix(np.zeros(self.feature_dim)).T) for i in np.arange(self.n_arms)]  # d * 1

        self.counts = np.zeros(self.n_arms, dtype=int)
        self.rewards = 0

    def select_arm(self, x: np.matrix) -> int:
        """Decide which arm should be selected.

        :param x: observed context matrix.

        :return: index of the selected arm.
        """
        if True in (self.counts < self.warmup):
            result = np.where(self.counts < self.warmup)[0][0]
        else:
            if self.data_size % self.sample_batch_size == 0:
                self.theta = [self.A_inv[i].dot(self.b[i]).A.reshape(self.feature_dim) for i in np.arange(self.n_arms)]
                sigma_hat = [self.sigma * self.A_inv[i] for i in np.arange(self.n_arms)]
                self.theta_tilde = np.concatenate([np.matrix(np.random.multivariate_normal(
                    self.theta[i], sigma_hat[i])).T for i in np.arange(self.n_arms)], axis=1)

            mu_hat = self.theta_tilde.T.dot(x)
            result = np.argmax(mu_hat)

        return result

    def update(self, x: np.matrix, chosen_arm: int, reward: float) -> None:
        """Update the information about the arms.

        :param x: observed context matrix.
        :param chosen_arm: index of the chosen arm.
        :param reward: reward from the chosen arm.
        """
        self.data_size += 1
        self.counts[chosen_arm] += 1
        self.rewards += reward
        self.A_inv[chosen_arm] -= self.A_inv[chosen_arm].dot(x.dot(x.T.dot(self.A_inv[chosen_arm]))) / (1 + x.T.dot(self.A_inv[chosen_arm].dot(x)))  # d * d
        self.b[chosen_arm] += x * reward  # d * 1

    def batch_update(self, x: np.matrix, chosen_arm: int, reward: float) -> None:
        """Update the information about the arms with a new batch of data.

        :param x: observed context matrix.
        :param chosen_arm: index of the chosen arm.
        :param reward: reward from the chosen arm.
        """
        self.data_size += 1
        self.counts[chosen_arm] += 1
        self.rewards += reward
        self._A_inv[chosen_arm] -= self._A_inv[chosen_arm].dot(x.dot(x.T.dot(self._A_inv[chosen_arm]))) / (1 + x.T.dot(self._A_inv[chosen_arm].dot(x)))  # d * d
        self._b[chosen_arm] += x * reward

        if self.data_size % self.batch_size == 0:
            self.A_inv = copy.deepcopy(self._A_inv)  # d * d
            self.b = copy.deepcopy(self._b)  # d * 1
        else:
            pass


class LogisticTS():
    """Logistic Thompson Sampling Algorithm for Contextual Multi-Armed Bandit Problem

    References
    -------
    [1] 本多淳也, 中村篤祥. バンディット問題の理論とアルゴリズム. 講談社 機械学習プロフェッショナルシリーズ.
    [2] Chapelle, Olivier and Li, Lihong.:
     An Empirical Evaluation of Thompson Sam- pling. In NIPS, pp. 2249–2257, 2011.
    """

    def __init__(self, n_arms: int, feature_dim: int, num_trial: int,
                 repeat: int= 1, warmup: int = 5, lam: float =0.1,
                 sample_batch_size: int= 5, batch_size: int =20) -> int:

        self.n_arms = n_arms
        self.feature_dim = feature_dim  # user_dim
        self.warmup = warmup
        self.lam = lam
        self.sample_batch_size = sample_batch_size

        self.batch_size = batch_size
        self.counts = np.zeros(n_arms, dtype="int")

        self.rewards = 0
        self.data_size = 0
        self.num_trial = num_trial
        self.repeat = repeat

        # matrix - (n_arms * user_dim) * batch_size
        self.data_stock = np.matrix(np.zeros((self.feature_dim * self.n_arms, self.num_trial)))  # (user_dim * arm_dim) * horizon
        self.reward_stock = np.zeros(self.num_trial)

        # array - (n_arms * user_dim),
        self.theta_tilde = np.zeros(self.feature_dim * self.n_arms)
        self.theta_hat = np.zeros(self.feature_dim * self.n_arms)
        self.hessian_inv = np.matrix(np.identity(self.feature_dim * self.n_arms))  # (user_dim * arm_dim) * (user_dim * arm_dim)

    def select_arm(self, x: np.array) -> int:
        if True in (self.counts < self.warmup):
            result = np.where(self.counts < self.warmup)[0][0]
        else:
            if self.data_size % self.sample_batch_size == 0:
                self.theta_tilde = np.matrix(np.random.multivariate_normal(mean=self.theta_hat, cov=self.hessian_inv)).T
            mu_hat = np.array([self.theta_tilde[i * self.feature_dim: (i + 1) * self.feature_dim].T.dot(x).A[0][0] for i in np.arange(self.n_arms)])
            result = np.argmax(mu_hat)
        return result

    def update(self, user_x: np.matrix, chosen_arm: int, reward: float) -> None:
        self.counts[chosen_arm] += 1
        self.rewards += reward
        x = np.matrix(np.zeros(self.feature_dim * self.n_arms)).T
        x[chosen_arm * self.feature_dim: (chosen_arm + 1) * self.feature_dim] = user_x  # (user_dim + arm_dim) * 1
        self.data_stock[:, self.data_size] = x  # (user_dim + arm_dim) * 1
        self.reward_stock[self.data_size] = reward
        self.data_size += 1

        if self.data_size % self.batch_size == 0:
            for i in np.arange(self.repeat):
                self.theta_hat, self.hessian_inv = self._update_theta_hat(self.theta_hat)

    def _calc_gradient(self, theta_hat) -> np.matrix:
        theta_hat = np.matrix(theta_hat).T  # (user_dim * arm_dim) * 1
        _gradient = theta_hat / self.lam  # (user_dim * arm_dim) * 1
        _data_stock = self.data_stock[:, :self.data_size]  # (user_dim * arm_dim) * data_size
        _gradient += _data_stock.dot((np.exp(theta_hat.T.dot(_data_stock)) / (1 + np.exp(theta_hat.T.dot(_data_stock)))).T)  # (user_dim * arm_dim) * 1
        _gradient -= np.sum(_data_stock[:, np.where(self.reward_stock == 1)[0]], axis=1)  # (user_dim * arm_dim) * 1
        return _gradient

    def _calc_hessian(self, theta_hat) -> np.matrix:
        theta_hat = np.matrix(theta_hat).T  # (user_dim * arm_dim) * 1
        _hessian = np.matrix(np.identity(self.feature_dim * self.n_arms)) / self.lam
        _data_stock = self.data_stock[:, :self.data_size]  # (user_dim * arm_dim) * data_size
        _exp_matrix = np.sqrt(np.exp(theta_hat.T.dot(_data_stock))) / (1 + np.exp(theta_hat.T.dot(_data_stock)))  # 1 * data_size
        _data_matrix = np.matrix(_data_stock.A * _exp_matrix.A)  # (user_dim * arm_dim) * data_size
        _hessian += _data_matrix.dot(_data_matrix.T)  # (user_dim * arm_dim) * (user_dim * arm_dim)
        return _hessian

    def _update_theta_hat(self, theta_hat) -> np.matrix:
        _theta_hat = np.matrix(theta_hat).T  # (user_dim * arm_dim) * 1
        _gradient = self._calc_gradient(theta_hat)
        _hessian_inv = np.linalg.inv(self._calc_hessian(theta_hat))
        _theta_hat -= _hessian_inv.dot(_gradient)
        return _theta_hat.A.reshape(self.feature_dim * self.n_arms), _hessian_inv

    def batch_update(self, user_x: np.matrix, chosen_arm: int, reward: float) -> None:
        self.update(user_x, chosen_arm, reward)


class ACTS(MABInterface):
    """Action Centered Thompson Sampling Algorithm for Contextual Multi-Armed Bandit Problem

    References
    -------
    [1] Kristjan Greenewald, Ambuj Tewari, Susan Murphy, and Predag Klasnja.:
     Action centered contextual bandits. In NIPS, 2017.
    """

    def __init__(self, n_arms: int, feature_dim: int, v: float=1.0,
                 pi_min: float=0.1, pi_max: float=0.9, warmup: int=10,
                 batch_size: int=100, sample_batch_size: int=20) -> None:
        """Initialize class.

        :param n_arms: the number of given arms.
        :param feature_dim: dimensions of context matrix.
        :param v:  the hyper-parameter which adjust the variance of posterior gaussian distribution.
        :param pi_min: the minimum probability of selecting a non-zero action.
        ;param pi_max: the maximum probability of selecting a non-zero action.
        :param warmup: how many times the algorithms randomly explore arms at first.
        :param batch_size: the size of information about rewards given in a update.
        """
        self.n_arms = n_arms
        self.feature_dim = feature_dim  # n_arms * user_dim
        self.warmup = warmup
        self.sigma = v ** 2  # v ** 2 ?
        self.pi_min = pi_min
        self.pi_max = pi_max
        self.a_bar = 0
        self.pi_t = pi_max
        self.sample_batch_size = sample_batch_size

        self.B_inv = [copy.deepcopy(np.matrix(np.identity(self.feature_dim))) for i in np.arange(self.n_arms)]
        self.b = [copy.deepcopy(np.matrix(np.zeros(self.feature_dim)).T) for i in np.arange(self.n_arms)]
        self.theta = [copy.deepcopy(np.zeros(self.feature_dim)) for i in np.arange(self.n_arms)]
        self.theta_tilde = np.matrix(np.zeros(shape=(self.feature_dim, self.n_arms)))

        self.data_size = 0
        self.batch_size = batch_size
        self._B_inv = [copy.deepcopy(np.matrix(np.identity(self.feature_dim))) for i in np.arange(self.n_arms)] * 1
        self._b = [copy.deepcopy(np.matrix(np.zeros(self.feature_dim)).T) for i in np.arange(self.n_arms)]
        self._theta = [copy.deepcopy(np.zeros(self.feature_dim)) for i in np.arange(self.n_arms)]

        self.counts_warmup = np.zeros(n_arms, dtype=int)
        self.counts = np.zeros(n_arms + 1, dtype=int)
        self.rewards = 0

    def select_arm(self, x: np.matrix) -> int:
        """Decide which arm should be selected.

        :param user_x: observed context matrix.

        :return: index of the selected arm.
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

            values = self.theta_tilde.T.dot(x)
            self.a_bar = np.argmax(values)
            mu_bar = self.theta_tilde[:, self.a_bar].T.dot(x)
            sigma_bar = self.sigma * (x.T.dot(self.B_inv[self.a_bar]).dot(x)).A[0]
            self.pi_t = 1.0 - np.clip(a=norm.cdf(x=0, loc=mu_bar, scale=sigma_bar), a_min=self.pi_min, a_max=self.pi_max)[0][0]

            result = np.random.choice([0, self.a_bar + 1], p=[1 - self.pi_t, self.pi_t])
        return result

    def update(self, x: np.matrix, chosen_arm: int, reward: float) -> None:
        """Update the information about the arms.

        :param user_x: observed context matrix.
        :param chosen_arm: index of the chosen arm.
        :param reward: reward from the chosen arm.
        """
        self.data_size += 1
        self.counts[chosen_arm] += 1
        self.rewards += reward
        _x = np.sqrt((1 - self.pi_t)) * self.pi_t * x
        self.B_inv[self.a_bar] -= self.B_inv[self.a_bar].dot(_x.dot(_x.T.dot(self.B_inv[self.a_bar]))) / (1 + _x.T.dot(self.B_inv[self.a_bar].dot(_x)))
        self.b[self.a_bar] += x * reward * (np.sign([chosen_arm]) - self.pi_t)
        self.theta[self.a_bar] = self.B_inv[self.a_bar].dot(self.b[self.a_bar]).A.reshape(self.feature_dim)

    def batch_update(self, x: np.matrix, chosen_arm: int, reward: float) -> None:
        """Update the information about the arms with a new batch of data.

        :param x: observed context matrix.
        :param chosen_arm: index of the chosen arm.
        :param reward: reward from the chosen arm.
        """
        self.data_size += 1
        self.counts[chosen_arm] += 1
        self.rewards += reward
        _x = (1 - self.pi_t) * self.pi_t * x
        self._B_inv[self.a_bar] -= self._B_inv[self.a_bar].dot(_x.dot(_x.T.dot(self._B_inv[self.a_bar]))) / (1 + _x.T.dot(self._B_inv[self.a_bar].dot(_x)))
        self._b[self.a_bar] += x * reward * (np.sign([chosen_arm]) - self.pi_t)
        self._theta[self.a_bar] = self._B_inv[self.a_bar].dot(self._b[self.a_bar]).A.reshape(self.feature_dim)

        if self.data_size % self.batch_size == 0:
            self.B_inv = copy.deepcopy(self._B_inv)  # d * d
            self.b = copy.deepcopy(self._b)  # d * 1
            self.theta = copy.deepcopy(self._theta)
        else:
            pass
