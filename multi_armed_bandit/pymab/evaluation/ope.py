"""Classes for Off-Policy Policy Evaluation."""
import copy
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator

from pymab.policy import BaseContextualPolicy, BasePolicy, BaseThompsonSampling


class OPEInterface(ABC):
    """Abstract base class for Off-Policy Evaluators."""

    @abstractmethod
    def estimate(self, X: np.ndarray, a: np.ndarray, r: np.ndarray) -> DataFrame:
        """Estimate the value of the new policy.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            A test sample collected by some past policies.

        a: array-like, shape = (n_samples, )
            Action log of the past policies.

        r: array-like, shape = (n_samples, )
            Reward log of the past policies.

        """
        pass


class ReplayMethod(OPEInterface):
    """Replay Method.

    policy_list: List[Union[BasePolicy, BaseThompsonSampling, BaseContextualPolicy]]
        A candidates of the new policy.

    n_iter: int
        The number of bootstrap iteration.

    References
    -------
    [1] L. Li, W. Chu, J. Langford, and X. Wang. Unbiased offline evaluation of contextual-bandit-based news article recommendation algorithms.
        In Proceedings of the Web Search and Data Mining (WSDM), pp. 297–306. ACM, 2011.

    """

    def __init__(self,
                 policy_list: List[Union[BasePolicy, BaseThompsonSampling, BaseContextualPolicy]],
                 n_iter: int) -> None:
        """Initialize Class."""
        self.n_policies = len(policy_list)
        self.policy_list = policy_list
        self.n_iter = n_iter

    def estimate(self, X: np.ndarray, a: np.ndarray, r: np.ndarray) -> DataFrame:
        """Estimate the value of the candidate policies.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            A test sample collected by some past policies.

        a: array-like, shape = (n_samples, )
            Action log of the past policies.

        r: array-like, shape = (n_samples, )
            Reward log of the past policies.

        """
        n_data = X.shape[0]
        values = np.zeros(self.n_policies * self.n_iter).reshape(self.n_iter, self.n_policies)

        for j, policy in enumerate(self.policy_list):
            for i in np.arange(self.n_iter):
                p = copy.deepcopy(policy)
                rewards: list = []
                idx = np.random.choice(n_data, size=n_data, replace=True)
                _X_boot, _a_boot, _r_boot = X[idx], a[idx], r[idx]

                for _x, _a, _r in zip(_X_boot, _a_boot, _r_boot):
                    x = np.array(_x)
                    chosen_arm = p.select_arm(x) if p._policy_type == "contextual" else p.select_arm()
                    if chosen_arm == _a:
                        rewards.append(_r)
                        p.update(x, chosen_arm, _r) if p._policy_type == "contextual" else p.update(chosen_arm, _r)

                values[i, j] = np.mean(rewards)

        return DataFrame(values, columns=[p.name for p in self.policy_list])


class DirectMethod(OPEInterface):
    """Direct Method.

    policy_list: List[Union[BasePolicy, BaseThompsonSampling, BaseContextualPolicy]]
        A candidates of the new policy.

    n_iter: int
        The number of bootstrap iteration.

    regression: int
        Whether the reward is continuous or not.

    References
    -------
    [1] M. Dudík, J. Langford, and L. Li. Doubly robust policy evaluation and learning. arXiv preprint arXiv:1103.4601, 2011.

    """

    def __init__(self,
                 policy_list: List[Union[BasePolicy, BaseThompsonSampling, BaseContextualPolicy]],
                 n_iter: int,
                 regression: bool=False) -> None:
        """Initialize Class."""
        self.n_policies = len(policy_list)
        self.policy_list = policy_list
        self.n_iter = n_iter
        self.regression = regression

    def fit(self, pom: BaseEstimator,
            X: np.ndarray, a: np.ndarray, r: np.ndarray) -> None:
        """Fit potential outcome models.

        Parameters
        ----------
        pom: BaseEstimator
            Potential Outcome Model

        X : array-like, shape = (n_samples, n_features)
            A test sample collected by some past policies.

        a: array-like, shape = (n_samples, )
            Action log of the past policies.

        r: array-like, shape = (n_samples, )
            Reward log of the past policies.

        """
        pred_list: list = []
        for w in np.unique(a):
            pom_ = pom
            pom_.fit(X[a == w], r[a == w])
            if self.regression:
                pred_list.append(np.expand_dims(pom_.predict(X), axis=1))
            else:
                pred_list.append(np.expand_dims(pom_.predict_proba(X)[:, 1], axis=1))

        self.r_pred = np.concatenate(pred_list, axis=1)

    def estimate(self, X: np.ndarray, a: np.ndarray, r: np.ndarray) -> DataFrame:
        """Estimate the value of the candidate policies.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            A test sample collected by some past policies.

        a: array-like, shape = (n_samples, )
            Action log of the past policies.

        r: array-like, shape = (n_samples, )
            Reward log of the past policies.

        """
        n_data = X.shape[0]
        values = np.zeros(self.n_policies * self.n_iter).reshape(self.n_iter, self.n_policies)

        for j, policy in enumerate(self.policy_list):
            for i in np.arange(self.n_iter):
                p = copy.deepcopy(policy)
                rewards: list = []
                idx = np.random.choice(n_data, size=n_data, replace=True)
                _X_boot, _a_boot, _r_boot, _r_pred_boot = X[idx], a[idx], r[idx], self.r_pred[idx]

                for _x, _a, _r, _r_pred in zip(_X_boot, _a_boot, _r_boot, _r_pred_boot):
                    x = np.array(_x)
                    chosen_arm = p.select_arm(x) if p._policy_type == "contextual" else p.select_arm()

                    reward = _r_pred[chosen_arm]
                    rewards.append(reward)
                    p.update(x, chosen_arm, reward) if p._policy_type == "contextual" else p.update(chosen_arm, reward)

                values[i, j] = np.mean(rewards)

        return DataFrame(values, columns=[p.name for p in self.policy_list])


class IPSEstimator(OPEInterface):
    """Inverse Propensity Score Estimator.

    policy_list: List[Union[BasePolicy, BaseThompsonSampling, BaseContextualPolicy]]
        A candidates of the new policy.

    n_iter: int
        The number of bootstrap iteration.

    References
    -------
    [1] M. Dudík, J. Langford, and L. Li. Doubly robust policy evaluation and learning. arXiv preprint arXiv:1103.4601, 2011.

    """

    def __init__(self,
                 policy_list: List[Union[BasePolicy, BaseThompsonSampling, BaseContextualPolicy]],
                 n_iter: int) -> None:
        """Initialize Class."""
        self.n_policies = len(policy_list)
        self.policy_list = policy_list
        self.n_iter = n_iter

    def fit(self, pse: BaseEstimator, X: np.ndarray, a: np.ndarray) -> None:
        """Fit propensity score estimator.

        Parameters
        ----------
        pse: BaseEstimator
            Propensity Score Estimator

        X : array-like, shape = (n_samples, n_features)
            A test sample collected by some past policies.

        a: array-like, shape = (n_samples, )
            Action log of the past policies.

        """
        pse_ = pse
        pse_.fit(X, a)
        self.ps = pse_.predict_proba(X)

    def estimate(self, X: np.ndarray, a: np.ndarray, r: np.ndarray) -> DataFrame:
        """Estimate the value of the candidate policies.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            A test sample collected by some past policies.

        a: array-like, shape = (n_samples, )
            Action log of the past policies.

        r: array-like, shape = (n_samples, )
            Reward log of the past policies.

        """
        n_data = X.shape[0]
        values = np.zeros(self.n_policies * self.n_iter).reshape(self.n_iter, self.n_policies)

        for j, policy in enumerate(self.policy_list):
            for i in np.arange(self.n_iter):
                p = copy.deepcopy(policy)
                rewards: list = []
                idx = np.random.choice(n_data, size=n_data, replace=True)
                _X_boot, _a_boot, _r_boot, _ps_boot = X[idx], a[idx], r[idx], self.ps[idx]

                for _x, _a, _r, _ps in zip(_X_boot, _a_boot, _r_boot, _ps_boot):
                    x = np.array(_x)
                    chosen_arm = p.select_arm(x) if p._policy_type == "contextual" else p.select_arm()

                    if chosen_arm == _a:
                        reward = _r / _ps[chosen_arm]
                        rewards.append(reward)
                        p.update(x, chosen_arm, reward) if p._policy_type == "contextual" else p.update(chosen_arm, reward)

                values[i, j] = np.mean(rewards)

        return DataFrame(values, columns=[p.name for p in self.policy_list])


class DREstimator(OPEInterface):
    """Doubly Robust Estimator.

    policy_list: List[Union[BasePolicy, BaseThompsonSampling, BaseContextualPolicy]]
        A candidates of the new policy.

    n_iter: int
        The number of bootstrap iteration.

    regression: int
        Whether the reward is continuous or not.

    References
    -------
    [1] M. Dudík, J. Langford, and L. Li. Doubly robust policy evaluation and learning. arXiv preprint arXiv:1103.4601, 2011.

    """

    def __init__(self,
                 policy_list: List[Union[BasePolicy, BaseThompsonSampling, BaseContextualPolicy]],
                 n_iter: int,
                 regression: bool=False) -> None:
        """Initialize Class."""
        self.n_policies = len(policy_list)
        self.policy_list = policy_list
        self.n_iter = n_iter
        self.regression = regression

    def fit(self, pom: BaseEstimator, pse: BaseEstimator,
            X: np.ndarray, a: np.ndarray, r: np.ndarray) -> None:
        """Fit potential outcome models.

        Parameters
        ----------
        pom: BaseEstimator
            Potential Outcome Model

        pse: BaseEstimator
            Propensity Score Estimator

        X : array-like, shape = (n_samples, n_features)
            A test sample collected by some past policies.

        a: array-like, shape = (n_samples, )
            Action log of the past policies.

        r: array-like, shape = (n_samples, )
            Reward log of the past policies.

        """
        pse_ = pse
        pse_.fit(X, a)
        self.ps = pse_.predict_proba(X)

        pred_list: list = []
        for w in np.unique(a):
            pom_ = pom
            pom_.fit(X[a == w], r[a == w])
            if self.regression:
                pred_list.append(np.expand_dims(pom_.predict(X), axis=1))
            else:
                pred_list.append(np.expand_dims(pom_.predict_proba(X)[:, 1], axis=1))

        self.r_pred = np.concatenate(pred_list, axis=1)

    def estimate(self, X: np.ndarray, a: np.ndarray, r: np.ndarray) -> DataFrame:
        """Estimate the value of the candidate policies.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            A test sample collected by some past policies.

        a: array-like, shape = (n_samples, )
            Action log of the past policies.

        r: array-like, shape = (n_samples, )
            Reward log of the past policies.

        """
        n_data = X.shape[0]
        values = np.zeros(self.n_policies * self.n_iter).reshape(self.n_iter, self.n_policies)

        for j, policy in enumerate(self.policy_list):
            for i in np.arange(self.n_iter):
                p = copy.deepcopy(policy)
                rewards: list = []
                idx = np.random.choice(n_data, size=n_data, replace=True)
                _X_boot, _a_boot, _r_boot, _r_pred_boot, _ps_boot = \
                    X[idx], a[idx], r[idx], self.r_pred[idx], self.ps[idx]

                for _x, _a, _r, _r_pred, _ps in zip(_X_boot, _a_boot, _r_boot, _r_pred_boot, _ps_boot):
                    x = np.array(_x)
                    chosen_arm = p.select_arm(x) if p._policy_type == "contextual" else p.select_arm()

                    reward = (np.array(chosen_arm == _a, dtype=int) * (_r - _r_pred[chosen_arm]) / _ps[chosen_arm]) + _r_pred[chosen_arm]
                    rewards.append(reward)
                    p.update(x, chosen_arm, reward) if p._policy_type == "contextual" else p.update(chosen_arm, reward)

                values[i, j] = np.mean(rewards)

        return DataFrame(values, columns=[p.name for p in self.policy_list])


class MRDREstimator(DREstimator):
    """More Robust Doubly Robust Estimator.

    policy_list: List[Union[BasePolicy, BaseThompsonSampling, BaseContextualPolicy]]
        A candidates of the new policy.

    n_iter: int
        The number of bootstrap iteration.

    regression: int
        Whether the reward is continuous or not.

    References
    -------
    [1] M. Farajtabar, Y. Chow, and M. Ghavamzadeh. More robust doubly robust off-policy evaluation. arXiv preprint arXiv:1802.03493, 2018.

    """

    def fit(self, pom: BaseEstimator, pse: BaseEstimator,
            X: np.ndarray, a: np.ndarray, r: np.ndarray) -> None:
        """Fit potential outcome models.

        Parameters
        ----------
        pom: BaseEstimator
            Potential Outcome Model

        pse: BaseEstimator
            Propensity Score Estimator

        X : array-like, shape = (n_samples, n_features)
            A test sample collected by some past policies.

        a: array-like, shape = (n_samples, )
            Action log of the past policies.

        r: array-like, shape = (n_samples, )
            Reward log of the past policies.

        """
        pse_ = pse
        pse_.fit(X, a)
        self.ps = pse_.predict_proba(X)

        pred_list: list = []
        for w in np.unique(a):
            pom_ = pom
            weight = (1.0 - self.ps[a == w, w]) / (self.ps[a == w, w] ** 2)
            pom_.fit(X[a == w], r[a == w], sample_weight=weight)
            if self.regression:
                pred_list.append(np.expand_dims(pom_.predict(X), axis=1))
            else:
                pred_list.append(np.expand_dims(pom_.predict_proba(X)[:, 1], axis=1))

        self.r_pred = np.concatenate(pred_list, axis=1)
