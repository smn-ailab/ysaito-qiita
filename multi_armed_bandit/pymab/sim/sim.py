"""This Module contains tools to evaluate Multi-Armed Bandit Algorithms."""
import copy
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
from pandas import DataFrame

from plotly.graph_objs import Figure, Layout, Scatter
from plotly.offline import iplot, plot
from pymab.bandit import BaseBandit
from pymab.policy import BaseContextualPolicy, BasePolicy, BaseThompsonSampling


class BanditSimulator():
    """Bandit Simulator.

    Parameters
    ----------
    policy_list: List[Union[BasePolicy, BaseThompsonSampling, BaseContextualPolicy]]
        List of Policy classes.

    bandit: BaseBandit
        A bandit class.

    n_rounds: int
        The number of rounds in a simulation.

    num_sims: int
        The number of simulation.

    contextual: bool
        Whether rewards are modeled as contextual or not.

    """

    def __init__(self, policy_list: List[Union[BasePolicy, BaseThompsonSampling, BaseContextualPolicy]],
                 bandit: BaseBandit, n_rounds: int, num_sims: int, contextual: bool=False) -> None:
        """Initialize Class."""
        self.policy_list = policy_list
        self.bandit = bandit
        self.n_rounds = n_rounds
        self.num_sims = num_sims
        self.contextual = contextual

    def generate_data(self) -> None:
        """Generate data for bandit simulations."""
        self.data = [np.concatenate([np.expand_dims(np.random.randint(2, size=self.bandit.n_features), axis=1)
                                     for i in np.arange(self.n_rounds)], axis=1)
                     for j in np.arange(self.num_sims)]

    def run_sim_stochastic(self) -> None:
        """Run simulations in stochastic bandit settings."""
        self.result_list: list = []

        for policy in self.policy_list:
            chosen_arms = np.zeros(self.num_sims * self.n_rounds, dtype=int)
            rewards = np.zeros(self.num_sims * self.n_rounds)
            regrets = np.zeros(self.num_sims * self.n_rounds)
            successes = np.zeros(self.num_sims * self.n_rounds, dtype=int)
            sim_nums = np.zeros(self.num_sims * self.n_rounds, dtype=int)
            rounds = np.zeros(self.num_sims * self.n_rounds, dtype=int)
            elapsed_time = np.zeros(self.num_sims)

            for sim in np.arange(self.num_sims):
                p, b = copy.deepcopy(policy), copy.deepcopy(self.bandit)

                start = time.time()
                for t in np.arange(self.n_rounds):
                    index, sim_nums[index], rounds[index] = \
                        sim * self.n_rounds + t, sim + 1, t + 1

                    chosen_arm = p.select_arm()
                    chosen_arms[index] = chosen_arm

                    reward = b.pull(chosen_arm)
                    rewards[index], regrets[index] = b.rewards, b.regrets
                    successes[index] = 1 if chosen_arm == b.best_arm else 0
                    p.update(chosen_arm, reward)

                elapsed_time[sim] = time.time() - start

            print(f"Avg Elapsed Time({self.n_rounds} iter) {policy.name} : {np.round(np.mean(elapsed_time), 3)}s")
            sim_data = [sim_nums, rounds, chosen_arms, rewards, regrets, successes]
            df = DataFrame({"sim_nums": sim_data[0], "rounds": sim_data[1], "chosen_arm": sim_data[2],
                            "rewards": sim_data[3], "regrets": sim_data[4], "successes": sim_data[5]}).set_index(["sim_nums", "rounds"])
            self.result_list.append(df)

    def run_sim_contextual(self) -> None:
        """Run simulations in contextual bandit settings."""
        self.result_list = []

        for policy in self.policy_list:
            n_features = self.bandit.n_features
            chosen_arms = np.zeros(self.num_sims * self.n_rounds, dtype=int)
            rewards = np.zeros(self.num_sims * self.n_rounds)
            regrets = np.zeros(self.num_sims * self.n_rounds)
            successes = np.zeros(self.num_sims * self.n_rounds, dtype=int)
            sim_nums = np.zeros(self.num_sims * self.n_rounds, dtype=int)
            rounds = np.zeros(self.num_sims * self.n_rounds, dtype=int)
            elapsed_time = np.zeros(self.num_sims)

            for sim in np.arange(self.num_sims):
                p, b = copy.deepcopy(policy), copy.deepcopy(self.bandit)

                start = time.time()
                for t in np.arange(self.n_rounds):
                    index, sim_nums[index], rounds[index] = \
                        sim * self.n_rounds + t, sim + 1, t + 1

                    x = np.copy(self.data[sim][:, t])

                    chosen_arm = p.select_arm(x) if p._policy_type == "contextual" else p.select_arm()
                    chosen_arms[index] = chosen_arm

                    reward = b.pull(x=x, chosen_arm=chosen_arm)
                    rewards[index], regrets[index] = b.rewards, b.regrets
                    successes[index] = 1 if chosen_arm == b.best_arm else 0
                    p.update(x, chosen_arm, reward) if p._policy_type == "contextual" else p.update(chosen_arm, reward)

            elapsed_time[sim] = time.time() - start
            print(f"Avg Elapsed Time({self.n_rounds} iter) {policy.name} : {np.round(np.mean(elapsed_time), 3)}s")
            sim_data = [sim_nums, rounds, chosen_arms, rewards, regrets, successes]
            df = DataFrame({"sim_nums": sim_data[0], "rounds": sim_data[1], "chosen_arm": sim_data[2],
                            "rewards": sim_data[3], "regrets": sim_data[4], "successes": sim_data[5]}).set_index(["sim_nums", "rounds"])
            self.result_list.append(df)

    def run_sim(self) -> None:
        """Run simulations."""
        if self.contextual:
            self.generate_data()
            self.run_sim_contextual()

        else:
            self.run_sim_stochastic()

    def plots(self) -> List[Scatter]:
        """Plot resutls.

        Returns
        -------
        fig_list: List[Figure]
            List of Figure isinstances to plot retulsts of bandit simulations.

        """
        fig_list: list = []

        for metric in ["rewards", "regrets", "successes"]:
            if metric == "successes":
                opacity, width = 0.9, 2
            else:
                opacity, width = 0.7, 5

            data = [Scatter(x=np.arange(self.n_rounds), y=df.mean(level="rounds")[metric],
                            opacity=opacity, line=dict(width=width), name=f"{policy.name}")
                    for df, policy in zip(self.result_list, self.policy_list)]

            layout = Layout(xaxis=dict(title="rounds"), yaxis=dict(title=metric))
            fig_list.append(Figure(data=data, layout=layout))

        return fig_list
