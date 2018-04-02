"""This Module contains tools to evaluate Multi-Armed Bandit Algorithms."""
import copy

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import plotly.graph_objs as go
import plotly.plotly as py
from plotly.graph_objs import *
from plotly.offline import init_notebook_mode, iplot, plot


class BernoulliArm():
    """Arm which gives reward(0.0 or 1.0) from a given bernoulli distribution."""

    def __init__(self, p: float) -> None:
        """Initialize class.

        :param p: the parameter of the bernoulli distribution.
        """
        self.p = p

    def draw(self) -> float:
        """Drow the arm to get a reward."""
        if np.random.rand() > self.p:
            return 0.0
        else:
            return 1.0


def sim_mabs(algo_list: list, arms: list, num_sims: int, horizon: int) -> list:
    """Run experiments on Multi-Armed Bandit problem using several algorithms.

    :param algo_list: a list of algorithms which are to be evaluated.
    :param arms: a list of arm.
    :param num_sims: the num of simulations.
    :param horizon: the num of trial in a single simulation.

    :return: a list of pd.DataFrame which contains results of the simulations for earch algorithms.
    """
    sim_data_list = []
    for algo in algo_list:
        chosen_arms = np.zeros(num_sims * horizon)
        rewards = np.zeros(num_sims * horizon)
        cumulative_rewards = np.zeros(num_sims * horizon)
        sim_nums = np.zeros(num_sims * horizon)
        times = np.zeros(num_sims * horizon)

        for sim in range(num_sims):
            a = copy.deepcopy(algo)
            sim = sim + 1

            for t in range(horizon):
                t += 1
                index = (sim - 1) * horizon + t - 1
                sim_nums[index] = sim
                times[index] = t

                chosen_arm = a.select_arm()
                chosen_arms[index] = chosen_arm

                reward = arms[chosen_arm].draw()
                rewards[index] = reward

                if t == 1:
                    cumulative_rewards[index] = reward
                else:
                    cumulative_rewards[index] = cumulative_rewards[index - 1] + reward

                a.update(chosen_arm, reward)

        sim_data = [sim_nums, times, chosen_arms, rewards, cumulative_rewards]
        df = DataFrame({"sim_nums": sim_data[0],
                        "times": sim_data[1],
                        "chosen_arm": sim_data[2],
                        "reward": sim_data[3],
                        "cumulative_rewards": sim_data[4]
                        }).set_index(["sim_nums", "times"])

        sim_data_list.append(df)

    return sim_data_list


def average_rewards(df_list: list, name_list: list) -> go.Scatter:
    """Evaluate the ability to explore the best arm during the given trials.

    :param df_list: a list of pd.DataFrame which contains results of the simulations for earch algorithms.
                    the output of the function sim_mabs.
    :param name_list: a list of the name of algorithms which are to be evaluated.

    :return: a scatter plot whose x and y axis are the num of trials and average rewards for each trial respectively.
    """
    data = []
    for df, name in zip(df_list, name_list):
        trace = Scatter(x=df.mean(level="times").index,
                        y=df.mean(level="times").reward,
                        mode="lines",
                        name=f"{name}")

        data.append(trace)

    layout = go.Layout(
        xaxis=dict(title="Times"),
        yaxis=dict(title="Average Reward"),
        title="Performance of Algorithms")
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


def cumulative_rewards(df_list: list, name_list: list) -> go.Scatter:
    """Evaluate the ability to maximize the cumulative reward during the given trials.

    :param df_list: a list of pd.DataFrame which contains results of the simulations for earch algorithms.
                    the output of the function sim_mabs.
    :param name_list: a list of the name of algorithms which are to be evaluated.

    :return: a scatter plot whose x and y axis are the num of trials and cumulative rewards for each trial respectively.
    """
    data = []
    for df, name in zip(df_list, name_list):
        trace = Scatter(x=df.mean(level="times").index,
                        y=df.mean(level="times").cumulative_rewards,
                        mode="lines",
                        name=f"{name}")

        data.append(trace)

    layout = go.Layout(
        xaxis=dict(title="Times"),
        yaxis=dict(title="Average Reward"),
        title="Cumulative Reward of Algorithms")
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
