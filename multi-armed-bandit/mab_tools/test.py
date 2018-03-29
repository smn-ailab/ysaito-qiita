import math
import random

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import plotly.graph_objs as go
import plotly.plotly as py
from plotly.graph_objs import *
from plotly.offline import init_notebook_mode, iplot, plot


class BernoulliArm():
    def __init__(self, p: float) -> None:
        self.p = p

    def draw(self) -> float:
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0


def test_algorithm(algo, arms: list, num_sims: int, horizon: int) -> list:
    chosen_arms = np.zeros(num_sims * horizon)
    rewards = np.zeros(num_sims * horizon)
    cumulative_rewards = np.zeros(num_sims * horizon)
    sim_nums = np.zeros(num_sims * horizon)
    times = np.zeros(num_sims * horizon)

    for sim in range(num_sims):
        sim = sim + 1
        algo.initialize(len(arms))

        for t in range(horizon):
            t = t + 1
            index = (sim - 1) * horizon + t - 1
            sim_nums[index] = sim
            times[index] = t

            chosen_arm = algo.select_arm()
            chosen_arms[index] = chosen_arm

            reward = arms[chosen_arms[index]].draw()
            rewards[index] = reward

            if t == 1:
                cumulative_rewards[index] = reward
            else:
                cumulative_rewards[index] = cumulative_rewards[index - 1] + reward

            algo.update(chosen_arm, reward)

    return [sim_nums, times, chosen_arms, rewards, cumulative_rewards]


def to_frame(test_list: list) -> pd.DataFrame:
    df = DataFrame({"sim_nums": test_list[0],
                    "times": test_list[1],
                    "chosen_arm": test_list[2],
                    "reward": test_list[3],
                    "cumulative_rewards": test_list[4]
                    }).set_index(["sim_nums", "times"])
    return df


def average_rewards(df_list: list, name_list: list) -> None:

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


def cumulative_rewards(df_list: list, name_list: list) -> None:

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
