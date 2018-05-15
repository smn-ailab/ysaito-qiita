"""This Module contains tools to evaluate Multi-Armed Bandit Algorithms."""
import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.pylabtools import figsize
from pandas import DataFrame, Series

import plotly.graph_objs as go
import plotly.plotly as py
from plotly.graph_objs import *
from plotly.offline import init_notebook_mode, iplot, plot


def sim_mabs_bern(algo_list: list, arms: list, num_sims: int, horizon: int, algo_name: list,
                  batch: bool =False, batch_size: int =0) -> list:
    """Run simulations Multi-Armed Bandit Algorithms on rewards given by Bernoulli distributions.

    :param algo_list: a list of simulated algorithms.
    :param arms: a list of means of Bernoulli Arms used in the simulations.
    :param num_sims: the number of simulations.
    :param horizon: the number of tiral in a simulation.
    :param algo_name: a list of names of the simulated algorithms.
    :param batch: whether simulations are run in the batch update situation or not.
    :param batch_size: the size of information about rewards given in a update.

    :return: a list of simulation results for each algorithm.
    """
    sim_data_list = []
    for i, algo in enumerate(algo_list):
        chosen_arms = np.zeros(num_sims * horizon, dtype=int)
        rewards = np.zeros(num_sims * horizon)
        cumulative_rewards = np.zeros(num_sims * horizon)
        p_max = max(arms)
        regrets = np.zeros(num_sims * horizon)
        cumulative_regrets = np.zeros(num_sims * horizon)
        successes = np.zeros(num_sims * horizon, dtype=int)
        sim_nums = np.zeros(num_sims * horizon, dtype=int)
        times = np.zeros(num_sims * horizon, dtype=int)
        elapsed_time = np.zeros(num_sims)

        if batch:
            algo.batch_size = batch_size

        for sim in range(num_sims):
            a = copy.deepcopy(algo)
            start = time.time()

            for t in range(horizon):
                t += 1
                index = sim * horizon + t - 1
                sim_nums[index] = sim + 1
                times[index] = t

                chosen_arm = a.select_arm()
                chosen_arms[index] = chosen_arm

                reward = np.random.binomial(n=1, p=arms[chosen_arm])
                rewards[index] = reward

                regret = p_max - arms[chosen_arm]
                regrets[index] = regret

                if t == 1:
                    cumulative_rewards[index] = reward
                    cumulative_regrets[index] = regret
                else:
                    cumulative_rewards[index] = cumulative_rewards[index - 1] + reward
                    cumulative_regrets[index] = cumulative_regrets[index - 1] + regret

                if chosen_arm == np.argmax(arms):
                    successes[index] = 1

                if batch:
                    a.batch_update(chosen_arm, reward)
                else:
                    a.update(chosen_arm, reward)
            elapsed_time[sim] = time.time() - start

        print(f"Avg Elapsed Time({horizon} iter) {algo_name[i]} : {round(np.mean(elapsed_time), 3)}s")
        sim_data = [sim_nums, times, chosen_arms, rewards, cumulative_rewards, regrets, cumulative_regrets, successes]
        df = DataFrame({"sim_nums": sim_data[0],
                        "times": sim_data[1],
                        "chosen_arm": sim_data[2],
                        "Reward": sim_data[3],
                        "Cumulative Rewards": sim_data[4],
                        "Regrets": sim_data[5],
                        "Cumulative Regrets": sim_data[6],
                        "Successes": sim_data[7],
                        }).set_index(["sim_nums", "times"])

        sim_data_list.append(df)

    return sim_data_list


def sim_mabs_norm(algo_list: list, locs: list, scales: list, num_sims: int, horizon: int, algo_name: list,
                  batch: bool =False, batch_size: int =0) -> list:
    """Run simulations Multi-Armed Bandit Algorithms on rewards given by Gaussian distributions.

    :param algo_list: a list of simulated algorithms.
    :param locs: a list of means of Gaussian Arms used in the simulations.
    :param scales: a list of variances of Gaussian Arms used in the simulations.
    :param num_sims: the number of simulations.
    :param horizon: the number of tiral in a simulation.
    :param algo_name: a list of names of the simulated algorithms.
    :param batch: whether simulations are run in the batch update situation or not.
    :param batch_size: the size of information about rewards given in a update.

    :return: a list of simulation results for each algorithm.
    """

    sim_data_list = []
    for i, algo in enumerate(algo_list):
        chosen_arms = np.zeros(num_sims * horizon, dtype=int)
        rewards = np.zeros(num_sims * horizon)
        cumulative_rewards = np.zeros(num_sims * horizon)
        mean_max = max(locs)
        regrets = np.zeros(num_sims * horizon)
        cumulative_regrets = np.zeros(num_sims * horizon)
        successes = np.zeros(num_sims * horizon, dtype=int)
        sim_nums = np.zeros(num_sims * horizon, dtype=int)
        times = np.zeros(num_sims * horizon, dtype=int)
        elapsed_time = np.zeros(num_sims)

        if batch:
            algo.batch_size = batch_size

        for sim in range(num_sims):
            a = copy.deepcopy(algo)
            start = time.time()

            for t in range(horizon):
                t += 1
                index = sim * horizon + t - 1
                sim_nums[index] = sim + 1
                times[index] = t

                chosen_arm = a.select_arm()
                chosen_arms[index] = chosen_arm

                reward = np.clip(np.random.normal(loc=locs[chosen_arm], scale=scales[chosen_arm]),
                                 0, np.max(locs) + 2 * np.max(scales))
                rewards[index] = reward

                regret = mean_max - locs[chosen_arm]
                regrets[index] = regret

                if t == 1:
                    cumulative_rewards[index] = reward
                    cumulative_regrets[index] = regret
                else:
                    cumulative_rewards[index] = cumulative_rewards[index - 1] + reward
                    cumulative_regrets[index] = cumulative_regrets[index - 1] + regret

                if chosen_arm == np.argmax(locs):
                    successes[index] = 1

                if batch:
                    a.batch_update(chosen_arm, reward)
                else:
                    a.update(chosen_arm, reward)
            elapsed_time[sim] = time.time() - start

        print(f"Avg Elapsed Time({horizon} iter) {algo_name[i]} : {round(np.mean(elapsed_time), 3)}s")
        sim_data = [sim_nums, times, chosen_arms, rewards, cumulative_rewards, regrets, cumulative_regrets, successes]
        df = DataFrame({"sim_nums": sim_data[0],
                        "times": sim_data[1],
                        "chosen_arm": sim_data[2],
                        "Reward": sim_data[3],
                        "Cumulative Rewards": sim_data[4],
                        "Regrets": sim_data[5],
                        "Cumulative Regrets": sim_data[6],
                        "Successes": sim_data[7],
                        }).set_index(["sim_nums", "times"])

        sim_data_list.append(df)

    return sim_data_list


def sigmoid(x: float) -> float:
    return (1.0 / 1.0 + np.exp(- x))


def sim_conmabs_bern(algo_list: list, arms: np.matrix, scale: float, num_sims: int, horizon: int, algo_name: list, context_key: list,
                     monitor: bool = False, batch: bool=False, batch_size: int=200) -> pd.DataFrame:
    """Run simulations Contextual Multi-Armed Bandit Algorithms on rewards given by Bernoulli distributions.

    :param algo_list: a list of simulated algorithms.
    :param arms: a matrix which contains linear parameters of every arm.
    :param scale: a variances of error given by a Gaussian distribution.
    :param num_sims: the number of simulations.
    :param horizon: the number of tiral in a simulation.
    :param algo_name: a list of names of the simulated algorithms.
    :param context_key: a list of bools which represent whether simulated algorithms are contextual or not.
    :param monitor: whether monitor simulation progress or not.
    :param batch: whether simulations are run in the batch update situation or not.
    :param batch_size: the size of information about rewards given in a update.

    :return: a list of simulation results for each algorithm.
    """
    sim_data_list = []
    for i, algo in enumerate(algo_list):
        chosen_arms = np.zeros(num_sims * horizon, dtype=int)
        successes = np.zeros(num_sims * horizon, dtype=int)
        rewards = np.zeros(num_sims * horizon)
        cumulative_rewards = np.zeros(num_sims * horizon)
        regrets = np.zeros(num_sims * horizon)
        cumulative_regrets = np.zeros(num_sims * horizon)
        sim_nums = np.zeros(num_sims * horizon, dtype=int)
        times = np.zeros(num_sims * horizon, dtype=int)
        elapsed_time = np.zeros(num_sims)

        for sim in range(num_sims):
            a = copy.deepcopy(algo)
            start = time.time()
            if batch:
                a.batch_size = batch_size

            for t in range(horizon):
                t += 1
                index = (sim - 1) * horizon + t - 1
                sim_nums[index] = sim + 1
                times[index] = t

                x = np.matrix(np.random.randint(2, size=arms.shape[1])).T
                e = np.random.normal(loc=0, scale=scale)

                if context_key[i]:
                    chosen_arm = a.select_arm(x)
                else:
                    chosen_arm = a.select_arm()
                chosen_arms[index] = chosen_arm

                p = sigmoid(arms[chosen_arm].dot(x) + e)
                reward = np.random.binomial(n=1, p=p)
                rewards[index] = reward
                regret = sigmoid(np.max(arms.dot(x))) - (arms[chosen_arm].dot(x))
                regrets[index] = regret

                if chosen_arm == np.argmax(arms.dot(x)):
                    successes[index] = 1

                if t == 1:
                    cumulative_regrets[index] = regret
                    cumulative_rewards[index] = reward
                else:
                    cumulative_regrets[index] = cumulative_regrets[index - 1] + regret
                    cumulative_rewards[index] = cumulative_rewards[index - 1] + reward

                if context_key[i]:
                    if batch:
                        a.batch_update(x, chosen_arm, reward[0][0])
                    else:
                        a.update(x, chosen_arm, reward[0][0])
                else:
                    if batch:
                        a.batch_update(chosen_arm, reward[0][0])
                    else:
                        a.update(chosen_arm, reward[0][0])

                if monitor:
                    if t == horizon:
                        print(f"sim {sim + 1}  100 % done : \
                        theta_loss {round(np.sqrt(mean_squared_error(arms, np.matrix(a.theta))), 3)}")

                    elif t % (0.25 * horizon) == 0:
                        print(f"sim {sim + 1} {100 * t / horizon} % done : \
                        theta_loss {round(np.sqrt(mean_squared_error(arms, np.matrix(a.theta))), 3)}")

        elapsed_time[sim] = time.time() - start
        print(f"Avg Elapsed Time({horizon} iter) {algo_name[i]} : {round(np.mean(elapsed_time), 3)}s")
        sim_data = [sim_nums, times, chosen_arms, rewards, cumulative_rewards,
                    regrets, cumulative_regrets, successes]

        df = DataFrame({"sim_nums": sim_data[0], "times": sim_data[1], "chosen_arm": sim_data[2],
                        "Rewards": sim_data[3], "Cumulative Rewards": sim_data[4],
                        "Regrets": sim_data[5], "Cumulative Regrets": sim_data[6],
                        "Successes": sim_data[7]}).set_index(["sim_nums", "times"])

        sim_data_list.append(df)

    return sim_data_list


def sim_conmabs_norm(algo_list: list, arms: np.matrix, scale: float, num_sims: int, horizon: int, algo_name: list, context_key: list,
                     monitor: bool = False, batch: bool=False, batch_size: int=200) -> pd.DataFrame:
    """Run simulations Contextual Multi-Armed Bandit Algorithms on rewards given by Gaussian distributions.

    :param algo_list: a list of simulated algorithms.
    :param arms: a matrix which contains linear parameters of every arm.
    :param scale: a variances of error given by a Gaussian distribution.
    :param num_sims: the number of simulations.
    :param horizon: the number of tiral in a simulation.
    :param algo_name: a list of names of the simulated algorithms.
    :param context_key: a list of bools which represent whether simulated algorithms are contextual or not.
    :param monitor: whether monitor simulation progress or not.
    :param batch: whether simulations are run in the batch update situation or not.
    :param batch_size: the size of information about rewards given in a update.

    :return: a list of simulation results for each algorithm.
    """
    sim_data_list = []
    for i, algo in enumerate(algo_list):
        chosen_arms = np.zeros(num_sims * horizon, dtype=int)
        successes = np.zeros(num_sims * horizon, dtype=int)
        rewards = np.zeros(num_sims * horizon)
        cumulative_rewards = np.zeros(num_sims * horizon)
        regrets = np.zeros(num_sims * horizon)
        cumulative_regrets = np.zeros(num_sims * horizon)
        sim_nums = np.zeros(num_sims * horizon, dtype=int)
        times = np.zeros(num_sims * horizon, dtype=int)
        elapsed_time = np.zeros(num_sims)

        for sim in range(num_sims):
            a = copy.deepcopy(algo)
            start = time.time()
            if batch:
                a.batch_size = batch_size

            for t in range(horizon):
                t += 1
                index = (sim - 1) * horizon + t - 1
                sim_nums[index] = sim + 1
                times[index] = t

                x = np.matrix(np.random.randint(2, size=arms.shape[1])).T
                e = np.random.normal(loc=0, scale=scale)

                if context_key[i]:
                    chosen_arm = a.select_arm(x)
                else:
                    chosen_arm = a.select_arm()
                chosen_arms[index] = chosen_arm

                reward = arms[chosen_arm].dot(x) + e
                rewards[index] = reward
                regret = np.max(arms.dot(x)) - arms[chosen_arm].dot(x)
                regrets[index] = regret

                if chosen_arm == np.argmax(arms.dot(x)):
                    successes[index] = 1

                if t == 1:
                    cumulative_regrets[index] = regret
                    cumulative_rewards[index] = reward
                else:
                    cumulative_regrets[index] = cumulative_regrets[index - 1] + regret
                    cumulative_rewards[index] = cumulative_rewards[index - 1] + reward

                if context_key[i]:
                    if batch:
                        a.batch_update(x, chosen_arm, reward[0][0])
                    else:
                        a.update(x, chosen_arm, reward[0][0])
                else:
                    if batch:
                        a.batch_update(chosen_arm, reward[0][0])
                    else:
                        a.update(chosen_arm, reward[0][0])

                if monitor:
                    if t == horizon:
                        print(f"sim {sim + 1}  100 % done : \
                        theta_loss {round(np.sqrt(mean_squared_error(arms, np.matrix(a.theta))), 3)}")

                    elif t % (0.25 * horizon) == 0:
                        print(f"sim {sim + 1} {100 * t / horizon} % done : \
                        theta_loss {round(np.sqrt(mean_squared_error(arms, np.matrix(a.theta))), 3)}")

        elapsed_time[sim] = time.time() - start
        print(f"Avg Elapsed Time({horizon} iter) {algo_name[i]} : {round(np.mean(elapsed_time), 3)}s")
        sim_data = [sim_nums, times, chosen_arms, rewards, cumulative_rewards,
                    regrets, cumulative_regrets, successes]

        df = DataFrame({"sim_nums": sim_data[0], "times": sim_data[1], "chosen_arm": sim_data[2],
                        "Rewards": sim_data[3], "Cumulative Rewards": sim_data[4],
                        "Regrets": sim_data[5], "Cumulative Regrets": sim_data[6],
                        "Successes": sim_data[7]}).set_index(["sim_nums", "times"])

        sim_data_list.append(df)

    return sim_data_list


def sim_acts_norm(algo_list: list, arms: np.matrix, base: np.matrix,
                  num_sims: int, horizon: int, algo_name: list, acts_key: list,
                  monitor: bool = False, batch: bool=False, batch_size: int=200) -> pd.DataFrame:
    """Run simulations Action-Centered Multi-Armed Bandit Algorithms on rewards given by Gaussian distributions.

    :param algo_list: a list of simulated algorithms.
    :param arms: a matrix which contains linear parameters of every arm.
    :param scale: a variances of error given by a Gaussian distribution.
    :param num_sims: the number of simulations.
    :param horizon: the number of tiral in a simulation.
    :param algo_name: a list of names of the simulated algorithms.
    :param context_key: a list of bools which represent whether simulated algorithms are contextual or not.
    :param monitor: whether monitor simulation progress or not.
    :param batch: whether simulations are run in the batch update situation or not.
    :param batch_size: the size of information about rewards given in a update.

    :return: a list of simulation results for each algorithm.
    """
    sim_data_list = []
    base_arms = np.concatenate([np.matrix(np.zeros(arms.shape[1])), arms], axis=0)
    for i, algo in enumerate(algo_list):

        n_arms = arms.shape[0]
        dim = arms.shape[1]
        chosen_arms = np.zeros(num_sims * horizon, dtype=int)
        successes = np.zeros(num_sims * horizon, dtype=int)
        cumulative_lifts = np.zeros(num_sims * horizon)
        base_rewards = np.zeros(num_sims * horizon)
        rewards = np.zeros(num_sims * horizon)
        cumulative_rewards = np.zeros(num_sims * horizon)
        regrets = np.zeros(num_sims * horizon)
        cumulative_regrets = np.zeros(num_sims * horizon)
        sim_nums = np.zeros(num_sims * horizon, dtype=int)
        times = np.zeros(num_sims * horizon, dtype=int)
        elapsed_time = np.zeros(num_sims)

        for sim in range(num_sims):
            a = copy.deepcopy(algo)
            start = time.time()
            if batch:
                a.batch_size = batch_size

            for t in range(horizon):
                t += 1
                index = (sim - 1) * horizon + t - 1
                sim_nums[index] = sim + 1
                times[index] = t

                x = np.matrix(np.clip(np.random.normal(loc=5, scale=2, size=dim), 0, 30)).T
                #x = np.matrix(np.random.randint(2, size=dim)).T
                e1 = np.random.normal(loc=0, scale=1.0)
                e2 = np.random.normal(loc=0, scale=1.0)

                chosen_arm = a.select_arm(x)
                chosen_arms[index] = chosen_arm

                #base_reward = base.dot(x) + e1
                base_reward = np.clip(np.random.normal(30, 30), 0, 60)
                base_rewards[index] = base_reward

                if acts_key[i]:
                    lift = base_arms[chosen_arm].dot(x)
                    reward = base_arms[chosen_arm].dot(x) + base_reward + e2
                    rewards[index] = reward
                    regret = np.max(base_arms.dot(x)) - base_arms[chosen_arm].dot(x)

                    if chosen_arm == np.argmax(base_arms.dot(x)):
                        successes[index] = 1
                else:
                    lift = arms[chosen_arm].dot(x)
                    reward = arms[chosen_arm].dot(x) + base_reward + e2
                    rewards[index] = reward
                    regret = np.max(base_arms.dot(x)) - arms[chosen_arm].dot(x)

                    if (chosen_arm + 1) == np.argmax(base_arms.dot(x)):
                        successes[index] = 1

                regrets[index] = regret

                if t == 1:
                    cumulative_lifts[index] = lift
                    cumulative_regrets[index] = regret
                    cumulative_rewards[index] = reward
                else:
                    cumulative_lifts[index] = cumulative_lifts[index - 1] + lift
                    cumulative_regrets[index] = cumulative_regrets[index - 1] + regret
                    cumulative_rewards[index] = cumulative_rewards[index - 1] + reward

                if batch:
                    a.batch_update(x, chosen_arm, reward[0][0])
                else:
                    a.update(x, chosen_arm, reward[0][0])

                if monitor:
                    if t == horizon:
                        print(f"sim {sim + 1}  100 % done : \
                        theta_loss {round(np.sqrt(mean_squared_error(arms, np.matrix(a.theta))), 3)}")

                    elif t % (0.25 * horizon) == 0:
                        print(f"sim {sim + 1} {100 * t / horizon} % done : \
                        theta_loss {round(np.sqrt(mean_squared_error(arms, np.matrix(a.theta))), 3)}")

        elapsed_time[sim] = time.time() - start
        print(f"Avg Elapsed Time({horizon} iter) {algo_name[i]} : {round(np.mean(elapsed_time), 3)}s")
        sim_data = [sim_nums, times, chosen_arms, base_rewards, rewards, cumulative_rewards,
                    regrets, cumulative_regrets, cumulative_lifts, successes]

        df = DataFrame({"sim_nums": sim_data[0], "times": sim_data[1], "chosen_arm": sim_data[2],
                        "Base Rewards": sim_data[3], "Rewards": sim_data[4], "Cumulative Rewards": sim_data[5],
                        "Regrets": sim_data[6], "Cumulative Regrets": sim_data[7], "Cumulative Lifts": sim_data[8],
                        "Successes": sim_data[9]}).set_index(["sim_nums", "times"])

        sim_data_list.append(df)

    return sim_data_list


def mab_plots(df_list: list, name_list: list, metric: str) -> go.Figure:
    """Evaluate the ability to explore the best arm during the given trials.

    :param df_list: a list of pd.DataFrame which contains results of the simulations for earch algorithms.
                    the output of the function sim_mabs_bern, sim_mabs_norm, etc.
    :param name_list: a list of the name of algorithms which are to be evaluated.
    :param metric: a metric used to evaluate Multi-Armed Bandit Algorithms.
    :return: a go.Figure incetance whose x and y axis are the num of trials and average rewards for each trial respectively.
    """

    data = []
    if metric == "Successes":
        _lw = 1.5
        _alpha = 0.9
    else:
        _lw = 3.0
        _alpha = 1.0

    for df, name in zip(df_list, name_list):
        trace = Scatter(x=df.mean(level="times").index,
                        y=df.mean(level="times")[metric],
                        opacity=_alpha,
                        line={"width": _lw},
                        mode="lines",
                        name=f"{name}")

        data.append(trace)

    layout = go.Layout(
        xaxis=dict(title="Times"),
        yaxis=dict(title=metric),
        title=f"{metric} of Bandit Algorithms")
    fig = go.Figure(data=data, layout=layout)
    return fig


def average_rewards(df_list: list, name_list: list) -> go.Figure:
    """Evaluate the ability to explore the best arm during the given trials.

    :param df_list: a list of pd.DataFrame which contains results of the simulations for earch algorithms.
                    the output of the function sim_mabs_bern, sim_mabs_norm, etc.
    :param name_list: a list of the name of algorithms which are to be evaluated.

    :return: a go.Figure incetance whose x and y axis are the num of trials and average rewards for each trial respectively.
    """
    data = []
    for df, name in zip(df_list, name_list):
        trace = Scatter(x=df.mean(level="times").index,
                        y=df.mean(level="times").reward,
                        opacity=0.9,
                        line={"width": 1.5},
                        mode="lines",
                        name=f"{name}")

        data.append(trace)

    layout = go.Layout(
        xaxis=dict(title="Times"),
        yaxis=dict(title="Average Rewards"),
        title="Average Rewards")
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


def cumulative_rewards(df_list: list, name_list: list) -> go.Figure:
    """Evaluate the ability to maximize the cumulative reward during the given trials.

    :param df_list: a list of pd.DataFrame which contains results of the simulations for earch algorithms.
                    the output of the function sim_mabs.
    :param name_list: a list of the name of algorithms which are to be evaluated.

    :return: a go.Figure incetance whose x and y axis are the num of trials and cumulative rewards for each trial respectively.
    """
    data = []
    for df, name in zip(df_list, name_list):
        trace = Scatter(x=df.mean(level="times").index,
                        y=df.mean(level="times").cumulative_rewards,
                        mode="lines",
                        line={"width": 3},
                        name=f"{name}")

        data.append(trace)

    layout = go.Layout(
        xaxis=dict(title="Times"),
        yaxis=dict(title="Average Reward"),
        title="Cumulative Rewards")
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


def success_rate(df_list: list, name_list: list) -> go.Figure:
    """Evaluate the ability to identify the best arm during the given trials.

    :param df_list: a list of pd.DataFrame which contains results of the simulations for earch algorithms.
                    the output of the function sim_mabs.
    :param name_list: a list of the name of algorithms which are to be evaluated.

    :return: a go.Figure incetance whose x and y axis are the num of trials and cumulative rewards for each trial respectively.
    """
    data = []
    for df, name in zip(df_list, name_list):
        trace = Scatter(x=df.mean(level="times").index,
                        y=df.mean(level="times").successes,
                        opacity=0.9,
                        line={"width": 1.5},
                        mode="lines",
                        name=f"{name}")

        data.append(trace)

    layout = go.Layout(
        xaxis=dict(title="Times"),
        yaxis=dict(title="Successes Rate"),
        title="Average Successes Rates")
    fig = go.Figure(data=data, layout=layout)
    return fig
