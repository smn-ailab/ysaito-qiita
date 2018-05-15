import argparse
import logging
import copy
import math
import time

import numpy as np
import random

from pandas import DataFrame, Series

import line_profiler

from mab_tools.contextual_mab import LogisticTS
from mab_tools.test_mab import sim_conmabs_norm

# ログ出力設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def run_mabs(player, z_dim, x_dim, n_arms, rounds):
    comon = np.matrix([np.random.normal(loc=-1, scale=0.25, size=z_dim)] * n_arms)
    unique = np.matrix([np.random.normal(loc=-1, scale=0.25, size=x_dim) for i in np.arange(n_arms)])
    arms = np.concatenate([comon, unique], axis=1)

    n_arms = arms.shape[0]
    dim = arms.shape[1]

    rewards = 0
    success = 0
    regret = 0
    
    times = 0
    start = time.time()

    for t in np.arange(0, rounds):
        x = np.matrix(np.random.randint(2, size=dim)).T

        i = player.select_arm(x)
        e = np.random.normal(loc=0, scale=0.1)

        reward = np.random.binomial(1, sigmoid(arms[i].dot(x)))
        regret += sigmoid(np.max(arms.dot(x))) - sigmoid(arms[i].dot(x))
        rewards += np.random.binomial(1, sigmoid(np.max(arms.dot(x))))
        player.update(x, i, reward)

        i_max = np.argmax(arms.dot(x))
        if i == i_max:
            success += 1

        times += 1
        if times % (0.25 * rounds) == 0:
            print(f"{100 * times / rounds} % done")
    print(f"{round(time.time() - start, 3)}s")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ContextualMABのプロファイル")
    parser.add_argument("n_arms", type=int)  # 腕の数
    parser.add_argument("z_dim", type=int)  # 共有特徴量の次元
    parser.add_argument("x_dim", type=int)  # 固有特徴量の次元
    parser.add_argument("rounds", type=int) # num_iter
    
    args = vars(parser.parse_args())  # プログラムの入力が辞書形式で登録

    n_arms = args["n_arms"]
    z_dim = args["z_dim"]
    x_dim = args["x_dim"]
    rounds = args["rounds"]


    # profileに用いるアルゴリズムの指定.
    algo = LogisticTS(n_arms=n_arms, feature_dim=z_dim+x_dim, num_trial = rounds)
    
    logger.info("シミュレーション開始")
    run_mabs(player=algo, z_dim=z_dim, x_dim=x_dim, n_arms=n_arms, rounds=rounds)
    