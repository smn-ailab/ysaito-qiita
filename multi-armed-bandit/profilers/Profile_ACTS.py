import argparse
import logging
import copy
import math
import time

import numpy as np
import random

from pandas import DataFrame, Series

import line_profiler

sys.path.append("/Users/y.saito/notebook/GeneralSemi/GeneralSeminar2018/ContextualMAB/")
from mab_tools.contextual_mab import ACTS
from mab_tools.test_mab import sim_conmabs_norm

# ログ出力設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_mabs(player, z_dim, x_dim, n_arms, rounds):
    times = 0
    start = time.time()

    for t in np.arange(0, rounds):
        x = np.matrix(np.random.randint(2, size=z_dim+x_dim-1)).T
        x = np.concatenate([np.matrix(np.array([1])).T, x])
        base = np.random.normal(5, 2)

        i = player.select_arm(x)

        if i == 0:
            reward = base
        else:
            reward = base + arms[i - 1].dot(x)

        player.update(x, i, reward)

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
    
    # arms指定
    common = np.matrix([np.random.normal(loc=3, scale=1, size=z_dim) for i in range(n_arms)])
    unique = np.matrix([np.random.normal(loc=-3, scale=1, size=x_dim) for i in range(n_arms)])
    arms = np.concatenate([common, unique], axis=1)


    # profileに用いるアルゴリズムの指定.
    algo = ACTS(n_arms=n_arms, feature_dim=z_dim+x_dim)
    
    logger.info("シミュレーション開始")
    run_mabs(player=algo, z_dim=z_dim, x_dim=x_dim, n_arms=n_arms, rounds=rounds)
    