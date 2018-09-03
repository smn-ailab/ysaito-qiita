import argparse
import logging
import sys

import copy
import math
import time

import numpy as np
import random

from pandas import DataFrame, Series

import line_profiler

sys.path.append("/Users/y.saito/notebook/GeneralSemi/GeneralSeminar2018/ContextualMAB/")
from mab_tools.contextual_mab import LinUCB
from mab_tools.test_mab import sim_conmabs_norm

# ログ出力設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ContextualMABのプロファイル")
    parser.add_argument("n_arms", type=int)  # 腕の数
    parser.add_argument("z_dim", type=int)  # 共有特徴量の次元
    parser.add_argument("x_dim", type=int)  # 固有特徴量の次元
    
    args = vars(parser.parse_args())  # プログラムの入力が辞書形式で登録

    n_arms = args["n_arms"]
    z_dim = args["z_dim"]
    x_dim = args["x_dim"]
    
    # armsの指定
    common = np.matrix([np.random.normal(loc=0.5, scale=0.3, size=z_dim)] * n_arms)
    unique = np.matrix([np.random.normal(loc=0.5, scale=0.3, size=x_dim) for i in range(n_arms)])
    arms = np.concatenate([common, unique], axis=1)

    # profileに用いるアルゴリズムの指定.
    algos = [LinUCB(n_arms=arms.shape[0], feature_dim=arms.shape[1])]
    key = [True]
    name_list = ["LinUCB"]
    
    logger.info("シミュレーション開始")
    result_list = sim_conmabs_norm(algo_list=algos, algo_name=name_list, context_key=key, arms=arms, scale=2.5, num_sims=10, horizon=3000)
    