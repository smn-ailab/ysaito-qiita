"""Some useful functions."""
from typing import Union

import numpy as np


def sigmoid(x: np.ndarray) -> float:
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


def _check_stochastic_input(n_arms: int, batch_size: int) -> None:

    if not isinstance(n_arms, int):
        raise TypeError("n_arms must be an integer.")
    if not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer.")


def _check_contextual_input(n_arms: int, n_features: int, warmup: int, batch_size: int) -> None:

    if not isinstance(n_arms, int):
        raise TypeError("n_arms must be an integer.")
    if not isinstance(n_features, int):
        raise TypeError("n_features must be an integer.")
    if not isinstance(warmup, int):
        raise TypeError("warmup must be an integer.")
    if not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer.")


def _check_x_input(x: np.ndarray) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise TypeError("x must be an array.")

    return np.expand_dims(x, axis=1)


def _check_update_input(chosen_arm: int, reward: Union[int, float]) -> None:

    if not isinstance(chosen_arm, (int, np.int64)):
        raise TypeError("chosen_arm must be an integer.")
    if not isinstance(reward, (int, float)):
        raise TypeError("reward must be an integer or a float.")
