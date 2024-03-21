from typing import List
from matplotlib import pyplot as plt
import numpy as np


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def rolling_count(a: np.ndarray, window: int | float) -> np.ndarray:
    """Don't use this for anything serious without testing, it probably has boundary effects
    also doesn't work if you don't start at 0"""
    pointer = window
    prev_pointer = 0 if type(window) == int else 0.0
    res = []
    while True:
        res.append(((prev_pointer < a) & (a < pointer)).sum())

        prev_pointer = pointer
        pointer += window

        if pointer > max(a):
            break

    return np.array(res)


def pad_lists_to_array(lists: List[List | np.ndarray]):
    max_length = max(len(lst) for lst in lists)
    padded_array = np.full((len(lists), max_length), np.nan)
    for i, lst in enumerate(lists):
        padded_array[i, : len(lst)] = lst
    return padded_array


def shaded_line_plot(arr: np.ndarray, x_axis, color: str, label: str):

    mean = np.mean(arr, 0)
    std = np.std(arr, 0)
    plt.plot(x_axis, mean, color=color, label=label)
    plt.fill_between(
        x_axis,
        np.subtract(mean, std),
        np.add(mean, std),
        alpha=0.2,
        color=color,
    )
    plt.fill_between(180, 200, 0, 9)
