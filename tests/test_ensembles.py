import concurrent.futures
from typing import Any

import numpy as np
from tqdm import tqdm

from viral.utils import shuffle_rows


def offline_reactivation_mocked(
    reactivation: np.ndarray, ensemble_matrix: np.ndarray
) -> np.ndarray:
    return reactivation @ ensemble_matrix.T


def test_shuffle_concurrencey() -> None:
    """Make sure that the concurrent shuffling doesn't do anything weird."""
    ensemble_matrix = np.random.randint(0, 1000, size=(100, 50))
    reactivation = np.random.rand(1000, 50)
    preactivation = np.random.rand(1000, 50)

    def compute_shuffled_strength(_: Any) -> tuple[np.ndarray, np.ndarray]:
        ensemble_matrix_shuffled = shuffle_rows(ensemble_matrix)
        return (
            offline_reactivation_mocked(
                reactivation=reactivation,
                ensemble_matrix=ensemble_matrix_shuffled,
            ),
            offline_reactivation_mocked(
                reactivation=preactivation,
                ensemble_matrix=ensemble_matrix_shuffled,
            ),
        )

    n_shuffles = 500
    reactivation_strength_shuffled = []
    preactivation_strength_shuffled = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(compute_shuffled_strength, range(n_shuffles)),
                total=n_shuffles,
            )
        )
        for reac, preac in results:
            reactivation_strength_shuffled.append(reac)
            preactivation_strength_shuffled.append(preac)

    reactivation_strength_shuffled = np.array(reactivation_strength_shuffled)
    preactivation_strength_shuffled = np.array(preactivation_strength_shuffled)

    assert (
        reactivation_strength_shuffled.shape
        == preactivation_strength_shuffled.shape
        == (n_shuffles, reactivation.shape[0], ensemble_matrix.shape[0])
    )

    def check_any_shuffles_the_same(arr: np.ndarray) -> bool:
        n_shuffles = arr.shape[0]
        first = arr[0, :, :]
        for shuffle in range(1, n_shuffles):
            if np.all(first == arr[shuffle, :, :]):
                return True
        return False

    assert not check_any_shuffles_the_same(reactivation_strength_shuffled)
    assert not check_any_shuffles_the_same(preactivation_strength_shuffled)

    ninety_5th_re = np.percentile(reactivation_strength_shuffled, 95, axis=0)
    ninety_5th_pre = np.percentile(preactivation_strength_shuffled, 95, axis=0)

    assert (
        ninety_5th_re.shape
        == ninety_5th_pre.shape
        == (
            reactivation.shape[0],
            ensemble_matrix.shape[0],
        )
    )
    assert np.all(ninety_5th_re > np.mean(reactivation_strength_shuffled, axis=0))
    assert np.all(ninety_5th_pre > np.mean(preactivation_strength_shuffled, axis=0))
