import concurrent.futures
import pytest
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
from tqdm import tqdm

from viral.utils import shuffle_rows
from viral.ensemble_reactivation import get_ssp_vectors


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


@pytest.fixture
def mock_trials() -> list[Mock]:
    """Creates a list of mocked TrialInfo objects for testing."""

    def mock_state(frame_idx: int, name: str = "trigger_panda") -> Mock:
        state = Mock()
        state.closest_frame_start = frame_idx
        state.name = name
        return state

    trial1 = Mock()
    trial1.rotary_encoder_position = [0, 10, 20, 30, 40, 50, 60]
    trial1.states_info = [mock_state(frame_idx) for frame_idx in np.arange(0, 7)]

    trial2 = Mock()
    trial2.rotary_encoder_position = [0, 15, 25, 35, 45, 55, 80]
    trial2.states_info = [mock_state(frame_idx) for frame_idx in np.arange(10, 17)]

    place_cells = np.zeros(shape=(10, 100))
    place_cells[:, 0:7] = 1  # activity in trial 1
    place_cells[:, 10:17] = 2  # activity in trial 2

    return [trial1, trial2], place_cells


def test_get_ssp_vectors(mock_trials) -> None:
    trials, place_cells = mock_trials
    sigma = 30
    mode = "above"
    speed_threshold = 1
    n_consecutive_samples = 1

    with patch(
        "viral.ensemble_reactivation.degrees_to_cm",
        side_effect=lambda x, _: np.array(x),
    ):
        with patch(
            "viral.ensemble_reactivation.get_wheel_circumference_from_rig",
            return_value=1,
        ):
            with patch(
                "viral.ensemble_reactivation.compute_speed_grosmark",
                side_effect=lambda position: np.ones_like(position, dtype=float) * 10,
            ):
                with patch(
                    "viral.ensemble_reactivation.gaussian_filter1d",
                    side_effect=lambda input, sigma, axis: input,
                ):
                    (
                        result_ssp_vectors,
                        result_position_vectors,
                        result_trial_start_indices,
                    ) = get_ssp_vectors(
                        trials,
                        place_cells,
                        sigma,
                        mode,
                        speed_threshold,
                        n_consecutive_samples,
                        None,
                    )

    expected_ssp_vectors = np.zeros(shape=(10, 14))
    expected_ssp_vectors[:, 0:7] = 1  # activity in trial 1
    expected_ssp_vectors[:, 7:14] = 2  # activity in trial 2
    expected_position_vectors = np.array(
        [0, 10, 20, 30, 40, 50, 60, 0, 15, 25, 35, 45, 55, 80]
    )
    expected_trial_start_indices = np.array([0, 7])
    assert np.array_equal(result_ssp_vectors, expected_ssp_vectors)
    assert np.array_equal(result_position_vectors, expected_position_vectors)
    assert np.array_equal(result_trial_start_indices, expected_trial_start_indices)


@pytest.fixture
def mock_trials_min_chunk_length() -> list[Mock]:
    """Creates a list of mocked TrialInfo objects for testing."""

    def mock_state(frame_idx: int, name: str = "trigger_panda") -> Mock:
        state = Mock()
        state.closest_frame_start = frame_idx
        state.name = name
        return state

    trial1 = Mock()
    trial1.rotary_encoder_position = [0, 10, 20, 30]
    trial1.states_info = [mock_state(frame_idx) for frame_idx in np.arange(0, 4)]

    trial2 = Mock()
    trial2.rotary_encoder_position = [0, 15, 25, 35, 45, 55, 80]
    trial2.states_info = [mock_state(frame_idx) for frame_idx in np.arange(10, 17)]

    place_cells = np.zeros(shape=(10, 100))
    place_cells[:, 0:4] = 1  # activity in trial 1
    place_cells[:, 10:17] = 2  # activity in trial 2

    return [trial1, trial2], place_cells


def test_get_ssp_vectors_min_chunk_length(mock_trials_min_chunk_length) -> None:
    trials, place_cells = mock_trials_min_chunk_length
    sigma = 30
    mode = "above"
    speed_threshold = 1
    n_consecutive_samples = 1
    min_chunk_len = 5

    with patch(
        "viral.ensemble_reactivation.degrees_to_cm",
        side_effect=lambda x, _: np.array(x),
    ):
        with patch("viral.utils.get_wheel_circumference_from_rig", return_value=1):
            with patch(
                "viral.ensemble_reactivation.compute_speed_grosmark",
                side_effect=lambda position: np.ones_like(position, dtype=float) * 10,
            ):
                with patch(
                    "viral.ensemble_reactivation.gaussian_filter1d",
                    side_effect=lambda input, sigma, axis: input,
                ):
                    (
                        result_ssp_vectors,
                        result_position_vectors,
                        result_trial_start_indices,
                    ) = get_ssp_vectors(
                        trials,
                        place_cells,
                        sigma,
                        mode,
                        speed_threshold,
                        n_consecutive_samples,
                        min_chunk_len,
                    )

    expected_ssp_vectors = np.zeros(shape=(10, 7))
    expected_ssp_vectors[:, 0:7] = 2  # activity in trial 2
    expected_position_vectors = np.array([0, 15, 25, 35, 45, 55, 80])
    expected_trial_start_indices = np.array([0])
    assert np.array_equal(result_ssp_vectors, expected_ssp_vectors)
    assert np.array_equal(result_position_vectors, expected_position_vectors)
    assert np.array_equal(result_trial_start_indices, expected_trial_start_indices)


@pytest.fixture
def mock_trials_first_chunk_filtered_out() -> list[Mock]:
    """Creates a list of mocked TrialInfo objects for testing."""

    def mock_state(frame_idx: int, name: str = "trigger_panda") -> Mock:
        state = Mock()
        state.closest_frame_start = frame_idx
        state.name = name
        return state

    trial1 = Mock()
    trial1.rotary_encoder_position = [0, 10, 20, 30, 40, 50, 60]
    trial1.states_info = [mock_state(frame_idx) for frame_idx in np.arange(0, 3)] + [
        mock_state(frame_idx) for frame_idx in np.arange(7, 11)
    ]

    trial2 = Mock()
    trial2.rotary_encoder_position = [0, 15, 25, 35, 45, 55, 80]
    trial2.states_info = [mock_state(frame_idx) for frame_idx in np.arange(10, 17)]

    place_cells = np.zeros(shape=(10, 100))
    place_cells[:, 0:3] = 1  # activity in trial 1, chunk 1
    place_cells[:, 7:11] = 1  # activity in trial 1, chunk 2
    place_cells[:, 10:17] = 2  # activity in trial 2

    return [trial1, trial2], place_cells


def test_get_ssp_vectors_first_chunk_filtered_out(
    mock_trials_first_chunk_filtered_out,
) -> None:
    trials, place_cells = mock_trials_first_chunk_filtered_out
    sigma = 30
    mode = "above"
    speed_threshold = 1
    n_consecutive_samples = 1
    min_chunk_len = 4

    with patch(
        "viral.ensemble_reactivation.degrees_to_cm",
        side_effect=lambda x, _: np.array(x),
    ):
        with patch("viral.utils.get_wheel_circumference_from_rig", return_value=1):
            with patch(
                "viral.ensemble_reactivation.compute_speed_grosmark",
                side_effect=lambda position: np.ones_like(position, dtype=float) * 10,
            ):
                with patch(
                    "viral.ensemble_reactivation.gaussian_filter1d",
                    side_effect=lambda input, sigma, axis: input,
                ):
                    (
                        result_ssp_vectors,
                        result_position_vectors,
                        result_trial_start_indices,
                    ) = get_ssp_vectors(
                        trials,
                        place_cells,
                        sigma,
                        mode,
                        speed_threshold,
                        n_consecutive_samples,
                        min_chunk_len,
                    )

    expected_ssp_vectors = np.zeros(shape=(10, 11))
    expected_ssp_vectors[:, 0:3] = 1  # activity in trial 1, chunk 2
    expected_ssp_vectors[:, 3:11] = 2  # activity in trial 2
    expected_position_vectors = np.array([30, 40, 50, 60, 0, 15, 25, 35, 45, 55, 80])
    expected_trial_start_indices = np.array([0, 4])
    assert np.array_equal(result_ssp_vectors, expected_ssp_vectors)
    assert np.array_equal(result_position_vectors, expected_position_vectors)
    assert np.array_equal(result_trial_start_indices, expected_trial_start_indices)


@pytest.fixture
def mock_trials_all_chunks_filtered_out() -> list[Mock]:
    """Creates a list of mocked TrialInfo objects for testing."""

    def mock_state(frame_idx: int, name: str = "trigger_panda") -> Mock:
        state = Mock()
        state.closest_frame_start = frame_idx
        state.name = name
        return state

    trial1 = Mock()
    trial1.rotary_encoder_position = [0, 10, 20, 30, 40, 50]
    trial1.states_info = [mock_state(frame_idx) for frame_idx in np.arange(0, 3)] + [
        mock_state(frame_idx) for frame_idx in np.arange(7, 10)
    ]

    trial2 = Mock()
    trial2.rotary_encoder_position = [0, 15, 25, 35, 45, 55, 80]
    trial2.states_info = [mock_state(frame_idx) for frame_idx in np.arange(10, 17)]

    place_cells = np.zeros(shape=(10, 100))
    place_cells[:, 0:3] = 1  # activity in trial 1, chunk 1
    place_cells[:, 7:10] = 1  # activity in trial 1, chunk 2
    place_cells[:, 10:17] = 2  # activity in trial 2

    return [trial1, trial2], place_cells


def test_get_ssp_vectors_all_chunks_filtered_out(
    mock_trials_all_chunks_filtered_out,
) -> None:
    trials, place_cells = mock_trials_all_chunks_filtered_out
    sigma = 30
    mode = "above"
    speed_threshold = 1
    n_consecutive_samples = 1
    min_chunk_len = 4

    with patch(
        "viral.ensemble_reactivation.degrees_to_cm",
        side_effect=lambda x, _: np.array(x),
    ):
        with patch("viral.utils.get_wheel_circumference_from_rig", return_value=1):
            with patch(
                "viral.ensemble_reactivation.compute_speed_grosmark",
                side_effect=lambda position: np.ones_like(position, dtype=float) * 10,
            ):
                with patch(
                    "viral.ensemble_reactivation.gaussian_filter1d",
                    side_effect=lambda input, sigma, axis: input,
                ):
                    (
                        result_ssp_vectors,
                        result_position_vectors,
                        result_trial_start_indices,
                    ) = get_ssp_vectors(
                        trials,
                        place_cells,
                        sigma,
                        mode,
                        speed_threshold,
                        n_consecutive_samples,
                        min_chunk_len,
                    )

    expected_ssp_vectors = np.zeros(shape=(10, 7))
    expected_ssp_vectors[:, 0:7] = 2  # activity in trial 2
    expected_position_vectors = np.array([0, 15, 25, 35, 45, 55, 80])
    expected_trial_start_indices = np.array([0])
    assert np.array_equal(result_ssp_vectors, expected_ssp_vectors)
    assert np.array_equal(result_position_vectors, expected_position_vectors)
    assert np.array_equal(result_trial_start_indices, expected_trial_start_indices)


# no trials won't return any trial indices, position vectors or ssp vectors as per code
