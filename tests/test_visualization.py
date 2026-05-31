import numpy as np

from crop_classification.visualization import majority_filter


def test_majority_filter_smooths_noise():
    arr = np.array(
        [
            [1, 1, 2],
            [1, 2, 2],
            [3, 2, 2],
        ],
        dtype=int,
    )
    filtered = majority_filter(arr, window_size=3)
    assert filtered.shape == arr.shape
    assert filtered[1, 1] == 2
    assert filtered[0, 0] == 1
