import numpy as np

from crop_classification.processing import get_cloud_mask


def test_get_cloud_mask_detects_invalid_pixels():
    data = np.array(
        [
            [4, 9, 3],
            [5, 11, 7],
            [0, 4, 4],
        ]
    )
    mask = get_cloud_mask(data)
    assert mask.shape == data.shape
    assert mask[0, 0] is False
    assert mask[0, 1] is True
    assert mask[0, 2] is True
    assert mask[1, 0] is False
    assert mask[1, 1] is True
