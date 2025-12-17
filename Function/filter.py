"""Filtering using Spectro-polarimetric Quad Bayer filter."""

import numpy as np


def filter_images(images, mosaic):
    """
    Filtering using Spectro-polarimetric Quad Bayer filter.

    Args:
        images: Input images array (rows, columns, channels)
        mosaic: Type of mosaic pattern ('pfa' or 'cpfa')

    Returns:
        SPQBf2: Filtered image with one dimension (rows x columns x 1)
    """
    rows, columns, _ = images.shape

    # Creation of the spectro-pola quad bayer filter
    if mosaic == 'cpfa':
        # Pattern: [[1 4 ; 10 7] [2 5; 11 8] ; [2 5; 11 8] [3 6 ; 12 9]]
        pattern = np.array([
            [1, 4, 2, 5],
            [10, 7, 11, 8],
            [2, 5, 3, 6],
            [11, 8, 12, 9]
        ])
    elif mosaic == 'pfa':
        # Pattern: [[1 ;4] [2 ;3]]
        pattern = np.array([
            [1, 2],
            [4, 3]
        ])
    else:
        raise ValueError(f"Unknown mosaic type: {mosaic}")

    x, y = pattern.shape
    n = len(np.unique(pattern))
    mosaic_pattern = np.tile(pattern, (rows // x, columns // y))

    # Create mask for each channel
    mask = np.zeros((rows, columns, n), dtype=bool)
    for i in range(1, n + 1):
        mask[:, :, i - 1] = (mosaic_pattern == i)

    # Filtering
    max_channel = np.max(pattern)
    B = np.zeros((rows, columns, max_channel))
    for i in range(1, max_channel + 1):
        channel_idx = i - 1
        if channel_idx < images.shape[2]:
            mask = (mosaic_pattern == i)
            B[:, :, channel_idx] = images[:, :, channel_idx] * mask

    SPQBf2 = np.sum(B, axis=2, keepdims=True)

    return SPQBf2
