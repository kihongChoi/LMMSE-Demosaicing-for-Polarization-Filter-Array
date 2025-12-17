"""This function demosaic a PFA image with pre-trained LMMSE algorithm.

References:
1. Spote A, Lapray PJ, Thomas JB, Farup I. Joint demosaicing of
   colour and polarisation from filter arrays.
   In 29th Color and Imaging Conference Final Program and Proceedings 2021.
   Society for Imaging Science and Technology.

2. Dumoulin R, Lapray P.-J., Thomas J.-B., (2022), Impact of training data on
   LMMSE demosaicing for Colour-Polarization Filter Array,  16th International
   Conference on Signal-Image Technology & Internet-Based Systems (SITIS),
   2022, Dijon, France.
"""

import numpy as np
from Function.blockproc import blockproc


def function_lmmse_demosaicing(mos_img, D):
    """
    Demosaic a PFA image with pre-trained LMMSE algorithm.

    Args:
        mos_img: Mosaiced image array
        D: Pre-trained demosaicing matrix

    Returns:
        DemosImg: Demosaiced image array of shape (rows, cols, channels, 4)
                  where 4 represents the 4 polarization angles (0°, 45°, 90°, 135°)
    """
    # Sizes definition
    height = 2  # height of the superpixel
    width = 2  # width of the superpixel
    P = 4  # number of color-pola channels

    rows, cols = mos_img.shape[:2]
    r_superpix = int(rows / height)  # number of superpixel in a line
    c_superpix = int(cols / width)  # number of superpixel in a column

    # Demosaicing
    def fun(x):
        return (D @ x.flatten()).reshape(height, width, P)

    # Handle single channel mosaiced image
    if len(mos_img.shape) == 2:
        mos_img_2d = mos_img
    elif len(mos_img.shape) == 3 and mos_img.shape[2] == 1:
        mos_img_2d = mos_img[:, :, 0]
    else:
        mos_img_2d = mos_img

    demos_img = blockproc(
        mos_img_2d,
        (height, width),
        fun,
        border_size=(3, 3),
        trim_border=False,
        use_parallel=True
    )

    # Reshape to (r_superpix*height, c_superpix*width, P/4, 4)
    DemosImg = demos_img.reshape(
        r_superpix * height,
        c_superpix * width,
        P // 4,
        4
    )

    return DemosImg
