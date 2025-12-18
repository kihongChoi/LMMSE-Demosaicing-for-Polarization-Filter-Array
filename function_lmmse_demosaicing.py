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


def function_lmmse_demosaicing(mos_img, D, mosaic='pfa'):
    """
    Demosaic a PFA/CPFA image with pre-trained LMMSE algorithm.

    Args:
        mos_img: Mosaiced image array
        D: Pre-trained demosaicing matrix
        mosaic: Type of mosaic ('pfa' or 'cpfa'). If not provided, will be inferred from D matrix shape.

    Returns:
        DemosImg: Demosaiced image array of shape (rows, cols, channels, 4)
                  where 4 represents the 4 polarization angles (0°, 45°, 90°, 135°)
    """
    # Always infer mosaic type from D matrix shape to ensure correctness
    # D matrix shape: (P * height * width, nh * nw * P)
    # For PFA: (4*2*2, 8*8*4) = (16, 256)
    # For CPFA: (12*4*4, 10*10*12) = (192, 1200)
    D_rows, D_cols = D.shape
    inferred_mosaic = None
    
    if D_rows == 16 and D_cols == 256:
        inferred_mosaic = 'pfa'
    elif D_rows == 192 and D_cols == 1200:
        inferred_mosaic = 'cpfa'
    else:
        # Try to infer from dimensions
        # PFA: D_rows should be 16, CPFA: D_rows should be 192
        if D_rows == 16:
            inferred_mosaic = 'pfa'
        elif D_rows == 192:
            inferred_mosaic = 'cpfa'
        else:
            raise ValueError(f"Cannot infer mosaic type from D matrix shape {D.shape}. Please specify mosaic='pfa' or 'cpfa'.")
    
    # Use inferred mosaic type, but warn if user specified a different type
    if mosaic != 'pfa' and mosaic != 'cpfa':
        # If mosaic was not explicitly set or was 'auto', use inferred
        mosaic = inferred_mosaic
    elif mosaic != inferred_mosaic:
        # User specified a different type than what D matrix indicates
        print(f"Warning: D matrix shape {D.shape} suggests mosaic type '{inferred_mosaic}', but '{mosaic}' was specified. Using '{inferred_mosaic}'.")
        mosaic = inferred_mosaic
    else:
        # Types match, use the specified one
        pass

    # Sizes definition
    if mosaic == 'cpfa':
        height = 4  # height of the superpixel
        width = 4  # width of the superpixel
        nh = 10  # number of neighbors per column
        nw = 10  # number of neighbors per line
        P = 12  # number of color-pola channels
    elif mosaic == 'pfa':
        height = 2  # height of the superpixel
        width = 2  # width of the superpixel
        nh = 8  # number of neighbors per column
        nw = 8  # number of neighbors per line
        P = 4  # number of color-pola channels
    else:
        raise ValueError(f"Unknown mosaic type: {mosaic}")

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

    # border_size should match the training parameters
    # For both PFA and CPFA, border_size is (3, 3) based on d_matrix.py
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
