"""Compute the Demosaicking Matrix of the LMMSE method."""

import numpy as np
from scipy.linalg import block_diag
from Function.blockproc import blockproc


def d_matrix(full_dataset, mos_dataset, folder_path='', mosaic='pfa'):
    """
    Compute the Demosaicking Matrix of the LMMSE method.

    Args:
        full_dataset: List of tuples (name, img_0, img_45, img_90, img_135)
        mos_dataset: List of tuples (name, mosaicked_image)
        folder_path: Path to save results
        mosaic: Type of mosaic ('pfa' or 'cpfa')

    Returns:
        DemosDataset: Demosaiced dataset
        y1: Unfolding of non-mosaicked images by neighborhood
        y: Unfolding of non-mosaicked images by superpixel
        D: Demosaicking matrix
    """
    # -------------------------------------------------------------------------
    #                    Compute the Demosaicking Matrix
    # -------------------------------------------------------------------------

    # Setting parameters
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

    rows, cols = full_dataset[0][1].shape[:2]
    if len(full_dataset[0][1].shape) == 3:
        color = full_dataset[0][1].shape[2]
    else:
        color = 1

    r_superpix = int(rows / height)  # number of superpixel in a line
    c_superpix = int(cols / width)  # number of superpixel in a column
    Len = len(full_dataset)  # number of images in the database

    print('Parameters set')

    # Compute y1 for all images of the database
    y1 = []
    for d in range(Len):
        im_nbr = d + 1
        print(f'Processing image {im_nbr}')
        
        y_per_img3 = np.zeros((nh * nw, P, r_superpix - 2, c_superpix - 2))

        # Reshape image dataset for simplicity
        matrix = np.stack([
            full_dataset[d][1],  # img_0
            full_dataset[d][2],  # img_45
            full_dataset[d][3],  # img_90
            full_dataset[d][4]   # img_135
        ], axis=2)
        
        # Reshape to (rows, cols, P)
        if matrix.shape[2] != P:
            # If grayscale, expand to match P channels
            if matrix.shape[2] == 4 and P == 4:
                pass  # Already correct
            else:
                # Expand grayscale to match P
                matrix = matrix.reshape(rows, cols, -1)
                if matrix.shape[2] < P:
                    # Repeat channels to match P
                    repeat_factor = P // matrix.shape[2]
                    matrix = np.repeat(matrix, repeat_factor, axis=2)[:, :, :P]

        # Compute the column vector for each superpixel
        y_per_img2 = np.zeros((nh * nw, P, r_superpix, c_superpix))
        
        for i in range(P):
            def fun(x):
                # Reshape to (1, 1, nw*nh) similar to MATLAB
                return x.flatten().reshape(1, 1, nw * nh)
            
            result = blockproc(
                matrix[:, :, i],
                (height, width),
                fun,
                border_size=(3, 3),
                trim_border=False,
                use_parallel=True
            )
            
            # MATLAB: permute(blockproc(...), [3 1 2])
            # Result shape is (1, 1, nw*nh, r_superpix, c_superpix) after blockproc
            # We need to reshape to (nw*nh, r_superpix, c_superpix)
            if len(result.shape) == 5:
                # (1, 1, nw*nh, r_superpix, c_superpix) -> (nw*nh, r_superpix, c_superpix)
                result_reshaped = result[0, 0, :, :, :]
            elif len(result.shape) == 3:
                # Already in correct shape or needs reshaping
                if result.shape[0] == 1 and result.shape[1] == 1:
                    result_reshaped = result[0, 0, :].reshape(nw * nh, r_superpix, c_superpix)
                else:
                    result_reshaped = result.reshape(nw * nh, r_superpix, c_superpix)
            else:
                result_reshaped = result.reshape(nw * nh, r_superpix, c_superpix)
            
            y_per_img2[:, i, :, :] = result_reshaped

        y_per_img3 = y_per_img2[:, :, 1:r_superpix-1, 1:c_superpix-1]
        y1_data = y_per_img3.reshape(nw * nh * P, (r_superpix - 2) * (c_superpix - 2))
        y1.append((full_dataset[d][0], y1_data))

    print('y1 computed')

    # Compute the auto-correlation R
    Y = np.zeros((P * nh * nw, P * nh * nw))
    for b in range(Len):
        im_nbr = b + 1
        print(f'Computing correlation for image {im_nbr}')
        A = y1[b][1]
        Y = Y + A @ A.T

    R = Y / (Len * r_superpix * c_superpix)
    print('R computed')

    # Compute S1
    block_16x100 = np.zeros((height * width, nh * nw))
    block_4x4 = np.diag(np.ones(width))

    s = 0
    for d in range(4, width + 4):  # d from 4 to width+3
        s += 1
        row_start = (s - 1) * width
        row_end = width * s
        col_start = (d - 1) * nh + 4
        col_end = (d - 1) * nh + width + 4
        block_16x100[row_start:row_end, col_start:col_end] = block_4x4

    # Create block diagonal matrix
    blocks = [block_16x100] * P
    S1 = block_diag(*blocks)

    print('S1 computed')

    # Compute y
    y = np.zeros((P * height * width, (r_superpix - 2) * (c_superpix - 2), Len))
    for f in range(Len):
        y[:, :, f] = S1 @ y1[f][1]

    print('y computed')

    # Compute M1
    pattern = np.array([[1, 2], [4, 3]])
    pattern = pattern - 1  # Now [[0, 1], [3, 2]]
    pattern_h, pattern_w = pattern.shape
    M1 = np.zeros((nh * nw, P * nh * nw))
    vn = (nh - pattern_h) // 2
    n_rep = (nh // pattern_h) + 1
    pattern_i = np.tile(pattern, (n_rep, n_rep))
    start_idx = abs(pattern_h - vn)
    pattern_total = pattern_i[
        start_idx:start_idx + nh,
        start_idx:start_idx + nh
    ] * (nh * nw)
    vec = pattern_total.flatten()

    for e in range(nh * nw):
        vec[e] = vec[e] + e
        idx = int(vec[e])
        if 0 <= idx < P * nh * nw:
            M1[e, idx] = 1

    print('M1 computed')

    # Compute D matrix
    M1_R_M1T = M1 @ R @ M1.T
    D = S1 @ R @ M1.T @ np.linalg.pinv(M1_R_M1T)
    print('D computed')

    # ----------------------------------------------------------------------- %
    #                    Apply the Demosaicking Matrix
    # ----------------------------------------------------------------------- %
    demos_dataset = []
    for f in range(Len):
        im_nbr = f + 1
        print(f'Demosaicing image {im_nbr}')

        def fun(x):
            return (D @ x.flatten()).reshape(height, width, P)

        mosaicked_img = mos_dataset[f][1]
        if len(mosaicked_img.shape) == 3 and mosaicked_img.shape[2] == 1:
            mosaicked_img = mosaicked_img[:, :, 0]

        demos_img = blockproc(
            mosaicked_img,
            (height, width),
            fun,
            border_size=(3, 3),
            trim_border=False,
            use_parallel=True
        )

        # Reshape to (r_superpix*height, c_superpix*width, P/4, 4)
        demos_img_reshaped = demos_img.reshape(
            r_superpix * height,
            c_superpix * width,
            P // 4,
            4
        )

        # Extract individual polarization images
        img_0 = demos_img_reshaped[:, :, :, 0] if P == 12 else demos_img_reshaped[:, :, 0]
        img_45 = demos_img_reshaped[:, :, :, 1] if P == 12 else demos_img_reshaped[:, :, 1]
        img_90 = demos_img_reshaped[:, :, :, 2] if P == 12 else demos_img_reshaped[:, :, 2]
        img_135 = demos_img_reshaped[:, :, :, 3] if P == 12 else demos_img_reshaped[:, :, 3]

        demos_dataset.append((
            mos_dataset[f][0],
            img_0,
            img_45,
            img_90,
            img_135
        ))

    print('Demosaicing of the dataset done')

    return demos_dataset, y1, y, D
