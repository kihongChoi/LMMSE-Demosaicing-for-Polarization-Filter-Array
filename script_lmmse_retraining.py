"""This script retrain the LMMSE algorithm from a dataset of images.

A small dataset of 3 images is provided as a demonstration.
Please use the same image structure as in Data/Dataset if you want to
retrain with your own dataset.
At the end of this Script, you will find the new trained D matrix in
folder "Data".

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

import os
import numpy as np
import scipy.io as sio
from Function.load_dataset import load_dataset
from Function.mosaicking import mosaicking
from Function.d_matrix import d_matrix


def main():
    """Main function for LMMSE retraining script."""
    # Select pfa or cpfa
    mosaic = 'pfa'

    # Load full resolution dataset
    path = os.getcwd()
    folder_path = os.path.join(path, 'Data', 'Dataset')
    nbr_of_img = 3
    full_dataset = load_dataset(
        save=True,
        folder=folder_path,
        nbr_of_img=nbr_of_img,
        mosaic=mosaic
    )

    # Mosaicking of the dataset
    mos_dataset = mosaicking(
        full_dataset,
        save=True,
        folder_path=folder_path,
        mosaic=mosaic
    )

    # Computation of D matrix (demosaicing matrix)
    demos_dataset, y1, y, D = d_matrix(
        full_dataset,
        mos_dataset,
        folder_path=folder_path,
        mosaic=mosaic
    )

    # Save the new re-trained D matrix
    output_path = os.path.join('Data', 'D_Matrix_retrained.mat')
    sio.savemat(output_path, {'D': D})
    print('Retrained D matrix saved')


if __name__ == '__main__':
    main()
