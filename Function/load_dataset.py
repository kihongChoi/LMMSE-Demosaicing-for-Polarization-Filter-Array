"""Function that converts the input dataset for mosaicing the images."""

import os
import numpy as np
from PIL import Image
import scipy.io as sio


def load_dataset(save=True, folder='', nbr_of_img=3, mosaic='pfa'):
    """
    Load dataset of images.

    All images are loaded as double (float64).
    The return dataset contains images like:
    name  RGB_img_0  RGB_img_45  RGB_img_90  RGB_img_135

    Args:
        save: Whether to save the dataset as .mat file
        folder: Path to the dataset folder
        nbr_of_img: Number of images to load
        mosaic: Type of mosaic ('pfa' or 'cpfa')

    Returns:
        Dataset: List of tuples (name, img_0, img_45, img_90, img_135)
    """
    path = folder
    dataset = []

    if mosaic == 'cpfa':
        for k in range(1, nbr_of_img + 1):
            img_0 = np.array(Image.open(os.path.join(path, f'0_{k}.png')), dtype=np.float64) / 255.0
            img_45 = np.array(Image.open(os.path.join(path, f'45_{k}.png')), dtype=np.float64) / 255.0
            img_90 = np.array(Image.open(os.path.join(path, f'90_{k}.png')), dtype=np.float64) / 255.0
            img_135 = np.array(Image.open(os.path.join(path, f'135_{k}.png')), dtype=np.float64) / 255.0

            dataset.append((str(k), img_0, img_45, img_90, img_135))
    elif mosaic == 'pfa':
        for k in range(1, nbr_of_img + 1):
            # Convert to grayscale for PFA
            img_0 = np.array(Image.open(os.path.join(path, f'0_{k}.png')).convert('L'), dtype=np.float64) / 255.0
            img_45 = np.array(Image.open(os.path.join(path, f'45_{k}.png')).convert('L'), dtype=np.float64) / 255.0
            img_90 = np.array(Image.open(os.path.join(path, f'90_{k}.png')).convert('L'), dtype=np.float64) / 255.0
            img_135 = np.array(Image.open(os.path.join(path, f'135_{k}.png')).convert('L'), dtype=np.float64) / 255.0

            dataset.append((str(k), img_0, img_45, img_90, img_135))
    else:
        raise ValueError(f"Unknown mosaic type: {mosaic}")

    if save:
        # Convert to cell array format for MATLAB compatibility
        dataset_dict = {}
        for i, (name, img_0, img_45, img_90, img_135) in enumerate(dataset):
            dataset_dict[f'Dataset_{i+1}'] = {
                'name': name,
                'img_0': img_0,
                'img_45': img_45,
                'img_90': img_90,
                'img_135': img_135
            }
        sio.savemat('Data/FullDataset.mat', {'Dataset': dataset_dict}, format='7.3')

    print('Dataset loaded')
    print('---------------------------------------------------')

    return dataset
