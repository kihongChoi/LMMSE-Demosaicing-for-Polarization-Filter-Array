"""Main code for mosaicing images in datasets."""

import numpy as np
import scipy.io as sio
from Function.filter import filter_images


def mosaicking(dataset, save=True, folder_path='', mosaic='pfa'):
    """
    Mosaicing images in datasets.

    Args:
        dataset: List of tuples (name, img_0, img_45, img_90, img_135)
        save: Whether to save the mosaicked dataset
        folder_path: Path to save the dataset
        mosaic: Type of mosaic ('pfa' or 'cpfa')

    Returns:
        MosDataset: List of tuples (name, mosaicked_image)
    """
    print('Mosaicking')
    print('---------------------------------------------------')

    mos_dataset = []
    for k in range(len(dataset)):
        name, img_0, img_45, img_90, img_135 = dataset[k]
        r, c = img_0.shape[:2]
        
        # Concatenate images along channel dimension
        if len(img_0.shape) == 2:  # Grayscale
            image = np.stack([img_0, img_45, img_90, img_135], axis=2)
        else:  # RGB
            image = np.concatenate([img_0, img_45, img_90, img_135], axis=2)
        
        # Reshape to (r, c, 4*channels)
        if len(image.shape) == 3:
            image = image.reshape(r, c, -1)
        
        # Apply filter
        mosaicked_img = filter_images(image, mosaic)
        mos_dataset.append((name, mosaicked_img))

    if save:
        mos_dataset_dict = {}
        for i, (name, mosaicked_img) in enumerate(mos_dataset):
            mos_dataset_dict[f'MosDataset_{i+1}'] = {
                'name': name,
                'mosaicked': mosaicked_img
            }
        sio.savemat('Data/MosDataset.mat', {'MosDataset': mos_dataset_dict}, format='7.3')

    print('Mosaicking done')

    return mos_dataset
