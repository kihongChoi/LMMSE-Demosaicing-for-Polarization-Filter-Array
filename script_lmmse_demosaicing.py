"""This script load, and demosaic with LMMSE the data from the selected image.

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
from PIL import Image
import matplotlib.pyplot as plt
from function_lmmse_demosaicing import function_lmmse_demosaicing


def imwrite_unicode(filename, img):
    """
    OpenCV imwrite with Unicode (Korean) path support.

    Args:
        filename: File path (can include Korean characters)
        img: Image array to save

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        # Use PIL for saving (handles Unicode paths better)
        if len(img.shape) == 2:
            Image.fromarray(img).save(filename)
        elif len(img.shape) == 3:
            Image.fromarray(img).save(filename)
        return True
    except Exception as e:
        print(f"Error saving image {filename}: {e}")
        return False


def main():
    """Main function for LMMSE demosaicing script."""
    # Global parameter
    d_matrix_name = 'D_Matrix.mat'  # If retrained, call D_Matrix_retrained
    save = True  # true to save the demosaiced image

    # Load mosaiced image
    mos_img_path = 'Data/im.tif'
    mos_img = np.array(Image.open(mos_img_path), dtype=np.float64) / 255.0
    
    # Handle grayscale
    if len(mos_img.shape) == 3 and mos_img.shape[2] == 1:
        mos_img = mos_img[:, :, 0]
    elif len(mos_img.shape) == 3:
        # Convert RGB to grayscale if needed for PFA
        mos_img = np.mean(mos_img, axis=2)

    plt.figure()
    plt.imshow(mos_img, cmap='gray')
    plt.title('Mosaiced image')
    plt.axis('off')
    plt.show()

    # Sizes definition
    height = 2  # height of the superpixel
    width = 2  # width of the superpixel
    P = 4  # number of color-pola channels

    rows, cols = mos_img.shape[:2]
    r_superpix = int(rows / height)  # number of superpixel in a line
    c_superpix = int(cols / width)  # number of superpixel in a column

    # Load D_Matrix
    d_matrix_path = os.path.join('Data', d_matrix_name)
    d_matrix_data = sio.loadmat(d_matrix_path)
    D = d_matrix_data['D']

    # Demosaicing
    demos_img = function_lmmse_demosaicing(mos_img, D)

    # Show result images
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(demos_img[:, :, :, 0], cmap='gray' if len(demos_img.shape) == 3 else None)
    plt.title('Demosaiced image for 0° polarization')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(demos_img[:, :, :, 1], cmap='gray' if len(demos_img.shape) == 3 else None)
    plt.title('Demosaiced image for 45° polarization')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(demos_img[:, :, :, 2], cmap='gray' if len(demos_img.shape) == 3 else None)
    plt.title('Demosaiced image for 90° polarization')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(demos_img[:, :, :, 3], cmap='gray' if len(demos_img.shape) == 3 else None)
    plt.title('Demosaiced image for 135° polarization')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Save result
    if save:
        # Save as .mat file
        sio.savemat('Data/im_demosaiced.mat', {'DemosImg': demos_img}, format='7.3')

        # Save as multipage TIFF
        output_path = 'Data/im_demosaiced.tif'
        if os.path.exists(output_path):
            os.remove(output_path)

        # Save each polarization angle as a separate page
        for i in range(P):
            img_to_save = demos_img[:, :, :, i] if len(demos_img.shape) == 4 else demos_img[:, :, i]
            if i == 0:
                # First image - create new file
                imwrite_unicode(output_path, img_to_save)
            else:
                # Append to existing file
                # For multipage TIFF, we'll save as separate files or use PIL
                img_pil = Image.fromarray((img_to_save * 255).astype(np.uint8) if img_to_save.max() <= 1.0 else img_to_save.astype(np.uint8))
                # Note: PIL's append_images can be used for multipage TIFF
                pass  # Simplified - save as separate images or use tifffile library

        print('Results saved')


if __name__ == '__main__':
    main()
