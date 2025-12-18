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

        # Handle shape: PIL doesn't support (rows, cols, 1) - squeeze single channel dimension
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img.squeeze(2)  # Remove single channel dimension: (rows, cols, 1) -> (rows, cols)
        
        # Use PIL for saving (handles Unicode paths better)
        if len(img.shape) == 2:
            Image.fromarray(img, mode='L').save(filename)
        elif len(img.shape) == 3:
            if img.shape[2] == 3:
                Image.fromarray(img, mode='RGB').save(filename)
            elif img.shape[2] == 4:
                Image.fromarray(img, mode='RGBA').save(filename)
            else:
                raise ValueError(f"Unsupported image shape: {img.shape}")
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")
        return True
    except Exception as e:
        print(f"Error saving image {filename}: {e}")
        return False


def main():
    """Main function for LMMSE demosaicing script."""
    # Global parameter
    d_matrix_name = 'D_Matrix.npy'  # If retrained, call D_Matrix_retrained.npy
    save = True  # true to save the demosaiced image
    mosaic = 'auto'  # 'pfa', 'cpfa', or 'auto' - will be auto-detected from D matrix shape

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

    # Load D_Matrix
    d_matrix_path = os.path.join('Data', d_matrix_name)
    D = np.load(d_matrix_path, allow_pickle=True)

    # Sizes definition (for display purposes)
    if mosaic == 'cpfa':
        height = 4
        width = 4
        P = 12
    else:  # pfa
        height = 2
        width = 2
        P = 4

    rows, cols = mos_img.shape[:2]
    r_superpix = int(rows / height)  # number of superpixel in a line
    c_superpix = int(cols / width)  # number of superpixel in a column

    # Demosaicing
    demos_img = function_lmmse_demosaicing(mos_img, D, mosaic=mosaic)

    # Show result images
    # demos_img shape: (rows, cols, channels, 4) where 4 is polarization angles
    # For PFA: channels=1 (grayscale), For CPFA: channels=3 (RGB)
    num_polarizations = 4  # Always 4 polarization angles (0°, 45°, 90°, 135°)
    
    plt.figure(figsize=(12, 10))
    for pol_idx in range(num_polarizations):
        plt.subplot(2, 2, pol_idx + 1)
        img_to_show = demos_img[:, :, :, pol_idx]
        
        # Clip values to [0, 1] for display
        img_to_show = np.clip(img_to_show, 0, 1)
        
        # Determine if grayscale or RGB
        if len(img_to_show.shape) == 2 or (len(img_to_show.shape) == 3 and img_to_show.shape[2] == 1):
            plt.imshow(img_to_show, cmap='gray')
        else:
            plt.imshow(img_to_show)
        
        pol_angles = ['0°', '45°', '90°', '135°']
        plt.title(f'Demosaiced image for {pol_angles[pol_idx]} polarization')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Save result
    if save:
        # Save as .npy file
        np.save('Data/im_demosaiced.npy', demos_img)

        # Save as multipage TIFF
        output_path = 'Data/im_demosaiced.tif'
        if os.path.exists(output_path):
            os.remove(output_path)

        # Save each polarization angle as a separate page
        # demos_img shape: (rows, cols, channels, 4) where 4 is polarization angles
        num_polarizations = 4  # Always 4 polarization angles
        for pol_idx in range(num_polarizations):
            img_to_save = demos_img[:, :, :, pol_idx]
            
            # Clip values to [0, 1] before saving
            img_to_save = np.clip(img_to_save, 0, 1)
            
            # Handle shape: remove single channel dimension if present
            if len(img_to_save.shape) == 3 and img_to_save.shape[2] == 1:
                img_to_save = img_to_save.squeeze(2)
            
            # Save each polarization as a separate file (simpler than multipage TIFF)
            pol_angles = ['0deg', '45deg', '90deg', '135deg']
            output_path_pol = f'Data/im_demosaiced_{pol_angles[pol_idx]}.tif'
            imwrite_unicode(output_path_pol, img_to_save)

        print('Results saved')


if __name__ == '__main__':
    main()
